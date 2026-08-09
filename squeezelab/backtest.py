"""
Portfolio backtester.

What changed from the 1.x engine, and why each change matters:

* **No same-bar lookahead.** The old loop read a bar's close, decided on a
  signal derived from that same bar, and filled at that same close. Every result
  it produced was unreachable in practice. Signals are now evaluated on bar `t`
  and filled at bar `t+1`'s open.
* **Trading days, not calendar days.** `(exit_date - entry_date).days` counted
  weekends, so a "5 day hold" was really three sessions over a weekend. Holds
  are counted in bars.
* **Intrabar exits.** Take-profit and stop-loss are checked against the bar's
  high and low rather than its close, so a 20% target inside the bar registers.
  When a bar touches both, the stop is assumed to fill first.
* **Costs.** Commission and slippage in basis points on both sides.
* **A real portfolio.** Cash, position sizing, a cap on concurrent positions,
  and a dated equity curve — so the summary can report drawdown, Sharpe and
  exposure instead of just summing trade percentages (which silently assumed
  compounding that never happened).
* **A benchmark.** Equal-weight buy-and-hold over the same universe and window,
  because a strategy's return means nothing without one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

TRADING_DAYS_YEAR = 252


@dataclass
class StrategyParams:
    entry_score: float = 70.0
    volume_mult: float = 1.5
    hold_days: int = 5          # trading days
    take_profit: float = 0.20
    stop_loss: float = -0.10
    trailing_stop: float | None = None  # e.g. 0.12 => exit 12% off the peak close
    position_pct: float = 0.20  # of starting capital, per position
    max_positions: int = 5
    starting_capital: float = 100_000.0
    commission_bps: float = 2.0
    slippage_bps: float = 10.0
    cooldown_days: int = 1      # bars to wait before re-entering the same name
    label: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    benchmark_curve: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "benchmark_curve": self.benchmark_curve,
            "summary": self.summary,
            "params": self.params,
            "notes": self.notes,
        }

    @property
    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


class SqueezeBacktester:
    """Long-only squeeze strategy over a shared calendar."""

    def __init__(self, params: StrategyParams | None = None, **overrides: Any) -> None:
        base = params or StrategyParams()
        if overrides:
            legacy = {
                "entry_score_threshold": "entry_score",
                "take_profit": "take_profit",
                "stop_loss": "stop_loss",
                "max_position_pct": "position_pct",
            }
            mapped = {legacy.get(k, k): v for k, v in overrides.items()}
            known = {k: v for k, v in mapped.items() if k in StrategyParams.__dataclass_fields__}
            base = StrategyParams(**{**base.to_dict(), **known})
        self.params = base
        self.trades = pd.DataFrame()

    # -- public -----------------------------------------------------------

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        prepared = {t: _prepare(df) for t, df in price_data.items()}
        prepared = {t: df for t, df in prepared.items() if df is not None and len(df) > 2}
        notes: list[str] = []
        if not prepared:
            return BacktestResult(summary={"error": "no usable price data"}, params=self.params.to_dict())

        merged = {t: _attach_signals(df, (signals or {}).get(t)) for t, df in prepared.items()}
        bases = {
            str(frame["si_basis"].iloc[0])
            for frame in (signals or {}).values()
            if isinstance(frame, pd.DataFrame) and "si_basis" in frame.columns and not frame.empty
        }
        notes.extend(sorted(bases))

        calendar = sorted({d for df in merged.values() for d in df["Date"]})
        result = self._simulate(merged, calendar)
        result.notes = notes + result.notes
        self.trades = result.trades_frame
        return result

    # Backwards-compatible entry point used by the 1.x CLI.
    def run_portfolio(
        self,
        price_data_dict: dict[str, pd.DataFrame],
        signals_dict: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        self._last = self.run(price_data_dict, signals_dict)
        return self._last.trades_frame

    def summarize(self) -> dict[str, Any]:
        last = getattr(self, "_last", None)
        if last is None:
            return {"error": "no backtest has been run"}
        return last.summary

    # -- simulation -------------------------------------------------------

    def _simulate(self, data: dict[str, pd.DataFrame], calendar: Sequence[pd.Timestamp]) -> BacktestResult:
        p = self.params
        lookups = {t: {row.Date: row for row in df.itertuples(index=False)} for t, df in data.items()}
        # Index of each ticker's bars so "5 trading days" means 5 of *its* bars.
        bar_index = {t: {row.Date: i for i, row in enumerate(df.itertuples(index=False))} for t, df in data.items()}

        cash = p.starting_capital
        positions: dict[str, dict[str, Any]] = {}
        cooldown: dict[str, pd.Timestamp] = {}
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        size = p.starting_capital * p.position_pct

        for day in calendar:
            # 1. Exits first, so capital freed today can be redeployed today.
            for ticker in list(positions):
                row = lookups[ticker].get(day)
                if row is None:
                    continue
                position = positions[ticker]
                exit_price, reason = self._exit_for_bar(row, position, bar_index[ticker].get(day, 0))
                if reason is None:
                    position["peak"] = max(position["peak"], float(row.Close))
                    continue
                fill = exit_price * (1 - (p.slippage_bps + p.commission_bps) / 10_000)
                proceeds = position["shares"] * fill
                cash += proceeds
                pnl = proceeds - position["cost"]
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": position["entry_date"].date().isoformat(),
                        "exit_date": pd.Timestamp(day).date().isoformat(),
                        "entry_price": round(position["entry_price"], 4),
                        "exit_price": round(fill, 4),
                        "shares": round(position["shares"], 4),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl / position["cost"] * 100, 2) if position["cost"] else 0.0,
                        "bars_held": bar_index[ticker].get(day, 0) - position["entry_bar"],
                        "entry_score": round(position["entry_score"], 2),
                        "exit_reason": reason,
                    }
                )
                del positions[ticker]
                cooldown[ticker] = day

            # 2. Entries, filled at today's open on yesterday's signal.
            free_slots = p.max_positions - len(positions)
            if free_slots > 0:
                for ticker, row in self._candidates(lookups, data, day, positions, cooldown, bar_index):
                    if free_slots <= 0:
                        break
                    entry_price = float(row.Open) * (1 + (p.slippage_bps + p.commission_bps) / 10_000)
                    if entry_price <= 0:
                        continue
                    budget = min(size, cash)
                    if budget < entry_price:
                        continue
                    shares = budget / entry_price
                    cost = shares * entry_price
                    cash -= cost
                    positions[ticker] = {
                        "shares": shares,
                        "entry_price": entry_price,
                        "cost": cost,
                        "entry_date": day,
                        "entry_bar": bar_index[ticker].get(day, 0),
                        "entry_score": float(getattr(row, "prev_score", 0.0) or 0.0),
                        "peak": float(row.Close),
                    }
                    free_slots -= 1

            # 3. Mark to market.
            holdings = 0.0
            for ticker, position in positions.items():
                row = lookups[ticker].get(day)
                price = float(row.Close) if row is not None else position["entry_price"]
                holdings += position["shares"] * price
            equity_curve.append(
                {
                    "date": pd.Timestamp(day).date().isoformat(),
                    "equity": round(cash + holdings, 2),
                    "cash": round(cash, 2),
                    "positions": len(positions),
                }
            )

        benchmark = _benchmark_curve(data, calendar, p.starting_capital)
        summary = summarise(trades, equity_curve, benchmark, p)
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            benchmark_curve=benchmark,
            summary=summary,
            params=p.to_dict(),
        )

    def _candidates(
        self,
        lookups: dict[str, dict[Any, Any]],
        data: dict[str, pd.DataFrame],
        day: pd.Timestamp,
        positions: dict[str, Any],
        cooldown: dict[str, pd.Timestamp],
        bar_index: dict[str, dict[Any, int]],
    ) -> Iterable[tuple[str, Any]]:
        """Tickers whose *previous* bar fired the entry rule, best score first."""
        p = self.params
        found: list[tuple[float, str, Any]] = []
        for ticker, lookup in lookups.items():
            if ticker in positions:
                continue
            row = lookup.get(day)
            if row is None:
                continue
            last_exit = cooldown.get(ticker)
            if last_exit is not None:
                gap = bar_index[ticker].get(day, 0) - bar_index[ticker].get(last_exit, 0)
                if gap < p.cooldown_days:
                    continue
            score = getattr(row, "prev_score", None)
            volume = getattr(row, "prev_volume", None)
            avg_volume = getattr(row, "prev_avg_volume", None)
            if score is None or score != score or score < p.entry_score:
                continue
            if not avg_volume or avg_volume != avg_volume:
                continue
            if not volume or volume < avg_volume * p.volume_mult:
                continue
            if not getattr(row, "Open", None) or float(row.Open) <= 0:
                continue
            found.append((float(score), ticker, row))
        found.sort(key=lambda item: item[0], reverse=True)
        return [(ticker, row) for _, ticker, row in found]

    def _exit_for_bar(self, row: Any, position: dict[str, Any], bar: int) -> tuple[float, str | None]:
        """Decide whether this bar closes the position, and at what price."""
        p = self.params
        entry = position["entry_price"]
        high = float(getattr(row, "High", row.Close))
        low = float(getattr(row, "Low", row.Close))
        close = float(row.Close)

        stop_price = entry * (1 + p.stop_loss)
        target_price = entry * (1 + p.take_profit)

        # A bar that touches both is assumed to hit the stop first.
        if low <= stop_price:
            return stop_price, "STOP_LOSS"
        if high >= target_price:
            return target_price, "TAKE_PROFIT"
        if p.trailing_stop:
            trail_price = position["peak"] * (1 - p.trailing_stop)
            if low <= trail_price < entry * (1 + p.take_profit):
                return min(trail_price, close if close < trail_price else trail_price), "TRAILING_STOP"
        if bar - position["entry_bar"] >= p.hold_days:
            return close, "TIME_EXIT"
        return close, None


# --------------------------------------------------------------------------
# Frame preparation
# --------------------------------------------------------------------------


def _prepare(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return None
    frame = df.copy()
    if "Date" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    try:
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    frame["Date"] = frame["Date"].dt.normalize()
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    for column in ("Open", "High", "Low"):
        if column not in frame.columns:
            frame[column] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0
    frame["avg_volume_20d"] = frame["Volume"].rolling(20, min_periods=10).mean()
    # Everything the entry rule reads is shifted by one bar — this is the
    # no-lookahead guarantee, enforced in one place.
    frame["prev_volume"] = frame["Volume"].shift(1)
    frame["prev_avg_volume"] = frame["avg_volume_20d"].shift(1)
    return frame


def _attach_signals(frame: pd.DataFrame, signals: pd.DataFrame | None) -> pd.DataFrame:
    frame = frame.copy()
    if signals is None or len(signals) == 0 or "squeeze_score" not in signals.columns:
        frame["score"] = float("nan")
    else:
        sig = signals.copy()
        stamp = "timestamp" if "timestamp" in sig.columns else "Date"
        sig[stamp] = pd.to_datetime(sig[stamp], errors="coerce")
        try:
            sig[stamp] = sig[stamp].dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        sig[stamp] = sig[stamp].dt.normalize()
        sig = sig.dropna(subset=[stamp]).drop_duplicates(subset=[stamp], keep="last")
        mapping = dict(zip(sig[stamp], sig["squeeze_score"]))
        frame["score"] = frame["Date"].map(mapping)
    frame["prev_score"] = frame["score"].shift(1)
    return frame


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _benchmark_curve(
    data: dict[str, pd.DataFrame],
    calendar: Sequence[pd.Timestamp],
    capital: float,
) -> list[dict[str, Any]]:
    """Equal-weight buy-and-hold across the same universe and window."""
    if not data or not calendar:
        return []
    per_name = capital / len(data)
    shares: dict[str, float] = {}
    for ticker, frame in data.items():
        first = frame.iloc[0]
        price = float(first.get("Open") or first["Close"])
        if price > 0:
            shares[ticker] = per_name / price
    curve: list[dict[str, Any]] = []
    last_price: dict[str, float] = {}
    lookups = {t: dict(zip(df["Date"], df["Close"])) for t, df in data.items()}
    for day in calendar:
        total = 0.0
        for ticker, count in shares.items():
            price = lookups[ticker].get(day)
            if price is None or price != price:
                price = last_price.get(ticker)
            else:
                last_price[ticker] = float(price)
            if price:
                total += count * float(price)
        curve.append({"date": pd.Timestamp(day).date().isoformat(), "equity": round(total, 2)})
    return curve


def summarise(
    trades: Sequence[dict[str, Any]],
    equity_curve: Sequence[dict[str, Any]],
    benchmark: Sequence[dict[str, Any]],
    params: StrategyParams,
) -> dict[str, Any]:
    equity = [point["equity"] for point in equity_curve]
    start = params.starting_capital
    end = equity[-1] if equity else start

    summary: dict[str, Any] = {
        "label": params.label,
        "starting_capital": start,
        "ending_capital": round(end, 2),
        "total_return_pct": round((end / start - 1) * 100, 2) if start else 0.0,
        "total_trades": len(trades),
        "sessions": len(equity_curve),
    }

    returns = _daily_returns(equity)
    summary["max_drawdown_pct"] = round(_max_drawdown(equity) * 100, 2)
    summary["cagr_pct"] = round(_cagr(equity_curve, start, end) * 100, 2)
    summary["sharpe"] = _sharpe(returns)
    summary["sortino"] = _sortino(returns)
    summary["volatility_pct"] = round(_stdev(returns) * math.sqrt(TRADING_DAYS_YEAR) * 100, 2)
    summary["exposure_pct"] = (
        round(sum(1 for p in equity_curve if p.get("positions")) / len(equity_curve) * 100, 2)
        if equity_curve
        else 0.0
    )

    if benchmark:
        bench_end = benchmark[-1]["equity"]
        bench_return = (bench_end / start - 1) * 100 if start else 0.0
        summary["benchmark_return_pct"] = round(bench_return, 2)
        summary["excess_return_pct"] = round(summary["total_return_pct"] - bench_return, 2)
        summary["benchmark_max_drawdown_pct"] = round(
            _max_drawdown([p["equity"] for p in benchmark]) * 100, 2
        )

    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        summary.update(
            {
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": round(len(wins) / len(pnls) * 100, 2),
                "avg_return": round(sum(pnls) / len(pnls), 2),
                "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
                "max_gain": round(max(pnls), 2),
                "max_loss": round(min(pnls), 2),
                "expectancy_pct": round(sum(pnls) / len(pnls), 2),
                "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
                "avg_hold_bars": round(sum(t["bars_held"] for t in trades) / len(trades), 1),
                "exits": _counts(t["exit_reason"] for t in trades),
                "by_ticker": _by_ticker(trades),
            }
        )
    else:
        summary.update(
            {
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "profit_factor": None,
                "exits": {},
                "by_ticker": [],
                "note": "no entries fired — try a lower score threshold or volume multiple",
            }
        )
    return summary


def _daily_returns(equity: Sequence[float]) -> list[float]:
    return [
        (equity[i] / equity[i - 1]) - 1
        for i in range(1, len(equity))
        if equity[i - 1]
    ]


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _sharpe(returns: Sequence[float]) -> float | None:
    if len(returns) < 2:
        return None
    deviation = _stdev(returns)
    if deviation == 0:
        return None
    mean = sum(returns) / len(returns)
    return round(mean / deviation * math.sqrt(TRADING_DAYS_YEAR), 2)


def _sortino(returns: Sequence[float]) -> float | None:
    downside = [r for r in returns if r < 0]
    if len(returns) < 2 or not downside:
        return None
    deviation = math.sqrt(sum(r ** 2 for r in downside) / len(returns))
    if deviation == 0:
        return None
    mean = sum(returns) / len(returns)
    return round(mean / deviation * math.sqrt(TRADING_DAYS_YEAR), 2)


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1)
    return worst


def _cagr(curve: Sequence[dict[str, Any]], start: float, end: float) -> float:
    if len(curve) < 2 or start <= 0 or end <= 0:
        return 0.0
    first = pd.Timestamp(curve[0]["date"])
    last = pd.Timestamp(curve[-1]["date"])
    years = max((last - first).days / 365.25, 1 / 365.25)
    return (end / start) ** (1 / years) - 1


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _by_ticker(trades: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(trade["ticker"], []).append(trade)
    rows = [
        {
            "ticker": ticker,
            "trades": len(items),
            "win_rate": round(sum(1 for t in items if t["pnl_pct"] > 0) / len(items) * 100, 1),
            "avg_return": round(sum(t["pnl_pct"] for t in items) / len(items), 2),
            "total_pnl": round(sum(t["pnl"] for t in items), 2),
        }
        for ticker, items in grouped.items()
    ]
    rows.sort(key=lambda r: r["total_pnl"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# Strategy comparison
# --------------------------------------------------------------------------

DEFAULT_VARIANTS: tuple[StrategyParams, ...] = (
    StrategyParams(label="baseline", entry_score=70, take_profit=0.20, stop_loss=-0.10, hold_days=5),
    StrategyParams(label="tight stop", entry_score=70, take_profit=0.20, stop_loss=-0.06, hold_days=5),
    StrategyParams(label="let it run", entry_score=70, take_profit=0.40, stop_loss=-0.12, hold_days=10),
    StrategyParams(label="high conviction", entry_score=80, take_profit=0.25, stop_loss=-0.10, hold_days=7),
    StrategyParams(label="trailing", entry_score=70, take_profit=0.50, stop_loss=-0.12, hold_days=15, trailing_stop=0.12),
)


def compare_strategies(
    price_data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame] | None = None,
    variants: Sequence[StrategyParams] | None = None,
) -> list[dict[str, Any]]:
    """Run several parameter sets over identical data and line the results up."""
    rows: list[dict[str, Any]] = []
    for params in variants or DEFAULT_VARIANTS:
        result = SqueezeBacktester(params).run(price_data, signals)
        summary = dict(result.summary)
        summary["params"] = params.to_dict()
        summary["equity_curve"] = result.equity_curve
        rows.append(summary)
    rows.sort(key=lambda r: r.get("total_return_pct", 0), reverse=True)
    return rows
