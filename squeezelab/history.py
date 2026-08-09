"""
Historical signal reconstruction.

The 1.x backtester fed a constant squeeze score into the strategy, which meant
the entry rule reduced to "volume spike" and the results said nothing about the
score. This module rebuilds a per-day score from the data that genuinely varies
day to day:

  * price, previous close, trailing 52-week range     — from the bars
  * 10-day average volume                             — from the bars
  * off-exchange short volume and its trend           — from FINRA's daily files

Short interest itself is only published semi-monthly and free history is not
available, so it is held at its most recent value across the window. That is a
real limitation, not a hidden one: every reconstructed frame carries a
`si_basis` column saying which fields were held constant, and the backtest
summary repeats it.

Borrow fees, shortable-share counts and option open interest are deliberately
*absent* from `STATIC_FIELDS`. All three are snapshots of the present with no
free history, and holding today's value across a year would be strictly worse
than holding short interest: today's 90% borrow fee is partly a consequence of
the squeeze the backtest is pretending to predict. So the borrow component
reconstructs from days-to-cover and float alone, the gamma component has no
inputs at all and is dropped and renormalised away, and the result understates
what a live scan sees rather than flattering it. Anyone adding one of these to
`STATIC_FIELDS` is adding lookahead, not coverage.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .engine import COMPONENTS, SqueezeEngine

TRADING_DAYS_YEAR = 252
STATIC_FIELDS = (
    "ticker", "name", "sector", "float", "shares_outstanding", "short_pct_float",
    "short_ratio", "shares_short", "shares_short_prior", "market_cap",
)


def reconstruct_scores(
    values: dict[str, Any],
    bars: pd.DataFrame,
    engine: SqueezeEngine | None = None,
    off_exchange_series: Sequence[dict[str, Any]] | None = None,
    warmup: int = 20,
) -> pd.DataFrame:
    """Per-bar squeeze scores for one ticker.

    Returns a frame with `timestamp`, `squeeze_score`, the component scores and
    `si_basis`. Rows before `warmup` bars are dropped so rolling inputs are
    meaningful.
    """
    engine = engine or SqueezeEngine()
    if bars is None or bars.empty or "Close" not in bars.columns:
        return pd.DataFrame(columns=["timestamp", "squeeze_score"])

    frame = bars.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "squeeze_score"])

    high_col = "High" if "High" in frame.columns else "Close"
    low_col = "Low" if "Low" in frame.columns else "Close"
    frame["roll_high"] = frame[high_col].rolling(TRADING_DAYS_YEAR, min_periods=warmup).max()
    frame["roll_low"] = frame[low_col].rolling(TRADING_DAYS_YEAR, min_periods=warmup).min()
    frame["prev_close"] = frame["Close"].shift(1)
    if "Volume" in frame.columns:
        frame["avg_volume_10d"] = frame["Volume"].rolling(10, min_periods=5).mean()
    else:
        frame["avg_volume_10d"] = None

    flow_lookup = _flow_lookup(off_exchange_series)
    static = {k: values.get(k) for k in STATIC_FIELDS if values.get(k) is not None}

    held = [f for f in ("short_pct_float", "short_ratio", "shares_short") if f in static]
    basis = "short interest held at latest value" if held else "no short interest available"

    rows: list[dict[str, Any]] = []
    for record in frame.iloc[warmup:].itertuples(index=False):
        daily = dict(static)
        daily.update(
            {
                "current_price": float(record.Close),
                "previous_close": _clean(getattr(record, "prev_close", None)),
                "fifty_two_week_high": _clean(getattr(record, "roll_high", None)),
                "fifty_two_week_low": _clean(getattr(record, "roll_low", None)),
                "avg_volume_10d": _clean(getattr(record, "avg_volume_10d", None)),
            }
        )
        stamp = pd.Timestamp(record.Date).normalize()
        flow = flow_lookup.get(stamp.date().isoformat())
        if flow:
            daily.update(flow)

        block = engine.score(daily)
        rows.append(
            {
                "timestamp": stamp,
                "squeeze_score": block["squeeze_score"],
                # Driven off COMPONENTS rather than listed, so a component
                # added to the engine cannot go missing from the frame the
                # backtest charts. A component with no reconstructable inputs
                # comes through as an all-None column, which is the honest
                # answer, not an omission.
                **{f"{key}_score": block[f"{key}_score"] for key in COMPONENTS},
                "si_basis": basis,
            }
        )
    return pd.DataFrame(rows)


def build_signals(
    records: dict[str, dict[str, Any]],
    price_data: dict[str, pd.DataFrame],
    engine: SqueezeEngine | None = None,
) -> dict[str, pd.DataFrame]:
    """Reconstruct signal frames for every ticker that has bars."""
    engine = engine or SqueezeEngine()
    signals: dict[str, pd.DataFrame] = {}
    for ticker, bars in price_data.items():
        values = records.get(ticker, {"ticker": ticker})
        frame = reconstruct_scores(
            values,
            bars,
            engine=engine,
            off_exchange_series=values.get("off_exchange_series"),
        )
        if not frame.empty:
            signals[ticker] = frame
    return signals


def _flow_lookup(series: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Turn FINRA's daily short-volume series into a per-date input map."""
    if not series:
        return {}
    points = [p for p in series if isinstance(p, dict) and p.get("date") and p.get("ratio") is not None]
    points.sort(key=lambda p: str(p["date"]))
    lookup: dict[str, dict[str, Any]] = {}
    for index, point in enumerate(points):
        window = points[max(0, index - 19) : index + 1]
        fast = points[max(0, index - 4) : index + 1]
        slow_ratio = sum(float(p["ratio"]) for p in window) / len(window)
        fast_ratio = sum(float(p["ratio"]) for p in fast) / len(fast)
        lookup[str(point["date"])[:10]] = {
            "off_exchange_short_pct": round(slow_ratio, 2),
            "off_exchange_trend": round(fast_ratio - slow_ratio, 2),
        }
    return lookup


def _clean(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed
