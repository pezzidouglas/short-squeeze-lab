"""
Backtester behaviour, with the lookahead and hold-length bugs pinned down.

These are the tests that would have failed against the 1.x engine.
"""

from __future__ import annotations

import pandas as pd
import pytest

from squeezelab.backtest import (
    SqueezeBacktester,
    StrategyParams,
    compare_strategies,
    summarise,
)

DATES = pd.bdate_range("2026-01-05", periods=30)


def frame(closes, opens=None, highs=None, lows=None, volumes=None):
    n = len(closes)
    opens = opens or [c * 0.99 for c in closes]
    highs = highs or [max(o, c) * 1.01 for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) * 0.99 for o, c in zip(opens, closes)]
    volumes = volumes or [1_000_000] * n
    return pd.DataFrame({
        "Date": DATES[:n], "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes, "ticker": "TEST",
    })


def signals(scores):
    return pd.DataFrame({"timestamp": DATES[: len(scores)], "squeeze_score": scores})


def flat_params(**kwargs):
    base = dict(
        entry_score=70, volume_mult=1.5, hold_days=5, take_profit=0.20, stop_loss=-0.10,
        position_pct=1.0, max_positions=1, starting_capital=10_000,
        commission_bps=0, slippage_bps=0, cooldown_days=0,
    )
    base.update(kwargs)
    return StrategyParams(**base)


# --------------------------------------------------------------------------
# No lookahead
# --------------------------------------------------------------------------


def test_entry_fills_on_the_bar_after_the_signal():
    closes = [10.0] * 30
    volumes = [1_000_000] * 30
    volumes[21] = 5_000_000              # the spike is on bar 21
    opens = [10.0] * 30
    opens[22] = 11.0                     # a distinctive open on the fill bar
    scores = [0.0] * 30
    scores[21] = 95.0                    # the signal is on bar 21

    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame(closes, opens=opens, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Fill is bar 22's open — not bar 21's close, which the signal was derived from.
    assert trade["entry_date"] == DATES[22].date().isoformat()
    assert trade["entry_price"] == pytest.approx(11.0)


def test_no_trade_when_the_signal_never_clears_the_threshold():
    volumes = [1_000_000] * 30
    volumes[21] = 5_000_000
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame([10.0] * 30, volumes=volumes)},
        {"TEST": signals([50.0] * 30)},
    )
    assert result.trades == []
    assert result.summary["total_trades"] == 0
    assert "no entries fired" in result.summary["note"]


def test_no_trade_without_a_volume_spike():
    scores = [0.0] * 30
    scores[21] = 95.0
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame([10.0] * 30)},
        {"TEST": signals(scores)},
    )
    assert result.trades == []


# --------------------------------------------------------------------------
# Exits
# --------------------------------------------------------------------------


def _entry_setup():
    """A frame that always enters at bar 22's open of 10.0."""
    volumes = [1_000_000] * 30
    volumes[21] = 5_000_000
    scores = [0.0] * 30
    scores[21] = 95.0
    return volumes, scores


def test_take_profit_fills_intrabar_on_the_high():
    volumes, scores = _entry_setup()
    closes = [10.0] * 30
    highs = [10.1] * 30
    highs[24] = 13.0            # touches +20% intrabar, closes back at 10
    lows = [9.9] * 30
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame(closes, opens=[10.0] * 30, highs=highs, lows=lows, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    trade = result.trades[0]
    assert trade["exit_reason"] == "TAKE_PROFIT"
    assert trade["exit_price"] == pytest.approx(12.0)   # the target, not the high
    assert trade["pnl_pct"] == pytest.approx(20.0, abs=0.01)


def test_stop_loss_fills_intrabar_on_the_low():
    volumes, scores = _entry_setup()
    lows = [9.9] * 30
    lows[23] = 8.0
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=[10.1] * 30, lows=lows, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    trade = result.trades[0]
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == pytest.approx(9.0)


def test_a_bar_touching_both_is_assumed_to_stop_out_first():
    volumes, scores = _entry_setup()
    highs = [10.1] * 30
    lows = [9.9] * 30
    highs[23], lows[23] = 13.0, 8.0      # both the target and the stop are inside bar 23
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=highs, lows=lows, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    assert result.trades[0]["exit_reason"] == "STOP_LOSS"


def test_time_exit_counts_trading_days_not_calendar_days():
    volumes, scores = _entry_setup()
    result = SqueezeBacktester(flat_params(hold_days=5)).run(
        {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=[10.1] * 30, lows=[9.9] * 30, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    trade = result.trades[0]
    assert trade["exit_reason"] == "TIME_EXIT"
    assert trade["bars_held"] == 5
    # Bar 22 + 5 bars = bar 27. Calendar-day counting would have exited early.
    assert trade["exit_date"] == DATES[27].date().isoformat()


def test_trailing_stop_exits_after_a_run_up():
    volumes, scores = _entry_setup()
    closes = [10.0] * 30
    for i in range(22, 26):
        closes[i] = 11.0 + (i - 22) * 0.2   # drifts up to 11.6
    closes[26] = 9.5                        # then gives it all back
    lows = [c * 0.99 for c in closes]
    lows[26] = 9.4
    result = SqueezeBacktester(
        flat_params(hold_days=40, take_profit=5.0, stop_loss=-0.90, trailing_stop=0.10)
    ).run(
        {"TEST": frame(closes, opens=[10.0] * 30, highs=[c * 1.01 for c in closes], lows=lows, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    assert result.trades[0]["exit_reason"] == "TRAILING_STOP"


def test_cooldown_blocks_an_immediate_re_entry():
    volumes = [1_000_000] * 30
    scores = [0.0] * 30
    for i in (21, 22, 23, 24, 25, 26):
        volumes[i] = 5_000_000
        scores[i] = 95.0
    with_cooldown = SqueezeBacktester(flat_params(hold_days=1, cooldown_days=5)).run(
        {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=[10.1] * 30, lows=[9.9] * 30, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    without = SqueezeBacktester(flat_params(hold_days=1, cooldown_days=0)).run(
        {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=[10.1] * 30, lows=[9.9] * 30, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    assert len(with_cooldown.trades) < len(without.trades)


# --------------------------------------------------------------------------
# Costs, portfolio, metrics
# --------------------------------------------------------------------------


def test_costs_eat_into_the_result():
    volumes, scores = _entry_setup()
    data = {"TEST": frame([10.0] * 30, opens=[10.0] * 30, highs=[10.1] * 30, lows=[9.9] * 30, volumes=volumes)}
    signal = {"TEST": signals(scores)}
    free = SqueezeBacktester(flat_params()).run(data, signal)
    costly = SqueezeBacktester(flat_params(commission_bps=10, slippage_bps=40)).run(data, signal)
    assert costly.trades[0]["pnl_pct"] < free.trades[0]["pnl_pct"]


def test_max_positions_caps_concurrent_exposure():
    volumes = [1_000_000] * 30
    volumes[21] = 5_000_000
    scores = [0.0] * 30
    scores[21] = 95.0
    data = {name: frame([10.0] * 30, opens=[10.0] * 30, volumes=volumes) for name in ("A", "B", "C")}
    signal = {name: signals(scores) for name in data}
    result = SqueezeBacktester(flat_params(max_positions=2, position_pct=0.4)).run(data, signal)
    assert max(p["positions"] for p in result.equity_curve) <= 2


def test_equity_curve_is_dated_and_starts_at_capital():
    volumes, scores = _entry_setup()
    result = SqueezeBacktester(flat_params()).run(
        {"TEST": frame([10.0] * 30, volumes=volumes)},
        {"TEST": signals(scores)},
    )
    assert len(result.equity_curve) == 30
    assert result.equity_curve[0]["equity"] == pytest.approx(10_000, abs=1)
    assert result.equity_curve[0]["date"] == DATES[0].date().isoformat()


def test_benchmark_tracks_buy_and_hold():
    closes = [10.0 + i for i in range(30)]
    result = SqueezeBacktester(flat_params()).run({"TEST": frame(closes)}, {"TEST": signals([0.0] * 30)})
    # Buys the first bar's open (9.90), holds to the last close (39.00).
    assert result.summary["benchmark_return_pct"] == pytest.approx(39.0 / 9.9 * 100 - 100, abs=0.5)
    assert result.summary["total_return_pct"] == pytest.approx(0.0, abs=0.01)
    assert result.summary["excess_return_pct"] < 0


def test_summary_metrics_on_a_known_curve():
    curve = [
        {"date": "2026-01-01", "equity": 100.0, "positions": 1},
        {"date": "2026-01-02", "equity": 120.0, "positions": 1},
        {"date": "2026-01-03", "equity": 90.0, "positions": 0},
        {"date": "2026-01-04", "equity": 110.0, "positions": 1},
    ]
    trades = [
        {"ticker": "A", "pnl": 20.0, "pnl_pct": 20.0, "bars_held": 2, "exit_reason": "TAKE_PROFIT"},
        {"ticker": "A", "pnl": -10.0, "pnl_pct": -10.0, "bars_held": 3, "exit_reason": "STOP_LOSS"},
    ]
    summary = summarise(trades, curve, [], StrategyParams(starting_capital=100.0, label="t"))
    assert summary["total_return_pct"] == pytest.approx(10.0)
    assert summary["max_drawdown_pct"] == pytest.approx(-25.0)
    assert summary["win_rate"] == 50.0
    assert summary["profit_factor"] == pytest.approx(2.0)
    assert summary["exits"] == {"TAKE_PROFIT": 1, "STOP_LOSS": 1}
    assert summary["exposure_pct"] == 75.0
    assert summary["by_ticker"][0]["ticker"] == "A"


def test_profit_factor_is_none_rather_than_infinite_when_nothing_lost():
    trades = [{"ticker": "A", "pnl": 5.0, "pnl_pct": 5.0, "bars_held": 1, "exit_reason": "TAKE_PROFIT"}]
    curve = [{"date": "2026-01-01", "equity": 100.0, "positions": 1}, {"date": "2026-01-02", "equity": 105.0, "positions": 0}]
    assert summarise(trades, curve, [], StrategyParams(starting_capital=100.0))["profit_factor"] is None


def test_empty_price_data_is_reported_not_raised():
    result = SqueezeBacktester().run({}, {})
    assert result.summary["error"] == "no usable price data"


def test_compare_strategies_runs_every_variant_on_the_same_data():
    volumes, scores = _entry_setup()
    data = {"TEST": frame([10.0] * 30, opens=[10.0] * 30, volumes=volumes)}
    rows = compare_strategies(data, {"TEST": signals(scores)})
    assert len(rows) == 5
    assert {r["label"] for r in rows} == {"baseline", "tight stop", "let it run", "high conviction", "trailing"}
    returns = [r["total_return_pct"] for r in rows]
    assert returns == sorted(returns, reverse=True)


def test_legacy_run_portfolio_and_summarize_still_work():
    volumes, scores = _entry_setup()
    bt = SqueezeBacktester(entry_score_threshold=70, take_profit=0.2, stop_loss=-0.1)
    trades = bt.run_portfolio({"TEST": frame([10.0] * 30, volumes=volumes)}, {"TEST": signals(scores)})
    assert isinstance(trades, pd.DataFrame)
    assert "total_trades" in bt.summarize()
