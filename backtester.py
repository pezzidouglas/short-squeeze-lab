"""
Compatibility shim for the 1.x module layout.

The backtester now lives in `squeezelab.backtest`. `SqueezeBacktester` keeps its
`run_portfolio` / `summarize` methods, but the simulation underneath fixes the
same-bar lookahead, counts holds in trading days, applies costs, and tracks a
real portfolio equity curve.
"""

from squeezelab.backtest import (
    BacktestResult,
    SqueezeBacktester,
    StrategyParams,
    compare_strategies,
    summarise,
)

__all__ = [
    "BacktestResult",
    "SqueezeBacktester",
    "StrategyParams",
    "compare_strategies",
    "summarise",
]
