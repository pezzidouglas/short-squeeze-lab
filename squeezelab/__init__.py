"""
Short Squeeze Lab — multi-source short squeeze detection, scoring and backtesting.

Typical use:

    from squeezelab import MarketDataService

    result = MarketDataService().scan()
    for row in result.rows[:10]:
        print(row["ticker"], row["squeeze_score"], row["data_confidence"])
"""

from .backtest import BacktestResult, SqueezeBacktester, StrategyParams, compare_strategies
from .compare import compare_presets, compare_sources, compare_tickers
from .config import DEFAULT_WATCHLIST, SETTINGS, WEIGHT_PRESETS, Settings
from .consolidate import ConsolidatedScan, TickerRecord, consolidate
from .engine import SqueezeEngine
from .history import build_signals, reconstruct_scores
from .service import MarketDataService, ScanResult

__version__ = "2.0.0"

__all__ = [
    "BacktestResult",
    "ConsolidatedScan",
    "DEFAULT_WATCHLIST",
    "MarketDataService",
    "SETTINGS",
    "ScanResult",
    "Settings",
    "SqueezeBacktester",
    "SqueezeEngine",
    "StrategyParams",
    "TickerRecord",
    "WEIGHT_PRESETS",
    "build_signals",
    "compare_presets",
    "compare_sources",
    "compare_strategies",
    "compare_tickers",
    "consolidate",
    "reconstruct_scores",
    "__version__",
]
