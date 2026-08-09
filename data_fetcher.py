"""
Compatibility shim for the 1.x module layout.

The data layer now lives in `squeezelab.sources` (several providers, merged with
provenance) and `squeezelab.service`. `DataFetcher` is kept so existing scripts
and notebooks keep working; new code should use `MarketDataService`.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd

from squeezelab.config import DEFAULT_WATCHLIST as _WATCHLIST
from squeezelab.engine import SqueezeEngine
from squeezelab.service import MarketDataService

DEFAULT_WATCHLIST = list(_WATCHLIST)

__all__ = ["DEFAULT_WATCHLIST", "DataFetcher"]


class DataFetcher:
    """Thin wrapper over `MarketDataService` with the old method names."""

    def __init__(self, watchlist: Sequence[str] | None = None) -> None:
        warnings.warn(
            "data_fetcher.DataFetcher is deprecated; use squeezelab.MarketDataService",
            DeprecationWarning,
            stacklevel=2,
        )
        self.watchlist = list(watchlist or DEFAULT_WATCHLIST)
        self.service = MarketDataService(watchlist=self.watchlist)

    def fetch_all(self, tickers: Sequence[str] | None = None) -> pd.DataFrame:
        scan, _ = self.service.consolidated(tickers or self.watchlist)
        rows = [{k: v for k, v in r.values.items() if not isinstance(v, (dict, list))} for r in scan.records]
        return pd.DataFrame(rows)

    def fetch_yf_info(self, ticker: str) -> dict | None:
        scan, _ = self.service.consolidated([ticker])
        record = scan.by_ticker().get(ticker.upper())
        return dict(record.values) if record else None

    def fetch_price_history(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame | None:
        return self.service.fetch_prices(ticker, period=period)

    def fetch_batch_prices(self, tickers: Sequence[str] | None = None, period: str = "1y") -> pd.DataFrame:
        frames = self.service.price_history(tickers or self.watchlist, period=period)
        return pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        return SqueezeEngine().rank_frame(df)
