"""
Shared test setup.

The environment is pinned before `squeezelab` is imported so no test can reach
the network, touch the developer's cache, or depend on which providers happen
to be configured on the machine running them.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("SQUEEZE_OFFLINE", "1")
os.environ.setdefault("SQUEEZE_SOURCES", "snapshot")
os.environ.setdefault("SQUEEZE_CACHE_TTL", "0")
os.environ.setdefault("SQUEEZE_DATA_DIR", tempfile.mkdtemp(prefix="squeeze-tests-"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from squeezelab.config import Settings  # noqa: E402
from squeezelab.service import MarketDataService  # noqa: E402
from squeezelab.sources.snapshot import SnapshotSource  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        cache_ttl_seconds=0,
        scan_cache_ttl=600,
        offline=True,
        enabled_sources=("snapshot",),
        max_workers=2,
    )


@pytest.fixture
def snapshot_source(settings) -> SnapshotSource:
    return SnapshotSource(settings)


@pytest.fixture
def service(settings, snapshot_source) -> MarketDataService:
    return MarketDataService(settings, sources=[snapshot_source], watchlist=snapshot_source.tickers())


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """A short, fully explicit OHLCV frame — every assertion can be read off it."""
    dates = pd.bdate_range("2026-01-01", periods=40)
    rows = []
    for i, day in enumerate(dates):
        close = 10 + i * 0.1
        rows.append(
            {
                "Date": day,
                "Open": close - 0.05,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1_000_000,
                "ticker": "TEST",
            }
        )
    return pd.DataFrame(rows)
