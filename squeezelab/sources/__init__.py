"""
Source registry.

Every source is registered here by key. `build_sources` turns the configured
list of keys into live instances; unknown keys are ignored rather than fatal so
a typo in `SQUEEZE_SOURCES` degrades instead of crashing a scheduled scan.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..config import Settings
from .base import DataSource, DiskCache, SourceError, SourceHealth, SourceMeta, SourceRecords
from .borrow import BorrowSource
from .finra import FinraSource
from .finviz import FinvizSource
from .options import OptionsSource
from .sec_edgar import SecEdgarSource
from .sec_ftd import SecFtdSource
from .snapshot import SnapshotSource
from .stooq import StooqSource
from .yahoo import YahooSource

SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "yahoo": YahooSource,
    "stooq": StooqSource,
    "finviz": FinvizSource,
    "finra": FinraSource,
    "sec": SecEdgarSource,
    "sec_ftd": SecFtdSource,
    "borrow": BorrowSource,
    "options": OptionsSource,
    "snapshot": SnapshotSource,
}

# Sources whose numbers need a denominator — float, spot price — that they
# deliberately do not fetch themselves, because refetching it would duplicate
# Yahoo/Finviz at the wrong layer. `MarketDataService` collects these in a
# second pass and injects the lookups from what the first pass returned; run
# concurrently with everything else they would simply have no denominator and
# would drop their only cross-ticker-comparable fields.
DEPENDENT_SOURCES: frozenset[str] = frozenset({"sec_ftd", "options"})

__all__ = [
    "DataSource",
    "DiskCache",
    "SourceError",
    "SourceHealth",
    "SourceMeta",
    "SourceRecords",
    "SOURCE_REGISTRY",
    "DEPENDENT_SOURCES",
    "build_sources",
    "describe_sources",
    "BorrowSource",
    "FinraSource",
    "FinvizSource",
    "OptionsSource",
    "SecEdgarSource",
    "SecFtdSource",
    "SnapshotSource",
    "StooqSource",
    "YahooSource",
]


def build_sources(keys: Sequence[str], settings: Settings | None = None) -> list[DataSource]:
    """Instantiate the requested sources, skipping unknown keys."""
    sources: list[DataSource] = []
    for key in keys:
        cls = SOURCE_REGISTRY.get(key.strip().lower())
        if cls is None:
            continue
        sources.append(cls(settings))
    return sources


def describe_sources(keys: Iterable[str] | None = None) -> list[dict]:
    """Static metadata for every known source — used by /api/sources."""
    selected = list(keys) if keys is not None else list(SOURCE_REGISTRY)
    out = []
    for key in selected:
        cls = SOURCE_REGISTRY.get(key)
        if cls is not None:
            out.append(cls.meta.to_dict())
    return out
