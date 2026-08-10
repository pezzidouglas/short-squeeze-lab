"""
Scan orchestration — specifically the two-pass collection.

Fails-to-deliver and option open interest are only comparable across tickers
once divided by a float, and the option near-the-money band needs a spot price
to exist at all. Neither source fetches those, deliberately. If the service
ran every source in one concurrent pass, both would silently drop their only
cross-ticker-comparable field and nothing would report a problem — which is
exactly the failure these tests exist to catch.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from squeezelab.service import MarketDataService
from squeezelab.sources import DEPENDENT_SOURCES, SOURCE_REGISTRY
from squeezelab.sources.base import DataSource, SourceMeta, SourceRecords


class _Primary(DataSource):
    """Stands in for Yahoo/Finviz: supplies the denominators."""

    meta = SourceMeta(
        key="yahoo", name="Primary", url="", cadence="daily",
        provides=("float", "current_price"), requires_network=False,
    )

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        return {t: {"float": 100e6, "current_price": 5.0} for t in tickers}


class _Dependent(DataSource):
    """Stands in for sec_ftd/options: records what it was handed."""

    meta = SourceMeta(
        key="options", name="Dependent", url="", cadence="daily",
        provides=("options_ntm_call_oi_pct_float",), requires_network=False,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.float_lookup = None
        self.price_lookup = None
        self.seen: dict[str, tuple[float | None, float | None]] = {}

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        out: SourceRecords = {}
        for ticker in tickers:
            shares = self.float_lookup(ticker) if self.float_lookup else None
            spot = self.price_lookup(ticker) if self.price_lookup else None
            self.seen[ticker] = (shares, spot)
            if shares:
                out[ticker] = {"options_ntm_call_oi_pct_float": 8e6 / shares * 100}
        return out


@pytest.fixture
def two_pass(settings) -> tuple[MarketDataService, _Dependent]:
    dependent = _Dependent(settings)
    service = MarketDataService(settings, sources=[_Primary(settings), dependent], watchlist=["GME"])
    return service, dependent


def test_the_registry_and_the_dependent_set_agree():
    """A key here that no longer exists in the registry would silently move a
    source back into the first pass, where it has no denominator."""
    assert DEPENDENT_SOURCES <= set(SOURCE_REGISTRY)


def test_dependent_sources_receive_the_first_pass_denominators(two_pass):
    service, dependent = two_pass
    collected, _ = service.collect(["GME"])
    assert dependent.seen["GME"] == (100e6, 5.0)
    assert collected["options"]["GME"]["options_ntm_call_oi_pct_float"] == pytest.approx(8.0)


def test_a_lookup_the_caller_already_wired_is_left_alone(settings):
    """Injecting a stub is how these sources are tested; the service must not
    overwrite one that is already set."""
    dependent = _Dependent(settings)
    dependent.float_lookup = lambda ticker: 50e6
    service = MarketDataService(settings, sources=[_Primary(settings), dependent], watchlist=["GME"])
    service.collect(["GME"])
    assert dependent.seen["GME"] == (50e6, 5.0)


def test_a_denominator_no_source_reported_reads_as_absent(settings):
    """Not as zero, and not as a crash — the dependent source is expected to
    degrade to the fields that need no denominator."""
    class _Empty(_Primary):
        def _fetch(self, tickers):
            return {t: {"name": "Anon"} for t in tickers}

    dependent = _Dependent(settings)
    service = MarketDataService(settings, sources=[_Empty(settings), dependent], watchlist=["GME"])
    collected, _ = service.collect(["GME"])
    assert dependent.seen["GME"] == (None, None)
    assert collected["options"] == {}


def test_the_scored_row_divides_by_the_same_float_the_engine_used(two_pass):
    """A percentage computed against a different denominator than the one on
    screen is worse than no percentage."""
    service, _ = two_pass
    result = service.scan(["GME"])
    row = result.rows[0]
    implied = row["options_ntm_call_oi_pct_float"] / 100 * row["float"]
    assert implied == pytest.approx(8e6)
