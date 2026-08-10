"""
The scan service: fan out across sources, consolidate, score.

Sources run concurrently because they are independent and network-bound; one
slow provider should not set the pace for the whole scan. If every live source
comes back empty and snapshot fallback is enabled, the bundled sample is used
and the result is labelled `snapshot` so nothing downstream mistakes it for
live data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .config import DEFAULT_WATCHLIST, SETTINGS, Settings
from .consolidate import ConsolidatedScan, consolidate, coverage_matrix, disagreement_report
from .engine import SqueezeEngine
from .sources import DEPENDENT_SOURCES, DataSource, build_sources


@dataclass
class ScanResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    health: list[dict[str, Any]] = field(default_factory=list)
    data_mode: str = "live"  # live | mixed | snapshot | empty
    preset: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""
    tickers_requested: int = 0
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # The merged records behind `rows`, kept so callers that need to rescore
    # (preset sensitivity, say) don't trigger a second fan-out. Not serialised.
    consolidated_scan: ConsolidatedScan | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "health": self.health,
            "data_mode": self.data_mode,
            "preset": self.preset,
            "weights": self.weights,
            "generated_at": self.generated_at,
            "tickers_requested": self.tickers_requested,
            "disagreements": self.disagreements,
            "coverage": self.coverage,
            "notes": self.notes,
        }


class MarketDataService:
    """Owns the source fleet and turns a watchlist into scored rows."""

    def __init__(
        self,
        settings: Settings | None = None,
        sources: Sequence[DataSource] | None = None,
        watchlist: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.watchlist = [t.upper() for t in (watchlist or DEFAULT_WATCHLIST)]
        if sources is not None:
            self.sources = list(sources)
        else:
            keys = list(self.settings.enabled_sources)
            if self.settings.allow_snapshot_fallback and "snapshot" not in keys:
                keys.append("snapshot")
            self.sources = build_sources(keys, self.settings)

    # -- raw collection ---------------------------------------------------

    def collect(self, tickers: Sequence[str]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[DataSource]]:
        """Run every source concurrently. Returns {source_key: records}."""
        live = [
            s for s in self.sources
            if s.meta.key != "snapshot" and s.meta.key not in DEPENDENT_SOURCES
        ]
        dependent = [s for s in self.sources if s.meta.key in DEPENDENT_SOURCES]
        snapshot = [s for s in self.sources if s.meta.key == "snapshot"]

        collected: dict[str, dict[str, dict[str, Any]]] = {}
        if live:
            collected.update(self._run(live, tickers))

        # Second pass. Fails-to-deliver and option open interest are only
        # comparable across tickers once divided by a float, and the option
        # near-the-money band needs a spot price to exist at all — neither of
        # which those sources fetch, precisely so they do not become a second
        # copy of Yahoo. Both expose the lookups as public attributes for the
        # caller to fill in, which is what this does with the first pass's
        # merged values.
        if dependent:
            self._wire_lookups(dependent, collected, tickers)
            collected.update(self._run(dependent, tickers))

        live_rows = sum(len(r) for r in collected.values())
        # Snapshot only runs when it is needed — as a filler for uncovered
        # tickers, or as the whole scan when nothing else answered.
        if snapshot and (live_rows == 0 or self._missing(collected, tickers)):
            for source in snapshot:
                collected[source.meta.key] = source.collect(tickers)

        return collected, list(self.sources)

    def _run(
        self, sources: Sequence[DataSource], tickers: Sequence[str]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Collect from a batch of sources concurrently."""
        workers = max(1, min(self.settings.max_workers, len(sources)))
        out: dict[str, dict[str, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for source, records in zip(sources, pool.map(lambda s: s.collect(tickers), sources)):
                out[source.meta.key] = records
        return out

    def _wire_lookups(
        self,
        dependent: Sequence[DataSource],
        collected: dict[str, dict[str, dict[str, Any]]],
        tickers: Sequence[str],
    ) -> None:
        """Give the second-pass sources their denominators from the first pass.

        The values are consolidated rather than read off one provider, so the
        float behind `ftd_pct_float` is the same float the scoring engine used
        — a percentage computed against a different denominator than the one on
        screen is worse than no percentage.

        A lookup that is already set (a test wiring a stub, say) is left alone.
        """
        merged = consolidate(collected, tickers).by_ticker()

        def _reader(field_name: str):
            def lookup(ticker: str) -> float | None:
                record = merged.get(ticker.upper())
                if record is None:
                    return None
                value = record.values.get(field_name)
                try:
                    number = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None
                return number if number > 0 else None

            return lookup

        for source in dependent:
            if getattr(source, "float_lookup", None) is None and hasattr(source, "float_lookup"):
                source.float_lookup = _reader("float")
            if getattr(source, "price_lookup", None) is None and hasattr(source, "price_lookup"):
                source.price_lookup = _reader("current_price")

    @staticmethod
    def _missing(collected: dict[str, dict[str, dict[str, Any]]], tickers: Sequence[str]) -> bool:
        covered: set[str] = set()
        for records in collected.values():
            covered.update(records.keys())
        return bool({t.upper() for t in tickers} - covered)

    def consolidated(self, tickers: Sequence[str] | None = None) -> tuple[ConsolidatedScan, list[DataSource]]:
        tickers = [t.upper() for t in (tickers or self.watchlist)]
        collected, sources = self.collect(tickers)
        scan = consolidate(collected, tickers)
        # Drop tickers no source could describe at all.
        scan.records = [r for r in scan.records if r.sources]
        return scan, sources

    # -- scored scan ------------------------------------------------------

    def scan(
        self,
        tickers: Sequence[str] | None = None,
        preset: str | None = None,
        weight_overrides: dict[str, float] | None = None,
    ) -> ScanResult:
        requested = [t.upper() for t in (tickers or self.watchlist)]
        scan, sources = self.consolidated(requested)
        engine = SqueezeEngine(preset=preset or self.settings.preset, weight_overrides=weight_overrides)
        rows = engine.rank(scan)

        health = [s.health.to_dict() for s in sources]
        used = {r_source for record in scan.records for r_source in record.sources}
        live_used = used - {"snapshot"}
        if not used:
            mode = "empty"
        elif not live_used:
            mode = "snapshot"
        elif "snapshot" in used:
            mode = "mixed"
        else:
            mode = "live"

        notes: list[str] = []
        if mode in {"snapshot", "mixed"}:
            notes.append(
                "Some or all rows come from the bundled sample snapshot — no live source covered them."
            )
        for source in sources:
            if source.health.status == "error":
                notes.append(f"{source.meta.name}: {source.health.detail}")

        return ScanResult(
            rows=rows,
            health=health,
            data_mode=mode,
            preset=engine.preset,
            weights=engine.weights,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tickers_requested=len(requested),
            disagreements=disagreement_report(scan),
            coverage=coverage_matrix(scan),
            notes=notes,
            consolidated_scan=scan,
        )

    # -- prices -----------------------------------------------------------

    def fetch_prices(self, ticker: str, period: str = "1y"):
        """First source that returns usable bars wins."""
        for source in self.sources:
            try:
                bars = source.fetch_prices(ticker, period=period)
            except Exception:  # noqa: BLE001 - a broken price feed falls through
                continue
            if bars is not None and len(bars) > 0:
                return bars
        return None

    def price_history(self, tickers: Sequence[str], period: str = "1y") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ticker in tickers:
            bars = self.fetch_prices(ticker, period=period)
            if bars is not None and len(bars) > 0:
                out[ticker.upper()] = bars
        return out
