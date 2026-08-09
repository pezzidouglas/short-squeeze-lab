"""
Field-level consolidation across sources.

Free short-interest data disagrees constantly — different settlement dates,
different float definitions, different refresh lags. The usual fix is to pick
one vendor and pretend. This module does the opposite: it keeps every source's
answer, picks a winner per field by explicit priority, and reports how far apart
the sources were, so a score built on a contested number can be discounted
rather than trusted blindly.

Outputs per ticker:
  values      — the merged record the scoring engine consumes
  provenance  — {field: winning source key}
  candidates  — {field: {source key: value}} (everything that was offered)
  agreement   — {field: 0-1} for fields at least two sources reported
  conflicts   — fields whose spread exceeds the field's tolerance
  confidence  — 0-100 blend of coverage, corroboration, agreement and freshness
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import date, datetime
from typing import Any, Iterable, Sequence

from .config import FIELD_SPECS, KEY_FIELDS, NUMERIC, SERIES, priority_for

# A field whose newest value is older than this loses freshness credit.
SI_STALE_DAYS = 30
# Spread beyond tolerance * this multiple counts as total disagreement.
DISAGREEMENT_SPAN = 5.0


@dataclass
class TickerRecord:
    ticker: str
    values: dict[str, Any] = dataclass_field(default_factory=dict)
    provenance: dict[str, str] = dataclass_field(default_factory=dict)
    candidates: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    agreement: dict[str, float] = dataclass_field(default_factory=dict)
    conflicts: list[dict[str, Any]] = dataclass_field(default_factory=list)
    confidence: float = 0.0
    confidence_parts: dict[str, float] = dataclass_field(default_factory=dict)
    sources: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.values)
        payload.update(
            {
                "ticker": self.ticker,
                "data_confidence": round(self.confidence, 1),
                "confidence_parts": {k: round(v, 3) for k, v in self.confidence_parts.items()},
                "provenance": self.provenance,
                "candidates": self.candidates,
                "field_agreement": {k: round(v, 3) for k, v in self.agreement.items()},
                "conflicts": self.conflicts,
                "sources": self.sources,
            }
        )
        return payload


@dataclass
class ConsolidatedScan:
    records: list[TickerRecord] = dataclass_field(default_factory=list)
    source_keys: list[str] = dataclass_field(default_factory=list)

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def by_ticker(self) -> dict[str, TickerRecord]:
        return {r.ticker: r for r in self.records}

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def consolidate(
    source_records: dict[str, dict[str, dict[str, Any]]],
    tickers: Sequence[str] | None = None,
) -> ConsolidatedScan:
    """Merge {source: {ticker: {field: value}}} into per-ticker records."""
    source_keys = list(source_records.keys())
    universe = _universe(source_records, tickers)

    records: list[TickerRecord] = []
    for ticker in universe:
        offered: dict[str, dict[str, Any]] = {}
        contributing: list[str] = []
        for source_key in source_keys:
            values = (source_records.get(source_key) or {}).get(ticker)
            if not values:
                continue
            contributing.append(source_key)
            for field_name, value in values.items():
                if value is None:
                    continue
                offered.setdefault(field_name, {})[source_key] = value

        record = TickerRecord(ticker=ticker, candidates=offered, sources=contributing)
        for field_name, by_source in offered.items():
            winner, value = _pick(field_name, by_source)
            record.values[field_name] = value
            record.provenance[field_name] = winner
            score, conflict = _assess(field_name, by_source)
            if score is not None:
                record.agreement[field_name] = score
            if conflict:
                record.conflicts.append(conflict)
        record.values["ticker"] = ticker
        record.provenance.setdefault("ticker", "derived")
        record.conflicts.sort(key=lambda c: c["spread"], reverse=True)
        _score_confidence(record)
        records.append(record)

    return ConsolidatedScan(records=records, source_keys=source_keys)


def _universe(
    source_records: dict[str, dict[str, dict[str, Any]]],
    tickers: Sequence[str] | None,
) -> list[str]:
    if tickers:
        return [t.upper() for t in tickers]
    seen: dict[str, None] = {}
    for records in source_records.values():
        for ticker in records:
            seen.setdefault(ticker.upper(), None)
    return list(seen)


def _pick(field_name: str, by_source: dict[str, Any]) -> tuple[str, Any]:
    """Highest-priority source that offered a value wins the field."""
    for source_key in priority_for(field_name):
        if source_key in by_source:
            return source_key, by_source[source_key]
    # Not in the priority list at all — fall back to insertion order.
    source_key = next(iter(by_source))
    return source_key, by_source[source_key]


def _assess(field_name: str, by_source: dict[str, Any]) -> tuple[float | None, dict[str, Any] | None]:
    """Return (agreement 0-1, conflict record) for a field's candidate values."""
    spec = FIELD_SPECS.get(field_name)
    if spec is None or spec.kind == SERIES or len(by_source) < 2:
        return None, None

    if spec.kind == NUMERIC:
        numbers = {k: float(v) for k, v in by_source.items() if isinstance(v, (int, float))}
        if len(numbers) < 2:
            return None, None
        spread = _relative_spread(numbers.values())
        tolerance = max(spec.tolerance, 1e-6)
        if spread <= tolerance:
            agreement = 1.0
        else:
            excess = (spread - tolerance) / (tolerance * DISAGREEMENT_SPAN)
            agreement = max(0.0, 1.0 - excess)
        conflict = None
        if spread > tolerance:
            low = min(numbers.items(), key=lambda kv: kv[1])
            high = max(numbers.items(), key=lambda kv: kv[1])
            conflict = {
                "field": field_name,
                "label": spec.label,
                "spread": round(spread, 4),
                "tolerance": tolerance,
                "agreement": round(agreement, 3),
                "low": {"source": low[0], "value": low[1]},
                "high": {"source": high[0], "value": high[1]},
                "values": {k: v for k, v in sorted(numbers.items())},
            }
        return agreement, conflict

    # Text / date: exact-match agreement, case and whitespace insensitive.
    normalised = {k: str(v).strip().lower() for k, v in by_source.items()}
    distinct = set(normalised.values())
    agreement = 1.0 if len(distinct) == 1 else 0.0
    conflict = None
    if agreement < 1.0:
        conflict = {
            "field": field_name,
            "label": spec.label,
            "spread": 1.0,
            "tolerance": 0.0,
            "agreement": 0.0,
            "values": dict(sorted(by_source.items())),
        }
    return agreement, conflict


def _relative_spread(values: Iterable[float]) -> float:
    values = [v for v in values]
    if not values:
        return 0.0
    low, high = min(values), max(values)
    scale = max(abs(low), abs(high))
    if scale == 0:
        return 0.0
    return (high - low) / scale


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


def _score_confidence(record: TickerRecord) -> None:
    present = [f for f in KEY_FIELDS if record.values.get(f) is not None]
    coverage = len(present) / len(KEY_FIELDS) if KEY_FIELDS else 0.0

    if present:
        corroboration = sum(
            min(len(record.candidates.get(f, {})), 3) / 3 for f in present
        ) / len(present)
    else:
        corroboration = 0.0

    scored = [record.agreement[f] for f in present if f in record.agreement]
    # No second opinion is not the same as agreement — score it as a neutral 0.6.
    agreement = sum(scored) / len(scored) if scored else 0.6

    freshness = _freshness(record.values.get("short_interest_date"))

    confidence = 100 * (0.40 * coverage + 0.25 * corroboration + 0.25 * agreement + 0.10 * freshness)
    record.confidence = max(0.0, min(100.0, confidence))
    record.confidence_parts = {
        "coverage": coverage,
        "corroboration": corroboration,
        "agreement": agreement,
        "freshness": freshness,
    }


def _freshness(si_date: Any) -> float:
    """1.0 for a short-interest date inside the current cycle, decaying after."""
    age = short_interest_age_days(si_date)
    if age is None:
        return 0.5  # unknown vintage
    if age <= SI_STALE_DAYS:
        return 1.0
    return max(0.0, 1.0 - (age - SI_STALE_DAYS) / 90)


def short_interest_age_days(si_date: Any, today: date | None = None) -> int | None:
    if not si_date:
        return None
    text = str(si_date)[:10]
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None
    return max(0, ((today or date.today()) - parsed).days)


# --------------------------------------------------------------------------
# Cross-scan reporting
# --------------------------------------------------------------------------


def disagreement_report(scan: ConsolidatedScan, limit: int = 25) -> list[dict[str, Any]]:
    """The widest source disagreements in a scan, worst first."""
    rows: list[dict[str, Any]] = []
    for record in scan.records:
        for conflict in record.conflicts:
            rows.append({"ticker": record.ticker, **conflict})
    rows.sort(key=lambda r: r["spread"], reverse=True)
    return rows[:limit]


def coverage_matrix(scan: ConsolidatedScan) -> dict[str, Any]:
    """Which source won which field, across the whole scan."""
    wins: dict[str, dict[str, int]] = {}
    offers: dict[str, dict[str, int]] = {}
    for record in scan.records:
        for field_name, source_key in record.provenance.items():
            wins.setdefault(field_name, {})
            wins[field_name][source_key] = wins[field_name].get(source_key, 0) + 1
        for field_name, by_source in record.candidates.items():
            offers.setdefault(field_name, {})
            for source_key in by_source:
                offers[field_name][source_key] = offers[field_name].get(source_key, 0) + 1
    return {
        "tickers": len(scan.records),
        "sources": scan.source_keys,
        "wins": wins,
        "offers": offers,
    }
