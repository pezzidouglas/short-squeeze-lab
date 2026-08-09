"""
Comparison views.

Three questions the dashboard needs answered, none of which a flat table can:

  compare_tickers  — put 2-4 names side by side on the metrics that drive the
                     score, mark which one leads each row, and express each
                     value as a z-score against the scanned universe so "34%
                     short" reads as "two standard deviations above the field"
  compare_sources  — for one ticker, show what every source said about every
                     field, who won, and how far apart they were
  compare_presets  — rescore the same data under each weight preset, so you can
                     see which names are robust and which only look good under
                     one set of assumptions
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .config import COMPONENT_LABELS, FIELD_SPECS, WEIGHT_PRESETS
from .consolidate import ConsolidatedScan, TickerRecord
from .engine import COMPONENTS, SqueezeEngine

# Metrics shown in a head-to-head, and which direction favours a squeeze.
# `higher_is_hotter` is about squeeze potential, not about being a good stock.
COMPARE_METRICS: tuple[tuple[str, bool], ...] = (
    ("squeeze_score", True),
    ("confidence_adjusted_score", True),
    ("data_confidence", True),
    ("short_pct_float", True),
    ("short_ratio", True),
    ("off_exchange_short_pct", True),
    ("off_exchange_trend", True),
    ("float", False),
    ("avg_volume_10d", False),
    ("market_cap", False),
    ("current_price", False),
)

EXTRA_LABELS = {
    "squeeze_score": ("Squeeze score", ""),
    "confidence_adjusted_score": ("Confidence-adjusted score", ""),
    "data_confidence": ("Data confidence", "%"),
}


def _label(field: str) -> tuple[str, str]:
    if field in EXTRA_LABELS:
        return EXTRA_LABELS[field]
    spec = FIELD_SPECS.get(field)
    return (spec.label, spec.unit) if spec else (field, "")


def _value(row: dict[str, Any], field: str) -> float | None:
    raw = row.get(field)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed


# --------------------------------------------------------------------------
# Ticker head-to-head
# --------------------------------------------------------------------------


def compare_tickers(rows: Sequence[dict[str, Any]], tickers: Sequence[str]) -> dict[str, Any]:
    """Side-by-side comparison of up to a handful of tickers."""
    wanted = [t.upper() for t in tickers]
    by_ticker = {str(r.get("ticker", "")).upper(): r for r in rows}
    selected = [by_ticker[t] for t in wanted if t in by_ticker]
    missing = [t for t in wanted if t not in by_ticker]

    stats = _universe_stats(rows)

    metrics: list[dict[str, Any]] = []
    for field, higher_is_hotter in COMPARE_METRICS:
        values = {str(r["ticker"]).upper(): _value(r, field) for r in selected}
        present = {k: v for k, v in values.items() if v is not None}
        label, unit = _label(field)
        leader = None
        if present:
            leader = (max if higher_is_hotter else min)(present, key=lambda k: present[k])
        metrics.append(
            {
                "field": field,
                "label": label,
                "unit": unit,
                "higher_is_hotter": higher_is_hotter,
                "values": values,
                "z_scores": {k: _z(v, stats.get(field)) for k, v in values.items()},
                "leader": leader,
            }
        )

    components = {
        str(r["ticker"]).upper(): [
            {
                "key": key,
                "label": COMPONENT_LABELS[key],
                "score": r.get(f"{key}_score"),
            }
            for key in COMPONENTS
        ]
        for r in selected
    }

    return {
        "tickers": [str(r["ticker"]).upper() for r in selected],
        "missing": missing,
        "names": {str(r["ticker"]).upper(): r.get("name") or r.get("ticker") for r in selected},
        "metrics": metrics,
        "components": components,
        "universe_size": len(rows),
        "verdict": _verdict(selected),
    }


def _universe_stats(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """(mean, stdev) per field across the scanned universe."""
    stats: dict[str, tuple[float, float]] = {}
    for field, _ in COMPARE_METRICS:
        values = [v for v in (_value(r, field) for r in rows) if v is not None]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        stats[field] = (mean, math.sqrt(variance))
    return stats


def _z(value: float | None, stat: tuple[float, float] | None) -> float | None:
    if value is None or stat is None:
        return None
    mean, deviation = stat
    if deviation == 0:
        return 0.0
    return round((value - mean) / deviation, 2)


def _verdict(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A one-line read: who leads, on what, and whether the data backs it."""
    if not selected:
        return {}
    leader = max(selected, key=lambda r: float(r.get("squeeze_score") or 0))
    strongest = None
    best_score = -1.0
    for key in COMPONENTS:
        score = leader.get(f"{key}_score")
        if score is not None and float(score) > best_score:
            best_score, strongest = float(score), key
    return {
        "leader": str(leader.get("ticker", "")).upper(),
        "score": leader.get("squeeze_score"),
        "driver": COMPONENT_LABELS.get(strongest or "", ""),
        "driver_score": round(best_score, 1) if best_score >= 0 else None,
        "confidence": leader.get("data_confidence"),
    }


# --------------------------------------------------------------------------
# Source-vs-source
# --------------------------------------------------------------------------


def compare_sources(record: TickerRecord | dict[str, Any]) -> dict[str, Any]:
    """Everything each source said about one ticker, field by field."""
    if isinstance(record, TickerRecord):
        candidates = record.candidates
        provenance = record.provenance
        agreement = record.agreement
        conflicts = record.conflicts
        ticker = record.ticker
        confidence = record.confidence
        parts = record.confidence_parts
        sources = record.sources
    else:
        candidates = record.get("candidates") or {}
        provenance = record.get("provenance") or {}
        agreement = record.get("field_agreement") or {}
        conflicts = record.get("conflicts") or []
        ticker = str(record.get("ticker", "")).upper()
        confidence = float(record.get("data_confidence") or 0)
        parts = record.get("confidence_parts") or {}
        sources = record.get("sources") or []

    fields: list[dict[str, Any]] = []
    for field, by_source in sorted(candidates.items()):
        if field == "ticker":
            continue
        label, unit = _label(field)
        fields.append(
            {
                "field": field,
                "label": label,
                "unit": unit,
                "winner": provenance.get(field),
                "agreement": agreement.get(field),
                "sources": len(by_source),
                "values": {k: _jsonable(v) for k, v in sorted(by_source.items())},
            }
        )
    # Contested fields first — that is what a reader is looking for here.
    fields.sort(key=lambda f: (f["agreement"] if f["agreement"] is not None else 1.1, -f["sources"]))

    return {
        "ticker": ticker,
        "sources": sources,
        "confidence": round(confidence, 1),
        "confidence_parts": parts,
        "fields": fields,
        "conflicts": conflicts,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return f"{len(value)} points"
    return value


# --------------------------------------------------------------------------
# Preset sensitivity
# --------------------------------------------------------------------------


def compare_presets(
    scan: ConsolidatedScan,
    presets: Iterable[str] | None = None,
    top: int = 15,
) -> dict[str, Any]:
    """Rescore the same scan under every weight preset.

    A name that tops the table under one preset and sinks under another is a
    name whose ranking is an artefact of the weights, not the data. `spread` is
    the gap between a ticker's best and worst score across presets.
    """
    names = list(presets or WEIGHT_PRESETS.keys())
    scored: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for preset in names:
        rows = SqueezeEngine(preset=preset).rank(scan)
        for row in rows:
            ticker = str(row["ticker"]).upper()
            scored.setdefault(ticker, {})[preset] = row["squeeze_score"]
            ranks.setdefault(ticker, {})[preset] = row["rank"]

    table: list[dict[str, Any]] = []
    for ticker, by_preset in scored.items():
        values = list(by_preset.values())
        table.append(
            {
                "ticker": ticker,
                "scores": by_preset,
                "ranks": ranks.get(ticker, {}),
                "mean": round(sum(values) / len(values), 2),
                "best": round(max(values), 2),
                "worst": round(min(values), 2),
                "spread": round(max(values) - min(values), 2),
            }
        )
    table.sort(key=lambda r: r["mean"], reverse=True)
    return {"presets": names, "rows": table[:top]}
