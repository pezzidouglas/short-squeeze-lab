"""
Squeeze scoring engine.

Six components, each scored 0-100 from its own inputs and then weighted:

  si        short interest as a share of float — the raw fuel
  borrow    what a short costs and how scarce the stock is — the cost of being wrong
  liquidity short position measured against real daily volume — the trap
  momentum  price compression plus a rising short position — the spark
  flow      FINRA off-exchange short volume and its trend — is it still on?
  gamma     near-the-money call open interest against the float — who is forced to buy

Two things differ from a naive weighted average, and both matter:

* A component with no inputs is *dropped and the remaining weights are
  renormalised*, rather than scored zero. A missing short-interest feed used to
  quietly push a name to the bottom of the table; now it shrinks the evidence
  base and shows up as reduced data confidence instead.
* The raw score is reported alongside a confidence-adjusted score. The raw
  number answers "how squeezy does this look"; the adjusted one answers "how
  much of that do the sources actually support".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import COMPONENT_LABELS, resolve_weights, setup_label
from .consolidate import ConsolidatedScan, TickerRecord, short_interest_age_days

COMPONENTS: tuple[str, ...] = ("si", "borrow", "liquidity", "momentum", "flow", "gamma")


@dataclass
class ComponentResult:
    key: str
    score: float | None
    inputs: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def piecewise(value: float, points: Sequence[tuple[float, float]]) -> float:
    """Interpolate `value` through ascending (input, score) breakpoints."""
    if value <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            if x1 == x0:
                return y1
            return y0 + (value - x0) / (x1 - x0) * (y1 - y0)
    return points[-1][1]


def _num(values: dict[str, Any], key: str) -> float | None:
    raw = values.get(key)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


class SqueezeEngine:
    def __init__(
        self,
        preset: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        # Legacy keyword arguments from the 1.x engine still work.
        si_weight: float | None = None,
        borrow_weight: float | None = None,
        liquidity_weight: float | None = None,
        momentum_weight: float | None = None,
        flow_weight: float | None = None,
    ) -> None:
        legacy = {
            "si": si_weight,
            "borrow": borrow_weight,
            "liquidity": liquidity_weight,
            "momentum": momentum_weight,
            "flow": flow_weight,
        }
        legacy = {k: v for k, v in legacy.items() if v is not None}
        if legacy and preset is None:
            preset = "classic"
        overrides = {**legacy, **(weight_overrides or {})}
        self.preset = (preset or "balanced").lower()
        self.weights = resolve_weights(self.preset, overrides or None)

    # -- components -------------------------------------------------------

    def score_si_density(self, values: dict[str, Any]) -> ComponentResult:
        """Short interest as a percentage of float."""
        si = _num(values, "short_pct_float")
        if si is None:
            shares_short = _num(values, "shares_short")
            float_shares = _num(values, "float")
            if shares_short and float_shares:
                si = shares_short / float_shares * 100
        if si is None:
            return ComponentResult("si", None, note="no short interest reported")
        score = piecewise(si, [(0, 0), (5, 20), (10, 38), (15, 55), (20, 68), (30, 88), (40, 97), (60, 100)])
        return ComponentResult("si", clamp(score), {"short_pct_float": round(si, 2)})

    def score_borrow_pressure(self, values: dict[str, Any]) -> ComponentResult:
        """What a short actually costs, and how scarce the stock is.

        Days-to-cover and float size are *proxies* for the price of borrow.
        When a stock-loan feed is enabled the price itself is available, so it
        takes the largest share and the proxies are demoted rather than dropped
        — one broker's indications are not the whole market.

        The proxy-only weights (0.28 / 0.12) are deliberately in the same 7:3
        ratio the component used before any loan data existed: `_weighted`
        renormalises over whatever is present, so a scan without `borrow`
        enabled scores identically to how it always did.
        """
        dtc = _num(values, "short_ratio")
        float_shares = _num(values, "float") or _num(values, "shares_outstanding")
        fee = _num(values, "borrow_fee_pct")
        available = _num(values, "shortable_shares")
        ftd_pct = _num(values, "ftd_pct_float")
        if dtc is None and float_shares is None and fee is None and available is None:
            return ComponentResult("borrow", None, note="no borrow rate, days-to-cover or float")

        parts: list[tuple[float, float]] = []
        inputs: dict[str, Any] = {}

        if fee is not None:
            # General collateral sits under ~1%/yr and says nothing. The curve
            # is steep through the single digits, where a name stops being
            # freely borrowable, and flattens past 50% where the rate is
            # already prohibitive and more of it changes no decision.
            parts.append((
                clamp(piecewise(fee, [(0, 0), (1, 8), (3, 30), (5, 45), (10, 62), (25, 80), (50, 92), (100, 100)])),
                0.40,
            ))
            inputs["borrow_fee_pct"] = round(fee, 2)

        if dtc is not None:
            dtc_score = piecewise(dtc, [(0, 0), (1, 18), (2, 35), (3, 52), (5, 76), (8, 92), (12, 100)])
            parts.append((dtc_score, 0.28))
            inputs["short_ratio"] = round(dtc, 2)

        if float_shares:
            # 10M float is as scarce as it gets; past 1B scarcity stops mattering.
            span = math.log10(1e9) - math.log10(1e7)
            scarcity = (math.log10(1e9) - math.log10(max(float_shares, 1e5))) / span
            scarcity_score = clamp(scarcity * 100)
            parts.append((scarcity_score, 0.12))
            inputs["float"] = float_shares

        if available is not None and float_shares:
            # Inventory left, as a share of float. Zero available is the single
            # most squeeze-relevant reading in the file — no new short can be
            # opened at that broker at any price — so it must score 100, which
            # is why this reads a count rather than testing it for truthiness.
            supply_pct = available / float_shares * 100
            basis = values.get("shortable_shares_basis")
            score = clamp(piecewise(supply_pct, [(0, 100), (0.05, 82), (0.2, 62), (0.5, 45), (1.0, 30), (3.0, 12), (8.0, 0)]))
            # A censored ">N" row is a floor on supply, so the real pressure is
            # at most what the floor implies. Scoring it as exact would invent
            # scarcity for the liquid names a human checks first.
            weight = 0.12 if basis != "floor" else 0.05
            parts.append((score, weight))
            inputs["shortable_pct_float"] = round(supply_pct, 4)
            if basis:
                inputs["shortable_shares_basis"] = basis

        if ftd_pct is not None:
            # Fails corroborate a hard borrow — the delivery leg is harder to
            # source — but the freshest datum is ~40 days old and fails arise
            # overwhelmingly from operational causes. Smallest weight in the
            # component, and never on its own: see `sec_ftd`'s module docstring.
            parts.append((
                clamp(piecewise(ftd_pct, [(0, 0), (0.05, 20), (0.2, 45), (0.5, 65), (1.0, 82), (3.0, 100)])),
                0.08,
            ))
            inputs["ftd_pct_float"] = round(ftd_pct, 4)
            if values.get("ftd_date"):
                # The vintage travels with the number, always.
                inputs["ftd_date"] = values["ftd_date"]

        return ComponentResult("borrow", _weighted(parts), inputs)

    def score_liquidity_trap(self, values: dict[str, Any]) -> ComponentResult:
        """The short position measured in days of *real* traded volume."""
        avg_vol = _num(values, "avg_volume_10d") or _num(values, "avg_volume_3m")
        shares_short = _num(values, "shares_short")
        if shares_short is None:
            si = _num(values, "short_pct_float")
            float_shares = _num(values, "float")
            if si and float_shares:
                shares_short = si / 100 * float_shares
        if not avg_vol or shares_short is None:
            return ComponentResult("liquidity", None, note="no volume or short share count")
        effective_dtc = shares_short / avg_vol
        score = piecewise(effective_dtc, [(0, 0), (1, 22), (2, 40), (3, 55), (5, 72), (8, 90), (12, 100)])
        return ComponentResult(
            "liquidity",
            clamp(score),
            {"effective_days_to_cover": round(effective_dtc, 2), "avg_volume": avg_vol},
        )

    def score_momentum_fuel(self, values: dict[str, Any]) -> ComponentResult:
        """Compression against the 52-week range, plus a growing short position."""
        price = _num(values, "current_price")
        high = _num(values, "fifty_two_week_high")
        low = _num(values, "fifty_two_week_low")
        parts: list[tuple[float, float]] = []
        inputs: dict[str, Any] = {}

        if price and high and low and high > low:
            position = clamp((price - low) / (high - low), 0.0, 1.0)
            # Coiled near the low is fuel; already extended is not.
            parts.append((clamp(100 - position * 100), 0.55))
            inputs["range_position_pct"] = round(position * 100, 1)

        now = _num(values, "shares_short")
        prior = _num(values, "shares_short_prior")
        if now and prior and prior > 0:
            change = (now - prior) / prior * 100
            parts.append((clamp(piecewise(change, [(-30, 0), (-10, 20), (0, 40), (10, 65), (25, 88), (50, 100)])), 0.30))
            inputs["short_change_pct"] = round(change, 1)

        prev = _num(values, "previous_close")
        if price and prev and prev > 0:
            day_change = (price - prev) / prev * 100
            parts.append((clamp(piecewise(day_change, [(-10, 0), (-3, 25), (0, 45), (3, 70), (8, 90), (15, 100)])), 0.15))
            inputs["day_change_pct"] = round(day_change, 2)

        if not parts:
            return ComponentResult("momentum", None, note="no price range or short interest history")
        return ComponentResult("momentum", _weighted(parts), inputs)

    def score_flow(self, values: dict[str, Any]) -> ComponentResult:
        """FINRA off-exchange short volume — is shorting still happening now?"""
        ratio = _num(values, "off_exchange_short_pct")
        if ratio is None:
            return ComponentResult("flow", None, note="no FINRA short volume")
        parts = [(clamp(piecewise(ratio, [(25, 0), (40, 30), (48, 55), (55, 75), (62, 90), (70, 100)])), 0.7)]
        inputs: dict[str, Any] = {"off_exchange_short_pct": round(ratio, 2)}
        trend = _num(values, "off_exchange_trend")
        if trend is not None:
            parts.append((clamp(piecewise(trend, [(-8, 0), (-3, 25), (0, 50), (3, 78), (8, 100)])), 0.3))
            inputs["off_exchange_trend"] = round(trend, 2)
        return ComponentResult("flow", _weighted(parts), inputs)

    def score_gamma_pressure(self, values: dict[str, Any]) -> ComponentResult:
        """Near-the-money call open interest against the float.

        Short interest says how many shares are owed; it says nothing about
        whether anyone is *forced to buy*. Calls already written in strikes a
        plausible move converts to in-the-money oblige whoever is short them to
        buy real shares to stay hedged, into a supply the short interest
        already says is tight.

        Scored only off the share-equivalent percentage of float, because that
        is the only cross-ticker comparable reading the chain produces: a raw
        contract count says nothing without knowing what it is a count against.
        Implied vol is carried as context and deliberately not scored — Yahoo
        inverts it from `lastPrice` rather than a mid, so on a strike that last
        traded days ago it is fiction.
        """
        pct_float = _num(values, "options_ntm_call_oi_pct_float")
        if pct_float is None:
            return ComponentResult(
                "gamma", None, note="no near-the-money call OI measured against a float"
            )

        parts = [(
            clamp(piecewise(pct_float, [(0, 0), (1, 18), (3, 38), (6, 57), (10, 73), (20, 90), (35, 100)])),
            0.7,
        )]
        inputs: dict[str, Any] = {
            "options_ntm_call_oi_pct_float": round(pct_float, 3),
            "options_ntm_call_oi": _num(values, "options_ntm_call_oi"),
        }

        turnover = _num(values, "options_call_volume_oi_ratio")
        if turnover is not None:
            # Open interest is T-1: OCC settles overnight, so on the day a ramp
            # is actually built the level above is stale and this ratio is the
            # only fresh column. Above 1.0 the session traded more contracts
            # than were open — the ladder being read is already wrong, upward.
            parts.append((
                clamp(piecewise(turnover, [(0, 0), (0.2, 25), (0.5, 50), (1.0, 75), (2.0, 92), (4.0, 100)])),
                0.3,
            ))
            inputs["options_call_volume_oi_ratio"] = round(turnover, 2)

        for context in ("options_atm_iv", "options_expiry", "options_window_days"):
            if values.get(context) is not None:
                inputs[context] = values[context]

        return ComponentResult(
            "gamma",
            _weighted(parts),
            inputs,
            note="" if turnover is not None else "open interest is T-1 and no fresh volume corroborates it",
        )

    # -- composite --------------------------------------------------------

    def score(self, values: dict[str, Any]) -> dict[str, Any]:
        """Score one consolidated record. Returns the scoring block only."""
        results = {
            "si": self.score_si_density(values),
            "borrow": self.score_borrow_pressure(values),
            "liquidity": self.score_liquidity_trap(values),
            "momentum": self.score_momentum_fuel(values),
            "flow": self.score_flow(values),
            "gamma": self.score_gamma_pressure(values),
        }

        available = {k: r for k, r in results.items() if r.score is not None and self.weights.get(k, 0) > 0}
        missing = [k for k in COMPONENTS if results[k].score is None]

        if available:
            total_weight = sum(self.weights[k] for k in available)
            effective = {k: self.weights[k] / total_weight for k in available}
            total = sum(available[k].score * effective[k] for k in available)
        else:
            effective = {}
            total = 0.0

        contributions = [
            {
                "key": k,
                "label": COMPONENT_LABELS[k],
                "score": round(results[k].score, 2) if results[k].score is not None else None,
                "weight": round(self.weights.get(k, 0.0), 4),
                "effective_weight": round(effective.get(k, 0.0), 4),
                "points": round(results[k].score * effective[k], 2) if k in effective else 0.0,
                "inputs": results[k].inputs,
                "note": results[k].note,
            }
            for k in COMPONENTS
        ]

        label, tier = setup_label(total)
        block: dict[str, Any] = {
            "squeeze_score": round(total, 2),
            "setup_label": label,
            "setup_tier": tier,
            "components": contributions,
            "components_missing": missing,
            "evidence_weight": round(sum(self.weights[k] for k in available), 4),
            "preset": self.preset,
        }
        for key in COMPONENTS:
            block[f"{key}_score"] = round(results[key].score, 2) if results[key].score is not None else None
        return block

    # -- ranking ----------------------------------------------------------

    def rank(self, scan: ConsolidatedScan | Iterable[Any]) -> list[dict[str, Any]]:
        """Score every record, add ranks and percentiles, sort by score."""
        records = _as_records(scan)
        rows: list[dict[str, Any]] = []
        for record in records:
            row = record.to_dict() if isinstance(record, TickerRecord) else dict(record)
            values = record.values if isinstance(record, TickerRecord) else row
            block = self.score(values)
            row.update(block)
            confidence = float(row.get("data_confidence") or 0.0)
            # Discount the score toward the middle of its band when the inputs
            # are thin — never silently replace the raw number.
            row["confidence_adjusted_score"] = round(
                block["squeeze_score"] * (0.65 + 0.35 * (confidence / 100)), 2
            )
            row["short_interest_age_days"] = short_interest_age_days(values.get("short_interest_date"))
            rows.append(row)

        rows.sort(key=lambda r: (r["squeeze_score"], r.get("data_confidence", 0)), reverse=True)
        _add_percentiles(rows)
        return rows

    def rank_frame(self, scan: ConsolidatedScan | Iterable[Any]):
        """Same as `rank`, as a pandas DataFrame (used by the CLI/CSV export)."""
        import pandas as pd

        rows = self.rank(scan)
        for row in rows:
            row.pop("candidates", None)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _weighted(parts: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(w for _, w in parts)
    if total_weight <= 0:
        return 0.0
    return clamp(sum(score * weight for score, weight in parts) / total_weight)


def _as_records(scan: ConsolidatedScan | Iterable[Any]) -> list[Any]:
    if isinstance(scan, ConsolidatedScan):
        return list(scan.records)
    if hasattr(scan, "to_dict") and hasattr(scan, "columns"):  # pandas DataFrame
        return list(scan.to_dict("records"))
    return list(scan)


def _add_percentiles(rows: list[dict[str, Any]]) -> None:
    count = len(rows)
    if not count:
        return
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        row["percentile"] = round((count - index) / count * 100, 1)

    by_sector: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sector.setdefault(str(row.get("sector") or "Unclassified"), []).append(row)
    for peers in by_sector.values():
        peers.sort(key=lambda r: r["squeeze_score"], reverse=True)
        total = len(peers)
        for index, row in enumerate(peers):
            row["sector_rank"] = index + 1
            row["sector_peers"] = total
            row["sector_percentile"] = round((total - index) / total * 100, 1)
