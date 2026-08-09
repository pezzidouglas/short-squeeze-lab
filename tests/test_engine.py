"""Scoring: component behaviour, missing-input handling, presets, ranking."""

from __future__ import annotations

import pytest

from squeezelab.config import resolve_weights, setup_label
from squeezelab.consolidate import consolidate
from squeezelab.engine import SqueezeEngine, piecewise

BASE = {
    "ticker": "TEST",
    "short_pct_float": 30.0,
    "short_ratio": 6.0,
    "shares_short": 30e6,
    "shares_short_prior": 25e6,
    "float": 100e6,
    "avg_volume_10d": 4e6,
    "current_price": 5.0,
    "previous_close": 4.9,
    "fifty_two_week_high": 20.0,
    "fifty_two_week_low": 4.0,
    "off_exchange_short_pct": 58.0,
    "off_exchange_trend": 2.0,
}


def with_(**overrides):
    return {**BASE, **overrides}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_piecewise_interpolates_and_clamps():
    points = [(0, 0), (10, 50), (20, 100)]
    assert piecewise(-5, points) == 0
    assert piecewise(5, points) == 25
    assert piecewise(15, points) == 75
    assert piecewise(99, points) == 100


def test_resolve_weights_normalises_and_validates():
    weights = resolve_weights("classic")
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["si"] == pytest.approx(0.40)
    with pytest.raises(ValueError):
        resolve_weights("nonsense")
    with pytest.raises(ValueError):
        resolve_weights("classic", {"bogus": 1.0})


@pytest.mark.parametrize("score, label", [(95, "HIGH SQUEEZE RISK"), (70, "ELEVATED"), (45, "WATCH"), (10, "LOW")])
def test_setup_label_bands(score, label):
    assert setup_label(score)[0] == label


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def test_components_are_monotonic_in_their_driver():
    engine = SqueezeEngine(preset="balanced")
    assert engine.score_si_density(with_(short_pct_float=40)).score > engine.score_si_density(with_(short_pct_float=10)).score
    assert engine.score_borrow_pressure(with_(short_ratio=10)).score > engine.score_borrow_pressure(with_(short_ratio=1)).score
    assert engine.score_liquidity_trap(with_(avg_volume_10d=1e6)).score > engine.score_liquidity_trap(with_(avg_volume_10d=50e6)).score
    assert engine.score_flow(with_(off_exchange_short_pct=65)).score > engine.score_flow(with_(off_exchange_short_pct=35)).score


def test_smaller_float_scores_higher_borrow_pressure():
    engine = SqueezeEngine()
    tight = engine.score_borrow_pressure(with_(float=15e6)).score
    loose = engine.score_borrow_pressure(with_(float=2e9)).score
    assert tight > loose


def test_price_near_the_low_is_more_momentum_fuel():
    engine = SqueezeEngine()
    coiled = engine.score_momentum_fuel(with_(current_price=4.5)).score
    extended = engine.score_momentum_fuel(with_(current_price=19.0)).score
    assert coiled > extended


def test_si_density_falls_back_to_shares_over_float():
    engine = SqueezeEngine()
    result = engine.score_si_density({"shares_short": 25e6, "float": 100e6})
    assert result.inputs["short_pct_float"] == pytest.approx(25.0)


def test_liquidity_derives_shares_short_when_absent():
    engine = SqueezeEngine()
    result = engine.score_liquidity_trap({"short_pct_float": 20, "float": 100e6, "avg_volume_10d": 2e6})
    assert result.inputs["effective_days_to_cover"] == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Missing inputs
# --------------------------------------------------------------------------


def test_missing_component_is_dropped_not_scored_zero():
    engine = SqueezeEngine(preset="balanced")
    full = engine.score(BASE)
    no_flow = engine.score({k: v for k, v in BASE.items() if not k.startswith("off_exchange")})

    assert "flow" in no_flow["components_missing"]
    assert no_flow["flow_score"] is None
    # Scoring it zero would have dragged the composite down by ~12 points.
    assert no_flow["squeeze_score"] > full["squeeze_score"] - 6
    # The surviving weights are renormalised back to 1.
    effective = sum(c["effective_weight"] for c in no_flow["components"])
    assert effective == pytest.approx(1.0)


def test_evidence_weight_reports_how_much_of_the_model_ran():
    engine = SqueezeEngine(preset="balanced")
    assert engine.score(BASE)["evidence_weight"] == pytest.approx(1.0)
    thin = engine.score({"ticker": "X", "short_pct_float": 30})
    assert 0 < thin["evidence_weight"] < 1.0


def test_a_record_with_nothing_scores_zero_rather_than_crashing():
    block = SqueezeEngine().score({"ticker": "EMPTY"})
    assert block["squeeze_score"] == 0.0
    assert len(block["components_missing"]) == 5


# --------------------------------------------------------------------------
# Presets & composite
# --------------------------------------------------------------------------


def test_classic_preset_ignores_the_flow_component():
    block = SqueezeEngine(preset="classic").score(BASE)
    flow = next(c for c in block["components"] if c["key"] == "flow")
    assert flow["weight"] == 0.0
    assert flow["points"] == 0.0


def test_legacy_keyword_arguments_still_select_classic_weights():
    engine = SqueezeEngine(si_weight=0.40, borrow_weight=0.25, liquidity_weight=0.20, momentum_weight=0.15)
    assert engine.preset == "classic"
    assert engine.weights["si"] == pytest.approx(0.40)


def test_flow_preset_rewards_off_exchange_shorting():
    hot = with_(off_exchange_short_pct=68, off_exchange_trend=6)
    cool = with_(off_exchange_short_pct=38, off_exchange_trend=-6)
    engine = SqueezeEngine(preset="flow")
    assert engine.score(hot)["squeeze_score"] > engine.score(cool)["squeeze_score"] + 10


def test_component_points_sum_to_the_composite():
    block = SqueezeEngine(preset="balanced").score(BASE)
    assert sum(c["points"] for c in block["components"]) == pytest.approx(block["squeeze_score"], abs=0.05)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_rank_sorts_and_adds_percentiles(service):
    scan, _ = service.consolidated()
    rows = SqueezeEngine(preset="balanced").rank(scan)
    scores = [r["squeeze_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["rank"] == 1
    assert rows[0]["percentile"] == 100.0
    assert all("sector_percentile" in r for r in rows)


def test_confidence_adjusted_score_never_exceeds_the_raw_score(service):
    scan, _ = service.consolidated()
    for row in SqueezeEngine().rank(scan):
        assert row["confidence_adjusted_score"] <= row["squeeze_score"] + 1e-9


def test_rank_accepts_plain_dicts_and_frames():
    engine = SqueezeEngine()
    rows = engine.rank([BASE])
    assert rows[0]["ticker"] == "TEST"

    frame = engine.rank_frame(consolidate({"yahoo": {"TEST": BASE}}, ["TEST"]))
    assert "squeeze_score" in frame.columns
    assert "candidates" not in frame.columns


def test_sector_ranking_is_within_sector():
    rows = SqueezeEngine().rank([
        with_(ticker="A", sector="Tech", short_pct_float=40),
        with_(ticker="B", sector="Tech", short_pct_float=10),
        with_(ticker="C", sector="Health", short_pct_float=12),
    ])
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["A"]["sector_rank"] == 1
    assert by_ticker["B"]["sector_rank"] == 2
    assert by_ticker["C"]["sector_rank"] == 1
    assert by_ticker["C"]["sector_peers"] == 1
