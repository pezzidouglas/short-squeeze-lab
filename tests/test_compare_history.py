"""Comparison views and historical signal reconstruction."""

from __future__ import annotations

import pandas as pd
import pytest

from squeezelab.compare import compare_presets, compare_sources, compare_tickers
from squeezelab.consolidate import consolidate
from squeezelab.engine import SqueezeEngine
from squeezelab.history import build_signals, reconstruct_scores

ROWS = [
    {
        "ticker": "AAA", "name": "Alpha", "squeeze_score": 88.0, "confidence_adjusted_score": 80.0,
        "data_confidence": 82.0, "short_pct_float": 35.0, "short_ratio": 9.0,
        "off_exchange_short_pct": 62.0, "off_exchange_trend": 1.5, "float": 40e6,
        "avg_volume_10d": 800_000, "market_cap": 1.2e9, "current_price": 12.0,
        "si_score": 95.0, "borrow_score": 90.0, "liquidity_score": 100.0,
        "momentum_score": 60.0, "flow_score": 84.0,
    },
    {
        "ticker": "BBB", "name": "Beta", "squeeze_score": 55.0, "confidence_adjusted_score": 48.0,
        "data_confidence": 70.0, "short_pct_float": 12.0, "short_ratio": 2.0,
        "off_exchange_short_pct": 48.0, "off_exchange_trend": -0.5, "float": 900e6,
        "avg_volume_10d": 9_000_000, "market_cap": 3.0e10, "current_price": 40.0,
        "si_score": 45.0, "borrow_score": 30.0, "liquidity_score": 55.0,
        "momentum_score": 70.0, "flow_score": 52.0,
    },
    {
        "ticker": "CCC", "name": "Gamma", "squeeze_score": 30.0, "confidence_adjusted_score": 26.0,
        "data_confidence": 60.0, "short_pct_float": 4.0, "short_ratio": 1.0,
        "off_exchange_short_pct": 41.0, "off_exchange_trend": 0.1, "float": 5e9,
        "avg_volume_10d": 40e6, "market_cap": 5e11, "current_price": 200.0,
        "si_score": 18.0, "borrow_score": 12.0, "liquidity_score": 22.0,
        "momentum_score": 40.0, "flow_score": 33.0,
    },
]


# --------------------------------------------------------------------------
# Ticker head-to-head
# --------------------------------------------------------------------------


def test_compare_tickers_marks_the_leader_per_row():
    payload = compare_tickers(ROWS, ["AAA", "BBB"])
    leaders = {m["field"]: m["leader"] for m in payload["metrics"]}
    assert leaders["squeeze_score"] == "AAA"
    assert leaders["short_pct_float"] == "AAA"
    # Smaller float and thinner volume favour a squeeze, so BBB does not lead.
    assert leaders["float"] == "AAA"
    assert leaders["avg_volume_10d"] == "AAA"


def test_compare_tickers_reports_z_scores_against_the_universe():
    payload = compare_tickers(ROWS, ["AAA", "CCC"])
    si = next(m for m in payload["metrics"] if m["field"] == "short_pct_float")
    assert si["z_scores"]["AAA"] > 0 > si["z_scores"]["CCC"]


def test_compare_tickers_names_missing_symbols():
    payload = compare_tickers(ROWS, ["AAA", "NOPE"])
    assert payload["missing"] == ["NOPE"]
    assert payload["tickers"] == ["AAA"]


def test_compare_tickers_verdict_names_the_driving_component():
    verdict = compare_tickers(ROWS, ["AAA", "BBB"])["verdict"]
    assert verdict["leader"] == "AAA"
    assert verdict["driver"] == "Liquidity trap"     # its highest component, at 100
    assert verdict["driver_score"] == 100.0


def test_compare_tickers_drops_metrics_nobody_reported():
    """A row of dashes says nothing a missing row does not say better — and it
    is what every opt-in feed would print for a scan that did not run it."""
    rows = [{"ticker": "X", "squeeze_score": 10.0}, {"ticker": "Y", "squeeze_score": 20.0}]
    payload = compare_tickers(rows, ["X", "Y"])
    fields = {m["field"] for m in payload["metrics"]}
    assert "squeeze_score" in fields
    assert "short_pct_float" not in fields
    assert "borrow_fee_pct" not in fields


def test_compare_tickers_keeps_a_metric_one_ticker_reported():
    """Partial coverage is a comparison; no coverage is not."""
    rows = [
        {"ticker": "X", "squeeze_score": 10.0, "short_pct_float": 22.0},
        {"ticker": "Y", "squeeze_score": 20.0},
    ]
    payload = compare_tickers(rows, ["X", "Y"])
    si = next(m for m in payload["metrics"] if m["field"] == "short_pct_float")
    assert si["leader"] == "X"
    assert si["values"] == {"X": 22.0, "Y": None}


# --------------------------------------------------------------------------
# Source-vs-source
# --------------------------------------------------------------------------


def test_compare_sources_puts_contested_fields_first():
    scan = consolidate(
        {
            "yahoo": {"GME": {"current_price": 28.4, "sector": "Consumer Cyclical", "short_ratio": 3.2}},
            "stooq": {"GME": {"current_price": 14.0}},
            "finviz": {"GME": {"sector": "Consumer Cyclical"}},
        },
        ["GME"],
    )
    detail = compare_sources(scan.by_ticker()["GME"])
    assert detail["fields"][0]["field"] == "current_price"
    assert detail["fields"][0]["agreement"] < 1.0
    assert detail["fields"][0]["values"] == {"stooq": 14.0, "yahoo": 28.4}
    assert detail["ticker"] == "GME"


def test_compare_sources_accepts_a_plain_row_dict():
    scan = consolidate({"yahoo": {"GME": {"current_price": 28.4}}}, ["GME"])
    row = scan.to_dicts()[0]
    detail = compare_sources(row)
    assert detail["ticker"] == "GME"
    assert any(f["field"] == "current_price" for f in detail["fields"])


def test_compare_sources_summarises_series_fields():
    series = [{"date": "2026-08-01", "ratio": 55.0}]
    scan = consolidate({"finra": {"GME": {"off_exchange_series": series}}}, ["GME"])
    detail = compare_sources(scan.by_ticker()["GME"])
    field = next(f for f in detail["fields"] if f["field"] == "off_exchange_series")
    assert field["values"]["finra"] == "1 points"


# --------------------------------------------------------------------------
# Preset sensitivity
# --------------------------------------------------------------------------


def test_compare_presets_exposes_ranking_that_depends_on_weights(service):
    scan, _ = service.consolidated()
    payload = compare_presets(scan, top=5)
    assert set(payload["presets"]) >= {"classic", "balanced", "flow"}
    for row in payload["rows"]:
        assert row["spread"] == pytest.approx(row["best"] - row["worst"], abs=0.01)
        assert set(row["ranks"]) == set(payload["presets"])
    means = [r["mean"] for r in payload["rows"]]
    assert means == sorted(means, reverse=True)


# --------------------------------------------------------------------------
# Historical reconstruction
# --------------------------------------------------------------------------


def test_reconstructed_scores_vary_with_price(sample_bars):
    values = {"ticker": "TEST", "short_pct_float": 25.0, "short_ratio": 5.0, "shares_short": 20e6, "float": 80e6}
    frame = reconstruct_scores(values, sample_bars, SqueezeEngine(), warmup=20)
    assert not frame.empty
    # The 1.x backtester fed a constant here; this must actually move.
    assert frame["squeeze_score"].nunique() > 1
    assert frame["si_basis"].iloc[0] == "short interest held at latest value"


def test_reconstruction_notes_when_no_short_interest_is_available(sample_bars):
    frame = reconstruct_scores({"ticker": "TEST"}, sample_bars, SqueezeEngine(), warmup=20)
    assert frame["si_basis"].iloc[0] == "no short interest available"


def test_reconstruction_uses_the_finra_series_for_the_flow_component(sample_bars):
    series = [
        {"date": d.date().isoformat(), "ratio": 65.0, "short": 650, "total": 1000}
        for d in sample_bars["Date"]
    ]
    values = {"ticker": "TEST", "short_pct_float": 25.0, "float": 80e6, "shares_short": 20e6}
    frame = reconstruct_scores(values, sample_bars, SqueezeEngine(preset="flow"), series, warmup=20)
    assert frame["flow_score"].notna().all()
    assert frame["flow_score"].iloc[-1] > 60


def test_reconstruction_returns_empty_for_empty_bars():
    frame = reconstruct_scores({"ticker": "X"}, pd.DataFrame(), SqueezeEngine())
    assert frame.empty


def test_build_signals_covers_every_ticker_with_bars(service):
    scan, _ = service.consolidated()
    records = {r.ticker: r.values for r in scan.records}
    tickers = list(records)[:3]
    prices = service.price_history(tickers, period="1y")
    signals = build_signals(records, prices)
    assert set(signals) == set(prices)
    for frame in signals.values():
        assert {"timestamp", "squeeze_score", "si_basis"} <= set(frame.columns)
        assert len(frame) > 200
