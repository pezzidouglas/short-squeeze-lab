"""Merging several sources into one record, with the disagreement kept visible."""

from __future__ import annotations

from datetime import date

import pytest

from squeezelab.consolidate import (
    consolidate,
    coverage_matrix,
    disagreement_report,
    short_interest_age_days,
)


def build(**by_source):
    """{'yahoo': {'GME': {...}}} shorthand."""
    return {source: {ticker: dict(values) for ticker, values in records.items()}
            for source, records in by_source.items()}


def test_priority_decides_the_winner():
    # FINRA outranks Yahoo for shares_short; Yahoo outranks Stooq for price.
    scan = consolidate(build(
        yahoo={"GME": {"shares_short": 50e6, "current_price": 28.40}},
        finra={"GME": {"shares_short": 55e6}},
        stooq={"GME": {"current_price": 28.10}},
    ), ["GME"])
    record = scan.by_ticker()["GME"]
    assert record.values["shares_short"] == 55e6
    assert record.provenance["shares_short"] == "finra"
    assert record.values["current_price"] == 28.40
    assert record.provenance["current_price"] == "yahoo"


def test_losing_values_are_kept_as_candidates():
    scan = consolidate(build(
        yahoo={"GME": {"shares_short": 50e6}},
        finra={"GME": {"shares_short": 55e6}},
    ), ["GME"])
    candidates = scan.by_ticker()["GME"].candidates["shares_short"]
    assert candidates == {"yahoo": 50e6, "finra": 55e6}


def test_close_values_agree_and_raise_no_conflict():
    # Price tolerance is 3%; a 0.4% gap is noise.
    scan = consolidate(build(
        yahoo={"GME": {"current_price": 28.40}},
        stooq={"GME": {"current_price": 28.30}},
    ), ["GME"])
    record = scan.by_ticker()["GME"]
    assert record.agreement["current_price"] == 1.0
    assert record.conflicts == []


def test_wide_gap_becomes_a_conflict_with_both_sides_named():
    scan = consolidate(build(
        yahoo={"GME": {"current_price": 28.40}},
        stooq={"GME": {"current_price": 14.00}},
    ), ["GME"])
    record = scan.by_ticker()["GME"]
    assert record.agreement["current_price"] < 1.0
    conflict = record.conflicts[0]
    assert conflict["field"] == "current_price"
    assert conflict["low"] == {"source": "stooq", "value": 14.0}
    assert conflict["high"] == {"source": "yahoo", "value": 28.4}
    assert conflict["spread"] == pytest.approx(0.5069, abs=1e-3)


def test_text_fields_agree_case_insensitively():
    scan = consolidate(build(
        yahoo={"GME": {"sector": "Consumer Cyclical"}},
        finviz={"GME": {"sector": "consumer cyclical  "}},
    ), ["GME"])
    assert scan.by_ticker()["GME"].agreement["sector"] == 1.0


def test_series_fields_are_never_averaged_or_compared():
    series = [{"date": "2026-08-01", "ratio": 55.0, "short": 1, "total": 2}]
    scan = consolidate(build(finra={"GME": {"off_exchange_series": series}}), ["GME"])
    record = scan.by_ticker()["GME"]
    assert record.values["off_exchange_series"] == series
    assert "off_exchange_series" not in record.agreement


def test_single_source_is_not_treated_as_agreement():
    scan = consolidate(build(yahoo={"GME": {"current_price": 28.4}}), ["GME"])
    record = scan.by_ticker()["GME"]
    assert "current_price" not in record.agreement
    # Neutral 0.6, not a free 1.0 — an uncorroborated number is not a verified one.
    assert record.confidence_parts["agreement"] == 0.6


def test_confidence_rises_with_coverage_and_corroboration():
    thin = consolidate(build(yahoo={"GME": {"current_price": 28.4}}), ["GME"])
    rich_values = {
        "current_price": 28.4, "float": 305e6, "short_pct_float": 18.5,
        "short_ratio": 3.2, "shares_short": 56e6, "avg_volume_10d": 5.8e6,
        "fifty_two_week_high": 64.0, "fifty_two_week_low": 9.9,
        "off_exchange_short_pct": 49.6,
        "short_interest_date": date.today().isoformat(),
    }
    rich = consolidate(build(
        yahoo={"GME": dict(rich_values)},
        finviz={"GME": dict(rich_values)},
    ), ["GME"])
    assert rich.by_ticker()["GME"].confidence > thin.by_ticker()["GME"].confidence + 30


def test_stale_short_interest_costs_freshness():
    fresh = consolidate(build(yahoo={"GME": {"short_interest_date": date.today().isoformat()}}), ["GME"])
    stale = consolidate(build(yahoo={"GME": {"short_interest_date": "2020-01-01"}}), ["GME"])
    assert fresh.by_ticker()["GME"].confidence_parts["freshness"] == 1.0
    assert stale.by_ticker()["GME"].confidence_parts["freshness"] == 0.0


def test_unknown_vintage_scores_between_fresh_and_stale():
    unknown = consolidate(build(yahoo={"GME": {"current_price": 1.0}}), ["GME"])
    assert unknown.by_ticker()["GME"].confidence_parts["freshness"] == 0.5


def test_short_interest_age_days():
    assert short_interest_age_days("2026-08-01", today=date(2026, 8, 9)) == 8
    assert short_interest_age_days(None) is None
    assert short_interest_age_days("not-a-date") is None


def test_disagreement_report_is_worst_first():
    scan = consolidate(build(
        yahoo={"GME": {"current_price": 28.4}, "AMC": {"current_price": 4.85}},
        stooq={"GME": {"current_price": 27.0}, "AMC": {"current_price": 1.00}},
    ), ["GME", "AMC"])
    rows = disagreement_report(scan)
    assert [r["ticker"] for r in rows] == ["AMC", "GME"]


def test_coverage_matrix_counts_wins_and_offers():
    scan = consolidate(build(
        yahoo={"GME": {"current_price": 28.4}},
        stooq={"GME": {"current_price": 28.3}},
    ), ["GME"])
    matrix = coverage_matrix(scan)
    assert matrix["wins"]["current_price"] == {"yahoo": 1}
    assert matrix["offers"]["current_price"] == {"yahoo": 1, "stooq": 1}


def test_tickers_with_no_data_still_appear_but_carry_no_sources():
    scan = consolidate(build(yahoo={"GME": {"current_price": 1.0}}), ["GME", "MISSING"])
    assert scan.by_ticker()["MISSING"].sources == []
