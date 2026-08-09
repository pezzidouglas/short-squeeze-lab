"""Source parsing and the health contract — all offline, all fixture-driven."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from squeezelab.sources.base import DataSource, DiskCache, SourceError, SourceMeta, to_float
from squeezelab.sources.finra import parse_daily_file, parse_short_interest, summarise_short_volume
from squeezelab.sources.finviz import parse_export
from squeezelab.sources.sec_edgar import latest_shares_outstanding, parse_cik_map
from squeezelab.sources.stooq import summarise_bars


# --------------------------------------------------------------------------
# Value parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None), ("", None), ("-", None), ("N/A", None),
        (12, 12.0), ("12.5", 12.5), ("1,234", 1234.0), ("$4.20", 4.2),
        ("12.34%", 12.34), ("1.5K", 1500.0), ("2.5M", 2_500_000.0),
        ("3B", 3e9), ("1T", 1e12), (True, None), ("abc", None),
    ],
)
def test_to_float_handles_provider_shapes(raw, expected):
    assert to_float(raw) == expected


def test_to_float_drops_nan():
    assert to_float(float("nan")) is None


# --------------------------------------------------------------------------
# FINRA
# --------------------------------------------------------------------------

DAILY_FIXTURE = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20260807|GME|500000|1000|1000000|B,Q,N
20260807|AMC|250000|0|1000000|B,Q,N
20260807|BAD|0|0|0|B
Record Count: 3
"""


def test_parse_daily_file():
    parsed = parse_daily_file(DAILY_FIXTURE)
    assert set(parsed) == {"GME", "AMC"}  # zero-volume row dropped
    assert parsed["GME"] == {"short": 500000.0, "exempt": 1000.0, "total": 1000000.0}


def test_parse_daily_file_ignores_junk():
    assert parse_daily_file("") == {}
    assert parse_daily_file("garbage\nmore|garbage") == {}


def test_summarise_short_volume_ratio_and_trend():
    sessions = []
    # Ten sessions at 40%, then ten at 60% — the fast window must read hotter.
    for i in range(20):
        ratio = 0.40 if i < 10 else 0.60
        day = date(2026, 7, 1 + i)
        sessions.append((day, {"GME": {"short": 1000 * ratio, "exempt": 0, "total": 1000}}))
    out = summarise_short_volume(sessions, ["GME"])
    assert out["GME"]["off_exchange_short_pct"] == pytest.approx(50.0, abs=0.01)
    assert out["GME"]["off_exchange_trend"] == pytest.approx(10.0, abs=0.01)
    assert len(out["GME"]["off_exchange_series"]) == 20


def test_summarise_short_volume_skips_uncovered_tickers():
    sessions = [(date(2026, 7, 1), {"GME": {"short": 1, "exempt": 0, "total": 2}})]
    assert "AMC" not in summarise_short_volume(sessions, ["GME", "AMC"])


def test_parse_short_interest_keeps_latest_settlement():
    payload = [
        {"symbolCode": "GME", "settlementDate": "2026-06-15", "currentShortPositionQuantity": 100},
        {"symbolCode": "GME", "settlementDate": "2026-07-15", "currentShortPositionQuantity": 200,
         "previousShortPositionQuantity": 100, "daysToCoverQuantity": 3.5},
    ]
    out = parse_short_interest(payload)
    assert out["GME"]["shares_short"] == 200
    assert out["GME"]["short_interest_date"] == "2026-07-15"
    assert out["GME"]["short_ratio"] == 3.5


# --------------------------------------------------------------------------
# Finviz
# --------------------------------------------------------------------------

FINVIZ_FIXTURE = (
    "No.,Ticker,Company,Sector,Industry,Market Cap,Outstanding,Float,"
    "Float Short,Short Ratio,Avg Volume,Price,Change,Volume\n"
    "1,GME,GameStop Corp,Consumer Cyclical,Specialty Retail,12.8B,440M,305M,18.50%,3.20,5.8M,28.40,1.2%,6M\n"
    "2,AMC,AMC Entertainment,Communication Services,Entertainment,1.6B,290M,263M,16.20%,2.80,12M,4.85,-0.5%,14M\n"
)


def test_parse_export_maps_and_derives_shares_short():
    out = parse_export(FINVIZ_FIXTURE)
    assert out["GME"]["short_pct_float"] == 18.50
    assert out["GME"]["short_ratio"] == 3.20
    assert out["GME"]["float"] == 305e6
    # shares_short is derived from float × short %, not taken on faith.
    assert out["GME"]["shares_short"] == pytest.approx(305e6 * 0.185)
    assert out["AMC"]["sector"] == "Communication Services"


def test_parse_export_handles_combined_short_column():
    text = "No.,Ticker,Float,Short Float / Ratio\n1,XYZ,10M,25.00% / 4.5\n"
    out = parse_export(text)
    assert out["XYZ"]["short_pct_float"] == 25.0
    assert out["XYZ"]["short_ratio"] == 4.5


def test_parse_export_rejects_wrong_payload():
    with pytest.raises(SourceError):
        parse_export("<html>login required</html>")


# --------------------------------------------------------------------------
# SEC
# --------------------------------------------------------------------------


def test_parse_cik_map_pads_cik():
    payload = {"0": {"cik_str": 1326380, "ticker": "gme", "title": "GameStop Corp"}}
    out = parse_cik_map(payload)
    assert out["GME"]["cik"] == "0001326380"
    assert out["GME"]["name"] == "GameStop Corp"


def test_latest_shares_outstanding_picks_newest():
    payload = {"units": {"shares": [
        {"end": "2026-03-31", "val": 400_000_000},
        {"end": "2026-06-30", "val": 445_000_000},
        {"end": "2025-12-31", "val": 380_000_000},
    ]}}
    assert latest_shares_outstanding(payload) == 445_000_000


def test_latest_shares_outstanding_empty():
    assert latest_shares_outstanding({}) is None


# --------------------------------------------------------------------------
# Stooq
# --------------------------------------------------------------------------


def test_summarise_bars(sample_bars):
    out = summarise_bars(sample_bars)
    assert out["current_price"] == pytest.approx(13.9)
    assert out["previous_close"] == pytest.approx(13.8)
    assert out["fifty_two_week_high"] == pytest.approx(14.1)
    assert out["fifty_two_week_low"] == pytest.approx(9.8)
    assert out["avg_volume_10d"] == 1_000_000


# --------------------------------------------------------------------------
# Health contract
# --------------------------------------------------------------------------


class _Boom(DataSource):
    meta = SourceMeta(key="boom", name="Boom", url="", cadence="daily", provides=(), requires_network=False)

    def _fetch(self, tickers):
        raise RuntimeError("upstream exploded")


class _Partial(DataSource):
    meta = SourceMeta(key="part", name="Part", url="", cadence="daily", provides=(), requires_network=False)

    def _fetch(self, tickers):
        return {tickers[0]: {"current_price": 10.0, "beta": None, "name": "  "}}


def test_a_failing_source_reports_health_instead_of_raising(settings):
    source = _Boom(settings)
    assert source.collect(["GME", "AMC"]) == {}
    assert source.health.status == "error"
    assert "upstream exploded" in source.health.detail


def test_partial_source_is_marked_partial_and_cleaned(settings):
    source = _Partial(settings)
    records = source.collect(["GME", "AMC"])
    assert source.health.status == "partial"
    assert source.health.coverage == 0.5
    # Empty and None values never make it into a record.
    assert records["GME"] == {"current_price": 10.0}


def test_disk_cache_roundtrip_and_ttl(tmp_path):
    cache = DiskCache(str(tmp_path), ttl_seconds=60)
    cache.set("ns", "key", {"a": 1})
    assert cache.get("ns", "key") == {"a": 1}

    expired = DiskCache(str(tmp_path), ttl_seconds=0)
    assert expired.get("ns", "key") is None


def test_snapshot_source_fetches_and_prices(snapshot_source):
    tickers = snapshot_source.tickers()
    assert tickers
    records = snapshot_source.collect(tickers[:3])
    assert set(records) == set(tickers[:3])
    assert records[tickers[0]]["short_pct_float"] > 0

    bars = snapshot_source.fetch_prices(tickers[0], period="1y")
    assert isinstance(bars, pd.DataFrame)
    assert len(bars) == 252
    assert {"Date", "Open", "High", "Low", "Close", "Volume"} <= set(bars.columns)


def test_snapshot_is_labelled_synthetic(snapshot_source):
    assert snapshot_source.payload()["synthetic"] is True
