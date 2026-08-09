"""SEC fails-to-deliver parsing, walk-back and health — all offline, all fixture-driven.

Nothing here touches the network: the transport tests inject a fake session so
the walk-back, the caching split and the ZIP handling are exercised against
bytes built in-process.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from squeezelab.config import Settings
from squeezelab.sources.sec_ftd import (
    INDEX_URLS,
    PERIOD_SEARCH_LIMIT,
    URL_TEMPLATES,
    SecFtdSource,
    candidate_periods,
    decode_ftd_bytes,
    extract_period_links,
    parse_ftd_file,
    period_end,
    period_stamp,
    previous_period,
    select_txt_member,
    summarise_ftd,
    table_from_cache,
    table_to_cache,
)
from squeezelab.sources.base import SourceError

# Every hazard from the file format lives in this fixture: header row, blank
# quantity, lowercase symbol, comma grouping, two CUSIPs on one symbol/date,
# a zero balance, a padded symbol, commas inside DESCRIPTION, a 0.00 price, a
# blank symbol, a truncated final line and a trailing blank line.
FTD_FIXTURE = """SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
20260616|36467W109|GME|1250000|GAMESTOP CORP NEW CL A|28.40
20260616|00165C104|AMC|48000|AMC ENTERTAINMENT HLDGS INC CL A|4.85
20260616|G0451H104|BAD||APEX HOLDINGS LTD ORD|
20260616|88160R101|tsla|12000|TESLA INC COM|241.10
20260617|36467W109|GME|1,410,000|GAMESTOP CORP NEW CL A|28.90
20260617|36467W208|GME|90,000|GAMESTOP CORP NEW CL A WHEN ISSUED|28.90
20260617|00165C104|AMC|0|AMC ENTERTAINMENT HLDGS INC CL A|4.90
20260617|46145F105|  SPRY |305000|ARS PHARMACEUTICALS, INC. COM|11.02
20260618|36467W109|GME|990000|GAMESTOP CORP NEW CL A|27.75
20260618|69422A100|PCT|220000|PURECYCLE TECHNOLOGIES INC CL A|0.00
20260618|123456789||75000|SOME TRUST UNIT SER 2026|3.10
20260618|46145F105|SPRY|412000|ARS PHARMACEUTICALS, INC. COM|11.44
20260619|36467W109|GME|1050000|GAMESTOP CORP NEW CL A|27.10
20260619|00165C104|AMC|61000|AMC ENTERTAINMENT HLDGS INC CL A|4.72
20260619|36467W109

"""

# The second half-month period, so the two-period merge has something to merge.
FTD_FIXTURE_PRIOR = """SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
20260602|36467W109|GME|400000|GAMESTOP CORP NEW CL A|26.10
20260603|36467W109|GME|410000|GAMESTOP CORP NEW CL A|26.30
"""

SCAN_DAY = date(2026, 8, 9)  # the worked example from the design spec


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------


def test_parse_ftd_file_keeps_only_usable_rows():
    parsed = parse_ftd_file(FTD_FIXTURE)
    # BAD has no quantity, the trust row has no symbol; AMC survives on its
    # other two dates despite the zero-balance row.
    assert set(parsed) == {"GME", "AMC", "TSLA", "SPRY", "PCT"}


def test_parse_ftd_file_sums_duplicate_cusips_on_one_date():
    """The when-issued line is the same ticker failing on the same day."""
    parsed = parse_ftd_file(FTD_FIXTURE)
    by_date = {stamp: shares for stamp, shares, _ in parsed["GME"]}
    assert by_date["20260617"] == 1_500_000


def test_parse_ftd_file_series_shape_and_order():
    parsed = parse_ftd_file(FTD_FIXTURE)
    assert [stamp for stamp, _, _ in parsed["GME"]] == [
        "20260616", "20260617", "20260618", "20260619",
    ]
    assert len(parsed["AMC"]) == 2  # the zero-balance date carries no information


def test_parse_ftd_file_normalises_symbols():
    parsed = parse_ftd_file(FTD_FIXTURE)
    assert parsed["TSLA"][0][1] == 12_000  # lowercase in the file
    assert parsed["SPRY"][0][1] == 305_000  # whitespace-padded in the file


def test_parse_ftd_file_treats_a_zero_price_as_missing():
    parsed = parse_ftd_file(FTD_FIXTURE)
    assert parsed["PCT"] == [("20260618", 220_000.0, None)]


def test_parse_ftd_file_survives_junk():
    assert parse_ftd_file("") == {}
    assert parse_ftd_file("garbage\nmore|garbage") == {}
    assert parse_ftd_file("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n") == {}


def test_parse_ftd_file_survives_a_stray_delimiter_in_the_description():
    """Numbers are anchored left and price right, so only the text is lost."""
    text = "20260616|36467W109|GME|1000|GAMESTOP CORP | CL A|28.40\n"
    assert parse_ftd_file(text)["GME"] == [("20260616", 1000.0, 28.40)]


def test_parse_ftd_file_handles_rows_without_a_price_column():
    text = "20260616|36467W109|GME|1000\n"
    assert parse_ftd_file(text)["GME"] == [("20260616", 1000.0, None)]


def test_parse_ftd_file_decodes_legacy_high_bytes():
    raw = "20260616|36467W109|GME|1000|CAF\xc9 HOLDINGS S.A.|1.10\n".encode("latin-1")
    assert parse_ftd_file(decode_ftd_bytes(raw))["GME"][0][1] == 1000.0


# --------------------------------------------------------------------------
# Container handling
# --------------------------------------------------------------------------


def _zip_bytes(text: str, member: str = "cnsfails202606b.txt") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text.encode("latin-1"))
    return buf.getvalue()


def test_select_txt_member_ignores_macos_resource_forks():
    assert select_txt_member(["__MACOSX/._x.txt", "cnsfails202606b.txt"]) == "cnsfails202606b.txt"
    assert select_txt_member(["data/", "data/cnsfails.TXT"]) == "data/cnsfails.TXT"
    assert select_txt_member(["readme.pdf"]) is None


def test_decode_ftd_bytes_reads_a_zip_whose_member_is_named_differently():
    body = _zip_bytes(FTD_FIXTURE, member="something-else.txt")
    assert "GAMESTOP" in decode_ftd_bytes(body)


def test_decode_ftd_bytes_sniffs_a_bare_text_payload():
    """Old vintages were served unzipped; the sniff costs no extra request."""
    assert decode_ftd_bytes(FTD_FIXTURE.encode("latin-1")).startswith("SETTLEMENT DATE|")


def test_decode_ftd_bytes_rejects_a_corrupt_archive():
    with pytest.raises(SourceError) as excinfo:
        decode_ftd_bytes(b"PK\x03\x04 not really a zip")
    assert "archive" in str(excinfo.value)


def test_decode_ftd_bytes_reports_the_namelist_when_no_txt_member_exists():
    body = _zip_bytes("nothing", member="cnsfails202606b.csv")
    with pytest.raises(SourceError) as excinfo:
        decode_ftd_bytes(body)
    assert "cnsfails202606b.csv" in str(excinfo.value)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_summarise_ftd_summarises_and_never_sums():
    out = summarise_ftd([parse_ftd_file(FTD_FIXTURE)], ["GME", "AMC"])
    gme = out["GME"]
    assert gme["ftd_shares_peak"] == 1_500_000
    assert gme["ftd_shares_avg"] == 1_197_500  # not the 4.79M total
    assert gme["ftd_persist_days"] == 4
    assert gme["ftd_notional"] == pytest.approx(1_500_000 * 28.90)
    assert gme["ftd_date"] == "2026-06-19"
    assert out["AMC"]["ftd_persist_days"] == 2


def test_summarise_ftd_series_is_ascending_and_carries_price():
    out = summarise_ftd([parse_ftd_file(FTD_FIXTURE)], ["GME", "PCT"])
    series = out["GME"]["ftd_series"]
    assert [p["date"] for p in series] == [
        "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19",
    ]
    assert series[1] == {"date": "2026-06-17", "shares": 1_500_000.0, "price": 28.90}
    assert out["PCT"]["ftd_series"][0]["price"] is None


def test_summarise_ftd_merges_periods_newest_last():
    periods = [parse_ftd_file(FTD_FIXTURE_PRIOR), parse_ftd_file(FTD_FIXTURE)]
    gme = summarise_ftd(periods, ["GME"])["GME"]
    assert gme["ftd_persist_days"] == 6
    assert gme["ftd_series"][0]["date"] == "2026-06-02"
    assert gme["ftd_date"] == "2026-06-19"


def test_summarise_ftd_skips_tickers_absent_from_the_file():
    assert summarise_ftd([parse_ftd_file(FTD_FIXTURE)], ["ZZZZ"]) == {}


def test_ftd_pct_float_is_claimed_only_with_an_injected_denominator():
    parsed = [parse_ftd_file(FTD_FIXTURE)]
    assert "ftd_pct_float" not in summarise_ftd(parsed, ["GME"])["GME"]
    out = summarise_ftd(parsed, ["GME"], float_lookup={"GME": 300_000_000}.get)
    assert out["GME"]["ftd_pct_float"] == pytest.approx(0.5)


@pytest.mark.parametrize("lookup", [
    lambda t: None,
    lambda t: 0,
    lambda t: -1,
    lambda t: (_ for _ in ()).throw(RuntimeError("denominator blew up")),
])
def test_a_broken_float_lookup_drops_one_field_not_the_record(lookup):
    out = summarise_ftd([parse_ftd_file(FTD_FIXTURE)], ["GME"], float_lookup=lookup)
    assert "ftd_pct_float" not in out["GME"]
    assert out["GME"]["ftd_shares_peak"] == 1_500_000


def test_ftd_trend_needs_more_than_the_fast_window():
    parsed = parse_ftd_file(FTD_FIXTURE)
    assert "ftd_trend" not in summarise_ftd([parsed], ["GME"])["GME"]

    # Twelve settlement dates: seven flat, then five at double the balance.
    points = [(f"202606{day:02d}", 100_000.0, 2.0) for day in range(1, 8)]
    points += [(f"202606{day:02d}", 200_000.0, 2.0) for day in range(8, 13)]
    out = summarise_ftd([{"XYZ": points}], ["XYZ"])
    window = (7 * 100_000 + 5 * 200_000) / 12
    assert out["XYZ"]["ftd_trend"] == pytest.approx((200_000 / window - 1) * 100, abs=0.01)


def test_unpunctuated_symbols_are_matched_best_effort():
    """FTD writes BRKB where Yahoo and Finviz write BRK-B."""
    text = "20260616|084670702|BRKB|5000|BERKSHIRE HATHAWAY INC CL B|412.00\n"
    out = summarise_ftd([parse_ftd_file(text)], ["BRK-B"])
    assert out["BRK-B"]["ftd_shares_peak"] == 5000


def test_cache_encoding_round_trips():
    parsed = parse_ftd_file(FTD_FIXTURE)
    assert table_from_cache(table_to_cache(parsed)) == parsed
    assert table_from_cache("not a table") == {}


# --------------------------------------------------------------------------
# Period arithmetic and the walk-back
# --------------------------------------------------------------------------


def test_period_boundaries():
    assert period_stamp((2026, 6, "b")) == "202606b"
    assert period_end((2026, 6, "a")) == date(2026, 6, 15)
    assert period_end((2026, 2, "b")) == date(2026, 2, 28)
    assert period_end((2024, 2, "b")) == date(2024, 2, 29)  # leap year
    assert previous_period((2026, 1, "a")) == (2025, 12, "b")
    assert previous_period((2026, 6, "b")) == (2026, 6, "a")


def test_candidate_periods_skips_what_sec_cannot_have_published_yet():
    """The worked example: on 2026-08-09 nothing after 2026-07a is worth a GET."""
    candidates = candidate_periods(SCAN_DAY, PERIOD_SEARCH_LIMIT)
    assert candidates[:3] == [(2026, 7, "a"), (2026, 6, "b"), (2026, 6, "a")]
    assert len(candidates) == PERIOD_SEARCH_LIMIT
    assert (2026, 7, "b") not in candidates  # closed 9 days ago, inside the lag floor


def test_extract_period_links_resolves_relative_hrefs():
    html = """
    <a href="/files/data/fails-deliver-data/cnsfails202606b.zip">June 2026 2nd half</a>
    <a href='https://www.sec.gov/files/data/fails-deliver-data/cnsfails202606a.ZIP'>June 1st half</a>
    <a href="/files/data/old/cnsfails200409a.txt">Sept 2004</a>
    <a href="/about.html">not a data file</a>
    """
    links = extract_period_links(html)
    assert set(links) == {(2026, 6, "b"), (2026, 6, "a"), (2004, 9, "a")}
    assert links[(2026, 6, "b")].startswith("https://www.sec.gov/files/data/")
    assert extract_period_links("") == {}


# --------------------------------------------------------------------------
# Transport, caching and health (fake session — still no network)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: bytes) -> None:
        self.status_code = status_code
        self.content = payload

    @property
    def text(self) -> str:
        return self.content.decode("latin-1")


class _FakeSession:
    """Serves a fixed URL -> payload map; everything else 404s like SEC does."""

    def __init__(self, routes: dict[str, bytes]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
        self.requested.append(url)
        payload = self.routes.get(url)
        if payload is None:
            return _FakeResponse(404, b"")
        return _FakeResponse(200, payload)


def _url(stamp: str, template: int = 0) -> str:
    return URL_TEMPLATES[template].format(yyyymm=stamp[:6], half=stamp[6:])


@pytest.fixture
def ftd_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        cache_ttl_seconds=1800,
        offline=False,
        sec_contact="me@example.com",
        http_retries=0,
        enabled_sources=("sec_ftd",),
    )


def _source(ftd_settings, routes, **kwargs):
    session = _FakeSession(routes)
    source = SecFtdSource(ftd_settings, session, today=SCAN_DAY, **kwargs)
    return source, session


def test_walk_back_stops_after_two_published_periods(ftd_settings):
    routes = {
        _url("202606b"): _zip_bytes(FTD_FIXTURE),
        _url("202606a"): _zip_bytes(FTD_FIXTURE_PRIOR, member="cnsfails202606a.txt"),
    }
    source, session = _source(ftd_settings, routes)

    records = source.collect(["GME", "AMC"])
    assert source.health.status == "ok"
    assert records["GME"]["ftd_persist_days"] == 6
    assert records["GME"]["ftd_date"] == "2026-06-19"
    # 2026-07a is the only unpublished candidate reached; both templates are
    # tried for it, then the two real periods answer on the first template.
    assert session.requested == [
        _url("202607a"), _url("202607a", 1), _url("202606b"), _url("202606a"),
    ]


def test_a_missing_period_is_not_pinned_in_the_long_cache(ftd_settings):
    """An FTD period that 404s today will exist in three weeks."""
    routes = {_url("202606b"): _zip_bytes(FTD_FIXTURE)}
    source, _ = _source(ftd_settings, routes)
    source.collect(["GME"])

    assert source.cache.get("sec_ftd_miss", "202607a") is True
    assert source.file_cache.get("sec_ftd", "202607a") is None
    assert source.file_cache.get("sec_ftd", "202606b")


def test_a_second_scan_is_served_from_cache(ftd_settings):
    routes = {
        _url("202606b"): _zip_bytes(FTD_FIXTURE),
        _url("202606a"): _zip_bytes(FTD_FIXTURE_PRIOR, member="cnsfails202606a.txt"),
    }
    source, session = _source(ftd_settings, routes)
    first = source.collect(["GME"])
    calls = len(session.requested)

    second = source.collect(["GME"])
    assert second == first
    assert len(session.requested) == calls  # nothing re-requested
    assert source.health.from_cache is True


def test_a_short_window_is_reported_but_still_returns_data(ftd_settings):
    routes = {_url("202606b"): _zip_bytes(FTD_FIXTURE)}
    source, _ = _source(ftd_settings, routes)
    records = source.collect(["GME"])
    assert records["GME"]["ftd_persist_days"] == 4
    assert "1 of 2 periods" in source.health.detail


def test_the_legacy_path_is_tried_on_a_404(ftd_settings):
    routes = {_url("202606b", 1): _zip_bytes(FTD_FIXTURE)}
    source, session = _source(ftd_settings, routes)
    assert source.collect(["GME"])["GME"]["ftd_shares_peak"] == 1_500_000
    assert _url("202606b") in session.requested  # the primary template first


def test_a_transport_failure_does_not_masquerade_as_a_missing_file(ftd_settings):
    class _DeadSession:
        def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
            raise OSError("connection reset")

    source = SecFtdSource(ftd_settings, _DeadSession(), today=SCAN_DAY)
    assert source.collect(["GME"]) == {}
    assert source.health.status == "error"
    assert "connection reset" in source.health.detail


def test_every_period_missing_reports_an_error_not_a_silent_empty(ftd_settings):
    source, _ = _source(ftd_settings, {})
    assert source.collect(["GME"]) == {}
    assert source.health.status == "error"
    assert "no fails-to-deliver file published" in source.health.detail


def test_a_corrupt_archive_is_reported_not_raised(ftd_settings):
    routes = {_url("202606b"): b"PK\x03\x04 truncated download"}
    source, _ = _source(ftd_settings, routes)
    assert source.collect(["GME"]) == {}
    assert source.health.status == "error"
    assert "archive" in source.health.detail


def test_a_ticker_absent_from_the_file_is_empty_not_an_error(ftd_settings):
    routes = {_url("202606b"): _zip_bytes(FTD_FIXTURE)}
    source, _ = _source(ftd_settings, routes)
    assert source.collect(["ZZZZ"]) == {}
    assert source.health.status == "empty"


def test_the_index_page_recovers_a_renamed_file_scheme(ftd_settings):
    """Last resort when SEC moves the directory again."""
    moved = "https://www.sec.gov/files/data/somewhere-new/cnsfails202606b.zip"
    routes = {
        INDEX_URLS[0]: f'<a href="{moved}">June 2026 2nd half</a>'.encode("latin-1"),
        moved: _zip_bytes(FTD_FIXTURE),
    }
    source, session = _source(ftd_settings, routes)
    records = source.collect(["GME"])
    assert records["GME"]["ftd_shares_peak"] == 1_500_000
    assert session.requested[-1] == moved


def test_float_lookup_is_wired_through_the_source(ftd_settings):
    routes = {_url("202606b"): _zip_bytes(FTD_FIXTURE)}
    source, _ = _source(ftd_settings, routes, float_lookup={"GME": 300_000_000}.get)
    assert source.collect(["GME"])["GME"]["ftd_pct_float"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Availability — same contract as the other SEC source
# --------------------------------------------------------------------------


def test_ftd_is_skipped_rather_than_erroring_without_a_contact(settings, monkeypatch):
    monkeypatch.setattr(settings, "sec_contact", "", raising=False)
    monkeypatch.setattr(settings, "offline", False, raising=False)
    source = SecFtdSource(settings)
    assert source.available() is False
    assert source.collect(["GME"]) == {}
    assert source.health.status == "skipped"


def test_ftd_declares_a_contact_in_its_user_agent(settings, monkeypatch):
    monkeypatch.setattr(settings, "sec_contact", "me@example.com", raising=False)
    monkeypatch.setattr(settings, "offline", False, raising=False)
    source = SecFtdSource(settings)
    assert source.available() is True
    assert "me@example.com" in source.user_agent()


def test_offline_mode_skips_the_source(settings, monkeypatch):
    monkeypatch.setattr(settings, "sec_contact", "me@example.com", raising=False)
    monkeypatch.setattr(settings, "offline", True, raising=False)
    assert SecFtdSource(settings).available() is False


def test_meta_matches_what_the_source_can_actually_emit(ftd_settings):
    routes = {_url("202606b"): _zip_bytes(FTD_FIXTURE)}
    source, _ = _source(ftd_settings, routes, float_lookup={"GME": 300_000_000}.get)
    emitted = set(source.collect(["GME"])["GME"])
    assert source.meta.key == "sec_ftd"
    assert source.meta.cadence == "semi-monthly"
    assert emitted <= set(source.meta.provides)
