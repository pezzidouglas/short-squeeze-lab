"""
Borrow source (IBKR shortable shares) — every test offline and fixture-driven.

The file format this source parses could not be verified against the live host,
so most of what is asserted here is *how the parser behaves when it is wrong*:
that it refuses an unrecognisable layout instead of guessing column positions,
that it reports what actually arrived, and that the numeric traps in the repo's
own helpers cannot silently corrupt a rate or delete a zero.
"""

from __future__ import annotations

import pytest

from squeezelab.config import Settings
from squeezelab.sources.base import DiskCache, SourceError, clean_record, positive, to_float
from squeezelab.sources.borrow import (
    DEFAULT_URLS,
    BorrowSource,
    build_symbol_index,
    cache_key,
    decode_body,
    describe_payload,
    implied_benchmark_rate,
    looks_like_shortable_file,
    parse_available,
    parse_bof_line,
    parse_header,
    parse_rate,
    parse_shortable_file,
    symbol_aliases,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# Every parsing hazard in the design is represented by exactly one row:
# BOM + CRLF; AAPL censored `>N`; GME an ordinary hard-to-borrow; AMC zero
# available; BRK B a space-separated class symbol with thousands commas; RUM WS
# a warrant with nothing known; PCT European decimal commas; INDI duplicated
# across two conids; SLS all-NA; a blank line; ZZZZ a short row; #EOF.
SHORTABLE_FIXTURE = (
    "﻿#BOF|usa|2026.08.07|18:15:02\r\n"
    "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE\r\n"
    "AAPL|USD|APPLE INC|265598|US0378331005|4.08|0.25|>10000000\r\n"
    "GME|USD|GAMESTOP CORP-CLASS A|36285627|US36467W1099|-21.75|26.00|175000\r\n"
    "AMC|USD|AMC ENTERTAINMENT HLDG-CLASS A|145733733|US00165C3025|-95.50|100.00|0\r\n"
    "BRK B|USD|BERKSHIRE HATHAWAY INC-CL B|13905888|US0846707026|4.30|0.30|1,250,000\r\n"
    "RUM WS|USD|RUMBLE INC - WARRANT|573823891|US78137L1176|NA|NA|NA\r\n"
    "PCT|USD|PURECYCLE TECHNOLOGIES INC|441234567|US74624P1012|-3,25|7,50|48000\r\n"
    "INDI|USD|INDIE SEMICONDUCTOR-A|500000001|US45569U1016|3.90|0.45|900000\r\n"
    "INDI|USD|INDIE SEMICONDUCTOR-A|500000002|US45569U1016|3.90|0.45|1500000\r\n"
    "SLS|USD|SELLAS LIFE SCIENCES GRP|NA|US81639W2044|NA|NA|NA\r\n"
    "\r\n"
    "ZZZZ|USD|TRUNCATED ROW\r\n"
    "#EOF\r\n"
)

TRUNCATED_FIXTURE = SHORTABLE_FIXTURE.replace("#EOF\r\n", "")

# The same rows under an aliased header, to prove nothing depends on the exact
# spelling IBKR happens to use today.
ALIASED_HEADER_FIXTURE = (
    "#BOF|usa|2026.08.07|18:15:02\n"
    "#SYMBOL|CURRENCY|NAME|REBATE|FEE RATE|QUANTITY\n"
    "GME|USD|GAMESTOP CORP|-21.75|26.00|175000\n"
    "#EOF\n"
)

# A warrant and its underlying, both present with real values, so the alias
# index has something it could plausibly collapse.
WARRANT_FIXTURE = (
    "#BOF|usa|2026.08.07|18:15:02\n"
    "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE\n"
    "RUM|USD|RUMBLE INC-CLASS A|1|US78137L1052|3.90|0.45|500000\n"
    "RUM WS|USD|RUMBLE INC - WARRANT|2|US78137L1176|-2.10|6.50|1000\n"
    "BRK B|USD|BERKSHIRE HATHAWAY INC-CL B|3|US0846707026|4.30|0.30|1250000\n"
    "#EOF\n"
)

# The rate columns no longer carry a rebate/fee pair. Row by row the two numbers
# no longer sum to anything like a funding rate, which is what the tripwire sees.
WRONG_COLUMN_FIXTURE = (
    "#BOF|usa|2026.08.07|18:15:02\n"
    "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE\n"
    "AAPL|USD|APPLE INC|1|US0378331005|0.25|48.00|>10000000\n"
    "GME|USD|GAMESTOP CORP|2|US36467W1099|26.00|51.20|175000\n"
    "AMC|USD|AMC ENTERTAINMENT|3|US00165C3025|100.00|44.10|0\n"
    "INDI|USD|INDIE SEMICONDUCTOR|4|US45569U1016|0.45|39.80|900000\n"
    "#EOF\n"
)


@pytest.fixture
def parsed():
    records, diagnostics = parse_shortable_file(SHORTABLE_FIXTURE)
    return records, diagnostics


@pytest.fixture(autouse=True)
def _no_ambient_opt_in(monkeypatch):
    """No test may inherit the developer's own opt-in from the environment."""
    monkeypatch.delenv("SQUEEZE_BORROW_ENABLED", raising=False)
    monkeypatch.delenv("IBKR_SHORTABLE_URL", raising=False)


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    """Settings that let `base.get` reach a stub session instead of the network."""
    return Settings(
        data_dir=str(tmp_path),
        cache_ttl_seconds=0,
        offline=False,
        http_retries=0,
        http_timeout=1.0,
        enabled_sources=("borrow",),
    )


class _StubResponse:
    def __init__(self, body: str | bytes, status_code: int = 200) -> None:
        self.status_code = status_code
        self.content = body.encode("utf-8") if isinstance(body, str) else body

    @property
    def text(self) -> str:  # pragma: no cover - the source decodes `content`
        return self.content.decode("utf-8", errors="replace")


class _StubSession:
    """Records every URL asked for, so "one request per scan" is testable."""

    def __init__(self, body: str | bytes, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return _StubResponse(self.body, self.status_code)


class _ExplodingSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        raise RuntimeError("TLS handshake failed")


# --------------------------------------------------------------------------
# Why the repo's own numeric helpers cannot be used here
# --------------------------------------------------------------------------


def test_the_shared_helpers_would_corrupt_this_file():
    """These four results are the entire reason `parse_rate`/`parse_available` exist."""
    assert to_float(">10000000") is None      # availability deleted for liquid names
    assert to_float("7,50") == 750.0          # a silent 100x on a European decimal
    assert positive("0") is None              # zero shortable is the loudest reading there is
    assert to_float("1,250,000") == 1_250_000.0  # here a comma *is* a group separator
    # ...and 0.0 does survive the rest of the pipeline, so it is worth keeping.
    assert clean_record({"shortable_shares": 0.0}) == {"shortable_shares": 0.0}


# --------------------------------------------------------------------------
# parse_rate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.25", 0.25),
        ("26.00", 26.0),
        ("-21.75", -21.75),      # negative rebates are normal for hard-to-borrow
        ("-95.50", -95.5),
        ("7,50", 7.5),           # European decimal comma, NOT 750
        ("-3,25", -3.25),
        ("4.08%", 4.08),
        (" 0.30 ", 0.30),
        ("300", 300.0),          # 300%/yr fees are real
        ("NA", None),
        ("N/A", None),
        ("-", None),
        ("", None),
        (None, None),
        ("abc", None),
        ("1,234,567", None),     # two commas: this is not a rate at all
        ("40000", None),         # outside the units tripwire
        ("-40000", None),
    ],
)
def test_parse_rate(raw, expected):
    assert parse_rate(raw) == expected


def test_parse_rate_treats_a_comma_as_a_group_separator_when_a_dot_is_present():
    # A '.' already holds the decimal point, so the comma can only be grouping.
    assert parse_rate("1,23.45") == 123.45
    # ...and the value that produces is then rejected by the units tripwire.
    assert parse_rate("1,234.56") is None


# --------------------------------------------------------------------------
# parse_available
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("175000", (175000.0, "exact")),
        ("0", (0.0, "exact")),                 # never dropped
        ("1,250,000", (1250000.0, "exact")),
        (">10000000", (10000000.0, "floor")),
        (">=10000000", (10000000.0, "floor")),
        ("> 500", (500.0, "floor")),
        ("NA", (None, None)),
        ("", (None, None)),
        (None, (None, None)),
        ("-1000", (None, None)),
        ("plenty", (None, None)),
    ],
)
def test_parse_available(raw, expected):
    assert parse_available(raw) == expected


# --------------------------------------------------------------------------
# Header and preamble
# --------------------------------------------------------------------------


def test_parse_bof_line():
    assert parse_bof_line("#BOF|usa|2026.08.07|18:15:02") == ("usa", "2026-08-07T18:15:02")


def test_parse_bof_line_tolerates_a_missing_or_odd_clock():
    assert parse_bof_line("#BOF|usa|2026.08.07") == ("usa", "2026-08-07")
    assert parse_bof_line("#BOF|usa|whenever|now") == ("usa", None)


def test_parse_header_is_position_independent():
    columns = parse_header("#AVAILABLE|FEERATE|SYM|REBATERATE|CUR")
    assert columns == {"available": 0, "fee": 1, "symbol": 2, "rebate": 3, "currency": 4}


def test_parse_header_rejects_a_line_with_no_symbol_column():
    assert parse_header("#CUR|NAME|ISIN") is None


def test_header_aliases_are_accepted():
    records, _ = parse_shortable_file(ALIASED_HEADER_FIXTURE)
    assert records["GME"] == {
        "borrow_fee_pct": 26.0,
        "borrow_rebate_pct": -21.75,
        "shortable_shares": 175000.0,
        "shortable_shares_basis": "exact",
        "borrow_as_of": "2026-08-07T18:15:02",
    }


def test_a_missing_header_refuses_and_quotes_what_arrived():
    """The refusal is the design: an unverifiable layout must not be guessed at."""
    body = (
        "#BOF|usa|2026.08.07|18:15:02\n"
        "AAPL,USD,APPLE INC,265598,US0378331005,4.08,0.25,>10000000\n"
    )
    with pytest.raises(SourceError) as excinfo:
        parse_shortable_file(body)
    message = str(excinfo.value)
    assert "unrecognised IBKR layout" in message
    assert "AAPL,USD,APPLE INC" in message


def test_a_header_without_fee_or_availability_is_not_enough():
    body = "#SYM|CUR|NAME|CON|ISIN\nGME|USD|GAMESTOP|1|US36467W1099\n"
    with pytest.raises(SourceError) as excinfo:
        parse_shortable_file(body)
    assert "fee" in str(excinfo.value)


def test_an_empty_body_refuses_rather_than_returning_nothing():
    with pytest.raises(SourceError):
        parse_shortable_file("")


# --------------------------------------------------------------------------
# parse_shortable_file — the row-level hazards
# --------------------------------------------------------------------------


def test_happy_path_field_mapping(parsed):
    records, _ = parsed
    assert records["GME"] == {
        "borrow_fee_pct": 26.0,
        "borrow_rebate_pct": -21.75,
        "shortable_shares": 175000.0,
        "shortable_shares_basis": "exact",
        "borrow_as_of": "2026-08-07T18:15:02",
    }


def test_bom_and_crlf_do_not_break_the_preamble(parsed):
    _, diagnostics = parsed
    assert diagnostics["country"] == "usa"
    assert diagnostics["as_of"] == "2026-08-07T18:15:02"
    assert diagnostics["truncated"] is False


def test_censored_availability_keeps_its_value_and_says_it_is_a_floor(parsed):
    records, _ = parsed
    assert records["AAPL"]["shortable_shares"] == 10_000_000.0
    assert records["AAPL"]["shortable_shares_basis"] == "floor"


def test_zero_availability_survives(parsed):
    """Regression against `positive()`: 0 shortable is the most squeeze-relevant
    reading in the file, and it is exactly the value a truthiness check eats."""
    records, _ = parsed
    assert records["AMC"]["shortable_shares"] == 0.0
    assert records["AMC"]["shortable_shares_basis"] == "exact"
    assert records["AMC"]["borrow_fee_pct"] == 100.0


def test_decimal_commas_are_not_multiplied_by_a_hundred(parsed):
    records, _ = parsed
    assert records["PCT"]["borrow_fee_pct"] == 7.5
    assert records["PCT"]["borrow_rebate_pct"] == -3.25


def test_thousands_commas_in_the_share_column(parsed):
    records, _ = parsed
    assert records["BRK B"]["shortable_shares"] == 1_250_000.0


def test_rows_with_nothing_known_emit_no_record_at_all(parsed):
    records, diagnostics = parsed
    assert "RUM WS" not in records
    assert "SLS" not in records
    assert diagnostics["rows_empty"] == 2


def test_a_short_row_is_skipped_and_counted_without_harming_its_neighbours(parsed):
    records, diagnostics = parsed
    assert "ZZZZ" not in records
    assert diagnostics["rows_skipped"] == 1
    assert set(records) == {"AAPL", "GME", "AMC", "BRK B", "PCT", "INDI"}
    assert diagnostics["symbols"] == 6


def test_duplicate_symbols_resolve_deterministically(parsed):
    records, diagnostics = parsed
    # Same currency, so the larger availability wins — not whichever came last.
    assert records["INDI"]["shortable_shares"] == 1_500_000.0
    assert diagnostics["duplicates"] == 1


def test_a_usd_row_beats_a_bigger_non_usd_row():
    body = (
        "#SYM|CUR|NAME|AVAILABLE\n"
        "XYZ|EUR|XYZ AG|9000000\n"
        "XYZ|USD|XYZ INC|1000\n"
    )
    records, _ = parse_shortable_file(body)
    assert records["XYZ"]["shortable_shares"] == 1000.0


def test_a_missing_eof_returns_records_and_flags_the_truncation():
    records, diagnostics = parse_shortable_file(TRUNCATED_FIXTURE)
    assert records["GME"]["borrow_fee_pct"] == 26.0
    assert diagnostics["truncated"] is True


def test_out_of_range_rates_are_dropped_and_counted():
    body = (
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\n"
        "XYZ|USD|XYZ INC|4.00|40000|5000\n"
    )
    records, diagnostics = parse_shortable_file(body)
    assert "borrow_fee_pct" not in records["XYZ"]
    assert records["XYZ"]["shortable_shares"] == 5000.0
    assert diagnostics["rates_dropped"] == 1
    # An NA is absent, not unusable, so it must not inflate that count.
    _, clean = parse_shortable_file(
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\nXYZ|USD|XYZ INC|NA|NA|5000\n"
    )
    assert clean["rates_dropped"] == 0


def test_a_pipe_inside_a_company_name_is_evidence_not_something_to_repair():
    body = (
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\n"
        "GOOD|USD|GOOD INC|4.00|0.30|5000\n"
        "BAD|USD|BAD | SONS INC|4.00|0.30|5000\n"
    )
    records, diagnostics = parse_shortable_file(body)
    assert set(records) == {"GOOD"}
    assert diagnostics["rows_skipped"] == 1


# --------------------------------------------------------------------------
# Symbol aliasing
# --------------------------------------------------------------------------


def test_symbol_aliases_substitute_separators_and_never_drop_a_token():
    assert symbol_aliases("BRK B") == {"BRKB", "BRK.B", "BRK-B", "BRK B"}
    assert "BRK" not in symbol_aliases("BRK B")
    assert symbol_aliases("GME") == set()


def test_class_symbols_resolve_through_the_alias_index(parsed):
    records, _ = parsed
    index = build_symbol_index(records)
    assert index["BRK.B"] == "BRK B"
    assert index["BRK-B"] == "BRK B"
    assert index["BRKB"] == "BRK B"
    assert index["BRK B"] == "BRK B"


def test_a_warrant_never_collapses_onto_its_underlying():
    """`RUM WS` is a different security from `RUM`; aliasing runs file->lookup
    only, so no generated spelling can ever reach the shorter symbol."""
    records, _ = parse_shortable_file(WARRANT_FIXTURE)
    index = build_symbol_index(records)
    assert index["RUM"] == "RUM"
    assert index["RUM.WS"] == "RUM WS"
    assert records[index["RUM"]]["borrow_fee_pct"] == 0.45


def test_an_alias_two_rows_both_claim_is_dropped_rather_than_guessed():
    body = (
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\n"
        "AB C|USD|AB C INC|4.00|0.30|5000\n"
        "A BC|USD|A BC INC|4.00|0.30|5000\n"
    )
    records, _ = parse_shortable_file(body)
    index = build_symbol_index(records)
    assert "ABC" not in index          # ambiguous
    assert index["AB.C"] == "AB C"     # unambiguous aliases still resolve
    assert index["A.BC"] == "A BC"


def test_an_exact_symbol_always_beats_an_alias():
    body = (
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\n"
        "ABC|USD|ABC INC|4.00|0.30|5000\n"
        "AB C|USD|AB C INC|4.00|9.90|1000\n"
    )
    records, _ = parse_shortable_file(body)
    index = build_symbol_index(records)
    assert index["ABC"] == "ABC"


# --------------------------------------------------------------------------
# The benchmark tripwire
# --------------------------------------------------------------------------


def test_implied_benchmark_is_plausible_on_a_correctly_read_file(parsed):
    records, _ = parsed
    assert implied_benchmark_rate(records) == pytest.approx(4.34, abs=0.01)


def test_implied_benchmark_flags_a_column_that_no_longer_means_what_it_says():
    records, _ = parse_shortable_file(WRONG_COLUMN_FIXTURE)
    benchmark = implied_benchmark_rate(records)
    assert benchmark is not None
    assert benchmark > 10.0  # nothing near an overnight funding rate


def test_implied_benchmark_cannot_see_a_straight_transposition(parsed):
    """An honest limit, asserted so nobody mistakes this check for a proof.

    Swapping the two rate columns preserves their sum, so the tripwire stays
    quiet while GME's 26% fee is reported as a 4.08% one. Only the header-driven
    parser guards that case.
    """
    records, _ = parsed
    swapped, _ = parse_shortable_file(
        SHORTABLE_FIXTURE.replace(
            "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE",
            "#SYM|CUR|NAME|CON|ISIN|FEERATE|REBATERATE|AVAILABLE",
        )
    )
    assert implied_benchmark_rate(swapped) == implied_benchmark_rate(records)
    assert swapped["GME"]["borrow_fee_pct"] == -21.75  # visibly wrong, silently accepted


def test_implied_benchmark_needs_more_than_a_couple_of_rows():
    body = (
        "#SYM|CUR|NAME|REBATERATE|FEERATE|AVAILABLE\n"
        "XYZ|USD|XYZ INC|4.00|0.30|5000\n"
    )
    records, _ = parse_shortable_file(body)
    assert implied_benchmark_rate(records) is None


# --------------------------------------------------------------------------
# Payload sniffing and refusal messages
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected",
    [
        ("", "empty response"),
        (None, "empty response"),
        ('<!DOCTYPE html><html><head><meta name="robots"></head>', "HTML page"),
        ("<html><body>nope</body></html>", "HTML page"),
        ("<div><noscript>enable js</noscript></div>", "HTML page"),
        ("something else entirely", "unexpected response"),
        ("#BOF usa 2026.08.07\nnothing that looks like a header", "unrecognised IBKR layout"),
    ],
)
def test_describe_payload(body, expected):
    assert expected in describe_payload(body)


def test_describe_payload_quotes_the_line_where_a_header_should_have_been():
    detail = describe_payload("#BOF usa 2026.08.07\nSYMBOL;CUR;FEE\nAAPL;USD;0.25\n")
    assert "SYMBOL;CUR;FEE" in detail


@pytest.mark.parametrize(
    "body, expected",
    [
        (SHORTABLE_FIXTURE, True),
        ("#SYM|CUR|NAME\nGME|USD|GAMESTOP\n", True),
        ("", False),
        (None, False),
        ("<!DOCTYPE html><table><tr><td>a|b</td></tr></table>", False),
        ("<html><noscript>x</noscript>a|b</html>", False),
        ("Service temporarily unavailable", False),
    ],
)
def test_looks_like_shortable_file(body, expected):
    assert looks_like_shortable_file(body) is expected


def test_decode_body_never_raises_on_bad_bytes():
    # Latin-1 accented name in a file we decode as UTF-8: replaced, not fatal.
    body = "#SYM|CUR|NAME\nSAN|EUR|BANCO SANTANDER S.A\xe9\n".encode("latin-1")
    text = decode_body(_StubResponse(body))
    assert "BANCO SANTANDER" in text


# --------------------------------------------------------------------------
# Cache key (DiskCache truncates at 120 chars)
# --------------------------------------------------------------------------


def test_cache_key_is_short_stable_and_order_independent():
    assert cache_key(["GME", "AMC"]) == cache_key(["amc", "gme"])
    assert cache_key(["GME"]).startswith("usa-")
    assert len(cache_key(["GME"])) < 120


def test_cache_key_does_not_collide_where_a_joined_ticker_key_would(tmp_path):
    """A 40-name watchlist joined with commas is ~320 chars; `DiskCache._path`
    keeps 120 of them, so two watchlists sharing a prefix share a cache file."""
    base = [f"TICK{i:03d}" for i in range(40)]
    cache = DiskCache(str(tmp_path), ttl_seconds=60)

    joined_a, joined_b = ",".join(base + ["ZZZA"]), ",".join(base + ["ZZZB"])
    assert cache._path("borrow", joined_a) == cache._path("borrow", joined_b)  # the bug

    key_a, key_b = cache_key(base + ["ZZZA"]), cache_key(base + ["ZZZB"])
    assert cache._path("borrow", key_a) != cache._path("borrow", key_b)


# --------------------------------------------------------------------------
# Source-level behaviour — no network, stub sessions only
# --------------------------------------------------------------------------


def test_source_is_skipped_until_it_is_opted_into(live_settings):
    """An unverified endpoint defaults to "skipped", not to a red scan."""
    source = BorrowSource(live_settings)
    assert source.available() is False
    assert source.collect(["GME"]) == {}
    assert source.health.status == "skipped"


def test_either_opt_in_switch_enables_the_source(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    assert BorrowSource(live_settings).available() is True
    monkeypatch.delenv("SQUEEZE_BORROW_ENABLED")
    monkeypatch.setenv("IBKR_SHORTABLE_URL", "https://mirror.example/usa.txt")
    assert BorrowSource(live_settings).available() is True


def test_offline_still_wins_over_the_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    offline = Settings(data_dir=str(tmp_path), cache_ttl_seconds=0, offline=True)
    assert BorrowSource(offline).available() is False


def test_collect_returns_canonical_fields_in_one_request(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    session = _StubSession(SHORTABLE_FIXTURE)
    source = BorrowSource(live_settings, session=session)

    records = source.collect(["GME", "AMC", "BRK.B", "NOSUCH"])

    assert records["GME"]["borrow_fee_pct"] == 26.0
    assert records["GME"]["borrow_as_of"] == "2026-08-07T18:15:02"
    assert records["AMC"]["shortable_shares"] == 0.0        # survives `clean_record`
    assert records["BRK.B"]["shortable_shares"] == 1_250_000.0
    assert "NOSUCH" not in records
    assert source.health.status == "partial"
    assert len(session.calls) == 1                          # whole file, never per ticker
    assert session.calls[0] == DEFAULT_URLS[0]


def test_the_file_is_parsed_once_per_process(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    session = _StubSession(SHORTABLE_FIXTURE)
    source = BorrowSource(live_settings, session=session)
    source.collect(["GME"])
    source.collect(["AMC"])
    assert len(session.calls) == 1


def test_a_cache_hit_is_reported_as_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    settings = Settings(data_dir=str(tmp_path), cache_ttl_seconds=600, offline=False, http_retries=0)
    BorrowSource(settings, session=_StubSession(SHORTABLE_FIXTURE)).collect(["GME"])

    # A fresh instance, so only the disk cache can answer — an empty body would
    # otherwise fail the sniff.
    second = BorrowSource(settings, session=_StubSession(""))
    records = second.collect(["GME"])
    assert records["GME"]["borrow_fee_pct"] == 26.0
    assert second.health.from_cache is True


def test_a_url_override_is_the_only_url_tried_and_is_what_meta_reports(live_settings, monkeypatch):
    monkeypatch.setenv("IBKR_SHORTABLE_URL", "https://mirror.example/usa.txt")
    session = _StubSession(SHORTABLE_FIXTURE)
    source = BorrowSource(live_settings, session=session)

    source.collect(["GME"])

    assert session.calls == ["https://mirror.example/usa.txt"]
    assert source.meta.url == "https://mirror.example/usa.txt"
    assert BorrowSource.meta.url == DEFAULT_URLS[0]  # the class default is untouched


def test_a_transport_failure_becomes_health_and_never_raises(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    session = _ExplodingSession()
    source = BorrowSource(live_settings, session=session)

    assert source.collect(["GME"]) == {}
    assert source.health.status == "error"
    assert "TLS handshake failed" in source.health.detail
    assert len(session.calls) == len(DEFAULT_URLS)  # default, then the one mirror


def test_an_html_body_is_described_rather_than_read_as_no_records(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession("<!DOCTYPE html><html>go away</html>"))

    assert source.collect(["GME"]) == {}
    assert source.health.status == "error"
    assert "HTML page" in source.health.detail


def test_an_unreadable_layout_surfaces_the_real_line(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    body = "#BOF|usa|2026.08.07|18:15:02\nAAPL,USD,APPLE INC,4.08,0.25,>10000000\n"
    source = BorrowSource(live_settings, session=_StubSession(body))

    assert source.collect(["AAPL"]) == {}
    assert source.health.status == "error"
    assert "AAPL,USD,APPLE INC" in source.health.detail


def test_no_matching_tickers_reads_differently_from_an_outage(live_settings, monkeypatch):
    """Without this detail a healthy fetch of a watchlist IBKR does not cover
    looks identical to a dead endpoint."""
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(SHORTABLE_FIXTURE))

    assert source.collect(["NOSUCH", "NEITHER"]) == {}
    assert source.health.status == "empty"
    assert "parsed 6 symbols" in source.health.detail
    assert "none of the 2 requested tickers" in source.health.detail


def test_durable_diagnostics_reach_the_health_line(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(TRUNCATED_FIXTURE))

    source.collect(["NOSUCH"])

    assert "truncated" in source.health.detail
    assert "malformed rows skipped" in source.health.detail


def test_an_implausible_benchmark_is_reported(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(WRONG_COLUMN_FIXTURE))

    records = source.collect(["GME", "AAPL", "AMC", "INDI"])

    assert records == {}                       # suspect fees are not worth serving
    assert source.health.status == "error"
    assert "column layout may have changed" in source.health.detail


def test_the_layout_tripwire_survives_the_partial_coverage_path(live_settings, monkeypatch):
    """The tripwire has to reach `health.detail` on the path this source is
    normally on, not just when coverage happens to be total.

    A plain `self.health.detail = ...` inside `_fetch` cannot do that:
    `DataSource.collect` overwrites the string with "<n>/<m> tickers covered"
    whenever some requested tickers are missing — and partial coverage is this
    source's steady state, so the one message worth reading would be the one
    always thrown away. Raising is what keeps it.
    """
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(WRONG_COLUMN_FIXTURE))

    # GME and AAPL are in the file, NOSUCH is not: partial coverage without the fix.
    source.collect(["GME", "AAPL", "NOSUCH"])

    assert "column layout may have changed" in source.health.detail
    assert "tickers covered" not in source.health.detail


def test_a_clean_run_records_the_benchmark_as_positive_evidence(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(SHORTABLE_FIXTURE))

    source.collect(["GME", "AMC", "AAPL", "PCT", "INDI", "BRK.B"])

    assert source.health.status == "ok"
    assert "implied benchmark rate 4.34%" in source.health.detail


def test_every_field_the_source_returns_is_one_it_declares(live_settings, monkeypatch):
    monkeypatch.setenv("SQUEEZE_BORROW_ENABLED", "1")
    source = BorrowSource(live_settings, session=_StubSession(SHORTABLE_FIXTURE))

    records = source.collect(["GME", "AMC", "AAPL", "PCT", "INDI", "BRK.B"])

    emitted = {field for values in records.values() for field in values}
    assert emitted <= set(source.meta.provides)
