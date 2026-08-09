"""Option-chain metrics, degradation and health — all offline, all fixture-driven.

Nothing here can reach the network: the chains are literal `pd.DataFrame`s built
from explicit row dicts, and the source-level tests inject a fake `yf` module so
the budget, the negative cache and the error/skip split are exercised in-process.
Per `conftest.py`, `SQUEEZE_OFFLINE=1` is pinned before `squeezelab` is imported.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from squeezelab.config import Settings
from squeezelab.sources import options as options_module
from squeezelab.sources.options import (
    CIRCUIT_BREAKER,
    LADDER_STRIKES,
    MIN_NTM_OI,
    ChainSlice,
    OptionsSource,
    atm_implied_vol,
    chain_to_rows,
    compute_options_metrics,
    in_ntm_band,
    is_standard_contract,
    oi_ladder,
    select_expiries,
    usable_iv,
)

SPOT = 20.00
FLOAT_SHARES = 20_000_000
EXPIRY = "2026-08-21"
DTE = 12
SCAN_DAY = date(2026, 8, 9)  # EXPIRY is 12 days out from here

# The band at this spot is [19.00, 26.00], so 15/17.5 are deep ITM and 30/35 are
# far OTM. Every expected number in this file is hand-computable off these rows.
LIQUID_CALL_ROWS = [
    #  strike,   oi,   volume,  iv
    (15.0, 5000, 200, 1.10),
    (17.5, 3000, 400, 0.90),
    (19.0, 1000, 800, 0.84),
    (20.0, 4000, 5000, 0.82),
    (21.0, 6000, 9000, 0.86),
    (22.5, 3000, 4000, 0.88),
    (25.0, 2000, 1000, 0.95),
    (30.0, 1500, 300, 1.20),
    (35.0, 500, 100, 1.60),
]
LIQUID_PUT_ROWS = [
    (15.0, 4000, 100, 1.15),
    (17.5, 3000, 250, 0.95),
    (20.0, 4000, 900, 0.83),
    (22.5, 2000, 150, 0.87),
]

NTM_CALL_OI_CONTRACTS = 1000 + 4000 + 6000 + 3000 + 2000        # 16,000
NTM_CALL_VOLUME = 800 + 5000 + 9000 + 4000 + 1000               # 19,800
TOTAL_CALL_OI = sum(row[1] for row in LIQUID_CALL_ROWS)         # 26,000
TOTAL_PUT_OI = sum(row[1] for row in LIQUID_PUT_ROWS)           # 13,000


def _contract_symbol(strike: float, right: str, root: str = "GME", expiry: str = EXPIRY) -> str:
    """An OCC-shaped symbol: ROOT + YYMMDD + C/P + 8-digit strike."""
    stamp = expiry.replace("-", "")[2:]
    return f"{root}{stamp}{right}{int(round(strike * 1000)):08d}"


def _frame(rows, right: str = "C", root: str = "GME", expiry: str = EXPIRY) -> pd.DataFrame:
    """A chain frame carrying the optional columns real yfinance frames have."""
    return pd.DataFrame(
        [
            {
                "contractSymbol": _contract_symbol(strike, right, root, expiry),
                "lastTradeDate": pd.Timestamp("2026-08-07T18:30:00Z"),
                "strike": float(strike),
                "lastPrice": round(max(0.05, SPOT - strike + 1.0), 2),
                "bid": 1.0,
                "ask": 1.2,
                "volume": float(volume) if volume is not None else float("nan"),
                "openInterest": float(oi) if oi is not None else float("nan"),
                "impliedVolatility": iv,
                "inTheMoney": strike < SPOT,
                "contractSize": "REGULAR",
                "currency": "USD",
            }
            for strike, oi, volume, iv in rows
        ]
    )


@pytest.fixture
def liquid_calls() -> pd.DataFrame:
    return _frame(LIQUID_CALL_ROWS)


@pytest.fixture
def liquid_puts() -> pd.DataFrame:
    return _frame(LIQUID_PUT_ROWS, right="P")


@pytest.fixture
def liquid_chain(liquid_calls, liquid_puts) -> ChainSlice:
    return ChainSlice(EXPIRY, DTE, chain_to_rows(liquid_calls), chain_to_rows(liquid_puts))


def _metrics(chains, *, spot=SPOT, float_shares=FLOAT_SHARES) -> dict:
    chains = chains if isinstance(chains, list) else [chains]
    return compute_options_metrics(chains, spot=spot, float_shares=float_shares)


# --------------------------------------------------------------------------
# Frame adapter
# --------------------------------------------------------------------------


def test_chain_to_rows_normalises_nan_to_none():
    frame = _frame([(20.0, None, None, 0.80)])
    row = chain_to_rows(frame)[0]
    assert row["open_interest"] is None
    assert row["volume"] is None
    assert row["strike"] == 20.0


def test_chain_to_rows_keeps_a_real_zero():
    """0 volume is a reading; None is the absence of one, and they differ."""
    row = chain_to_rows(_frame([(20.0, 0, 0, 0.80)]))[0]
    assert row["open_interest"] == 0.0
    assert row["volume"] == 0.0


EMPTY_COLUMNS = ("strike", "lastPrice", "volume", "openInterest", "impliedVolatility")


def test_chain_to_rows_handles_empty_frame():
    empty = pd.DataFrame({column: pd.Series(dtype="float64") for column in EMPTY_COLUMNS})
    assert chain_to_rows(empty) == []
    assert chain_to_rows(pd.DataFrame()) == []  # version drift: no columns at all
    assert chain_to_rows(None) == []
    assert chain_to_rows("not a frame") == []


def test_chain_to_rows_drops_adjusted_contracts():
    """Adjusted series deliver something other than 100 shares."""
    frame = _frame([(20.0, 4000, 100, 0.80)], root="GME1")
    assert chain_to_rows(frame) == []

    non_regular = _frame([(20.0, 4000, 100, 0.80)])
    non_regular.loc[0, "contractSize"] = "MINI"
    assert chain_to_rows(non_regular) == []


def test_is_standard_contract_defaults_to_keeping_the_row():
    # Neither revealing column present: the row is kept and the 100-share
    # multiplier is assumed, which is the documented limitation.
    assert is_standard_contract({"strike": 20.0}) is True
    assert is_standard_contract({"contractSymbol": "GME260821C00020000"}) is True


def test_optional_columns_absent_still_computes(liquid_calls, liquid_puts):
    """The version-drift guard: enrichment columns must never be load-bearing."""
    optional = ["contractSymbol", "bid", "ask", "inTheMoney", "contractSize", "lastTradeDate", "currency"]
    stripped_calls = liquid_calls.drop(columns=optional)
    stripped_puts = liquid_puts.drop(columns=optional)

    full = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(liquid_calls), chain_to_rows(liquid_puts)))
    thin = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(stripped_calls), chain_to_rows(stripped_puts)))
    assert thin == full


# --------------------------------------------------------------------------
# Near-the-money band
# --------------------------------------------------------------------------


def test_ntm_band_excludes_deep_itm_and_far_otm():
    assert [s for s in (15.0, 17.5, 19.0, 20.0, 21.0, 22.5, 25.0, 30.0, 35.0) if in_ntm_band(s, SPOT)] == [
        19.0, 20.0, 21.0, 22.5, 25.0
    ]


def test_ntm_band_includes_its_own_edges():
    """The bounds are float products; a strike sitting exactly on one counts."""
    assert in_ntm_band(SPOT * 0.95, SPOT) is True
    assert in_ntm_band(SPOT * 1.30, SPOT) is True


def test_ntm_call_oi_is_shares_not_contracts(liquid_chain):
    values = _metrics(liquid_chain)
    assert values["options_ntm_call_oi"] == NTM_CALL_OI_CONTRACTS * 100 == 1_600_000


def test_pct_float_arithmetic(liquid_chain):
    assert _metrics(liquid_chain)["options_ntm_call_oi_pct_float"] == pytest.approx(8.00)


def test_volume_oi_ratio(liquid_chain):
    values = _metrics(liquid_chain)
    assert values["options_call_volume_oi_ratio"] == pytest.approx(
        round(NTM_CALL_VOLUME / NTM_CALL_OI_CONTRACTS, 2)
    )
    assert values["options_call_volume_oi_ratio"] == 1.24


def test_volume_oi_ratio_absent_when_no_contract_reported_volume():
    """All-missing volume would sum to 0 and read as a measured zero."""
    rows = [(19.0, 1000, None, 0.84), (20.0, 4000, None, 0.82), (21.0, 6000, None, 0.86)]
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(_frame(rows)), []))
    assert values["options_ntm_call_oi"] == 11_000 * 100
    assert "options_call_volume_oi_ratio" not in values


def test_call_put_oi_ratio(liquid_chain):
    values = _metrics(liquid_chain)
    assert values["options_call_put_oi_ratio"] == pytest.approx(TOTAL_CALL_OI / TOTAL_PUT_OI)
    assert values["options_call_put_oi_ratio"] == 2.00


def test_call_put_oi_ratio_absent_without_puts(liquid_calls):
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(liquid_calls), []))
    assert "options_call_put_oi_ratio" not in values


# --------------------------------------------------------------------------
# ATM implied volatility
# --------------------------------------------------------------------------


def test_atm_iv_is_median_and_scaled_to_percent(liquid_chain):
    # Five nearest strikes by (|strike - spot|, strike): 20, 19, 21, 17.5, 22.5.
    # 17.5 and 22.5 are equidistant, which is what pins the tie-break.
    assert _metrics(liquid_chain)["options_atm_iv"] == pytest.approx(86.0)


def test_atm_iv_tie_break_is_deterministic(liquid_calls):
    rows = chain_to_rows(liquid_calls)
    assert atm_implied_vol(rows, SPOT) == atm_implied_vol(list(reversed(rows)), SPOT)


def test_atm_iv_above_one_hundred_percent_is_not_clipped():
    rows = [(20.0, 100, 10, 1.80), (21.0, 100, 10, 1.90), (19.0, 100, 10, 1.70)]
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(_frame(rows)), []))
    assert values["options_atm_iv"] == pytest.approx(180.0)


@pytest.mark.parametrize("raw", [0.00001, 12.5, None, float("nan"), -0.5])
def test_usable_iv_rejects_sentinels_and_garbage(raw):
    assert usable_iv(raw) is None


def test_atm_iv_discards_sentinel_and_garbage():
    """The outliers must not merely be outvoted — they must not take a slot."""
    rows = [
        (19.0, 200, 10, 0.00001),   # Yahoo's floor sentinel
        (20.0, 200, 10, 12.5),      # 1250% IV on an untraded strike
        (21.0, 200, 10, 0.80),
        (22.5, 200, 10, 0.90),
        (25.0, 200, 10, 1.00),
    ]
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(_frame(rows)), []))
    # Median of the three survivors, not of five values two of which are noise.
    assert values["options_atm_iv"] == pytest.approx(90.0)


def test_atm_iv_uses_the_anchor_expiry_only(liquid_calls):
    """Blending two points on the term structure describes neither."""
    far_rows = [(strike, oi, vol, 0.40) for strike, oi, vol, _ in LIQUID_CALL_ROWS]
    chains = [
        ChainSlice(EXPIRY, DTE, chain_to_rows(liquid_calls), []),
        ChainSlice("2026-09-18", 40, chain_to_rows(_frame(far_rows, expiry="2026-09-18")), []),
    ]
    assert _metrics(chains)["options_atm_iv"] == pytest.approx(86.0)


def test_atm_iv_needs_a_spot(liquid_calls):
    assert atm_implied_vol(chain_to_rows(liquid_calls), None) is None


# --------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------


def test_ladder_is_capped_and_sorted():
    rows = [(float(s), 500, 50, 0.80) for s in range(5, 40)]
    ladder = oi_ladder([ChainSlice(EXPIRY, DTE, chain_to_rows(_frame(rows)), [])], SPOT)
    assert len(ladder) == LADDER_STRIKES
    strikes = [entry["strike"] for entry in ladder]
    assert strikes == sorted(strikes)
    # Capped by distance from spot, then presented ascending.
    assert strikes == [float(s) for s in range(14, 26)]


def test_ladder_reports_contracts_not_shares(liquid_chain):
    ladder = _metrics(liquid_chain)["options_oi_ladder"]
    assert len(ladder) == len(LIQUID_CALL_ROWS)  # 9 strikes, under the cap
    by_strike = {entry["strike"]: entry for entry in ladder}
    assert by_strike[21.0]["call_oi"] == 6000       # contracts, not 600,000 shares
    assert by_strike[21.0]["call_volume"] == 9000


def test_ladder_sums_a_strike_across_expiries(liquid_calls):
    chains = [
        ChainSlice(EXPIRY, DTE, chain_to_rows(liquid_calls), []),
        ChainSlice("2026-09-18", 40, chain_to_rows(_frame(LIQUID_CALL_ROWS, expiry="2026-09-18")), []),
    ]
    by_strike = {entry["strike"]: entry for entry in oi_ladder(chains, SPOT)}
    assert by_strike[21.0]["call_oi"] == 12000


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


def test_below_min_oi_publishes_nothing_built_on_it():
    """A 1.67 ratio off 5 contracts is noise wearing a metric's clothes."""
    rows = [(20.0, 3, 5, 0.80), (21.0, 2, 1, 0.82)]
    assert sum(row[1] for row in rows) < MIN_NTM_OI
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(_frame(rows)), []))
    for absent in (
        "options_ntm_call_oi",
        "options_ntm_call_oi_pct_float",
        "options_call_volume_oi_ratio",
        "options_oi_ladder",
    ):
        assert absent not in values
    # The IV diagnostic does not depend on the OI floor, so it survives.
    assert values["options_atm_iv"] == pytest.approx(81.0)


def test_degenerate_chain_returns_nothing_rather_than_zeros():
    """NaN OI, NaN volume, both IV sentinels, a dead zero row and a stale print."""
    rows = [
        (19.0, None, None, 0.00001),
        (20.0, None, None, 12.5),
        (21.0, 0, 0, None),
        (35.0, None, None, None),
    ]
    frame = _frame(rows)
    frame.loc[3, "lastPrice"] = 0.01
    values = _metrics(ChainSlice(EXPIRY, DTE, chain_to_rows(frame), []))
    assert values == {}


def test_missing_float_keeps_shares_drops_pct(liquid_chain):
    values = _metrics(liquid_chain, float_shares=None)
    assert values["options_ntm_call_oi"] == 1_600_000
    assert "options_ntm_call_oi_pct_float" not in values


def test_missing_spot_keeps_call_put_ratio(liquid_chain):
    values = _metrics(liquid_chain, spot=None)
    assert values["options_call_put_oi_ratio"] == 2.00
    for absent in (
        "options_ntm_call_oi",
        "options_ntm_call_oi_pct_float",
        "options_call_volume_oi_ratio",
        "options_oi_ladder",
        "options_atm_iv",
    ):
        assert absent not in values


def test_no_chains_returns_empty():
    assert compute_options_metrics([], spot=SPOT, float_shares=FLOAT_SHARES) == {}
    assert compute_options_metrics([ChainSlice(EXPIRY, DTE, [], [])], spot=SPOT) == {}


# --------------------------------------------------------------------------
# Expiry selection
# --------------------------------------------------------------------------


def test_expiry_window_filters_by_dte():
    expiries = ["2026-08-09", "2026-08-12", "2026-10-08"]  # DTE 0, 3, 60
    assert select_expiries(expiries, SCAN_DAY) == [("2026-08-12", 3)]


def test_expiry_window_takes_the_two_nearest_inside_it():
    expiries = ["2026-09-18", "2026-08-12", "2026-08-09", "2026-10-08"]  # 40, 3, 0, 60
    assert select_expiries(expiries, SCAN_DAY) == [("2026-08-12", 3), ("2026-09-18", 40)]


def test_expiry_window_ignores_unparseable_dates():
    assert select_expiries(["not-a-date", None, "2026-08-12"], SCAN_DAY) == [("2026-08-12", 3)]


def test_window_days_reports_the_farthest_expiry(liquid_calls):
    near = ChainSlice("2026-08-12", 3, chain_to_rows(liquid_calls), [])
    far = ChainSlice("2026-09-18", 40, chain_to_rows(_frame(LIQUID_CALL_ROWS, expiry="2026-09-18")), [])

    single = _metrics(near)
    assert single["options_expiry"] == "2026-08-12"
    assert single["options_window_days"] == 3

    both = _metrics([far, near])  # deliberately out of order
    assert both["options_expiry"] == "2026-08-12"  # anchor is the nearest
    assert both["options_window_days"] == 40
    assert both["options_ntm_call_oi"] == 2 * NTM_CALL_OI_CONTRACTS * 100


# --------------------------------------------------------------------------
# Source health contract
# --------------------------------------------------------------------------


class _FakeChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


class _FakeTicker:
    """Stand-in for `yf.Ticker`, driven entirely by in-process frames."""

    def __init__(self, expiries=(), chains=None, spot=SPOT, error: Exception | None = None) -> None:
        self._expiries = tuple(expiries)
        self._chains = chains or {}
        self._spot = spot
        self._error = error
        self.chain_calls: list[str] = []
        self.option_list_calls = 0

    @property
    def options(self):
        self.option_list_calls += 1
        if self._error is not None:
            raise self._error
        return self._expiries

    def option_chain(self, expiry: str) -> _FakeChain:
        self.chain_calls.append(expiry)
        if expiry not in self._chains:
            raise KeyError(f"no chain for {expiry}")
        return self._chains[expiry]

    @property
    def fast_info(self):
        return {"last_price": self._spot}


class _FakeYf:
    def __init__(self, tickers: dict[str, _FakeTicker]) -> None:
        self.tickers = tickers

    def Ticker(self, symbol: str) -> _FakeTicker:  # noqa: N802 - mirrors yfinance
        return self.tickers.get(symbol, _FakeTicker())


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    """Settings that let the source actually run, with every network path faked."""
    return Settings(
        data_dir=str(tmp_path),
        cache_ttl_seconds=0,
        offline=False,
        enabled_sources=("options",),
        max_workers=2,
        request_delay=0.0,
    )


def _source(live_settings, fake_yf, monkeypatch, **kwargs) -> OptionsSource:
    monkeypatch.setattr(options_module, "yf", fake_yf)
    return OptionsSource(live_settings, today=SCAN_DAY, **kwargs)


def test_source_is_skipped_without_yfinance(live_settings, monkeypatch):
    """A missing package is a configuration gap, not a provider outage."""
    monkeypatch.setattr(options_module, "yf", None)
    source = OptionsSource(live_settings, today=SCAN_DAY)
    assert source.available() is False
    assert source.collect(["GME"]) == {}
    assert source.health.status == "skipped"


def test_source_is_skipped_when_disabled_by_env(live_settings, monkeypatch):
    monkeypatch.setattr(options_module, "yf", _FakeYf({}))
    monkeypatch.setenv("SQUEEZE_OPTIONS_ENABLED", "0")
    source = OptionsSource(live_settings, today=SCAN_DAY)
    assert source.available() is False
    assert source.collect(["GME"]) == {}
    assert source.health.status == "skipped"


def test_source_is_skipped_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(options_module, "yf", _FakeYf({}))
    source = OptionsSource(Settings(data_dir=str(tmp_path), offline=True), today=SCAN_DAY)
    assert source.available() is False


def test_ticker_without_options_does_not_raise(live_settings, monkeypatch):
    """Most of a squeeze watchlist has no chain at all; that is 'empty', not 'error'."""
    fake = _FakeYf({"SPRY": _FakeTicker(expiries=())})
    source = _source(live_settings, fake, monkeypatch)
    assert source.collect(["SPRY"]) == {}
    assert source.health.status == "empty"
    assert "list no options" in source.health.detail


def test_a_ticker_without_options_is_negatively_cached(live_settings, monkeypatch):
    fake = _FakeYf({"SPRY": _FakeTicker(expiries=())})
    source = _source(live_settings, fake, monkeypatch)
    source.collect(["SPRY"])
    source.collect(["SPRY"])
    # The second scan costs nothing: the 7-day negative cache answered it.
    assert fake.tickers["SPRY"].option_list_calls == 1


def test_source_maps_a_full_chain(live_settings, monkeypatch, liquid_calls, liquid_puts):
    fake = _FakeYf(
        {"GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)})}
    )
    source = _source(
        live_settings,
        fake,
        monkeypatch,
        float_lookup=lambda t: FLOAT_SHARES,
        price_lookup=lambda t: SPOT,
    )
    records = source.collect(["GME"])
    assert source.health.status == "ok"
    assert records["GME"]["options_ntm_call_oi"] == 1_600_000
    assert records["GME"]["options_ntm_call_oi_pct_float"] == pytest.approx(8.00)
    assert records["GME"]["options_expiry"] == EXPIRY
    assert len(records["GME"]["options_oi_ladder"]) == len(LIQUID_CALL_ROWS)


def test_source_falls_back_to_fast_info_for_spot(live_settings, monkeypatch, liquid_calls, liquid_puts):
    fake = _FakeYf(
        {"GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)})}
    )
    source = _source(live_settings, fake, monkeypatch)
    records = source.collect(["GME"])
    assert records["GME"]["options_ntm_call_oi"] == 1_600_000
    # No float injected, so the only cross-ticker comparable field is dropped.
    assert "options_ntm_call_oi_pct_float" not in records["GME"]


def test_a_bad_lookup_costs_one_field_not_the_source(live_settings, monkeypatch, liquid_calls, liquid_puts):
    def explode(_ticker):
        raise RuntimeError("float service is down")

    fake = _FakeYf(
        {"GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)})}
    )
    source = _source(live_settings, fake, monkeypatch, float_lookup=explode, price_lookup=lambda t: SPOT)
    records = source.collect(["GME"])
    assert source.health.status == "ok"
    assert "options_ntm_call_oi_pct_float" not in records["GME"]


def test_one_failing_expiry_does_not_lose_the_other(live_settings, monkeypatch, liquid_calls, liquid_puts):
    fake = _FakeYf(
        {
            "GME": _FakeTicker(
                expiries=("2026-08-12", EXPIRY),
                chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)},  # 2026-08-12 raises
            )
        }
    )
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    records = source.collect(["GME"])
    assert records["GME"]["options_expiry"] == EXPIRY
    assert records["GME"]["options_window_days"] == DTE


def test_every_ticker_failing_is_an_error_not_an_empty_scan(live_settings, monkeypatch):
    fake = _FakeYf({t: _FakeTicker(error=RuntimeError("429 too many requests")) for t in ("GME", "AMC")})
    source = _source(live_settings, fake, monkeypatch)
    assert source.collect(["GME", "AMC"]) == {}
    assert source.health.status == "error"
    assert "option-chain requests failed" in source.health.detail


def test_a_lone_failure_beside_names_with_no_chain_is_not_an_outage(live_settings, monkeypatch):
    """An empty scan is the ordinary end of a watchlist, not a dead provider.

    SPRY lists no chain, TSBX lists one nothing qualified in, AMC failed. Two of
    the three answered, so the provider is demonstrably up and the one failure is
    a partial-coverage note — reading this as "every request failed" turns a scan
    that worked as designed red.
    """
    thin = _frame([(20.0, None, None, 12.5)])  # a real chain; no OI, no usable IV
    fake = _FakeYf(
        {
            "SPRY": _FakeTicker(expiries=()),
            "TSBX": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(thin, thin)}),
            "AMC": _FakeTicker(error=RuntimeError("429 too many requests")),
        }
    )
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    assert source.collect(["SPRY", "TSBX", "AMC"]) == {}
    assert source.health.status == "empty"
    assert "1 ticker(s) failed" in source.health.detail


def test_a_provider_answering_only_with_nothing_still_trips_the_breaker(tmp_path, monkeypatch):
    """A skip is not proof of life, so it must not clear the failure streak.

    Single-worker settings pin the order: the failures and the skips strictly
    alternate, so a streak a skip resets never reaches the breaker and a provider
    that fails half its requests would be allowed to run the budget out.
    """
    settings = Settings(
        data_dir=str(tmp_path), cache_ttl_seconds=0, offline=False,
        enabled_sources=("options",), max_workers=1, request_delay=0.0,
    )
    tickers: list[str] = []
    tickers_by_symbol: dict[str, _FakeTicker] = {}
    for i in range(CIRCUIT_BREAKER):
        tickers_by_symbol[f"BAD{i}"] = _FakeTicker(error=RuntimeError("429 too many requests"))
        tickers_by_symbol[f"NIL{i}"] = _FakeTicker(expiries=())
        tickers += [f"BAD{i}", f"NIL{i}"]

    source = _source(settings, _FakeYf(tickers_by_symbol), monkeypatch)
    assert source.collect(tickers) == {}
    assert "consecutive failures" in source.health.detail
    # The breaker fired on the last failure, so the skip behind it never ran.
    assert tickers_by_symbol[f"NIL{CIRCUIT_BREAKER - 1}"].option_list_calls == 0


def test_a_provider_failure_is_not_cached_as_no_options(live_settings, monkeypatch):
    """Caching an outage for a week would keep the ticker dark long after it lifts."""
    fake = _FakeYf({"GME": _FakeTicker(error=RuntimeError("429 too many requests"))})
    source = _source(live_settings, fake, monkeypatch)
    source.collect(["GME"])
    source.collect(["GME"])
    assert fake.tickers["GME"].option_list_calls == 2


def test_one_bad_ticker_does_not_end_the_scan(live_settings, monkeypatch, liquid_calls, liquid_puts):
    fake = _FakeYf(
        {
            "GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)}),
            "AMC": _FakeTicker(error=RuntimeError("boom")),
        }
    )
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    records = source.collect(["GME", "AMC"])
    assert set(records) == {"GME"}
    assert source.health.status == "partial"


def test_expiries_outside_the_window_yield_nothing(live_settings, monkeypatch, liquid_calls, liquid_puts):
    far = "2027-01-15"
    fake = _FakeYf({"GME": _FakeTicker(expiries=(far,), chains={far: _FakeChain(liquid_calls, liquid_puts)})})
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    assert source.collect(["GME"]) == {}
    assert source.health.status == "empty"
    assert fake.tickers["GME"].chain_calls == []  # no request was ever worth making


def test_ticker_cap_is_applied_in_watchlist_order(live_settings, monkeypatch, liquid_calls, liquid_puts):
    chains = {EXPIRY: _FakeChain(liquid_calls, liquid_puts)}
    fake = _FakeYf({t: _FakeTicker(expiries=(EXPIRY,), chains=chains) for t in ("GME", "AMC")})
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    source.max_tickers = 1
    records = source.collect(["GME", "AMC"])
    assert set(records) == {"GME"}
    # The cap must cost requests, not just records: AMC is never touched.
    assert fake.tickers["AMC"].option_list_calls == 0


def test_an_exhausted_budget_stops_the_run(live_settings, monkeypatch, liquid_calls, liquid_puts):
    chains = {EXPIRY: _FakeChain(liquid_calls, liquid_puts)}
    fake = _FakeYf({t: _FakeTicker(expiries=(EXPIRY,), chains=chains) for t in ("GME", "AMC")})
    source = _source(live_settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    source.budget = -1.0  # already spent before the first unit of work
    assert source.collect(["GME", "AMC"]) == {}
    assert source.health.status == "empty"
    assert "budget" in source.health.detail


@pytest.fixture
def caching_settings(tmp_path) -> Settings:
    """Like `live_settings`, but with the derived-record cache actually on."""
    return Settings(
        data_dir=str(tmp_path), cache_ttl_seconds=600, offline=False,
        enabled_sources=("options",), max_workers=1, request_delay=0.0,
    )


def _gme(liquid_calls, liquid_puts) -> _FakeYf:
    return _FakeYf(
        {"GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)})}
    )


def test_derived_cache_turns_over_on_a_moved_spot(
    caching_settings, monkeypatch, liquid_calls, liquid_puts
):
    """Spot cuts the band, so a cache key without it serves the wrong strikes."""
    quote = {"spot": SPOT}
    fake = _gme(liquid_calls, liquid_puts)
    source = _source(caching_settings, fake, monkeypatch, price_lookup=lambda t: quote["spot"])
    assert source.collect(["GME"])["GME"]["options_ntm_call_oi"] == 1_600_000

    quote["spot"] = 30.00  # the band is now [28.50, 39.00]: only 30 and 35 qualify
    records = source.collect(["GME"])
    assert records["GME"]["options_ntm_call_oi"] == (1500 + 500) * 100


def test_derived_cache_turns_over_on_a_revised_float(
    caching_settings, monkeypatch, liquid_calls, liquid_puts
):
    """Float is the headline denominator; a stale one halves or doubles the number."""
    shares = {"float": FLOAT_SHARES}
    fake = _gme(liquid_calls, liquid_puts)
    source = _source(
        caching_settings, fake, monkeypatch,
        float_lookup=lambda t: shares["float"], price_lookup=lambda t: SPOT,
    )
    first = source.collect(["GME"])
    assert first["GME"]["options_ntm_call_oi_pct_float"] == pytest.approx(8.00)

    shares["float"] = FLOAT_SHARES * 2
    second = source.collect(["GME"])
    assert second["GME"]["options_ntm_call_oi_pct_float"] == pytest.approx(4.00)


def test_a_penny_of_drift_still_answers_from_the_derived_cache(
    caching_settings, monkeypatch, liquid_calls, liquid_puts
):
    """Quantised, not raw: a key that tracked every tick would never hit."""
    quote = {"spot": SPOT}
    fake = _gme(liquid_calls, liquid_puts)
    source = _source(caching_settings, fake, monkeypatch, price_lookup=lambda t: quote["spot"])
    source.collect(["GME"])

    fake.tickers["GME"].chain_calls.clear()
    quote["spot"] = SPOT + 0.01  # too small to move a band edge by a strike
    assert source.collect(["GME"])["GME"]["options_ntm_call_oi"] == 1_600_000
    assert fake.tickers["GME"].chain_calls == []


def test_ladder_and_series_survive_the_disk_cache(tmp_path, monkeypatch, liquid_calls, liquid_puts):
    """Everything published has to be JSON-serialisable — DiskCache dumps it."""
    settings = Settings(
        data_dir=str(tmp_path), cache_ttl_seconds=600, offline=False,
        enabled_sources=("options",), max_workers=1, request_delay=0.0,
    )
    fake = _FakeYf(
        {"GME": _FakeTicker(expiries=(EXPIRY,), chains={EXPIRY: _FakeChain(liquid_calls, liquid_puts)})}
    )
    source = _source(settings, fake, monkeypatch, price_lookup=lambda t: SPOT)
    first = source.collect(["GME"])

    warm = OptionsSource(settings, today=SCAN_DAY, price_lookup=lambda t: SPOT)
    fake.tickers["GME"].chain_calls.clear()
    assert warm.collect(["GME"]) == first
    assert fake.tickers["GME"].chain_calls == []  # a warm scan costs zero requests
    assert warm.health.from_cache is True
