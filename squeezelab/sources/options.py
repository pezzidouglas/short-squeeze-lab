"""
Option-chain source — gamma-side pressure, measured against the float.

Short interest says how many shares are owed. It says nothing about the other
half of a squeeze: whether anyone is *forced to buy*. Near-the-money call open
interest is the one free datum that speaks to that. Contracts already written in
strikes a plausible move converts from out-of-the-money to in-the-money oblige
whoever is short them to buy real shares to stay hedged, into a supply the short
interest already says is tight. That is the necessary condition for a reflexive
upside move, and it is what separates a gamma ramp from an ordinary crowded
short.

Four properties of this data shape every decision in this module:

* `openInterest` is T-1. OCC settles overnight, so on the day a ramp actually
  happens the OI is stale and `volume` is the only fresh column. That is why
  `options_call_volume_oi_ratio` ships next to the level rather than instead of
  it: it says whether the OI you are reading is about to be wrong, and in which
  direction.
* `impliedVolatility` is inverted from `lastPrice`, not from a mid, so on a
  strike whose last trade was days ago it is fiction. Yahoo emits sentinels
  (~1e-5) and garbage (hundreds of percent) instead of nulls — hence
  `IV_FLOOR`/`IV_CEIL`, and hence IV is published as unscored context only.
* `volume` and `openInterest` arrive as NaN, not 0, on quiet contracts.
* OI counts contracts, not positioning, and cannot be signed. The customary
  "dealers are short the calls" reading inverts wherever buy-writes and
  covered-call funds dominate. Read these fields as a filter that removes names,
  never as a trigger that selects them.

Deliberately not computed: IV skew (no delta column, and a difference of two
lastPrice-derived IVs amplifies that error instead of averaging it away), dollar
gamma (needs a Black-Scholes gamma off the same unreliable IV plus an assumption
about which side dealers are on), max pain (a price level, which is a category
error for a screener), and implied borrow from put-call parity (needs quality
two-sided quotes on exactly the strikes where quotes are worst).

There is no HTTP fallback when `yfinance` is missing. The options endpoint needs
Yahoo's cookie/crumb dance, and a half-working transport is worse than reporting
"skipped".
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any, Callable, Sequence

from .base import DataSource, DiskCache, SourceError, SourceMeta, SourceRecords, positive, to_float

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Computation constants
# --------------------------------------------------------------------------

CONTRACT_MULTIPLIER = 100  # shares per standard equity option contract

# The band is asymmetric on purpose. Deep-ITM calls are already hedged near
# delta 1 and generate almost no new buying; the dangerous inventory is the OTM
# strikes just above spot. An IV-scaled band (±1 expected move) would be more
# principled and is deliberately rejected: it would couple the most robust
# metric here to its least reliable input, so one bad IV print would silently
# move the denominator of the headline number. A fixed band is crude, but it is
# crude in a way a reader can see — it over-counts a 25%-IV megacap and
# under-counts a 200%-IV microcap, and says so.
NTM_BAND_LOW = 0.95
NTM_BAND_HIGH = 1.30

MIN_DTE = 1        # expiration day is decayed OI and closing churn
MAX_DTE = 45       # front weekly + monthly; past this, hedging urgency collapses
MAX_EXPIRIES = 2
MIN_NTM_OI = 100   # contracts; below this every ratio built on it is noise
ATM_STRIKES = 5
IV_FLOOR = 0.02    # Yahoo's near-zero sentinel
IV_CEIL = 5.00     # 500% IV on a penny strike is a parse artefact, not a quote
LADDER_STRIKES = 12

# OCC symbol tail: YYMMDD + C/P + 8-digit strike. Whatever precedes it is the
# root, and an adjusted series carries a digit there (GME -> GME1).
OCC_SUFFIX_LEN = 15

MAX_THREADS = 4          # yfinance shares one session and rate limiter
CIRCUIT_BREAKER = 5      # consecutive failures before the run gives up
EXPIRY_CACHE_TTL = 60 * 60 * 6           # expiration lists change weekly
NO_OPTIONS_TTL = 60 * 60 * 24 * 7        # "this ticker has no chain" stays true

# Significant figures the two denominators are quantised to inside the derived
# cache key. Spot keeps three because the band edges sit 5%/30% away from it, so
# a sub-percent tick cannot move which strikes are selected; float keeps two
# because it is revised in lumps and the percentage it feeds is published to
# four decimals off a number that is itself an estimate.
SPOT_KEY_SIGFIGS = 3
FLOAT_KEY_SIGFIGS = 2


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ChainSlice:
    """One expiry's calls and puts, already flattened off pandas."""

    expiry: str
    dte: int
    calls: list[dict[str, Any]]
    puts: list[dict[str, Any]]


class OptionsSource(DataSource):
    meta = SourceMeta(
        key="options",
        name="Yahoo option chains",
        url="https://finance.yahoo.com",
        cadence="realtime",
        provides=(
            "options_ntm_call_oi", "options_ntm_call_oi_pct_float",
            "options_call_volume_oi_ratio", "options_call_put_oi_ratio",
            "options_atm_iv", "options_oi_ladder", "options_expiry",
            "options_window_days",
        ),
        notes=(
            "Near-the-money call open interest against the float — the only free read on "
            "forced hedging demand. Needs yfinance; off by default because a cold scan "
            "costs up to four requests per ticker (an expiry list, one per selected expiry, "
            "and a spot). OI is T-1 and unsigned, so this is a filter that removes names, "
            "not a trigger that selects them."
        ),
    )

    def __init__(
        self,
        *args: Any,
        today: date | None = None,
        float_lookup: Callable[[str], float | None] | None = None,
        price_lookup: Callable[[str], float | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._today = today
        # Float and spot are denominators, not option data. Fetching them here
        # would duplicate Yahoo/Finviz at the wrong layer, so the caller injects
        # them the way SecFtdSource takes its float; without them the dependent
        # fields are simply not claimed.
        self.float_lookup = float_lookup
        self.price_lookup = price_lookup
        self.budget = _env_float("SQUEEZE_OPTIONS_BUDGET", 25.0)
        self.max_expiries = max(1, _env_int("SQUEEZE_OPTIONS_EXPIRIES", MAX_EXPIRIES))
        self.max_tickers = max(0, _env_int("SQUEEZE_OPTIONS_MAX_TICKERS", 40))
        self.enabled = _env_bool("SQUEEZE_OPTIONS_ENABLED", True)
        # Two extra tiers beside the derived-record cache, following FINRA's
        # two-TTL split. The negative cache is the highest-leverage entry here:
        # a squeeze watchlist is mostly small caps with no listed chain, and
        # without it every scan re-pays a request to rediscover that.
        self.expiry_cache = DiskCache(self.settings.cache_dir, EXPIRY_CACHE_TTL)
        self.miss_cache = DiskCache(self.settings.cache_dir, NO_OPTIONS_TTL)

    @property
    def today(self) -> date:
        return self._today or date.today()

    def available(self) -> bool:
        # yfinance carries Yahoo's cookie/crumb handling and there is no usable
        # HTTP fallback for the options endpoint, so a missing package is a
        # configuration gap — "skipped" — not a provider outage that should turn
        # every scan red.
        return super().available() and yf is not None and self.enabled

    # -- per-ticker work --------------------------------------------------

    def _stagger(self) -> None:
        """Space out real requests only.

        yfinance shares one global session and rate limiter across threads, so
        unstaggered parallelism turns a slow scan into a 429 storm. It sits on
        the request paths rather than on the per-ticker unit so a warm scan,
        which issues nothing, also waits for nothing.
        """
        if self.settings.request_delay > 0:
            time.sleep(self.settings.request_delay)

    def _expiries(self, ticker: str, handle: Any) -> list[str]:
        cached = self.expiry_cache.get("options_expiries", ticker)
        if isinstance(cached, list):
            self.health.from_cache = True
            return [str(e)[:10] for e in cached]
        self._stagger()
        try:
            raw = handle.options
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            raise SourceError(f"expiry list failed for {ticker}: {exc}") from exc
        expiries = [str(e)[:10] for e in (raw or ())]
        if expiries:
            # An empty list is not cached here; the 7-day negative cache owns it.
            self.expiry_cache.set("options_expiries", ticker, expiries)
        return expiries

    def _chain(self, ticker: str, handle: Any, expiry: str, dte: int) -> ChainSlice:
        self._stagger()
        try:
            chain = handle.option_chain(expiry)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            raise SourceError(f"option chain {expiry} failed for {ticker}: {exc}") from exc
        return ChainSlice(
            expiry=expiry,
            dte=dte,
            calls=chain_to_rows(getattr(chain, "calls", None)),
            puts=chain_to_rows(getattr(chain, "puts", None)),
        )

    def _spot(self, ticker: str, handle: Any) -> float | None:
        """Spot from the caller when it has one, otherwise from yfinance.

        Wiring `price_lookup` is the point: a scan already holds a price from
        Yahoo or Stooq, and reusing it drops a cold ticker from the four
        requests its budget assumes (expiry list, one per selected expiry, and
        this one) to three.
        """
        injected = _call_lookup(self.price_lookup, ticker)
        if injected:
            return injected
        fast = getattr(handle, "fast_info", None)
        for key in ("last_price", "lastPrice", "regularMarketPrice", "previous_close"):
            try:
                value = fast[key] if fast is not None else None
            except Exception:  # noqa: BLE001 - fast_info raises on unknown keys
                continue
            spot = positive(value)
            if spot:
                return spot
        return None

    def _ticker_record(self, ticker: str) -> dict[str, Any] | None:
        """Canonical values for one ticker, or None when it lists no options.

        Returning None rather than {} is what lets `_fetch` tell "this small cap
        has no chain" (coverage we never had) apart from "the provider failed"
        (an outage) — the two must not land on the same health status.
        """
        if self.miss_cache.get("options_none", ticker):
            self.health.from_cache = True
            return None
        try:
            handle = yf.Ticker(ticker)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            raise SourceError(f"yfinance could not open {ticker}: {exc}") from exc

        expiries = self._expiries(ticker, handle)
        if not expiries:
            # Only a genuinely empty list is cached. A request that *failed*
            # says nothing about whether the chain exists, and pinning that for
            # a week would keep the ticker dark long after Yahoo recovered.
            self.miss_cache.set("options_none", ticker, True)
            return None
        selected = select_expiries(expiries, self.today, max_expiries=self.max_expiries)
        if not selected:
            return None  # every listed expiry sits outside the window

        # Resolved before the cache probe because they are inputs to the cached
        # block, not labels on it: the band is cut around spot and the headline
        # percentage divides by float, so a key built from the expiries alone
        # would serve one scan's strike selection against another's price. Both
        # enter the key quantised — keyed raw, a one-cent tick would miss every
        # time and the cache would never answer anything.
        spot = self._spot(ticker, handle)
        float_shares = _call_lookup(self.float_lookup, ticker)
        cache_key = "_".join(
            [
                ticker,
                *(expiry for expiry, _ in selected),
                _key_bucket(spot, SPOT_KEY_SIGFIGS),
                _key_bucket(float_shares, FLOAT_KEY_SIGFIGS),
            ]
        )
        cached = self.cache.get("options", cache_key)
        if isinstance(cached, dict) and cached:
            self.health.from_cache = True
            return cached

        chains: list[ChainSlice] = []
        failures = 0
        for expiry, dte in selected:
            try:
                chains.append(self._chain(ticker, handle, expiry, dte))
            except SourceError:
                failures += 1  # one bad expiry must not cost the other one
        if not chains:
            if failures:
                raise SourceError(f"every option chain request for {ticker} failed")
            return None

        values = compute_options_metrics(chains, spot=spot, float_shares=float_shares)
        if values:
            self.cache.set("options", cache_key, values)
        return values or None

    def _ticker_unit(self, ticker: str, state: _RunState) -> dict[str, Any] | None:
        """One unit of work that never raises — a bad frame is health, not an outage."""
        with state.lock:
            if state.stopped:
                return None
            if time.monotonic() > state.deadline:
                # A wall-clock deadline, not a request count, is the load-bearing
                # control: a count does not protect against one ticker that hangs
                # for the full HTTP timeout.
                state.stopped = f"stopped after the {self.budget:.0f}s budget"
                return None
        try:
            values = self._ticker_record(ticker)
        except Exception as exc:  # noqa: BLE001 - yfinance raises a wide variety of types
            with state.lock:
                state.errors += 1
                state.streak += 1
                if state.streak >= CIRCUIT_BREAKER and not state.stopped:
                    state.stopped = f"stopped after {state.streak} consecutive failures: {exc}"
            return None
        with state.lock:
            if values is None:
                # A skip does not clear the streak. "No chain" is indistinguishable
                # from a provider answering every request with nothing, and a
                # breaker that a skip resets would never trip on that provider —
                # only a ticker that actually came back with data proves the
                # provider is alive.
                state.skipped += 1
            else:
                state.streak = 0
        return values

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        selected = list(tickers)[: self.max_tickers]
        if not selected:
            return {}
        notes: list[str] = []
        if len(selected) < len(tickers):
            notes.append(f"capped at {self.max_tickers} tickers")

        state = _RunState(deadline=time.monotonic() + self.budget)
        workers = max(1, min(self.settings.max_workers, MAX_THREADS))
        records: SourceRecords = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ticker, values in zip(selected, pool.map(lambda t: self._ticker_unit(t, state), selected)):
                if values:
                    records[ticker] = values

        if state.stopped:
            notes.append(state.stopped)
        if state.errors:
            notes.append(f"{state.errors} ticker(s) failed")
        if not records and state.skipped and not state.errors:
            # The distinction the health line exists to make: "these names have
            # no chain worth reading" is coverage we never had, and it must not
            # read like a provider that went dark.
            notes.append(f"{state.skipped} ticker(s) list no options inside {MAX_DTE}d")
        if notes:
            # Best effort: `collect` replaces the detail with its coverage line
            # on a partial run, so this survives on ok/empty/error only. The
            # coverage number it substitutes says most of it, and inventing a
            # second channel for the rest would be worse than losing it.
            self.health.detail = "; ".join(notes)[:300]
        if state.errors and not records and not state.skipped:
            # Only when *nothing* came back alive: an empty `records` on its own
            # is the ordinary outcome of a watchlist whose names either list no
            # chain or list one nothing qualified in, and calling that an outage
            # would turn the health line red on a scan that worked exactly as
            # designed. A ticker the provider answered — even with "no options" —
            # is proof it is up.
            raise SourceError(f"all {state.errors} option-chain requests failed")
        return records


@dataclass
class _RunState:
    """Shared budget/failure bookkeeping for one `_fetch`, guarded by its lock."""

    deadline: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    errors: int = 0
    skipped: int = 0
    streak: int = 0
    stopped: str = ""


# --------------------------------------------------------------------------
# Frame adapter (the only pandas-aware code in this module)
# --------------------------------------------------------------------------


def chain_to_rows(frame: Any) -> list[dict[str, Any]]:
    """Flatten a yfinance chain frame into plain rows.

    Everything downstream works on lists of dicts so the cached artifact and the
    test fixture are the same object, and so NaN -> None normalisation happens in
    exactly one place. Only `strike` is required: every other column, including
    the four yfinance nominally guarantees, is read through `.get`, because which
    columns a given yfinance version emits is not something this module can
    verify — and a missing optional column must cost enrichment, not the metric.
    """
    if frame is None:
        return []
    try:
        columns = set(frame.columns)
        raw_rows = frame.to_dict(orient="records")
    except (AttributeError, TypeError, ValueError):
        return []
    if "strike" not in columns:
        return []

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        strike = positive(raw.get("strike"))
        if strike is None or not is_standard_contract(raw):
            continue
        rows.append(
            {
                "strike": strike,
                # 0 is a real reading ("nobody traded this") and must survive;
                # only negatives and NaN are dropped.
                "open_interest": _non_negative(raw.get("openInterest")),
                "volume": _non_negative(raw.get("volume")),
                "implied_volatility": usable_iv(raw.get("impliedVolatility")),
                "last_price": to_float(raw.get("lastPrice")),
            }
        )
    return rows


def is_standard_contract(row: dict[str, Any]) -> bool:
    """Reject adjusted deliverables when the optional columns reveal them.

    Post-split and post-merger contracts deliver something other than 100
    shares, and they cluster in exactly the distressed, heavily shorted names
    this tool watches, so counting them at the standard multiplier inflates the
    headline number. Both columns that expose them are optional enrichment: when
    neither is present the row is kept and the shares figure silently carries
    the 100-share assumption — which is why the dashboard has to say the
    multiplier is assumed rather than pretend otherwise.
    """
    size = row.get("contractSize")
    if isinstance(size, str) and size.strip() and size.strip().upper() != "REGULAR":
        return False
    symbol = row.get("contractSymbol")
    if isinstance(symbol, str) and len(symbol) > OCC_SUFFIX_LEN:
        root = symbol[:-OCC_SUFFIX_LEN]
        if root and root[-1].isdigit():
            return False
    return True


# --------------------------------------------------------------------------
# Metrics (pure functions — unit tested without pandas or yfinance)
# --------------------------------------------------------------------------


def days_to_expiry(expiry: Any, today: date) -> int | None:
    try:
        return (date.fromisoformat(str(expiry)[:10]) - today).days
    except (TypeError, ValueError):
        return None


def select_expiries(
    expiries: Sequence[Any],
    today: date,
    *,
    max_expiries: int = MAX_EXPIRIES,
) -> list[tuple[str, int]]:
    """The expiries worth spending a request on, nearest first.

    Expiration day itself is dropped: its OI is decayed and its volume is
    closing churn. Past MAX_DTE the hedging urgency that makes this metric
    mechanical is gone, and including LEAPS OI would dilute the one number here
    that has a mechanism behind it while making it monotonically larger for any
    heavily optionable large cap.
    """
    dated: list[tuple[str, int]] = []
    for expiry in expiries or ():
        dte = days_to_expiry(expiry, today)
        if dte is None or dte < MIN_DTE or dte > MAX_DTE:
            continue
        dated.append((str(expiry)[:10], dte))
    dated.sort(key=lambda item: (item[1], item[0]))
    return dated[: max(0, max_expiries)]


def in_ntm_band(strike: float, spot: float) -> bool:
    """Is this strike inside the band where hedging demand is convex?

    A hair of tolerance on each edge: both bounds are a float ratio times a
    float price, so a strike sitting exactly on 0.95x could otherwise land on
    either side of the comparison for no reason a reader could predict.
    """
    tolerance = abs(spot) * 1e-9
    return spot * NTM_BAND_LOW - tolerance <= strike <= spot * NTM_BAND_HIGH + tolerance


def usable_iv(value: Any) -> float | None:
    """An implied vol worth believing, as a 0-1 fraction, or None.

    Yahoo derives IV from `lastPrice`, so an untraded strike yields either its
    near-zero floor sentinel or a value in the hundreds of percent. Neither is a
    quote, and neither belongs in a median.
    """
    iv = to_float(value)
    if iv is None or iv < IV_FLOOR or iv > IV_CEIL:
        return None
    return iv


def atm_implied_vol(rows: Sequence[dict[str, Any]], spot: float | None) -> float | None:
    """Median IV of the strikes nearest spot, as a 0-1 fraction.

    Filtering before selecting matters: a sentinel must not consume a slot a
    real quote would have filled. The strike is carried in the sort key as a
    tie-break because on a symmetric ladder two strikes are equidistant from
    spot, and without it the median flips between runs on identical input.
    """
    if not spot or spot <= 0:
        return None
    # Re-applying the filter keeps this correct for rows built by hand as well
    # as rows off `chain_to_rows`; `usable_iv` is idempotent.
    usable = [(row["strike"], usable_iv(row.get("implied_volatility"))) for row in rows]
    usable = [(strike, iv) for strike, iv in usable if iv is not None]
    if not usable:
        return None
    usable.sort(key=lambda item: (abs(item[0] - spot), item[0]))
    return median([iv for _, iv in usable[:ATM_STRIKES]])


def oi_ladder(
    chains: Sequence[ChainSlice],
    spot: float,
    *,
    limit: int = LADDER_STRIKES,
) -> list[dict[str, float]]:
    """The `limit` strikes nearest spot, ascending, with OI in **contracts**.

    The scalar cannot tell one 12,000-contract wall at $30 from 12,000 contracts
    smeared over ten strikes, and only the first pins or ramps — the ladder is
    the qualitative check that stops the scalar from being trusted blindly. OI is
    summed across the included expiries at each strike, because a wall at a
    strike is a wall whether it sits in the weekly or the monthly. The unit is
    contracts, deliberately unlike the scalar's shares: the ladder is read
    against the chain in front of a human, not against the float.
    """
    by_strike: dict[float, dict[str, float]] = {}
    for chain in chains:
        for row in chain.calls:
            entry = by_strike.setdefault(row["strike"], {"call_oi": 0.0, "call_volume": 0.0})
            entry["call_oi"] += row["open_interest"] or 0.0
            entry["call_volume"] += row["volume"] or 0.0
    if not by_strike:
        return []
    nearest = sorted(by_strike, key=lambda strike: (abs(strike - spot), strike))[: max(0, limit)]
    return [
        {
            "strike": strike,
            "call_oi": round(by_strike[strike]["call_oi"], 2),
            "call_volume": round(by_strike[strike]["call_volume"], 2),
        }
        for strike in sorted(nearest)
    ]


def compute_options_metrics(
    chains: Sequence[ChainSlice],
    *,
    spot: float | None = None,
    float_shares: float | None = None,
) -> dict[str, Any]:
    """Canonical fields for one ticker from its selected chains.

    Every field degrades on its own: no float drops the percentage but keeps the
    share count, no spot drops everything that needs a band but keeps the
    call/put ratio, and an OI total under the noise floor publishes nothing built
    on it rather than a ratio derived from five contracts.
    """
    usable = sorted(
        [chain for chain in chains if chain and (chain.calls or chain.puts)],
        key=lambda chain: (chain.dte, chain.expiry),
    )
    if not usable:
        return {}

    values: dict[str, Any] = {}
    call_oi_total = _open_interest_total(chain.calls for chain in usable)
    put_oi_total = _open_interest_total(chain.puts for chain in usable)
    if call_oi_total is not None and put_oi_total:
        # Context only, never scored: the arithmetic is exact but the reading is
        # weak — put OI is inflated by hedges and collars carrying no view, and
        # nothing here distinguishes bought calls from sold ones.
        values["options_call_put_oi_ratio"] = round(call_oi_total / put_oi_total, 2)

    # IV comes from the anchor expiry alone. Blending two expiries' IVs would
    # average across two different points on the term structure and produce a
    # number that describes neither.
    atm_iv = atm_implied_vol(usable[0].calls, spot)
    if atm_iv is not None:
        # A de-rater ("is this already priced?"), not a predictor. Values above
        # 100% are real and must not be clipped.
        values["options_atm_iv"] = round(atm_iv * 100, 2)

    if spot and spot > 0:
        band = [row for chain in usable for row in chain.calls if in_ntm_band(row["strike"], spot)]
        ntm_oi = _sum_field(band, "open_interest")
        if ntm_oi >= MIN_NTM_OI:
            values["options_ntm_call_oi"] = round(ntm_oi * CONTRACT_MULTIPLIER, 2)
            if float_shares and float_shares > 0:
                values["options_ntm_call_oi_pct_float"] = round(
                    ntm_oi * CONTRACT_MULTIPLIER / float_shares * 100, 4
                )
            if any(row["volume"] is not None for row in band):
                # Only claimed when at least one contract reported volume: with
                # every value missing the sum is 0 and the ratio would read as a
                # measured "no fresh activity" that was never measured.
                values["options_call_volume_oi_ratio"] = round(_sum_field(band, "volume") / ntm_oi, 2)
            ladder = oi_ladder(usable, spot)
            if ladder:
                values["options_oi_ladder"] = ladder

    if not values:
        return {}
    # Attached last so they describe something: a record carrying only an expiry
    # would claim coverage it does not have.
    values["options_expiry"] = usable[0].expiry
    values["options_window_days"] = usable[-1].dte
    return values


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _non_negative(value: Any) -> float | None:
    parsed = to_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _key_bucket(value: float | None, sigfigs: int) -> str:
    """A cache-key fragment for a denominator, stable under noise.

    Written as mantissa+exponent rather than a rounded float so the fragment
    does not carry binary-repr artefacts into a filename. A missing value gets
    its own token: "no float" and "some float" produce different records and
    must not collide.
    """
    if not value or value <= 0:
        return "na"
    exponent = math.floor(math.log10(value)) - sigfigs + 1
    return f"{round(value / 10 ** exponent)}e{exponent}"


def _sum_field(rows: Sequence[dict[str, Any]], field_name: str) -> float:
    return sum(row[field_name] or 0.0 for row in rows)


def _open_interest_total(row_groups: Any) -> float | None:
    """Total open interest across groups of rows, or None when none reported any."""
    total = 0.0
    seen = False
    for rows in row_groups:
        for row in rows:
            if row["open_interest"] is not None:
                total += row["open_interest"]
                seen = True
    return total if seen else None


def _call_lookup(lookup: Callable[[str], float | None] | None, ticker: str) -> float | None:
    if lookup is None:
        return None
    try:
        return positive(lookup(ticker))
    except Exception:  # noqa: BLE001 - a bad denominator drops one field, not the source
        return None
