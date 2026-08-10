"""
Short Squeeze Lab — configuration.

Holds the canonical field vocabulary shared by every data source, the per-field
source priority used when sources disagree, the scoring weight presets, and the
runtime settings read from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Canonical field vocabulary
# --------------------------------------------------------------------------
# Every source normalises whatever it fetches into these names. `tolerance` is
# the relative difference (0-1) two sources may report before we call it a
# genuine disagreement rather than rounding/timing noise.

TEXT = "text"
NUMERIC = "numeric"
DATE = "date"
SERIES = "series"  # a list of points; merged by priority, never averaged


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    label: str
    unit: str = ""
    tolerance: float = 0.05
    # Fields used by the scoring engine; missing ones cost data confidence.
    key: bool = False


FIELD_SPECS: dict[str, FieldSpec] = {
    f.name: f
    for f in [
        FieldSpec("ticker", TEXT, "Ticker"),
        FieldSpec("name", TEXT, "Company"),
        FieldSpec("sector", TEXT, "Sector"),
        FieldSpec("industry", TEXT, "Industry"),
        FieldSpec("exchange", TEXT, "Exchange"),
        FieldSpec("cik", TEXT, "SEC CIK"),
        FieldSpec("current_price", NUMERIC, "Price", "$", 0.03, key=True),
        FieldSpec("previous_close", NUMERIC, "Prev close", "$", 0.03),
        FieldSpec("market_cap", NUMERIC, "Market cap", "$", 0.10),
        FieldSpec("float", NUMERIC, "Float", "sh", 0.15, key=True),
        FieldSpec("shares_outstanding", NUMERIC, "Shares out", "sh", 0.10),
        FieldSpec("short_pct_float", NUMERIC, "Short % float", "%", 0.20, key=True),
        FieldSpec("short_ratio", NUMERIC, "Days to cover", "d", 0.25, key=True),
        FieldSpec("shares_short", NUMERIC, "Shares short", "sh", 0.20, key=True),
        FieldSpec("shares_short_prior", NUMERIC, "Shares short (prior)", "sh", 0.20),
        FieldSpec("short_interest_date", DATE, "SI as of"),
        FieldSpec("avg_volume_10d", NUMERIC, "Avg vol 10d", "sh", 0.25, key=True),
        FieldSpec("avg_volume_3m", NUMERIC, "Avg vol 3m", "sh", 0.25),
        FieldSpec("fifty_two_week_high", NUMERIC, "52w high", "$", 0.05, key=True),
        FieldSpec("fifty_two_week_low", NUMERIC, "52w low", "$", 0.05, key=True),
        FieldSpec("beta", NUMERIC, "Beta", "", 0.30),
        FieldSpec("off_exchange_short_pct", NUMERIC, "Off-exch short vol", "%", 0.15, key=True),
        FieldSpec("off_exchange_trend", NUMERIC, "Off-exch 5d vs 20d", "pp", 0.30),
        FieldSpec("off_exchange_series", SERIES, "Off-exch short vol history"),
        # -- Stock loan (IBKR) ------------------------------------------------
        # The measured cost and availability of borrow, rather than the
        # days-to-cover proxy for it.
        FieldSpec("borrow_fee_pct", NUMERIC, "Borrow fee", "%/yr", 0.25),
        FieldSpec("borrow_rebate_pct", NUMERIC, "Rebate rate", "%/yr", 0.25),
        FieldSpec("shortable_shares", NUMERIC, "Shortable shares", "sh", 0.30),
        FieldSpec("shortable_shares_basis", TEXT, "Shortable basis"),
        FieldSpec("borrow_as_of", DATE, "Borrow as of"),
        # -- Fails to deliver (SEC) -------------------------------------------
        # ~40 days stale by construction, and a weak corroboration of borrow
        # scarcity rather than a reading of short interest. `ftd_date` travels
        # with every number here for exactly that reason.
        FieldSpec("ftd_shares_peak", NUMERIC, "FTD peak", "sh", 0.10),
        FieldSpec("ftd_shares_avg", NUMERIC, "FTD average", "sh", 0.10),
        FieldSpec("ftd_persist_days", NUMERIC, "FTD settlement days", "d", 0.20),
        FieldSpec("ftd_pct_float", NUMERIC, "FTD peak % float", "%", 0.15),
        FieldSpec("ftd_notional", NUMERIC, "FTD peak notional", "$", 0.15),
        FieldSpec("ftd_trend", NUMERIC, "FTD recent vs window", "%", 0.30),
        FieldSpec("ftd_date", DATE, "FTD as of"),
        FieldSpec("ftd_series", SERIES, "FTD history"),
        # -- Option chain (Yahoo) ---------------------------------------------
        # Forced-hedging demand. Open interest is T-1 and unsigned, so the
        # volume/OI ratio ships beside the level to say how stale it is.
        FieldSpec("options_ntm_call_oi", NUMERIC, "NTM call OI", "contracts", 0.15),
        FieldSpec("options_ntm_call_oi_pct_float", NUMERIC, "NTM call OI % float", "%", 0.15),
        FieldSpec("options_call_volume_oi_ratio", NUMERIC, "Call volume / OI", "x", 0.25),
        FieldSpec("options_call_put_oi_ratio", NUMERIC, "Call / put OI", "x", 0.25),
        FieldSpec("options_atm_iv", NUMERIC, "ATM implied vol", "%", 0.20),
        FieldSpec("options_expiry", DATE, "Nearest expiry"),
        FieldSpec("options_window_days", NUMERIC, "Expiry window", "d", 0.30),
        FieldSpec("options_oi_ladder", SERIES, "Call OI by strike"),
    ]
}

# Coverage of these fields is 40% of a ticker's data confidence, so membership
# is restricted to fields a *default* scan can actually obtain. Borrow, FTD and
# option fields are all opt-in (a credential or an env flag), and marking them
# key would score every stock scanner user's data as incomplete for declining
# to enable a source they were never asked to enable.
KEY_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELD_SPECS.items() if s.key)

# --------------------------------------------------------------------------
# Per-field source priority
# --------------------------------------------------------------------------
# First source in the list that reports a usable value wins. Everything else is
# kept as a candidate so the dashboard can show the disagreement instead of
# hiding it. Sources not listed for a field fall back to DEFAULT_PRIORITY.

DEFAULT_PRIORITY: tuple[str, ...] = (
    "yahoo", "finviz", "finra", "sec", "stooq", "borrow", "options", "sec_ftd", "snapshot",
)

FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    # FINRA publishes the official semi-monthly short interest; it is the
    # authority even though it lags. Finviz refreshes faster than Yahoo.
    "shares_short": ("finra", "finviz", "yahoo", "snapshot"),
    "short_interest_date": ("finra", "yahoo", "snapshot"),
    "short_pct_float": ("finra", "finviz", "yahoo", "snapshot"),
    "short_ratio": ("finviz", "finra", "yahoo", "snapshot"),
    # Off-exchange short volume only comes from FINRA's daily RegSHO files.
    "off_exchange_short_pct": ("finra", "snapshot"),
    "off_exchange_trend": ("finra", "snapshot"),
    "off_exchange_series": ("finra", "snapshot"),
    # Prices: Stooq is an independent end-of-day cross-check on Yahoo.
    "current_price": ("yahoo", "stooq", "finviz", "snapshot"),
    "previous_close": ("yahoo", "stooq", "snapshot"),
    "avg_volume_10d": ("yahoo", "stooq", "finviz", "snapshot"),
    "avg_volume_3m": ("yahoo", "stooq", "finviz", "snapshot"),
    "fifty_two_week_high": ("stooq", "yahoo", "snapshot"),
    "fifty_two_week_low": ("stooq", "yahoo", "snapshot"),
    # SEC filings are authoritative for the share count.
    "shares_outstanding": ("sec", "yahoo", "finviz", "snapshot"),
    "cik": ("sec",),
    "float": ("yahoo", "finviz", "snapshot"),
    "sector": ("finviz", "yahoo", "snapshot"),
    "industry": ("finviz", "yahoo", "snapshot"),
    "name": ("yahoo", "sec", "finviz", "snapshot"),
}

# Borrow, FTD and option fields have exactly one live producer each, so listing
# them says little about precedence — but a field absent from this map falls
# back to DEFAULT_PRIORITY, which puts Yahoo and Finviz ahead of the source
# that actually produces it, and reads in `coverage_matrix` as "nobody owns
# it". The offline snapshot trails each one so it fills a gap and never wins.
for _sole_source, _fields in (
    ("borrow", ("borrow_fee_pct", "borrow_rebate_pct", "shortable_shares",
                "shortable_shares_basis", "borrow_as_of")),
    ("sec_ftd", ("ftd_shares_peak", "ftd_shares_avg", "ftd_persist_days", "ftd_pct_float",
                 "ftd_notional", "ftd_trend", "ftd_date", "ftd_series")),
    ("options", ("options_ntm_call_oi", "options_ntm_call_oi_pct_float",
                 "options_call_volume_oi_ratio", "options_call_put_oi_ratio",
                 "options_atm_iv", "options_expiry", "options_window_days",
                 "options_oi_ladder")),
):
    for _field_name in _fields:
        FIELD_PRIORITY[_field_name] = (_sole_source, "snapshot")
del _sole_source, _fields, _field_name


def priority_for(field_name: str) -> tuple[str, ...]:
    return FIELD_PRIORITY.get(field_name, DEFAULT_PRIORITY)


# --------------------------------------------------------------------------
# Scoring weight presets
# --------------------------------------------------------------------------
# Each preset sums to 1.0 across the six components. `classic` reproduces the
# original 40/25/20/15 model, carrying a zero on every *component* added since,
# so historical scores stay comparable. A zero-weight component is excluded
# from scoring outright rather than scored and multiplied by nothing, so a
# whole new component cannot perturb it even indirectly.
#
# What `classic` does not freeze is the data inside the components it does
# weight: enabling the stock-loan feed sharpens `borrow` under every preset,
# this one included. It is a weighting, not a data vintage — comparability
# holds across presets on one scan, and across scans run on the same sources.

WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "classic": {
        "si": 0.40,
        "borrow": 0.25,
        "liquidity": 0.20,
        "momentum": 0.15,
        "flow": 0.00,
        "gamma": 0.00,
    },
    "balanced": {
        "si": 0.28,
        "borrow": 0.20,
        "liquidity": 0.16,
        "momentum": 0.10,
        "flow": 0.16,
        "gamma": 0.10,
    },
    "tight_float": {
        "si": 0.28,
        "borrow": 0.28,
        "liquidity": 0.26,
        "momentum": 0.06,
        "flow": 0.05,
        "gamma": 0.07,
    },
    "momentum": {
        "si": 0.20,
        "borrow": 0.12,
        "liquidity": 0.13,
        "momentum": 0.32,
        "flow": 0.13,
        "gamma": 0.10,
    },
    "flow": {
        "si": 0.18,
        "borrow": 0.14,
        "liquidity": 0.13,
        "momentum": 0.08,
        "flow": 0.35,
        "gamma": 0.12,
    },
    # Leans on the two components that need an opt-in source: measured borrow
    # cost and near-the-money call open interest. Scores every name in a scan
    # that has neither identically to `balanced` minus those inputs, because a
    # component with no data is dropped and the rest renormalised — so this
    # preset is only worth selecting with `borrow` and `options` enabled.
    "gamma": {
        "si": 0.18,
        "borrow": 0.24,
        "liquidity": 0.12,
        "momentum": 0.08,
        "flow": 0.08,
        "gamma": 0.30,
    },
}

DEFAULT_PRESET = "balanced"

COMPONENT_LABELS: dict[str, str] = {
    "si": "Short interest density",
    "borrow": "Borrow pressure",
    "liquidity": "Liquidity trap",
    "momentum": "Momentum fuel",
    "flow": "Off-exchange flow",
    "gamma": "Forced-hedging demand",
}


def resolve_weights(preset: str | None = None, overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Return a normalised weight map for a preset name, with optional overrides."""
    name = (preset or DEFAULT_PRESET).lower()
    if name not in WEIGHT_PRESETS:
        raise ValueError(f"unknown weight preset {preset!r}; choose from {sorted(WEIGHT_PRESETS)}")
    weights = dict(WEIGHT_PRESETS[name])
    if overrides:
        unknown = set(overrides) - set(weights)
        if unknown:
            raise ValueError(f"unknown weight component(s): {sorted(unknown)}")
        weights.update({k: float(v) for k, v in overrides.items()})
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


# --------------------------------------------------------------------------
# Setup labels
# --------------------------------------------------------------------------

SETUP_BANDS: tuple[tuple[float, str, str], ...] = (
    (80.0, "HIGH SQUEEZE RISK", "high"),
    (60.0, "ELEVATED", "elevated"),
    (40.0, "WATCH", "watch"),
    (0.0, "LOW", "low"),
)


def setup_label(score: float) -> tuple[str, str]:
    """Map a squeeze score to its (label, css tier)."""
    for threshold, label, tier in SETUP_BANDS:
        if score >= threshold:
            return label, tier
    return "LOW", "low"


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

DEFAULT_WATCHLIST: tuple[str, ...] = (
    "SPRY", "PCT", "INDI", "SLS", "RUM", "LUCK", "PSNY",
    "GME", "AMC", "NVAX", "LCID",
    "RIVN", "BYND", "CVNA", "UPST", "AFRM", "HOOD",
    "TSLA", "AAPL", "MSFT", "NVDA", "AMD", "INTC",
    "PLTR", "SOFI", "NIO", "XPEV", "LI",
    "ASTS", "RKLB", "IONQ", "QBTS",
    "SOUN", "BBAI", "SMCI", "MSTR",
)


# --------------------------------------------------------------------------
# Runtime settings
# --------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


@dataclass
class Settings:
    """Runtime knobs. Everything is overridable through the environment."""

    data_dir: str = field(default_factory=lambda: os.environ.get("SQUEEZE_DATA_DIR", "data"))
    # Disk cache for source responses. 0 disables it.
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("SQUEEZE_CACHE_TTL", 1800))
    # In-process cache for whole scans, so one dashboard load doesn't fan out
    # across every provider once per endpoint. Independent of the disk cache.
    scan_cache_ttl: int = field(default_factory=lambda: _env_int("SQUEEZE_SCAN_TTL", 600))
    http_timeout: float = field(default_factory=lambda: _env_float("SQUEEZE_HTTP_TIMEOUT", 12.0))
    http_retries: int = field(default_factory=lambda: _env_int("SQUEEZE_HTTP_RETRIES", 2))
    max_workers: int = field(default_factory=lambda: _env_int("SQUEEZE_MAX_WORKERS", 6))
    request_delay: float = field(default_factory=lambda: _env_float("SQUEEZE_REQUEST_DELAY", 0.15))
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SQUEEZE_USER_AGENT",
            "short-squeeze-lab/2.0 (+https://github.com/pezzidouglas/short-squeeze-lab)",
        )
    )
    # SEC requires a contact address in the UA string for its APIs.
    sec_contact: str = field(default_factory=lambda: os.environ.get("SEC_CONTACT_EMAIL", ""))
    enabled_sources: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip()
            for s in os.environ.get("SQUEEZE_SOURCES", "yahoo,stooq,finviz,finra,sec").split(",")
            if s.strip()
        )
    )
    # When no live source answers, fall back to the bundled snapshot so the
    # dashboard and CLI still work offline.
    allow_snapshot_fallback: bool = field(
        default_factory=lambda: _env_bool("SQUEEZE_SNAPSHOT_FALLBACK", True)
    )
    offline: bool = field(default_factory=lambda: _env_bool("SQUEEZE_OFFLINE", False))
    preset: str = field(default_factory=lambda: os.environ.get("SQUEEZE_PRESET", DEFAULT_PRESET))

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_dir, "cache")

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "scan_cache_ttl": self.scan_cache_ttl,
            "http_timeout": self.http_timeout,
            "max_workers": self.max_workers,
            "enabled_sources": list(self.enabled_sources),
            "allow_snapshot_fallback": self.allow_snapshot_fallback,
            "offline": self.offline,
            "preset": self.preset,
        }


SETTINGS = Settings()
