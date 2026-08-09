"""
Base plumbing shared by every data source: the source contract, a small
disk-backed HTTP cache, retrying requests, and per-source health reporting.

A source never raises into the caller. It reports what it managed to fetch and
records why the rest is missing, so a dead provider degrades the scan instead of
killing it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import SETTINGS, Settings

try:  # pragma: no cover - exercised implicitly
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Records & health
# --------------------------------------------------------------------------

# A source returns {ticker: {canonical_field: value}}. Values must already be
# in canonical units (percentages as 0-100, share counts as raw shares).
SourceRecords = dict[str, dict[str, Any]]


@dataclass
class SourceHealth:
    """What happened the last time a source ran."""

    key: str
    name: str
    status: str = "idle"  # ok | partial | empty | error | disabled | skipped
    detail: str = ""
    tickers_requested: int = 0
    tickers_returned: int = 0
    fields_returned: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    last_run: str | None = None

    @property
    def coverage(self) -> float:
        if not self.tickers_requested:
            return 0.0
        return round(self.tickers_returned / self.tickers_requested, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "tickers_requested": self.tickers_requested,
            "tickers_returned": self.tickers_returned,
            "fields_returned": self.fields_returned,
            "coverage": self.coverage,
            "latency_ms": round(self.latency_ms, 1),
            "from_cache": self.from_cache,
            "last_run": self.last_run,
        }


@dataclass
class SourceMeta:
    key: str
    name: str
    url: str
    cadence: str  # "realtime" | "daily" | "semi-monthly" | "quarterly" | "static"
    provides: tuple[str, ...]
    notes: str = ""
    requires_network: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "url": self.url,
            "cadence": self.cadence,
            "provides": list(self.provides),
            "notes": self.notes,
            "requires_network": self.requires_network,
        }


class SourceError(Exception):
    """Raised inside a source; converted into health state by `collect`."""


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


class DiskCache:
    """Tiny JSON cache so repeated scans don't hammer free endpoints."""

    def __init__(self, root: str, ttl_seconds: int) -> None:
        self.root = root
        self.ttl = ttl_seconds

    def _path(self, namespace: str, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:120]
        return os.path.join(self.root, namespace, f"{safe}.json")

    def get(self, namespace: str, key: str) -> Any | None:
        if self.ttl <= 0:
            return None
        path = self._path(namespace, key)
        try:
            age = time.time() - os.path.getmtime(path)
            if age > self.ttl:
                return None
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        if self.ttl <= 0:
            return
        path = self._path(namespace, key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(value, fh, default=str)
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError):
            pass  # a cache miss is never fatal


# --------------------------------------------------------------------------
# Source contract
# --------------------------------------------------------------------------


class DataSource:
    """Base class. Subclasses implement `_fetch` and optionally `fetch_prices`."""

    meta: SourceMeta

    def __init__(self, settings: Settings | None = None, session: Any = None) -> None:
        self.settings = settings or SETTINGS
        self.cache = DiskCache(self.settings.cache_dir, self.settings.cache_ttl_seconds)
        self.health = SourceHealth(key=self.meta.key, name=self.meta.name)
        self._session = session

    # -- networking -------------------------------------------------------

    @property
    def session(self):
        if self._session is None:
            if requests is None:  # pragma: no cover
                raise SourceError("the `requests` package is not installed")
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent()})
        return self._session

    def user_agent(self) -> str:
        return self.settings.user_agent

    def get(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        """GET with retries and exponential backoff. Returns a Response."""
        if self.settings.offline:
            raise SourceError("offline mode is enabled")
        last_error: Exception | None = None
        for attempt in range(self.settings.http_retries + 1):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.settings.http_timeout,
                )
                if resp.status_code == 200:
                    return resp
                # 4xx other than rate limiting will not improve on retry.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise SourceError(f"HTTP {resp.status_code} for {url}")
                last_error = SourceError(f"HTTP {resp.status_code} for {url}")
            except SourceError:
                raise
            except Exception as exc:  # network errors, TLS, timeouts
                last_error = exc
            if attempt < self.settings.http_retries:
                time.sleep(min(2 ** attempt, 8) * 0.5)
        raise SourceError(str(last_error or "request failed"))

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def get_text(self, url: str, **kwargs: Any) -> str:
        return self.get(url, **kwargs).text

    # -- contract ---------------------------------------------------------

    def available(self) -> bool:
        """Cheap pre-flight check; skip the source when it cannot possibly work."""
        if self.meta.requires_network and self.settings.offline:
            return False
        if self.meta.requires_network and requests is None:  # pragma: no cover
            return False
        return True

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        raise NotImplementedError

    def fetch_prices(self, ticker: str, period: str = "1y") -> Any | None:
        """Optional. Return an OHLCV DataFrame with a `Date` column, or None."""
        return None

    # -- driver -----------------------------------------------------------

    def collect(self, tickers: Sequence[str]) -> SourceRecords:
        """Run the source, capturing timing, coverage and failures as health."""
        tickers = [t.upper() for t in tickers]
        self.health = SourceHealth(key=self.meta.key, name=self.meta.name)
        self.health.tickers_requested = len(tickers)
        self.health.last_run = _now_iso()

        if not self.available():
            self.health.status = "skipped"
            self.health.detail = "source unavailable in this environment"
            return {}

        started = time.perf_counter()
        try:
            records = self._fetch(tickers) or {}
        except SourceError as exc:
            self.health.status = "error"
            self.health.detail = str(exc)[:300]
            self.health.latency_ms = (time.perf_counter() - started) * 1000
            return {}
        except Exception as exc:  # noqa: BLE001 - a broken source must not kill the scan
            self.health.status = "error"
            self.health.detail = f"{type(exc).__name__}: {exc}"[:300]
            self.health.latency_ms = (time.perf_counter() - started) * 1000
            return {}

        cleaned = {
            ticker: clean_record(values)
            for ticker, values in records.items()
            if values
        }
        cleaned = {t: v for t, v in cleaned.items() if v}

        self.health.latency_ms = (time.perf_counter() - started) * 1000
        self.health.tickers_returned = len(cleaned)
        self.health.fields_returned = sum(len(v) for v in cleaned.values())
        if not cleaned:
            self.health.status = "empty"
            self.health.detail = self.health.detail or "no records returned"
        elif len(cleaned) < len(tickers):
            self.health.status = "partial"
            self.health.detail = f"{len(cleaned)}/{len(tickers)} tickers covered"
        else:
            self.health.status = "ok"
        return cleaned


# --------------------------------------------------------------------------
# Helpers used by concrete sources
# --------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_float(value: Any) -> float | None:
    """Parse numbers out of the many shapes free providers hand back."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return None if result != result else result  # drop NaN
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text in {"-", "--", "N/A", "NA", "null", "None"}:
        return None
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    elif text[-1] in "KMBT":
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1]]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def positive(value: Any) -> float | None:
    """A value that is only meaningful when strictly positive."""
    parsed = to_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def clean_record(values: dict[str, Any]) -> dict[str, Any]:
    """Strip empty values so a source never claims a field it does not have."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out


def chunked(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
