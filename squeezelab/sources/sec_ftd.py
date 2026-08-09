"""
SEC fails-to-deliver source — CNS open fails balances as a borrow-scarcity proxy.

A fail-to-deliver is a trade that did not settle on time at NSCC's Continuous Net
Settlement system. The `QUANTITY (FAILS)` column is the *open fails balance* on a
settlement date, not new fails and not a running total, which is why this module
summarises it with max/mean/persistence and never sums it — summing double-counts
the same open fail on every day it stays open and produces a number that can
exceed the float while describing nothing.

What it is good for: names that are hard to borrow fail more, because the
delivery leg is harder to source, so persistent large fails corroborate a tight
borrow. What it is not: it is not naked short selling, not "synthetic shares",
and not hidden short interest — fails arise overwhelmingly from operational
causes (late settlement instructions, restricted/legended shares, ETF
create/redeem, options assignment timing, corporate-action re-orgs, transfer-agent
lag). Reg SHO also forces close-out, so large balances mechanically resolve.

The other defining property is staleness: SEC publishes roughly a month after a
half-month period closes, so the freshest datum is typically ~40 days old at scan
time. That is structural context, never a timing signal — which is why
`ftd_date` is emitted alongside every number and must be surfaced with it.

The file itself has no float column, so `ftd_pct_float` (the only cross-ticker
comparable reading) is claimed only when a denominator is injected by the caller,
the same way `FinraSource` only claims short interest when it has credentials.

SEC asks automated clients to identify themselves; set `SEC_CONTACT_EMAIL` so the
User-Agent carries a contact address, per their access policy.
"""

from __future__ import annotations

import io
import re
import zipfile
from calendar import monthrange
from datetime import date, timedelta
from typing import Any, Callable, Sequence

from .base import DataSource, DiskCache, SourceError, SourceMeta, SourceRecords, positive, to_float

# SEC has moved this directory at least once, so the templates are tried in
# order and only ever advanced on a 404 — a transport failure is not evidence
# that the file lives somewhere else.
URL_TEMPLATES = (
    "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/cnsfails{yyyymm}{half}.zip",
    "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{yyyymm}{half}.zip",
)

# Last resort when every template 404s: the naming scheme has probably changed
# again, and one GET of the landing page recovers the real hrefs.
INDEX_URLS = (
    "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
    "https://www.sec.gov/data/foiadocsfailsdatahtm",
)

PERIODS_WANTED = 2            # ~21 settlement dates, one calendar month of window
PERIOD_SEARCH_LIMIT = 8       # candidates tried before giving up (4 months back)
PUBLICATION_LAG_FLOOR = 14    # days; SEC has never published faster than this
FAST_WINDOW = 5               # dates in the short window used for the trend
FILE_CACHE_TTL = 60 * 60 * 24 * 180  # published FTD files are immutable

# (year, month, "a"|"b") — "a" covers settlement dates 01-15, "b" 16-end.
Period = tuple[int, int, str]
# One symbol's fails history: (YYYYMMDD, shares, price or None), ascending.
FtdPoint = tuple[str, float, float | None]
FtdTable = dict[str, list[FtdPoint]]

_HTTP_4XX_RE = re.compile(r"HTTP 4\d\d")
_LINK_RE = re.compile(
    r"""href\s*=\s*["']?(?P<url>[^"'>\s]*cnsfails(?P<year>\d{4})(?P<month>\d{2})(?P<half>[ab])\.(?:zip|txt))""",
    re.IGNORECASE,
)


class SecFtdSource(DataSource):
    meta = SourceMeta(
        key="sec_ftd",
        name="SEC fails-to-deliver",
        url="https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
        cadence="semi-monthly",
        provides=(
            "ftd_shares_peak", "ftd_shares_avg", "ftd_persist_days", "ftd_pct_float",
            "ftd_notional", "ftd_trend", "ftd_date", "ftd_series",
        ),
        notes=(
            "CNS open fails balances, ~a month stale by construction and lagging rather than "
            "leading. A weak corroboration of borrow scarcity, not short interest. Symbols are "
            "keyed without CUSIP, so a reused ticker can attribute a dead issuer's fails to a "
            "new one; the two-period window is the only mitigation. Set SEC_CONTACT_EMAIL."
        ),
    )

    def __init__(
        self,
        *args: Any,
        today: date | None = None,
        float_lookup: Callable[[str], float | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._today = today
        # The file carries no float column, so the denominator for the only
        # cross-ticker comparable field is injected rather than fetched here —
        # refetching float would duplicate Yahoo/Finviz at the wrong layer.
        self.float_lookup = float_lookup
        # Published periods never change, so successful parses get a long TTL.
        # Misses deliberately do NOT go here (see `_period_table`).
        self.file_cache = DiskCache(self.settings.cache_dir, FILE_CACHE_TTL)

    @property
    def today(self) -> date:
        return self._today or date.today()

    def available(self) -> bool:
        # SEC returns 403 to clients that do not identify themselves, so a
        # missing contact is a configuration gap, not a provider outage, and
        # must read as "skipped" instead of turning every scan red.
        return super().available() and bool(self.settings.sec_contact)

    def user_agent(self) -> str:
        return f"short-squeeze-lab {self.settings.sec_contact}".strip()

    # -- one period -------------------------------------------------------

    def _cached_table(self, period: Period) -> FtdTable | None:
        cached = self.file_cache.get("sec_ftd", period_stamp(period))
        if not cached:
            return None
        self.health.from_cache = True
        return table_from_cache(cached) or None

    def _download_table(self, period: Period, urls: Sequence[str]) -> FtdTable | None:
        """Fetch and parse one period from an ordered list of URLs, or None."""
        body: bytes | None = None
        for url in urls:
            try:
                body = self.get(url).content
                break
            except SourceError as exc:
                if _is_missing(exc):
                    continue  # wrong path (or unpublished); try the next one
                raise  # a transport failure says nothing about the file existing
        if body is None:
            return None
        table = parse_ftd_file(decode_ftd_bytes(body))
        if table:
            self.file_cache.set("sec_ftd", period_stamp(period), table_to_cache(table))
        return table or None

    def _period_table(self, period: Period) -> FtdTable | None:
        stamp = period_stamp(period)
        cached = self._cached_table(period)
        if cached is not None:
            return cached
        # Misses go in the SHORT cache, unlike FINRA's daily files: a missing
        # daily file (weekend/holiday) never appears, but an FTD period that
        # 404s today will exist in three weeks. Pinning it in the long cache
        # would freeze the source on stale data for half a year.
        if self.cache.get("sec_ftd_miss", stamp):
            return None
        table = self._download_table(period, [t.format(yyyymm=stamp[:6], half=stamp[6:]) for t in URL_TEMPLATES])
        if not table:
            self.cache.set("sec_ftd_miss", stamp, True)
            return None
        return table

    def recent_periods(self, limit: int = PERIODS_WANTED) -> list[tuple[Period, FtdTable]]:
        """Most recent `limit` published periods, newest last (mirrors FINRA)."""
        periods: list[tuple[Period, FtdTable]] = []
        for period in candidate_periods(self.today, PERIOD_SEARCH_LIMIT):
            if len(periods) >= limit:
                break
            table = self._period_table(period)
            if table:
                periods.append((period, table))
        periods.reverse()
        return periods

    def _periods_from_index(self) -> list[tuple[Period, FtdTable]]:
        """Recover from a renamed file scheme by scraping the landing page."""
        links: dict[Period, str] = {}
        for url in INDEX_URLS:
            try:
                links = extract_period_links(self.get(url).text)
            except SourceError:
                continue
            if links:
                break
        cutoff = self.today - timedelta(days=PUBLICATION_LAG_FLOOR)
        periods: list[tuple[Period, FtdTable]] = []
        for period in sorted(links, reverse=True):
            if len(periods) >= PERIODS_WANTED:
                break
            if period_end(period) > cutoff:
                continue
            # The miss cache is keyed to the template URLs, so it says nothing
            # about the href we just discovered; only the long cache applies.
            table = self._cached_table(period) or self._download_table(period, (links[period],))
            if table:
                periods.append((period, table))
        periods.reverse()
        return periods

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        periods = self.recent_periods()
        if not periods:
            periods = self._periods_from_index()
        if not periods:
            raise SourceError(
                f"no fails-to-deliver file published in the last {PERIOD_SEARCH_LIMIT} periods"
            )
        if len(periods) < PERIODS_WANTED:
            self.health.detail = (
                f"only {len(periods)} of {PERIODS_WANTED} periods available; window is shorter than usual"
            )
        return summarise_ftd([table for _, table in periods], tickers, self.float_lookup)


# --------------------------------------------------------------------------
# Period arithmetic (pure functions — unit tested without network)
# --------------------------------------------------------------------------


def period_stamp(period: Period) -> str:
    """(2026, 6, 'b') -> '202606b', the token both the URL and the cache use."""
    year, month, half = period
    return f"{year:04d}{month:02d}{half.lower()}"


def period_end(period: Period) -> date:
    """Last settlement date the period can contain."""
    year, month, half = period
    if half.lower() == "a":
        return date(year, month, 15)
    return date(year, month, monthrange(year, month)[1])


def period_start(period: Period) -> date:
    year, month, half = period
    return date(year, month, 1 if half.lower() == "a" else 16)


def previous_period(period: Period) -> Period:
    year, month, half = period
    if half.lower() == "b":
        return (year, month, "a")
    if month == 1:
        return (year - 1, 12, "b")
    return (year, month - 1, "b")


def candidate_periods(today: date, limit: int = PERIOD_SEARCH_LIMIT) -> list[Period]:
    """Periods worth requesting on `today`, newest first.

    Anything whose last settlement date is inside PUBLICATION_LAG_FLOOR is
    dropped without a request: SEC has never published that fast, so those
    candidates are guaranteed 404s. Worst case SEC gets quicker and we are one
    period staler than we could be — cheap insurance against burning two
    requests on every scan.
    """
    cutoff = today - timedelta(days=PUBLICATION_LAG_FLOOR)
    period: Period = (today.year, today.month, "b")
    out: list[Period] = []
    # The skipped head is bounded (at most three periods can sit inside the lag
    # floor), so a small allowance over `limit` always fills the list.
    for _ in range(limit + 6):
        if len(out) >= limit:
            break
        if period_end(period) <= cutoff:
            out.append(period)
        period = previous_period(period)
    return out


# --------------------------------------------------------------------------
# Container / payload handling (pure)
# --------------------------------------------------------------------------


def select_txt_member(names: Sequence[str]) -> str | None:
    """Pick the data member of an FTD archive; the name never matches the zip."""
    for name in names or []:
        if name.endswith("/"):
            continue
        base = name.rsplit("/", 1)[-1]
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name or base.startswith("._"):
            continue
        if base.lower().endswith(".txt"):
            return name
    return None


def decode_ftd_bytes(body: bytes) -> str:
    """Turn a downloaded FTD payload into text, ZIP or bare .txt.

    Sniffing the payload rather than the extension covers the very old vintages
    that were served unzipped without spending another request to find out.
    Decoding is latin-1 because security descriptions carry legacy high bytes
    and latin-1 cannot fail — a UnicodeDecodeError reaching the caller would be
    a bug, not a data problem.
    """
    raw = body or b""
    if raw[:2] == b"PK":
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
            names = archive.namelist()
            member = select_txt_member(names)
            if member is None:
                # A silent empty result here is indistinguishable from "the
                # ticker is not in the file", so say what was actually in there.
                raise SourceError(f"no .txt member in the FTD archive; saw {names[:10]}")
            raw = archive.read(member)
        except SourceError:
            raise
        except (zipfile.BadZipFile, KeyError, RuntimeError, OSError, ValueError, EOFError) as exc:
            raise SourceError(f"unreadable FTD archive: {type(exc).__name__}: {exc}") from exc
    return raw.decode("latin-1")


def parse_ftd_file(text: str) -> FtdTable:
    """Parse `SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY|DESCRIPTION|PRICE` rows.

    Numbers are anchored from the left and the price from the right, so a stray
    delimiter inside the free-text DESCRIPTION costs the description only. Rows
    with no symbol or no positive balance carry no information and are dropped
    rather than keyed on an empty string or a zero.
    """
    balances: dict[str, dict[str, float]] = {}
    prices: dict[str, dict[str, float | None]] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue  # trailer text, or a truncated final line
        stamp = parts[0].strip()
        # The general guard that also kills the header and any repeated header.
        if len(stamp) != 8 or not stamp.isdigit():
            continue
        symbol = parts[2].strip().upper()
        shares = positive(parts[3])
        if not symbol or shares is None:
            continue
        price = positive(parts[-1]) if len(parts) >= 5 else None
        by_date = balances.setdefault(symbol, {})
        # Same symbol on the same date under two CUSIPs (when-issued lines,
        # class re-orgs, post-split old/new) is one question: how many shares
        # of this ticker were failing that day. Sum them.
        by_date[stamp] = by_date.get(stamp, 0.0) + shares
        seen = prices.setdefault(symbol, {})
        if seen.get(stamp) is None:
            seen[stamp] = price
    return {
        symbol: [(stamp, by_date[stamp], prices[symbol].get(stamp)) for stamp in sorted(by_date)]
        for symbol, by_date in balances.items()
    }


def extract_period_links(html: str) -> dict[Period, str]:
    """Map periods to absolute file URLs found on the FTD landing page."""
    links: dict[Period, str] = {}
    for match in _LINK_RE.finditer(html or ""):
        url = match.group("url")
        if url.startswith("//"):
            url = f"https:{url}"
        elif url.startswith("/"):
            url = f"https://www.sec.gov{url}"
        period = (int(match.group("year")), int(match.group("month")), match.group("half").lower())
        links.setdefault(period, url)
    return links


# --------------------------------------------------------------------------
# Aggregation (pure)
# --------------------------------------------------------------------------


def summarise_ftd(
    periods: Sequence[FtdTable],
    tickers: Sequence[str],
    float_lookup: Callable[[str], float | None] | None = None,
) -> SourceRecords:
    """Aggregate parsed periods into canonical per-ticker fields.

    Balances are summarised, never summed: peak and persistence are what carry
    the signal, and a total across settlement dates would count the same open
    fail once per day it stayed open.
    """
    merged, aliases = _merge_periods(periods)
    records: SourceRecords = {}
    for ticker in {t.upper() for t in tickers}:
        points = merged.get(ticker)
        if points is None:
            # FTD symbols appear unpunctuated where vendors use hyphens or dots
            # (BRKB vs BRK-B). Best-effort, and only after the exact key misses.
            points = merged.get(aliases.get(_squash(ticker), ""))
        if not points:
            continue
        shares = [p[1] for p in points]
        peak = max(shares)
        peak_price = next((p[2] for p in points if p[1] == peak and p[2] is not None), None)
        values: dict[str, Any] = {
            "ftd_shares_peak": round(peak, 2),
            "ftd_shares_avg": round(sum(shares) / len(shares), 2),
            "ftd_persist_days": len(points),
            "ftd_date": _iso(points[-1][0]),
            "ftd_series": [
                {"date": _iso(stamp), "shares": value, "price": price}
                for stamp, value, price in points
            ],
        }
        if peak_price:
            # A size filter that needs no external data: 5M failing shares of a
            # $0.30 stock is noise.
            values["ftd_notional"] = round(peak * peak_price, 2)
        if len(points) > FAST_WINDOW:
            window = sum(shares) / len(shares)
            fast = sum(shares[-FAST_WINDOW:]) / FAST_WINDOW
            if window > 0:
                values["ftd_trend"] = round((fast / window - 1) * 100, 2)
        shares_float = _lookup_float(float_lookup, ticker)
        if shares_float:
            values["ftd_pct_float"] = round(peak / shares_float * 100, 4)
        records[ticker] = values
    return records


def table_to_cache(table: FtdTable) -> dict[str, list[list[Any]]]:
    """Compact encoding for the disk cache — lists, not dicts, per point."""
    return {
        symbol: [[stamp, shares, price] for stamp, shares, price in points]
        for symbol, points in table.items()
    }


def table_from_cache(payload: Any) -> FtdTable:
    """Inverse of `table_to_cache`, tolerant of anything else on disk."""
    if not isinstance(payload, dict):
        return {}
    out: FtdTable = {}
    for symbol, points in payload.items():
        restored: list[FtdPoint] = []
        for row in points or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            shares = to_float(row[1])
            if shares is None:
                continue
            price = to_float(row[2]) if len(row) > 2 else None
            restored.append((str(row[0]), shares, price))
        if restored:
            out[str(symbol).upper()] = sorted(restored)
    return out


def _merge_periods(periods: Sequence[FtdTable]) -> tuple[FtdTable, dict[str, str]]:
    """Fold periods into one ascending series per symbol, plus an alias index."""
    balances: dict[str, dict[str, float]] = {}
    prices: dict[str, dict[str, float | None]] = {}
    for table in periods or []:
        for symbol, points in (table or {}).items():
            by_date = balances.setdefault(symbol, {})
            seen = prices.setdefault(symbol, {})
            for stamp, shares, price in points or []:
                by_date[stamp] = by_date.get(stamp, 0.0) + shares
                if seen.get(stamp) is None:
                    seen[stamp] = price
    merged: FtdTable = {
        symbol: [(stamp, by_date[stamp], prices[symbol].get(stamp)) for stamp in sorted(by_date)]
        for symbol, by_date in balances.items()
    }
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for symbol in merged:
        key = _squash(symbol)
        prior = aliases.get(key)
        if prior is None or symbol == key:
            aliases[key] = symbol
        elif prior != symbol and prior != key:
            ambiguous.add(key)  # two symbols collapse to it; guessing is worse
    for key in ambiguous:
        aliases.pop(key, None)
    return merged, aliases


def _lookup_float(float_lookup: Callable[[str], float | None] | None, ticker: str) -> float | None:
    if float_lookup is None:
        return None
    try:
        return positive(float_lookup(ticker))
    except Exception:  # noqa: BLE001 - a bad denominator drops one field, not the source
        return None


def _squash(symbol: str) -> str:
    return symbol.replace("-", "").replace(".", "")


def _iso(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if len(stamp) == 8 and stamp.isdigit() else stamp


def _is_missing(error: Exception) -> bool:
    """base.get reports a 4xx as `HTTP <code> for <url>`; anything else is transport."""
    return bool(_HTTP_4XX_RE.search(str(error)))
