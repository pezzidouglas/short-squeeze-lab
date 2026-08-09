"""
Borrow source — IBKR's shortable-share counts and indicative borrow fees.

Every other input in this repo scores "borrow pressure" from days-to-cover and
float size: two proxies for a quantity neither of them measures. The price and
availability of stock loan *is* that quantity, and IBKR publishes both, free and
unauthenticated, refreshed intraday. It is the only source here that measures a
short's cost of carry instead of inferring it, and the only one that is not
lagged by a semi-monthly reporting cycle.

It is also narrower than it looks, and the scoring engine should treat it that
way: this is one broker's inventory, the rates are indications rather than
struck loans, and expensive-to-borrow names underperform on average. A high fee
raises variance, not expected return. Use it as pressure, never as a trigger.

Opt-in, deliberately — and as of the first live run, **the default URLs do not
work**. CI reached `ftp3.interactivebrokers.com:443` and got a TCP connect
timeout, not a 404 or a redirect: nothing is listening for HTTPS on that host.
The parser below is exercised entirely against fixtures and is believed
correct; the transport is not. Until someone supplies a URL that actually
serves the file, this source is a well-tested reader with nothing to read.

That is exactly why it defaults off. Set `SQUEEZE_BORROW_ENABLED=1` to try the
defaults anyway, or — better — point `IBKR_SHORTABLE_URL` at a mirror known to
serve `usa.txt` over HTTP(S). Defaulting it on would paint every scan red for
every user; "skipped" costs one environment variable instead. Same reasoning as
`SecEdgarSource.available()` — a missing setting, not a broken provider.

The historical anonymous-FTP location, `ftp://ftp3.interactivebrokers.com/usa.txt`,
is the manual fallback for a human, and is the likely reason the HTTPS attempt
times out: the service is real, but it speaks FTP. It is deliberately absent
from DEFAULT_URLS — `requests` cannot speak FTP, so it would only burn a retry
cycle on the way to a `SourceError`.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from statistics import median
from typing import Any, Sequence

from .base import DataSource, SourceError, SourceMeta, SourceRecords

COUNTRY_FILE = "usa.txt"
# Seconds allowed per candidate URL. Bounds the whole source at roughly
# len(DEFAULT_URLS) * this, so an unreachable endpoint costs a few seconds of
# scan latency rather than over a minute.
PROBE_TIMEOUT = 6.0
# At most two: every miss costs `http_retries + 1` requests and two sleeps
# inside `base.get()`, so a long candidate list buys nothing but latency.
DEFAULT_URLS: tuple[str, ...] = (
    f"https://ftp3.interactivebrokers.com/{COUNTRY_FILE}",
    f"https://ftp2.interactivebrokers.com/{COUNTRY_FILE}",
)

# Header tokens are matched by alias so a rename does not kill the source. The
# column *order* is never assumed; see `parse_header`.
COLUMN_ALIASES: dict[str, str] = {
    "SYM": "symbol",
    "SYMBOL": "symbol",
    "CUR": "currency",
    "CURRENCY": "currency",
    "FEERATE": "fee",
    "FEE": "fee",
    "FEE RATE": "fee",
    "REBATERATE": "rebate",
    "REBATE": "rebate",
    "AVAILABLE": "available",
    "AVAILABLESHARES": "available",
    "QUANTITY": "available",
}

NA_TOKENS = frozenset({"-", "--", "N/A", "NA", "#N/A", "NULL", "NONE"})
# Two characters first, or ">=" leaves a stray "=" behind.
CAP_TOKENS = (">=", ">")

# A units tripwire, generous on purpose: 300%/yr fees are real, 40000 is a
# change of units.
RATE_LIMIT = 1000.0

# `rebate + fee` should be one benchmark short rate; see `implied_benchmark_rate`.
BENCHMARK_BAND = (0.0, 10.0)
BENCHMARK_MIN_ROWS = 3

SNIFF_CHARS = 2000


class BorrowSource(DataSource):
    meta = SourceMeta(
        key="borrow",
        name="IBKR shortable shares",
        url=DEFAULT_URLS[0],
        cadence="realtime",
        provides=(
            "borrow_fee_pct", "shortable_shares", "shortable_shares_basis",
            "borrow_rebate_pct", "borrow_as_of",
        ),
        notes=(
            "One broker's indicative stock-loan book, snapshotted intraday. Opt in with "
            "SQUEEZE_BORROW_ENABLED=1 or IBKR_SHORTABLE_URL; the endpoint is public but unsupported."
        ),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.url_override = os.environ.get("IBKR_SHORTABLE_URL", "").strip()
        self.enabled = _env_flag("SQUEEZE_BORROW_ENABLED") or bool(self.url_override)
        self._parsed: tuple[dict[str, dict[str, Any]], dict[str, Any]] | None = None
        if self.url_override:
            # `meta.url` is the answer to "where did this number come from", so
            # it has to name the URL actually fetched, not the default we skipped.
            self.meta = replace(self.meta, url=self.url_override)

    def urls(self) -> tuple[str, ...]:
        return (self.url_override,) if self.url_override else DEFAULT_URLS

    def available(self) -> bool:
        return super().available() and self.enabled

    # -- transport --------------------------------------------------------

    def _table(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """The whole country file, fetched once and memoised per process.

        One request per scan for ~40k rows, never one request per ticker.
        """
        if self._parsed is not None:
            return self._parsed
        refusals: list[str] = []
        for url in self.urls():
            try:
                # Single attempt, short budget. These are *candidates*, not a
                # known-good endpoint: the whole point of the list is that the
                # first one may not answer, and CI proved it does not — the
                # default 3-attempts-at-12s per candidate spent 75 seconds
                # discovering that ftp3.interactivebrokers.com does not serve
                # HTTPS. The file is ~4MB, so the read budget stays generous;
                # it is the reachability probe that has to be cheap.
                response = self.get(url, timeout=PROBE_TIMEOUT, retries=0)
            except SourceError as exc:
                refusals.append(str(exc))
                continue
            text = decode_body(response)
            if not looks_like_shortable_file(text):
                refusals.append(describe_payload(text))
                continue
            # A layout refusal propagates rather than falling through to the
            # mirror: the mirror carries the same format, and that refusal is
            # the observation this source exists to make.
            self._parsed = parse_shortable_file(text)
            return self._parsed
        raise SourceError(refusals[0] if refusals else "no IBKR shortable URL was reachable")

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        wanted = [t.upper() for t in tickers]
        key = cache_key(wanted)
        cached = self.cache.get("borrow", key)
        if isinstance(cached, dict) and cached:
            self.health.from_cache = True
            return {ticker: dict(values) for ticker, values in cached.items()}

        records, diagnostics = self._table()

        tripwire = layout_tripwire(records)
        if tripwire:
            # Raised, not assigned. `collect()` overwrites `health.detail` with
            # "<n>/<m> tickers covered" on the partial path — the ordinary
            # outcome for this source, since IBKR does not list every name — so
            # a plain `self.health.detail = tripwire` here is discarded before
            # anyone can read it. The `SourceError` path is the only one that
            # reports the message verbatim, and it is also the honest status:
            # if the rate columns no longer mean what the parser thinks, every
            # fee below is suspect, and the module already refuses rather than
            # guesses on the layouts `parse_header` can see.
            raise SourceError(tripwire)

        index = build_symbol_index(records)
        matched: SourceRecords = {t: dict(records[index[t]]) for t in wanted if t in index}

        notes = _diagnose(records, diagnostics)
        if not matched:
            # Partial coverage is this source's steady state — IBKR does not
            # offer every OTC name, warrant or recent IPO — so "nothing matched"
            # has to read differently from an outage.
            notes.append(
                f"parsed {diagnostics['symbols']:,} symbols; none of the {len(wanted)} "
                "requested tickers are in IBKR's US book"
            )
        # `collect()` keeps this detail on the empty and error paths but replaces
        # it with a coverage line on the partial path, which is the common one.
        # Nothing here may be the *only* place a problem is reported — anything
        # that must always be seen is raised instead (see `layout_tripwire`).
        self.health.detail = "; ".join(notes)[:300]

        if matched:
            # The parsed records for this watchlist, not the multi-MB body: the
            # default 1800s TTL is longer than the file's refresh interval on
            # purpose. Borrow fees do not move enough in 15 minutes to change a
            # score band, and hammering a free file from CI is how you get blocked.
            self.cache.set("borrow", key, matched)
        return matched


# --------------------------------------------------------------------------
# Parsing (pure functions — unit tested without a source instance or network)
# --------------------------------------------------------------------------


def parse_shortable_file(text: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse the pipe file into {file symbol: canonical fields} plus diagnostics.

    Raises `SourceError` when the layout is unrecognisable, quoting what actually
    arrived. That refusal is the design: the column order in this file could not
    be verified before writing the parser, so the first real run is meant to
    print the truth into a health line rather than silently read a -21.75%
    rebate as a 26% fee. A positional fallback would hide the one failure worth
    seeing.
    """
    lines = _strip_bom(text or "").splitlines()
    diagnostics: dict[str, Any] = {
        "country": None,
        "as_of": None,
        "columns": [],
        "rows": 0,
        "symbols": 0,
        "rows_skipped": 0,
        "rows_empty": 0,
        "duplicates": 0,
        "rates_dropped": 0,
        "truncated": True,  # until a #EOF says otherwise
    }
    columns: dict[str, int] | None = None
    width = 0
    records: dict[str, dict[str, Any]] = {}
    currencies: dict[str, str] = {}
    availability: dict[str, float | None] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("#BOF"):
            diagnostics["country"], diagnostics["as_of"] = parse_bof_line(line)
            continue
        if upper.startswith("#EOF"):
            diagnostics["truncated"] = False
            continue
        if columns is None and line.startswith("#"):
            # Must be tried before the generic comment skip: the header also
            # starts with '#', and a naive startswith("#") filter eats the one
            # line the parser cannot work without.
            candidate = parse_header(line)
            if candidate is not None:
                columns, width = candidate, len(line.split("|"))
                diagnostics["columns"] = sorted(columns)
                if "fee" not in columns and "available" not in columns:
                    raise SourceError(
                        "IBKR layout has neither a fee nor an availability column; "
                        f"header was: {line[:120]!r}"
                    )
                continue
        if line.startswith("#"):
            continue
        if columns is None:
            raise SourceError(f"unrecognised IBKR layout; first data line was: {line[:120]!r}")

        diagnostics["rows"] += 1
        parts = line.split("|")
        if len(parts) != width:
            # A '|' inside a company name shifts every later column. A
            # wrong-width row is evidence, not something to repair with a
            # maxsplit heuristic.
            diagnostics["rows_skipped"] += 1
            continue
        symbol = parts[columns["symbol"]].strip().upper()
        if not symbol:
            diagnostics["rows_skipped"] += 1
            continue

        fee = _read_rate(parts, columns, "fee", diagnostics)
        rebate = _read_rate(parts, columns, "rebate", diagnostics)
        shares, basis = (None, None)
        if "available" in columns:
            shares, basis = parse_available(parts[columns["available"]])

        if fee is None and rebate is None and shares is None:
            # Nothing is known about this name. An empty dict would be dropped
            # downstream anyway, but it would also look like coverage on the way.
            diagnostics["rows_empty"] += 1
            continue

        values: dict[str, Any] = {}
        if fee is not None:
            values["borrow_fee_pct"] = fee
        if rebate is not None:
            values["borrow_rebate_pct"] = rebate
        if shares is not None:
            values["shortable_shares"] = shares
            values["shortable_shares_basis"] = basis
        if diagnostics["as_of"]:
            values["borrow_as_of"] = diagnostics["as_of"]

        currency = parts[columns["currency"]].strip().upper() if "currency" in columns else ""
        if symbol in records:
            diagnostics["duplicates"] += 1
            if not _prefer_row(currency, shares, currencies.get(symbol, ""), availability.get(symbol)):
                continue
        records[symbol] = values
        currencies[symbol] = currency
        availability[symbol] = shares

    if columns is None:
        raise SourceError(
            f"unrecognised IBKR layout; no header and no data rows in {len(lines)} lines"
        )
    diagnostics["symbols"] = len(records)
    return records, diagnostics


def parse_rate(token: Any) -> float | None:
    """Parse an annualised percentage rate on a 0-100 scale, or None.

    `base.to_float` cannot be used here: it reads "7,50" as 750.0, a silent 100x
    error that still looks exactly like a borrow fee. Rates never carry thousands
    separators, so in this column a lone comma is a decimal point and two commas
    mean we are not looking at a rate at all. The asymmetry with
    `parse_available` is deliberate.
    """
    text = _clean_token(token)
    if text is None:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.count(",") > 1:
        return None
    if "," in text:
        # A lone comma is a European decimal point unless a '.' already holds
        # that role, in which case the comma can only be a group separator.
        text = text.replace(",", "" if "." in text else ".")
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or abs(value) > RATE_LIMIT:
        return None
    return value


def parse_available(token: Any) -> tuple[float | None, str | None]:
    """Parse a shortable-share count into (shares, basis), both None when absent.

    Two of the repo's helpers are traps here. `to_float(">10000000")` is None,
    which would delete availability for every liquid name — the rows a human
    checks first. `positive("0")` is None, and zero available is the single most
    squeeze-relevant reading in the file: no new short can be opened at that
    broker at any price. So this returns the count plus how to read it — "floor"
    when the file censored it as `>N`, because a bare 10000000 would be treated
    downstream as an exact measurement.
    """
    text = _clean_token(token)
    if text is None:
        return None, None
    basis = "exact"
    for prefix in CAP_TOKENS:
        if text.startswith(prefix):
            text, basis = text[len(prefix):].strip(), "floor"
            break
    # In this column a comma *is* a thousands separator — the opposite of `parse_rate`.
    digits = text.replace(",", "")
    if not digits.isdigit():
        return None, None
    return float(digits), basis


def parse_header(line: str) -> dict[str, int] | None:
    """Map canonical column name -> index, or None when this is not a header.

    Header-driven rather than positional because the column order was the one
    thing the design could not verify, and reading REBATERATE where FEERATE was
    meant produces plausible-looking garbage instead of an error.
    """
    columns: dict[str, int] = {}
    for index, token in enumerate(line.split("|")):
        canonical = COLUMN_ALIASES.get(token.lstrip("#").strip().upper())
        if canonical and canonical not in columns:
            columns[canonical] = index
    return columns if "symbol" in columns else None


def parse_bof_line(line: str) -> tuple[str | None, str | None]:
    """Harvest (country, naive ISO timestamp) from `#BOF|usa|2026.08.07|18:15:02`.

    The timestamp is stored exactly as published, with no timezone attached.
    IBKR does not say which zone it is in, and converting an unknown zone is
    worse than not converting one.
    """
    parts = [p.strip() for p in line.split("|")]
    country = parts[1].lower() if len(parts) > 1 and parts[1] else None
    day = parts[2].replace(".", "-").replace("/", "-") if len(parts) > 2 else ""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return country, None
    clock = parts[3] if len(parts) > 3 else ""
    if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", clock):
        return country, f"{day}T{clock}"
    return country, day


def build_symbol_index(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map every safe lookup spelling to the file symbol that owns it.

    Aliases are generated *from* the file symbol by substituting its own
    separators, never by dropping a token. The direction is the safety property:
    `RUM WS` (a warrant) yields RUM.WS / RUM-WS / RUMWS and can never collapse
    onto RUM, which is a different security. An exact symbol always beats an
    alias, and an alias two rows both claim is dropped rather than guessed.
    """
    index = {symbol: symbol for symbol in records}
    claims: dict[str, set[str]] = {}
    for symbol in records:
        for alias in symbol_aliases(symbol):
            if alias in index:
                continue
            claims.setdefault(alias, set()).add(symbol)
    for alias, owners in claims.items():
        if len(owners) == 1:
            index[alias] = next(iter(owners))
    return index


def symbol_aliases(symbol: str) -> set[str]:
    """Alternative spellings of a class or warrant symbol: `BRK B` -> BRK.B, BRK-B, BRKB."""
    parts = [p for p in re.split(r"[ .\-/]+", symbol.strip().upper()) if p]
    if len(parts) < 2:
        return set()
    return {joiner.join(parts) for joiner in ("", ".", "-", " ")}


def implied_benchmark_rate(records: dict[str, dict[str, Any]]) -> float | None:
    """Median of `rebate + fee`, which should be a single benchmark short rate.

    If the two rate columns mean what the design thinks they mean, every row's
    rebate and fee sum to roughly the overnight funding rate. A median far
    outside a plausible band is evidence that a column now carries something
    else — a units change, a sign flip, a renamed field — none of which a type
    check would notice.

    It is blind to one thing: a straight rebate<->fee transposition preserves the
    sum. `parse_header` is what guards that case.
    """
    sums = [
        row["borrow_rebate_pct"] + row["borrow_fee_pct"]
        for row in records.values()
        if row.get("borrow_rebate_pct") is not None and row.get("borrow_fee_pct") is not None
    ]
    if len(sums) < BENCHMARK_MIN_ROWS:
        return None
    return round(median(sums), 2)


def layout_tripwire(records: dict[str, dict[str, Any]]) -> str | None:
    """The refusal message when the rate columns stop meaning what we think.

    Split out of `_diagnose` because this is the one diagnosis that may not be a
    note: notes reach `health.detail`, and `collect()` overwrites that string on
    the partial path. `_fetch` raises this, so it lands on the one path
    `collect()` reports verbatim.

    None when the implied benchmark rate is plausible, or when too few rows
    carry both a rebate and a fee for the median to mean anything.
    """
    benchmark = implied_benchmark_rate(records)
    if benchmark is None or BENCHMARK_BAND[0] <= benchmark <= BENCHMARK_BAND[1]:
        return None
    return f"column layout may have changed: implied benchmark rate {benchmark:.1f}% is implausible"


def looks_like_shortable_file(text: str | None) -> bool:
    """Cheap sniff run before parsing, so an HTML body is reported as HTML."""
    head = _strip_bom(text or "").lstrip()[:SNIFF_CHARS]
    if not head:
        return False
    lowered = head.lower()
    if lowered.startswith("<!doctype") or lowered.startswith("<html") or "<noscript" in lowered:
        return False
    return "|" in head


def describe_payload(text: str | None) -> str:
    """Turn a non-conforming body into a health line someone can act on.

    Same job as `stooq.describe_refusal`. A 200 that is not the file we asked for
    otherwise surfaces as "no records returned", which is indistinguishable from
    an outage and tells the reader nothing about which of the two it was.
    """
    stripped = _strip_bom(text or "")
    body = " ".join(stripped.split())
    if not body:
        return "empty response"
    lowered = body.lower()
    if lowered.startswith("<!doctype") or lowered.startswith("<html") or "<noscript" in lowered:
        return (
            "IBKR served an HTML page instead of the pipe file — usual from cloud/datacenter "
            "IPs, and usually fine from a normal connection"
        )
    if "#BOF" in stripped.upper():
        return f"unrecognised IBKR layout; header line was: {_line_after_bof(stripped)[:120]}"
    return f"unexpected response: {body[:120]}"


def decode_body(response: Any) -> str:
    """Decode the body ourselves — the file's encoding is not guaranteed UTF-8.

    `resp.text` guesses a charset from headers and content sniffing, and an
    accented company name is not worth a `UnicodeDecodeError` escaping `_fetch`.
    """
    raw = getattr(response, "content", None)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if raw is not None:
        return str(raw)
    return str(getattr(response, "text", "") or "")


def cache_key(tickers: Sequence[str]) -> str:
    """A short, stable cache key for one watchlist.

    `DiskCache._path` truncates keys to 120 characters, so the joined-ticker key
    the other sources build collides between any two watchlists sharing a
    120-character prefix — they silently read each other's cache file.
    """
    joined = ",".join(sorted({t.strip().upper() for t in tickers if t and t.strip()}))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"{COUNTRY_FILE.split('.', 1)[0]}-{digest}"


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _diagnose(records: dict[str, dict[str, Any]], diagnostics: dict[str, Any]) -> list[str]:
    """Health notes about how the parse went, most alarming first.

    Everything here is survivable — a note is advisory, and `collect()` may drop
    the lot on the partial path. The implausible-benchmark case is deliberately
    absent: it is a `SourceError` raised by `_fetch` via `layout_tripwire`,
    because it is the one finding that must never be advisory.
    """
    notes: list[str] = []
    benchmark = implied_benchmark_rate(records)
    plausible = benchmark is not None and BENCHMARK_BAND[0] <= benchmark <= BENCHMARK_BAND[1]
    if diagnostics.get("truncated"):
        notes.append("download looks truncated (no #EOF); tail-of-alphabet symbols may be missing")
    if diagnostics.get("rows_skipped"):
        notes.append(f"{diagnostics['rows_skipped']} malformed rows skipped")
    if diagnostics.get("rates_dropped"):
        notes.append(f"{diagnostics['rates_dropped']} unusable rate values dropped")
    if plausible:
        # Positive evidence that the rate columns were read correctly.
        notes.append(f"implied benchmark rate {benchmark:.2f}%")
    return notes


def _read_rate(
    parts: Sequence[str],
    columns: dict[str, int],
    name: str,
    diagnostics: dict[str, Any],
) -> float | None:
    """Read one rate cell, counting values that were present but unusable."""
    index = columns.get(name)
    if index is None:
        return None
    raw = parts[index]
    value = parse_rate(raw)
    if value is None and _clean_token(raw) is not None:
        diagnostics["rates_dropped"] += 1
    return value


def _prefer_row(
    new_currency: str,
    new_shares: float | None,
    old_currency: str,
    old_shares: float | None,
) -> bool:
    """Deterministic winner for duplicate symbol rows (multiple conids or venues).

    USD first, then the larger availability, then first-seen. Spelled out so dict
    ordering never decides which listing a scan reports.
    """
    if (new_currency == "USD") != (old_currency == "USD"):
        return new_currency == "USD"
    # Explicit None checks, not `or`: zero available is a real reading and must
    # not be demoted to "missing" the way `x or -1` would demote it.
    left = -1.0 if new_shares is None else new_shares
    right = -1.0 if old_shares is None else old_shares
    return left > right


def _clean_token(token: Any) -> str | None:
    """Strip a cell down to something worth parsing, or None if it says nothing."""
    if token is None:
        return None
    text = str(token).strip()
    if not text or text.upper() in NA_TOKENS:
        return None
    return text


def _line_after_bof(text: str) -> str:
    """The line where a header should have been — quoted back at the operator."""
    seen_bof = False
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if line.upper().startswith("#BOF"):
            seen_bof = True
            continue
        if seen_bof:
            return line
    return "(nothing followed #BOF)"


def _strip_bom(text: str) -> str:
    # A BOM is not whitespace to `str.split`, so it survives every other cleanup
    # and turns the first line into something that matches nothing.
    return text.lstrip("\ufeff")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
