"""
FINRA source — official off-exchange short sale volume, and (optionally) the
consolidated short interest file.

This is the source that adds something genuinely new. Yahoo, Finviz and friends
all recycle the same semi-monthly exchange short interest number, which can be
two weeks stale by the time you read it. FINRA's RegSHO daily files are
published every trading day and tell you what share of off-exchange volume was
sold short *yesterday* — a same-week read on whether shorts are still pressing.

Two products are used:

* Daily short sale volume — https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
  Pipe-delimited, free, no credentials. Immutable once published, so it is
  cached aggressively.
* Consolidated short interest — FINRA's data API, which requires OAuth
  credentials. Enabled only when FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET
  are set; otherwise the source simply does not claim those fields.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Sequence

from .base import DataSource, DiskCache, SourceError, SourceMeta, SourceRecords, positive, to_float

DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
SHORT_INTEREST_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"

LOOKBACK_DAYS = 20          # trading days of short volume to aggregate
FAST_WINDOW = 5             # short window used for the trend
CALENDAR_SEARCH_LIMIT = 40  # calendar days to walk back looking for files
DAILY_CACHE_TTL = 60 * 60 * 24 * 30  # published files never change


class FinraSource(DataSource):
    meta = SourceMeta(
        key="finra",
        name="FINRA RegSHO",
        url="https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data",
        cadence="daily",
        provides=(
            "off_exchange_short_pct", "off_exchange_trend", "off_exchange_series",
            "shares_short", "short_pct_float", "short_interest_date",
        ),
        notes="Daily off-exchange short volume, free and unlagged. Consolidated short interest needs API credentials.",
    )

    def __init__(self, *args: Any, today: date | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._today = today
        # Published daily files are immutable, so they get their own long TTL.
        self.daily_cache = DiskCache(self.settings.cache_dir, DAILY_CACHE_TTL)
        self.client_id = os.environ.get("FINRA_API_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("FINRA_API_CLIENT_SECRET", "").strip()

    @property
    def today(self) -> date:
        return self._today or date.today()

    @property
    def has_credentials(self) -> bool:
        return bool(self.client_id and self.client_secret)

    # -- daily short volume ----------------------------------------------

    def _daily_map(self, day: date) -> dict[str, dict[str, float]] | None:
        """Parsed {symbol: {short, exempt, total}} for one session, or None."""
        stamp = day.strftime("%Y%m%d")
        cached = self.daily_cache.get("finra_daily", stamp)
        if cached is not None:
            self.health.from_cache = True
            return cached or None
        try:
            text = self.get_text(DAILY_URL.format(yyyymmdd=stamp))
        except SourceError:
            # Weekend, holiday, or not published yet — remember the miss cheaply.
            self.daily_cache.set("finra_daily", stamp, {})
            return None
        parsed = parse_daily_file(text)
        self.daily_cache.set("finra_daily", stamp, parsed)
        return parsed or None

    def recent_sessions(self, limit: int = LOOKBACK_DAYS) -> list[tuple[date, dict[str, dict[str, float]]]]:
        """Most recent `limit` published sessions, newest last."""
        sessions: list[tuple[date, dict[str, dict[str, float]]]] = []
        day = self.today
        for _ in range(CALENDAR_SEARCH_LIMIT):
            if len(sessions) >= limit:
                break
            day -= timedelta(days=1)
            if day.weekday() >= 5:
                continue
            data = self._daily_map(day)
            if data:
                sessions.append((day, data))
        sessions.reverse()
        return sessions

    # -- consolidated short interest (credentialed) -----------------------

    def _access_token(self) -> str | None:
        if not self.has_credentials:
            return None
        cached = self.cache.get("finra_token", "token")
        if isinstance(cached, str) and cached:
            return cached
        try:
            resp = self.session.post(
                TOKEN_URL,
                params={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=self.settings.http_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"FINRA token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise SourceError(f"FINRA token request returned HTTP {resp.status_code}")
        token = (resp.json() or {}).get("access_token")
        if token:
            self.cache.set("finra_token", "token", token)
        return token

    def _short_interest(self, tickers: Sequence[str]) -> SourceRecords:
        token = self._access_token()
        if not token:
            return {}
        payload = {
            "limit": 5000,
            "compareFilters": [
                {"fieldName": "symbolCode", "fieldValue": t, "compareType": "EQUAL"} for t in tickers
            ],
        }
        try:
            resp = self.session.post(
                SHORT_INTEREST_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=self.settings.http_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"FINRA short interest request failed: {exc}") from exc
        if resp.status_code != 200:
            raise SourceError(f"FINRA short interest returned HTTP {resp.status_code}")
        return parse_short_interest(resp.json())

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        sessions = self.recent_sessions()
        records: SourceRecords = {}
        if sessions:
            records = summarise_short_volume(sessions, tickers)
        if self.has_credentials:
            try:
                for ticker, values in self._short_interest(tickers).items():
                    records.setdefault(ticker, {}).update(values)
            except SourceError as exc:
                self.health.detail = f"short interest unavailable: {exc}"[:200]
        if not records and not sessions:
            raise SourceError("no RegSHO daily files were reachable")
        return records


# --------------------------------------------------------------------------
# Parsing / aggregation (pure functions — unit tested without network)
# --------------------------------------------------------------------------


def parse_daily_file(text: str) -> dict[str, dict[str, float]]:
    """Parse a pipe-delimited CNMSshvol file into {symbol: {short, exempt, total}}."""
    out: dict[str, dict[str, float]] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("Date|") or line.lower().startswith("record count"):
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        symbol = parts[1].strip().upper()
        short = to_float(parts[2])
        exempt = to_float(parts[3])
        total = to_float(parts[4])
        if not symbol or total is None or total <= 0 or short is None:
            continue
        out[symbol] = {"short": short, "exempt": exempt or 0.0, "total": total}
    return out


def summarise_short_volume(
    sessions: Sequence[tuple[date, dict[str, dict[str, float]]]],
    tickers: Sequence[str],
) -> SourceRecords:
    """Aggregate per-session short volume into canonical per-ticker fields."""
    records: SourceRecords = {}
    for ticker in {t.upper() for t in tickers}:
        series: list[dict[str, Any]] = []
        for day, table in sessions:
            row = table.get(ticker)
            if not row or not row.get("total"):
                continue
            series.append(
                {
                    "date": day.isoformat() if isinstance(day, date) else str(day),
                    "ratio": round(row["short"] / row["total"] * 100, 2),
                    "short": row["short"],
                    "total": row["total"],
                }
            )
        if not series:
            continue
        overall = _ratio(series)
        fast = _ratio(series[-FAST_WINDOW:])
        values: dict[str, Any] = {
            "off_exchange_short_pct": overall,
            "off_exchange_series": series,
        }
        if fast is not None and overall is not None and len(series) > FAST_WINDOW:
            values["off_exchange_trend"] = round(fast - overall, 2)
        records[ticker] = values
    return records


def _ratio(series: Sequence[dict[str, Any]]) -> float | None:
    total = sum(float(p["total"]) for p in series)
    if total <= 0:
        return None
    short = sum(float(p["short"]) for p in series)
    return round(short / total * 100, 2)


def parse_short_interest(payload: Any) -> SourceRecords:
    """Map FINRA's consolidated short interest rows to canonical fields."""
    rows = payload if isinstance(payload, list) else (payload or {}).get("data") or []
    records: SourceRecords = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbolCode") or row.get("issueSymbolIdentifier") or "").strip().upper()
        if not symbol:
            continue
        settlement = row.get("settlementDate") or row.get("lastUpdateDate")
        current = positive(row.get("currentShortPositionQuantity"))
        previous = positive(row.get("previousShortPositionQuantity"))
        days_to_cover = to_float(row.get("daysToCoverQuantity"))
        existing = records.get(symbol, {})
        # Keep the most recent settlement per symbol.
        if existing and str(existing.get("short_interest_date") or "") >= str(settlement or ""):
            continue
        values: dict[str, Any] = {}
        if current:
            values["shares_short"] = current
        if previous:
            values["shares_short_prior"] = previous
        if days_to_cover:
            values["short_ratio"] = days_to_cover
        if settlement:
            values["short_interest_date"] = str(settlement)[:10]
        if values:
            records[symbol] = values
    return records
