"""
SEC EDGAR source — the authoritative share count.

Float and shares-outstanding are the denominators under every squeeze metric, and
vendor copies of them drift after buybacks, ATM offerings and reverse splits —
exactly the corporate actions that show up in heavily shorted names. EDGAR's
company-concept API carries the number straight off the cover page of the latest
10-Q/10-K, so it is the tie-breaker when Yahoo and Finviz disagree.

SEC asks automated clients to identify themselves; set `SEC_CONTACT_EMAIL` so the
User-Agent carries a contact address, per their access policy.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import DataSource, SourceError, SourceMeta, SourceRecords, positive

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json"
)


class SecEdgarSource(DataSource):
    meta = SourceMeta(
        key="sec",
        name="SEC EDGAR",
        url="https://www.sec.gov/edgar",
        cadence="quarterly",
        provides=("cik", "name", "shares_outstanding"),
        notes="Filed share count from the latest 10-Q/10-K cover page. Set SEC_CONTACT_EMAIL.",
    )

    def user_agent(self) -> str:
        contact = self.settings.sec_contact
        return f"{self.settings.user_agent} {contact}".strip()

    # -- ticker -> CIK ----------------------------------------------------

    def _cik_map(self) -> dict[str, dict[str, str]]:
        cached = self.cache.get("sec", "company_tickers")
        if isinstance(cached, dict) and cached:
            self.health.from_cache = True
            return cached
        payload = self.get_json(TICKERS_URL)
        mapping = parse_cik_map(payload)
        if mapping:
            self.cache.set("sec", "company_tickers", mapping)
        return mapping

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        mapping = self._cik_map()
        if not mapping:
            raise SourceError("could not load the SEC ticker/CIK map")

        records: SourceRecords = {}
        for ticker in tickers:
            entry = mapping.get(ticker.upper())
            if not entry:
                continue
            values: dict[str, Any] = {"cik": entry["cik"], "name": entry.get("name")}
            cached = self.cache.get("sec_shares", ticker)
            shares = positive(cached) if cached is not None else None
            if shares is None:
                try:
                    payload = self.get_json(CONCEPT_URL.format(cik=entry["cik"]))
                except SourceError:
                    payload = None
                shares = latest_shares_outstanding(payload) if payload else None
                if shares:
                    self.cache.set("sec_shares", ticker, shares)
            else:
                self.health.from_cache = True
            if shares:
                values["shares_outstanding"] = shares
            records[ticker.upper()] = values
        return records


# --------------------------------------------------------------------------
# Parsing (pure — unit tested without network)
# --------------------------------------------------------------------------


def parse_cik_map(payload: Any) -> dict[str, dict[str, str]]:
    """company_tickers.json is a dict of index -> {cik_str, ticker, title}."""
    rows = payload.values() if isinstance(payload, dict) else payload
    mapping: dict[str, dict[str, str]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik = row.get("cik_str") or row.get("cik")
        if not ticker or cik is None:
            continue
        mapping[ticker] = {"cik": str(cik).zfill(10), "name": str(row.get("title") or "").strip()}
    return mapping


def latest_shares_outstanding(payload: Any) -> float | None:
    """Pick the most recently filed EntityCommonStockSharesOutstanding value."""
    units = ((payload or {}).get("units") or {}).get("shares") or []
    best_date = ""
    best_value: float | None = None
    for entry in units:
        if not isinstance(entry, dict):
            continue
        value = positive(entry.get("val"))
        stamp = str(entry.get("end") or entry.get("filed") or "")
        if value is None or not stamp:
            continue
        if stamp >= best_date:
            best_date, best_value = stamp, value
    return best_value
