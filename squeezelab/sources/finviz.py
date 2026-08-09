"""
Finviz source — the ownership screener export (view 131).

Finviz refreshes short interest faster than Yahoo and is the only free source
here that gives float short and days-to-cover in one request for a whole
watchlist. The export endpoint expects an `auth` token on modern accounts; set
`FINVIZ_AUTH_TOKEN` to use it. Without a token Finviz will usually refuse, and
the source reports itself unavailable rather than poisoning the scan.
"""

from __future__ import annotations

import os
from io import StringIO
from typing import Any, Sequence

import pandas as pd

from .base import DataSource, SourceError, SourceMeta, SourceRecords, chunked, positive, to_float

EXPORT_URL = "https://finviz.com/export.ashx"
OWNERSHIP_VIEW = "131"
BATCH_SIZE = 40

COLUMN_MAP = {
    "Ticker": "ticker",
    "Company": "name",
    "Sector": "sector",
    "Industry": "industry",
    "Market Cap": "market_cap",
    "Outstanding": "shares_outstanding",
    "Float": "float",
    "Float Short": "short_pct_float",
    "Short Float": "short_pct_float",
    "Short Float / Ratio": "short_pct_float",
    "Short Ratio": "short_ratio",
    "Avg Volume": "avg_volume_3m",
    "Price": "current_price",
}


class FinvizSource(DataSource):
    meta = SourceMeta(
        key="finviz",
        name="Finviz screener",
        url="https://finviz.com/screener.ashx",
        cadence="daily",
        provides=(
            "name", "sector", "industry", "market_cap", "shares_outstanding", "float",
            "short_pct_float", "short_ratio", "avg_volume_3m", "current_price",
        ),
        notes="Ownership export (v=131). Needs FINVIZ_AUTH_TOKEN; refreshes short interest faster than Yahoo.",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.auth_token = os.environ.get("FINVIZ_AUTH_TOKEN", "").strip()

    def available(self) -> bool:
        return super().available() and bool(self.auth_token)

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        records: SourceRecords = {}
        failures: list[str] = []
        for batch in chunked(list(tickers), BATCH_SIZE):
            key = ",".join(batch)
            cached = self.cache.get("finviz", key)
            text: str | None = cached if isinstance(cached, str) else None
            if text is None:
                try:
                    text = self.get_text(
                        EXPORT_URL,
                        params={"v": OWNERSHIP_VIEW, "t": key, "auth": self.auth_token},
                    )
                except SourceError as exc:
                    failures.append(str(exc))
                    continue
                if not text or "Ticker" not in text.split("\n", 1)[0]:
                    failures.append("unexpected export payload (auth token rejected?)")
                    continue
                self.cache.set("finviz", key, text)
            else:
                self.health.from_cache = True
            records.update(parse_export(text))
        if not records and failures:
            raise SourceError(failures[0])
        return records


def parse_export(text: str) -> SourceRecords:
    """Turn a Finviz CSV export into canonical records."""
    try:
        frame = pd.read_csv(StringIO(text))
    except (ValueError, pd.errors.ParserError) as exc:
        raise SourceError(f"could not parse Finviz export: {exc}") from exc
    if "Ticker" not in frame.columns:
        raise SourceError("Finviz export is missing a Ticker column")

    records: SourceRecords = {}
    for _, row in frame.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        if not ticker:
            continue
        values: dict[str, Any] = {}
        for column, canonical in COLUMN_MAP.items():
            if column not in frame.columns or canonical in values:
                continue
            raw = row[column]
            if canonical in {"name", "sector", "industry"}:
                text_value = str(raw).strip()
                if text_value and text_value.lower() != "nan":
                    values[canonical] = text_value
            elif canonical == "short_pct_float":
                # Newer exports ship "12.34% / 3.2" in one column.
                parsed = _short_float_and_ratio(raw)
                values.update(parsed)
            elif canonical in {"short_ratio"}:
                parsed_ratio = to_float(raw)
                if parsed_ratio is not None:
                    values.setdefault(canonical, parsed_ratio)
            else:
                parsed_value = positive(raw)
                if parsed_value is not None:
                    values[canonical] = parsed_value
        # Finviz gives float short as a percentage; derive the share count.
        if values.get("short_pct_float") and values.get("float"):
            values["shares_short"] = values["float"] * values["short_pct_float"] / 100
        if values:
            records[ticker] = values
    return records


def _short_float_and_ratio(raw: Any) -> dict[str, float]:
    """Handle both the plain `12.34%` column and the combined `12.34% / 3.2`."""
    out: dict[str, float] = {}
    text = str(raw)
    parts = [p.strip() for p in text.split("/")]
    pct = to_float(parts[0])
    if pct is not None:
        out["short_pct_float"] = pct
    if len(parts) > 1:
        ratio = to_float(parts[1])
        if ratio is not None:
            out["short_ratio"] = ratio
    return out
