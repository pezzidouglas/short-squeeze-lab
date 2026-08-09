"""
Stooq source — free end-of-day OHLCV, no key, no rate limit worth worrying about.

Stooq exists here as an *independent* price feed: it is the cross-check that
catches a stale or wrong Yahoo quote, and it supplies the 52-week range and
volume averages straight from the bars rather than from a vendor's summary
field. It is also the price feed the backtester falls back to when `yfinance`
is not installed.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Sequence

import pandas as pd

from .base import DataSource, SourceError, SourceMeta, SourceRecords

CSV_URL = "https://stooq.com/q/d/l/"


class StooqSource(DataSource):
    meta = SourceMeta(
        key="stooq",
        name="Stooq EOD",
        url="https://stooq.com",
        cadence="daily",
        provides=(
            "current_price", "previous_close", "avg_volume_10d", "avg_volume_3m",
            "fifty_two_week_high", "fifty_two_week_low",
        ),
        notes="Independent end-of-day bars. Used to cross-check Yahoo prices and to back the backtester.",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._price_cache: dict[str, pd.DataFrame] = {}

    # -- transport --------------------------------------------------------

    @staticmethod
    def symbol(ticker: str) -> str:
        return f"{ticker.lower().replace('.', '-')}.us"

    def _bars(self, ticker: str) -> pd.DataFrame | None:
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        cached = self.cache.get("stooq", ticker)
        text: str | None = cached if isinstance(cached, str) else None
        if text is None:
            text = self.get_text(CSV_URL, params={"s": self.symbol(ticker), "i": "d"})
            if not text or not text.lstrip().lower().startswith("date"):
                # Stooq answers 200 with a plain-text message when it refuses —
                # most often a daily hit limit on shared/cloud IPs. Record what
                # it actually said, or the health line just reads "no records".
                self.health.detail = describe_refusal(text)
                return None
            self.cache.set("stooq", ticker, text)
        else:
            self.health.from_cache = True

        try:
            frame = pd.read_csv(StringIO(text))
        except (ValueError, pd.errors.ParserError):
            return None
        if frame.empty or "Close" not in frame.columns:
            return None
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        if frame.empty:
            return None
        frame["ticker"] = ticker
        self._price_cache[ticker] = frame
        return frame

    # -- contract ---------------------------------------------------------

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        records: SourceRecords = {}
        failures = 0
        for ticker in tickers:
            try:
                bars = self._bars(ticker)
            except SourceError:
                failures += 1
                continue
            if bars is None or bars.empty:
                continue
            records[ticker] = summarise_bars(bars)
        if not records and failures:
            raise SourceError(f"all {failures} Stooq requests failed")
        return records

    def fetch_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame | None:
        try:
            bars = self._bars(ticker)
        except SourceError:
            return None
        if bars is None or bars.empty:
            return None
        days = _period_to_days(period)
        if days:
            cutoff = bars["Date"].max() - pd.Timedelta(days=days)
            bars = bars[bars["Date"] >= cutoff]
        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume", "ticker") if c in bars.columns]
        return bars[keep].reset_index(drop=True)


def describe_refusal(text: str | None) -> str:
    """Turn a non-CSV Stooq response into a health message worth reading."""
    body = " ".join((text or "").split())[:120]
    if not body:
        return "empty response"
    if "exceeded" in body.lower() and "limit" in body.lower():
        return f"rate limited by Stooq: {body}"
    return f"unexpected response: {body}"


def summarise_bars(bars: pd.DataFrame) -> dict[str, Any]:
    """Derive the quote-shaped fields a scan needs from raw daily bars."""
    close = bars["Close"]
    year = bars.tail(252)
    values: dict[str, Any] = {
        "current_price": float(close.iloc[-1]),
        "fifty_two_week_high": float(year["High"].max()) if "High" in year else float(year["Close"].max()),
        "fifty_two_week_low": float(year["Low"].min()) if "Low" in year else float(year["Close"].min()),
    }
    if len(close) > 1:
        values["previous_close"] = float(close.iloc[-2])
    if "Volume" in bars.columns:
        volume = bars["Volume"].dropna()
        if len(volume) >= 10:
            values["avg_volume_10d"] = float(volume.tail(10).mean())
        if len(volume) >= 20:
            values["avg_volume_3m"] = float(volume.tail(63).mean())
    return values


def _period_to_days(period: str) -> int | None:
    period = (period or "").strip().lower()
    if period in {"max", "", "all"}:
        return None
    units = {"d": 1, "mo": 30, "m": 30, "y": 365}
    for suffix in ("mo", "y", "d", "m"):
        if period.endswith(suffix):
            try:
                return int(float(period[: -len(suffix)]) * units[suffix])
            except ValueError:
                return None
    return None
