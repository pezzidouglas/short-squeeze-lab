"""
Yahoo Finance source.

Uses `yfinance` when it is installed (it handles Yahoo's cookie/crumb dance) and
falls back to the public quoteSummary endpoint otherwise. Yahoo is the broadest
single source we have: fundamentals, float, exchange short interest and prices.
Its short interest is a stale copy of the exchange semi-monthly file, which is
why FINRA outranks it in `FIELD_PRIORITY`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from .base import DataSource, SourceError, SourceMeta, SourceRecords, positive, to_float

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

QUOTE_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
MODULES = "defaultKeyStatistics,summaryDetail,price,assetProfile"


class YahooSource(DataSource):
    meta = SourceMeta(
        key="yahoo",
        name="Yahoo Finance",
        url="https://finance.yahoo.com",
        cadence="realtime",
        provides=(
            "name", "sector", "industry", "exchange", "current_price", "previous_close",
            "market_cap", "float", "shares_outstanding", "short_pct_float", "short_ratio",
            "shares_short", "shares_short_prior", "short_interest_date", "avg_volume_10d",
            "avg_volume_3m", "fifty_two_week_high", "fifty_two_week_low", "beta",
        ),
        notes="Free, broad coverage. Short interest lags the exchange file by up to two weeks.",
    )

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        records: SourceRecords = {}
        errors = 0
        for ticker in tickers:
            cached = self.cache.get("yahoo", ticker)
            if cached is not None:
                records[ticker] = cached
                self.health.from_cache = True
                continue
            try:
                info = self._info(ticker)
            except SourceError:
                errors += 1
                continue
            if not info:
                continue
            values = self._map(info)
            if values:
                records[ticker] = values
                self.cache.set("yahoo", ticker, values)
        if not records and errors:
            raise SourceError(f"all {errors} Yahoo requests failed")
        return records

    # -- transport --------------------------------------------------------

    def _info(self, ticker: str) -> dict[str, Any]:
        if yf is not None:
            try:
                return dict(yf.Ticker(ticker).info or {})
            except Exception as exc:  # noqa: BLE001 - yfinance raises many types
                raise SourceError(f"yfinance failed for {ticker}: {exc}") from exc
        payload = self.get_json(
            QUOTE_SUMMARY.format(ticker=ticker), params={"modules": MODULES}
        )
        result = ((payload or {}).get("quoteSummary") or {}).get("result") or []
        if not result:
            return {}
        merged: dict[str, Any] = {}
        for module in result[0].values():
            if isinstance(module, dict):
                merged.update(module)
        # quoteSummary wraps numbers as {"raw": x, "fmt": "..."}.
        return {k: (v.get("raw") if isinstance(v, dict) and "raw" in v else v) for k, v in merged.items()}

    # -- mapping ----------------------------------------------------------

    @staticmethod
    def _map(info: dict[str, Any]) -> dict[str, Any]:
        shares_out = positive(info.get("sharesOutstanding"))
        shares_short = positive(info.get("sharesShort"))
        short_pct = to_float(info.get("shortPercentOfFloat"))
        if short_pct is not None:
            # Yahoo reports a 0-1 fraction; canonical form is 0-100.
            short_pct = short_pct * 100 if short_pct <= 1.5 else short_pct
        elif shares_short and positive(info.get("floatShares")):
            short_pct = shares_short / float(info["floatShares"]) * 100

        return {
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "exchange": info.get("exchange") or info.get("fullExchangeName"),
            "current_price": positive(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            ),
            "previous_close": positive(info.get("previousClose") or info.get("regularMarketPreviousClose")),
            "market_cap": positive(info.get("marketCap")),
            "float": positive(info.get("floatShares")) or shares_out,
            "shares_outstanding": shares_out,
            "short_pct_float": short_pct,
            "short_ratio": to_float(info.get("shortRatio")),
            "shares_short": shares_short,
            "shares_short_prior": positive(info.get("sharesShortPriorMonth")),
            "short_interest_date": _epoch_to_date(info.get("dateShortInterest")),
            "avg_volume_10d": positive(info.get("averageVolume10days") or info.get("averageDailyVolume10Day")),
            "avg_volume_3m": positive(info.get("averageVolume") or info.get("averageDailyVolume3Month")),
            "fifty_two_week_high": positive(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": positive(info.get("fiftyTwoWeekLow")),
            "beta": to_float(info.get("beta")),
        }

    # -- prices -----------------------------------------------------------

    def fetch_prices(self, ticker: str, period: str = "1y") -> Any | None:
        if yf is None or self.settings.offline:
            return None
        try:
            hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        except Exception:  # noqa: BLE001
            return None
        if hist is None or hist.empty:
            return None
        hist = hist.reset_index()
        rename = {c: c.title() for c in hist.columns if c.lower() in {"open", "high", "low", "close", "volume", "date"}}
        hist = hist.rename(columns=rename)
        if "Date" in hist.columns:
            try:
                hist["Date"] = hist["Date"].dt.tz_localize(None)
            except (AttributeError, TypeError):
                pass
        hist["ticker"] = ticker
        return hist[[c for c in ("Date", "Open", "High", "Low", "Close", "Volume", "ticker") if c in hist.columns]]


def _epoch_to_date(value: Any) -> str | None:
    epoch = to_float(value)
    if not epoch:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None
