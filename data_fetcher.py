"""
Short Squeeze Lab — Data Fetcher Module
Pulls short interest, price, volume, and fundamental data from free sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json

# Known heavily-shorted watchlist + some popular names
DEFAULT_WATCHLIST = [
    "SPRY", "PCT", "INDI", "SLS", "RUM", "LUCK", "PSNY",
    "GME", "AMC", "BBBYQ", "MULN", "NVAX", "GOEV", "LCID",
    "RIVN", "BYND", "CVNA", "UPST", "AFRM", "HOOD",
    "TSLA", "AAPL", "MSFT", "NVDA", "AMD", "INTC",
    "PLTR", "SOFI", "NIO", "XPEV", "LI",
    "ASTS", "RKLB", "IONQ", "QBTS",
    "SOUN", "BBAI", "SMCI", "MSTR",
]

class DataFetcher:
    def __init__(self, watchlist=None):
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch_yf_info(self, ticker):
        """Pull key stats from Yahoo Finance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info

            # Key metrics for squeeze detection
            data = {
                "ticker": ticker,
                "name": info.get("shortName", info.get("longName", ticker)),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "float": info.get("floatShares", info.get("sharesOutstanding", 0)),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "short_ratio": info.get("shortRatio", 0),  # Days to cover
                "short_pct_float": info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") else 0,
                "short_pct_shares": info.get("sharesShort", 0) / info.get("sharesOutstanding", 1) * 100 if info.get("sharesShort") else 0,
                "shares_short": info.get("sharesShort", 0),
                "shares_short_prior": info.get("sharesShortPriorMonth", 0),
                "avg_volume_10d": info.get("averageVolume10days", 0),
                "avg_volume_3m": info.get("averageVolume", 0),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "beta": info.get("beta", 1.0),
                "recommendation": info.get("recommendationKey", "none"),
                "target_mean_price": info.get("targetMeanPrice", 0),
                "timestamp": datetime.now().isoformat(),
            }
            return data
        except Exception as e:
            print(f"[ERROR] yf info for {ticker}: {e}")
            return None

    def fetch_price_history(self, ticker, period="6mo", interval="1d"):
        """Pull OHLCV history for backtesting."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            if hist.empty:
                return None
            hist.reset_index(inplace=True)
            hist["ticker"] = ticker
            return hist
        except Exception as e:
            print(f"[ERROR] price history for {ticker}: {e}")
            return None

    def fetch_finviz_screener(self, filters="sh_short_o15"):
        """
        Attempt to fetch Finviz screener export.
        Note: Finviz may block automated requests; this is a best-effort fallback.
        """
        url = f"https://finviz.com/export.ashx?v=111&f={filters}&o=-shortinterestshare"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                return df
        except Exception as e:
            print(f"[WARN] Finviz fetch failed: {e}")
        return pd.DataFrame()

    def fetch_all(self, tickers=None):
        """Fetch info for entire watchlist."""
        tickers = tickers or self.watchlist
        results = []
        for i, ticker in enumerate(tickers):
            print(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
            data = self.fetch_yf_info(ticker)
            if data:
                results.append(data)
            time.sleep(0.3)  # Be nice to Yahoo
        return pd.DataFrame(results)

    def fetch_batch_prices(self, tickers=None, period="1y"):
        """Fetch price history for all tickers."""
        tickers = tickers or self.watchlist
        all_hist = []
        for ticker in tickers:
            hist = self.fetch_price_history(ticker, period=period)
            if hist is not None:
                all_hist.append(hist)
        if all_hist:
            return pd.concat(all_hist, ignore_index=True)
        return pd.DataFrame()


if __name__ == "__main__":
    df = DataFetcher().fetch_all(["GME", "AMC", "TSLA"])
    print(df[["ticker", "short_pct_float", "short_ratio", "avg_volume_10d"]].to_string())
