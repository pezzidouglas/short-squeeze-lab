#!/usr/bin/env python3
"""
Regenerate `squeezelab/data/snapshot.json` — the bundled offline sample.

The snapshot exists so the dashboard, CLI and tests run with no network. It is
deterministic (seeded per ticker), internally consistent (the quote fields are
derived from the generated bars, not invented separately) and clearly labelled
as sample data everywhere it surfaces.

    python scripts/build_snapshot.py

Nothing in here is market data. It is a fixture whose only job is to exercise
the real scoring, consolidation and backtest code paths.
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import date, timedelta

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "squeezelab", "data", "snapshot.json")

# ticker, name, sector, price, market cap, float, short % float, days to cover,
# annualised drift, annualised vol, off-exchange short volume %
PROFILES = [
    ("SPRY", "ARS Pharmaceuticals", "Healthcare", 28.50, 2.10e9, 45e6, 34.6, 8.2, 0.35, 0.85, 62.0),
    ("PCT", "PureCycle Technologies", "Basic Materials", 4.20, 3.80e8, 85e6, 36.3, 12.5, -0.30, 0.95, 58.5),
    ("SLS", "SELLAS Life Sciences", "Healthcare", 12.40, 8.90e8, 28e6, 32.2, 9.8, 0.15, 1.10, 60.1),
    ("INDI", "indie Semiconductor", "Technology", 6.80, 1.20e9, 95e6, 31.2, 7.4, -0.20, 0.80, 55.3),
    ("RUM", "Rumble Inc", "Technology", 14.20, 5.40e9, 120e6, 26.2, 4.5, 0.25, 0.75, 53.8),
    ("PSNY", "Polestar Automotive", "Consumer Cyclical", 1.15, 2.50e9, 850e6, 6.5, 2.1, -0.45, 0.90, 51.2),
    ("GME", "GameStop Corp", "Consumer Cyclical", 28.40, 1.28e10, 305e6, 18.5, 3.2, 0.05, 0.70, 49.6),
    ("AMC", "AMC Entertainment", "Communication Services", 4.85, 1.60e9, 263e6, 16.2, 2.8, -0.25, 0.80, 52.4),
    ("CVNA", "Carvana Co", "Consumer Cyclical", 185.00, 3.80e10, 105e6, 22.4, 4.1, 0.40, 0.65, 47.9),
    ("NVAX", "Novavax Inc", "Healthcare", 8.40, 1.30e9, 98e6, 19.8, 5.5, -0.10, 0.85, 54.7),
    ("BYND", "Beyond Meat Inc", "Consumer Defensive", 5.20, 3.40e8, 58e6, 28.5, 6.2, -0.35, 0.95, 56.9),
    ("UPST", "Upstart Holdings", "Financial Services", 42.10, 3.90e9, 82e6, 24.1, 3.8, 0.10, 0.85, 50.8),
    ("AFRM", "Affirm Holdings", "Financial Services", 58.30, 1.85e10, 250e6, 12.4, 2.6, 0.20, 0.70, 48.2),
    ("HOOD", "Robinhood Markets", "Financial Services", 31.70, 2.80e10, 720e6, 6.8, 1.9, 0.30, 0.65, 46.5),
    ("LCID", "Lucid Group", "Consumer Cyclical", 2.35, 5.60e9, 620e6, 27.9, 4.9, -0.30, 0.85, 55.1),
    ("RIVN", "Rivian Automotive", "Consumer Cyclical", 12.80, 1.30e10, 880e6, 13.6, 3.1, -0.15, 0.70, 49.9),
    ("SOUN", "SoundHound AI", "Technology", 9.60, 3.70e9, 300e6, 29.4, 3.4, 0.45, 1.05, 57.2),
    ("BBAI", "BigBear.ai Holdings", "Technology", 4.45, 1.10e9, 180e6, 25.7, 3.9, 0.20, 1.00, 56.4),
    ("IONQ", "IonQ Inc", "Technology", 38.20, 8.90e9, 210e6, 21.3, 3.3, 0.35, 0.95, 52.7),
    ("ASTS", "AST SpaceMobile", "Technology", 24.60, 7.40e9, 165e6, 18.9, 4.2, 0.30, 0.90, 53.1),
    ("MSTR", "MicroStrategy Inc", "Technology", 340.00, 6.80e10, 135e6, 12.4, 1.8, 0.25, 0.80, 45.8),
    ("SMCI", "Super Micro Computer", "Technology", 41.50, 2.45e10, 480e6, 15.7, 2.4, -0.05, 0.85, 48.6),
    ("TSLA", "Tesla Inc", "Consumer Cyclical", 245.00, 7.80e11, 2.80e9, 3.2, 0.9, 0.05, 0.55, 43.2),
    ("PLTR", "Palantir Technologies", "Technology", 78.00, 1.75e11, 1.90e9, 4.5, 1.2, 0.30, 0.50, 42.7),
    ("AAPL", "Apple Inc", "Technology", 225.00, 3.40e12, 1.50e10, 0.8, 0.4, 0.10, 0.28, 40.1),
]

BARS = 320
AS_OF = date(2026, 8, 7)


def trading_days(end: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    return days


def build_path(rng: random.Random, price: float, drift: float, vol: float, count: int) -> list[float]:
    """A GBM path rescaled so the final close equals the quoted price."""
    daily_drift = drift / 252
    daily_vol = vol / math.sqrt(252)
    path = [1.0]
    for _ in range(count - 1):
        shock = rng.gauss(daily_drift - 0.5 * daily_vol**2, daily_vol)
        path.append(path[-1] * math.exp(shock))
    scale = price / path[-1]
    return [max(round(p * scale, 4), 0.01) for p in path]


def build_bars(rng: random.Random, closes: list[float], avg_volume: float, dates: list[date]) -> dict:
    opens, highs, lows, volumes = [], [], [], []
    for index, close in enumerate(closes):
        prev = closes[index - 1] if index else close
        open_price = round(prev * (1 + rng.gauss(0, 0.006)), 4)
        span = abs(rng.gauss(0, 0.018)) + 0.004
        high = round(max(open_price, close) * (1 + span), 4)
        low = round(min(open_price, close) * (1 - span), 4)
        move = abs(close / prev - 1) if prev else 0.0
        # Volume rides the size of the move, with the odd genuine spike.
        multiplier = 1 + move * 22 + (rng.expovariate(1 / 0.35) if rng.random() < 0.10 else 0)
        volumes.append(int(max(avg_volume * multiplier * rng.uniform(0.6, 1.4), 1000)))
        opens.append(max(open_price, 0.01))
        highs.append(max(high, 0.01))
        lows.append(max(min(low, open_price, close), 0.01))
    return {
        "dates": [d.isoformat() for d in dates],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def build() -> dict:
    records: dict[str, dict] = {}
    prices: dict[str, dict] = {}
    dates = trading_days(AS_OF, BARS)

    for ticker, name, sector, price, market_cap, float_shares, short_pct, dtc, drift, vol, flow in PROFILES:
        rng = random.Random(f"squeeze-lab::{ticker}")
        closes = build_path(rng, price, drift, vol, BARS)
        avg_volume = max(float_shares / rng.uniform(60, 220), 50_000)
        bars = build_bars(rng, closes, avg_volume, dates)
        prices[ticker] = bars

        year_high = max(bars["high"][-252:])
        year_low = min(bars["low"][-252:])
        vol_10d = sum(bars["volume"][-10:]) / 10
        vol_3m = sum(bars["volume"][-63:]) / 63
        shares_short = float_shares * short_pct / 100
        prior = shares_short * rng.uniform(0.82, 1.14)

        series = []
        for offset in range(20):
            day = dates[-(20 - offset)]
            ratio = max(min(flow + rng.gauss(0, 2.4) + offset * 0.12, 85.0), 15.0)
            total = bars["volume"][-(20 - offset)]
            series.append(
                {
                    "date": day.isoformat(),
                    "ratio": round(ratio, 2),
                    "short": round(total * ratio / 100, 0),
                    "total": float(total),
                }
            )
        slow = sum(p["ratio"] for p in series) / len(series)
        fast = sum(p["ratio"] for p in series[-5:]) / 5

        records[ticker] = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "exchange": "NASDAQ",
            "current_price": round(closes[-1], 2),
            "previous_close": round(closes[-2], 2),
            "market_cap": market_cap,
            "float": float_shares,
            "shares_outstanding": round(float_shares * rng.uniform(1.02, 1.35)),
            "short_pct_float": short_pct,
            "short_ratio": dtc,
            "shares_short": round(shares_short),
            "shares_short_prior": round(prior),
            "short_interest_date": (AS_OF - timedelta(days=rng.randint(5, 22))).isoformat(),
            "avg_volume_10d": round(vol_10d),
            "avg_volume_3m": round(vol_3m),
            "fifty_two_week_high": round(year_high, 2),
            "fifty_two_week_low": round(year_low, 2),
            "beta": round(rng.uniform(0.8, 2.6), 2),
            "off_exchange_short_pct": round(slow, 2),
            "off_exchange_trend": round(fast - slow, 2),
            "off_exchange_series": series,
        }

    return {
        "schema": 2,
        "synthetic": True,
        "captured_at": AS_OF.isoformat(),
        "description": (
            "Deterministic sample dataset for offline use. Not market data — "
            "generated by scripts/build_snapshot.py so the dashboard, CLI and "
            "tests exercise the real scoring and backtest code without a network."
        ),
        "records": records,
        "prices": prices,
    }


if __name__ == "__main__":
    payload = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH} ({len(payload['records'])} tickers, {size_kb:.0f} KB)")
