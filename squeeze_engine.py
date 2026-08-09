"""
Short Squeeze Lab — Squeeze Scoring Engine
Calculates composite squeeze scores and detects setup conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class SqueezeEngine:
    """
    Composite squeeze scoring system.

    Components:
    - Short Interest Density (40%): % of float shorted
    - Borrow Cost Pressure (25%): Proxy via days-to-cover + price proximity to 52w low
    - Liquidity Trap (20%): Days to cover relative to float turnover
    - Momentum Fuel (15%): Volume spike + price compression
    """

    def __init__(self, 
                 si_weight=0.40,
                 borrow_weight=0.25,
                 liquidity_weight=0.20,
                 momentum_weight=0.15):
        self.weights = {
            "si": si_weight,
            "borrow": borrow_weight,
            "liquidity": liquidity_weight,
            "momentum": momentum_weight,
        }

        # Thresholds
        self.si_high = 30.0      # % of float
        self.si_med = 15.0
        self.dtc_high = 5.0      # Days to cover
        self.dtc_med = 3.0
        self.float_max = 100e6   # Small float preference

    def _normalize(self, value, low, high, cap=100):
        """Normalize value to 0-100 scale."""
        if pd.isna(value):
            return 0
        score = (value - low) / (high - low) * 100
        return min(max(score, 0), cap)

    def score_si_density(self, row):
        """Score based on short interest % of float."""
        si = row.get("short_pct_float", 0)
        if si >= self.si_high:
            return 100
        elif si >= self.si_med:
            return 50 + (si - self.si_med) / (self.si_high - self.si_med) * 50
        else:
            return self._normalize(si, 0, self.si_med, 50)

    def score_borrow_pressure(self, row):
        """Score based on days-to-cover + float scarcity."""
        dtc = row.get("short_ratio", 0)
        float_shares = row.get("float", 0)

        dtc_score = 0
        if dtc >= self.dtc_high:
            dtc_score = 100
        elif dtc >= self.dtc_med:
            dtc_score = 50 + (dtc - self.dtc_med) / (self.dtc_high - self.dtc_med) * 50
        else:
            dtc_score = self._normalize(dtc, 0, self.dtc_med, 50)

        # Small float bonus
        float_score = 100 if float_shares < self.float_max else 50

        return (dtc_score * 0.7) + (float_score * 0.3)

    def score_liquidity_trap(self, row):
        """Score liquidity risk — high SI + low volume = trap."""
        avg_vol = row.get("avg_volume_10d", row.get("avg_volume_3m", 1))
        shares_short = row.get("shares_short", 0)

        if avg_vol <= 0:
            return 0

        dtc = shares_short / avg_vol
        if dtc >= 10:
            return 100
        elif dtc >= 5:
            return 70 + (dtc - 5) / 5 * 30
        elif dtc >= 2:
            return 40 + (dtc - 2) / 3 * 30
        else:
            return self._normalize(dtc, 0, 2, 40)

    def score_momentum_fuel(self, row):
        """Score price/volume momentum potential."""
        price = row.get("current_price", 0)
        high_52w = row.get("fifty_two_week_high", 0)
        low_52w = row.get("fifty_two_week_low", 0)

        if high_52w <= low_52w:
            return 50

        # Price near 52w low = more compression = more fuel
        range_pct = (price - low_52w) / (high_52w - low_52w)
        compression_score = 100 - (range_pct * 100)  # Lower price = higher score

        # Short interest increasing
        short_now = row.get("shares_short", 0)
        short_prior = row.get("shares_short_prior", 0)
        if short_prior > 0:
            change_pct = (short_now - short_prior) / short_prior * 100
            increase_score = min(max(change_pct, 0), 100)
        else:
            increase_score = 0

        return (compression_score * 0.6) + (increase_score * 0.4)

    def calculate_score(self, row):
        """Calculate composite squeeze score for a single row."""
        si = self.score_si_density(row)
        borrow = self.score_borrow_pressure(row)
        liquidity = self.score_liquidity_trap(row)
        momentum = self.score_momentum_fuel(row)

        total = (si * self.weights["si"] +
                 borrow * self.weights["borrow"] +
                 liquidity * self.weights["liquidity"] +
                 momentum * self.weights["momentum"])

        return {
            "squeeze_score": round(total, 2),
            "si_score": round(si, 2),
            "borrow_score": round(borrow, 2),
            "liquidity_score": round(liquidity, 2),
            "momentum_score": round(momentum, 2),
        }

    def rank(self, df):
        """Add squeeze scores to dataframe and sort."""
        if df.empty:
            return df

        scores = df.apply(self.calculate_score, axis=1)
        score_df = pd.DataFrame(scores.tolist(), index=df.index)
        df = pd.concat([df, score_df], axis=1)

        # Add setup labels
        def label(row):
            if row["squeeze_score"] >= 80:
                return "🔥 HIGH SQUEEZE RISK"
            elif row["squeeze_score"] >= 60:
                return "⚠️ ELEVATED"
            elif row["squeeze_score"] >= 40:
                return "📊 WATCH"
            else:
                return "✅ LOW"

        df["setup_label"] = df.apply(label, axis=1)

        # Sort by score descending
        df = df.sort_values("squeeze_score", ascending=False)
        return df


if __name__ == "__main__":
    import data_fetcher as dfetch
    fetcher = dfetch.DataFetcher()
    data = fetcher.fetch_all(["GME", "AMC", "TSLA"])
    engine = SqueezeEngine()
    ranked = engine.rank(data)
    print(ranked[["ticker", "squeeze_score", "setup_label", "short_pct_float", "short_ratio"]].to_string())
