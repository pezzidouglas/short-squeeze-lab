"""
Short Squeeze Lab — Backtester
Simulates squeeze strategy on historical price data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SqueezeBacktester:
    """
    Backtest a long-only squeeze strategy:
    - Entry: squeeze_score > threshold + volume > 1.5x avg
    - Exit: N days hold OR +20% gain OR -10% loss
    """

    def __init__(self,
                 entry_score_threshold=70,
                 volume_mult=1.5,
                 hold_days=5,
                 take_profit=0.20,
                 stop_loss=-0.10,
                 max_position_pct=0.20):
        self.entry_score = entry_score_threshold
        self.volume_mult = volume_mult
        self.hold_days = hold_days
        self.tp = take_profit
        self.sl = stop_loss
        self.max_pos = max_position_pct
        self.trades = []

    def run_single(self, ticker, price_df, signals_df):
        """
        Run backtest for single ticker.
        price_df: columns [Date, Open, High, Low, Close, Volume, ticker]
        signals_df: pre-computed squeeze scores aligned by date
        """
        if price_df.empty:
            return []

        price_df = price_df.copy().sort_values("Date").reset_index(drop=True)

        # Calculate rolling average volume
        price_df["avg_volume_20d"] = price_df["Volume"].rolling(20).mean()

        trades = []
        in_position = False
        entry_date = None
        entry_price = None

        for i, row in price_df.iterrows():
            date = row["Date"]
            price = row["Close"]
            volume = row["Volume"]
            avg_vol = row["avg_volume_20d"]

            # Get signal for this date
            signal_row = signals_df[signals_df["timestamp"] == date] if "timestamp" in signals_df.columns else None
            squeeze_score = signal_row["squeeze_score"].values[0] if signal_row is not None and len(signal_row) > 0 else 0

            if not in_position:
                # Entry check
                vol_spike = volume > (avg_vol * self.volume_mult) if avg_vol > 0 else False
                if squeeze_score >= self.entry_score and vol_spike:
                    in_position = True
                    entry_date = date
                    entry_price = price
            else:
                # Exit checks
                days_held = (date - entry_date).days if isinstance(date, pd.Timestamp) else 0
                pnl_pct = (price - entry_price) / entry_price

                exit_reason = None
                if pnl_pct >= self.tp:
                    exit_reason = "TAKE_PROFIT"
                elif pnl_pct <= self.sl:
                    exit_reason = "STOP_LOSS"
                elif days_held >= self.hold_days:
                    exit_reason = "TIME_EXIT"

                if exit_reason:
                    trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(price, 2),
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "days_held": days_held,
                        "exit_reason": exit_reason,
                    })
                    in_position = False
                    entry_date = None
                    entry_price = None

        return trades

    def run_portfolio(self, price_data_dict, signals_dict):
        """
        Run backtest across multiple tickers.
        price_data_dict: {ticker: price_df}
        signals_dict: {ticker: signals_df}
        """
        all_trades = []
        for ticker, price_df in price_data_dict.items():
            signals_df = signals_dict.get(ticker, pd.DataFrame())
            trades = self.run_single(ticker, price_df, signals_df)
            all_trades.extend(trades)

        self.trades = pd.DataFrame(all_trades)
        return self.trades

    def summarize(self):
        """Generate performance summary."""
        if self.trades.empty:
            return {"error": "No trades executed"}

        t = self.trades
        wins = t[t["pnl_pct"] > 0]
        losses = t[t["pnl_pct"] <= 0]

        summary = {
            "total_trades": len(t),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(t) * 100, 2),
            "avg_return": round(t["pnl_pct"].mean(), 2),
            "avg_win": round(wins["pnl_pct"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loss": round(losses["pnl_pct"].mean(), 2) if len(losses) > 0 else 0,
            "max_gain": round(t["pnl_pct"].max(), 2),
            "max_loss": round(t["pnl_pct"].min(), 2),
            "profit_factor": round(abs(wins["pnl_pct"].sum() / losses["pnl_pct"].sum()), 2) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else float('inf'),
            "avg_hold_days": round(t["days_held"].mean(), 1),
            "total_return": round(t["pnl_pct"].sum(), 2),
        }

        # Exit reason breakdown
        summary["exits"] = t["exit_reason"].value_counts().to_dict()

        return summary


if __name__ == "__main__":
    # Quick test
    bt = SqueezeBacktester()
    print("Backtester initialized. Run via main.py or app.py")
