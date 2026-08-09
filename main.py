#!/usr/bin/env python3
"""
Short Squeeze Lab — CLI Runner
Usage:
    python main.py scan              # Run squeeze scan
    python main.py backtest          # Run backtest
    python main.py serve             # Start dashboard server
    python main.py export            # Export results to CSV/JSON
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

from data_fetcher import DataFetcher
from squeeze_engine import SqueezeEngine
from backtester import SqueezeBacktester

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cmd_scan():
    print("=" * 60)
    print("🔥 SHORT SQUEEZE SCANNER")
    print("=" * 60)

    fetcher = DataFetcher()
    engine = SqueezeEngine()

    print(f"\nFetching data for {len(fetcher.watchlist)} tickers...")
    df = fetcher.fetch_all()

    if df.empty:
        print("❌ No data fetched. Check connections.")
        return

    ranked = engine.rank(df)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{OUTPUT_DIR}/squeeze_scan_{timestamp}.csv"
    json_path = f"{OUTPUT_DIR}/squeeze_scan_{timestamp}.json"

    ranked.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(ranked.to_dict("records"), f, indent=2, default=str)

    # Print top 15
    print(f"\n📊 TOP 15 SQUEEZE CANDIDATES:")
    print("-" * 100)
    display = ranked.head(15)[["ticker", "name", "squeeze_score", "setup_label", 
                                "short_pct_float", "short_ratio", "current_price", "market_cap"]]
    print(display.to_string(index=False))
    print("-" * 100)

    high_risk = ranked[ranked["squeeze_score"] >= 80]
    print(f"\n🔥 HIGH RISK (Score ≥ 80): {len(high_risk)} tickers")
    if len(high_risk) > 0:
        print(high_risk[["ticker", "squeeze_score", "short_pct_float", "short_ratio"]].to_string(index=False))

    print(f"\n💾 Saved to: {csv_path} | {json_path}")
    return ranked

def cmd_backtest():
    print("=" * 60)
    print("🎲 SQUEEZE BACKTESTER")
    print("=" * 60)

    fetcher = DataFetcher()
    bt = SqueezeBacktester()

    # Use a subset for speed
    test_tickers = ["GME", "AMC", "TSLA", "NVAX", "BYND", "CVNA", "UPST"]
    print(f"\nBacktesting {len(test_tickers)} tickers over 1 year...")

    price_data = {}
    signals_data = {}

    for ticker in test_tickers:
        print(f"  Fetching {ticker}...", end=" ")
        hist = fetcher.fetch_price_history(ticker, period="1y")
        if hist is not None and not hist.empty:
            price_data[ticker] = hist
            # Generate daily squeeze scores from historical data
            # In real impl, you'd compute this from historical SI data
            signals_data[ticker] = pd.DataFrame({
                "timestamp": hist["Date"],
                "squeeze_score": [65] * len(hist)  # Placeholder
            })
            print(f"✓ {len(hist)} bars")
        else:
            print("✗ no data")

    trades_df = bt.run_portfolio(price_data, signals_data)
    summary = bt.summarize()

    print(f"\n📈 BACKTEST RESULTS:")
    print("-" * 50)
    for k, v in summary.items():
        if k != "exits":
            print(f"  {k.replace('_', ' ').title()}: {v}")
    print("-" * 50)

    if not trades_df.empty:
        print(f"\nLast 10 trades:")
        print(trades_df.tail(10).to_string(index=False))

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_df.to_csv(f"{OUTPUT_DIR}/backtest_trades_{timestamp}.csv", index=False)
        with open(f"{OUTPUT_DIR}/backtest_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)

def cmd_serve():
    print("🚀 Starting dashboard server...")
    os.system("python app.py")

def cmd_export():
    print("📤 Exporting latest scan...")
    df = cmd_scan()
    if df is not None:
        # Also save as latest
        df.to_csv(f"{OUTPUT_DIR}/latest_scan.csv", index=False)
        print(f"\n📁 Latest scan saved to {OUTPUT_DIR}/latest_scan.csv")

COMMANDS = {
    "scan": cmd_scan,
    "backtest": cmd_backtest,
    "serve": cmd_serve,
    "export": cmd_export,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
