"""
Short Squeeze Lab — Flask API Server
Serves dashboard and REST endpoints for squeeze data & backtests.
"""

from flask import Flask, jsonify, send_from_directory, request
import os
import json
from datetime import datetime

from data_fetcher import DataFetcher
from squeeze_engine import SqueezeEngine
from backtester import SqueezeBacktester

app = Flask(__name__, static_folder="static")

# Cache
_data_cache = None
_cache_time = None
CACHE_TTL_MINUTES = 30

def get_squeeze_data(force_refresh=False):
    """Fetch and rank squeeze candidates."""
    global _data_cache, _cache_time

    if not force_refresh and _data_cache is not None:
        if _cache_time and (datetime.now() - _cache_time).seconds < CACHE_TTL_MINUTES * 60:
            return _data_cache

    fetcher = DataFetcher()
    engine = SqueezeEngine()

    df = fetcher.fetch_all()
    if df.empty:
        return []

    ranked = engine.rank(df)
    _data_cache = ranked.to_dict("records")
    _cache_time = datetime.now()
    return _data_cache

@app.route("/")
def index():
    return send_from_directory("static", "dashboard.html")

@app.route("/api/squeeze-data")
def api_squeeze_data():
    force = request.args.get("refresh", "false").lower() == "true"
    data = get_squeeze_data(force_refresh=force)
    return jsonify(data)

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run backtest on current watchlist."""
    from data_fetcher import DataFetcher

    fetcher = DataFetcher()
    bt = SqueezeBacktester()

    # Fetch price history for all tickers
    tickers = fetcher.watchlist[:20]  # Limit for speed
    price_data = {}
    signals_data = {}

    for ticker in tickers:
        hist = fetcher.fetch_price_history(ticker, period="1y")
        if hist is not None and not hist.empty:
            price_data[ticker] = hist
            # Create dummy signals dataframe for backtest compatibility
            signals_data[ticker] = pd.DataFrame({
                "timestamp": hist["Date"],
                "squeeze_score": [70] * len(hist)  # Simplified; real impl would compute per-day
            })

    trades_df = bt.run_portfolio(price_data, signals_data)
    summary = bt.summarize()

    return jsonify({
        "summary": summary,
        "trades": trades_df.to_dict("records") if not trades_df.empty else []
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
