#!/usr/bin/env python3
"""
Embed a precomputed payload into `static/index.html`.

The dashboard prefers the live API, but the same file is also deployed as a
standalone page (tiiny.site, GitHub Pages, a shared HTML file) where there is no
API to call. This script runs a real scan, a real backtest and a real strategy
sweep, then writes the results into the `#embedded-data` block so the static
copy shows numbers this codebase actually produced.

    python scripts/build_static.py                 # whatever sources are enabled
    SQUEEZE_SOURCES=snapshot python scripts/build_static.py   # offline sample

The page labels its own data mode, so a snapshot-built file says "sample" in the
header rather than passing synthetic numbers off as live.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squeezelab.backtest import SqueezeBacktester, StrategyParams, compare_strategies  # noqa: E402
from squeezelab.config import SETTINGS, WEIGHT_PRESETS  # noqa: E402
from squeezelab.consolidate import coverage_matrix, disagreement_report  # noqa: E402
from squeezelab.history import build_signals  # noqa: E402
from squeezelab.service import MarketDataService  # noqa: E402
from squeezelab.sources import describe_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "static", "index.html")
MARKER = re.compile(
    r'(<script id="embedded-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def build_payload(universe: int, period: str) -> dict:
    service = MarketDataService(SETTINGS)
    result = service.scan()
    rows = [{k: v for k, v in row.items() if k != "candidates"} for row in result.rows]

    scan_obj = result.consolidated_scan
    sources = service.sources
    tickers = [str(r["ticker"]) for r in rows[:universe]]
    prices = service.price_history(tickers, period=period)

    backtest_payload = None
    strategies_payload = None
    if prices:
        records = {str(r["ticker"]).upper(): r for r in rows}
        signals = build_signals(records, prices)
        outcome = SqueezeBacktester(StrategyParams(label="baseline")).run(prices, signals)
        backtest_payload = {
            "backtest": outcome.to_dict(),
            "tickers": sorted(prices),
            "period": period,
            "data_mode": result.data_mode,
        }
        strategies_payload = {
            "strategies": compare_strategies(prices, signals),
            "tickers": sorted(prices),
            "period": period,
        }

    return {
        "generated_at": result.generated_at,
        "presets": {name: {"weights": weights} for name, weights in WEIGHT_PRESETS.items()},
        "scan": {
            "ok": True,
            "rows": rows,
            "data_mode": result.data_mode,
            "preset": result.preset,
            "weights": result.weights,
            "generated_at": result.generated_at,
            "notes": result.notes,
        },
        "sources": {
            "ok": True,
            "health": result.health,
            "catalog": describe_sources([s.meta.key for s in sources]),
            "coverage": coverage_matrix(scan_obj),
            "disagreements": disagreement_report(scan_obj),
            "data_mode": result.data_mode,
        },
        "backtest": backtest_payload,
        "strategies": strategies_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=int, default=15, help="tickers to include in the embedded backtest")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--out", default=TARGET)
    args = parser.parse_args()

    payload = build_payload(args.universe, args.period)
    with open(args.out, "r", encoding="utf-8") as fh:
        html = fh.read()
    if not MARKER.search(html):
        print("could not find the #embedded-data block in", args.out)
        return 1

    blob = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")
    html = MARKER.sub(lambda m: m.group(1) + "\n" + blob + "\n" + m.group(3), html, count=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_kb = os.path.getsize(args.out) / 1024
    rows = len(payload["scan"]["rows"])
    trades = len((payload.get("backtest") or {}).get("backtest", {}).get("trades", []))
    print(
        f"embedded {rows} rows, {trades} trades into {args.out} "
        f"({size_kb:.0f} KB, mode={payload['scan']['data_mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
