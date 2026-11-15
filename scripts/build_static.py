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

from squeezelab import playbook as pb  # noqa: E402
from squeezelab.backtest import SqueezeBacktester, StrategyParams, compare_strategies  # noqa: E402
from squeezelab.config import SETTINGS, WEIGHT_PRESETS  # noqa: E402
from squeezelab.consolidate import coverage_matrix, disagreement_report  # noqa: E402
from squeezelab.history import build_signals  # noqa: E402
from squeezelab.service import MarketDataService  # noqa: E402
from squeezelab.sources import describe_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "static", "index.html")
ODDS = os.path.join(ROOT, "backtests", "playbook_odds.json")
OOS = os.path.join(ROOT, "backtests", "playbook_oos.json")
SCORECARD = os.path.join(ROOT, "backtests", "strategy_scorecard.json")
SI_CACHE = os.path.join(ROOT, "data", "cache", "playbook", "finra_si")
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

    playbook_payload = build_playbook(service, rows, prices)

    return {
        "generated_at": result.generated_at,
        "playbook": playbook_payload,
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


def latest_finra_si(tickers: list[str]) -> tuple[dict[str, dict], str | None]:
    """Newest FINRA short interest that was public by today (same lag as the backtest)."""
    try:
        import pandas as pd

        dates = pb.finra_settlement_dates("2025-01-01")
        today = pd.Timestamp.today().normalize()
        usable = [d for d in dates if pd.Timestamp(d) + pd.offsets.BDay(pb.SI_PUBLICATION_LAG_BDAYS) <= today]
        if not usable:
            return {}, None
        day = usable[-1]
        rows = pb.finra_si_for_date(day, SI_CACHE)
        want = {t.upper() for t in tickers}
        out = {}
        for r in rows:
            sym = str(r.get("symbolCode", "")).upper().replace(".", "-").replace("/", "-")
            if sym in want:
                out[sym] = {
                    "shares_short": r.get("currentShortPositionQuantity"),
                    "si_adv": r.get("averageDailyVolumeQuantity"),
                    "shares_short_prior": r.get("previousShortPositionQuantity"),
                    "short_ratio": r.get("daysToCoverQuantity"),
                }
        return out, day
    except Exception as exc:  # noqa: BLE001 - fall back to the scan's own short data
        print("FINRA point-in-time short interest unavailable:", exc)
        return {}, None


def build_playbook(service: MarketDataService, rows: list[dict], prices: dict) -> dict | None:
    """Today's action/levels per name plus the cached backtest odds."""
    odds = None
    if os.path.exists(ODDS):
        with open(ODDS, encoding="utf-8") as fh:
            odds = json.load(fh)
    tickers = [str(r["ticker"]).upper() for r in rows]
    si_map, si_date = latest_finra_si(tickers)
    calls = []
    bars_by: dict = {}
    for row in rows:
        t = str(row["ticker"]).upper()
        # Match the dividend/split-adjusted backtest bars. The latest close is
        # still today's quote; adjusted history makes ATR internally consistent.
        try:
            import yfinance as yf
            bars = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True).reset_index()
        except Exception:
            bars = None  # never silently replace prices with synthetic data
        bars_by[t] = bars
        calls.append(pb.todays_call(t, bars, row, odds, si=si_map.get(t)))
        calls[-1]["short_data"] = f"FINRA {si_date}" if t in si_map else "scan (Yahoo)"
    calls = pb.rank_calls(calls)
    summary = None
    if odds:
        summary = {k: odds.get(k) for k in ("generated_at", "rule", "rule_text", "period", "universe", "tiers",
                                             "tier_labels", "combined", "by_year", "robustness", "data_coverage", "caveats")}
    oos = None
    if os.path.exists(OOS):
        with open(OOS, encoding="utf-8") as fh:
            raw = json.load(fh)
        oos = {k: v for k, v in raw.items() if k != "results"}  # drop the 108-candidate grid
    strategies = None
    if os.path.exists(SCORECARD):
        with open(SCORECARD, encoding="utf-8") as fh:
            card = json.load(fh)
        strategies = {k: v for k, v in card.items()}
        strategies["families"] = [{k: v for k, v in f.items() if k != "all_variants"} for f in card.get("families", [])]
        strategies["signals"] = []
        if card.get("passing"):
            try:
                sys.path.insert(0, os.path.join(ROOT, "scripts"))
                import strategy_search  # noqa: PLC0415

                strategies["signals"] = strategy_search.todays_signals(card, bars_by)
            except Exception as exc:  # noqa: BLE001 - no signals beats invented ones
                print("strategy signals failed:", exc)
                strategies["signals_error"] = str(exc)
    return {"calls": calls, "odds": summary, "oos": oos, "strategies": strategies, "si_settlement_date": si_date, "account_risk_pct": 0.25,
            "min_edge_pct": pb.MIN_EDGE_PCT, "min_trades": pb.MIN_TRADES}


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
