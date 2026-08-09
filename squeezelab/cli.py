"""
Command line interface.

    python main.py scan [--preset balanced] [--tickers GME,AMC] [--top 20]
    python main.py sources [--ticker GME]
    python main.py compare GME AMC SPRY
    python main.py backtest [--tickers ...] [--period 1y] [--entry-score 70]
    python main.py strategies
    python main.py presets
    python main.py serve [--port 5000]
    python main.py export [--format csv|json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Sequence

from .backtest import SqueezeBacktester, StrategyParams, compare_strategies
from .compare import compare_presets, compare_sources, compare_tickers
from .config import SETTINGS, WEIGHT_PRESETS
from .history import build_signals
from .service import MarketDataService

RULE = "─" * 78


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def human(value: Any, unit: str = "") -> str:
    if value is None:
        return "–"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "sh" or abs(number) >= 1e6:
        for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(number) >= divisor:
                return f"{number / divisor:.2f}{suffix}"
    if unit == "%":
        return f"{number:.1f}%"
    if unit == "$":
        return f"${number:,.2f}"
    return f"{number:,.2f}"


def bar(score: float | None, width: int = 12) -> str:
    if score is None:
        return " " * width
    filled = int(round(max(0.0, min(100.0, float(score))) / 100 * width))
    return "█" * filled + "·" * (width - filled)


def header(title: str) -> None:
    print(f"\n{title}\n{RULE}")


def print_notes(result: Any) -> None:
    for note in getattr(result, "notes", []) or []:
        print(f"  ! {note}")


def service_for(args: argparse.Namespace) -> MarketDataService:
    tickers = _split(getattr(args, "tickers", None))
    return MarketDataService(SETTINGS, watchlist=tickers or None)


def _split(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    service = service_for(args)
    result = service.scan(preset=args.preset)
    if not result.rows:
        print("No data returned by any source.")
        print_notes(result)
        return 1

    header(f"SQUEEZE SCAN · preset={result.preset} · mode={result.data_mode} · {result.generated_at}")
    print(f"{'#':>3} {'TICKER':<7}{'SCORE':>7} {'BAR':<13}{'CONF':>6}{'SI%':>7}{'DTC':>6}{'OFFEX':>7}{'FLOAT':>9}  SETUP")
    for row in result.rows[: args.top]:
        print(
            f"{row['rank']:>3} {row['ticker']:<7}"
            f"{row['squeeze_score']:>7.1f} {bar(row['squeeze_score']):<13}"
            f"{row.get('data_confidence', 0):>6.0f}"
            f"{human(row.get('short_pct_float'), '%'):>7}"
            f"{human(row.get('short_ratio')):>6}"
            f"{human(row.get('off_exchange_short_pct'), '%'):>7}"
            f"{human(row.get('float'), 'sh'):>9}  {row['setup_label']}"
        )
    print(RULE)

    high = [r for r in result.rows if r["squeeze_score"] >= 80]
    print(f"  {len(high)} name(s) at HIGH SQUEEZE RISK · {len(result.rows)} scanned")
    print("  sources: " + ", ".join(f"{h['key']}={h['status']}({h['tickers_returned']})" for h in result.health))
    if result.disagreements:
        worst = result.disagreements[0]
        print(
            f"  widest source disagreement: {worst['ticker']} {worst['label']} "
            f"{worst['spread'] * 100:.0f}% apart"
        )
    print_notes(result)

    if args.save:
        _save(result, args.format)
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    service = service_for(args)
    scan, sources = service.consolidated()

    header("SOURCE HEALTH")
    for source in sources:
        health = source.health
        print(
            f"  {health.key:<9} {health.status:<8} "
            f"{health.tickers_returned:>3}/{health.tickers_requested:<3} tickers  "
            f"{health.fields_returned:>4} fields  {health.latency_ms:>7.0f} ms"
            + (f"  — {health.detail}" if health.detail else "")
        )

    if args.ticker:
        record = scan.by_ticker().get(args.ticker.upper())
        if record is None:
            print(f"\n{args.ticker.upper()} not found in the scan.")
            return 1
        detail = compare_sources(record)
        header(f"SOURCE COMPARISON · {detail['ticker']} · confidence {detail['confidence']}")
        for field in detail["fields"]:
            agreement = field["agreement"]
            flag = "  " if agreement is None or agreement >= 0.99 else "!!"
            values = "  ".join(f"{k}={human(v, field['unit'])}" for k, v in field["values"].items())
            print(f" {flag} {field['label']:<26} won by {field['winner']:<9} {values}")
        return 0

    from .consolidate import disagreement_report

    rows = disagreement_report(scan, limit=12)
    if rows:
        header("WIDEST SOURCE DISAGREEMENTS")
        for row in rows:
            print(
                f"  {row['ticker']:<7}{row['label']:<24}"
                f"{row['spread'] * 100:>6.0f}% apart   "
                f"{row['low']['source']}={human(row['low']['value'])}  "
                f"{row['high']['source']}={human(row['high']['value'])}"
                if "low" in row
                else f"  {row['ticker']:<7}{row['label']:<24}values differ"
            )
    else:
        print("\nNo cross-source disagreements — only one source covered these fields.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    tickers = [t.upper() for t in args.symbols]
    if len(tickers) < 2:
        print("Give at least two tickers, e.g. `compare GME AMC`.")
        return 1

    service = MarketDataService(SETTINGS)
    result = service.scan(preset=args.preset)
    payload = compare_tickers(result.rows, tickers)
    if not payload["tickers"]:
        print("None of those tickers are in the scan.")
        return 1

    columns = payload["tickers"]
    header("HEAD TO HEAD · " + "  vs  ".join(columns))
    width = max(12, max(len(c) for c in columns) + 4)
    print(f"{'METRIC':<28}" + "".join(f"{c:>{width}}" for c in columns))
    print(RULE)
    for metric in payload["metrics"]:
        cells = []
        for ticker in columns:
            value = metric["values"].get(ticker)
            text = human(value, metric["unit"])
            if metric["leader"] == ticker and value is not None:
                text = f"*{text}"
            cells.append(f"{text:>{width}}")
        print(f"{metric['label']:<28}" + "".join(cells))

    print(RULE)
    print(f"{'COMPONENT':<28}" + "".join(f"{c:>{width}}" for c in columns))
    for index, component in enumerate(next(iter(payload["components"].values()), [])):
        cells = []
        for ticker in columns:
            score = payload["components"][ticker][index]["score"]
            cells.append(f"{human(score):>{width}}")
        print(f"{component['label']:<28}" + "".join(cells))

    verdict = payload["verdict"]
    if verdict:
        print(
            f"\n  {verdict['leader']} leads at {verdict['score']} — driven by "
            f"{verdict['driver'].lower()} ({verdict['driver_score']}), "
            f"data confidence {verdict['confidence']}."
        )
    print("  * = leads this row (for squeeze potential, not investment merit)")
    print_notes(result)
    if payload["missing"]:
        print(f"  not in scan: {', '.join(payload['missing'])}")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    service = service_for(args)
    scan, _ = service.consolidated()
    sensitivity = compare_presets(scan, top=args.top)
    names = sensitivity["presets"]

    header("PRESET SENSITIVITY · does the ranking survive a change of weights?")
    print(f"{'TICKER':<8}" + "".join(f"{n:>14}" for n in names) + f"{'SPREAD':>9}")
    print(RULE)
    for row in sensitivity["rows"]:
        cells = "".join(f"{row['scores'].get(n, 0):>14.1f}" for n in names)
        print(f"{row['ticker']:<8}{cells}{row['spread']:>9.1f}")
    print(RULE)
    print("  A wide spread means the ranking is an artefact of the weights, not the data.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    service = MarketDataService(SETTINGS)
    result = service.scan(preset=args.preset)
    tickers = _split(args.tickers) or [str(r["ticker"]) for r in result.rows[: args.universe]]

    print(f"Fetching {len(tickers)} price histories ({args.period})…")
    prices = service.price_history(tickers, period=args.period)
    if not prices:
        print("No price history available from any source.")
        return 1

    records = {str(r["ticker"]).upper(): r for r in result.rows}
    signals = build_signals(records, prices)
    params = StrategyParams(
        entry_score=args.entry_score,
        volume_mult=args.volume_mult,
        hold_days=args.hold_days,
        take_profit=args.take_profit / 100,
        stop_loss=-abs(args.stop_loss) / 100,
        max_positions=args.max_positions,
        label="cli",
    )
    outcome = SqueezeBacktester(params).run(prices, signals)
    summary = outcome.summary

    header(f"BACKTEST · {len(prices)} tickers · {args.period} · entry score ≥ {args.entry_score}")
    rows = [
        ("Total return", f"{summary.get('total_return_pct', 0):+.2f}%"),
        ("Benchmark (equal-weight hold)", f"{summary.get('benchmark_return_pct', 0):+.2f}%"),
        ("Excess vs benchmark", f"{summary.get('excess_return_pct', 0):+.2f}%"),
        ("CAGR", f"{summary.get('cagr_pct', 0):+.2f}%"),
        ("Max drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%"),
        ("Sharpe / Sortino", f"{summary.get('sharpe')} / {summary.get('sortino')}"),
        ("Trades", str(summary.get("total_trades", 0))),
        ("Win rate", f"{summary.get('win_rate', 0):.1f}%"),
        ("Avg win / avg loss", f"{summary.get('avg_win', 0):+.2f}% / {summary.get('avg_loss', 0):+.2f}%"),
        ("Profit factor", str(summary.get("profit_factor"))),
        ("Avg hold", f"{summary.get('avg_hold_bars', 0)} bars"),
        ("Time in market", f"{summary.get('exposure_pct', 0):.0f}%"),
    ]
    for label, value in rows:
        print(f"  {label:<32}{value:>16}")
    if summary.get("exits"):
        print("  " + "  ".join(f"{k}={v}" for k, v in summary["exits"].items()))
    for note in outcome.notes:
        print(f"  ! {note}")
    print_notes(result)

    if outcome.trades:
        print("\n  Last 8 trades")
        print(f"  {'TICKER':<8}{'ENTRY':<12}{'EXIT':<12}{'P&L':>9}{'BARS':>6}  REASON")
        for trade in outcome.trades[-8:]:
            print(
                f"  {trade['ticker']:<8}{trade['entry_date']:<12}{trade['exit_date']:<12}"
                f"{trade['pnl_pct']:>8.2f}%{trade['bars_held']:>6}  {trade['exit_reason']}"
            )
    if args.save:
        _save_backtest(outcome)
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    service = MarketDataService(SETTINGS)
    result = service.scan(preset=args.preset)
    tickers = _split(args.tickers) or [str(r["ticker"]) for r in result.rows[: args.universe]]
    prices = service.price_history(tickers, period=args.period)
    if not prices:
        print("No price history available from any source.")
        return 1
    records = {str(r["ticker"]).upper(): r for r in result.rows}
    signals = build_signals(records, prices)

    header(f"STRATEGY COMPARISON · {len(prices)} tickers · {args.period}")
    print(f"{'STRATEGY':<18}{'RETURN':>9}{'VS BM':>9}{'MAXDD':>9}{'SHARPE':>8}{'TRADES':>8}{'WIN%':>7}{'PF':>7}")
    print(RULE)
    for row in compare_strategies(prices, signals):
        print(
            f"{row['label']:<18}"
            f"{row.get('total_return_pct', 0):>8.2f}%"
            f"{row.get('excess_return_pct', 0):>8.2f}%"
            f"{row.get('max_drawdown_pct', 0):>8.2f}%"
            f"{_fmt(row.get('sharpe')):>8}"
            f"{row.get('total_trades', 0):>8}"
            f"{row.get('win_rate', 0):>7.1f}"
            f"{_fmt(row.get('profit_factor')):>7}"
        )
    print(RULE)
    print("  Same data, same window — only the exit rules differ.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import create_app

    app = create_app()
    port = args.port or int(os.environ.get("PORT", 5000))
    print(f"Dashboard on http://localhost:{port}  (sources: {', '.join(SETTINGS.enabled_sources)})")
    app.run(host=args.host, port=port, debug=args.debug)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    service = service_for(args)
    result = service.scan(preset=args.preset)
    if not result.rows:
        print("Nothing to export.")
        return 1
    paths = _save(result, args.format, latest=True)
    for path in paths:
        print(f"  wrote {path}")
    return 0


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _save(result: Any, fmt: str, latest: bool = False) -> list[str]:
    import pandas as pd

    os.makedirs(SETTINGS.data_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: list[str] = []
    rows = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in result.rows]

    if fmt in {"csv", "both"}:
        path = os.path.join(SETTINGS.data_dir, f"squeeze_scan_{stamp}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)
        if latest:
            latest_path = os.path.join(SETTINGS.data_dir, "latest_scan.csv")
            pd.DataFrame(rows).to_csv(latest_path, index=False)
            written.append(latest_path)
    if fmt in {"json", "both"}:
        path = os.path.join(SETTINGS.data_dir, f"squeeze_scan_{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        written.append(path)
        if latest:
            latest_path = os.path.join(SETTINGS.data_dir, "latest_scan.json")
            with open(latest_path, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, default=str)
            written.append(latest_path)
    return written


def _save_backtest(outcome: Any) -> None:
    import pandas as pd

    os.makedirs(SETTINGS.data_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(outcome.trades).to_csv(
        os.path.join(SETTINGS.data_dir, f"backtest_trades_{stamp}.csv"), index=False
    )
    with open(os.path.join(SETTINGS.data_dir, f"backtest_summary_{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump(outcome.summary, fh, indent=2, default=str)


def _fmt(value: Any) -> str:
    return "–" if value is None else f"{value}"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="squeeze-lab", description="Short Squeeze Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--preset", default=SETTINGS.preset, choices=sorted(WEIGHT_PRESETS), help="weight preset")
        sp.add_argument("--tickers", help="comma-separated ticker list (defaults to the watchlist)")

    scan = sub.add_parser("scan", help="rank the watchlist")
    common(scan)
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--save", action="store_true", help="write the scan to data/")
    scan.add_argument("--format", default="both", choices=["csv", "json", "both"])
    scan.set_defaults(func=cmd_scan)

    sources = sub.add_parser("sources", help="source health and cross-source disagreements")
    common(sources)
    sources.add_argument("--ticker", help="show every source's value for one ticker")
    sources.set_defaults(func=cmd_sources)

    compare = sub.add_parser("compare", help="head-to-head comparison")
    compare.add_argument("symbols", nargs="+")
    compare.add_argument("--preset", default=SETTINGS.preset, choices=sorted(WEIGHT_PRESETS))
    compare.set_defaults(func=cmd_compare)

    presets = sub.add_parser("presets", help="how much the ranking depends on the weights")
    common(presets)
    presets.add_argument("--top", type=int, default=15)
    presets.set_defaults(func=cmd_presets)

    backtest = sub.add_parser("backtest", help="run the strategy over reconstructed signals")
    common(backtest)
    backtest.add_argument("--period", default="1y")
    backtest.add_argument("--universe", type=int, default=15, help="how many top-ranked names to test")
    backtest.add_argument("--entry-score", type=float, default=70.0, dest="entry_score")
    backtest.add_argument("--volume-mult", type=float, default=1.5, dest="volume_mult")
    backtest.add_argument("--hold-days", type=int, default=5, dest="hold_days")
    backtest.add_argument("--take-profit", type=float, default=20.0, dest="take_profit", help="percent")
    backtest.add_argument("--stop-loss", type=float, default=10.0, dest="stop_loss", help="percent")
    backtest.add_argument("--max-positions", type=int, default=5, dest="max_positions")
    backtest.add_argument("--save", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    strategies = sub.add_parser("strategies", help="compare several exit rule sets on the same data")
    common(strategies)
    strategies.add_argument("--period", default="1y")
    strategies.add_argument("--universe", type=int, default=15)
    strategies.set_defaults(func=cmd_strategies)

    serve = sub.add_parser("serve", help="start the dashboard")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--debug", action="store_true")
    serve.set_defaults(func=cmd_serve)

    export = sub.add_parser("export", help="write the latest scan to data/")
    common(export)
    export.add_argument("--format", default="both", choices=["csv", "json", "both"])
    export.set_defaults(func=cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
