#!/usr/bin/env python3
"""
Backtest the playbook rule (squeezelab/playbook.py) per squeeze-score tier and
write backtests/playbook_odds.json, which build_static.py reads to put
historical odds next to today's calls.

Universe (chosen point-in-time, to limit hindsight): every exchange-listed
symbol that, on any of the selection dates below, had FINRA days-to-cover >= 5
and average daily volume >= 500k shares. A name only becomes tradable from the
first selection date it qualified on. Delisted names Yahoo no longer serves
drop out, so results carry survivorship bias (they are likely flattered).

    python scripts/backtest_playbook.py               # full run (~10-20 min first time, cached after)
    python scripts/backtest_playbook.py --max-tickers 100   # quick smoke test

Caches live in data/cache/playbook/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squeezelab import playbook as pb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache", "playbook")
OUT = os.path.join(ROOT, "backtests", "playbook_odds.json")
START = "2022-01-14"
PRICE_START = "2021-01-01"
SELECTION_TARGETS = ["2022-01-14", "2022-07-15", "2023-01-13", "2023-07-14", "2024-01-12",
                     "2024-07-15", "2025-01-15", "2025-07-15", "2026-01-15", "2026-07-15"]
LISTED = {"NNM", "NYSE", "AMEX", "ARCA", "SC", "BATS", "NCM", "NGM", "NMS"}


def _download(chunk: list[str]) -> dict[str, pd.DataFrame | None]:
    import yfinance as yf

    out: dict[str, pd.DataFrame | None] = {t: None for t in chunk}
    try:
        raw = yf.download(chunk, start=PRICE_START, auto_adjust=True, group_by="ticker",
                          progress=False, threads=False)
    except Exception as exc:  # noqa: BLE001
        print("  download error", exc, flush=True)
        return out
    for t in chunk:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            out[t] = pb.normalise_bars(sub.dropna(how="all").reset_index())
        except Exception:  # noqa: BLE001
            out[t] = None
    return out


def load_prices(tickers: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    """Daily bars from Yahoo, cached. Yahoo throttles bulk pulls and reports a
    throttled symbol as "possibly delisted", so misses are retried in small,
    paced batches over several passes before a symbol is counted as missing."""
    path = os.path.join(CACHE, "prices.pkl")
    cached: dict[str, pd.DataFrame | None] = pd.read_pickle(path) if os.path.exists(path) else {}
    last_needed = pd.Timestamp.today().normalize() - pd.offsets.BDay(3)
    todo = [t for t in tickers if refresh or cached.get(t) is None
            or cached[t]["Date"].max() < last_needed]
    print(f"prices: {len(tickers) - len(todo)} cached, {len(todo)} to download", flush=True)
    for attempt in range(4):
        if not todo:
            break
        missed = []
        for i in range(0, len(todo), 40):
            chunk = todo[i:i + 40]
            got = _download(chunk)
            for t, df in got.items():
                if df is not None and len(df):
                    cached[t] = df
                else:
                    missed.append(t)
                    cached.setdefault(t, None)
            os.makedirs(CACHE, exist_ok=True)
            pd.to_pickle(cached, path)
            time.sleep(2 + 3 * attempt)
        print(f"  pass {attempt + 1}: {len(todo) - len(missed)} downloaded, {len(missed)} missing", flush=True)
        if len(missed) == len(todo):
            break
        todo = missed
        time.sleep(30)
    return {t: cached.get(t) for t in tickers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--refresh-prices", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    dates = pb.finra_settlement_dates("2021-06-01")
    print(f"FINRA settlement dates: {len(dates)} ({dates[0]} .. {dates[-1]})", flush=True)
    si = pb.load_si_history(dates, os.path.join(CACHE, "finra_si"))
    print(f"FINRA short-interest rows: {len(si):,}", flush=True)

    # Point-in-time universe selection.
    first_ok: dict[str, pd.Timestamp] = {}
    for target in SELECTION_TARGETS:
        cand = [d for d in dates if d <= target]
        if not cand:
            continue
        day = pd.Timestamp(cand[-1])
        snap = si[(si["settlement_date"] == day) & si["market"].isin(LISTED)
                  & (si["short_ratio"] >= 5) & (si["si_adv"] >= 500_000)]
        for t in snap["ticker"]:
            if t not in first_ok:
                first_ok[t] = snap.loc[snap["ticker"] == t, "available"].iloc[0]
    tickers = sorted(t for t in first_ok if t.isascii() and t.replace("-", "").isalpha())
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"universe: {len(tickers)} symbols selected point-in-time", flush=True)

    shares = pd.DataFrame()  # Exclude historical float/share count: filing-availability history is unavailable.
    print(f"SEC shares-outstanding rows: {len(shares):,}", flush=True)
    prices = load_prices(tickers, args.refresh_prices)

    si_by = {t: g.reset_index(drop=True) for t, g in si[si["ticker"].isin(tickers)].groupby("ticker")}
    sh_by = {t: g.reset_index(drop=True) for t, g in shares[shares["ticker"].isin(tickers)].groupby("ticker")} if len(shares) else {}

    rule = pb.PRIMARY_RULE
    all_trades: list[dict] = []
    signals_scored: list[dict] = []  # for the robustness grid
    missing_prices = 0
    for n, t in enumerate(tickers):
        bars = prices.get(t)
        if bars is None or len(bars) < 60:
            missing_prices += 1
            continue
        ind = pb.add_indicators(bars, rule)
        start = max(pd.Timestamp(START), first_ok[t])
        mask = pb.trigger_mask(ind, rule) & (ind["Date"] >= start)
        idx = list(np.flatnonzero(mask.to_numpy()))
        if not idx:
            continue
        tiers = {}
        for i in idx:
            row = ind.iloc[i]
            when = row["Date"]
            si_row = pb.asof(si_by.get(t), when)
            sh_row = pb.asof(sh_by.get(t), when)
            score, tier, comps = pb.tier_score(pb.pit_values(row, si_row, sh_row.get("shares_outstanding") if sh_row else None))
            tiers[i] = (score, tier, comps, si_row is not None, sh_row is not None)
            signals_scored.append({"ticker": t, "i": i, "tier": tier})
        for tr in pb.simulate(ind, idx, rule):
            score, tier, comps, has_si, has_sh = tiers[tr["signal_idx"]]
            tr.update(ticker=t, score=score, tier=tier, has_si=has_si, has_shares=has_sh)
            all_trades.append(tr)
        if n % 200 == 0:
            print(f"  scored {n}/{len(tickers)} tickers, {len(all_trades)} trades", flush=True)

    trades = pd.DataFrame(all_trades)
    print(f"trades: {len(trades)}", flush=True)

    def stats(df: pd.DataFrame) -> dict:
        return pb.summarize(df.to_dict("records"))

    tiers = {k: stats(trades[trades["tier"] == k]) for k in pb.TIER_ORDER}
    combined = {
        "high+elevated": stats(trades[trades["tier"].isin(["high", "elevated"])]),
        "all breakouts (no score filter)": stats(trades),
    }
    years = {}
    for tier in pb.TIER_ORDER:
        sub = trades[trades["tier"] == tier]
        years[tier] = {y: stats(g) for y, g in sub.groupby(sub["entry_date"].str[:4])}
    data_cov = {
        "with_point_in_time_short_interest_pct": round(float(trades["has_si"].mean() * 100), 1) if len(trades) else 0,
        "with_point_in_time_shares_outstanding_pct": round(float(trades["has_shares"].mean() * 100), 1) if len(trades) else 0,
    }

    # Robustness: vary one knob at a time around the primary rule (not used for the odds).
    grid = []
    for label, change in [("hold 5d", {"max_hold": 5}), ("hold 10d", {"max_hold": 10}), ("hold 20d", {"max_hold": 20}),
                          ("target 1.5R", {"reward_risk": 1.5}), ("target 3R", {"reward_risk": 3.0}),
                          ("stop 1.5 ATR", {"stop_atr": 1.5}), ("stop 3 ATR", {"stop_atr": 3.0})]:
        alt = pb.Rule(**{**rule.to_dict(), **change})
        tier_of = {(s["ticker"], s["i"]): s["tier"] for s in signals_scored}
        res: dict[str, list] = {"high+elevated": [], "watch": [], "low": []}
        for t in trades["ticker"].unique():
            ind = pb.add_indicators(prices[t], alt)
            idx = [s["i"] for s in signals_scored if s["ticker"] == t]
            for tr in pb.simulate(ind, idx, alt):
                tier = tier_of[(t, tr["signal_idx"])]
                res["high+elevated" if tier in ("high", "elevated") else tier].append({**tr, "ticker": t})
        grid.append({"variant": label, **{k: {kk: pb.summarize(v).get(kk) for kk in ("trades", "win_rate_pct", "expectancy_pct", "expectancy_r")} for k, v in res.items()}})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": rule.to_dict(),
        "rule_text": [
            "Trigger: daily close above the prior 20-session high on volume >= 2x the prior 20-session average (price >= $1, 20-day avg $ volume >= $2M).",
            "Entry: next session's open. Stop: entry - 2 x ATR(14). Target: entry + 2 x stop distance (2:1). Time exit: close of session 15.",
            "Costs 12 bps per side; gaps through the stop fill at the open; if a bar hits stop and target, the stop is assumed first.",
        ],
        "period": {"from": START, "to": str(trades["exit_date"].max()) if len(trades) else None},
        "universe": {
            "selected": len(tickers),
            "with_prices": len(tickers) - missing_prices,
            "missing_prices": missing_prices,
            "traded": int(trades["ticker"].nunique()) if len(trades) else 0,
            "rule": "exchange-listed symbols with FINRA days-to-cover >= 5 and avg daily volume >= 500k on any of " + ", ".join(SELECTION_TARGETS) + "; tradable only from the first date they qualified",
        },
        "tiers": tiers,
        "tier_labels": pb.TIER_LABELS,
        "combined": combined,
        "by_year": years,
        "robustness": grid,
        "data_coverage": data_cov,
        "caveats": [
            "Survivorship bias: symbols that were delisted, went bankrupt or changed ticker mostly have no Yahoo history, so they are missing (%d of %d selected). Real results were likely worse." % (missing_prices, len(tickers)),
            "FINRA short interest/DTC are lagged 9 business days after settlement; original release vintages are unavailable, so subsequent revisions and exact holiday release lags can bias results. No current short metrics are applied to history.",
            "Odds tiers are a proxy, NOT the full live squeeze score: historical float/SI-density and flow are excluded. Borrow uses FINRA DTC; liquidity uses the same report short shares/volume; momentum uses trailing prices and published short-position changes. Today uses exactly that proxy too.",
            "Max DD is a synthetic realized-trade sequence risking 1% per trade, NOT a mark-to-market investable portfolio; overlapping positions, halts and liquidity constraints are not modelled. Wilson intervals assume independent trades, whereas names/signals are correlated.",
            "Exploratory in-sample research, no independent out-of-sample validation; small positive averages are not evidence of a reliable edge. Paper trading only until validated.",
            "Past results do not predict future ones. Not financial advice.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    trades.drop(columns=["signal_idx"]).to_csv(os.path.join(os.path.dirname(args.out), "playbook_trades.csv.gz"), index=False, compression="gzip")
    print(json.dumps({"tiers": {k: {kk: v.get(kk) for kk in ("trades", "win_rate_pct", "expectancy_pct", "expectancy_r", "max_drawdown_pct_at_1pct_risk")} for k, v in tiers.items()}, "combined": {k: {kk: v.get(kk) for kk in ("trades", "win_rate_pct", "expectancy_pct")} for k, v in combined.items()}}, indent=1))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
