#!/usr/bin/env python3
"""Out-of-sample check for the playbook's labelling rule.

Fit (choose) on trades entered 2022-2024, test untouched on 2025-2026.
Candidates: tier group x stop ATR x reward:risk x max hold (27 exit variants x
4 groups = 108 candidates). The in-sample winner (>= MIN_TRADES trades) is then
scored on the test years. Signals and tiers are exactly those in
backtests/playbook_trades.csv.gz (same point-in-time universe and proxy tiers);
each variant re-simulates exits from the cached Yahoo bars.

Writes backtests/playbook_oos.json, which build_static.py embeds.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squeezelab import playbook as pb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "backtests", "playbook_trades.csv.gz")
PRICES = os.path.join(ROOT, "data", "cache", "playbook", "prices.pkl")
OUT = os.path.join(ROOT, "backtests", "playbook_oos.json")
SPLIT = "2025-01-01"
GROUPS = {"all": ("high", "elevated", "watch", "low"), "high+elevated": ("high", "elevated"),
          "watch": ("watch",), "low": ("low",)}
KEYS = ("trades", "win_rate_pct", "win_rate_ci95", "expectancy_pct", "expectancy_r", "profit_factor")


def brief(trades):
    s = pb.summarize(trades)
    return {k: s.get(k) for k in KEYS}


def robust(trades, seed=0):
    """Ticker-clustered bootstrap 95% CI of the mean return, plus per-year means."""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    grp = [g["ret_pct"].to_numpy(float) for _, g in df.groupby("ticker")]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(3000):
        pick = rng.integers(0, len(grp), len(grp))
        a = np.concatenate([grp[i] for i in pick])
        means.append(a.mean())
    years = {y: {"trades": int(len(g)), "win_rate_pct": round(float((g["ret_pct"] > 0).mean() * 100), 1),
                 "expectancy_pct": round(float(g["ret_pct"].mean()), 2)}
             for y, g in df.groupby(df["entry_date"].str[:4])}
    return {"expectancy_ci95_cluster_bootstrap": [round(float(x), 2) for x in np.percentile(means, [2.5, 97.5])], "by_year": years}


def main() -> int:
    led = pd.read_csv(LEDGER)
    prices = pd.read_pickle(PRICES)
    base = pb.PRIMARY_RULE
    sig = {}  # ticker -> [(idx, tier)]
    for t, g in led.groupby("ticker"):
        bars = prices.get(t)
        if bars is None:
            continue
        dates = pd.to_datetime(bars["Date"]).dt.strftime("%Y-%m-%d")
        pos = {d: i for i, d in enumerate(dates)}
        sig[t] = [(pos[d], tier) for d, tier in zip(g["signal_date"], g["tier"]) if d in pos]

    variants = list(itertools.product((1.5, 2.0, 3.0), (1.5, 2.0, 3.0), (10, 15, 20)))
    results = []
    ind_cache = {}
    for stop, rr, hold in variants:
        rule = pb.Rule(**{**base.to_dict(), "stop_atr": stop, "reward_risk": rr, "max_hold": hold})
        by_tier_is: dict[str, list] = {k: [] for k in pb.TIER_ORDER}
        by_tier_oos: dict[str, list] = {k: [] for k in pb.TIER_ORDER}
        for t, items in sig.items():
            if t not in ind_cache:
                ind_cache[t] = pb.add_indicators(prices[t], base)  # ATR(14) is rule-independent here
            tier_of = dict(items)
            for tr in pb.simulate(ind_cache[t], [i for i, _ in items], rule):
                tr["ticker"] = t
                (by_tier_is if tr["entry_date"] < SPLIT else by_tier_oos)[tier_of[tr["signal_idx"]]].append(tr)
        for gname, tiers in GROUPS.items():
            ins = [x for k in tiers for x in by_tier_is[k]]
            oos = [x for k in tiers for x in by_tier_oos[k]]
            results.append({"group": gname, "stop_atr": stop, "reward_risk": rr, "max_hold": hold,
                            "in_sample": brief(ins), "out_of_sample": brief(oos), "_oos": oos})
        print(f"variant stop={stop} rr={rr} hold={hold} done", flush=True)

    eligible = [r for r in results if r["in_sample"]["trades"] >= pb.MIN_TRADES]
    best = max(eligible, key=lambda r: r["in_sample"]["expectancy_pct"])
    current = next(r for r in results if r["group"] == "watch" and (r["stop_atr"], r["reward_risk"], r["max_hold"]) == (2.0, 2.0, 15))
    baseline = next(r for r in results if r["group"] == "all" and (r["stop_atr"], r["reward_risk"], r["max_hold"]) == (2.0, 2.0, 15))
    passed_is = [r for r in eligible if r["in_sample"]["expectancy_pct"] >= pb.MIN_EDGE_PCT]
    passed_both = [r for r in passed_is if (r["out_of_sample"]["trades"] or 0) >= pb.MIN_TRADES
                   and r["out_of_sample"]["expectancy_pct"] >= pb.MIN_EDGE_PCT]
    for r in (best, current, baseline):
        r["out_of_sample_robustness"] = robust(r["_oos"])
    rb = best["out_of_sample_robustness"]
    last_year = max(rb["by_year"]) if rb else None
    checks = {
        "oos_trades_ok": (best["out_of_sample"]["trades"] or 0) >= pb.MIN_TRADES,
        "oos_avg_meets_bar": (best["out_of_sample"]["expectancy_pct"] or -99) >= pb.MIN_EDGE_PCT,
        "oos_ci_excludes_zero": bool(rb) and rb["expectancy_ci95_cluster_bootstrap"][0] > 0,
        "latest_year_not_losing": bool(rb) and rb["by_year"][last_year]["expectancy_pct"] >= 0,
    }
    adopt = all(checks.values())
    for r in results:
        r.pop("_oos", None)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {"fit": "entries 2022-01-14 to 2024-12-31", "test": "entries 2025-01-01 to " + str(led["entry_date"].max())},
        "candidates": len(results),
        "bar": {"min_trades": pb.MIN_TRADES, "min_expectancy_pct": pb.MIN_EDGE_PCT},
        "current_watch_tier": current,
        "all_breakouts": baseline,
        "best_in_sample": best,
        "in_sample_passers": len(passed_is),
        "passers_both_periods": len(passed_both),
        "share_of_is_passers_that_held_up_pct": round(100 * len(passed_both) / len(passed_is), 1) if passed_is else None,
        "adoption_checks": checks,
        "adopted": adopt,
        "decision": ("adopt the in-sample winner for Watch/Buy labels" if adopt else
                     "keep every name on Skip; the best 2022-2024 rule failed: " + ", ".join(
                         {"oos_trades_ok": "too few 2025-26 trades", "oos_avg_meets_bar": "2025-26 average under the bar",
                          "oos_ci_excludes_zero": "its 2025-26 95% range includes zero",
                          "latest_year_not_losing": "it lost money in " + str(last_year)}[k] for k, v in checks.items() if not v)),
        "results": results,
        "note": "The CI and latest-year checks were added after first seeing the by-year split, which makes them stricter, not looser; the pre-set bar alone (>=100 trades, >=+0.5%/trade) would have passed the best candidate. Signals are the ledger's (one position per ticker under the primary rule); exit variants re-simulated independently. Same survivorship and FINRA-vintage caveats as the main backtest.",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: payload[k] for k in ("split", "current_watch_tier", "all_breakouts", "best_in_sample", "in_sample_passers", "passers_both_periods", "adoption_checks", "decision")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
