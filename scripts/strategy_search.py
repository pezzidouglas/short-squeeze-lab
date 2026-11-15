#!/usr/bin/env python3
"""Honest multi-family strategy search for the high-short-interest universe.

Families (small, pre-declared parameter grids):
  1 breakout_regime  - 20d-high breakout on 2x volume + market regime + SI-change filter (long)
  2 fade_blowoff     - short a high-SI name after a +30..100% spike in 3-5 days on 3x volume
  3 covering         - days-to-cover falls sharply between FINRA reports while price rises (long)
  4 pullback         - high-DTC name in an uptrend pulls back to its 10/20d average and holds (long)
  5 weekly_topn      - each week hold the top-N names by a score, equal weight (long-only portfolio)

Protocol: every number comes from the same point-in-time data the playbook uses
(universe chosen on fixed FINRA dates, FINRA short interest lagged 9 business
days, Yahoo daily bars). Signals use information up to the signal day's close;
entries are at the next session's open. Costs 12 bps per side; short trades
also pay borrow (base 35%/yr, sensitivity 20% and 50%). Parameters are chosen on
entries 2022-01-14..2024-12-31 only, and the chosen variant is then scored once
on 2025-01-01..latest. Pass bar (trade families): >=100 out-of-sample trades,
OOS average >= +0.5%/trade net, ticker-clustered bootstrap 95% CI above zero,
latest calendar year not negative. Portfolio: OOS Sharpe above SPY's, OOS CAGR
above SPY's, latest year not negative.

    .venv/bin/python scripts/strategy_search.py      # ~3-6 min with warm caches

Writes backtests/strategy_scorecard.json (embedded by build_static.py).
"""
from __future__ import annotations

import itertools
import json
import math
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
BENCH = os.path.join(ROOT, "data", "cache", "strategy", "bench.pkl")
OUT = os.path.join(ROOT, "backtests", "strategy_scorecard.json")
START = "2022-01-14"
SPLIT = "2025-01-01"
COST = 0.0012
BORROW_BASE, BORROW_SENS = 0.35, (0.20, 0.50)
MIN_TRADES, MIN_EDGE = 100, 0.5
SELECTION_TARGETS = ["2022-01-14", "2022-07-15", "2023-01-13", "2023-07-14", "2024-01-12",
                     "2024-07-15", "2025-01-15", "2025-07-15", "2026-01-15", "2026-07-15"]
LISTED = {"NNM", "NYSE", "AMEX", "ARCA", "SC", "BATS", "NCM", "NGM", "NMS"}


# ---------------------------------------------------------------- data

def load_bench(refresh: bool = False) -> dict[str, pd.DataFrame]:
    if os.path.exists(BENCH) and not refresh:
        b = pd.read_pickle(BENCH)
        if b["SPY"]["Date"].max() >= pd.Timestamp.today().normalize() - pd.offsets.BDay(3):
            return b
    import yfinance as yf
    out = {}
    for t in ("SPY", "^VIX"):
        raw = yf.download(t, start="2021-01-01", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        out[t] = pb.normalise_bars(raw.reset_index())
    os.makedirs(os.path.dirname(BENCH), exist_ok=True)
    pd.to_pickle(out, BENCH)
    return out


def load_universe():
    try:
        dates = pb.finra_settlement_dates("2021-06-01")
    except Exception:  # noqa: BLE001 - offline: use what is cached
        dates = sorted(f[:-5] for f in os.listdir(os.path.join(CACHE, "finra_si")) if f.endswith(".json"))
    si = pb.load_si_history(dates, os.path.join(CACHE, "finra_si"))
    si["available"] = si["available"].astype("datetime64[ns]")
    si["settlement_date"] = si["settlement_date"].astype("datetime64[ns]")
    first_ok: dict[str, pd.Timestamp] = {}
    for target in SELECTION_TARGETS:
        cand = [d for d in dates if d <= target]
        if not cand:
            continue
        day = pd.Timestamp(cand[-1])
        snap = si[(si["settlement_date"] == day) & si["market"].isin(LISTED)
                  & (si["short_ratio"] >= 5) & (si["si_adv"] >= 500_000)]
        for t, av in zip(snap["ticker"], snap["available"]):
            first_ok.setdefault(t, av)
    tickers = sorted(t for t in first_ok if t.isascii() and t.replace("-", "").isalpha())
    return si, first_ok, tickers


def prep(bars: pd.DataFrame, si_t: pd.DataFrame | None, first: pd.Timestamp, mkt: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["Date"] = df["Date"].astype("datetime64[ns]")
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    for n in (10, 20, 50, 200):
        df[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
    df["prior_high"] = h.shift(1).rolling(20, min_periods=20).max()
    df["prior_avg_vol"] = v.shift(1).rolling(20, min_periods=20).mean()
    df["dollar_vol"] = (c * v).rolling(20, min_periods=10).mean()
    for k in (3, 5, 10, 63):
        df[f"ret{k}"] = c / c.shift(k) - 1
    df["prev_close"] = pc
    df["avg_volume_10d"] = v.rolling(10, min_periods=5).mean()
    df["roll_high"] = h.rolling(252, min_periods=20).max()
    df["roll_low"] = l.rolling(252, min_periods=20).min()
    cols = ["short_ratio", "shares_short", "shares_short_prior", "si_adv", "available", "dtc_prev"]
    if si_t is not None and len(si_t):
        s = si_t.sort_values("available").copy()
        s["dtc_prev"] = s["short_ratio"].shift(1)
        df = pd.merge_asof(df.sort_values("Date"), s[cols], left_on="Date", right_on="available", direction="backward")
    else:
        for col in cols:
            df[col] = np.nan
    df["si_chg"] = df["shares_short"] / df["shares_short_prior"].replace(0, np.nan) - 1
    df["dtc_chg"] = df["short_ratio"] / df["dtc_prev"].replace(0, np.nan) - 1
    df["new_report"] = df["available"].notna() & (df["available"] != df["available"].shift(1))
    df = df.merge(mkt, on="Date", how="left")
    df["eligible"] = (df["Date"] >= max(pd.Timestamp(START), first)) & (c >= 1) & (df["dollar_vol"] >= 2e6) & df["atr"].notna()
    return df.reset_index(drop=True)


def market_frame(bench) -> pd.DataFrame:
    spy, vix = bench["SPY"].copy(), bench["^VIX"].copy()
    spy["spy_ma200"] = spy["Close"].rolling(200, min_periods=200).mean()
    m = pd.DataFrame({"Date": spy["Date"].astype("datetime64[ns]"), "spy_up": spy["Close"] > spy["spy_ma200"]})
    vix["vix_ok"] = vix["Close"] <= 1.15 * vix["Close"].rolling(20, min_periods=20).mean()
    m = m.merge(pd.DataFrame({"Date": vix["Date"].astype("datetime64[ns]"), "vix_ok": vix["vix_ok"]}), on="Date", how="left")
    return m


# ---------------------------------------------------------------- simulation

def simulate(df: pd.DataFrame, sig: np.ndarray, side: str, stop_atr: float | None, rr: float | None,
             hold: int, borrow: float = 0.0) -> list[dict]:
    """One position per ticker. Entry next open. side 'long'/'short'.
    stop_atr None = no stop; rr None = no profit target (time exit only)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = df["atr"].to_numpy(float)
    dates = df["Date"].to_numpy()
    n = len(df)
    out, busy = [], -1
    for s in np.flatnonzero(sig):
        e = s + 1
        if s <= busy or e >= n or not atr[s] > 0:
            continue
        px = o[e]
        risk = stop_atr * atr[s] if stop_atr else None
        if side == "long":
            stop = px - risk if risk else -np.inf
            tgt = px + rr * risk if (risk and rr) else np.inf
        else:
            stop = px + risk if risk else np.inf
            tgt = px - rr * risk if (risk and rr) else -np.inf
        exit_px = reason = None
        x = e
        for j in range(e, min(n, e + hold)):
            x = j
            if side == "long":
                if j > e and o[j] <= stop: exit_px, reason = o[j], "stop"; break
                if l[j] <= stop: exit_px, reason = stop, "stop"; break
                if j > e and o[j] >= tgt: exit_px, reason = o[j], "target"; break
                if h[j] >= tgt: exit_px, reason = tgt, "target"; break
            else:
                if j > e and o[j] >= stop: exit_px, reason = o[j], "stop"; break
                if h[j] >= stop: exit_px, reason = stop, "stop"; break
                if j > e and o[j] <= tgt: exit_px, reason = o[j], "target"; break
                if l[j] <= tgt: exit_px, reason = tgt, "target"; break
        if reason is None:
            if x - e + 1 < hold and x == n - 1:
                continue  # still open at data end
            exit_px, reason = c[x], "time"
        days = (pd.Timestamp(dates[x]) - pd.Timestamp(dates[e])).days + 1
        if side == "long":
            ret = exit_px * (1 - COST) / (px * (1 + COST)) - 1
        else:
            ret = (px * (1 - COST) - exit_px * (1 + COST)) / px - borrow * days / 365
        out.append({"entry_date": pd.Timestamp(dates[e]).strftime("%Y-%m-%d"),
                    "exit_date": pd.Timestamp(dates[x]).strftime("%Y-%m-%d"),
                    "ret_pct": ret * 100, "days": days, "reason": reason, "side": side,
                    "gross_pct": ((exit_px / px - 1) if side == "long" else (1 - exit_px / px)) * 100})
        busy = x
    return out


def cluster_ci(trades: list[dict], reps: int = 2000, seed: int = 0) -> list[float]:
    if len(trades) < 5:
        return [None, None]
    df = pd.DataFrame(trades)
    grp = [g.to_numpy(float) for _, g in df.groupby("ticker")["ret_pct"]]
    sums = np.array([g.sum() for g in grp]); cnts = np.array([len(g) for g in grp])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(grp), (reps, len(grp)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return [round(float(x), 2) for x in np.percentile(means, [2.5, 97.5])]


def stats(trades: list[dict], spy_ret=None, full: bool = False) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([t["ret_pct"] for t in trades])
    out = {"trades": int(len(r)), "tickers": len({t["ticker"] for t in trades}),
           "win_rate_pct": round(float((r > 0).mean() * 100), 1),
           "avg_pct": round(float(r.mean()), 2), "median_pct": round(float(np.median(r)), 2),
           "t_stat": round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else None}
    if full:
        out["ci95"] = cluster_ci(trades)
        df = pd.DataFrame(trades)
        out["by_year"] = {y: {"trades": int(len(g)), "win_rate_pct": round(float((g["ret_pct"] > 0).mean() * 100), 1),
                              "avg_pct": round(float(g["ret_pct"].mean()), 2)}
                          for y, g in df.groupby(df["entry_date"].str[:4])}
        srt = np.sort(r)[::-1]
        out["avg_ex_top5_pct"] = round(float(srt[5:].mean()), 2) if len(srt) > 10 else None
        out["worst_pct"] = round(float(r.min()), 1)
        out["best_pct"] = round(float(r.max()), 1)
        if spy_ret is not None:
            s = np.array([spy_ret(t["entry_date"], t["exit_date"]) for t in trades])
            out["spy_same_days_avg_pct"] = round(float(np.nanmean(s)), 2)
            sign = np.array([1 if t["side"] == "long" else -1 for t in trades])
            out["avg_vs_spy_pct"] = round(float(np.nanmean(r - sign * s)), 2)
    return out


def bh_stats(rets: pd.Series, periods: int = 52) -> dict:
    """rets: weekly simple returns indexed by date."""
    if rets.empty:
        return {}
    eq = (1 + rets).cumprod()
    yrs = len(rets) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std(ddof=1) * math.sqrt(periods)
    dd = (eq / eq.cummax() - 1).min()
    by_year = {str(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in rets.groupby(rets.index.year)}
    return {"weeks": int(len(rets)), "cagr_pct": round(float(cagr * 100), 1), "vol_pct": round(float(vol * 100), 1),
            "sharpe": round(float(rets.mean() / rets.std(ddof=1) * math.sqrt(periods)), 2) if rets.std() > 0 else None,
            "max_dd_pct": round(float(dd * 100), 1), "total_pct": round(float((eq.iloc[-1] - 1) * 100), 1), "by_year": by_year}


# ---------------------------------------------------------------- families

def fam_breakout(df, p):
    m = (df["Close"] > df["prior_high"]) & (df["Volume"] >= 2 * df["prior_avg_vol"]) & df["eligible"]
    if p["regime"] == "spy>200d":
        m &= df["spy_up"].fillna(False).astype(bool)
    elif p["regime"] == "spy>200d & vix calm":
        m &= df["spy_up"].fillna(False).astype(bool) & df["vix_ok"].fillna(False).astype(bool)
    if p["si"] == "rising":
        m &= df["si_chg"] > 0
    elif p["si"] == "falling":
        m &= df["si_chg"] < 0
    return m.to_numpy(bool)


def fam_fade(df, p):
    m = (df[f"ret{p['window']}"] >= p["spike"]) & (df["Volume"] >= 3 * df["prior_avg_vol"]) & df["eligible"] \
        & (df["short_ratio"] >= 3)
    return m.to_numpy(bool)


def fam_covering(df, p):
    m = df["new_report"] & (df["dtc_prev"] >= 5) & (df["dtc_chg"] <= -p["dtc_drop"]) \
        & (df["ret10"] > p["price_up"]) & df["eligible"]
    if p["shares_short_falling"]:
        m &= df["si_chg"] < 0
    return m.fillna(False).to_numpy(bool)


def fam_pullback(df, p):
    ma = df[f"ma{p['ma']}"]
    m = (df["Close"] > df["ma50"]) & (df["ma50"] > df["ma200"]) & (df["prev_close"] > ma.shift(1)) \
        & (df["Low"] <= ma) & (df["Close"] >= ma) & (df["short_ratio"] >= p["min_dtc"]) & df["eligible"]
    return m.fillna(False).to_numpy(bool)


FAMILIES = {
    "breakout_regime": {
        "title": "Breakout + market filter",
        "plain": "Buy a high-short-interest stock when it closes at a 20-day high on double volume, but only in a healthy market and/or when short sellers are adding.",
        "side": "long", "signal": fam_breakout,
        "grid": [dict(regime=r, si=s, stop_atr=e[0], rr=e[1], hold=e[2]) for r in ("none", "spy>200d", "spy>200d & vix calm")
                 for s in ("any", "rising", "falling") for e in ((2.0, 2.0, 15), (3.0, 3.0, 20))],
    },
    "fade_blowoff": {
        "title": "Short the blow-off spike",
        "plain": "After a heavily shorted stock jumps 30-100% in a few days on huge volume, bet it falls back (short sale, borrow fees charged).",
        "side": "short", "signal": fam_fade,
        "grid": [dict(spike=s, window=w, hold=hd, stop_atr=st, rr=None) for s in (0.3, 0.5, 1.0) for w in (3, 5)
                 for hd in (5, 10, 20) for st in (None, 3.0)],
    },
    "covering": {
        "title": "Short-covering momentum",
        "plain": "Buy when FINRA's new report shows days-to-cover dropped sharply while the price rose - shorts appear to be covering.",
        "side": "long", "signal": fam_covering,
        "grid": [dict(dtc_drop=d, price_up=u, shares_short_falling=f, hold=hd, stop_atr=2.0, rr=None)
                 for d in (0.2, 0.35) for u in (0.0, 0.10) for f in (False, True) for hd in (10, 20)],
    },
    "pullback": {
        "title": "Buy the pullback",
        "plain": "Buy a heavily shorted stock in an uptrend when it dips to its 10- or 20-day average and holds it.",
        "side": "long", "signal": fam_pullback,
        "grid": [dict(ma=m, min_dtc=d, stop_atr=s, rr=2.0, hold=hd) for m in (10, 20) for d in (5, 8)
                 for s in (1.5, 2.0) for hd in (10, 20)],
    },
}


def label(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def run_trade_family(key, spec, frames, spy_ret):
    rows, chosen = [], None
    for p in spec["grid"]:
        trades = []
        for t, df in frames.items():
            sig = spec["signal"](df, p)
            if not sig.any():
                continue
            for tr in simulate(df, sig, spec["side"], p.get("stop_atr"), p.get("rr"), p["hold"],
                               borrow=BORROW_BASE if spec["side"] == "short" else 0.0):
                tr["ticker"] = t
                trades.append(tr)
        ins = [x for x in trades if x["entry_date"] < SPLIT]
        oos = [x for x in trades if x["entry_date"] >= SPLIT]
        rows.append({"params": p, "is": stats(ins), "oos": stats(oos), "_is": ins, "_oos": oos})
    elig = [r for r in rows if r["is"]["trades"] >= MIN_TRADES] or rows
    best = max(elig, key=lambda r: r["is"].get("avg_pct", -1e9))
    best["is"] = stats(best["_is"], spy_ret, full=True)
    best["oos"] = stats(best["_oos"], spy_ret, full=True)
    o = best["oos"]
    latest = max(o.get("by_year") or {"0": None})
    checks = {
        "oos_trades>=100": o["trades"] >= MIN_TRADES,
        "oos_avg>=+0.5%": (o.get("avg_pct") or -99) >= MIN_EDGE,
        "oos_ci_above_0": bool(o.get("ci95")) and o["ci95"][0] is not None and o["ci95"][0] > 0,
        "latest_year_not_negative": bool(o.get("by_year")) and o["by_year"][latest]["avg_pct"] >= 0,
    }
    sens = {}
    if spec["side"] == "short":  # borrow-rate sensitivity for the chosen variant (re-price, same trades)
        for b in BORROW_SENS:
            adj = [x["ret_pct"] + (BORROW_BASE - b) * x["days"] / 365 * 100 for x in best["_oos"]]
            sens[f"oos_avg_at_{int(b*100)}pct_borrow"] = round(float(np.mean(adj)), 2) if adj else None
        g = [x["gross_pct"] for x in best["_oos"]]
        sens["oos_avg_before_costs_and_borrow"] = round(float(np.mean(g)), 2) if g else None
        sens["oos_raw_move_of_spiked_stock_pct"] = round(float(-np.mean(g)), 2) if g else None
    peek = sum(1 for r in rows if r["oos"]["trades"] >= MIN_TRADES and r["oos"].get("avg_pct", -99) >= MIN_EDGE)
    n_var = len(rows)
    result = {
        "key": key, "title": spec["title"], "plain": spec["plain"], "side": spec["side"], "variants_tried": n_var,
        "chosen_params": best["params"], "chosen_label": label(best["params"]),
        "selection": f"highest 2022-24 avg/trade among {len(elig)} variants with >= {MIN_TRADES} in-sample trades" if elig is not rows else "no variant had 100 in-sample trades; highest avg chosen",
        "in_sample": best["is"], "out_of_sample": best["oos"], "checks": checks, "passed": all(checks.values()),
        "borrow_sensitivity": sens or None,
        "multiple_testing": {
            "expected_max_t_under_null": round(math.sqrt(2 * math.log(n_var)), 2) if n_var > 1 else None,
            "chosen_in_sample_t": best["is"].get("t_stat"),
            "variants_that_would_pass_oos_avg_bar_if_peeked": peek,
        },
        "all_variants": [{"params": r["params"], "is": r["is"], "oos": r["oos"]} for r in rows],
    }
    return result, best


def run_portfolio(frames, bench, first_ok):
    spy = bench["SPY"].copy(); spy["Date"] = spy["Date"].astype("datetime64[ns]")
    cal = spy["Date"]
    wk = cal.groupby(cal.dt.to_period("W-FRI")).max()
    sig_dates = [d for d in wk if d >= pd.Timestamp(START)]
    nxt = {d: cal[cal > d].min() for d in sig_dates}
    exec_dates = [nxt[d] for d in sig_dates if pd.notna(nxt[d])]
    sig_dates = sig_dates[:len(exec_dates)]
    opens, feats = {}, {}
    sigset = set(sig_dates)
    for t, df in frames.items():
        s = df.set_index("Date")
        opens[t] = s["Open"]
        w = s.loc[s.index.isin(sigset)]
        if w.empty:
            continue
        ok = w["eligible"] & (w["short_ratio"] >= 5) & (w["Close"] >= 3)
        w = w[ok]
        if w.empty:
            continue
        proxy = []
        for _, row in w.iterrows():
            si = {"shares_short": row["shares_short"], "shares_short_prior": row["shares_short_prior"],
                  "short_ratio": row["short_ratio"], "si_adv": row["si_adv"]}
            proxy.append(pb.tier_score(pb.pit_values(row, si, None))[0])
        feats[t] = pd.DataFrame({"dtc": w["short_ratio"], "si_rising": w["si_chg"], "momentum": w["ret63"],
                                 "proxy": proxy}, index=w.index)
    # Forward-fill so a name that stops trading mid-week exits at its last price
    # (no peeking at whether it still trades next week).
    O_raw = pd.DataFrame(opens).sort_index()
    O = O_raw.ffill().reindex(exec_dates)
    live_at = O_raw.reindex(exec_dates).notna()
    F = pd.concat(feats, names=["ticker", "Date"]).reset_index()
    # forward open-to-open return from exec date k to exec date k+1
    fwd = O.shift(-1) / O - 1
    fwd = fwd.clip(upper=5.0)  # guard against bad split prints (+500%/week cap; counts reported)
    capped = int((O.shift(-1) / O - 1 > 5.0).sum().sum())
    exec_of = dict(zip(sig_dates, exec_dates))
    F["exec"] = F["Date"].map(exec_of)
    spy_o = spy.set_index("Date")["Open"].reindex(exec_dates)
    spy_w = (spy_o.shift(-1) / spy_o - 1).dropna()

    def port(score, n):
        rets, prev = {}, set()
        for d, g in F.groupby("exec"):
            if d not in fwd.index:
                continue
            g = g.dropna(subset=[score])
            r = fwd.loc[d].fillna(0.0)
            live = live_at.loc[d]
            g = g[g["ticker"].map(lambda x: bool(live.get(x, False)))]
            if n:
                g = g.nlargest(n, score)
            names = set(g["ticker"])
            if not names or pd.isna(spy_w.get(d)):
                continue
            # traded notional (sells + buys) as a share of the book, equal weights, drift ignored
            turn = len(names - prev) / len(names) + (len(prev - names) / len(prev) if prev else 0.0)
            rets[d] = float(np.mean([r[x] for x in names])) - COST * turn
            prev = names
        return pd.Series(rets).sort_index()

    grid = [dict(score=s, top_n=n) for s in ("dtc", "si_rising", "momentum", "proxy") for n in (10, 25)]
    rows = []
    for p in grid:
        r = port(p["score"], p["top_n"])
        rows.append({"params": p, "_r": r, "is": bh_stats(r[r.index < SPLIT]), "oos": bh_stats(r[r.index >= SPLIT])})
    F["all"] = 1.0
    ew = port("all", 0)
    best = max(rows, key=lambda x: x["is"].get("sharpe") or -9)
    r = best["_r"]
    ex = (r - spy_w.reindex(r.index)).dropna()
    exo = ex[ex.index >= SPLIT]
    rng = np.random.default_rng(1)
    bmeans = []
    arr = exo.to_numpy()
    for _ in range(2000):  # 4-week block bootstrap of weekly excess return
        idx = np.concatenate([np.arange(s, s + 4) % len(arr) for s in rng.integers(0, len(arr), len(arr) // 4 + 1)])[:len(arr)]
        bmeans.append(arr[idx].mean())
    spy_is, spy_oos = bh_stats(spy_w[spy_w.index < SPLIT]), bh_stats(spy_w[spy_w.index >= SPLIT])
    o = best["oos"]
    latest = max(o["by_year"]) if o.get("by_year") else None
    checks = {
        "oos_sharpe_above_spy": (o.get("sharpe") or -9) > (spy_oos.get("sharpe") or 9),
        "oos_cagr_above_spy": (o.get("cagr_pct") or -99) > (spy_oos.get("cagr_pct") or 99),
        "latest_year_not_negative": bool(latest) and o["by_year"][latest] >= 0,
    }
    return {
        "key": "weekly_topn", "title": "Weekly top-N portfolio",
        "plain": "Every Friday rank heavily shorted stocks (days-to-cover >= 5, price >= $3) by a score, hold the top N equally weighted for the next week.",
        "side": "long", "variants_tried": len(grid), "chosen_params": best["params"], "chosen_label": label(best["params"]),
        "selection": "highest 2022-24 Sharpe", "in_sample": best["is"], "out_of_sample": o,
        "equal_weight_universe": {"in_sample": bh_stats(ew[ew.index < SPLIT]), "out_of_sample": bh_stats(ew[ew.index >= SPLIT])},
        "spy": {"in_sample": spy_is, "out_of_sample": spy_oos},
        "oos_weekly_excess_vs_spy_pct": {"mean": round(float(exo.mean() * 100), 3),
                                         "ci95_block_bootstrap": [round(float(x * 100), 3) for x in np.percentile(bmeans, [2.5, 97.5])]},
        "checks": checks, "passed": all(checks.values()),
        "multiple_testing": {"variants_tried": len(grid),
                             "variants_beating_spy_sharpe_oos_if_peeked": sum(1 for x in rows if (x["oos"].get("sharpe") or -9) > (spy_oos.get("sharpe") or 9))},
        "notes": f"Execution at Monday (next session) open, 12 bps per side on replaced names; weekly name returns capped at +500% ({capped} cases) to guard against bad prints.",
        "all_variants": [{"params": x["params"], "is": x["is"], "oos": x["oos"]} for x in rows],
    }


def todays_signals(scorecard: dict, bars_by_ticker: dict, max_cards: int = 5) -> list[dict]:
    """Signals on the latest bar for every PASSING trade family (none = []).
    bars_by_ticker: today's scan names -> adjusted daily bars. Uses the cached
    FINRA history (same 9-business-day lag) and SPY/VIX regime data."""
    passing = [f for f in scorecard.get("families", []) if f.get("passed") and f["key"] in FAMILIES]
    if not passing or not bars_by_ticker:
        return []
    bench = load_bench()
    mkt = market_frame(bench)
    dates = sorted(f[:-5] for f in os.listdir(os.path.join(CACHE, "finra_si")) if f.endswith(".json"))
    si = pb.load_si_history(dates, os.path.join(CACHE, "finra_si"))
    si["available"] = si["available"].astype("datetime64[ns]")
    si_by = {t: g for t, g in si[si["ticker"].isin(list(bars_by_ticker))].groupby("ticker")}
    out = []
    for fam in passing:
        spec, p = FAMILIES[fam["key"]], fam["chosen_params"]
        for t, bars in bars_by_ticker.items():
            b = pb.normalise_bars(bars) if bars is not None else None
            if b is None or len(b) < 210:
                continue
            df = prep(b, si_by.get(t), pd.Timestamp("1900-01-01"), mkt)
            df["eligible"] = df["eligible"] & (df["Date"] >= pd.Timestamp("1900-01-02"))
            sig = spec["signal"](df, p)
            if not sig[-1]:
                continue
            last = df.iloc[-1]
            ref, atr = float(last["Close"]), float(last["atr"])
            k = p.get("stop_atr")
            short = spec["side"] == "short"
            stop = (ref + k * atr if short else ref - k * atr) if k else None
            tgt = (ref - p["rr"] * k * atr if short else ref + p["rr"] * k * atr) if (k and p.get("rr")) else None
            o = fam["out_of_sample"]
            out.append({"family": fam["key"], "title": fam["title"], "ticker": t, "side": spec["side"],
                        "as_of": last["Date"].strftime("%Y-%m-%d"), "close": round(ref, 4),
                        "stop": round(stop, 4) if stop else None, "target": round(tgt, 4) if tgt else None,
                        "hold_days": p["hold"], "oos_win_rate_pct": o.get("win_rate_pct"), "oos_avg_pct": o.get("avg_pct"),
                        "oos_trades": o.get("trades")})
    return out[: max_cards * max(1, len(passing))]


def main() -> int:
    t0 = time.time()
    bench = load_bench()
    si, first_ok, tickers = load_universe()
    prices = pd.read_pickle(os.path.join(CACHE, "prices.pkl"))
    mkt = market_frame(bench)
    si_by = {t: g for t, g in si[si["ticker"].isin(tickers)].groupby("ticker")}
    frames = {}
    for t in tickers:
        b = prices.get(t)
        if b is None or len(b) < 60:
            continue
        frames[t] = prep(b, si_by.get(t), first_ok[t], mkt)
    print(f"prepared {len(frames)} tickers in {time.time()-t0:.0f}s", flush=True)

    spy = bench["SPY"].copy(); spy["Date"] = spy["Date"].dt.strftime("%Y-%m-%d")
    so = dict(zip(spy["Date"], spy["Open"])); sc = dict(zip(spy["Date"], spy["Close"]))

    def spy_ret(a, b):
        return (sc[b] / so[a] - 1) * 100 if a in so and b in sc else np.nan

    fams, best_rows = [], {}
    for key, spec in FAMILIES.items():
        res, best = run_trade_family(key, spec, frames, spy_ret)
        fams.append(res); best_rows[key] = best
        print(f"{key}: {res['chosen_label']} IS {res['in_sample'].get('trades')}/{res['in_sample'].get('avg_pct')} "
              f"OOS {res['out_of_sample'].get('trades')}/{res['out_of_sample'].get('avg_pct')} ci {res['out_of_sample'].get('ci95')} "
              f"passed={res['passed']} ({time.time()-t0:.0f}s)", flush=True)
    port = run_portfolio(frames, bench, first_ok)
    fams.append(port)
    print(f"weekly_topn: {port['chosen_label']} OOS {port['out_of_sample']} SPY {port['spy']['out_of_sample']} passed={port['passed']}", flush=True)

    # SPY buy-and-hold, daily, same split
    s = bench["SPY"].set_index("Date")["Close"]
    def bh(a, b=None):
        x = s[(s.index >= a) & ((s.index < b) if b else True)]
        d = x.pct_change().dropna()
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        return {"from": str(x.index[0].date()), "to": str(x.index[-1].date()), "total_pct": round(float((x.iloc[-1] / x.iloc[0] - 1) * 100), 1),
                "cagr_pct": round(float(((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1) * 100), 1),
                "sharpe": round(float(d.mean() / d.std() * math.sqrt(252)), 2), "max_dd_pct": round(float(((x / x.cummax()) - 1).min() * 100), 1)}

    total_variants = sum(f["variants_tried"] for f in fams)
    passing = [f["key"] for f in fams if f["passed"]]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": str(max(df["Date"].max() for df in frames.values()).date()),
        "split": {"fit": f"{START} to 2024-12-31", "test": f"{SPLIT} to latest"},
        "universe": {"tickers_with_prices": len(frames), "selected": len(tickers)},
        "costs": {"per_side_bps": 12, "short_borrow_annual_base_pct": int(BORROW_BASE * 100),
                  "short_borrow_sensitivity_pct": [int(b * 100) for b in BORROW_SENS]},
        "pass_bar": {"trade_families": f">= {MIN_TRADES} OOS trades, OOS avg >= +{MIN_EDGE}%/trade net, ticker-clustered bootstrap 95% CI above 0, latest year not negative",
                     "portfolio": "OOS Sharpe and CAGR above SPY buy-and-hold, latest year not negative"},
        "total_variants_tried": total_variants,
        "families": fams,
        "passing": passing,
        "spy_buy_hold": {"in_sample": bh(START, SPLIT), "out_of_sample": bh(SPLIT)},
        "verdict": ("no squeeze strategy passed" if not passing else "passed: " + ", ".join(passing)),
        "caveats": [
            "Survivorship bias: delisted/bankrupt names mostly have no Yahoo history, so long results are flattered and short results understated for the worst blow-ups.",
            "FINRA short interest is lagged 9 business days; revisions and holiday release dates are not modelled exactly.",
            f"{total_variants} variants were tried across 5 families. Only the in-sample winner per family was scored out of sample; with this many tries some in-sample winners will look good by chance.",
            "Short trades assume shares are always available to borrow at 35%/yr; real hard-to-borrow names can cost far more, be recalled, or be unborrowable, and losses on shorts are unlimited (gaps through stops).",
            "Trades are independent per ticker (one position at a time) and do not model portfolio capital limits, halts, or slippage beyond 12 bps.",
            "Past results do not predict future ones. Not financial advice.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"verdict: {payload['verdict']}; {total_variants} variants; wrote {OUT} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
