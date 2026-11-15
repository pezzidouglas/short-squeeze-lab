#!/usr/bin/env python3
"""Short-side strategy research on point-in-time short-interest data.

Strategies (each with a small, pre-declared parameter grid):
  a short_basket   - each month short the most-shorted names (top decile/quintile by
                     days-to-cover or short interest % of shares), equal weight, 100% short
  b long_short     - long the least-shorted, short the most-shorted (50% / 50% of account)
  c failed_squeeze - trade-level: short a heavily shorted stock after a spike to a new high
                     fails (closes well below the peak and under its 20-day average)
  d hedged         - short the most-shorted basket (50%) + long SPY (50%), market-neutral-ish

Data/no-lookahead rules: FINRA short interest is used only from 9 business days after
its settlement date; SEC cover-page shares outstanding only from 60 days after the
period end; prices are Yahoo daily bars. Signals are formed on the last trading day of
each month at the close; trades happen at the next session's open and are held to the
next rebalance open. Costs: 12 bps per side on traded weight. Short positions pay an
annual borrow fee by tier: 5% (DTC < 5 and SI% < 10), 20% (DTC 5-10 or SI% 10-20),
50% (DTC >= 10 or SI% >= 20); sensitivity at half and double. No interest is credited
on cash or short proceeds (conservative). Universe: US-listed operating companies
(SEC ticker list) with price >= $5 and 20-day average dollar volume >= $5M.

Parameters are chosen on 2022-01..2024-12 (highest in-sample Sharpe) and scored once on
2025-01..latest. Pass bar: OOS net return positive after borrow, OOS Sharpe > 0.5
(hedged: OOS Sharpe also above SPY's), latest calendar year not negative.

    .venv/bin/python scripts/short_search.py

Writes backtests/short_scorecard.json (embedded by build_static.py).
"""
from __future__ import annotations

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
EXTRA = os.path.join(ROOT, "data", "cache", "short", "prices_extra.pkl")
BENCH = os.path.join(ROOT, "data", "cache", "strategy", "bench.pkl")
OUT = os.path.join(ROOT, "backtests", "short_scorecard.json")
START, SPLIT = "2022-01-01", "2025-01-01"
COST = 0.0012
LISTED = {"NNM", "NYSE", "AMEX", "SC", "BATS", "NCM", "NGM", "NMS"}
MIN_PRICE, MIN_DV = 5.0, 5e6
SH_LAG_DAYS = 60
FEE_MULTS = (0.5, 1.0, 2.0)


def fee_tier(dtc: np.ndarray, sip: np.ndarray) -> np.ndarray:
    sip = np.nan_to_num(sip, nan=0.0)
    f = np.full(dtc.shape, 0.05)
    f[(dtc >= 5) | (sip >= 0.10)] = 0.20
    f[(dtc >= 10) | (sip >= 0.20)] = 0.50
    return f


# ---------------------------------------------------------------- data

def refresh_extra(max_age_bdays: int = 3) -> None:
    """Incrementally refresh the extra (non-squeeze-universe) Yahoo bars when stale."""
    if not os.path.exists(EXTRA):
        return
    cached = pd.read_pickle(EXTRA)
    need = pd.Timestamp.today().normalize() - pd.offsets.BDay(max_age_bdays)
    todo = [t for t, df in cached.items() if df is not None and df["Date"].max() < need
            and df["Date"].max() >= need - pd.Timedelta(days=30)]  # skip long-dead symbols
    if not todo:
        return
    import yfinance as yf
    print(f"refreshing {len(todo)} extra price series", flush=True)
    for i in range(0, len(todo), 80):
        ch = todo[i:i + 80]
        try:
            raw = yf.download(ch, start="2021-01-01", auto_adjust=True, group_by="ticker", progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001 - keep the old bars
            print("  download error", exc, flush=True)
            continue
        for t in ch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = pb.normalise_bars(sub.dropna(how="all").reset_index())
                if df is not None and len(df) and df["Date"].max() >= cached[t]["Date"].max():
                    cached[t] = df
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
    pd.to_pickle(cached, EXTRA)


def load_all():
    t0 = time.time()
    refresh_extra()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import strategy_search  # noqa: PLC0415 - shared SPY/VIX cache (refreshes when stale)
    bench = strategy_search.load_bench()
    spy = bench["SPY"].copy(); spy["Date"] = pd.to_datetime(spy["Date"]).astype("datetime64[ns]")
    cal = pd.DatetimeIndex(spy["Date"])
    prices = dict(pd.read_pickle(os.path.join(CACHE, "prices.pkl")))
    if os.path.exists(EXTRA):
        for t, df in pd.read_pickle(EXTRA).items():
            if prices.get(t) is None and df is not None:
                prices[t] = df
    ct = json.load(open(os.path.join(CACHE, "sec", "company_tickers.json")))
    norm = lambda s: s.upper().replace(".", "-").replace("/", "-")  # noqa: E731
    tick2cik = {norm(v["ticker"]): int(v["cik_str"]) for v in ct.values()}
    dates = sorted(f[:-5] for f in os.listdir(os.path.join(CACHE, "finra_si")) if f[0] == "2")
    si = pb.load_si_history(dates, os.path.join(CACHE, "finra_si"))
    si = si[si["market"].isin(LISTED) & si["ticker"].isin(tick2cik)]
    keep = sorted(t for t in si["ticker"].unique() if prices.get(t) is not None and len(prices[t]) > 60)
    si = si[si["ticker"].isin(keep)]
    print(f"tickers with SI+prices: {len(keep)}", flush=True)

    def wide(col):
        return pd.DataFrame({t: prices[t].assign(Date=pd.to_datetime(prices[t]["Date"])).set_index("Date")[col]
                             for t in keep}).reindex(cal)
    O, C, V = wide("Open"), wide("Close"), wide("Volume")
    # shares outstanding (SEC cover page frames), usable SH_LAG_DAYS after period end
    rows = []
    for f in os.listdir(os.path.join(CACHE, "sec")):
        if not f.startswith("CY"):
            continue
        d = json.load(open(os.path.join(CACHE, "sec", f)))
        for r in d.get("data", []):
            rows.append((r["cik"], pd.Timestamp(r["end"]) + pd.Timedelta(days=SH_LAG_DAYS), float(r["val"])))
    sh = pd.DataFrame(rows, columns=["cik", "avail", "shares"]).sort_values("avail")
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    return bench, spy, cal, O, C, V, si, sh, tick2cik, keep


def month_ends(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(cal, index=cal)
    me = s.groupby(cal.to_period("M")).max()
    return [d for d in me if d >= pd.Timestamp("2021-12-01")]


def build_panel(cal, O, C, V, si, sh, tick2cik, keep):
    """One row per (signal month-end, ticker) with point-in-time features + next-period return."""
    sigs = month_ends(cal)
    nxt = {d: cal[cal.get_loc(d) + 1] if cal.get_loc(d) + 1 < len(cal) else None for d in sigs}
    dv = (C * V).rolling(20, min_periods=10).mean()
    ma200 = C.rolling(200, min_periods=200).mean()
    mom = C.shift(21) / C.shift(252) - 1
    mom3 = C / C.shift(63) - 1
    si_av = si.sort_values("available")
    sh_by = {c: g for c, g in sh.groupby("cik")}
    out = []
    for k, d in enumerate(sigs):
        e = nxt[d]
        if e is None:
            break
        e2 = nxt[sigs[k + 1]] if k + 1 < len(sigs) and nxt.get(sigs[k + 1]) is not None else None
        rep = si_av[(si_av["available"] <= d) & (si_av["settlement_date"] >= d - pd.Timedelta(days=35))]
        rep = rep.sort_values("available").drop_duplicates("ticker", keep="last").set_index("ticker")
        tk = [t for t in rep.index if t in C.columns]
        df = pd.DataFrame(index=tk)
        df["dtc"] = rep.loc[tk, "short_ratio"].astype(float)
        df["shares_short"] = rep.loc[tk, "shares_short"].astype(float)
        df["close"] = C.loc[d, tk]; df["dv"] = dv.loc[d, tk]
        df["ma200"] = ma200.loc[d, tk]; df["mom12_1"] = mom.loc[d, tk]; df["mom3"] = mom3.loc[d, tk]
        so = []
        for t in tk:
            g = sh_by.get(tick2cik.get(t))
            v = np.nan
            if g is not None:
                g2 = g[(g["avail"] <= d) & (g["avail"] >= d - pd.Timedelta(days=200))]
                if len(g2):
                    v = g2["shares"].iloc[-1]
            so.append(v)
        df["shares_out"] = so
        df["si_pct"] = df["shares_short"] / df["shares_out"]
        df.loc[(df["si_pct"] > 1.5) | (df["si_pct"] <= 0), "si_pct"] = np.nan  # bad share counts
        o1 = O.loc[e, tk]
        if e2 is not None:
            o2 = O.loc[e2, tk]
            # delisted / missing at next open: use last close available before e2
            last_c = C.loc[e:e2, tk].ffill().iloc[-1]
            o2 = o2.fillna(last_c)
            df["ret"] = o2 / o1 - 1
        else:  # current, open period: no realised return
            df["ret"] = np.nan
        df["signal_date"] = d; df["exec_date"] = e; df["exit_date"] = e2
        df = df[(df["close"] >= MIN_PRICE) & (df["dv"] >= MIN_DV) & o1.notna()]
        out.append(df.reset_index().rename(columns={"index": "ticker"}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- portfolio construction

def pick(month: pd.DataFrame, p: dict, side: str) -> pd.DataFrame:
    col = p["signal"]
    m = month.dropna(subset=[col])
    if len(m) < 50:
        return m.iloc[0:0]
    q = 0.10 if p["bucket"] == "decile" else 0.20
    if side == "short":
        cut = m[col].quantile(1 - q)
        s = m[m[col] >= cut]
        if p["filter"] == "neg_mom12_1":
            s = s[s["mom12_1"] < 0]
        elif p["filter"] == "below_200d":
            s = s[s["close"] < s["ma200"]]
        return s
    cut = m[col].quantile(q)
    return m[m[col] <= cut]


def series_for(panel: pd.DataFrame, p: dict, kind: str, spy_m: pd.Series, fee_mult: float = 1.0, flat_fee: float | None = None):
    """Monthly net returns (index = exec date). kind: short_basket | long_short | hedged."""
    rets, prev_s, prev_l, names_hist, n_short = {}, {}, {}, {}, {}
    avg_fee = series_for.avg_fee = []
    for d, month in panel.groupby("exec_date", sort=True):
        month = month.set_index("ticker")
        realised = month["ret"].notna().any()
        s = pick(month.reset_index(), p, "short").set_index("ticker") if True else None
        if len(s) < 5:
            if not realised:
                names_hist[d] = []
            rets[d] = 0.0 if realised else np.nan
            prev_s = {}
            continue
        ws = {t: 1 / len(s) for t in s.index}
        fee = fee_tier(s["dtc"].to_numpy(float), s["si_pct"].to_numpy(float)) * fee_mult
        if flat_fee is not None:
            fee = np.full(len(s), flat_fee)
        avg_fee.append(float(np.mean(fee)))
        names_hist[d] = list(s.index); n_short[d] = len(s)
        if not realised:
            continue
        days = (pd.Timestamp(s["exit_date"].iloc[0]) - pd.Timestamp(d)).days
        r_s = s["ret"].fillna(0).to_numpy(float)
        short_ret = float(np.mean(-r_s - fee * days / 365))
        turn_s = sum(abs(ws.get(t, 0) - prev_s.get(t, 0)) for t in set(ws) | set(prev_s))
        prev_s = ws
        if kind == "short_basket":
            rets[d] = short_ret - COST * turn_s
        elif kind == "hedged":
            rets[d] = 0.5 * short_ret + 0.5 * float(spy_m.get(d, 0.0)) - COST * 0.5 * turn_s
        else:
            lg = pick(month.reset_index(), p, "long").set_index("ticker")
            wl = {t: 1 / len(lg) for t in lg.index}
            turn_l = sum(abs(wl.get(t, 0) - prev_l.get(t, 0)) for t in set(wl) | set(prev_l))
            prev_l = wl
            long_ret = float(lg["ret"].fillna(0).mean())
            rets[d] = 0.5 * long_ret + 0.5 * short_ret - COST * 0.5 * (turn_s + turn_l)
    r = pd.Series(rets).sort_index()
    return r.dropna(), names_hist, n_short


def mstats(r: pd.Series) -> dict:
    if len(r) < 3:
        return {"months": int(len(r))}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    return {"months": int(len(r)), "cagr_pct": round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 1),
            "total_pct": round(float((eq.iloc[-1] - 1) * 100), 1),
            "avg_month_pct": round(float(r.mean() * 100), 2),
            "vol_pct": round(float(r.std(ddof=1) * math.sqrt(12) * 100), 1),
            "sharpe": round(float(r.mean() / r.std(ddof=1) * math.sqrt(12)), 2) if r.std() > 0 else None,
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "win_months_pct": round(float((r > 0).mean() * 100), 0),
            "worst_month_pct": round(float(r.min() * 100), 1),
            "by_year": {str(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in r.groupby(r.index.year)}}


# ---------------------------------------------------------------- failed squeeze (trade level)

def failed_squeeze(cal, O, C, V, si, spy_close, grid):
    """Short after a failed spike: within the last 10 sessions the stock closed at a 60-day
    high after rising >= `rise` from its 20-day low; today it closes >= `drop` below that peak
    and under its 20-day average; latest FINRA DTC >= `min_dtc`. Entry next open; exit at
    stop (k x ATR above entry), or after `hold` sessions."""
    H = C  # close-based to keep it simple and robust to bad intraday prints
    hi60 = C.rolling(60, min_periods=60).max()
    lo20 = C.rolling(20, min_periods=20).min()
    ma20 = C.rolling(20, min_periods=20).mean()
    dv = (C * V).rolling(20, min_periods=10).mean()
    atr = (C.diff().abs()).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    at_high = (C >= hi60) & (C / lo20.shift(1) - 1 >= 0.0)
    rise20 = C / lo20 - 1
    si_av = si.sort_values("available")
    dtc_w = si_av.pivot_table(index="available", columns="ticker", values="short_ratio", aggfunc="last")
    dtc_w = dtc_w.reindex(dtc_w.index.union(cal)).ffill().reindex(cal)
    dtc_w = dtc_w.reindex(columns=C.columns)
    peak_close = C.where(at_high).ffill(limit=10)
    peak_rise = rise20.where(at_high).ffill(limit=10)
    elig = (C >= MIN_PRICE) & (dv >= MIN_DV) & (pd.Series(cal, index=cal) >= pd.Timestamp(START)).to_numpy()[:, None]
    results = []
    Oa, Ca = O.to_numpy(float), C.to_numpy(float)
    for p in grid:
        sig = (elig & (peak_rise >= p["rise"]) & (C <= peak_close * (1 - p["drop"])) & (C < ma20)
               & (dtc_w >= p["min_dtc"])).to_numpy(bool)
        dtc_a, atr_a = dtc_w.to_numpy(float), atr.to_numpy(float)
        trades = []
        for j, t in enumerate(C.columns):
            busy = -1
            for i in np.flatnonzero(sig[:, j]):
                e = i + 1
                if i <= busy or e >= len(cal) or not np.isfinite(Oa[e, j]):
                    continue
                px = Oa[e, j]; stop = px + p["stop_atr"] * atr_a[i, j]
                x, exit_px = e, None
                for k in range(e, min(len(cal), e + p["hold"])):
                    x = k
                    if not np.isfinite(Ca[k, j]):
                        break
                    if Ca[k, j] >= stop:  # close-based stop, filled at that close (optimistic for gaps)
                        exit_px = Ca[k, j]; break
                if exit_px is None:
                    if x - e + 1 < p["hold"] and x >= len(cal) - 1:
                        continue
                    exit_px = Ca[x, j] if np.isfinite(Ca[x, j]) else px
                days = (cal[x] - cal[e]).days + 1
                fee = fee_tier(np.array([dtc_a[i, j]]), np.array([np.nan]))[0]
                gross = 1 - exit_px / px
                trades.append({"ticker": t, "entry_date": cal[e], "exit_date": cal[x], "gross": gross,
                               "days": days, "fee": fee,
                               "ret": (px * (1 - COST) - exit_px * (1 + COST)) / px - fee * days / 365})
                busy = x
        results.append((p, pd.DataFrame(trades)))
    return results


def trades_to_monthly(tr: pd.DataFrame, slots: int = 10, fee_mult: float = 1.0) -> pd.Series:
    """Portfolio view: each trade uses 1/slots of the account (max `slots` open at once,
    first-come); P&L booked in the exit month."""
    if tr.empty:
        return pd.Series(dtype=float)
    tr = tr.sort_values("entry_date").copy()
    tr["ret_adj"] = tr["ret"] - (fee_mult - 1) * tr["fee"] * tr["days"] / 365
    open_exits, taken = [], []
    for _, row in tr.iterrows():
        open_exits = [x for x in open_exits if x >= row["entry_date"]]
        if len(open_exits) < slots:
            open_exits.append(row["exit_date"]); taken.append(row)
    t = pd.DataFrame(taken)
    m = t.groupby(t["exit_date"].dt.to_period("M"))["ret_adj"].sum() / slots
    idx = pd.period_range(pd.Timestamp(START), t["exit_date"].max(), freq="M")
    m = m.reindex(idx, fill_value=0.0)
    m.index = m.index.to_timestamp()
    return m


# ---------------------------------------------------------------- main

GRID = [dict(signal=s, bucket=b, filter=f) for s in ("dtc", "si_pct") for b in ("decile", "quintile")
        for f in ("none", "neg_mom12_1", "below_200d")]
FS_GRID = [dict(rise=r, drop=d, min_dtc=m, stop_atr=s, hold=h) for r in (0.3, 0.6) for d in (0.15, 0.25)
           for m in (5, 8) for s in (2.0, 4.0) for h in (10, 20)]

STRATS = {
    "short_basket": {"title": "Short the most-shorted stocks (monthly basket)",
                     "plain": "Each month, short the stocks with the highest days-to-cover (or short % of shares), equal weight. Optional filter: only names already falling (negative 12-month momentum or below the 200-day average)."},
    "long_short": {"title": "Long least-shorted / short most-shorted",
                   "plain": "Each month, 50% of the account long the least-shorted stocks and 50% short the most-shorted ones (the classic days-to-cover spread)."},
    "hedged": {"title": "Short most-shorted basket, hedged with SPY",
               "plain": "Each month, 50% short the most-shorted basket and 50% long SPY, so a rising market does not sink the shorts."},
    "failed_squeeze": {"title": "Short the failed squeeze",
                       "plain": "Short a heavily shorted stock after it spikes to a new 60-day high and then breaks down: closes 15-25% below the peak and under its 20-day average. Stop above entry, time exit."},
}


def check(o: dict, kind: str, spy_oos: dict) -> dict:
    latest = max(o["by_year"]) if o.get("by_year") else None
    c = {"oos_net_return_positive": (o.get("total_pct") or -1) > 0,
         "oos_sharpe>0.5": (o.get("sharpe") or -9) > 0.5,
         "latest_year_not_negative": bool(latest) and o["by_year"][latest] >= 0}
    if kind == "hedged":
        c["oos_sharpe_above_spy"] = (o.get("sharpe") or -9) > (spy_oos.get("sharpe") or 9)
    return c


def main() -> int:
    t0 = time.time()
    bench, spy, cal, O, C, V, si, sh, tick2cik, keep = load_all()
    panel = build_panel(cal, O, C, V, si, sh, tick2cik, keep)
    print(f"panel {len(panel):,} rows, {panel['exec_date'].nunique()} months, si_pct coverage "
          f"{panel['si_pct'].notna().mean():.0%} ({time.time()-t0:.0f}s)", flush=True)
    so = spy.set_index("Date")["Open"]
    execs = sorted(panel["exec_date"].unique())
    spy_m = pd.Series({pd.Timestamp(a): so[b] / so[a] - 1 for a, b in zip(execs[:-1], execs[1:])})
    spy_is, spy_oos = mstats(spy_m[spy_m.index < SPLIT]), mstats(spy_m[spy_m.index >= SPLIT])
    fams, total = [], 0
    current_names = {}
    for kind in ("short_basket", "long_short", "hedged"):
        rows = []
        for p in GRID:
            r, names, nshort = series_for(panel, p, kind, spy_m)
            rows.append({"params": p, "r": r, "names": names, "n": nshort,
                         "is": mstats(r[r.index < SPLIT]), "oos": mstats(r[r.index >= SPLIT])})
        total += len(rows)
        best = max(rows, key=lambda x: x["is"].get("sharpe") or -9)
        o = best["oos"]
        sens = {}
        for fm in FEE_MULTS:
            rr, _, _ = series_for(panel, best["params"], kind, spy_m, fee_mult=fm)
            sens[f"borrow_x{fm:g}"] = {"is": mstats(rr[rr.index < SPLIT]), "oos": mstats(rr[rr.index >= SPLIT])}
        rr0, _, _ = series_for(panel, best["params"], kind, spy_m, fee_mult=0.0)
        sens["no_borrow_fee"] = {"is": mstats(rr0[rr0.index < SPLIT]), "oos": mstats(rr0[rr0.index >= SPLIT])}
        series_for(panel, best["params"], kind, spy_m)
        avg_tier_fee = round(float(np.mean(series_for.avg_fee)) * 100, 1)
        flat = {}
        for ff in (0.01, 0.03, 0.05, 0.10):
            rf, _, _ = series_for(panel, best["params"], kind, spy_m, flat_fee=ff)
            of = mstats(rf[rf.index >= SPLIT])
            flat[f"flat_{int(ff*100)}pct"] = {"is": mstats(rf[rf.index < SPLIT]), "oos": of,
                                             "would_pass": all(check(of, kind, spy_oos).values())}
        sens["flat_fee_what_if"] = flat
        sens["avg_assumed_tier_fee_pct"] = avg_tier_fee
        checks = check(o, kind, spy_oos)
        last_d = max(best["names"]) if best["names"] else None
        current_names[kind] = (best["params"], last_d, best["names"].get(last_d, []))
        fams.append({"key": kind, **STRATS[kind], "variants_tried": len(rows), "chosen_params": best["params"],
                     "selection": "highest 2022-24 Sharpe", "in_sample": best["is"], "out_of_sample": o,
                     "avg_names_short": round(float(np.mean(list(best["n"].values()))), 0) if best["n"] else 0,
                     "borrow_sensitivity": sens, "checks": checks, "passed": all(checks.values()),
                     "variants_passing_if_peeked": sum(1 for x in rows if all(check(x["oos"], kind, spy_oos).values())),
                     "monthly_returns_pct": {str(k.date()): round(float(v * 100), 2) for k, v in best["r"].items()},
                     "all_variants": [{"params": x["params"], "is": x["is"], "oos": x["oos"]} for x in rows]})
        print(f"{kind}: {best['params']} IS {best['is'].get('sharpe')}/{best['is'].get('cagr_pct')} "
              f"OOS {o.get('sharpe')}/{o.get('cagr_pct')} {o.get('by_year')} passed={all(checks.values())} "
              f"({time.time()-t0:.0f}s)", flush=True)
    # failed squeeze
    fs = failed_squeeze(cal, O, C, V, si, None, FS_GRID)
    total += len(fs)
    rows = []
    for p, tr in fs:
        m = trades_to_monthly(tr)
        rows.append({"params": p, "tr": tr, "m": m, "is": mstats(m[m.index < SPLIT]), "oos": mstats(m[m.index >= SPLIT]),
                     "n_is": int((tr["entry_date"] < SPLIT).sum()) if len(tr) else 0})
    elig = [x for x in rows if x["n_is"] >= 50] or rows
    best = max(elig, key=lambda x: x["is"].get("sharpe") or -9)
    tr = best["tr"]
    tro = tr[tr["entry_date"] >= SPLIT] if len(tr) else tr
    tri = tr[tr["entry_date"] < SPLIT] if len(tr) else tr
    def tstats(t):
        if t is None or len(t) == 0:
            return {"trades": 0}
        return {"trades": int(len(t)), "win_rate_pct": round(float((t["ret"] > 0).mean() * 100), 1),
                "avg_pct": round(float(t["ret"].mean() * 100), 2), "median_pct": round(float(t["ret"].median() * 100), 2),
                "worst_pct": round(float(t["ret"].min() * 100), 1), "avg_hold_days": round(float(t["days"].mean()), 1)}
    sens = {}
    for fm in FEE_MULTS + (0.0,):
        mm = trades_to_monthly(tr, fee_mult=fm)
        sens[f"borrow_x{fm:g}" if fm else "no_borrow_fee"] = {"is": mstats(mm[mm.index < SPLIT]), "oos": mstats(mm[mm.index >= SPLIT])}
    o = best["oos"]
    checks = check(o, "failed_squeeze", spy_oos)
    fams.append({"key": "failed_squeeze", **STRATS["failed_squeeze"], "variants_tried": len(rows),
                 "chosen_params": best["params"], "selection": "highest 2022-24 Sharpe (>= 50 in-sample trades)",
                 "in_sample": best["is"], "out_of_sample": o, "trades_in_sample": tstats(tri), "trades_out_of_sample": tstats(tro),
                 "borrow_sensitivity": sens, "checks": checks, "passed": all(checks.values()),
                 "variants_passing_if_peeked": sum(1 for x in rows if all(check(x["oos"], "failed_squeeze", spy_oos).values())),
                 "portfolio_note": "10 equal slots of 10% each; P&L booked in the exit month",
                 "all_variants": [{"params": x["params"], "is": x["is"], "oos": x["oos"]} for x in rows]})
    print(f"failed_squeeze: {best['params']} IS {best['is'].get('sharpe')} OOS {o.get('sharpe')}/{o.get('cagr_pct')} "
          f"{o.get('by_year')} trades {tstats(tro)} passed={all(checks.values())}", flush=True)

    passing = [f["key"] for f in fams if f["passed"]]
    # today's (current month) short list for passing monthly strategies
    short_list = []
    last_panel = panel[panel["exec_date"] == panel["exec_date"].max()].set_index("ticker")
    for f in fams:
        if not f["passed"] or f["key"] == "failed_squeeze":
            continue
        params, d, names = current_names[f["key"]]
        for t in names:
            r = last_panel.loc[t] if t in last_panel.index else None
            if r is None:
                continue
            fee = float(fee_tier(np.array([r["dtc"]]), np.array([r["si_pct"]]))[0])
            short_list.append({"strategy": f["key"], "ticker": t, "signal_date": str(pd.Timestamp(r["signal_date"]).date()),
                               "rebalance_open": str(pd.Timestamp(d).date()), "ref_close": round(float(r["close"]), 2),
                               "dtc": round(float(r["dtc"]), 2),
                               "si_pct": round(float(r["si_pct"]) * 100, 1) if pd.notna(r["si_pct"]) else None,
                               "mom12_1_pct": round(float(r["mom12_1"]) * 100, 1) if pd.notna(r["mom12_1"]) else None,
                               "borrow_assumed_pct": int(fee * 100)})
    # Long-only reading of the same signal: did the most-shorted basket lag SPY (before any
    # borrow fee)? Uses the short_basket winner's basket, equal weight, held long, no costs.
    sb = next(f for f in fams if f["key"] == "short_basket")
    rows_l = {}
    for d, month in panel.groupby("exec_date", sort=True):
        if month["ret"].isna().all():
            continue
        b = pick(month, sb["chosen_params"], "short")
        if len(b) >= 5 and d in spy_m.index:
            rows_l[d] = (float(b["ret"].fillna(0).mean()), float(spy_m[d]))
    lag = pd.DataFrame(rows_l, index=["basket", "spy"]).T.sort_index()
    def lagstats(x):
        return {"months": int(len(x)), "basket_total_pct": round(float(((1 + x["basket"]).prod() - 1) * 100), 1),
                "spy_total_pct": round(float(((1 + x["spy"]).prod() - 1) * 100), 1),
                "months_basket_lagged_spy_pct": round(float((x["basket"] < x["spy"]).mean() * 100), 0),
                "by_year": {str(y): {"basket_pct": round(float(((1 + g["basket"]).prod() - 1) * 100), 1),
                                     "spy_pct": round(float(((1 + g["spy"]).prod() - 1) * 100), 1)} for y, g in x.groupby(x.index.year)}}
    laggard = {"params": sb["chosen_params"], "in_sample": lagstats(lag[lag.index < SPLIT]), "out_of_sample": lagstats(lag[lag.index >= SPLIT])}
    cur_exec = panel["exec_date"].max()
    cur = pick(panel[panel["exec_date"] == cur_exec], sb["chosen_params"], "short").sort_values(sb["chosen_params"]["signal"], ascending=False)
    laggard["current_count"] = int(len(cur))
    laggard["as_of_signal_date"] = str(pd.Timestamp(cur["signal_date"].iloc[0]).date()) if len(cur) else None
    laggard["current_top"] = [{"ticker": r.ticker, "close": round(float(r.close), 2), "dtc": round(float(r.dtc), 2),
                               "si_pct": round(float(r.si_pct) * 100, 1) if pd.notna(r.si_pct) else None,
                               "pct_vs_200d": round(float(r.close / r.ma200 - 1) * 100, 1) if pd.notna(r.ma200) else None,
                               "borrow_tier_pct": int(fee_tier(np.array([r.dtc]), np.array([r.si_pct]))[0] * 100)}
                              for r in cur.head(12).itertuples()]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": str(C.dropna(how="all").index.max().date()),
        "split": {"fit": "2022-01 to 2024-12", "test": "2025-01 to latest"},
        "universe": {"tickers": len(keep), "avg_eligible_per_month": int(panel.groupby("exec_date").size().mean()),
                     "si_pct_coverage_pct": round(float(panel["si_pct"].notna().mean() * 100), 1),
                     "rules": f"US-listed operating companies (SEC ticker list), price >= ${MIN_PRICE:g}, 20-day $ volume >= ${MIN_DV/1e6:g}M, FINRA report <= 35 days old"},
        "costs": {"per_side_bps": 12, "borrow_tiers": "5%/yr (DTC<5 & SI%<10), 20% (DTC 5-10 or SI% 10-20), 50% (DTC>=10 or SI%>=20)",
                  "borrow_sensitivity": "x0.5, x1, x2 and zero", "cash_interest": "none credited"},
        "pass_bar": "OOS (2025-latest) net total return > 0 after borrow, OOS Sharpe > 0.5, latest year not negative; hedged also needs OOS Sharpe above SPY",
        "total_variants_tried": total, "families": fams, "passing": passing,
        "spy": {"in_sample": spy_is, "out_of_sample": spy_oos},
        "short_list": short_list,
        "laggard_basket": laggard,
        "verdict": "no short strategy passed" if not passing else "passed: " + ", ".join(passing),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"SPY IS {spy_is} OOS {spy_oos}")
    print(f"verdict: {payload['verdict']}; {total} variants; {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
