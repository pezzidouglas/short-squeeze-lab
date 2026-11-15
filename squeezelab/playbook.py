"""Reproducible breakout playbook and honest proxy-tier historical statistics.

Close > prior 20-session high, volume >= 2x prior 20-session mean. Buy next
open, stop 2 ATR(14) below actual entry, target 4 ATR above, exit by session 15.
Costs 12bps each way; gaps through stops fill at open; ambiguous bars stop first.

Historical float/SI-density and off-exchange flow cannot be reconstructed
reliably. Odds tiers therefore use a DISTINCT proxy, not the full live score:
FINRA report DTC, short shares / report ADV (matching share units across splits),
trailing price momentum and published short-position changes. Reports are
lagged 9 business days from settlement; subsequent revisions/release calendars
remain a limitation. Current share counts are NEVER projected backward.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

from .config import setup_label
from .engine import SqueezeEngine

FINRA_SI_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SI_PUBLICATION_LAG_BDAYS = 9
# A tier only gets "Buy"/"Watch" calls if its backtest made at least this much
# per trade on at least this many trades; otherwise its names are "Avoid".
MIN_EDGE_PCT = 0.5
MIN_TRADES = 100
TIER_ORDER = ("high", "elevated", "watch", "low")
TIER_LABELS = {"high": "High (proxy 80+)", "elevated": "Elevated (proxy 60-80)", "watch": "Watch (proxy 40-60)", "low": "Low (proxy <40)"}


@dataclass(frozen=True)
class Rule:
    lookback: int = 20
    volume_mult: float = 2.0
    atr_period: int = 14
    stop_atr: float = 2.0
    reward_risk: float = 2.0
    max_hold: int = 15
    min_price: float = 1.0
    min_dollar_volume: float = 2_000_000.0
    cost_bps: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_RULE = Rule()


# --------------------------------------------------------------------------
# Point-in-time data
# --------------------------------------------------------------------------


def _ua() -> str:
    contact = os.environ.get("SEC_CONTACT_EMAIL", "pezzidouglas@gmail.com")
    return f"short-squeeze-lab/2.0 {contact}"


def _post_finra(body: dict[str, Any], retries: int = 4) -> list[dict[str, Any]]:
    for attempt in range(retries):
        try:
            resp = requests.post(
                FINRA_SI_URL, json=body, timeout=60,
                headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": _ua()},
            )
            if resp.status_code == 200:
                return resp.json() if resp.text.strip() else []
            if resp.status_code == 204:
                return []
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"FINRA short interest request failed: {body.get('dateRangeFilters')}")


def finra_settlement_dates(start: str = "2021-06-01", probe: str = "GME") -> list[str]:
    rows = _post_finra({
        "limit": 5000, "fields": ["settlementDate"],
        "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": probe}],
        "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": start, "endDate": "2100-01-01"}],
    })
    return sorted({r["settlementDate"] for r in rows})


SI_FIELDS = ["symbolCode", "marketClassCode", "currentShortPositionQuantity", "previousShortPositionQuantity",
             "averageDailyVolumeQuantity", "daysToCoverQuantity", "settlementDate"]


def finra_si_for_date(day: str, cache_dir: str, min_adv: int = 100_000) -> list[dict[str, Any]]:
    """Every symbol's short position for one settlement date (cached forever)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{day}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _post_finra({
            "limit": 5000, "offset": offset, "fields": SI_FIELDS,
            "compareFilters": [{"compareType": "GREATER", "fieldName": "averageDailyVolumeQuantity", "fieldValue": min_adv}],
            "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": day, "endDate": day}],
        })
        rows.extend(page)
        if len(page) < 5000:
            break
        offset += 5000
    if rows:  # never cache an empty answer for a date that should exist
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, separators=(",", ":"))
    return rows


def load_si_history(dates: Sequence[str], cache_dir: str) -> pd.DataFrame:
    frames = []
    for day in dates:
        rows = finra_si_for_date(day, cache_dir)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "symbolCode": "ticker", "marketClassCode": "market", "currentShortPositionQuantity": "shares_short",
        "previousShortPositionQuantity": "shares_short_prior", "averageDailyVolumeQuantity": "si_adv",
        "daysToCoverQuantity": "short_ratio", "settlementDate": "settlement_date",
    })
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["available"] = df["settlement_date"] + pd.offsets.BDay(SI_PUBLICATION_LAG_BDAYS)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)
    return df.sort_values(["ticker", "available"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Indicators, triggers, scoring
# --------------------------------------------------------------------------


def normalise_bars(bars: pd.DataFrame) -> pd.DataFrame | None:
    if bars is None or len(bars) == 0:
        return None
    df = bars.copy()
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    df = df[(df["Close"] > 0) & (df["Open"] > 0)].drop_duplicates("Date", keep="last").reset_index(drop=True)
    return df[["Date"] + need] if len(df) else None


def add_indicators(df: pd.DataFrame, rule: Rule = PRIMARY_RULE) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["Close"].shift(1)
    tr = pd.concat([out["High"] - out["Low"], (out["High"] - prev_close).abs(), (out["Low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / rule.atr_period, adjust=False, min_periods=rule.atr_period).mean()
    out["prior_high"] = out["High"].shift(1).rolling(rule.lookback, min_periods=rule.lookback).max()
    out["prior_avg_vol"] = out["Volume"].shift(1).rolling(rule.lookback, min_periods=rule.lookback).mean()
    out["avg_volume_10d"] = out["Volume"].rolling(10, min_periods=5).mean()
    out["roll_high"] = out["High"].rolling(252, min_periods=20).max()
    out["roll_low"] = out["Low"].rolling(252, min_periods=20).min()
    out["prev_close"] = prev_close
    out["dollar_vol"] = (out["Close"] * out["Volume"]).rolling(20, min_periods=10).mean()
    return out


def trigger_mask(ind: pd.DataFrame, rule: Rule = PRIMARY_RULE) -> pd.Series:
    return (
        (ind["Close"] > ind["prior_high"])
        & (ind["Volume"] >= rule.volume_mult * ind["prior_avg_vol"])
        & (ind["Close"] >= rule.min_price)
        & (ind["dollar_vol"] >= rule.min_dollar_volume)
        & ind["atr"].notna()
    )


def pit_values(row: Any, si: dict[str, Any] | None, shares_out: float | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "current_price": float(row.Close),
        "previous_close": _f(row.prev_close),
        "fifty_two_week_high": _f(row.roll_high),
        "fifty_two_week_low": _f(row.roll_low),
        "avg_volume_10d": _f(row.avg_volume_10d),
    }
    if si:
        for key in ("shares_short", "shares_short_prior", "short_ratio"):
            if si.get(key) is not None and not _isnan(si.get(key)):
                values[key] = float(si[key])
    # Historical float/filing release timestamps are not available. Never use
    # current shares outstanding or a guessed SEC reporting lag in past scores.
    # FINRA's report volume and shares-short use the same share units; using
    # those together also avoids mismatching split-adjusted Yahoo volume with
    # historical pre-split short-share counts.
    if si and _f(si.get("si_adv")) and float(si["si_adv"]) > 0:
        values["avg_volume_10d"] = float(si["si_adv"])
    else:
        values.pop("shares_short", None)
        values.pop("shares_short_prior", None)
    return values


_ENGINE = SqueezeEngine("balanced")


def tier_score(values: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
    block = _ENGINE.score(values)
    return block["squeeze_score"], block["setup_tier"], {k: block[f"{k}_score"] for k in ("si", "borrow", "liquidity", "momentum")}


# --------------------------------------------------------------------------
# Trade simulation
# --------------------------------------------------------------------------


def simulate(ind: pd.DataFrame, signal_idx: Iterable[int], rule: Rule = PRIMARY_RULE) -> list[dict[str, Any]]:
    """One position per ticker at a time; returns trades with R multiples."""
    o, h, l, c = (ind[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = ind["atr"].to_numpy(float)
    dates = ind["Date"].to_numpy()
    n = len(ind)
    cost = rule.cost_bps / 10_000
    trades = []
    busy_until = -1
    for s in sorted(signal_idx):
        e = s + 1
        if s <= busy_until or e >= n or not atr[s] > 0:
            continue
        entry = o[e] * (1 + cost)
        risk = rule.stop_atr * atr[s]
        stop, target = entry - risk, entry + rule.reward_risk * risk
        exit_px, reason, x = None, None, e
        for j in range(e, min(n, e + rule.max_hold)):
            x = j
            if j > e and o[j] <= stop:
                exit_px, reason = o[j], "stop (gap)"
                break
            if l[j] <= stop:
                exit_px, reason = stop, "stop"
                break
            if j > e and o[j] >= target:
                exit_px, reason = o[j], "target (gap)"
                break
            if h[j] >= target:
                exit_px, reason = target, "target"
                break
        if reason is None:
            if x - e + 1 < rule.max_hold and x == n - 1:
                continue  # still open at the end of the data: not a completed trade
            exit_px, reason = c[x], "time"
        exit_net = exit_px * (1 - cost)
        ret = exit_net / entry - 1
        trades.append({
            "signal_date": pd.Timestamp(dates[s]).date().isoformat(),
            "entry_date": pd.Timestamp(dates[e]).date().isoformat(),
            "exit_date": pd.Timestamp(dates[x]).date().isoformat(),
            "entry": round(entry, 4), "exit": round(exit_net, 4),
            "ret_pct": round(ret * 100, 3), "r": round((exit_net - entry) / risk, 3),
            "bars": int(x - e + 1), "reason": reason, "signal_idx": int(s),
        })
        busy_until = x
    return trades


def summarize(trades: Sequence[dict[str, Any]], risk_pct: float = 1.0) -> dict[str, Any]:
    if not trades:
        return {"trades": 0}
    rs = np.array([t["r"] for t in trades], float)
    rets = np.array([t["ret_pct"] for t in trades], float)
    wins, losses = rets > 0, rets <= 0
    ordered = sorted(trades, key=lambda t: (t["exit_date"], t["entry_date"]))
    equity, peak, mdd = 1.0, 1.0, 0.0
    for t in ordered:
        equity *= 1 + risk_pct / 100 * t["r"]
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    win_rate = float(wins.mean())
    lo, hi = _wilson(int(wins.sum()), len(rets))
    reasons: dict[str, int] = {}
    for t in trades:
        key = t["reason"].split(" ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "trades": int(len(rets)),
        "tickers": len({t.get("ticker") for t in trades}),
        "win_rate_pct": round(win_rate * 100, 1),
        "win_rate_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
        "avg_win_pct": round(float(rets[wins].mean()), 2) if wins.any() else None,
        "avg_loss_pct": round(float(rets[losses].mean()), 2) if losses.any() else None,
        "expectancy_pct": round(float(rets.mean()), 2),
        "median_ret_pct": round(float(np.median(rets)), 2),
        "expectancy_r": round(float(rs.mean()), 3),
        "profit_factor": round(float(rets[wins].sum() / -rets[losses].sum()), 2) if losses.any() and rets[losses].sum() < 0 else None,
        "max_drawdown_pct_at_1pct_risk": round(mdd * 100, 1),
        "avg_bars": round(float(np.mean([t["bars"] for t in trades])), 1),
        "exits": reasons,
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _isnan(value: Any) -> bool:
    try:
        return float(value) != float(value)
    except (TypeError, ValueError):
        return False


def asof(frame: pd.DataFrame | None, when: pd.Timestamp) -> dict[str, Any] | None:
    """Latest row of a per-ticker frame whose `available` date is <= when."""
    if frame is None or frame.empty:
        return None
    idx = frame["available"].searchsorted(when, side="right") - 1
    if idx < 0:
        return None
    return frame.iloc[idx].to_dict()


# --------------------------------------------------------------------------
# Today's calls
# --------------------------------------------------------------------------


def todays_call(ticker: str, bars: pd.DataFrame | None, row: dict[str, Any], odds: dict[str, Any] | None,
                si: dict[str, Any] | None = None, account_risk_pct: float = 0.25) -> dict[str, Any]:
    """Action, levels and historical odds for one name from today's scan."""
    rule = Rule(**{k: v for k, v in ((odds or {}).get("rule") or {}).items() if k in Rule.__dataclass_fields__}) if odds else PRIMARY_RULE
    out: dict[str, Any] = {"ticker": ticker}
    df = normalise_bars(bars) if bars is not None else None
    if df is None or len(df) < rule.lookback + rule.atr_period + 2:
        out.update(action="No data", reason="not enough price history to set levels")
        return out
    ind = add_indicators(df, rule)
    last = ind.iloc[-1]
    if si is None:
        si = {"short_ratio": row.get("short_ratio")}
    values = pit_values(last, si, row.get("shares_outstanding") or row.get("float"))
    score, tier, comps = tier_score(values)
    atr = float(last["atr"])
    trigger_price = float(last["prior_high"])
    vol_needed = rule.volume_mult * float(last["prior_avg_vol"])
    hit = bool(trigger_mask(ind, rule).iloc[-1])
    ref = float(last["Close"]) if hit else trigger_price
    risk = rule.stop_atr * atr
    stop = ref - risk
    target = ref + rule.reward_risk * risk
    # Not triggered: tomorrow's prior-20-session window ends with today's bar.
    if not hit:
        trigger_price = float(ind["High"].iloc[-rule.lookback:].max())
        ref = trigger_price
        stop, target = ref - risk, ref + rule.reward_risk * risk
        vol_needed = rule.volume_mult * float(ind["Volume"].iloc[-rule.lookback:].mean())
    tier_stats = ((odds or {}).get("tiers") or {}).get(tier) or {}
    expectancy = tier_stats.get("expectancy_pct")
    n = tier_stats.get("trades", 0)
    liquid = float(last["dollar_vol"] or 0) >= rule.min_dollar_volume and float(last["Close"]) >= rule.min_price
    if not liquid:
        action, why = "Avoid", "too illiquid or under $1 for the rule"
    elif expectancy is None or n < MIN_TRADES:
        action, why = "Avoid", f"too few historical trades in the {tier} tier to trust ({n})"
    elif expectancy < MIN_EDGE_PCT:
        action, why = "Avoid", f"no real edge: {tier}-tier breakouts averaged {expectancy:+.2f}% per trade (n={n}); needs >= +{MIN_EDGE_PCT}%"
    elif hit:
        action, why = "Buy trigger hit", "closed above its 20-day high on 2x+ volume today"
    else:
        action, why = "Watch", "wait for a close above the trigger on 2x volume"
    dist = (trigger_price / float(last["Close"]) - 1) * 100
    out.update({
        "action": action,
        "reason": why,
        "close": round(float(last["Close"]), 4),
        "as_of": pd.Timestamp(last["Date"]).date().isoformat(),
        "trigger_price": round(trigger_price, 4),
        "trigger_distance_pct": round(dist, 1),
        "volume_needed": int(vol_needed),
        "today_volume": int(last["Volume"]),
        "est_entry": round(ref, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "stop_pct": round(-risk / ref * 100, 1),
        "target_pct": round(rule.reward_risk * risk / ref * 100, 1),
        "reward_risk": rule.reward_risk,
        "atr": round(atr, 4),
        "max_hold_days": rule.max_hold,
        "account_risk_pct": account_risk_pct if action != "Avoid" else 0.0,
        "position_pct_of_account": None if action == "Avoid" else round(min(100.0, account_risk_pct / (risk / ref * 100) * 100), 1),
        "odds_score": score,
        "odds_tier": tier,
        "odds_components": comps,
        "tier_trades": n,
        "tier_win_rate_pct": tier_stats.get("win_rate_pct"),
        "tier_win_rate_ci95": tier_stats.get("win_rate_ci95"),
        "tier_expectancy_pct": expectancy,
        "tier_expectancy_r": tier_stats.get("expectancy_r"),
        "tier_avg_win_pct": tier_stats.get("avg_win_pct"),
        "tier_avg_loss_pct": tier_stats.get("avg_loss_pct"),
    })
    return out


def rank_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"Buy trigger hit": 0, "Watch": 1, "Avoid": 2, "No data": 3}

    def key(c: dict[str, Any]) -> tuple:
        return (order.get(c.get("action"), 9), -(c.get("tier_expectancy_pct") or -99),
                abs(c.get("trigger_distance_pct") or 999))
    return sorted(calls, key=key)
