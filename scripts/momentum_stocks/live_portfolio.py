#!/usr/bin/env python3
"""Live two-sleeve portfolio engine (paper) — Alpaca.

Sleeve A: equity momentum (Clenow Stocks on the Move) — rebalanced weekly (Wed).
Sleeve B: ETF cross-asset trend (Clenow Following the Trend) — evaluated daily.

Capital split MOM_ALLOC / TREND_ALLOC of live account equity. Signals computed
on yfinance daily closes (same as the backtest); execution + account state via
Alpaca. Fractional shares -> dollar sizing, no rounding.

DRY-RUN by default: prints the reconciled order list and submits nothing.
Pass --live to actually submit market orders.

State (trend trailing stops) persisted in live/portfolio/state.json.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Repo root derived from this file, not hardcoded: the CI runner checks the repo
# out at a different path than the dev machine.
ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "live/portfolio"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "state.json"
LOG_PATH = STATE_DIR / "orders.jsonl"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
EQUITY_LOG = STATE_DIR / "equity.jsonl"

# --- Alpaca paper creds (paper account PA3VMR6S2EQN) ---
# Never hardcode: this repo is public. Supply via env / GitHub Secrets.
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_BASE = os.environ.get("ALPACA_BASE", "https://paper-api.alpaca.markets")

def _require_creds():
    """Checked lazily, not at import: the pure signal functions below are imported
    by the backtest/validation harnesses, which need no account access."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        sys.exit("ALPACA_KEY / ALPACA_SECRET not set — export them or add repo secrets.")

# --- allocation ---
MOM_ALLOC = 0.60
TREND_ALLOC = 0.40

# --- momentum params (Clenow) ---
REG_WINDOW, MA_TREND, MA_INDEX = 90, 100, 200
ATR_WIN_M, RISK_M = 20, 0.001
GAP_MAX, MIN_PRICE, MAX_WEIGHT_M = 0.15, 5.0, 0.15
HOLD_PCT = 0.20

# --- trend params (Clenow) ---
MA_FAST, MA_SLOW, BREAKOUT = 50, 100, 50
ATR_WIN_T, RISK_T, STOP_ATR = 100, 0.002, 3.0
TREND_GROSS = 1.5

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EEM", "EFA", "TLT", "IEF", "LQD", "HYG",
                "GLD", "SLV", "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "VNQ"]


# ============ Alpaca REST ============
def _req(method, path, body=None):
    _require_creds()
    url = ALPACA_BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get_equity():
    return float(_req("GET", "/v2/account")["equity"])


def get_positions():
    return {p["symbol"]: float(p["qty"]) for p in _req("GET", "/v2/positions")}


def submit_order(symbol, qty, side):
    return _req("POST", "/v2/orders", {
        "symbol": symbol, "qty": str(abs(qty)), "side": side,
        "type": "market", "time_in_force": "day"})


# ============ data + indicators ============
def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rolling_slope_r2(close, window):
    y = np.log(close.values.astype(float))
    x = np.arange(window)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    w = y[-window:]
    if len(w) < window or np.isnan(w).any():
        return None
    slope = (w * xc).sum() / sxx
    yhat = slope * xc + w.mean()
    ss_res = ((w - yhat) ** 2).sum()
    ss_tot = ((w - w.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return (math.exp(slope) ** 250 - 1) * r2


def fetch(tickers, period="2y"):
    df = yf.download(tickers, period=period, progress=False, auto_adjust=True,
                     group_by="ticker", threads=True)
    out = {}
    multi = isinstance(df.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if multi:
                sub = df[t].copy()
            else:
                sub = df.copy()  # single ticker, flat columns
            sub = sub.dropna(subset=["Close"])
            if len(sub) >= MA_INDEX:
                out[t] = sub
        except Exception:
            pass
    return out


# ============ momentum sleeve targets ($ per symbol) ============
def momentum_targets(equity, data, spy_close, held):
    """Pure: data = {sym: df as-of}, spy_close = SPY close series as-of,
    held = set of momentum stocks currently held. Implements Clenow's hysteresis
    (keep a held name while it stays in the top HOLD_PCT and above MA100)."""
    # size off the sleeve's OWN capital so the sleeve == MOM_ALLOC x the backtested
    # book (same names, scaled), not a concentrated top-of-list subset.
    sleeve_eq = MOM_ALLOC * equity
    budget = sleeve_eq
    regime_ok = spy_close.iloc[-1] > spy_close.rolling(MA_INDEX).mean().iloc[-1]
    ranked = []
    for sym, df in data.items():
        c = df["Close"]
        score = rolling_slope_r2(c, REG_WINDOW)
        if score is None:
            continue
        px = float(c.iloc[-1])
        ma100 = c.rolling(MA_TREND).mean().iloc[-1]
        gap = c.pct_change().abs().rolling(REG_WINDOW).max().iloc[-1]
        a20 = atr(df, ATR_WIN_M).iloc[-1]
        if math.isnan(ma100) or math.isnan(a20) or a20 <= 0:
            continue
        qual = px > ma100 and gap < GAP_MAX and score > 0 and px >= MIN_PRICE
        ranked.append((score, sym, px, ma100, a20, qual))
    ranked.sort(reverse=True)
    rank_of = {r[1]: k for k, r in enumerate(ranked)}
    info = {r[1]: r for r in ranked}
    keep_cut = int(len(ranked) * HOLD_PCT)

    def size(px, a20):
        return min((sleeve_eq * RISK_M / a20) * px, MAX_WEIGHT_M * sleeve_eq)

    targets, used = {}, 0.0
    # 1) keep held names still in top band and above MA100 (hysteresis).
    # Iterate in RANK order: deterministic (a set's order varies per process, which
    # would make the live book depend on hash seed) and keeps the strongest first
    # when the budget binds.
    for sym in sorted(held, key=lambda s: rank_of.get(s, 10 ** 9)):
        r = info.get(sym)
        if r is None:
            continue
        _, _, px, ma100, a20, _ = r
        if rank_of[sym] <= keep_cut and px > ma100:
            d = size(px, a20)
            if used + d <= budget:
                targets[sym] = d
                used += d
    # 2) fill remaining budget with new top-ranked qualified names (only if regime on)
    if regime_ok:
        for score, sym, px, ma100, a20, qual in ranked:
            if sym in targets or not qual or rank_of[sym] > keep_cut:
                continue
            d = size(px, a20)
            if used + d > budget:
                continue
            targets[sym] = d
            used += d
    note = f"{len(targets)} names, ${used:,.0f}/{budget:,.0f}" + \
           ("" if regime_ok else " [regime OFF: no new buys, holds only]")
    return targets, note


# ============ trend sleeve targets ($ signed per ETF) ============
def trend_targets(equity, data, state):
    """Pure: data = {etf: df as-of}. state['trend'] holds open-position extremes."""
    # size off the sleeve's OWN capital (== TREND_ALLOC x the backtested trend book)
    sleeve_eq = TREND_ALLOC * equity
    budget_gross = TREND_GROSS * sleeve_eq
    tstate = state.get("trend", {})
    targets, gross, notes = {}, 0.0, []
    for sym in ETF_UNIVERSE:
        df = data.get(sym)
        if df is None:
            continue
        c = df["Close"]
        px = float(c.iloc[-1])
        maf = c.rolling(MA_FAST).mean().iloc[-1]
        mas = c.rolling(MA_SLOW).mean().iloc[-1]
        hi = c.rolling(BREAKOUT).max().iloc[-1]
        lo = c.rolling(BREAKOUT).min().iloc[-1]
        a = atr(df, ATR_WIN_T).iloc[-1]
        if any(math.isnan(v) for v in (maf, mas, hi, lo, a)) or a <= 0:
            continue
        st = tstate.get(sym)  # {'side':1/-1,'extreme':px,'shares':n}
        side = 0
        if st:  # manage open position: trailing stop
            side = st["side"]
            if side == 1:
                st["extreme"] = max(st["extreme"], px)
                if px <= st["extreme"] - STOP_ATR * a:
                    side = 0
            else:
                st["extreme"] = min(st["extreme"], px)
                if px >= st["extreme"] + STOP_ATR * a:
                    side = 0
        fresh = False
        if side == 0:  # look for fresh entry
            if maf > mas and px >= hi:
                side = 1
            elif maf < mas and px <= lo:
                side = -1
            else:
                tstate.pop(sym, None)
                continue
            # size ONCE at entry (Clenow sizes on entry and holds the share count
            # until the exit — do NOT resize daily, that only adds turnover).
            st = {"side": side, "extreme": px, "shares": sleeve_eq * RISK_T / a}
            fresh = True
        dollars = st["shares"] * px          # fixed shares -> $ varies with price
        # the gross cap gates NEW entries only; an already-open position is never
        # force-closed just because gross drifted above the cap.
        if fresh and gross + abs(dollars) > budget_gross:
            tstate.pop(sym, None)
            continue
        tstate[sym] = st
        gross += abs(dollars)
        targets[sym] = side * dollars
    state["trend"] = tstate
    return targets, f"{len(targets)} ETF positions, gross ${gross:,.0f}/{budget_gross:,.0f}"


# ============ reconcile ============
def latest_price(sym, cache):
    if sym in cache:
        return cache[sym]
    try:
        p = float(yf.Ticker(sym).fast_info["lastPrice"])
    except Exception:
        p = None
    cache[sym] = p
    return p


def write_heartbeat(ok: bool, **detail):
    """Always written, even on failure — the watchdog reads this to know we ran.
    A run that errors still leaves a heartbeat with ok=False, so 'crashed' is
    distinguishable from 'never started'."""
    HEARTBEAT_PATH.write_text(json.dumps(
        {"ts": datetime.now(timezone.utc).isoformat(), "ok": ok, **detail}, indent=2))


def log_equity(equity: float):
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "equity": equity}) + "\n")


def run_once(args):
    today = datetime.now(timezone.utc)
    is_wed = today.weekday() == 2 or args.force_wed
    state = json.load(open(STATE_PATH)) if STATE_PATH.exists() else {}

    equity = get_equity()
    print(f"=== Live portfolio {today.date()} | equity ${equity:,.0f} | "
          f"{'LIVE' if args.live else 'DRY-RUN'} ===")
    log_equity(equity)   # track-record curve, one point per run

    current = get_positions()
    held_stocks = {s for s in current if s not in ETF_UNIVERSE}

    # sleeve targets ($)
    etf_data = fetch(ETF_UNIVERSE)
    if is_wed:
        stock_tickers = json.load(open(ROOT / "data/momentum/sp500_current.json"))
        stock_data = fetch(stock_tickers)
        spy_close = etf_data["SPY"]["Close"]
        mom, mnote = momentum_targets(equity, stock_data, spy_close, held_stocks)
        state["momentum"] = mom  # persist last momentum target on Wed
    else:
        mom = state.get("momentum", {})
        mnote = "held (non-Wednesday, no momentum rebalance)"
    trend, tnote = trend_targets(equity, etf_data, state)
    print(f"  momentum: {mnote}")
    print(f"  trend:    {tnote}")

    # combined target $ (long-only stocks + signed ETFs)
    target_usd = {}
    for s, d in mom.items():
        target_usd[s] = target_usd.get(s, 0) + d
    for s, d in trend.items():
        target_usd[s] = target_usd.get(s, 0) + d

    pxcache = {}
    all_syms = set(target_usd) | set(current)
    orders = []
    for sym in sorted(all_syms):
        px = latest_price(sym, pxcache)
        if not px:
            continue
        tgt_qty = target_usd.get(sym, 0) / px
        cur_qty = current.get(sym, 0)
        delta = tgt_qty - cur_qty
        if abs(delta * px) < 50:  # ignore <$50 drift
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append((sym, round(delta, 4), side, round(tgt_qty, 3), cur_qty, round(px, 2)))

    print(f"\n  {'SYM':<6}{'ORDER':>12}{'side':>6}{'target':>12}{'current':>12}{'px':>9}")
    print("  " + "-" * 57)
    for sym, delta, side, tgt, cur, px in orders:
        print(f"  {sym:<6}{delta:>12}{side:>6}{tgt:>12}{cur:>12}{px:>9}")
    if not orders:
        print("  (no orders — portfolio already at target)")

    sent, failed = 0, 0
    if args.live and orders:
        print("\n  submitting...")
        for sym, delta, side, *_ in orders:
            try:
                submit_order(sym, delta, side)
                sent += 1
                print(f"    {side} {abs(delta)} {sym} OK")
            except Exception as e:
                failed += 1
                print(f"    {side} {sym} FAILED: {repr(e)[:120]}")
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"ts": today.isoformat(), "equity": equity,
                                "sent": sent, "failed": failed,
                                "orders": [list(o) for o in orders]}) + "\n")

    if args.live or is_wed:
        json.dump(state, open(STATE_PATH, "w"), indent=2)

    # a run where every order failed is NOT a healthy run — say so
    healthy = not (args.live and orders and sent == 0)
    write_heartbeat(healthy, equity=equity, mode="live" if args.live else "dry-run",
                    orders=len(orders), sent=sent, failed=failed,
                    momentum=len(mom), trend=len(trend))
    print("\ndone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="submit orders (default dry-run)")
    ap.add_argument("--force-wed", action="store_true", help="run momentum regardless of weekday")
    args = ap.parse_args()
    try:
        run_once(args)
    except Exception as e:
        # still leave a heartbeat so the watchdog sees "ran but crashed",
        # not "never started" — different problems, different fixes.
        write_heartbeat(False, error=repr(e)[:300])
        raise


if __name__ == "__main__":
    main()
