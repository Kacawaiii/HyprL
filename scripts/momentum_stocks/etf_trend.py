#!/usr/bin/env python3
"""Clenow 'Following the Trend' core strategy on a diversified ETF universe.

Book: Clenow, *Following the Trend*, ch. 4 (core strategy rules, p.76). This is
the decorrelated sleeve: it can go SHORT and trades bonds/commodities/FX, so it
is genuinely orthogonal to a long-only equity book.

Rules (exact, p.78):
  * Long entries only if MA50 > MA100; short entries only if MA50 < MA100.
  * Buy when today's close is the highest close of the past 50 days.
  * Sell short when today's close is the lowest close of the past 50 days.
  * Size = equity * risk_factor / ATR100  (risk_factor = 0.002 = 20 bps).
  * Exit long when price falls 3*ATR100 below its highest close since entry.
  * Exit short when price rises 3*ATR100 above its lowest close since entry.
  * Signals on close[t], executed at open[t+1] (no look-ahead).
Cash-account realism: unlevered, gross exposure capped at MAX_GROSS (positions
scaled down pro-rata when the ATR sizing would exceed it). Reports realized
leverage. Costs COST_BPS per side.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kyo/HyprL")
ETF = ROOT / "data/momentum/etf"

MA_FAST, MA_SLOW = 50, 100
BREAKOUT = 50
ATR_WIN = 100
RISK_FACTOR = 0.002
STOP_ATR = 3.0
MAX_GROSS = 1.5          # max gross exposure (1.0 = unlevered; 1.5 = modest Reg-T)
COST_BPS = 5.0
INITIAL = 100_000.0


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load():
    data = {}
    for f in ETF.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        data[f.stem] = df
    return data


@dataclass
class P:
    side: int          # +1 long, -1 short
    shares: float
    entry: float
    extreme: float     # highest close since entry (long) / lowest (short)


def _clean_run(data, feats, cal, start_i, we, max_gross):
    """Clean, unambiguous accounting pass (cash + MTM, next-open execution)."""
    def val(sym, date, field):
        ser = feats[sym][field]
        if date in ser.index:
            v = ser.loc[date]
            return float(v) if not (isinstance(v, float) and math.isnan(v)) else None
        return None

    cash = INITIAL
    pos: dict[str, P] = {}
    pending = []
    curve, levs = [], []
    n_trades = 0

    for i in range(start_i, len(cal)):
        date = cal[i]
        if we is not None and date > we:
            break

        # execute pending at today's open
        for sym, action, side, shares in pending:
            op = val(sym, date, "o")
            if op is None:
                continue
            cost = abs(shares if action == "enter" else pos.get(sym, P(0,0,0,0)).shares) * op * COST_BPS / 1e4
            if action == "enter" and sym not in pos:
                cash -= cost
                pos[sym] = P(side, shares, op, op)
            elif action == "exit" and sym in pos:
                p = pos[sym]
                cash += p.side * p.shares * (op - p.entry) - cost
                del pos[sym]
                n_trades += 1
        pending = []

        # MTM equity: cash + open PnL
        equity = cash + sum(p.side * p.shares * ((val(s, date, "c") or p.entry) - p.entry)
                            for s, p in pos.items())
        curve.append((date, equity))
        gross = sum(p.shares * (val(s, date, "c") or p.entry) for s, p in pos.items())
        levs.append(gross / equity if equity > 0 else 0)

        # signals on close
        new_pending = []
        for sym, p in list(pos.items()):
            cp, a = val(sym, date, "c"), val(sym, date, "atr")
            if cp is None or a is None:
                continue
            if p.side == 1:
                p.extreme = max(p.extreme, cp)
                if cp <= p.extreme - STOP_ATR * a:
                    new_pending.append((sym, "exit", 1, p.shares))
            else:
                p.extreme = min(p.extreme, cp)
                if cp >= p.extreme + STOP_ATR * a:
                    new_pending.append((sym, "exit", -1, p.shares))
        exiting = {x[0] for x in new_pending}
        for sym, s in feats.items():
            if sym in pos or sym in exiting:
                continue
            cp, hi, lo = val(sym, date, "c"), val(sym, date, "hi"), val(sym, date, "lo")
            mf, ms, a = val(sym, date, "ma_f"), val(sym, date, "ma_s"), val(sym, date, "atr")
            if None in (cp, hi, lo, mf, ms, a) or a <= 0:
                continue
            side = 0
            if mf > ms and cp >= hi:
                side = 1
            elif mf < ms and cp <= lo:
                side = -1
            if side == 0:
                continue
            # projected gross cap (unlevered-ish)
            shares = math.floor(equity * RISK_FACTOR / a)
            if shares <= 0:
                continue
            if (gross + shares * cp) > max_gross * equity:
                continue
            gross += shares * cp
            new_pending.append((sym, "enter", side, shares))
        pending = new_pending

    eqs = pd.Series({d: v for d, v in curve})
    return eqs, np.mean(levs), n_trades


def metrics(eqs):
    r = eqs.pct_change().dropna()
    ann = (eqs.iloc[-1] / eqs.iloc[0]) ** (252 / len(eqs)) - 1
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    dn = r[r < 0].std()
    so = r.mean() / dn * math.sqrt(252) if dn and dn > 0 else 0
    mdd = -((eqs - eqs.cummax()) / eqs.cummax()).min()
    return ann, sh, so, mdd


def main():
    data = load()
    windows = [
        ("FULL 2007-2026", "2007-06-01", None),
        ("2008 crisis 07-09", "2007-06-01", "2009-12-31"),
        ("recovery 10-14", "2010-01-01", "2014-12-31"),
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("covid 19-21", "2019-01-01", "2021-12-31"),
        ("recent 22-26", "2022-01-01", "2026-07-11"),
    ]
    print(f"ETF trend (Clenow core, {len(data)} ETFs, risk 20bps, gross cap {MAX_GROSS})")
    print(f"{'Period':<20}{'AnnRet%':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD%':>8}{'Lev':>6}")
    print("-" * 60)
    shps = []
    full_eqs = None
    for name, ws, we in windows:
        eqs, lev, nt = _prep_and_run(data, ws, we)
        a, s, so, dd = metrics(eqs)
        if name != "FULL 2007-2026":
            shps.append(s)
        else:
            full_eqs = eqs
        print(f"{name:<20}{a*100:>9.1f}{s:>8.2f}{so:>9.2f}{dd*100:>8.1f}{lev:>6.2f}")
    mean = sum(shps) / len(shps)
    std = (sum((x - mean) ** 2 for x in shps) / len(shps)) ** 0.5
    print("-" * 60)
    print(f"sub-period Sharpe: mean={mean:.2f} std={std:.2f} CoV={std/mean*100:.0f}% min={min(shps):.2f}")
    full_eqs.to_frame("equity").to_csv(ROOT / "data/momentum/etf_trend_equity.csv")


def _prep_and_run(data, ws, we):
    feats = {}
    for sym, df in data.items():
        c = df["Close"]
        feats[sym] = dict(o=df["Open"], c=c,
                          ma_f=c.rolling(MA_FAST).mean(), ma_s=c.rolling(MA_SLOW).mean(),
                          hi=c.rolling(BREAKOUT).max(), lo=c.rolling(BREAKOUT).min(),
                          atr=atr(df, ATR_WIN))
    cal = data["SPY"].index
    start_i = MA_SLOW + 5
    if ws:
        start_i = max(start_i, cal.searchsorted(pd.Timestamp(ws)))
    wend = pd.Timestamp(we) if we else None
    return _clean_run(data, feats, cal, start_i, wend, MAX_GROSS)


if __name__ == "__main__":
    main()
