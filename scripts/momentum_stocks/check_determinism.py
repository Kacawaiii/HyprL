#!/usr/bin/env python3
"""Determinism check: the live momentum book must not depend on PYTHONHASHSEED.

Runs the momentum walk-forward and prints a hash of the equity curve. Run twice
with different PYTHONHASHSEED; the digests must match.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/kyo/HyprL")
sys.argv = ["x"]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cb = load_mod("cb", ROOT / "scripts/momentum_stocks/clenow_backtest.py")
lp = load_mod("lp", ROOT / "scripts/momentum_stocks/live_portfolio.py")

prices = cb.load_prices()
spy = prices["SPY"]["Close"]
current_list = set(json.load(open(ROOT / "data/momentum/sp500_current.json")))
stock_dfs = {s: prices[s] for s in current_list if s in prices}
cal = spy.index
start = cal.searchsorted(pd.Timestamp("2018-01-01"))  # shorter window: fast check
INIT, COST = 100_000.0, 5.0 / 1e4


def slice_asof(dfs, date, n=230):
    out = {}
    for s, df in dfs.items():
        if date in df.index:
            i = df.index.get_loc(date)
            if i >= 210:
                out[s] = df.iloc[max(0, i - n):i + 1]
    return out


cash, sh, curve = INIT, {}, []
for k in range(start, len(cal)):
    date = cal[k]
    eq = cash + sum(q * float(stock_dfs[s]["Close"].loc[date])
                    for s, q in sh.items() if date in stock_dfs[s].index)
    curve.append(eq)
    if date.weekday() != 2:
        continue
    data = slice_asof(stock_dfs, date)
    held = {s for s, q in sh.items() if q != 0}
    tgt, _ = lp.momentum_targets(eq, data, spy.loc[:date], held)
    for s in list(sh) + list(tgt):
        if date not in stock_dfs.get(s, pd.DataFrame()).index:
            continue
        px = float(stock_dfs[s]["Close"].loc[date])
        tq = tgt.get(s, 0.0) / px
        dq = tq - sh.get(s, 0.0)
        if abs(dq * px) < 1:
            continue
        cash -= dq * px + abs(dq * px) * COST
        sh[s] = tq
    sh = {s: q for s, q in sh.items() if abs(q) > 1e-9}

digest = hashlib.sha256(",".join(f"{v:.6f}" for v in curve).encode()).hexdigest()[:16]
print(f"seed={sys.flags.hash_randomization} final={curve[-1]:,.2f} digest={digest}")
