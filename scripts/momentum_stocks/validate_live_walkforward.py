#!/usr/bin/env python3
"""Walk-forward the LIVE engine's own decision functions over history.

Drives live_portfolio.momentum_targets and .trend_targets bar-by-bar on cached
data (current S&P 500 list + ETF universe), simulating execution, and reports
momentum-only / trend-only / combined metrics. Confirms the deployed code path
reproduces the strategy — no implementation drift between live and backtest.

Universe note: momentum uses the CURRENT S&P list (what the live engine trades
going forward) => survivorship-optimistic vs the PIT backtest; the point here is
code-path equivalence and a sane curve, not the survivorship-honest number.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
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

INIT = 100_000.0
COST = 5.0 / 1e4

# --- load cached data ---
prices = cb.load_prices()
spy = prices["SPY"]["Close"]
current_list = set(json.load(open(ROOT / "data/momentum/sp500_current.json")))
stock_dfs = {s: prices[s] for s in current_list if s in prices}
etf_dfs = {s: pd.read_parquet(ROOT / f"data/momentum/etf/{s}.parquet")
           for s in lp.ETF_UNIVERSE}
for d in etf_dfs.values():
    d.index = pd.to_datetime(d.index)

cal = spy.index
start = cal.searchsorted(pd.Timestamp("2010-01-01"))  # 3y warmup from 2007 data


def metrics(eqs):
    r = eqs.pct_change().dropna()
    ann = (eqs.iloc[-1] / eqs.iloc[0]) ** (252 / len(eqs)) - 1
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    mdd = -((eqs - eqs.cummax()) / eqs.cummax()).min()
    return ann * 100, sh, mdd * 100, r


def slice_asof(dfs, date, n=230):
    out = {}
    for s, df in dfs.items():
        if date in df.index:
            i = df.index.get_loc(date)
            if i >= 210:
                out[s] = df.iloc[max(0, i - n):i + 1]
    return out


def walk_momentum():
    cash = INIT
    sh = {}
    curve = []
    for k in range(start, len(cal)):
        date = cal[k]
        eq = cash + sum(q * float(stock_dfs[s]["Close"].loc[date])
                        for s, q in sh.items() if date in stock_dfs[s].index)
        curve.append((date, eq))
        if date.weekday() != 2:
            continue
        data = slice_asof(stock_dfs, date)
        spy_slice = spy.loc[:date]
        held = {s for s, q in sh.items() if q != 0}
        tgt, _ = lp.momentum_targets(eq, data, spy_slice, held)
        # reconcile to target dollars
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
    return pd.Series({d: v for d, v in curve})


def walk_trend():
    """Signals on close[t], executed at open[t+1] — matches etf_trend and the real
    live cron (which submits after the close and fills at the next open)."""
    cash = INIT
    sh = {}
    state = {"trend": {}}
    curve = []
    pending = {}   # sym -> target qty, to execute at next open
    for k in range(start, len(cal)):
        date = cal[k]
        # 1) execute yesterday's signals at today's OPEN
        for s, tq in pending.items():
            df = etf_dfs.get(s)
            if df is None or date not in df.index:
                continue
            op = float(df["Open"].loc[date])
            dq = tq - sh.get(s, 0.0)
            if abs(dq * op) < 1:
                continue
            cash -= dq * op + abs(dq * op) * COST
            sh[s] = tq
        sh = {s: q for s, q in sh.items() if abs(q) > 1e-9}
        pending = {}

        # 2) mark equity at today's close
        eq = cash + sum(q * float(etf_dfs[s]["Close"].loc[date])
                        for s, q in sh.items() if date in etf_dfs[s].index)
        curve.append((date, eq))

        # 3) compute signals on today's close -> execute next open
        data = slice_asof(etf_dfs, date)
        if not data:
            continue
        tgt, _ = lp.trend_targets(eq, data, state)  # signed dollars
        for s in set(list(sh) + list(tgt)):
            df = etf_dfs.get(s)
            if df is None or date not in df.index:
                continue
            px = float(df["Close"].loc[date])
            pending[s] = tgt.get(s, 0.0) / px
    return pd.Series({d: v for d, v in curve})


print("walking momentum (live code)...")
m_eq = walk_momentum()
print("walking trend (live code)...")
t_eq = walk_trend()

# reference backtest-engine curves on SAME current-list universe
print("reference: clenow_backtest engine (current list)...")
feats = cb.precompute(stock_dfs)
idx = prices["_GSPC"]["Close"] if "_GSPC" in prices else spy
ref_m = cb.run(data={**stock_dfs, "SPY": prices["SPY"], "_GSPC": prices.get("_GSPC", prices["SPY"])},
               feats=feats, index_close=idx, index_ma200=idx.rolling(200).mean(),
               win_start="2010-01-01", verbose=False)["_eqs"]

am, sm, dm, rm = metrics(m_eq)
at, st, dt, rt = metrics(t_eq)
arm, srm, drm, _ = metrics(ref_m)

# Combined book. Each sleeve already sizes off its OWN allocation (60% / 40%) on a
# 100k base, so each curve is that sleeve's contribution to the whole account —
# the contributions ADD (re-weighting 0.6/0.4 here would double-scale).
mr = m_eq.pct_change().dropna(); tr = t_eq.pct_change().dropna()
common = mr.index.intersection(tr.index)
comb = mr.loc[common] + tr.loc[common]
ceq = (1 + comb).cumprod() * INIT
ac, sc, dc, _ = metrics(ceq)
corr = np.corrcoef(mr.loc[common], tr.loc[common])[0, 1]

print(f"\n{'book (2010-2026)':<34}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}")
print("-" * 59)
print(f"{'LIVE momentum (walk-fwd)':<34}{am:>9.1f}{sm:>8.2f}{dm:>8.1f}")
print(f"{'  ref clenow_backtest engine':<34}{arm:>9.1f}{srm:>8.2f}{drm:>8.1f}")
print(f"{'LIVE trend (walk-fwd)':<34}{at:>9.1f}{st:>8.2f}{dt:>8.1f}")
print(f"{"LIVE combo (mom+trend)":<34}{ac:>9.1f}{sc:>8.2f}{dc:>8.1f}")
print("-" * 59)
print(f"live momentum vs ref engine: Sharpe drift = {abs(sm-srm):.2f}  "
      f"(should be small; universe/exec identical)")
print(f"momentum-trend correlation: {corr:+.2f}")
