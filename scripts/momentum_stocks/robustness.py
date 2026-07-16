#!/usr/bin/env python3
"""Parameter robustness sweep + Monte Carlo for the Clenow momentum backtest.

Robustness: vary one knob at a time around Clenow's canonical values. A real
edge should degrade gracefully, not live on a single lucky parameter.
Monte Carlo: block-bootstrap the weekly equity returns to get a distribution
of final return / max DD / Sharpe (guards against lucky trade ordering).
"""
import json
import math
import numpy as np
import pandas as pd

import clenow_backtest as cb

data = cb.load_prices()
idx_key = "_GSPC" if "_GSPC" in data else "SPY"
index_close = data[idx_key]["Close"]
index_ma200 = index_close.rolling(cb.MA_INDEX).mean()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}

# cache precompute per reg_window (only knob that changes features)
_feat_cache = {}
def feats_for(w):
    if w not in _feat_cache:
        _feat_cache[w] = cb.precompute(stocks, reg_window=w)
    return _feat_cache[w]


def one(reg_window=90, risk=0.001, hold=0.20, cost=5.0):
    cb.REG_WINDOW = reg_window
    cb.RISK_FACTOR = risk
    cb.HOLD_PCT = hold
    cb.COST_BPS = cost
    r = cb.run(data=data, feats=feats_for(reg_window), index_close=index_close,
               index_ma200=index_ma200, verbose=False)
    s = r["strategy"]
    return s, r["_eqs"]


BASE = dict(reg_window=90, risk=0.001, hold=0.20, cost=5.0)
print("=== PARAMETER ROBUSTNESS (one knob at a time) ===")
print(f"{'config':<26}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}{'Calmar':>8}{'Trades':>8}")
print("-" * 67)
base_eqs = None
sweeps = {
    "reg_window": [60, 90, 120],
    "risk":       [0.0005, 0.001, 0.0015],
    "hold":       [0.10, 0.20, 0.30],
    "cost":       [5.0, 10.0, 20.0],
}
for knob, vals in sweeps.items():
    for v in vals:
        cfg = dict(BASE); cfg[knob] = v
        s, eqs = one(**cfg)
        tag = f"{knob}={v}" + (" *" if v == BASE[knob] else "")
        print(f"{tag:<26}{s['annualized_pct']:>9}{s['sharpe']:>8}"
              f"{s['max_dd_pct']:>8}{s['calmar']:>8}{s['trades']:>8}")
        if v == BASE[knob] and base_eqs is None:
            base_eqs = eqs
    print()

# ===== Monte Carlo: block bootstrap of weekly equity returns =====
print("=== MONTE CARLO (block-bootstrap weekly returns, 2000 paths) ===")
eqs = base_eqs
weekly = eqs.resample("W").last().dropna()
rets = weekly.pct_change().dropna().values
n = len(rets)
block = 8  # ~2 months blocks to preserve autocorrelation
rng = np.random.default_rng(42)
finals, mdds, sharpes = [], [], []
for _ in range(2000):
    path = []
    while len(path) < n:
        start = rng.integers(0, n - block)
        path.extend(rets[start:start + block])
    path = np.array(path[:n])
    eq = 100000 * np.cumprod(1 + path)
    finals.append(eq[-1] / 100000 - 1)
    peak = np.maximum.accumulate(eq)
    mdds.append(((eq - peak) / peak).min())
    sharpes.append(path.mean() / path.std() * math.sqrt(52) if path.std() > 0 else 0)

def pct(a, p): return round(float(np.percentile(a, p)) * (100 if abs(np.percentile(a,p))<10 else 1), 2)
finals = np.array(finals); mdds = np.array(mdds); sharpes = np.array(sharpes)
print(f"Total return : p5={np.percentile(finals,5)*100:6.1f}%  "
      f"med={np.percentile(finals,50)*100:6.1f}%  p95={np.percentile(finals,95)*100:6.1f}%")
print(f"Max drawdown : p5={np.percentile(mdds,95)*100:6.1f}%  "
      f"med={np.percentile(mdds,50)*100:6.1f}%  worst={mdds.min()*100:6.1f}%")
print(f"Sharpe (ann) : p5={np.percentile(sharpes,5):6.2f}  "
      f"med={np.percentile(sharpes,50):6.2f}  p95={np.percentile(sharpes,95):6.2f}")
print(f"P(total return > 0): {(finals>0).mean()*100:.1f}%   "
      f"P(maxDD worse than -35%): {(mdds<-0.35).mean()*100:.1f}%")
