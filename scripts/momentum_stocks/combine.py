#!/usr/bin/env python3
"""Test the diversification payoff: momentum (trend) + MR (counter-trend).

Runs both strategies full-period on the same PIT universe, aligns daily returns,
measures their correlation, and reports the Sharpe of a 50/50 combined book.
Carver's free lunch: two ~equal-Sharpe strategies with low correlation combine
to a higher Sharpe than either alone.
"""
import math
import numpy as np
import pandas as pd

import clenow_backtest as cb
import mr_backtest as mr

data = cb.load_prices()
membership = cb.load_membership()
idx_key = "_GSPC" if "_GSPC" in data else "SPY"
index_close = data[idx_key]["Close"]
index_ma200 = index_close.rolling(cb.MA_INDEX).mean()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}

# momentum PIT
mfeats = cb.precompute(stocks)
mres = cb.run(data=data, feats=mfeats, index_close=index_close,
              index_ma200=index_ma200, verbose=False, membership=membership)
m_eq = mres["_eqs"]

# mean reversion RSI<10 PIT
rfeats = mr.precompute(stocks, rsi_entry=10.0)
rres = mr.run(data, rfeats, membership, rsi_entry=10.0)
r_eq = rres["_eqs"]

# align on common dates
idx = m_eq.index.intersection(r_eq.index)
mr_ = m_eq.reindex(idx).pct_change().dropna()
rr_ = r_eq.reindex(idx).pct_change().dropna()
common = mr_.index.intersection(rr_.index)
mr_, rr_ = mr_.loc[common], rr_.loc[common]

# SPY for reference
spy = data["SPY"]["Close"].reindex(common).ffill().pct_change().dropna()


def stats(r):
    sh = r.mean() / r.std() * math.sqrt(252)
    eq = (1 + r).cumprod()
    mdd = -((eq - eq.cummax()) / eq.cummax()).min()
    ann = eq.iloc[-1] ** (252 / len(r)) - 1
    return ann, sh, mdd


corr = float(np.corrcoef(mr_.values, rr_.values)[0, 1])
print(f"daily-return correlation momentum vs MR: {corr:.3f}")
print()
print(f"{'book':<26}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}")
print("-" * 51)
for name, r in [("Momentum (PIT)", mr_), ("Mean-reversion RSI2<10", rr_),
                ("SPY buy&hold", spy)]:
    a, s, d = stats(r)
    print(f"{name:<26}{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}")

print("-" * 51)
for w in [0.5, 0.4, 0.6]:
    comb = w * mr_ + (1 - w) * rr_
    a, s, d = stats(comb)
    print(f"{'combo '+str(int(w*100))+'/'+str(int((1-w)*100))+' mom/MR':<26}"
          f"{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}")

# vol-parity weighting (inverse-vol)
vm, vr = mr_.std(), rr_.std()
wm = (1/vm) / (1/vm + 1/vr)
comb = wm * mr_ + (1 - wm) * rr_
a, s, d = stats(comb)
print(f"{'combo inverse-vol':<26}{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}"
      f"   (wm={wm:.2f})")
