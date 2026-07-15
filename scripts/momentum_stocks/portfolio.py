#!/usr/bin/env python3
"""Decorrelation payoff: equity momentum + ETF cross-asset trend.

The ETF trend sleeve is weak standalone (Sharpe ~0.4) but if it is decorrelated
from the equity momentum sleeve, combining them lifts the portfolio Sharpe above
either leg (Carver's free lunch). This measures the correlation and the combined
book.
"""
import math
import numpy as np
import pandas as pd

import clenow_backtest as cb
import etf_trend as et

# momentum PIT daily equity
data = cb.load_prices()
membership = cb.load_membership()
idx_close = data["_GSPC"]["Close"] if "_GSPC" in data else data["SPY"]["Close"]
idx_ma = idx_close.rolling(cb.MA_INDEX).mean()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
m_eq = cb.run(data=data, feats=cb.precompute(stocks), index_close=idx_close,
              index_ma200=idx_ma, verbose=False, membership=membership)["_eqs"]

# ETF trend daily equity (full period)
etf_data = et.load()
t_eq, lev, nt = et._prep_and_run(etf_data, "2007-06-01", None)

# SPY
spy_eq = data["SPY"]["Close"]

# align
idx = m_eq.index.intersection(t_eq.index)
m = m_eq.reindex(idx).pct_change().dropna()
t = t_eq.reindex(idx).pct_change().dropna()
common = m.index.intersection(t.index)
m, t = m.loc[common], t.loc[common]
spy = spy_eq.reindex(common).ffill().pct_change().dropna()
common = m.index.intersection(spy.index)
m, t, spy = m.loc[common], t.loc[common], spy.loc[common]


def stats(r):
    sh = r.mean() / r.std() * math.sqrt(252)
    eq = (1 + r).cumprod()
    mdd = -((eq - eq.cummax()) / eq.cummax()).min()
    ann = eq.iloc[-1] ** (252 / len(r)) - 1
    return ann, sh, mdd


print("=== correlations (daily returns) ===")
print(f"  momentum vs ETF-trend : {np.corrcoef(m, t)[0,1]:+.3f}")
print(f"  momentum vs SPY       : {np.corrcoef(m, spy)[0,1]:+.3f}")
print(f"  ETF-trend vs SPY      : {np.corrcoef(t, spy)[0,1]:+.3f}")
print()
print(f"{'book':<28}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}")
print("-" * 53)
for name, r in [("Equity momentum (PIT)", m), ("ETF cross-asset trend", t),
                ("SPY buy&hold", spy)]:
    a, s, d = stats(r)
    print(f"{name:<28}{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}")
print("-" * 53)
# combined books
for w in [0.5, 0.6, 0.7]:
    comb = w * m + (1 - w) * t
    a, s, d = stats(comb)
    print(f"{'combo '+str(int(w*100))+'/'+str(int((1-w)*100))+' mom/trend':<28}"
          f"{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}")
# inverse-vol
vm, vt = m.std(), t.std()
wm = (1/vm) / (1/vm + 1/vt)
comb = wm * m + (1 - wm) * t
a, s, d = stats(comb)
print(f"{'combo inverse-vol':<28}{a*100:>9.1f}{s:>8.2f}{d*100:>8.1f}   (wm={wm:.2f})")

# recent regime only
mask = m.index >= pd.Timestamp("2022-01-01")
mr_, tr_ = m[mask], t[mask]
cr = np.corrcoef(mr_, tr_)[0, 1]
combr = 0.5 * mr_ + 0.5 * tr_
print(f"\nrecent 22-26: corr={cr:+.2f}  "
      f"mom Sh={stats(mr_)[1]:.2f}  trend Sh={stats(tr_)[1]:.2f}  "
      f"combo Sh={stats(combr)[1]:.2f}")
