#!/usr/bin/env python3
"""Vol-targeting (Moreira-Muir vol-managed) overlay on the PIT momentum returns.

Moreira & Muir 2017 JF: scale next-period exposure by the inverse of recent
realized variance. For momentum-type strategies the effect survives OOS
(Cederburg et al 2020). We apply it to the strategy's own daily returns:

    r_managed(t) = clip(c / vol_{t-1}, 0, LEV_CAP) * r(t)

vol_{t-1} = trailing realized vol (annualized) of strategy returns.
c is set so the managed series matches a chosen target vol full-sample.
"""
import math
import numpy as np
import pandas as pd

import clenow_backtest as cb

# --- build the honest PIT full-period daily equity curve ---
data = cb.load_prices()
idx_key = "_GSPC" if "_GSPC" in data else "SPY"
index_close = data[idx_key]["Close"]
index_ma200 = index_close.rolling(cb.MA_INDEX).mean()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
feats = cb.precompute(stocks)
membership = cb.load_membership()

r = cb.run(data=data, feats=feats, index_close=index_close,
           index_ma200=index_ma200, verbose=False, membership=membership)
eqs = r["_eqs"]
rets = eqs.pct_change().dropna()


def metrics(series_ret):
    mean, sd = series_ret.mean(), series_ret.std()
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else 0
    eq = (1 + series_ret).cumprod()
    peak = eq.cummax()
    mdd = -((eq - peak) / peak).min()
    ann = eq.iloc[-1] ** (252 / len(series_ret)) - 1
    dn = series_ret[series_ret < 0].std()
    sortino = mean / dn * math.sqrt(252) if dn and dn > 0 else 0
    return ann, sharpe, sortino, mdd


def vol_managed(rets, target_vol, vol_win=20, lev_cap=2.0):
    realized = rets.rolling(vol_win).std() * math.sqrt(252)
    scale = (target_vol / realized).shift(1).clip(0, lev_cap)  # lag to avoid look-ahead
    managed = (scale * rets).dropna()
    return managed


print(f"PIT base:  bars={len(rets)}  vol={rets.std()*math.sqrt(252)*100:.1f}%")
a, s, so, dd = metrics(rets)
print(f"{'config':<22}{'AnnRet%':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD%':>8}")
print("-" * 56)
print(f"{'base (no overlay)':<22}{a*100:>9.1f}{s:>8.2f}{so:>9.2f}{dd*100:>8.1f}")
for tv in [0.10, 0.12, 0.15, 0.20]:
    for cap in [1.0, 1.5, 2.0]:
        m = vol_managed(rets, tv, lev_cap=cap)
        a, s, so, dd = metrics(m)
        print(f"{'VT '+str(int(tv*100))+'% cap'+str(cap):<22}"
              f"{a*100:>9.1f}{s:>8.2f}{so:>9.2f}{dd*100:>8.1f}")
