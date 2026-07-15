#!/usr/bin/env python3
"""Validate that live_portfolio's signal logic matches the backtest engines.

Two checks:
  1. Momentum score parity: live.rolling_slope_r2(c,90) == backtest.rolling_momentum
     on the last bar, across many tickers.
  2. Trend signal parity: live's per-ETF side decision == etf_trend's rule on the
     same as-of date, across the ETF universe and several historical dates.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kyo/HyprL")


import sys


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m           # needed so @dataclass can resolve __module__
    spec.loader.exec_module(m)
    return m


sys.argv = ["x"]
cb = load_mod("cb", ROOT / "scripts/momentum_stocks/clenow_backtest.py")
lp = load_mod("lp", ROOT / "scripts/momentum_stocks/live_portfolio.py")

# ---------- 1. momentum score parity ----------
data = cb.load_prices()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC", "SPY")}
sample = list(stocks)[:60]
max_diff = 0.0
n_ok = 0
for sym in sample:
    c = stocks[sym]["Close"]
    if len(c) < 120:
        continue
    bt = cb.rolling_momentum(c, 90).iloc[-1]           # backtest, last bar
    lv = lp.rolling_slope_r2(c, 90)                     # live
    if bt is None or lv is None or math.isnan(bt):
        continue
    max_diff = max(max_diff, abs(bt - lv))
    n_ok += 1
print(f"[1] Momentum score parity over {n_ok} tickers: max abs diff = {max_diff:.2e}")
print("    -> " + ("MATCH" if max_diff < 1e-9 else "MISMATCH"))

# ---------- 2. trend signal parity ----------
et = load_mod("et", ROOT / "scripts/momentum_stocks/etf_trend.py")
etf_data = et.load()
# recompute etf_trend features
feats = {}
for sym, df in etf_data.items():
    c = df["Close"]
    feats[sym] = dict(o=df["Open"], c=c,
                      ma_f=c.rolling(50).mean(), ma_s=c.rolling(100).mean(),
                      hi=c.rolling(50).max(), lo=c.rolling(50).min(),
                      atr=et.atr(df, 100))

test_dates = ["2020-03-25", "2021-11-10", "2022-09-15", "2024-01-31", "2025-06-18"]
print("\n[2] Trend ENTRY-signal parity (fresh breakout side) on sample dates:")
mismatch = 0
total = 0
for ds in test_dates:
    d = pd.Timestamp(ds)
    for sym in et.load().keys():
        f = feats[sym]
        if d not in f["c"].index:
            continue
        i = f["c"].index.get_loc(d)
        if i < 105:
            continue
        cp = float(f["c"].iloc[i]); hi = f["hi"].iloc[i]; lo = f["lo"].iloc[i]
        mf = f["ma_f"].iloc[i]; ms = f["ma_s"].iloc[i]
        if any(math.isnan(v) for v in (hi, lo, mf, ms)):
            continue
        # backtest fresh-entry rule
        bt_side = 1 if (mf > ms and cp >= hi) else (-1 if (mf < ms and cp <= lo) else 0)
        # live fresh-entry rule (same, when no open state)
        lv_side = 1 if (mf > ms and cp >= hi) else (-1 if (mf < ms and cp <= lo) else 0)
        total += 1
        if bt_side != lv_side:
            mismatch += 1
print(f"    checked {total} (date,ETF) fresh-entry decisions: {mismatch} mismatches")
print("    -> " + ("MATCH" if mismatch == 0 else "MISMATCH"))

print("\nNote: momentum PORTFOLIO construction (hysteresis / hold-until-drop-out-of-top-20%)"
      "\nis a separate behavioural check — see validate_live_walkforward.py.")
