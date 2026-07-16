#!/usr/bin/env python3
"""Final prop calibration: FTMO vs Trade The Pool rules on the HONEST momentum edge.

Uses the survivorship-corrected (point-in-time) momentum curve — not the
inflated current-list one — and Monte-Carlos both firms' rule sets across
position sizes.

FTMO 2-Step : +10% target | -10% static max loss | -5% daily
Trade The Pool (swing): +15% target | -7% static max loss | -3% daily
"""
import importlib.util
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

CURVE = ROOT / "data/momentum/momentum_pit_equity.csv"
if not CURVE.exists():
    print("building honest PIT momentum curve...")
    data = cb.load_prices()
    membership = cb.load_membership()
    idx = data["_GSPC"]["Close"] if "_GSPC" in data else data["SPY"]["Close"]
    stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
    eqs = cb.run(data=data, feats=cb.precompute(stocks), index_close=idx,
                 index_ma200=idx.rolling(200).mean(), verbose=False,
                 membership=membership)["_eqs"]
    eqs.to_frame("equity").to_csv(CURVE)
else:
    eqs = pd.read_csv(CURVE, index_col=0, parse_dates=True)["equity"]

rets = eqs.pct_change().dropna().values
vol = rets.std() * math.sqrt(252) * 100
ann = ((1 + rets.mean()) ** 252 - 1) * 100
sharpe = rets.mean() / rets.std() * math.sqrt(252)
peak = np.maximum.accumulate(eqs.values)
mdd = ((eqs.values - peak) / peak).min() * 100
print(f"HONEST momentum (PIT): ann {ann:.1f}%  vol {vol:.1f}%  "
      f"Sharpe {sharpe:.2f}  MaxDD {-mdd:.1f}%  Calmar {ann/-mdd:.2f}")

RULES = {
    "FTMO 2-Step":      dict(target=10.0, maxloss=10.0, daily=5.0),
    "Trade The Pool":   dict(target=15.0, maxloss=7.0,  daily=3.0),
}
N_PATHS = 3000
CAP = 750          # ~3 years of patience
BLOCK = 5


def sim(rets, k, rule, rng):
    n = len(rets)
    eq = 100.0
    day = 0
    while day < CAP:
        s = rng.integers(0, n - BLOCK)
        for r in rets[s:s + BLOCK]:
            day += 1
            prev = eq
            eq *= (1 + r * k)
            if (prev - eq) >= rule["daily"]:
                return False, day
            if eq <= 100 - rule["maxloss"]:
                return False, day
            if eq >= 100 + rule["target"]:
                return True, day
            if day >= CAP:
                break
    return False, day


for name, rule in RULES.items():
    print(f"\n=== {name}  (+{rule['target']:.0f}% / -{rule['maxloss']:.0f}% max "
          f"/ -{rule['daily']:.0f}% daily, no time limit) ===")
    print(f"{'size k':>7}{'vol%':>7}{'P(pass)':>10}{'med days':>10}{'med months':>12}")
    print("-" * 46)
    rng = np.random.default_rng(3)
    best = None
    for k in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        res = [sim(rets, k, rule, rng) for _ in range(N_PATHS)]
        p = sum(1 for ok, _ in res if ok) / len(res)
        d = [dd for ok, dd in res if ok]
        med = int(np.median(d)) if d else -1
        print(f"{k:>7.2f}{vol*k:>7.1f}{p*100:>9.1f}%"
              f"{med if med > 0 else '-':>10}{(med/21 if med > 0 else 0):>12.1f}")
        if best is None or p > best[1]:
            best = (k, p, med)
    print("-" * 46)
    print(f"BEST: k={best[0]} -> P(pass)={best[1]*100:.1f}%, "
          f"median {best[2]} days (~{best[2]/21:.1f} months)")
