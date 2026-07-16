#!/usr/bin/env python3
"""Monte-Carlo a prop-firm challenge against our real strategy returns.

A challenge is a BARRIER problem, not a return problem: hit +target before the
max-drawdown barrier. For a drifted random walk, P(hit +a before -b) depends on
mu/sigma^2 — so scaling position size by k sends mu/sigma^2 -> mu/(k sigma^2).
=> Trading SMALLER raises P(pass) when there is no time limit. This script
measures that on our actual return distribution instead of assuming normality.

Rules modelled (FTMO-style, phase 1):
  * pass  : equity >= 100 + TARGET_PCT
  * fail  : equity <= 100 - MAX_LOSS_PCT      (static, from initial balance)
  * fail  : intraday/daily loss > DAILY_LOSS_PCT of initial balance
  * optional time limit (0 = unlimited, FTMO current)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kyo/HyprL")

TARGET_PCT = 10.0
MAX_LOSS_PCT = 10.0
DAILY_LOSS_PCT = 5.0
N_PATHS = 4000
MAX_DAYS_CAP = 2000          # hard cap so unlimited-time paths terminate
BLOCK = 5                    # block bootstrap length (preserve autocorrelation)


def load_returns(which="trend"):
    if which == "trend":
        p = ROOT / "data/momentum/etf_trend_equity.csv"
        eq = pd.read_csv(p, index_col=0, parse_dates=True)["equity"]
    else:
        p = ROOT / "data/momentum/equity_curve.csv"
        eq = pd.read_csv(p, index_col=0, parse_dates=True)["equity"]
    return eq.pct_change().dropna().values


def simulate(rets, k, time_limit=0, rng=None):
    """Return (passed, days). One challenge path with size multiplier k."""
    rng = rng or np.random.default_rng()
    n = len(rets)
    eq = 100.0
    day = 0
    limit = time_limit if time_limit > 0 else MAX_DAYS_CAP
    while day < limit:
        start = rng.integers(0, n - BLOCK)
        for r in rets[start:start + BLOCK]:
            day += 1
            day_start = eq
            eq *= (1 + r * k)
            # daily loss rule (vs initial balance)
            if (day_start - eq) >= DAILY_LOSS_PCT:
                return False, day
            if eq <= 100 - MAX_LOSS_PCT:
                return False, day
            if eq >= 100 + TARGET_PCT:
                return True, day
            if day >= limit:
                break
    return False, day  # ran out of time (or cap)


def sweep(which="trend", time_limit=0):
    rets = load_returns(which)
    ann_vol = rets.std() * math.sqrt(252) * 100
    ann_ret = ((1 + rets.mean()) ** 252 - 1) * 100
    sharpe = rets.mean() / rets.std() * math.sqrt(252)
    print(f"strategy '{which}': ann {ann_ret:.1f}%  vol {ann_vol:.1f}%  Sharpe {sharpe:.2f}")
    print(f"rules: +{TARGET_PCT:.0f}% target | -{MAX_LOSS_PCT:.0f}% max loss | "
          f"-{DAILY_LOSS_PCT:.0f}% daily | time limit: "
          f"{'none' if time_limit == 0 else str(time_limit)+'d'}\n")
    print(f"{'size k':>7}{'implied vol%':>14}{'P(pass)':>10}{'P(fail)':>10}"
          f"{'med days (pass)':>17}")
    print("-" * 58)
    rng = np.random.default_rng(7)
    out = []
    for k in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        res = [simulate(rets, k, time_limit, rng) for _ in range(N_PATHS)]
        passed = [d for ok, d in res if ok]
        p = len(passed) / len(res)
        med = int(np.median(passed)) if passed else -1
        out.append((k, p, med))
        print(f"{k:>7.2f}{ann_vol*k:>14.1f}{p*100:>9.1f}%{(1-p)*100:>9.1f}%"
              f"{med if med > 0 else '-':>17}")
    best = max(out, key=lambda x: x[1])
    print("-" * 58)
    print(f"best: k={best[0]} -> P(pass)={best[1]*100:.1f}%, median {best[2]} days")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "trend"
    print("=" * 58)
    print("NO TIME LIMIT (FTMO current)")
    print("=" * 58)
    sweep(which, time_limit=0)
    print("\n" + "=" * 58)
    print("30-DAY TIME LIMIT (older / stricter firms)")
    print("=" * 58)
    sweep(which, time_limit=30)
