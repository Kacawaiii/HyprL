#!/usr/bin/env python3
"""What Sharpe does a prop challenge actually require?

Grid over (Sharpe, size k) -> P(pass) and median days, under FTMO-style rules.
Answers: given our honest edge ceiling (~0.6-0.8), is a challenge viable, and
how long does passing take?
"""
import math
import numpy as np

TARGET, MAXLOSS, DAILY = 10.0, 10.0, 5.0
N = 3000
CAP = 400          # practical patience cap in trading days (~1.6 years)


def sim(sharpe, k, base_vol_ann, rng, limit):
    """Synthetic daily returns with the given annualised Sharpe."""
    vol_d = base_vol_ann / math.sqrt(252)
    mu_d = sharpe * base_vol_ann / 252
    eq = 100.0
    for day in range(1, limit + 1):
        r = (mu_d + vol_d * rng.standard_normal()) * k
        prev = eq
        eq *= (1 + r)
        if (prev - eq) >= DAILY:
            return False, day
        if eq <= 100 - MAXLOSS:
            return False, day
        if eq >= 100 + TARGET:
            return True, day
    return False, limit


def best_for(sharpe, limit, base_vol=0.12):
    rng = np.random.default_rng(11)
    best = None
    for k in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        res = [sim(sharpe, k, base_vol, rng, limit) for _ in range(N)]
        p = sum(1 for ok, _ in res if ok) / len(res)
        days = [d for ok, d in res if ok]
        med = int(np.median(days)) if days else -1
        if best is None or p > best[1]:
            best = (k, p, med)
    return best


for limit, label in [(CAP, f"patience {CAP}d (~1.6y)"), (60, "60-day limit"), (30, "30-day limit")]:
    print(f"\n=== {label} ===")
    print(f"{'Sharpe':>7}{'best k':>8}{'P(pass)':>10}{'med days':>10}")
    print("-" * 35)
    for s in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
        k, p, med = best_for(s, limit)
        print(f"{s:>7.1f}{k:>8.2f}{p*100:>9.1f}%{med if med > 0 else '-':>10}")
