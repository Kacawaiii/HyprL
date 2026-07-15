#!/usr/bin/env python3
"""Cost + parameter stress test for the Connors MR strategy.

The strategy trades ~19k times, so it lives or dies on transaction costs.
Test full-period AND recent-regime Sharpe across cost levels and position
counts. If it survives 15-20 bps round-trip it is deployable; if it dies at
10 bps it is a backtest artifact.
"""
import math
import clenow_backtest as cb
import mr_backtest as mr

data = cb.load_prices()
membership = cb.load_membership()
stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
feats10 = mr.precompute(stocks, rsi_entry=10.0)

WINS = [("FULL", None, None), ("recent 22-26", "2022-01-01", "2026-07-11")]

print("=== COST STRESS (RSI2<10, max 10 pos, 10% each) ===")
print(f"{'cost/side':<12}{'FULL Shp':>10}{'FULL Ann%':>11}{'recent Shp':>12}{'trades':>9}")
print("-" * 54)
for cost in [5.0, 10.0, 15.0, 20.0, 30.0]:
    mr.COST_BPS = cost
    row = {}
    for name, ws, we in WINS:
        r = mr.run(data, feats10, membership, ws, we, rsi_entry=10.0)
        row[name] = r
    print(f"{cost:<12.0f}{row['FULL']['sharpe']:>10}{row['FULL']['ann']:>11}"
          f"{row['recent 22-26']['sharpe']:>12}{row['FULL']['trades']:>9}")

mr.COST_BPS = 10.0  # realistic large-cap round-trip ~10bps
print("\n=== POSITION-COUNT SENSITIVITY (cost=10bps) ===")
print(f"{'max_pos/wgt':<14}{'FULL Shp':>10}{'FULL Ann%':>11}{'MaxDD%':>9}{'recent Shp':>12}")
print("-" * 56)
for maxp, wgt in [(5, 0.20), (10, 0.10), (15, 0.0667), (20, 0.05)]:
    mr.MAX_POS = maxp
    mr.WEIGHT_PER = wgt
    rf = mr.run(data, feats10, membership, None, None, rsi_entry=10.0)
    rr = mr.run(data, feats10, membership, "2022-01-01", "2026-07-11", rsi_entry=10.0)
    print(f"{str(maxp)+' / '+str(int(wgt*100))+'%':<14}{rf['sharpe']:>10}"
          f"{rf['ann']:>11}{rf['maxdd']:>9}{rr['sharpe']:>12}")
