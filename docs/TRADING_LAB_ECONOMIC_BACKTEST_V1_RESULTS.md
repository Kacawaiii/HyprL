# Trading Lab — economic backtest V1 results

**STATUS: EXPLORATORY ECONOMIC BACKTEST**

**SOURCE PREDICTIONS: V2 EXPLORATORY** (already observed once, on a spent corpus)

**EXECUTION MODEL: SYNTHETIC, NOT BROKER-SPECIFIC**

**COMMERCIAL EDGE: NOT ESTABLISHED**

These are simulated portfolios, not trades. Nothing here was executed, no money
moved, and no broker was involved.

## Provenance

| | |
|---|---|
| engine commit | `9dc86f2e442df65fcc9b324d3c3aef34f5ff2db0` (committed **before** the first P&L) |
| source predictions | Benchmark V2 exploratory, commit `4900eaeff49001ecaa8b7b5f5bcde26a7fe74b62` |
| corpus | `coinbase_history_v1`, content `688c250d…` |
| execution spec | `99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb` |
| signal spec | `7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939` |
| risk spec | `f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad` |
| window | 2025-09-08T02:00Z → 2026-07-29T22:00Z, 7787 hourly marks |

Costs: 10 bps fee per fill, 5 bps slippage against the trade, 100,000 USD initial
equity. Synthetic infrastructure figures — not any exchange's schedule, not tuned,
not an account.

## Results

| | BTC-USD | ETH-USD |
|---|---|---|
| initial equity | 100,000.00 | 100,000.00 |
| final equity (net) | 96,338.20 | 95,417.96 |
| **net return** | **-3.6618 %** | **-4.5820 %** |
| gross return (costs removed) | -1.5792 % | +1.2379 % |
| net P&L | -3,661.80 | -4,582.04 |
| gross P&L | -1,579.25 | 1,237.91 |
| total fees | 1,404.67 | 3,853.96 |
| slippage cost | 702.33 | 1,926.98 |
| total execution cost | 2,107.00 | 5,780.94 |
| turnover ratio | 14.26 | 39.47 |
| max drawdown | -3.8524 % | -4.5895 % |
| annualized Sharpe | -3.2578 | -2.6410 |
| fills | 554 | 1082 |
| rebalances offered | 7728 | 7728 |
| expired targets (gap) | 0 | 0 |
| average absolute exposure | +0.2304 % | +0.4666 % |
| time in market | +5.2780 % | +9.9653 % |
| results hash | `4617c6151da9cb6560299149047e6d7954c81e9699b16b044afb48991551bb77` | `47f0e8e324d4b58b28cf15e29ceb2bca25f434279ef6ac915a4bd4f429063bf1` |

Sharpe is descriptive, annualised by √8760 on hourly net-equity returns, risk-free
rate zero. No significance is claimed and none is testable from one path.

## What the numbers say

Both products end **net negative**: -3.6618 % on BTC-USD and
-4.5820 % on ETH-USD.

The two products fail differently, and the difference is the interesting part:

* **BTC-USD** was already losing before costs — gross -1.5792 % —
  and costs of 2,107.00 deepened it.
* **ETH-USD** was slightly *positive* before costs — gross +1.2379 %,
  about 1,237.91 — and 5,780.94 of fees and
  slippage turned that into -4.5820 %. Execution cost alone is roughly
  4.7× the gross gain.

Turnover is the mechanism: 14.3× and
39.5× equity traded over eleven months, at 15 bps of
round-trip friction per unit of notional, against a signal whose measured rank
correlation was near zero.

The strategy is also barely invested — average absolute exposure
+0.2304 % (BTC) and +0.4666 % (ETH),
in the market +5.2780 % and +9.9653 %
of the time. The Signal V1 threshold is rarely crossed, so most bars hold nothing
while the few that trade still pay full friction.

No target expired: every one of the 7728 predictions had a contiguous next bar.

## Technical engine quality vs strategy economic quality

These are different questions and this document keeps them apart.

* **Engine**: deterministic (both products replayed byte-identically from scratch),
  independently re-auditable (final equity, fees, slippage, net return, drawdown and
  turnover all recomputed from the stored fills and equity curve alone, matching
  exactly), causally bounded (no same-bar fill, no stale post-gap execution), and
  fail-closed on insolvency. 8 engine mutants killed.
* **Strategy**: loses money under these costs.

A negative net return is not a failure of the backtester. A backtester that
reported a profit here would be the thing to distrust.

## What this does not establish

The predictions are V2 **exploratory**, computed on a corpus already observed under
V1. The signal threshold was never optimised. The risk cap was never optimised. The
cost model is synthetic. There is no confirmatory test, no broker, no live
execution, and no position/turnover policy tuned for anything.

```
economic_backtest_v1_completed = true
economic_results_reproducible  = true
confirmatory_economic_result   = false
commercial_edge_established    = false
```

Improving this result by lowering the fee assumption, raising the threshold, or
retuning the risk cap would be fitting the strategy to a corpus that has now been
observed three times. Any such change is a new, explicitly labelled experiment.
