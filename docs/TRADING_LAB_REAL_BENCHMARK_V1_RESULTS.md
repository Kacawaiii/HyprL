# Trading Lab — real benchmark V1 results

**STATUS: FIRST FROZEN REAL-MARKET BENCHMARK RESULT**

These are the first real predictive scores HyprL has ever produced. The protocol
was registered and committed before any of them existed, and nothing in it was
touched afterwards. The numbers below are reported exactly as the runner emitted
them, including the periods where the models did worse than nothing.

## Provenance

| | |
|---|---|
| benchmark contract commit | `1c92eb95317db2794f468de100ebd01cfad12bfb` |
| runner commit | `54244bef023e3379220004f70d23e2fc922e5e1f` (committed **before** the first score) |
| corpus | `coinbase_history_v1`, Coinbase Exchange REST, 1h |
| corpus spec hash | `7a9de4d8331ece3856408636ad650a8dff44625a83c1a64fa9c134bc7627cdbd` |
| corpus content hash | `688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b` |
| date range | 2025-08-01T00:00:00Z → 2026-07-31T23:00:00Z (inclusive openings) |
| BTC benchmark spec hash | `dd7c474b34489ad3ca0c8ce906187821039fd9f0c50bff4a03b2da8cd20332a6` |
| ETH benchmark spec hash | `3d418bbbe6b0a418b2821fd44cbaa09273f8457f7bc08ee5c9c95d43956ad5f9` |
| environment | Python 3.12.3, scikit-learn 1.8.0, xgboost 3.1.3 |

The runner refuses to start unless the rebuilt spec hash, the corpus hashes and
the fold geometry all match what was frozen. All three matched.

## Protocol (unchanged, see TRADING_LAB_REAL_BENCHMARK_V1.md)

Target `forward_return` h=4 on 1h bars. Features, in this exact order:
`return_1`, `ema_12`, `ema_26`, `rsi_14`, `atr_14`. Expanding walk-forward with
min_train 720, validation 168, test 168, step 168, purge 4. Candidates: Ridge
(alpha 1.0) and XGBoost V1. Selection on the **validation** block only —
`rank_ic` → MAE → RMSE → `candidate_id` — then a fresh refit on train+validation
before the test block is touched. BTC-USD and ETH-USD are separate experiments;
their observations are never pooled.

Geometry actually executed, both products: 8750 series points, 10 declared gaps,
8663 usable rows, **46 folds**, minimum effective validation 164, **7728**
out-of-sample observations, all timestamps unique.

## Global out-of-sample results

| | BTC-USD | ETH-USD |
|---|---|---|
| `rank_ic` | **-0.003041** | **+0.004177** |
| MAE | 0.00686238 | 0.00989532 |
| RMSE | 0.01017800 | 0.01493139 |
| observations | 7728 | 7728 |
| folds | 46 | 46 |

Both rank correlations sit within a few thousandths of zero — BTC slightly
negative, ETH slightly positive. No threshold for "good" was defined before the
run and none is invented now.

## Model selection (validation only)

| | BTC-USD | ETH-USD |
|---|---|---|
| ridge folds won | 22 | 20 |
| xgboost folds won | 24 | 26 |
| decided by | {'rank_ic': 46} | {'rank_ic': 46} |
| winner changes | 23 | 14 |
| longest run | 5 | 8 |

Every fold was decided by the primary criterion (`rank_ic` on validation); the
MAE and RMSE fallbacks were never reached. The selected candidate alternates
frequently. No global winner is declared — that would be a second, unregistered
selection rule.

## Quarterly periods (calendar, fixed before the run)

BTC-USD:

| period | n | `rank_ic` | MAE | RMSE |
|---|---|---|---|---|
| P1 2025-08 → 2025-11 | 1261 | +0.007877 | 0.00554154 | 0.00810941 |
| P2 2025-11 → 2026-02 | 2208 | +0.026623 | 0.00677082 | 0.01021266 |
| P3 2026-02 → 2026-05 | 2136 | -0.015532 | 0.00840993 | 0.01213029 |
| P4 2026-05 → 2026-08 | 2123 | -0.015350 | 0.00618515 | 0.00902814 |

ETH-USD:

| period | n | `rank_ic` | MAE | RMSE |
|---|---|---|---|---|
| P1 2025-08 → 2025-11 | 1261 | +0.016306 | 0.00933940 | 0.01375774 |
| P2 2025-11 → 2026-02 | 2208 | +0.035384 | 0.01029412 | 0.01603486 |
| P3 2026-02 → 2026-05 | 2136 | -0.032978 | 0.01159988 | 0.01680623 |
| P4 2026-05 → 2026-08 | 2123 | +0.002126 | 0.00809575 | 0.01214661 |

Sign counts — BTC: 2 positive, 2 negative, 0 undefined.
ETH: 3 positive, 1 negative, 0 undefined.
No quarter is omitted, and the negative ones are not hidden.

## Sensitivity scenarios (diagnostic only)

| scenario | BTC `rank_ic` | ETH `rank_ic` |
|---|---|---|
| central (Ridge 1.0) | -0.003041 | +0.004177 |
| ridge_low (Ridge 0.5) | -0.004035 | +0.005565 |
| ridge_high (Ridge 2.0) | -0.002119 | +0.004392 |

Varying the ridge penalty moves the global figure by less than two thousandths.
No scenario is designated best; the ranking between them is not a selection.

## Reproducibility

Both products were run a second time from scratch: identical
`benchmark_results_hash`, dataset hash, selection and robustness result hashes,
prediction records, global metrics and selection counts. The global and quarterly
metrics were also recomputed independently from the stored prediction records
alone, without refitting anything, and matched exactly.

| | BTC-USD | ETH-USD |
|---|---|---|
| `benchmark_results_hash` | `0343014d81c0cef7ba65ec5a64d326f55bd92cc2d139596fd79fc32eb19902ae` | `db4a0e8198e8c650a87d918b759da6cf40aa8865a95a918ff3fc04188d92062d` |
| `dataset_hash` | `f37c0b05d4278f3f9da8de10db7593472d4c9ff1d855d7f05a13924f91ffdf7a` | `4417a1da8e692663c7c4a618e9016164dec77ccc26d58edb263e832c80b0f044` |

Every out-of-sample prediction is stored, so anyone can re-derive these numbers
without trusting the runner.

## What this does and does not establish

The predictive quality of the V1 feature set has now been **measured** on a real
frozen corpus under a pre-registered protocol. That is the claim. It is not a
claim that the quality is good: both global rank correlations are indistinguishable
from zero at this sample size, and the sign flips across quarters.

**Commercial profitability was not evaluated.** There is still no policy mapping a
prediction to a position — no threshold, no sizing, no turnover — and no fee or
slippage model. A `rank_ic` is not a P&L, and none is implied here.

## Data limitation

This is a **frozen historical Coinbase candle benchmark**, not a strict exchange
point-in-time revision backtest. `historical_candle_corpus = true`,
`point_in_time_exchange_revision_history = false`. Lookahead *between bars* is
prevented and enforced; what cannot be excluded is that a candle revised by the
exchange between its original hour and the 2026 capture carries the revised value.
That limitation is stated because it is true, not as a reason to discount results
that happen to be unexciting — it applies equally whichever way the numbers fell.

```
real_benchmark_v1_completed        = true
real_scores_observed               = true
benchmark_results_reproducible     = true
predictive_quality_measured        = true
positive_predictive_edge_established = undetermined (both global rank_ic ~ 0)
commercial_edge_established        = false
```
