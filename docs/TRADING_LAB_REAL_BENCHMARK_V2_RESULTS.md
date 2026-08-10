# Trading Lab — real benchmark V2 results (exploratory)

**STATUS: EXPLORATORY V2 RESULT**

**CONFIRMATORY RESULT: NO**

**CORPUS ALREADY OBSERVED UNDER V1: YES**

This is the second experiment run on the same Coinbase corpus. That corpus spent its
test blocks answering V1, so nothing below can establish an edge, whatever the numbers
say. A confirmatory answer requires the future window registered at the bottom of this
page, which has not happened yet and has not been downloaded.

## Provenance

| | |
|---|---|
| V2 contract commit | `4354e6be1520b662b3fdeeffd68e0ce3926d0e48` |
| runner commit | `937808f52e161f8a9f2b36d79d4eb434193ebaae` (committed **before** the first V2 score) |
| corpus | `coinbase_history_v1`, content hash `688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b` |
| corpus role | development/exploratory — already observed under V1 |
| BTC V2 spec hash | `ec4d19b55e36ea67119b31289d798adfd37cdb3b9e9ded73cc5fbcdd2c245d79` |
| ETH V2 spec hash | `db371b153824d337db1a848fef9a1658a603e84c1911fd3c9de68ab01b0566b0` |
| environment | Python 3.12.3, scikit-learn 1.8.0, xgboost 3.1.3 |

## The contract, unchanged from what was pre-registered

Features, in this exact order: `return_1`, `return_4`, `return_12`,
`ema_spread_12_26`, `rsi_14`, `atr_pct_14`. Target `forward_return` h=4 on 1h bars.
Walk-forward 720 / 168 / 168 / 168, purge 4, expanding. Ridge alpha 1.0 and XGBoost V1.
Selection `validation-rank-ic-mae-rmse-v1` on the validation block only, then a fresh
refit on train+validation. Robustness quarters P1–P4. BTC-USD and ETH-USD evaluated
separately, never pooled.

Geometry executed, both products: 8750 series points, 10 declared gaps, 8663 usable
rows, 46 folds, minimum effective validation 164, 7728 out-of-sample observations, all
timestamps unique — identical to V1, so the two are directly comparable.

## Global out-of-sample results

| | BTC-USD | ETH-USD |
|---|---|---|
| `rank_ic` | **-0.007970** | **-0.012406** |
| MAE | 0.00649862 | 0.00899958 |
| RMSE | 0.00963739 | 0.01354742 |
| observations | 7728 | 7728 |
| selection | ridge 24, xgboost 22 | ridge 20, xgboost 26 |
| decided by | {'rank_ic': 46} | {'rank_ic': 46} |
| winner changes | 23 | 25 |
| longest run | 7 | 6 |

Every fold was decided by the primary criterion; the MAE and RMSE fallbacks were never
reached.

## Quarterly periods (calendar, fixed before V1 ever ran)

BTC-USD:

| period | n | `rank_ic` | MAE | RMSE |
|---|---|---|---|---|
| P1 2025-08 → 11 | 1261 | -0.025484 | 0.00574852 | 0.00830910 |
| P2 2025-11 → 02 | 2208 | +0.005626 | 0.00653451 | 0.00985459 |
| P3 2026-02 → 05 | 2136 | +0.000011 | 0.00771846 | 0.01127837 |
| P4 2026-05 → 08 | 2123 | -0.050491 | 0.00567952 | 0.00825242 |

ETH-USD:

| period | n | `rank_ic` | MAE | RMSE |
|---|---|---|---|---|
| P1 2025-08 → 11 | 1261 | -0.025272 | 0.00906230 | 0.01338233 |
| P2 2025-11 → 02 | 2208 | +0.025818 | 0.00947100 | 0.01466549 |
| P3 2026-02 → 05 | 2136 | -0.028408 | 0.00999336 | 0.01464154 |
| P4 2026-05 → 08 | 2123 | -0.029875 | 0.00747215 | 0.01106052 |

BTC: 2 positive, 2 negative.
ETH: 1 positive, 3 negative.
Nothing is omitted, and the negative quarters are the majority.

## Sensitivity scenarios (diagnostic only)

| scenario | BTC `rank_ic` | ETH `rank_ic` |
|---|---|---|
| central (Ridge 1.0) | -0.007970 | -0.012406 |
| ridge_low (Ridge 0.5) | -0.007971 | -0.012435 |
| ridge_high (Ridge 2.0) | -0.007982 | -0.012362 |

The ridge penalty moves the global figure by less than a ten-thousandth. No scenario
is designated best.

## V1 ↔ V2, descriptively

No success threshold was pre-registered for either experiment, so this is a comparison
of numbers and nothing more. "Improved" below means only "moved in the stated
direction".

BTC-USD:

| metric | V1 | V2 | delta | direction |
|---|---|---|---|---|
| `rank_ic` | -0.003041 | -0.007970 | -0.004929 | moved further below zero |
| MAE | 0.00686238 | 0.00649862 | -0.000364 | numerically lower |
| RMSE | 0.01017800 | 0.00963739 | -0.000541 | numerically lower |

ETH-USD:

| metric | V1 | V2 | delta | direction |
|---|---|---|---|---|
| `rank_ic` | +0.004177 | -0.012406 | -0.016583 | moved further below zero |
| MAE | 0.00989532 | 0.00899958 | -0.000896 | numerically lower |
| RMSE | 0.01493139 | 0.01354742 | -0.001384 | numerically lower |

Quarterly sign patterns (P1→P4):

| | V1 | V2 |
|---|---|---|
| BTC-USD | `+ + − −` | `− + + −` |
| ETH-USD | `+ + − +` | `− + − −` |

Model selection counts:

| | V1 | V2 |
|---|---|---|
| BTC-USD | ridge 22 / xgboost 24 | ridge 24 / xgboost 22 |
| ETH-USD | ridge 20 / xgboost 26 | ridge 20 / xgboost 26 |

## Interpretation

The V2 hypothesis was that replacing nominal level features with relative,
dimensionless ones would change what the models can learn. On this exploratory corpus
the ranking metric did not improve: both global rank correlations moved further below
zero, and the quarterly signs remain mixed and unstable. The error metrics fell for
both assets — the relative representation produces numerically smaller absolute and
squared errors — but a lower MAE on a near-zero-correlation forecast is not evidence of
ranking skill, and the contract's primary criterion is the rank correlation.

Read against the pre-registered readings, this is case B: **the relative feature
representation did not produce important ranking power on this exploratory corpus.**

It remains true that neither experiment tested more than its own representation,
models, horizon and geometry. Neither shows that no predictive information exists in
hourly crypto.

## Reproducibility

Both products were re-run from scratch: identical `benchmark_results_hash`, dataset
hash, selection and robustness result hashes, prediction records, global metrics,
selection counts, periods, scenarios and experiment metadata. The global and quarterly
metrics were independently recomputed from the stored prediction records alone, without
refitting, and matched exactly.

| | BTC-USD | ETH-USD |
|---|---|---|
| `benchmark_results_hash` | `e810c0e24ea979310cb987df0c6a22ec15d137b913e0d1a13379dead5ac03beb` | `5421de013f27add40d2232da46f7ed371f850c2c66014884a5bf5e873816bd39` |
| `dataset_hash` | `4375db4ff2528702b6f25d130d9df2a6e40ac4b8ae4db2aebee5d22d9ef2639f` | `b3712a90d1d8c5ef038eea2d2b8cc09109874b8bb2d95be313a59f2fb780f904` |

Every out-of-sample prediction is stored, so these numbers can be re-derived without
trusting the runner.

## Still exploratory, and the holdout is still untouched

```
experiment_type       = exploratory
confirmatory_result   = false
corpus_role           = development/exploratory
```

The confirmatory window remains **2026-09-01T00:00:00Z → 2026-11-30T23:00:00Z**, BTC-USD
and ETH-USD, 1h — uncaptured, unobserved, single-use. It was not shortened, not moved
earlier, and nothing was downloaded for it. Having now seen two experiments on the same
corpus, the case for keeping that window sealed is stronger, not weaker.

No V3 is proposed here. Building one directly from these results would tighten the
research loop around a corpus that has already answered twice.

## Data limitation and scope

Frozen historical Coinbase candle benchmark, not a strict exchange point-in-time
revision backtest (`point_in_time_exchange_revision_history = false`). Lookahead
between bars is prevented and enforced; a candle revised by the exchange before the 2026
capture would carry its revised value.

No prediction→position policy, no threshold, no sizing, no turnover, no fees, no
slippage. **Commercial profitability was not evaluated**, and
`commercial_edge_established` remains false.
