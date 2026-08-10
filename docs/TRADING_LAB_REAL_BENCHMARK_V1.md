# Trading Lab — real benchmark V1

**STATUS: FROZEN BEFORE FIRST REAL PREDICTIVE SCORE**

**NO REAL PREDICTIVE SCORE HAS BEEN OBSERVED UNDER THIS CONTRACT.**

This document fixes the first real-market experiment completely — data, target,
features, geometry, candidates, selection rule and diagnostic periods — while the
result is still unknown to everyone, including the author. That ordering is the
entire point. Choosing a fold size or a feature set after glimpsing a rank
correlation is the oldest way to manufacture a backtest, and writing the choice
into git beforehand is the only defence that survives review.

Any change to what follows **after** a score has been observed does not produce a
corrected V1. It produces V2, and it is a different experiment. V1 is never
rewritten.

## Corpus binding

| | |
|---|---|
| corpus | `coinbase_history_v1` (Phase 4A, Coinbase Exchange REST) |
| `corpus_spec_hash` | `7a9de4d8331ece3856408636ad650a8dff44625a83c1a64fa9c134bc7627cdbd` |
| `corpus_content_hash` | `688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b` |
| range | 2025-08-01T00:00:00Z → 2026-07-31T23:00:00Z, inclusive openings |
| rows | 8750 per product, 10 declared gaps per product, never filled |

Both hashes enter the benchmark identity. Substituting the corpus changes
`benchmark_spec_hash`, so a "V1 result" cannot quietly come to mean two things.

## Products — two separate experiments

`BTC-USD` and `ETH-USD` are evaluated **independently**. No pooling of
out-of-sample observations, no cross-asset features, no combined "crypto" score.
Identical protocol, different subject.

| product | `benchmark_spec_hash` |
|---|---|
| BTC-USD | `dd7c474b34489ad3ca0c8ce906187821039fd9f0c50bff4a03b2da8cd20332a6` |
| ETH-USD | `3d418bbbe6b0a418b2821fd44cbaa09273f8457f7bc08ee5c9c95d43956ad5f9` |

## Target

`forward_return`, horizon **4 bars** on the **1h** timeframe — the Phase 2C
contract, unchanged. A label at T reaches T+4, so block boundaries are purged in
market time and no label window crosses one.

## Features — exactly five, in this order

The order is part of the contract. It is not sorted, and `ModelSpec` receives
this exact schema.

| # | column | indicator | parameters |
|---|---|---|---|
| 1 | `return_1` | `simple_return` | — |
| 2 | `ema_12` | `ema` | period=12 |
| 3 | `ema_26` | `ema` | period=26 |
| 4 | `rsi_14` | `rsi` | period=14 |
| 5 | `atr_14` | `atr` | period=14 |

The intent is coarse coverage — short return, fast trend, slower trend, momentum,
volatility — not a feature zoo. `return_1` is the one-bar causal return added as a
versioned indicator in the preceding commit; it reuses the semantics already
established for `FeaturePoint.simple_return` rather than introducing a second
definition. No imputation, no forward-fill; unusable rows are excluded by the
existing Phase 2C rule.

## Walk-forward geometry

| field | value |
|---|---|
| `min_train_rows` | 720 (30 days of hourly bars) |
| `validation_rows` | 168 (7 days) |
| `test_rows` | 168 (7 days) |
| `step_rows` | 168 |
| `purge_rows` | 4 |

Expanding window, no shuffling. `step_rows == test_rows`, so out-of-sample windows
never overlap. `purge_rows` is a floor: the Phase 2D market-time purge remains the
real authority, and it is what guarantees no label window crosses a boundary.

Measured on the frozen corpus with the five features above (`build_folds` only —
no model was fitted): **46 folds per product**, every effective validation block
**164 rows** (far above the required 3), **7728** out-of-sample observations per
product, all timestamps unique. Adding `return_1` did not change this geometry —
warm-up is dominated by `ema_26`, and that was measured rather than assumed.

## Candidates

Two, both frozen in Phase 3 and untouched here:

* **Ridge** — alpha 1.0, cholesky solver, train-only standardisation
* **XGBoost** — V1 config (100 trees, depth 3, lr 0.05, `subsample=1`,
  `colsample_bytree=1`, `n_jobs=1`, `random_state=0`), no scaling

No hyperparameter is tuned in this contract.

## Selection — `validation-rank-ic-mae-rmse-v1`

Validation block only, lexicographic and deterministic: highest validation
`rank_ic` → lowest MAE → lowest RMSE → lexicographic `candidate_id`. A defined
correlation always beats an undefined one, and an undefined one is never read as
zero. Folds with fewer than **3** effective validation observations are refused.

Refit policy `refit-train-plus-validation-v1`: after the choice, a fresh
instance of the winner is fitted on train + validation, permitted only once every
validation label window is proven to close before the test block opens. The model
that faces test is therefore not the model that won validation, and both fitted
hashes are recorded separately.

The test block never participates in the choice.

## Robustness periods — calendar quarters

* P1 `[2025-08-01, 2025-11-01)`
* P2 `[2025-11-01, 2026-02-01)`
* P3 `[2026-02-01, 2026-05-01)`
* P4 `[2026-05-01, 2026-08-01)`

Calendar boundaries, fixed before any score, never redrawn to balance observation
counts or to move a disappointing stretch out of view. Gaps stay where they are.

## Sensitivity scenarios — diagnostic only

| scenario | Ridge alpha | XGBoost |
|---|---|---|
| `central` | 1.0 | V1 |
| `ridge_low` | 0.5 | V1 |
| `ridge_high` | 2.0 | V1 |

One axis, three points. There is no `best_scenario` and none will be added:
picking the best-scoring scenario would be tuning on test wearing a diagnostic's
clothes.

## Anti-peek policy

`scripts/trading_lab/real_benchmark.py` is declarative. It has no
`run_benchmark`, and it never calls `fit`, `predict`, `evaluate_selection` or any
metric function — asserted both by source inspection and by a test that replaces
every one of those surfaces with a function that raises before building the specs.
Reading a candidate's `model_spec_hash` requires constructing a predictor, which
is not training one.

## Point-in-time limitation, restated

The corpus is a historical candle capture, not a point-in-time record of exchange
revisions (`historical_candle_corpus = true`,
`point_in_time_exchange_revision_history = false`). The benchmark prevents
lookahead across bars; it cannot prove that a later correction to a candle was not
already folded into the captured value. That is a limitation of the data, not a
causal defect in the engine.

## Status

```
benchmark_contract_v1_frozen   = true
real_scores_observed           = false
predictive_quality_established = false
commercial_edge_established    = false
```
