# Trading Lab — Phase 3: predictive engine

Phase 3 puts real models on the causal data plumbing built in Phases 1 and 2.
It establishes that the predictive machinery is *sound*. It does **not**
establish that the models are profitable, and this document is careful to keep
those two claims apart.

## Pipeline

```
MarketSnapshot → MarketSeries → causal indicators → Dataset (forward_return, h=4, 1h)
    → expanding walk-forward folds (train / validation / test, purged)
    → validation-only model selection
    → fresh refit on train + validation
    → test out-of-sample records
    → robustness diagnostics
```

Every layer consumes only the proven layer below it. Nothing bypasses
`replay_snapshot`, and nothing re-implements fold construction or the metrics.

## Models (V1, frozen)

| candidate | module | notes |
|---|---|---|
| `ridge_regression` | `scripts/trading_lab/models.py` | alpha 1.0, cholesky solver, train-only standardisation |
| `xgboost_regression` | `scripts/trading_lab/models.py` | 100 trees, depth 3, lr 0.05, `subsample=1`, `colsample_bytree=1`, `n_jobs=1`, `random_state=0`, no scaling |

Neither configuration has been tuned. They are two candidates, not a shortlist
with a favourite.

**Numeric boundary.** Phases 1 and 2 are `Decimal` end to end. The solvers are
not: both models cross to `float64` for fitting and come back to `Decimal` at a
fixed exponent, pinned to the module's own precision so the result does not
depend on the caller's `decimal` context. Inference is pure `Decimal`. This is
Decimal on both banks of one float64 river — not Decimal end to end.

## Target

`forward_return`, horizon 4 bars, 1h timeframe. A label at T knows the close at
T+4, so the tail of every block is purged in **market time** before a temporal
boundary — not by counting rows, which gaps make unreliable.

## Selection rule — `validation-rank-ic-mae-rmse-v1`

Lexicographic and fully deterministic:

1. highest **validation** `rank_ic` (a defined value always beats an undefined one)
2. lowest validation MAE
3. lowest validation RMSE
4. lexicographic `candidate_id` — never a random tie-break

`rank_ic` is a Spearman rank correlation over one evaluation block for a single
instrument. It is **not** a cross-sectional information coefficient. An
undefined `rank_ic` stays `None`; it is never read as 0, which would be the
different claim "measured, and found no relationship".

A fold whose effective validation block holds fewer than **3** scorable
observations is refused. Over two non-constant points a Spearman correlation is
mechanically ±1, so the primary criterion would tie every time and MAE would
silently become the real selector. The check runs on the block actually produced
for the fold, never on the nominal parameters.

## Refit — `refit-train-plus-validation-v1`

After the choice, a **new** instance of the winner is fitted on train +
validation. This is permitted only when every validation label window closes
strictly before the test block opens, verified per fold in market time.

The model that faces test is therefore *not* the model that won validation.
Both fitted hashes are recorded separately (`selection_fit_hash`,
`final_fit_hash`) rather than blurred into one.

## Test data

The test block is never used to choose anything. `validate_candidates` and
`select` have no test parameter at all — there is no channel through which test
information could arrive. Test metrics are descriptive, computed after the
choice is final, and they never feed back into hyperparameters, the selection
rule, or the feature schema.

## Robustness (Phase 3D) — diagnostic only

Selection stability, validation geometry, selection margins and temporal
subperiod metrics. Subperiod boundaries are fixed in advance or derived from
timestamps alone, never from scores. Every subperiod metric is recomputed from
its own records; the official global figure stays the one computed over all
concatenated out-of-sample records and is never rebuilt by averaging.

There is deliberately **no** `best_candidate`, `majority_winner` or
`best_scenario`. Counting fold wins and then crowning a champion would be a
second selection rule, chosen after the test metrics were visible. The selected
candidate may legitimately differ from fold to fold.

## Dependencies — the `[ml]` contract

* **Core (Phase 1 + 2)** — `market_bar`, `coinbase_candles`, `market_data_store`,
  `market_snapshots`, `market_series`, `market_indicators`, `market_dataset`,
  `walk_forward` — requires **no** ML library. Importing any of them never pulls
  in scikit-learn or xgboost.
* **Phase 3** requires the optional extra: `pip install hyprl[ml]`.
  `scripts/trading_lab/models.py` is the single ML-coupled module; missing
  libraries raise an `ImportError` at import time naming the extra.

Test suites:

```bash
pytest -q tests/crypto                 # everything (needs [ml])
pytest -q tests/crypto -m "not ml"     # core only, no ML libraries needed
```

Phase 3 tests carry the `ml` marker at module granularity. A handful of pure
selection-rule tests would run without the extra; they are marked with the rest
rather than split hair-thin.

## Trading costs

**Not available.** There is no causal policy mapping a predicted forward return
to a position: no entry threshold, no turnover, no sizing rule. Applying fees to
a raw forward return would be arithmetic dressed up as a cost model. This waits
for a signal/portfolio layer.

## Technical correctness ≠ predictive edge

Phase 3 is technically closed: the pipeline is causal, deterministic, auditable
and free of the leakage paths it was built to exclude.

Predictive quality is **not** established. The repository tracks no reproducible
real historical dataset large enough to evaluate it — the tracked Coinbase
fixtures hold 2 and 1 candles respectively, which are unit-test fixtures, not
history. Every score quoted during Phase 3 (`rank_ic` around +0.8 for individual
models, around +0.6 post-selection) comes from deterministic *synthetic*
fixtures whose closes follow a periodic formula. Those numbers are **protocol
checks**: they show the machinery computes and compares correctly. They are not
evidence of an edge, of alpha, of expected performance, or of profitability, and
must never be quoted as such.
