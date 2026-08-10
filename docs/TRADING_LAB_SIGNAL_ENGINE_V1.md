# Trading Lab — signal engine V1

**THIS SIGNAL CONTRACT HAS NOT BEEN OPTIMIZED FOR PROFITABILITY.**

**SIGNAL STRENGTH IS NOT POSITION SIZE.**

A model that predicts `forward_return` has not yet said what to do. The step from
"the model predicts +0.4 %" to "buy" is where a backtest most easily acquires an edge
it did not earn, so it gets its own layer, its own frozen contract and its own hash
instead of living as an `if prediction > 0` inside a loop.

## Scope — and where it deliberately stops

```
Prediction → SignalDecision → [Phase 5B] PositionTarget → [Phase 5C] Execution
                    ▲
              this document
```

This layer produces a direction and a descriptive intensity. It knows nothing about
capital, position size, orders, brokers, turnover, fees, slippage, equity or profit,
and none of those words appear in its API. Mixing them in would make it impossible to
attribute any later number to the component that caused it.

## The contract

| | |
|---|---|
| rule | `static-symmetric-threshold-v1` |
| schema | `trading-lab.signal-engine.v1` |
| prediction horizon | 4 bars (matches the V1/V2 target) |
| long threshold | `+0.0025` |
| short threshold | `-0.0025` |
| full-strength excess | `0.01` |
| boundary semantics | `strict` |
| `spec_hash` | `7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939` |

### Direction

```
prediction >  +0.0025  → LONG
prediction <  -0.0025  → SHORT
otherwise              → FLAT
```

Strict inequalities. A prediction sitting **exactly** on a threshold is FLAT, and both
boundaries are tested at the boundary value itself.

### Strength

For a LONG or SHORT decision:

```
excess   = |prediction| - threshold
strength = min(1, excess / 0.01)
```

FLAT is always strength 0. So +0.25 % of threshold plus a further 1.00 % of predicted
move reads as full intensity, and anything beyond clamps at 1.

**Strength is not a capital fraction.** `strength = 0.8` means the prediction sits 80 %
of the way to the saturation excess. It does **not** mean 80 % of a portfolio.
Translating intensity into exposure requires position caps, volatility scaling and
portfolio state — that is the risk engine's job in Phase 5B, not this file's.

## Why these numbers, and what they are not

The thresholds are a plain symmetric rule chosen in advance for the infrastructure.
They were **not** derived from the observed V1 or V2 prediction distributions. That
corpus is spent: calibrating on results already seen would launder an observed outcome
into what looks like a design decision, and the resulting signal would inherit all the
selection bias of the experiments that produced it.

```
SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED = true
```

No claim is made that ±0.25 % is optimal, or even reasonable, for any asset. Any future
tuning belongs to a separate, pre-registered experiment with its own identity — never to
a quiet edit of this file.

## Causality guarantees

* **No labels.** `generate_signal` has no parameter through which an actual forward
  return could arrive, and `signal_from_prediction_record` reads only `bar_open_at` and
  `prediction`. A record whose `actual_forward_return` raises on access still produces a
  signal — that is a test, not a promise.
* **No calibration.** Nothing here is a quantile, a percentile, a z-score, a rolling
  standard deviation, a "top 10 %", or a Platt/isotonic fit. Every one of those would
  make the signal at T depend on data that arrives after T, or on the distribution of the
  very series being scored.
* **No clock.** The decision timestamp comes from the prediction. Replaying a historical
  record yields the identical decision and the identical hash, today or next year.
* **No global context.** A decision depends only on its own prediction, the frozen spec
  and its provenance. Appending later predictions leaves every earlier decision
  byte-identical.
* **No silent repair.** A batch with out-of-order or duplicated timestamps fails closed.
  Sorting the input would hide precisely the defect worth surfacing.

## Identity

`SignalSpec.spec_hash` covers the rule, both thresholds, the strength scale, the horizon
and the boundary semantics — the definition.

`SignalDecision.decision_hash` covers the spec hash, the timestamp, the prediction, the
direction, the strength, and the provenance triple (`source_model_spec_hash`,
`source_fitted_hash`, `source_benchmark_spec_hash`) — so a decision is traceable to the
exact model, the exact fitted state and the exact benchmark that produced its input.

`SignalSeries.series_hash` covers the ordered run, so a reordering is a different series.

All arithmetic is `Decimal` at precision 34, pinned inside
`localcontext`, and therefore independent of the caller's decimal context. NaN and
Infinity fail closed, as do malformed timestamps and provenance hashes.

## What this does not establish

No economic backtest exists. No paper trading exists. No prediction→position policy
exists. `commercial_edge_established` remains **false**, and this layer cannot change
that: it measures nothing.

It is also worth stating plainly that the V1 and V2 benchmarks found rank correlations
near zero. A signal layer built on top of a near-zero-information forecast will produce
near-random directions. This contract exists so that the machinery is correct and
auditable when a forecast worth acting on eventually arrives — not because one has.
