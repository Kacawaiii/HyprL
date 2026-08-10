# Trading Lab — real benchmark V2

**STATUS: FROZEN BEFORE FIRST V2 SCORE**

**EXPERIMENT TYPE: EXPLORATORY ON THE V1 CORPUS / CONFIRMATORY ONLY ON A FUTURE HOLDOUT**

No V2 predictive score has been computed. Not one fit, not one prediction, not one
metric. This document exists so that the second experiment is fully specified while
its outcome is still unknown.

## What V1 actually found

| | BTC-USD | ETH-USD |
|---|---|---|
| global OOS `rank_ic` | −0.003041 | +0.004177 |
| P1 2025-08 → 11 | +0.007877 | +0.016306 |
| P2 2025-11 → 02 | +0.026623 | +0.035384 |
| P3 2026-02 → 05 | −0.015532 | −0.032978 |
| P4 2026-05 → 08 | −0.015350 | +0.002126 |

Global rank correlations are close to zero and their signs are not stable across the
fixed calendar subperiods.

That is the whole claim. V1's technical integrity passed and its benchmark completed,
but it tested **one** representation, with two model configurations, one horizon and
one fold geometry. It does not show that no predictive information exists in hourly
crypto, and no statistical significance test was pre-registered, so none is asserted
here either.

V1 remains immutable. Its features, horizon, geometry, models, selection rule and
recorded results are untouched, including the quarters that came out negative.

## The V2 hypothesis

V1 fed the models several **nominal level** features: `ema_12`, `ema_26`, `atr_14`.
A level that reads as "high" in one price regime reads as "low" in another. Ridge
compensates partly through train-only standardisation, but a gradient-boosted tree
can learn a threshold on a raw level that simply does not survive the next regime.

V2 replaces those levels with **relative, dimensionless** quantities and changes
nothing else.

This is a hypothesis for a new experiment, not a diagnosis of V1. The V1 outcome has
not been attributed to this cause, and no V2 feature was chosen by measuring anything
against the target on the observed corpus.

## What changes, and what does not

Changed — the feature representation, in this exact order:

| # | column | indicator | parameters | why |
|---|---|---|---|---|
| 1 | `return_1` | `simple_return` | — | 1-bar momentum (unchanged from V1) |
| 2 | `return_4` | `return_over_period` | period=4 | momentum at the label's own horizon |
| 3 | `return_12` | `return_over_period` | period=12 | slower momentum |
| 4 | `ema_spread_12_26` | `ema_spread` | fast=12, slow=26 | trend, normalised by the slow EMA |
| 5 | `rsi_14` | `rsi` | period=14 | already dimensionless in V1 |
| 6 | `atr_pct_14` | `atr_percent` | period=14 | volatility as a fraction of price |

Unchanged — everything else, so the delta is exactly one thing:

* products `BTC-USD` and `ETH-USD`, evaluated separately, never pooled
* timeframe 1h, target `forward_return`, horizon 4
* walk-forward 720 / 168 / 168 / 168, purge 4, expanding
* Ridge alpha 1.0 and XGBoost V1, byte-identical configurations
* selection `validation-rank-ic-mae-rmse-v1`, floor 3 validation observations
* refit `refit-train-plus-validation-v1`
* robustness quarters P1–P4 and the three sensitivity scenarios

`simple_return` keeps its parameterless identity precisely because V1 committed to
its spec hash; `return_over_period` is a separate primitive that agrees with it
exactly at period=1, so the repository holds one definition of a one-bar return
reachable under two identities.

## Geometry on the V1 corpus (measured, no model fitted)

Both products: 8750 series points, 10 declared gaps, 8663 usable rows, **46 folds**,
every effective validation block 164 rows, **7728** out-of-sample observations, all
timestamps unique, no label window crossing a block boundary.

Identical to V1's geometry — warm-up is still dominated by the 26-period EMA — so the
two experiments are directly comparable.

## Identity

| | `benchmark_spec_hash` |
|---|---|
| BTC-USD V2 | `ec4d19b55e36ea67119b31289d798adfd37cdb3b9e9ded73cc5fbcdd2c245d79` |
| ETH-USD V2 | `db371b153824d337db1a848fef9a1658a603e84c1911fd3c9de68ab01b0566b0` |

Both differ from their V1 counterparts. The corpus hashes are bound into the
identity, as is the exploratory status and the future holdout definition.

## The V1 corpus is spent

```
V1_CORPUS_ROLE_FOR_V2 = "development/exploratory"
confirmatory_holdout  = false
```

The window 2025-08-01 → 2026-07-31 has already surrendered its test blocks under V1.
It cannot also serve as an untouched holdout for a hypothesis formed *after* seeing
those results. Any V2 number computed on it must be labelled:

```
exploratory_v2_result = true
confirmatory_result   = false
```

That holds even if V2 returns a `rank_ic` of +0.50. There is no exception, and a
large exploratory number is a reason to run the confirmatory test, never a substitute
for it.

## Future confirmatory holdout — registered now

| | |
|---|---|
| holdout id | `coinbase_confirmatory_2026q4` |
| window | 2026-09-01T00:00:00Z → 2026-11-30T23:00:00Z |
| products | BTC-USD, ETH-USD |
| timeframe | 1h |
| role | confirmatory |
| captured | **no** — this window has not happened yet, and nothing was downloaded |

After 2026-11-30: capture once, freeze the corpus exactly as Phase 4A did, and run V2
**exactly as registered above**. If V2 changes before then, the holdout must be
re-pointed at the new version explicitly — a contract cannot be validated against
data by a protocol it did not predate.

### One use only

Once that window is evaluated it is spent too. Any further hypothesis needs a fresh
holdout or must be labelled exploratory. This rule is written down now, before there
is any temptation to bend it.

## No promise

Nothing here predicts that V2 will do better than V1. It may do worse. The point of
pre-registering it is that either outcome will be reportable without argument.

## Still out of scope

No prediction→position policy, no threshold, no sizing, no turnover, no fees, no
slippage, no P&L. `commercial_edge_established` remains **false** and is not a
question this experiment can answer.
