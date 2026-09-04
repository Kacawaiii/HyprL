# US Equity Benchmark V1 — exploratory causal price-return benchmark

```
STATUS: EXPLORATORY

DAILY US EQUITY PRICE-RETURN BENCHMARK

NOT A TRADING STRATEGY

NOT A BACKTEST

NOT TOTAL RETURN

NO EXECUTION MODEL

NO COMMERCIAL EDGE CLAIM

FUTURE CONFIRMATORY HOLDOUT RESERVED AND UNOBSERVED
```

---

## 1. The question, and only that question

> Is there any measurable out-of-sample signal on a 5-session horizon, from a
> simple linear model, under a protocol frozen before the results were seen?

**Answer: no. No predictive edge established.**

Ridge loses to both fixed baselines on every instrument and on every pooled
metric. The macro rank IC is negative. Directional accuracy is below a coin
flip. This is a valid result, not a failed run — the leakage audit is clean
and the protocol executed exactly as frozen.

What it is not: a signal, a backtest, a P&L, a portfolio, or evidence about
any horizon, feature set or model other than the one specified below.

## 2. Source and analytical semantics

| | |
|---|---|
| source corpus | `yahoo_us_equity_daily_v1`, local, verified, gitignored |
| corpus content hash | `64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024` |
| corpus spec hash | `b7ad1e33b9896418e81f5e386ddc5384e0e2caca4854d467ac894eec25987dae` |
| calendar | `US_EQUITY_REGULAR`, `1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314` |
| exploratory range | 2024-08-01 → 2026-07-31 (501 sessions × 4 instruments) |
| analytical view | **SPLIT_ADJUSTED**, via the real `apply_splits` |
| dividends | **excluded** |

The target is a **split-adjusted price return**. Dividends are requested and
stored upstream in the hashed raw payload but are never applied here, so this
is *not* total shareholder return and the phrase is not used for any number in
this document.

Zero splits fall in this range, so the adjusted values equal RAW exactly
today. The *identity* still differs deliberately: a corpus that later contains
a split would otherwise destroy every return computed across it, and the
failure would look like a market event rather than a bug.

**Session-ordinal indexing.** The repository's indicators define a gap as "the
next bar is not one timeframe later", which is correct for crypto and wrong
for a market that closes every weekend. Measured on this corpus, applying them
verbatim fragments 501 sessions into 113 runs of at most five, and `return20`
never becomes computable at all. Bars are therefore projected onto the frozen
calendar's session sequence before any indicator sees them; the indicator
maths is reused unchanged. A genuinely missing session still breaks the run.

## 3. Protocol, frozen before any result existed

| spec | hash |
|---|---|
| research (binds all of the below + the data) | `be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177` |
| analytical view | `3dc527718ea389088b5e17ba4609a501604cea22b495249857a34296aedfb7bc` |
| features | `2d0fb6472704ba8ea4fa4429e8a253c4c31faf4f581ddf3d209f632ba67df591` |
| target | `60ba7d9346e55572bd790a7233635e4bf673fb1e486532e856073af8314183e0` |
| walk-forward | `a8b6418962eed360006bbc336967a819b24b6dcccf1d7ce2877b480fa8d50a65` |
| model | `cbbbe425a330c7fab8bd6ca1c266a6d3eebea0818f59addf7e774de4118c7162` |
| reserved holdout | `b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9` |

**Features** (six, dimensionless, causal): `return1`, `return5`, `return20`,
`ema_spread10_20`, `rsi14`, `atr_pct14`. No instrument identity, no calendar
position, no volume.

**Target**: `close[t+5] / close[t] - 1`, where `t+5` is five *trading
sessions* on the frozen calendar — never five calendar days.

**Walk-forward**: rolling train 252, purge 5, test 63, step 63. The purge
equals the target horizon, so the last training label cannot reach the test
block. No validation split, because nothing is selected.

**Model**: one `Ridge(alpha=1.0, solver=cholesky)` per instrument on a
`StandardScaler` fitted on train only. No pooling, no tuning, no CV.

**Baselines**: `ZERO` (predict 0) and `TRAIN_MEAN` (per-fold train mean).

## 4. Dataset

501 sessions → **476 eligible rows** per instrument (20 warm-up for
`return20`, 5 for the forward target), identical across all four because they
share one calendar. → **3 complete folds**, **189 OOS rows** per instrument,
**756 OOS rows** total. Incomplete folds are not produced; the final five
sessions get no fabricated target.

## 5. Results

Per instrument, out-of-sample, concatenated across the 3 folds:

| instrument | Ridge MAE | ZERO MAE | TRAIN_MEAN MAE | Ridge RMSE | ZERO RMSE | rank IC | dir. acc. |
|---|---|---|---|---|---|---|---|
| xnas:AAPL | 0.029677 | **0.027274** | 0.027228 | 0.035508 | **0.033515** | −0.0771 | 42.86 % |
| xnas:MSFT | 0.033546 | **0.030785** | 0.031052 | 0.045280 | **0.041906** | −0.0510 | 46.56 % |
| xnas:NVDA | 0.039784 | **0.037647** | 0.038193 | 0.049625 | **0.048094** | −0.0436 | 47.09 % |
| xnas:QQQ  | 0.021086 | 0.021264 | **0.020597** | 0.026855 | **0.026352** | +0.0706 | 52.38 % |

Aggregate over all 756 OOS rows:

| metric | Ridge | ZERO | TRAIN_MEAN |
|---|---|---|---|
| MAE | 0.031023 | **0.029243** | 0.029268 |
| RMSE | 0.040295 | 0.038363 | **0.038300** |
| macro rank IC | **−0.02527** | — | — |
| directional accuracy | **47.22 %** | — | — |

- **Ridge vs ZERO**: worse. MAE +6.1 %, RMSE +5.0 % relative.
- **Ridge vs TRAIN_MEAN**: worse. MAE +6.0 %, RMSE +5.2 % relative.
- 1 of 4 instruments has a positive rank IC (QQQ, +0.0706); the macro mean is
  negative.
- Directional accuracy is 47.22 %, below 50 %, with all 756 rows scored.

**No predictive edge established.** A single positive rank IC on one
instrument out of four, against a negative macro mean and a model that loses
to predicting zero, is noise and is reported as noise.

## 6. Leakage audit

Recomputed from the run itself, per instrument:

| check | AAPL | MSFT | NVDA | QQQ |
|---|---|---|---|---|
| purge violations | 0 | 0 | 0 | 0 |
| train/test overlap | 0 | 0 | 0 | 0 |
| duplicate OOS predictions | 0 | 0 | 0 | 0 |
| target horizon violations | 0 | 0 | 0 | 0 |
| OOS rows = distinct sessions | 189 = 189 | 189 = 189 | 189 = 189 | 189 = 189 |

Scaler and model fit sizes are asserted equal to the train block on every
fold, so neither ever saw a test row.

## 7. Limitations

- Two years, four large-cap US names, one horizon, one model, one alpha. A
  negative result here says nothing about other horizons, features, models or
  instruments — and no such variant was tried, deliberately.
- 756 OOS observations across 3 folds is small. The overlapping 5-session
  targets inside each test block are autocorrelated, so the effective sample
  is smaller than the row count suggests.
- The source is an unofficial provider and the corpus is not a strict
  point-in-time revision history.
- Survivorship: these four instruments were chosen in Phase 6D as liquid,
  currently-listed names. That is a selection, and it is not corrected for.
- No execution, cost, slippage or capacity assumption exists anywhere in this
  phase, so nothing here can be converted into an expected return.

## 8. Protocol spent; holdout reserved

This corpus and this spec are now **exploratory-spent** for this question:
the results have been observed, so they cannot serve as independent
confirmation of the same hypothesis again.

A separate **equity confirmatory holdout** was reserved *before* this run —
`equity_confirmatory_2027q1`, 2026-12-01 → 2027-02-28, single-use — and
remains `captured=false`, `observed=false`, `spent=false`. It has never been
requested, and any research range touching it is refused. The crypto
confirmatory holdout is likewise untouched.

No feature, horizon, alpha or metric was changed after the results were seen.
A V2 would require a new phase and a new justification, not a patch here.

`commercial_edge_established = false`.
