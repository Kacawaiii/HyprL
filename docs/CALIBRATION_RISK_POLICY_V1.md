# Probability calibration and paper protection V1

This slice defines new policies beside the frozen model, risk, backtest and replay
references. It performs no real fitting, source request, external inference or order.
The demonstration and its API responses are explicitly synthetic. Actual model
calibration remains WAITING_AUTHORIZATION; prediction intervals remain absent.

The authoritative specification is
`artifacts/calibration_risk_policy_spec_v1.json`, revision 1, canonically bound in
`scripts/trading_lab/policies/spec.py`. A changed specification requires a new
revision and bindings; old artifacts remain rejected, without migration.

## Calibration

`ScoreObservation` binds a binary `forward_return_gt_zero` label to product,
model artifact, decision, horizon, label realization and actual label availability.
Scores must already lie in [0, 1]. A point return forecast is never silently
converted into a probability. The original score remains alongside its calibrated
output and calibration identity.

`fit_isotonic` uses pooled adjacent violators, pooling equal scores before pooling
decreasing blocks. Inference uses a right-continuous step function, clipping outside
the training score endpoints. It accepts only synthetic training observations,
with labels available strictly before fitting and validation. Validation and test
labels, duplicates and mixed model/product/horizon bindings are refused. Every
fold gets a new immutable artifact and training population digest. The demo reuses
the existing expanding `walk_forward.build_folds`, including its horizon purge.

V1 refuses fewer than 100 training decisions, fewer than 25 labels in either class,
or fewer than five distinct scores. These are declared operational floors, not
statistical guarantees. `evaluate_test` accepts unique, realized, available test
labels only. Diagnostics are refused below 50 decisions or 10 labels per class;
refusal yields null metrics and no diagram points, never a zero metric.

Reliability uses ten equal-width bins, including probability 1 in the last bin.
Empty bins carry null means and frequencies. Binary Brier is the mean squared
error of individual forecasts. The Murphy decomposition is exact for the
**bin-mean forecasts**:

`binned Brier = reliability - resolution + uncertainty`

The individual-forecast Brier also retains its binning residual:

`Brier = binned Brier + binning residual`

The residual is not suppressed or claimed nonnegative. The cockpit displays both
raw-model and calibrated Brier, and labels the decomposition as calibrated.
Training and diagnostic periods, populations and identities are visible in Expert.
The aggregate concatenates disjoint test blocks; a later expanding fold may train
on past observations, while every test probability retains its original fold.
Synthetic Brier changes provide no evidence of real calibration or an edge.

## TP/SL

`ProtectionPlan` binds an explicit entry fill and source prediction to product,
PAPER/SHADOW mode, side, horizon, policy version and both levels. Each valid strategy
level takes precedence independently. A strategy level needs its method, identity
and a provision time no later than entry. Invalid supplied levels are rejected,
without replacing them with defaults. Missing levels use the fixed, unoptimized
entry-distance method: TP 2%, SL 1%, mirrored for shorts. Levels bracket entry.
Every level exposes its value, origin, method, source identity and provision time.

The simulator reuses the native `market_series.SeriesPoint` OHLC shape and the
fees/adverse slippage in `economic_backtest.EXECUTION_SPEC_V1`. Entry is an already
specified fill, so entry slippage is not charged twice; fees apply to both legs.
Results are unit-position evidence, with no portfolio, funding or borrow claim.
Only complete bars starting at entry or later are considered; incomplete future
bars stay pending. OHLC crossings are observable at bar close; their exact intrabar
execution time is unknown. Observed bars receive a population digest.

- Opening through SL fills at the open, retaining adverse gaps.
- Opening through TP fills at TP, without favorable gap improvement.
- Touching both inside a bar closes at SL first and flags ambiguity; a separate
  TP-first sensitivity records the alternate result.
- A missing interval stops with NOT_OBSERVED; later prices cannot establish a
  fill across that gap and no P&L is invented.
- An untriggered position closes at the last complete bar of its stated horizon.

Both policies consult the registered per-product protected intervals even for
synthetic dates. They do not open price datasets or holdouts. Real fitting and
real-data simulation are disabled in this revision. No frozen prediction, model,
risk plan, result or ledger is written or reinterpreted.

## Reproduce and read

From the repository, with its Python environment active, choose a **new** runtime:

```bash
python -m scripts.trading_lab.policies.demo --root var/policy-demo-v1
python -m scripts.trading_lab.app_api.server --data-root var/empty-data \
  --policy-root var/policy-demo-v1 --host 127.0.0.1 --port 8787
```

Run `npm run dev` in `apps/web`, then open `/policies`. The sidebar and cockpit TP/SL
card link there, preserving the cockpit selection. Evidence is attributed to its
own synthetic model, independent of the selected frozen reference. Beginner shows
probabilities, reliability and level methods; Expert adds splits, decomposition,
costs and identities. Both show sample refusal, missing evidence and integrity
errors. No policy worker runs during an HTTP read.

Read-only V1 routes:

| Route | Result |
| --- | --- |
| `/api/v1/policies/definitions` | Specification, hash, sample floors and real-fitting authorization state |
| `/api/v1/policies/report` | Verified separate synthetic report; NOT_CONFIGURED without a runtime |
| `/api/v1/policies/report?product=BTC-USD` | Selected evidence with whole-report identity and selection hash |

GET/HEAD only; writes are refused. Old policy hashes, corrupt reports and mismatched
probability/level provenance fail with 409. Unavailable files fail with 503, malformed
queries with 400, unknown products with 404. Report size/population budgets are
bounded. Digests detect corruption; they are not signatures or source attestation.

`tests/policies` covers PAV, causality, bindings, refusal, Brier residuals, protected
dates, long/short fills, costs, gaps, ambiguity, missing bars, expiry, API integrity
and read-only behavior. The generated, native-shape synthetic web fixture has an
API drift guard. Refresh it with `python -m tests.policies.export_views` only when
the public synthetic contract changes. `apps/web/src/test/policies.test.tsx` covers
mode/selection persistence, diagrams, origins, refusal, loading, retry and absence.
The new Python suite is included in `sources-ci` without removing any existing gate.

For an optional local browser check, build the cockpit into `var/policy-web-dist`,
serve that build with the synthetic report using `--dist-root`, and run:

```bash
npm install --prefix var/policy-browser --no-package-lock --no-save playwright-core @sparticuz/chromium
node tests/policies/browser_check.mjs http://127.0.0.1:8787
```

This checks desktop/mobile layout, reliability, ambiguity, selection, mode and
keyboard focus with reduced motion. It refuses external hosts and aborts browser
requests outside the local API origin. Dependencies, temporary files and screenshots
stay ignored under `var/`.
