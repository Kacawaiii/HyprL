# Research registry and Model Observability v1

This offline slice consumes [CONTRACTS_V1](CONTRACTS_V1.md), [Model Lab](MODEL_LAB_V1.md),
[COMPARISON_PROTOCOL_V2](COMPARISON_PROTOCOL_V2.md), the existing paper engine, its frozen
signal/risk/execution policies, and per-product research protection. It enables no capture,
real training, external model call, broker, or real order. Private state stays under ignored `var/`.

## Registry and bounded proposals

`research.proposals.propose(dataset)` deterministically prepares at most two synthetic experiments:
the existing synthetic Ridge and the local momentum adapter. It calls Model Lab's existing
`prepare_experiment`; it never performs training. Each immutable `Hypothesis` records the statement,
mechanism, falsification, sources, features, target, horizon, population, baselines, chronological
splits with purge/embargo, train-only transformations, costs, calendar, availability, exclusions,
research-protection binding, frozen decision criteria and bounded research budgets.

The complete plan and its decision criteria have separate canonical SHA-256 identities. Register
the plan and reserve a PREPARED trial before submitting a worker job. Reservations consume a
persistent budget transactionally, including concurrent attempts and abandoned trials. The trial
must match the registered dataset, splits, baselines, costs and criteria. At most eight trials per
hypothesis, 10,000 rows and 180 seconds per workload are admitted; the proposal engine uses a
600-row ceiling and inherits Model Lab's tighter worker bounds. Registry preparation grants no
research permission. Synthetic experiments remain exploratory.

Trial states append: PREPARED, RUNNING, COMPLETE, FAILED, ABANDONED, BLOCKED. Outcomes retain
POSITIVE, NULL, NEGATIVE, ABANDONED, ERROR and PENDING. Terminal trials cannot be rewritten or
reopened; further work needs another budgeted trial. Histories preserve the original preparation
hash and criterion hash. A failed criterion is retained, with its metrics, rather than hidden.
The HTTP surface is read-only; this developer API is not an untrusted workload execution service.

The comparison plan consumes protocol V2's exact hash, features, criteria, costs, splits, roles and
budgets. The hypothesis scalar horizon is the primary crypto horizon; the exploratory equity
session rules remain in the bound protocol. Crypto stays primary; equities remain exploratory without a verdict. There is no execution
path for this comparison in this slice. `pair_decisions` refuses mismatched populations, duplicate
keys, labels, availability clocks, splits or costs. It does not repair a comparison by silently
intersecting populations. The readiness diagnostic retains WAITING_DATA, missing causal overlap,
operator review and scoped authorizations, with concrete next actions. Backfills cannot invent
historical availability; already studied windows remain exploratory and holdouts remain closed.

## Immutable prediction ledger

`ResearchStore.issue` preserves the exact shared `PredictionRecord` and its fingerprint. The input
evidence separately binds its actual feature values, snapshot, selected event identities, baselines,
input quality, split and provenance. Issuing another output or input evidence under the same identity
is refused. Methodless confidence is refused; absent uncertainty remains null.

Labels use the shared `LabelRecord`, bound to prediction hash, product and horizon. Realization
before the horizon is refused. Availability and recording clocks both control visibility. Corrections
append a new version; an existing label version cannot change. Equal recording clocks preserve
arrival sequence. A prediction remains PENDING until a visible label exists. Reading a historical
view never writes anything or changes the original prediction.

Signals, risk decisions and proposed positions emitted after the prediction append separate
`DecisionObservation` records. Fills, no-fill observations, expiry and errors append separate
`ExecutionObservation` records. Fill observations preserve fees, adverse slippage, target versus
marked realized exposure, observation delay, source fill identity and method. A final proposal
without a next-bar observation stays PENDING. Execution availability uses the paper engine's
**fill-bar close observation**, rather than pretending its opening was already observed.

Model Lab import verifies the dataset, prepared/completed manifests, model artifacts, prediction
artifact and per-prediction inputs. Original snapshots and prediction fingerprints survive. ZERO
and TRAIN_MEAN predictions are retained; the mean uses only the registered train population.
Shadow observations come from a read-only, hash-verified view of the new worker's private database.
Inference latency and availability remain unknown when Model Lab did not record them.

The authorized real demo consumes only the frozen May–July v2 corpus into a **new** private database,
loads the frozen v2 artifacts without fitting, and requires the same result hashes as the committed
reference. It does not open a live or archived runtime store. Original paper evidence did not record
full feature values or an InformationSnapshot: its adapter retains a clearly named legacy input
certificate and original feature/prediction hashes, with explicit NOT_RECORDED diagnostics. It does
not fabricate a shared snapshot or feature vector. Historical bar-close availability remains an
assumption, execution costs remain synthetic, and the spent OOS window remains exploratory. Its
four-hour tail labels stay pending and its unobserved telemetry stays unknown.

The private SQLite store uses WAL/FULL, append-only evidence triggers, canonical content hashes,
a global hash chain, verified reads, transactional reservations and persistent size/count budgets
(100,000 records, 256 MiB, 2 MiB per record). Read-only consumers use SQLite `mode=ro`. Unknown store
schemas fail closed without migration. The chain detects content/index changes and missing internal
records; it is local integrity evidence, not an externally signed audit journal.

## Monitoring

Every view binds a single model contract, artifact, product and horizon, an explicit `as_of`, a
population fingerprint and a versioned method. The immutable reference records its exact cohort,
selection, cutoff, observations, feature distributions, baseline measurements and method. A future
or differently bound reference is refused. References are created explicitly, never overwritten
or silently fitted to the current sample.

`monitoring-method-v1` reports:

- Decision freshness, observed hourly decision gaps, explicit input gaps and missing/partial quality.
- Observed inference attempts, errors, availability and latency distribution. Without attempts,
  availability stays null. Telemetry binds the model artifact, product and horizon.
- Finite feature/return distributions with sample, mean, population standard deviation and quantiles.
  Invalid raw outputs remain in the ledger and are excluded from numeric metrics with a diagnostic.
- PSI using five reference-quantile bins, deduplicated boundaries and a `1e-6` pseudocount; two-sample
  KS statistic with no p-value. At least ten samples in each population are required. PSI > 0.2 or
  KS > 0.3 is a descriptive drift flag. Constant references remain valid.
- Regimes under `price-regimes-v1`: HIGH_VOL when decision-time `atr_pct_14 >= 0.03`, otherwise UP
  when `return_4 >= 0.01`, DOWN when `return_4 <= -0.01`, otherwise RANGE. Missing inputs give UNKNOWN.
  Future labels never define a regime.
- Forward-return MAE/MSE/RMSE by UTC month, product and regime, using only visible labels.
  Baseline advantage is `1 - model_MSE / baseline_MSE` on exactly the same observed pairs, with sample,
  population hash and method. Zero baseline MSE is undefined. Fewer than ten pairs is insufficient.
  A decline greater than 0.05 versus the reference flags a descriptive performance drop.

Classifications are MISSING_DATA, TECHNICAL_DEGRADATION, DRIFT and PERFORMANCE_DROP; multiple can
coexist. Freshness > 7,200 seconds, attempt error rate > 5% or p95 latency > 1,000 ms are explicitly
versioned operational thresholds. A replay's present-day age is not a live freshness claim. These
thresholds and descriptive baseline differences establish no statistical edge. Serial dependence
and overlapping horizons require the separately frozen comparison method for any confirmatory
statement; this module claims no calibrated probabilities or intervals.

## Reproducible demo and API

With the repository Python dependencies installed:

```bash
python -m scripts.trading_lab.research.demo --root var/trading_lab/research-demo --paper-replay
python -m scripts.trading_lab.app_api.server --port 8790 --research-root var/trading_lab/research-demo/registry
```

A previously created private replay can be reused with `--replay-database` pointing under this
worktree's `var/`; its complete read-only event chain and count must equal the frozen reference.
No archived or live store is accepted.

The demo requires a new root and preserves prior runs. It creates a synthetic dataset in the existing
isolated worker, prepares and registers both hypotheses before their jobs, imports emitted predictions
and labels, imports actual shadow decisions/executions, and creates validation references and test
monitoring. A separate labelled synthetic scenario injects gaps, latency, errors, drift and performance
loss and verifies all four classifications. `--paper-replay` additionally runs the authorized frozen
replay without fitting and creates descriptive May references versus June–July observations,
retaining unknown real feature values and inference telemetry; omit it for a wholly synthetic demo. Only private runtime stores are written.

All endpoints below are GET/HEAD; POST returns 405. Reads never launch a workload or ingest a label.
Unconfigured stores return 503, malformed selection/cursors 400, absent evidence at T 404 and corrupt
identities 409. Errors never expose operator locations. Existing research benchmark/equity routes
and authenticated Model Lab controls keep their behavior.

| Path under `/api/v1` | Result |
|---|---|
| `/research/hypotheses` | Paged immutable plans |
| `/research/hypotheses/{hash}` | Plan, criterion hash and full trial history at T |
| `/research/experiments` | Paged append-only trial observations, including failed/negative/abandoned outcomes |
| `/research/experiments/{hash}` | Prepared or completed Model Lab manifest and bound trial history |
| `/research/proposals` | Deterministic local engine capabilities and preparation instructions |
| `/research/comparison` | Exact protocol V2 plan/hash and actionable WAITING_DATA diagnostic |
| `/observability/predictions` | Paged original predictions and compact causal state; full evidence at the detail path |
| `/observability/predictions/{hash}` | Prediction view at T; pending labels stay pending |
| `/observability/references` and `/observability/references/{hash}` | Versioned reference evidence |
| `/observability/monitoring` | Explicit `as_of`; optional product, model_id, artifact_hash, start/end, split and reference_hash |
| `/observability/replays` | Count/identity evidence from authorized frozen replay imports |
| `/observability/health` | Verified chain, method/regime definitions and read-only capability |

Lists accept `as_of`, `limit` (default 100, maximum 200) and a query-bound `cursor`. Prediction lists
also accept product, model_id and a half-open start/end interval. Continuation requires the original
`as_of`; a cursor for another selection is rejected. Full feature and snapshot evidence lives in the
private store and is not committed. This local read-only surface does not implement multi-project B2B
authorization; that is the separate B2B slice.

Validation: 48 research cases; the combined required source/application gate, shared platform,
Model Lab and research suites finished with **689 passed / 2 BLOCKED** (private FOMC fixtures and
`hyprl_api` absent). Event/protection/protocol, supervised EDGAR and frozen paper/engine regressions:
**496 passed**. The committed evidence artifact contains only counts, identities,
method bindings and explicit limitations. Unavailable legacy gate checks remain BLOCKED.
