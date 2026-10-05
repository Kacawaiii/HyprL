# Model Lab v1

The available path is **InformationSnapshot → versioned dataset → isolated synthetic training →
temporal validation → economic backtest → offline synthetic shadow replay**. It consumes
[CONTRACTS_V1](CONTRACTS_V1.md), the existing crypto indicators/Ridge, scoring, signal, risk,
economic-backtest and PaperEngine implementations. No capture, real training, remote model
inference, broker connection or real order is enabled. Existing paper v1/v2 artifacts and the
May–July OOS v2 replay keep their exact identities and results.

## Datasets and admissibility

`platform.datasets.build_versioned_dataset` joins the existing hourly crypto feature dataset to
a pinned `SnapshotBuilder` for every decision. Callers supply the series and matching OHLCV
availability attestations; price clocks are checked against every dependency, including the
entire causal prefix needed by EMA. Snapshot state, source-specific horizons, inventory/revision
distinctions, coverage and quality are retained. Price evidence must match the selected snapshot.

The manifest records products, a half-open decision period, `forward_return`, a whole-hour horizon,
feature definitions, snapshot identities, counts, exclusions and research-protection identity.
Its fingerprint binds the complete contract; the data digest also binds labels and backtest bars.
Rows carry feature/snapshot hashes, a price-attestation prefix digest and label realization and
availability clocks. Labels stay outside snapshots and inference inputs. Unknown sources remain
unknown. Selecting an unavailable event feature excludes a decision; it never inserts zero.

Protected input is refused **before price access**. Labels reaching a protected boundary, event
windows touching a protected interval, warm-up/gaps, unavailable price dependencies, stale prices
and unresolved selected event features have explicit exclusion reasons. The authoritative
`research_protection` contracts are consumed without edits; an old protection binding is rejected.

V1 dataset construction supports BTC-USD/ETH-USD hourly prices and horizons from 1 to 24 hours.
The current registered models support exactly four hours and the six frozen price feature columns.
The offline Python builder can select snapshot V2 event columns for a future compatible adapter.
The HTTP path creates **only server-generated synthetic** prices (80–600 bars per product), with
synthetic at-close availability. It accepts no uploaded data, corpus locations or source URLs.
There is no real joint price/event training population in this delivery. Equity datasets and other
calendars require another implementation; they are not advertised as supported.

## Models and adapters

| Model | Registration | Supported capabilities | Identity and limits |
|---|---|---|---|
| `paper-ridge-v1` | internal | predict, serialize, infer | exact PAPER_MODEL_SPEC_V1 and original fitted/artifact hashes; frozen, no train method |
| `paper-ridge-v2` | internal | predict, serialize, infer | exact PAPER_MODEL_SPEC_V2 and original fitted/artifact hashes; frozen, no train method |
| `synthetic-ridge-v1` | internal demo | train, predict, serialize, infer | new synthetic-only artifact using the existing Ridge implementation; alpha 1.0, cholesky, train-only standardization |
| `local-momentum-v1` | external local Python entry point | train, predict, serialize, infer | four times the last hourly return; initialization validates synthetic training; no fitted transform |

Every `ModelContract` declares inputs, outputs, horizon, versions, row limits and capabilities.
Only `return` is provided. Target price, class, probabilities, quantiles and scenarios are explicitly
null; no calibrated confidence is claimed. `FrozenPaperAdapter.load` uses the original verified
loader and binds the artifact to its product. Serialization copies the original record exactly.

`ModelRegistry.register_entry_point("package.module:create_adapter")` is an operator/developer
Python interface. A factory returns an adapter with `contract`, `train(rows, synthetic=...)`,
`predict(rows)` and `serialize()` as declared; the demonstration adapters also implement
`restore(artifact)`. Prediction rows expose features and clocks, with no label attribute. The
independent `platform.local_momentum:create_adapter` is registered through this interface.
Duplicate model identities are refused. HTTP never accepts a Python import or executable path.
Workers use the shipped registry; adding another worker adapter requires installing/registering
it in operator-controlled code and obtaining any authorization its workload needs.

## Experiments and reproduction

`prepare_experiment` records the hypothesis/mechanism/falsification, fixed parameters and criterion,
both ZERO and TRAIN_MEAN baselines, split populations, costs, calendar, budgets and model/dataset
contract hashes **before execution**. PREPARED and COMPLETE manifests are separate immutable
records, bound by `prepared_hash`. Failed, interrupted and cancelled jobs retain the prepared
configuration; their results view reports the corresponding manifest status.

There is one chronological train/validation/test split. Training labels must be realized and
available strictly before validation starts; validation labels must be available strictly before
test starts. The label tail is purged in market time. A configurable 0–24 h embargo drops decisions
at the starts of validation and test; the default is one hour. Each split records its exact
population hash, bounds and exclusions. No validation selection or post-validation refit occurs.
Ridge transformations and the TRAIN_MEAN baseline are fitted on train only. Restored serialized
adapters perform validation/test inference. Every prediction is a shared `PredictionRecord`.

Metrics reuse walk-forward MAE, RMSE and rank IC. The predeclared demo criterion compares test MAE
strictly against both baselines. Null IC, negative results and a failed criterion are kept. Economic
backtests reuse the frozen signal/risk/execution contracts (10 bps fees, 5 bps adverse slippage and
the existing terminal liquidation rule). Shadow replay uses PaperEngine's distinct next-bar fill
observation and no-terminal-liquidation policy. The two accounting views are identified separately.
Shadow accounts are independent per product and their session hash binds the new experiment.

Results include model artifacts, parameters, splits, baseline metrics, predictions, backtests,
shadow chain verification and hashes for artifact lookup. Numeric library versions, Python version,
Decimal precision and public implementation digests bind the runtime. The worker pins numeric
thread counts to one. Repeating the same prepared configuration in that runtime yields identical
canonical results and shadow chains; a changed runtime fails closed. Reproduction creates a fresh
private shadow database, preserving every previous run. This proves infrastructure, not an edge.

## Jobs and resource limits

`platform.jobs` persists IDs, state, progress, structured log codes, cancellation, resource budgets,
errors and content-addressed artifacts in SQLite WAL/FULL. States are QUEUED, RUNNING, COMPLETE,
FAILED, CANCELLED and BLOCKED. Only dataset and experiment workloads are dispatched.

A root has one supervisor owner (`flock`), one spawned model worker and at most eight queued/running
jobs. The worker has CPU, address-space and file-size limits, a wall-clock watchdog, no core dumps
and one numeric thread. Defaults: 120 s wall, 60 s CPU, 1024 MiB address space, 32 MiB per output file.
Maximums: 180 s wall, 90 s CPU, 1024 MiB address space, 32 MiB output. The persistent lab budget is
1000 jobs and 128 MiB of serialized artifacts; individual artifacts are bounded at 24 MiB.

Cancellation is persistent and interrupts a running worker. The final transition cannot publish a
successful result after a cancellation request. Supervisor shutdown terminates its own worker;
queued work can resume at restart. Previously RUNNING work is marked FAILED with an interruption
code, rather than silently resumed. A second execution lock and an orphan-parent watchdog prevent
overlapping workloads after a crash. Errors expose stable codes, not exception text or locations.
These are local trusted-code process limits, not a sandbox for untrusted plugins.

## CLI demonstration

From the worktree, with the existing Python dependencies:

```bash
python -m scripts.trading_lab.platform.model_lab_demo --root var/trading_lab/model-lab-ridge --bars 120
python -m scripts.trading_lab.platform.model_lab_demo --root var/trading_lab/model-lab-momentum --model local-momentum-v1 --bars 120
```

Each command creates a synthetic two-product dataset and runs its experiment twice in workers.
It exits unsuccessfully if the two canonical result hashes differ. It prints synthetic labels,
counts, metrics, hashes and limitations. Runtime stores and logs stay in ignored `var/`.
The existing snapshot evidence demo (`platform.demo`) is unchanged.

## Local API

The source API remains read-only by default. Opt in to lab controls by setting the private
`HYPRL_MODEL_LAB_TOKEN` environment value (32–256 ASCII characters) and running:

```bash
python -m scripts.trading_lab.app_api.server --port 8790 --model-lab-root var/trading_lab/model-lab-api
```

The lab requires a loopback listener. Every lab request needs `Authorization: Bearer ...`; tokens
are never accepted in URLs. Browser Origin requests are refused. This is a single local operator
lab, not the separate multi-project B2B authorization surface. It does not change source/store
permissions or the existing GET/HEAD endpoints. Unconfigured controls report unavailable.

| Method | Path | Request/result |
|---|---|---|
| GET/HEAD | `/api/v1/lab/models` | contracts, identities and registration method |
| POST | `/api/v1/lab/datasets` | `{"synthetic":true,"products":["BTC-USD","ETH-USD"],"start":"2026-06-01T00:00:00Z","bars":120,"target":"forward_return","horizon_seconds":14400,"seed":7}` → 202 with job ID |
| GET/HEAD | `/api/v1/lab/datasets/{hash}` | manifest, exclusions and fingerprint; no source data export |
| POST | `/api/v1/lab/experiments` | `{"dataset_hash":"...","model_id":"synthetic-ridge-v1","embargo_seconds":3600}` → 202, prepared manifest and job ID |
| GET/HEAD | `/api/v1/lab/jobs` | bounded recent jobs, states and limits |
| GET/HEAD | `/api/v1/lab/jobs/{id}` | status, progress, structured logs and worker PID |
| POST | `/api/v1/lab/jobs/{id}/cancel` | `{}` → current status and cancellation request |
| GET/HEAD | `/api/v1/lab/jobs/{id}/results` | pending/terminal state or full synthetic result and hash |
| GET/HEAD | `/api/v1/lab/artifacts/{kind}/{hash}` | kind = model, predictions, backtests, shadow or experiment |

JSON control bodies are bounded at 16 KiB. Stored artifact identities are verified on every read;
corruption returns an integrity conflict (409) before any artifact is exposed. Unknown parameters, uploaded datasets, client paths,
import strings and unsupported model/horizon/schema combinations are refused. A semantically
invalid dataset configuration fails in its worker with a durable diagnostic. Source POSTs still
return 405. Frozen real replay evidence remains available at the unchanged read-only
`GET /api/v1/paper/replay` and its existing pages; Model Lab never refits or rewrites that evidence.

Validation: `python -m pytest tests/model_lab -q` — **54 passed** locally. The suites cover hash mutations, causal price
dependencies, holdout-before-read checks, temporal leakage and embargo, train-only transforms,
serialized inference without labels, frozen model identities, separate worker PIDs, resource
failures, cancellation, restart, queue ownership and the authenticated HTTP flow.

The [digest-only demonstration evidence](artifacts/model_lab_v1_evidence.json) records both two-run
reproductions in the recorded runtime. The shared dataset has 182 included decisions, 58 exclusions
and 240 snapshots. Each experiment emits 80 validation/test prediction records and a verified
364-event synthetic shadow chain. Ridge meets the synthetic MAE criterion; momentum fails it for
both products and keeps its result. Reference file digests are recorded without copying artifacts.
These synthetic outcomes establish no real model advantage.

Waiting work: real training and remote-model authorization; a real joint price/event population;
equity/other calendar adapters; a user-facing model registration flow under project permissions;
and the observability/cockpit slices consuming the emitted records.
