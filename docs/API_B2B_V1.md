# B2B API v1

The dedicated local listener exposes authenticated project resources under
`/api/b2b/v1/projects/{project_id}`. It reuses the platform's immutable v1 contracts,
causal source readers, Model Lab adapters and isolated workers, and research monitoring.
It supports synthetic experiments and reads operator-configured archives without changing them.
No endpoint captures data, trains on real data, imports client code, calls a remote model,
starts a broker, or issues an order.

The [generated OpenAPI 3.1 document](artifacts/b2b_openapi_v1.json) describes requests,
responses, permissions and export rights. It is generated from the route registry and shared
dataclass contracts, validated with `openapi-spec-validator`, and checked against actual HTTP
responses with JSON Schema in `tests/b2b`. Every route, including the live OpenAPI document,
requires authentication. The existing cockpit/application listener remains a separate local
operator surface; it is not a tenant endpoint. B2B clients use this dedicated listener.

## Reproducible local client

From the checkout, with the existing Python runtime dependencies installed:

```bash
python -m examples.b2b_client --demo
```

This starts an ephemeral loopback listener, generates a random key in memory, registers
`demo-momentum` using the installed `local-momentum-v1` adapter, builds a labelled synthetic
120-hour BTC dataset, prepares an experiment with pre-fixed criteria, waits for its isolated
worker, and reads results and predictions over HTTP. Only the key digest reaches the server
configuration. Runtime artifacts and audit entries stay in ignored `var/trading_lab/b2b-demo/`.
Use `--demo-root` to choose a new private runtime for an independent demonstration.

The example prints only synthetic labels, counts, digests, the retained positive/negative
criterion result and limitations. It does not print credentials or private roots. On the
initial verified run it returned 40 predictions and retained the momentum adapter's negative
result against its baselines. This demonstrates infrastructure and establishes no market edge.

## Private configuration and startup

For a persistent local server, create a private configuration and keep its client key in
the environment. Never put credentials, private configuration, stores or logs in Git.
`PRIVATE_CONFIG` denotes an operator-chosen file in a private directory; `PRIVATE_RUNTIME`
denotes a new directory separate from every archive. The configure command creates a 0600
file exclusively and never overwrites one or writes a plaintext key.

```bash
export HYPRL_B2B_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m scripts.trading_lab.b2b.configure --output "$PRIVATE_CONFIG" --project demo
python -m scripts.trading_lab.b2b.server --config "$PRIVATE_CONFIG" --root "$PRIVATE_RUNTIME" --port 8791
```

In another local shell with the same private client key:

```bash
python -m examples.b2b_client --base-url http://127.0.0.1:8791 --project demo
```

The server accepts only literal IPv4/IPv6 loopback addresses. The example refuses remote
hosts, redirects and environment proxies. TLS termination and remote hosting are outside
this slice; no service or proxy configuration is changed.

The configuration has schema `b2b-config-v1` and exactly `schema`, `projects`, `keys`.
Project and key identifiers are public slugs, starting with a lowercase letter, followed
by up to 47 lowercase letters, digits, underscores or hyphens.

| Project field | Meaning |
|---|---|
| `request_budget`, `job_budget` | Required integers, 0–1,000,000; lifetime spend is durable across restart and key rotation |
| `products` | Explicit product grants from the existing registry; synthetic workloads support BTC-USD/ETH-USD |
| `sources` | Explicit `fomc`/`edgar` archive grants; ungranted sources remain NOT_CONFIGURED |
| `exports` | Explicit subset of `dataset_manifest`, `dataset_rows`, `model_artifact`; defaults empty |
| `fomc_store`, `edgar_store`, `price_root`, `research_root` | Optional absolute private roots, set only by the operator; no request accepts a root |
| `synthetic_sources` | Optional boolean, defaults false; explicitly labels synthetic source archives in snapshots |

Keys contain exactly `key_id`, `project_id`, `key_sha256`, `permissions`, `expires_at`,
`enabled`. A key belongs to one configured project and must have an offset-qualified expiry
and an explicit boolean enabled state. The hash is SHA-256 over
`b"hyprl-b2b-api-key-v1\0" + key.encode("ascii")`; keys must be 32–256 ASCII characters.
Use random 256-bit keys, not passwords. Comparison is constant-time. The API never stores
a plaintext credential and accepts keys only as `Authorization: Bearer <key>`, never in URLs.
Config files must be regular, private files with no group/other permissions. Configuration
is loaded at startup; restart with the edited private config to rotate, disable or revoke keys.
Spend remains attached to the project. Changing a budget limit does not erase previous spend.

| Key permission | Operations |
|---|---|
| `read` | Project, contracts, snapshots/events/data, models, job state, experiment results/predictions, monitoring |
| `models:write` | Register a project alias for an installed synthetic adapter |
| `datasets:write` | Submit an explicitly synthetic dataset job |
| `experiments:write` | Prepare and enqueue an experiment using a project-owned dataset and registered alias |
| `jobs:cancel` | Request cancellation of a project-owned job |
| `export` | Export only kinds also granted in the project's `exports` |
| `audit:read` | Read a verified, project-filtered audit page |

## Wire contracts and endpoints

Success responses contain `schema="b2b-response-v1"`, `api_version="1.0.0"`, `project_id`,
`request_id`, and `data`. Errors contain `schema="b2b-error-v1"`, the same version and request
ID, and a fixed `error` code. HEAD performs the same auth, quota and audit checks as GET and
returns no body. Responses have `Cache-Control: no-store` and `X-Request-ID`.

All paths below are relative to `/api/b2b/v1/projects/{project_id}`. Unknown and repeated
query parameters, duplicate JSON members, nonfinite JSON, client paths/imports/URLs and
unsupported fields are refused. JSON bodies must be 1–16,384 bytes with one Content-Length
and application/json; chunked requests and browser Origin requests are refused.

| Method | Path | Contract / behavior |
|---|---|---|
| GET/HEAD | project root | Grants, lifetime budgets and worker limit |
| GET/HEAD | `/contracts` | Shared provider, snapshot, model, dataset, experiment, prediction, label schemas and installed adapter capabilities |
| GET/HEAD | `/snapshots` | InformationSnapshot v1, its canonical fingerprint and native source identities |
| GET/HEAD | `/events` | Current causally selected events, source states and snapshot hash |
| GET/HEAD | `/events/{event_id}/revisions` | Selected revision plus historical attested observations within T and the source's own horizon |
| GET/HEAD | `/normalized-data` | Native FOMC normalized fields / EDGAR filing fields, zero-padded CIK, price states and snapshot hash; no raw bodies |
| GET/HEAD | `/models`, `/models/{model_id}` | Project registration and the adapter's unchanged ModelContract and digest |
| POST | `/models` | `{"model_id":"demo-momentum","adapter_id":"local-momentum-v1"}` → 201; identical registrations are idempotent, conflicting aliases return 409 |
| POST | `/datasets` | `{"synthetic":true,"products":["BTC-USD"],"bars":120,"seed":7}` → 202 and a persistent job ID |
| GET/HEAD | `/datasets/{dataset_hash}/manifest` | DatasetManifest and fingerprint, requires export permission + `dataset_manifest` |
| GET/HEAD | `/datasets/{dataset_hash}/export` | Full verified synthetic dataset, requires export permission + `dataset_rows` |
| POST | `/experiments` | `{"dataset_hash":"...","model_id":"demo-momentum","embargo_seconds":3600}` → 202 with the PREPARED manifest and fingerprint |
| GET/HEAD | `/jobs`, `/jobs/{job_id}` | Owned jobs, persistent state, progress, structured log codes, resource limits |
| POST | `/jobs/{job_id}/cancel` | `{}`; retains terminal results and requests cancellation of pending/running jobs |
| GET/HEAD | `/experiments/{job_id}` | PREPARED/RUNNING/terminal manifest view for an experiment |
| GET/HEAD | `/experiments/{job_id}/results` | Pending/terminal state or verified result summary; no model artifact, training rows or shadow dump |
| GET/HEAD | `/experiments/{job_id}/predictions` | Paged immutable PredictionRecord v1 values emitted by that experiment |
| GET/HEAD | `/experiments/{job_id}/models/{model_id}/export?product=BTC-USD` | Verified model artifact; requires export permission + `model_artifact` |
| GET/HEAD | `/observability/predictions` | Project archive predictions with causal late-label/execution enrichment; `as_of` and granted `product` required |
| GET/HEAD | `/observability/predictions/{prediction_hash}` | Immutable prediction, exact input evidence, append-only late labels and executions; `as_of` and granted `product` required |
| GET/HEAD | `/observability/monitoring` | Existing versioned monitoring methods, distributions, regimes, drift and paired performance; `as_of` and granted `product` required |
| GET/HEAD | `/audit` | Append-only audit entries from this project with chain verification |

The live generated document is at authenticated GET/HEAD `/api/b2b/v1/openapi.json`, within
the standard success envelope's `data`. Regenerate/check the public artifact with:

```bash
python -m scripts.trading_lab.b2b.openapi
python -m scripts.trading_lab.b2b.openapi --check
```

Snapshot/event/data reads require `as_of`, comma-separated granted `products` and explicit
`visibility_mode=DURABLE_OBSERVED`. Optional `fomc_horizon` and `edgar_horizon` name independent
commit-sequence horizons. There is no invented common horizon. Publication text, observation,
ingestion and attested availability remain separate; acquisition today cannot make a historical
publication available yesterday. UNRESOLVED, NOT_OBSERVED, absent sources, integrity errors,
protected prices and partial coverage remain visible. Source provenance includes archive-wide
digests and coverage; event records and price selections are restricted to the granted products.

Job/prediction pages use `after` offsets and `limit` (1–200), returning nullable `next_after`.
Job pages are ordered by creation and ID; prediction artifacts are immutable. Audit pages use
`after` sequence numbers and `limit`. Observability pages preserve the existing cursor bound to
the original explicit T and selection. They never truncate a filtered search silently.

The model alias names the registration; the native contract's `model_id` remains the installed
adapter identity. Only `synthetic-ridge-v1` and `local-momentum-v1` can be registered for new
experiments. Installed frozen paper models are capability descriptors only; real refitting
and arbitrary adapter installation are unavailable through HTTP. Model outputs retain all six
kinds with unsupported values explicitly null. No calibrated uncertainty is implied.

Read a dataset job's results before preparing an experiment: a verified successful job binds its
dataset hash to its project. A digest alone never grants access. Dataset summaries omit the manifest
unless both manifest export rights are present; internal experiment preparation still works.
The `result_hash` binds the complete worker artifact; `result` is a permission-filtered summary
and is not claimed to hash to that full artifact's identity. Predictions and model exports have
their own complete immutable record/artifact identities.

## Isolation, budgets and audit

Each key is bound to exactly one project. Cross-project paths return 403; an unknown or
foreign object under the caller's own project returns 404. Job/model/dataset lookups require
ownership before reading content. A successful owned dataset job grants only its verified
dataset digest. Data grants are rechecked when results or datasets are read, including after
configuration changes. Project research archives must be distinct, with no overlapping roots;
the operator curates each archive and the API also checks selected products and reference bindings.

One existing JobRunner owns the B2B root and schedules one isolated process across all projects.
Admission charges the job quota and records ownership in the same SQLite transaction that
enqueues the job. Queue failures roll back those charges. Cancellation and failed jobs retain
their original charge and evidence. CPU, memory, output and wall limits remain enforced by the
existing worker. The API does not execute training inside a request.

Authenticated, authorized requests consume a request unit before query/body validation, including
HEAD and invalid selections. Authentication/project/permission failures are audited without
consuming a project's request budget. Request and job budgets are lifetime ceilings; no automatic
reset or billing is implemented. Exhaustion returns 429 and is audited. Exhausted clients cannot
poll/cancel through the API until the operator changes the private request ceiling; the bounded
worker still stops at its own resource limits. Rejected authentication remains audited even
without a known project. This local v1 has no public-network anti-abuse gateway.

Audit entries contain only server-generated request IDs, UTC time, configured public project/key
IDs, fixed operation names, status and outcome codes. They exclude credentials, bodies, queries,
private paths and exception text. SQLite triggers forbid update/delete and a SHA-256 chain is
verified before project-filtered reads. It detects local corruption; there is no external anchor
against a database owner rewriting the entire chain. If durable audit recording fails, the API
returns 503, including when a mutation already committed. Do not blindly retry mutations after
a transport/503 failure: inspect the project job list/audit first. Workload submission is not an
idempotent retry protocol. Native structured worker logs remain private runtime evidence.

Archives are opened through existing `read_only=True` source/research readers. Both use the
shared stable private DB/WAL reader so SQLite cannot create WAL indexes in the original archives.
The B2B state root must be a new/marked private directory, separate from every read-only input.
Exports never expose source bodies or repair/copy data into the public repository.

| HTTP | Typical fixed code |
|---|---|
| 400 | `INVALID_REQUEST`, `INVALID_QUERY`, `INVALID_BODY`, `INVALID_SOURCE_HORIZON` |
| 401 | `AUTH_REQUIRED` (absent, malformed, wrong, expired or disabled credential) |
| 403 | `PROJECT_DENIED`, `PERMISSION_DENIED`, `DATA_PERMISSION_DENIED`, `EXPORT_DENIED`, `ORIGIN_DENIED` |
| 404 | `RESOURCE_NOT_FOUND` |
| 405 | `METHOD_NOT_ALLOWED` |
| 409 | `ARTIFACT_INTEGRITY_ERROR`, `AUDIT_INTEGRITY_ERROR`, `MODEL_ID_CONFLICT` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 429 | `REQUEST_BUDGET_EXHAUSTED`, `JOB_BUDGET_EXHAUSTED`, `MODEL_BUDGET_EXHAUSTED`, `WORKER_QUEUE_EXHAUSTED`, `PERSISTENT_JOB_BUDGET_EXHAUSTED` |
| 503 | `EVIDENCE_UNAVAILABLE` (unconfigured research archive), `AUDIT_UNAVAILABLE` |

## Verification and limits

```bash
python -m pip install jsonschema==4.26.0 openapi-spec-validator==0.9.0
python -m pytest tests/b2b tests/research/test_read_only.py -q
```

Tests exercise failed/expired/disabled auth, permission checks, cross-project IDs, revoked data
rights, concurrent quota races, restart persistence, transactional queue admission, structured
audit denials/successes and tamper detection, export rights, native synthetic source shapes,
causal/pinned revision selection, integrity failures and unchanged archive bytes, the complete
HTTP client flow and worker isolation. Generated OpenAPI is standard-validated; HTTP responses
are checked against its schemas. `sources-ci` adds this suite without weakening any existing step.

Monitoring reads a configured private project ResearchStore and returns 503 when absent. Worker
experiments produce prediction records immediately; publishing them to an observability archive
uses the already-delivered offline `research.importers.import_model_lab` workflow. HTTP reads
do not publish records, replay archives, refit or fabricate inference telemetry. Future labels
remain PENDING until realization, availability and recording permit their read-time enrichment.
Real training, remote adapters, new captures, billing, automatic budget periods and externally
anchored auditing remain outside this delivery. Data/auth gaps stay WAITING_DATA or
WAITING_AUTHORIZATION; skipped legacy checks stay BLOCKED and are never counted as passes.
