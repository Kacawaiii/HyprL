# Shared platform contracts v1

The implementation is `scripts/trading_lab/platform/contracts.py`. Every record is a frozen,
keyword-only dataclass. Nested JSON objects and arrays are copied and frozen too. `to_dict()` returns
an independent JSON value; `from_dict()` requires the exact schema. `canonical_json()` uses the source
canonical serializer (sorted keys, compact separators, ASCII escapes, finite numbers only).
`identity` / `fingerprint` is SHA-256 of that complete JSON, including `schema`. Hashes are lowercase
64-digit hex. Instants require an explicit offset and top-level clocks normalize to UTC. No record
contains private filesystem locations or raw captured bodies. Semantic changes require a new schema.

| Record / schema | Required content |
|---|---|
| ProviderContract / `provider-contract-v1` | provider_id, version, capabilities, identities, formats, clocks, limits, corrections, historical_availability, health, evidence, activation, shape_verification |
| InformationSnapshot / `information-snapshot-v1` | as_of, products, companies, sources, prices, events, features, policies, coverage, quality; synthetic defaults false |
| ModelContract / `model-contract-v1` | model_id, version, inputs, outputs, horizons_seconds, capabilities, limits, implementation_version; synthetic defaults false |
| DatasetManifest / `dataset-manifest-v1` | dataset_id, version, products, decision_start/end, target, horizon_seconds, snapshot_hashes, features_hash, exclusions, splits, policies, counts, synthetic |
| ExperimentManifest / `experiment-manifest-v1` | experiment_id, version, dataset_hash, model_contract_hash, hypothesis, parameters, splits, baselines, decision_criteria, costs, budgets, status, artifacts, synthetic |
| PredictionRecord / `prediction-record-v1` | prediction_id, model_id, model_contract_hash, artifact_hash, product, decision_at, horizon_seconds, snapshot_hash, features_hash, event_ids, outputs; nullable signal, risk, proposed_position, execution, uncertainty, costs; errors; synthetic |
| LabelRecord / `label-record-v1` | label_id, prediction_id, prediction_hash, product, horizon_seconds, realized_at, available_at, recorded_at, target, value, provenance, version |

Model capabilities are an explicit subset of `train`, `predict`, `serialize`, `infer`: registration
never implies a capability. Model output declarations and prediction outputs contain all six keys:
`return`, `target_price`, `class`, `probabilities`, `quantiles`, `scenarios`. Unsupported outputs are
JSON null. Declarations describe their type/method; predictions carry the value. A confidence or
interval claim needs its named method and limitations in `uncertainty`; this contract does not confer
calibration. Horizons are positive seconds. Adapters declare their own model and implementation versions.

Dataset intervals are half-open. Exclusions retain reason, product and decision time; splits should
record temporal ranges, purge and embargo; policies bind transformations and research protection.
Experiment criteria and budgets are recorded before execution. Status is PREPARED, RUNNING, COMPLETE,
FAILED, CANCELLED or BLOCKED. Artifacts and source evidence use digests, public URLs and counts.

Predictions are immutable evidence. A future label is a separate append-only record bound to the exact
prediction hash, product and horizon. `enrich_prediction(prediction, labels, as_of=...)` returns a view,
PENDING until both label availability and recording are at or before T. It refuses mismatched labels
or realization before the horizon. Label corrections append another version; they never replace the
original prediction or earlier labels. Execution updates likewise belong to later observation records,
not mutations of an issued prediction.

Snapshot source sections each contain their own H, P when resolved, source snapshot identity, spec and
provider contract bindings, read state and coverage. There is no global H. UNRESOLVED, NOT_OBSERVED,
NOT_CONFIGURED, INTEGRITY_ERROR, NOT_APPLICABLE, UNKNOWN_MAPPING and PROTECTED survive composition.
A RESOLVED read proves a causal prefix, not complete historical coverage. Event clocks separately retain
declared publication, observed_at, ingested_at, attested available_at and revision identities. A backfill
retains the acquisition's attested availability. Price providers must attest availability explicitly;
bar close time alone does not establish historical availability. Snapshot features bind V1/V2 policies
and all historical dependencies, with partial coverage exposed. No snapshot grants research permission.

Validation: `python -m pytest tests/platform/test_contracts.py -q` covers every record, canonical round
trips, deep immutability, explicit absent outputs, invalid declarations and late-label causality.
