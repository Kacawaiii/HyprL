/**
 * Wire shapes of the Model Lab (`/api/v1/lab/*`), research registry
 * (`/api/v1/research/*`) and observability (`/api/v1/observability/*`) endpoints.
 *
 * Decimal values travel as strings and are never parsed for a decision. Monitoring
 * statistics are JSON numbers, as served. An optional output the model does not
 * provide is `null` -- it is shown as "not provided", never as zero.
 */

export interface ModelContract {
  schema: string;
  model_id: string;
  version: string;
  inputs: { features: string[]; products: string[]; target: string };
  outputs: {
    return: string | null; target_price: string | null; class: string | null;
    probabilities: string | null; quantiles: string | null; scenarios: string | null;
  };
  horizons_seconds: number[];
  capabilities: string[];
  limits: Record<string, string | number | boolean | null>;
  implementation_version: string;
  synthetic: boolean;
}

export interface ModelDescriptor {
  contract: ModelContract;
  fingerprint: string;
  registration: string;
}

export interface LabModels {
  models: ModelDescriptor[];
  external_registration: string;
}

export type JobState = 'QUEUED' | 'RUNNING' | 'COMPLETE' | 'FAILED' | 'CANCELLED' | 'CANCELLING' | string;

export interface JobLog { sequence: number; code: string; progress: number; at: number }

export interface JobStatus {
  id: string;
  kind: 'dataset' | 'experiment' | string;
  state: JobState;
  progress: number;
  cancel_requested: boolean;
  worker_pid: number | null;
  created_at: number;
  updated_at: number;
  result_hash: string | null;
  error_code: string | null;
  limits: { wall_seconds: number; cpu_seconds: number; memory_mb: number; output_mb: number };
  logs: JobLog[];
}

export interface LabJobs { jobs: JobStatus[]; worker_limit: number; synthetic_only: boolean }

export interface Exclusion { decision_at: string; product: string; reason: string; snapshot_hash?: string }

export interface DatasetManifest {
  schema: string;
  dataset_id: string;
  version: string;
  products: string[];
  decision_start: string;
  decision_end: string;
  target: string;
  horizon_seconds: number;
  counts: { by_product: Record<string, number>; excluded: number; included: number };
  exclusions: Exclusion[];
  features_hash: string;
  snapshot_hashes: string[];
  policies: { columns: string[]; event_columns: string[]; availability: string; calendar: string } & Record<string, unknown>;
  splits: { method: string; state: string };
  synthetic: boolean;
}

export interface DatasetResult {
  state: JobState;
  result_hash: string;
  result: { dataset_hash: string; manifest: DatasetManifest; synthetic: boolean };
}

export interface SplitRole { first: string; last: string; population_hash: string; rows: number }

export interface MetricBlock { count: number; mae: string | null; rmse: string | null; rank_ic: string | null }

export interface ExperimentManifest {
  schema: string;
  experiment_id: string;
  dataset_hash: string;
  model_contract_hash: string;
  baselines: string[];
  budgets: { cpu_seconds: number; memory_mb: number; output_mb: number; wall_seconds: number };
  decision_criteria: { metric: string; rule: string; commercial_claim: boolean };
  hypothesis: { statement: string; mechanism: string; falsification: string; scientific_claim: boolean };
  parameters: {
    model_id: string; alpha: number | null; transformations: string;
    runtime: Record<string, unknown> & { source_hashes?: Record<string, string> };
  };
  splits: {
    method: string; embargo_seconds: number; purge_seconds: number; refit_after_validation: boolean;
    roles: Record<'train' | 'validation' | 'test', SplitRole>;
    exclusions: Exclusion[];
  };
  artifacts: Record<string, unknown>;
  status: string;
  synthetic: boolean;
}

export interface BacktestMetrics {
  net_return: string; gross_return: string; max_drawdown: string; fill_count: number;
  total_execution_cost: string; total_fees: string; total_slippage_cost: string;
  final_equity: string; initial_equity: string; annualized_sharpe: string | null;
  turnover_ratio: string; average_abs_exposure: string;
}

export interface ExperimentResult {
  state: JobState;
  result_hash: string;
  result: {
    schema: string;
    fingerprint: string;
    manifest: ExperimentManifest;
    metrics: Record<string, Record<'validation' | 'test', Record<string, MetricBlock>>>;
    criteria_met: Record<string, boolean>;
    limitations: string[];
    backtests: Record<string, {
      metrics: BacktestMetrics; cost_model: string; confirmatory: boolean; live_execution: boolean;
      window: Record<string, string>; signal_series_hash: string; position_target_series_hash: string;
    }>;
    shadow: { mode: string; broker_connected: boolean; events: number;
      chain: { events: number; head_hash: string; verified: boolean; session_id: string } };
    models: Record<string, Record<string, unknown>>;
    predictions: { fingerprint: string; split: string; record: PredictionRecord }[];
  };
}

/** One prediction as the ledger stores it. Optional outputs are `null` when not provided. */
export interface PredictionRecord {
  schema: string;
  prediction_id: string;
  model_id: string;
  model_contract_hash: string;
  artifact_hash: string;
  product: string;
  decision_at: string;
  horizon_seconds: number;
  snapshot_hash: string;
  features_hash: string;
  event_ids: string[];
  outputs: {
    return: string | null; target_price: string | null; class: string | null;
    probabilities: unknown; quantiles: unknown; scenarios: unknown;
  };
  signal: unknown;
  risk: unknown;
  proposed_position: unknown;
  execution: unknown;
  uncertainty: unknown;
  errors: unknown[];
  costs: unknown;
  synthetic: boolean;
}

export interface LabelRecord {
  label_id: string;
  realized_at: string;
  available_at: string;
  recorded_at: string;
  target: string;
  value: string;
  version: string;
  provenance: { dataset_hash?: string; method: string; synthetic?: boolean };
}

export type LabelState = 'AVAILABLE' | 'PENDING' | string;

export interface InputQuality {
  gaps: number; scope: string; state: string; source_states: Record<string, string>;
}

export interface PredictionSummary {
  prediction_hash: string;
  label_state: LabelState;
  latest_label: LabelRecord | null;
  label_versions: number;
  execution_state: string;
  latest_execution: unknown;
  latest_decision: unknown;
  input_quality: InputQuality | null;
  detail_path: string;
}

export interface LedgerRow {
  sequence: number;
  identity: string;
  recorded_at: string;
  chain_hash: string;
  payload: PredictionRecord;
  view: PredictionSummary;
}

export interface PageInfo { returned: number; has_more: boolean; next_cursor: string | null }

export interface ResearchPage<T> {
  schema: string;
  as_of: string;
  records: T[];
  page: PageInfo;
}

export interface PredictionView {
  prediction: PredictionRecord;
  prediction_hash: string;
  label_state: LabelState;
  labels: LabelRecord[];
  decisions: unknown[];
  executions: unknown[];
  execution_state: string;
  recorded_at: string;
  inputs: {
    baselines: Record<string, string>;
    features: [string, string][];
    input_quality: InputQuality;
    provenance: Record<string, unknown>;
    snapshot: {
      as_of: string;
      events: unknown[];
      coverage: { state: string; complete_history: boolean; limits: string[]; source_states: Record<string, string> };
      quality: { state: string; price_states: Record<string, string>; source_states: Record<string, string> };
    } & Record<string, unknown>;
  } | null;
}

export interface Distribution {
  count: number; min: number | null; max: number | null; mean: number | null;
  std: number | null; p50: number | null; p95: number | null;
}

export interface DriftEntry {
  state: string; reference_count: number; current_count: number;
  psi: number | null; ks_statistic: number | null; method: string; minimum_sample?: number;
}

export interface ErrorSummary { sample: number; mae: number; mse: number; rmse: number }

export interface EdgeEntry {
  sample: number; mse_reduction: number | null; model_mse_on_same_pairs: number;
  population_hash: string; method: string; scope: string; state: string; uncertainty: unknown;
}

export interface PerformanceBlock {
  sample: number; pending: number; method: string;
  model: ErrorSummary | null;
  baselines: Record<string, ErrorSummary>;
  edge: Record<string, EdgeEntry>;
  uncertainty: unknown;
  label_policy?: string;
}

export interface Classification {
  category: 'MISSING_DATA' | 'TECHNICAL_DEGRADATION' | 'DRIFT' | 'PERFORMANCE_DROP' | string;
  method: string;
  [key: string]: unknown;
}

export interface MonitoringView {
  schema: string;
  as_of: string;
  selection: { product?: string; model_id?: string };
  sample: number;
  population_hash: string;
  method: Record<string, unknown> & { version: string; limitations: string[] };
  method_hash: string;
  regime_definition: { version: string; definition: string[]; inputs: string[]; clock: string };
  regime_hash: string;
  reference_hash: string | null;
  freshness: {
    seconds_since_last_decision: number | null; threshold_seconds: number;
    decision_gaps: number | null; cadence_seconds: number | null;
    input_gaps: number; known_input_gaps: number; input_gap_unknown_rows: number;
  };
  output_quality: { invalid_returns: number; prediction_error_rows: number };
  input_quality: { sample: number; missing_features: number; not_valid: number };
  inference: {
    attempts: number; errors: number; availability: number | null; state: string;
    latency_ms: Distribution;
  };
  distributions: { prediction: Distribution; features: Record<string, Distribution> };
  drift: Record<string, DriftEntry>;
  performance: PerformanceBlock;
  performance_by_product: Record<string, PerformanceBlock>;
  performance_by_period: Record<string, PerformanceBlock>;
  performance_by_regime: Record<string, PerformanceBlock>;
  classification: Classification[];
  scope: string;
  uncertainty: unknown;
  limitations: string[];
}

export interface MonitoringReference {
  reference_id: string;
  population_hash: string;
  selection: { model_id: string; product: string; split: string };
  binding: { artifact_hash: string; model_id: string; product: string; horizon_seconds: number };
  synthetic: boolean;
}

export interface ReferenceRow { sequence: number; identity: string; recorded_at: string; payload: MonitoringReference }

export interface ObservabilityHealth {
  records: number; head_hash: string; verified: boolean; read_only: boolean;
  network_requests: number; workload_execution: boolean;
  method: { version: string; limitations: string[] };
  regime_definition: { version: string; definition: string[] };
}

export interface HypothesisRecord {
  hypothesis_id: string;
  statement: string;
  mechanism: string;
  falsification: string;
  scope: string;
  target: string;
  horizon_seconds: number;
  sources: string[];
  baselines: string[];
  synthetic: boolean;
  version: string;
  decision_criteria: Record<string, unknown>;
  budgets: Record<string, unknown>;
}

export interface HypothesisRow { sequence: number; identity: string; recorded_at: string; payload: HypothesisRecord }

export interface TrialRecord {
  trial_id: string;
  experiment_hash: string;
  hypothesis_hash: string;
  criteria_hash: string;
  state: string;
  outcome: string;
  synthetic?: boolean;
}

export interface TrialRow { sequence: number; identity: string; recorded_at: string; payload: TrialRecord }

export interface HypothesisDetail extends HypothesisRow {
  criteria_hash: string;
  trial_history: TrialRow[];
}

export interface ComparisonReadiness {
  schema: string;
  protocol_hash: string;
  state: string;
  execution_enabled: boolean;
  paired_decisions: number;
  reasons: string[];
  actions: string[];
  studied_windows: string;
  crypto_role: string;
  equity_role: string;
  limitations: string[];
  hypothesis_hash: string;
  criteria_hash: string;
}

export interface ProposalCatalogue {
  engine: string;
  external_model_calls: number;
  execution_enabled: boolean;
  max_proposals: number;
  max_trials_per_hypothesis: number;
  models: string[];
  synthetic_only: boolean;
  preparation: string;
}
