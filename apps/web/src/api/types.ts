/**
 * Payload types for the read-only application API.
 *
 * These mirror `scripts/trading_lab/app_api` and are centralised here on
 * purpose: scattering interfaces across components is how a frontend ends up
 * with three slightly different ideas of what a signal is.
 *
 * Decimal-precision values arrive as strings and STAY strings. Parsing a
 * price into a JS number silently rounds it, and this app never needs the
 * arithmetic -- it displays what Python computed.
 */

export type SignalDirection = 'LONG' | 'FLAT' | 'SHORT';
export type PositionSide = 'LONG' | 'FLAT' | 'SHORT';

export interface Health {
  status: string;
  api_version: string;
  core_status: string;
}

export interface Capabilities {
  market_history: boolean;
  signal_engine: boolean;
  position_target: boolean;
  economic_backtest: boolean;
  paper_trading: boolean;
  live_trading: boolean;
  realtime_stream: boolean;
}

export interface SignalEngineInfo {
  protocol: string;
  rule: string;
  spec_hash: string;
  frozen: boolean;
  optimized: boolean;
  long_threshold: string;
  short_threshold: string;
  full_strength_excess: string;
  prediction_horizon: number;
  boundary_semantics: string;
}

export interface RiskEngineInfo {
  protocol: string;
  spec_hash: string;
  frozen: boolean;
  optimized: boolean;
  max_long_exposure: string;
  max_short_exposure: string;
  risk_scale: string;
  volatility_scaling_enabled: boolean;
  strength_mapping_version: string;
}

export interface ConfirmatoryHoldout {
  holdout_id: string;
  range_start: string;
  range_end: string;
  products: string[];
  timeframe: string;
  captured: boolean;
  single_use: boolean;
}

export interface SystemInfo {
  api_version: string;
  signal_engine: SignalEngineInfo;
  risk_engine: RiskEngineInfo;
  market_data: {
    available: boolean;
    corpus_id: string | null;
    corpus_spec_hash: string | null;
    corpus_content_hash: string | null;
    products: string[];
    timeframe: string;
    point_in_time_revision_history: boolean;
  };
  benchmarks: {
    v1_available: boolean;
    v2_exploratory_available: boolean;
    v2_confirmatory_observed: boolean;
    confirmatory_holdout: ConfirmatoryHoldout | null;
  };
  capabilities: Capabilities;
}

export interface ProductSummary {
  product: string;
  timeframe: string;
  rows: number;
  first_open: string;
  last_open: string;
  missing_openings: number;
  /** No live feed exists; this is null rather than a fabricated number. */
  latest_price: string | null;
  latest_price_available: boolean;
}

export interface BenchmarkProductEntry {
  product: string;
  rank_ic: string;
  mae: string;
  rmse: string;
  observations: number;
  folds: number;
  benchmark_spec_hash: string;
  dataset_hash: string;
  benchmark_results_hash: string;
}

export interface BenchmarkSummary {
  version: string;
  protocol_version: string;
  experiment_type: string;
  confirmatory_result: boolean;
  corpus_content_hash: string;
  products: BenchmarkProductEntry[];
}

export interface Overview {
  api_version: string;
  system_status: string;
  signal_spec_hash: string;
  risk_spec_hash: string;
  products: ProductSummary[];
  benchmarks: BenchmarkSummary[];
  capabilities: Capabilities;
}

export interface MarketsIndex {
  timeframe: string;
  corpus_id?: string;
  corpus_content_hash?: string;
  products: Array<{
    product: string;
    rows: number;
    first_open: string;
    last_open: string;
    missing_openings: number;
  }>;
}

export interface Candle {
  bar_open_at: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

export interface Page {
  returned: number;
  limit?: number;
  has_more: boolean;
  next_cursor: string | null;
}

export interface CandlePage {
  product: string;
  timeframe: string;
  candles: Candle[];
  page: Page;
}

export interface ChartMetadata {
  source_timeframe: string;
  aggregation: 'none' | 'ohlc-bucket';
  bucket_size: number;
  source_count: number;
  returned_count: number;
  max_points: number;
  aggregated: boolean;
}

export interface ChartSeries {
  product: string;
  series: Candle[];
  metadata: ChartMetadata;
}

export interface SignalsView {
  available: boolean;
  reason?: string;
  signal_spec: SignalEngineInfo & { optimized: boolean };
  decisions: Array<{
    timestamp: string;
    prediction: string;
    direction: SignalDirection;
    strength: string;
    signal_spec_hash: string;
    decision_hash: string;
  }>;
  page: Page;
}

export interface RiskView {
  available: boolean;
  reason?: string;
  risk_spec: RiskEngineInfo & { risk_scale_rule_version: string };
  targets: Array<{
    timestamp: string;
    side: PositionSide;
    target_exposure: string;
    signal_strength: string;
    position_target_hash: string;
  }>;
  page: Page;
}

export interface BenchmarkPeriod {
  start: string;
  end: string;
  rank_ic: string | null;
  mae: string | null;
  rmse: string | null;
  observations: number;
}

export interface BenchmarkDetail {
  version: string;
  product: string;
  protocol_version: string;
  experiment_type: string;
  confirmatory_result: boolean;
  benchmark_spec_hash: string;
  dataset_hash: string;
  benchmark_results_hash: string;
  corpus_spec_hash: string;
  corpus_content_hash: string;
  geometry: Record<string, number>;
  global_test_metrics: {
    rank_ic: string | null;
    mae: string | null;
    rmse: string | null;
    observations: number;
  };
  selection_counts: Record<string, number>;
  selection_reasons: Record<string, number>;
  periods: BenchmarkPeriod[];
  scenarios: Array<{
    scenario_id: string;
    rank_ic: string | null;
    mae: string | null;
    rmse: string | null;
  }>;
}

export interface ApiErrorPayload {
  error: string;
  api_version?: string;
}
