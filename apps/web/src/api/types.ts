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

export interface ExecutionContract {
  protocol: string;
  execution_spec_hash: string;
  fee_rate: string;
  slippage_rate: string;
  initial_equity: string;
  currency: string;
  fill_policy: string;
  mark_policy: string;
  instrument_model: string;
  cost_model: string;
  optimized: boolean;
  exchange_account_specific: boolean;
}

export interface BacktestMetrics {
  initial_equity: string;
  final_equity: string;
  net_return: string;
  gross_return: string;
  net_pnl: string;
  gross_pnl: string;
  total_fees: string;
  total_slippage_cost: string;
  total_execution_cost: string;
  turnover_ratio: string;
  max_drawdown: string;
  annualized_sharpe: string | null;
  periods_per_year: number;
  fill_count: number;
  rebalance_count: number;
  expired_target_count: number;
  average_abs_exposure: string;
  exposure_time_fraction: string;
}

export interface BacktestSummary {
  version: string;
  product: string;
  experiment_type: string;
  confirmatory: boolean;
  live_execution: boolean;
  cost_model: string;
  source_benchmark_protocol: string;
  economic_backtest_spec_hash: string;
  economic_results_hash: string;
  window: Record<string, string>;
  metrics: BacktestMetrics;
}

export interface BacktestsIndex {
  available: boolean;
  reason: string | null;
  execution_spec: ExecutionContract;
  signal_spec_hash: string;
  risk_spec_hash: string;
  runs: BacktestSummary[];
}

export interface BacktestEquityPoint {
  timestamp: string;
  equity: string;
  position_quantity: string;
  target_exposure: string;
  realized_exposure: string;
  cumulative_fees: string;
}

export interface BacktestEquity {
  version: string;
  product: string;
  series: BacktestEquityPoint[];
  metadata: {
    source_count: number;
    returned_count: number;
    max_points: number;
    aggregation: string;
    aggregated: boolean;
    initial_equity: string;
  };
}

export interface BacktestFill {
  timestamp: string;
  side: string;
  reference_price: string;
  fill_price: string;
  quantity_delta: string;
  notional: string;
  fee: string;
  slippage_cost: string;
  position_after: string;
  equity_after: string;
}

export interface BacktestFillPage {
  version: string;
  product: string;
  fills: BacktestFill[];
  page: { returned: number; has_more: boolean; total: number; next_cursor: string | null };
}

export interface EmbargoState {
  product: string;
  protected_product: boolean;
  embargoed: boolean;
  window_active: boolean;
  window_elapsed: boolean;
  start: string;
  end: string;
  closes_at: string;
  holdout_id: string;
  holdout_hash: string;
  observed: boolean;
  reason: string;
}

export interface PaperStatus {
  available: boolean;
  reason: string | null;
  shadow_mode: boolean;
  real_money: boolean;
  broker_connected: boolean;
  paper_model_spec_hash: string;
  paper_model_optimized: boolean;
  signal_spec_hash: string;
  risk_spec_hash: string;
  paper_execution: {
    spec_hash: string;
    fee_rate: string;
    slippage_rate: string;
    initial_equity: string;
    currency: string;
    fill_price_policy: string;
    fill_observation_policy: string;
    terminal_liquidation: boolean;
    cost_model: string;
    differs_from_backtest: string[];
  };
  protected_holdout: {
    holdout_id: string; products: string[]; start: string; end: string;
    holdout_hash: string; observed: boolean;
  };
  session: { session_id: string; products: string[]; started_at: string } | null;
  products: string[];
  embargo: Record<string, EmbargoState>;
  events?: number;
}

export interface PaperProductState {
  product: string;
  available: boolean;
  reason: string | null;
  status: string;
  embargo: EmbargoState;
  last_candle: Record<string, string> | null;
  last_prediction: Record<string, string> | null;
  last_signal: Record<string, string> | null;
  last_target: Record<string, string> | null;
  last_fill: Record<string, string> | null;
  portfolio: Record<string, string> | null;
  last_event_at: string | null;
  gap_count: number;
  pipeline_latency?: Record<string, string>;
}

export interface PaperEvent {
  event_id: number;
  event_type: string;
  event_at: string;
  product: string | null;
  natural_key: string | null;
  payload: Record<string, unknown>;
  event_hash: string;
}

export interface PaperEventsPage {
  available: boolean;
  reason: string | null;
  events: PaperEvent[];
  page: { returned: number; last_event_id: number | null };
}

export interface PaperEquity {
  available: boolean;
  reason: string | null;
  product: string;
  series: Array<{ timestamp: string; equity: string; position_quantity: string;
                  cumulative_fees: string }>;
  metadata: { returned_count: number; source_count: number; max_points?: number };
}

/* --- operations (Phase 5E) ------------------------------------------------
 *
 * The ops surface reports on the machine, never on the market. No price, no
 * prediction and no absolute path appears in any of these payloads. */

export type HealthState = 'HEALTHY' | 'DEGRADED' | 'ERROR' | 'STOPPED' | 'EMBARGOED';

export interface HealthRecord {
  record_id: number;
  observed_at: string;
  component: string;
  status: HealthState;
  latency_ms: number | null;
  error_code: string | null;
  details: Record<string, unknown>;
}

export interface HealthHistory {
  available: boolean;
  components: string[];
  states: HealthState[];
  retention: number;
  latest?: Record<string, { status: HealthState; observed_at: string; error_code: string | null }>;
  records: HealthRecord[];
}

export interface OpsRuntime {
  runtime_schema_version: string;
  layout: {
    runtime_schema_version: string;
    directories: Record<string, boolean>;
    paper_database_present: boolean;
    ops_database_present: boolean;
    settings_present: boolean;
  };
  app: {
    state: string;
    reason: string;
    pid?: number;
    host?: string;
    port?: number;
    started_at?: string;
    uptime_seconds?: number | null;
    rss_bytes?: number | null;
  };
  paper_session: { session_id: string; products: string[]; started_at: string } | null;
  snapshots: {
    snapshot_every_events: number;
    status: HealthState;
    error_code?: string;
    products: Record<string, {
      events_since_last_snapshot: number;
      snapshot_due: boolean;
      has_snapshot: boolean;
    }>;
  };
  real_money: boolean;
  broker_connected: boolean;
}

export interface OpsRecovery {
  last_shutdown_clean: boolean | null;
  recovery_performed: boolean;
  event_chain_verified: boolean | null;
  latest_snapshot_verified: boolean | null;
  status: HealthState;
  error_code: string | null;
  events: number;
  sessions: number;
}

export interface OpsStorage {
  paper_database_bytes: number;
  ops_database_bytes: number;
  log_bytes: number;
  export_bytes: number;
  events: number;
  sessions: number;
  snapshots: number;
  log_cap_bytes: number;
  paper_events_retention: string;
  database_warning: string | null;
}

export interface OperationalSettings {
  theme: 'dark' | 'light' | 'system';
  sidebar_collapsed: boolean;
  default_product: string;
  default_chart_window: string;
  time_display: 'utc' | 'local';
  log_retention_preset: string;
  launch_browser: boolean;
  paper_auto_start: boolean;
  schema_version?: string;
}

export interface OpsSettings {
  schema_version: string;
  defaults: OperationalSettings;
  allowed_fields: string[];
  forbidden_trading_fields: string[];
  trading_contracts_immutable: boolean;
  options: Record<string, string[]>;
  current: OperationalSettings;
}

/* --- instruments and providers (Phase 6A) ---------------------------------
 *
 * The registry is the single source of which markets exist. The frontend
 * keeps no list of its own: a hardcoded ["BTC-USD", "ETH-USD"] is a second
 * registry that drifts the day a third instrument is added, and it would
 * still be offering a market the backend had removed. */

export type AssetClass = 'CRYPTO' | 'EQUITY' | 'ETF' | 'INDEX' | 'FX';

export interface Instrument {
  metadata_version: string;
  instrument_id: string;
  venue: string;
  symbol: string;
  asset_class: AssetClass;
  base_asset: string;
  quote_asset: string;
  price_currency: string;
  timezone: string;
  trading_calendar: string;
  native_timeframes: string[];
  quantity_precision: number;
  price_precision: number;
  display_name: string;
  instrument_spec_hash: string;
  providers: string[];
  /** Whether this build trades the instrument or only describes it. */
  tradable: boolean;
  /** What the committed artefacts call it. Null for anything untradable,
   *  because only a tradable instrument appears in an artefact that uses one. */
  legacy_product_id: string | null;
  /** Whether verified local research history exists for it on this machine.
   *  Independent of `tradable`: an instrument can have years of history and
   *  still be one this build refuses to trade. */
  research: InstrumentResearchStatus;
}

export interface InstrumentResearchStatus {
  local_corpus_available: boolean;
  local_corpus_status: CorpusStatus | null;
  corpus_id?: string;
  provider?: string;
  source_timeframe?: string;
  adjustment?: string;
  official_contract?: boolean;
  redistribution_permitted?: boolean;
  rows?: number;
}

export type CorpusStatus = 'AVAILABLE' | 'NOT_INSTALLED' | 'INVALID' | 'CORRUPT';

/** Per-instrument detail. Diagnostic only -- the corpus is atomic, so this
 *  never decides whether anything may be charted. */
export interface CorpusInstrumentDiagnostic {
  instrument_id: string;
  present: boolean;
  rows: number;
  content_hash_matches: boolean;
  canonical_sha256_matches: boolean;
  problem: string | null;
}

export interface ResearchCorpusMetadata {
  provider: string;
  corpus_id: string;
  source_timeframe: string;
  adjustment: string;
  session: string;
  official_contract: boolean;
  redistribution_permitted: boolean;
  local_verified: boolean;
  source_kind: string;
  live: boolean;
  realtime: boolean;
}

export interface ResearchCorpusStatus {
  api_version: string;
  schema_version: string;
  status: CorpusStatus;
  available: boolean;
  reasons: string[];
  instruments: CorpusInstrumentDiagnostic[];
  corpus_id: string | null;
  provider_id: string | null;
  timeframe: string | null;
  adjustment_policy: string | null;
  session_type: string | null;
  requested_range: { start: string; end: string } | null;
  corpus_spec_hash: string | null;
  calendar_spec_hash: string | null;
  corpus_content_hash: string | null;
  official_contract: boolean;
  redistribution_permitted: boolean;
  instruments_expected: string[];
  rows_total?: number;
  expected_sessions?: number;
  capabilities: {
    local_history: boolean;
    live: boolean;
    realtime: boolean;
    prediction: boolean;
    backtest: boolean;
    paper_trading: boolean;
    tradable: boolean;
    download: boolean;
  };
  metadata?: ResearchCorpusMetadata;
}

export interface ResearchBar {
  instrument_id: string;
  bar_open_at: string;
  bar_close_at: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  session_date: string;
}

export interface ResearchBarPage {
  api_version: string;
  instrument_id: string;
  metadata: ResearchCorpusMetadata;
  bars: ResearchBar[];
  page: {
    returned: number;
    limit: number;
    has_more: boolean;
    next_cursor: string | null;
  };
}

export interface InstrumentGroup {
  asset_class: AssetClass;
  instruments: Instrument[];
}

export interface InstrumentsIndex {
  api_version: string;
  count: number;
  tradable_count: number;
  asset_classes: InstrumentGroup[];
  instruments: Instrument[];
}

export interface ProviderCapabilities {
  historical_bars: boolean;
  latest_closed_bar: boolean;
  realtime_ticks: boolean;
  order_book: boolean;
  corporate_actions: boolean;
  market_calendar: string;
  /** How current the most recent row is. END_OF_DAY is never a live price. */
  data_freshness: string;
  authenticated: boolean;
  private_account_data: boolean;
}

export interface Provider {
  schema_version: string;
  provider_id: string;
  display_name: string;
  capabilities: ProviderCapabilities;
  instruments: string[];
}

export interface ProvidersIndex {
  api_version: string;
  schema_version: string;
  count: number;
  providers: Provider[];
}

/** A calendar as the server computed it.
 *
 *  Every derived number arrives finished. The browser never counts bars in a
 *  session or scales a Sharpe ratio: that would be a second implementation of
 *  the session rules, and the two would disagree the first time a holiday
 *  moved. `bars_per_day` is null when the timeframe does not divide the
 *  session evenly -- a 6h30 session holds no whole number of hourly bars, and
 *  saying so beats rounding. */
export interface TradingCalendarView {
  schema_version: string;
  calendar_id: string;
  description: string;
  available: boolean;
  reason?: string;
  timeframe?: string;
  bars_per_day: number | null;
  annualization_periods: number | null;
  timeframe_note?: string;
  spec?: {
    calendar_provider: string;
    calendar_provider_version: string;
    calendar_name: string;
    timezone: string;
    session_type: string;
  };
  spec_hash?: string;
  instruments?: string[];
}

export interface CalendarsIndex {
  api_version: string;
  count: number;
  calendars: TradingCalendarView[];
}

/** One real session, already in UTC, already measured. */
export interface TradingSessionView {
  session_date: string;
  open_at: string;
  close_at: string;
  session_type: string;
  duration_seconds: number;
  early_close: boolean;
  expected_bars: number;
}

export interface InstrumentSessions {
  api_version: string;
  instrument_id: string;
  timeframe: string;
  tradable: boolean;
  start: string;
  end: string;
  calendar: TradingCalendarView;
  /** A market with no session boundaries has none to enumerate, and
   *  `session_count` is null rather than a fabricated row per day. */
  continuous: boolean;
  session_count: number | null;
  early_close_count?: number;
  sessions: TradingSessionView[];
}

export interface InstrumentDetail extends Instrument {
  calendar: TradingCalendarView;
  provider_details: Provider[];
}

/* --- portfolio (Phase 6B) --------------------------------------------------
 *
 * One shared cash ledger across instruments. Every figure is computed in
 * Python and arrives as a Decimal string; the browser formats, never
 * computes. */

export interface PortfolioContract {
  protocol: string;
  portfolio_spec_hash: string;
  frozen: boolean;
  optimized: boolean;
  base_currency: string;
  initial_equity: string;
  max_instrument_abs_exposure: string;
  max_gross_exposure: string;
  max_net_abs_exposure: string;
  allocation_rule: string;
  simultaneous_rebalance_rule: string;
  cash_model: string;
  short_model: string;
  gross_cap_rationale: string;
}

export interface PortfolioStatus {
  api_version: string;
  available: boolean;
  reason: string | null;
  portfolio: PortfolioContract;
  instruments: string[];
  shared_capital: boolean;
  real_money: boolean;
  broker_connected: boolean;
  commercial_edge_established: boolean;
}

export interface PortfolioMetrics {
  initial_equity: string;
  final_equity: string;
  gross_pnl: string;
  net_pnl: string;
  gross_return: string;
  net_return: string;
  total_fees: string;
  total_slippage_cost: string;
  total_execution_cost: string;
  portfolio_turnover: string;
  max_drawdown: string;
  annualized_sharpe: string;
  average_gross_exposure: string;
  average_abs_net_exposure: string;
  max_observed_gross_exposure: string;
  fill_count: number;
  rebalance_count: number;
}

export interface PortfolioRun {
  version: string;
  protocol: string;
  experiment_type: string;
  confirmatory: boolean;
  instruments: string[];
  result_hash: string;
  portfolio_backtest_spec_hash: string;
}

export interface PortfolioBacktestsIndex {
  api_version: string;
  available: boolean;
  reason: string | null;
  portfolio: PortfolioContract;
  runs: PortfolioRun[];
}

export interface PortfolioDetail {
  api_version: string;
  version: string;
  available: boolean;
  portfolio: PortfolioContract;
  instruments: string[];
  experiment_type: string;
  confirmatory: boolean;
  live_execution: boolean;
  cost_model: string;
  commercial_edge_established: boolean;
  metrics: PortfolioMetrics;
  gross_metrics: PortfolioMetrics;
  result_hash: string;
  source: Record<string, unknown>;
  alignment: Record<string, number>;
  equity_points: number;
  fill_count: number;
}

export interface PortfolioEquityPoint {
  timestamp: string;
  equity: string;
  gross_exposure: string;
  net_exposure: string;
}

export interface PortfolioEquity {
  api_version: string;
  version: string;
  series: PortfolioEquityPoint[];
  metadata: {
    source_count: number; returned_count: number; max_points: number;
    aggregation: string; aggregated: boolean; initial_equity: string;
  };
}

export interface PortfolioFill {
  timestamp: string;
  instrument_id: string;
  side: string;
  reference_price: string;
  fill_price: string;
  quantity_delta: string;
  notional: string;
  fee: string;
  slippage_cost: string;
  position_after: string;
}

export interface PortfolioFillPage {
  api_version: string;
  version: string;
  fills: PortfolioFill[];
  page: { returned: number; total: number; has_more: boolean; next_cursor: string | null };
}

export interface InstrumentAttribution {
  instrument_id: string;
  gross_pnl: string;
  fees: string;
  slippage_cost: string;
  execution_cost: string;
  net_pnl: string;
  turnover: string;
  average_abs_exposure: string;
  fill_count: number;
}

export interface PortfolioAttribution {
  api_version: string;
  version: string;
  attribution: InstrumentAttribution[];
  reconciliation: Record<string, string>;
  metrics: { net_pnl: string; gross_pnl: string; total_execution_cost: string };
}

/* --- shared paper portfolio (Phase 6C) -------------------------------------
 *
 * One cash ledger across instruments. The legacy per-product sessions are a
 * separate shape on purpose: their equity is not this portfolio's history and
 * is never added to it. */

export interface PaperPortfolioPosition {
  instrument_id: string;
  quantity: string;
  mark_price: string;
  market_value: string;
  target_exposure: string;
  cumulative_fees: string;
  cumulative_slippage_cost: string;
  cumulative_gross_pnl: string;
}

export interface PaperPortfolioState {
  timestamp: string;
  cash: string;
  equity: string;
  gross_exposure: string;
  net_exposure: string;
  cumulative_fees: string;
  cumulative_slippage_cost: string;
  positions: PaperPortfolioPosition[];
}

export interface PendingBatch {
  decision_at: string;
  required: string[];
  ready: string[];
  missing: string[];
  complete: boolean;
}

export interface PaperPortfolioStatus {
  api_version: string;
  mode: string;
  available: boolean;
  reason: string | null;
  shadow_mode: boolean;
  shared_capital: boolean;
  real_money: boolean;
  broker_connected: boolean;
  commercial_edge_established: boolean;
  portfolio_spec_hash: string;
  initial_equity: string;
  max_instrument_abs_exposure: string;
  max_gross_exposure: string;
  allocation_rule: string;
  simultaneous_rebalance_rule: string;
  paper_execution: {
    spec_hash: string; fill_price_policy: string; fill_observation_policy: string;
  };
  protected_holdout: {
    holdout_id: string; start: string; end: string; holdout_hash: string;
    observed: boolean;
  };
  instruments: string[];
  session_id: string | null;
  active_session: Record<string, unknown> | null;
  embargo: Record<string, { embargoed: boolean; reason: string }>;
  events?: number;
  chain?: { verified: boolean; events?: number };
  snapshot_verified?: boolean;
  state?: PaperPortfolioState | null;
  pending_batches?: Record<string, PendingBatch>;
  fill_count?: number;
  rebalance_count?: number;
}

export interface PaperPortfolioPending {
  api_version: string;
  available: boolean;
  status: string;
  pending_batches: Record<string, PendingBatch>;
}

export interface PaperPortfolioEquity {
  api_version: string;
  available: boolean;
  series: { timestamp: string; equity: string; cash: string;
            gross_exposure: string; net_exposure: string }[];
  metadata: { source_count: number; returned_count?: number; max_points?: number };
}

export interface PaperLegacy {
  api_version: string;
  available: boolean;
  label: string;
  shared_capital: boolean;
  note: string;
  session: Record<string, unknown> | null;
  sessions: number;
  events: number;
}

// ---- official event sources (FOMC V1, read-only store) ----------------------------------------

export interface FomcSourceBase {
  api_version: string;
  source: 'fomc';
  provider_id: string;
  spec_revision: number;
  spec_hash: string;
  schema_version: string;
  read_only: true;
}

export interface FomcSourceStatus extends FomcSourceBase {
  status: 'AVAILABLE' | 'NOT_CONFIGURED' | 'REJECTED';
  store: string | null;
  reason?: string;
  horizon?: number;
  first_durable_activity?: string | null;
  last_durable_activity?: string | null;
  /** SERVER_NOW_LB: the latest instant the store attests; a later read is unresolved. */
  suggested_as_of?: string | null;
  counts?: { epochs: number; responses: number; revisions: number; observations: number; cycles: number };
}

export interface FomcSnapshotHeader {
  policy: string;
  spec_hash: string;
  mode: string;
  T: string;
  H: number;
  P?: number | null;
  read_state: string;
  identity: string;
}

export interface FomcHealthSurface {
  result_state: string | null;
  reason: string;
  check: number | null;
  check_at: string | null;
  outcome?: string | null;
  record?: number | null;
  attempt?: number | null;
}

export interface FomcItemSummary {
  sid: string;
  state: string;
  step: number;
  revision: string | null;
  content_hash: string | null;
  live_available: boolean | null;
  title: string | null;
  official_statement_date: string | null;
  declared_release_at: string | null;
  declared_release_text: string | null;
  declared_release_trust_verdict: string | null;
  observation_mode: string | null;
  canonical_source_url: string | null;
  content_domain: string | null;
  observations: number;
}

export interface FomcSnapshotView extends FomcSourceBase {
  pagination?: SourcePagination;
  snapshot: FomcSnapshotHeader;
  discovery: { state: string; cycle_id?: number | null; B?: number | null } | null;
  health: Record<string, FomcHealthSurface> | null;
  items: FomcItemSummary[];
}

export interface FomcRevision {
  revision_id: string;
  committed_seq: number;
  content_hash: string | null;
  first_raw_sha256: string | null;
  observation_mode: string | null;
  content_identity: { identity: string; canonicalizer: string; domain: string; bytes_sha256: string } | null;
}

export interface FomcObservation {
  record: number;
  attempt: number;
  mode: string;
  verdict: string;
  status: number | null;
  observed_at: string | null;
  wall_at_receipt: string | null;
  request_url: string | null;
  final_url: string | null;
  redirect_chain: string[] | null;
  raw_sha256: string | null;
  byte_length: number | null;
  processing_outcome: string | null;
  revision: string | null;
}

export interface FomcItemDetail extends FomcSourceBase {
  pagination?: SourcePagination;
  snapshot: FomcSnapshotHeader;
  item: (Record<string, unknown> & { sid: string; state: string }) | null;
  revisions: FomcRevision[];
  observations: FomcObservation[];
}

export interface FomcReplay extends FomcSourceBase {
  snapshot: FomcSnapshotHeader;
  replay_identity: string | null;
  identical: boolean;
  error: string | null;
}

// ---- SEC EDGAR (offline slice, read-only store) -----------------------------------------------

export interface EdgarSourceBase {
  api_version: string;
  source: 'edgar';
  provider_id: string;
  spec_revision: number;
  spec_hash: string;
  schema_version: string;
  read_only: true;
}

export interface EdgarSourceStatus extends EdgarSourceBase {
  status: 'AVAILABLE' | 'NOT_CONFIGURED' | 'REJECTED';
  store: string | null;
  reason?: string;
  horizon?: number;
  first_durable_activity?: string | null;
  last_durable_activity?: string | null;
  suggested_as_of?: string | null;
  watchlist?: string[];
  counts?: { epochs: number; responses: number; revisions: number; observations: number; absences: number };
}

export interface EdgarHealth {
  result_state: string | null;
  reason: string;
  check_at: string | null;
  attempt: number | null;
  record: number | null;
}

export interface EdgarFilingSummary {
  accession_number: string;
  cik: string;
  form: string;
  filing_date: string;
  report_date: string | null;
  items: string | null;
  state: 'PRESENT' | 'ABSENT_FROM_LISTING';
  revisions_seen: number;
  observations: number;
  first_available_at: string | null;
  first_observed_at: string | null;
  amendment_link: string | null;
  /** Provenance only: never an availability. */
  acceptance_datetime_text: string;
  entity_name: string | null;
}

export interface EdgarSnapshotView extends EdgarSourceBase {
  pagination?: SourcePagination;
  snapshot: FomcSnapshotHeader;
  watchlist: string[] | null;
  health: Record<string, EdgarHealth> | null;
  filings: EdgarFilingSummary[];
}

export interface EdgarFilingDetail extends EdgarSourceBase {
  pagination?: SourcePagination;
  snapshot: FomcSnapshotHeader;
  filing: (Record<string, unknown> & { accession_number: string; state: string }) | null;
  revisions: { revision: string; committed_seq: number; content_sha256: string; first_record: number;
    fields: Record<string, string | number | null> }[];
  observations: { record: number; observed_at: string | null; raw_sha256: string; position: number;
    revision: string; entity_name: string | null }[];
  absences: { record: number; observed_at: string | null; raw_sha256: string; filing_date: string;
    listing_oldest_filing_date: string }[];
}

export interface EdgarReplay extends EdgarSourceBase {
  snapshot: FomcSnapshotHeader;
  replay_identity: string | null;
  identical: boolean;
  error: string | null;
}

/** Optional on older source API responses. Totals describe the complete frozen read. */
export interface SourcePagination {
  limit: number;
  totals: Record<string, number>;
  next_cursor: string | null;
}

export interface SourceTimeline {
  snapshot: FomcSnapshotHeader;
  rows: { committed_seq: number; kind: string; key: string | null; body: Record<string, unknown> }[];
  pagination?: SourcePagination;
}
