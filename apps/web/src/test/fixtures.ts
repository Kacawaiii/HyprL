/** Payload fixtures mirroring real API responses (values copied from committed
 *  artefacts, so a shape change in Python shows up as a failing test here). */
import type { BenchmarkSummary, CandlePage, ChartSeries, MarketsIndex, Overview, RiskView, SignalsView, SystemInfo } from '../api/types';

export const capabilities = {
  market_history: true, signal_engine: true, position_target: true,
  economic_backtest: true,  paper_trading: true, live_trading: false,
  realtime_stream: false,
};

export const overview: Overview = {
  api_version: 'trading-lab.app-api.v1',
  system_status: 'ready',
  signal_spec_hash: '7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939',
  risk_spec_hash: 'f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad',
  products: [
    { product: 'BTC-USD', timeframe: '1h', rows: 8750, first_open: '2025-08-01T00:00:00+00:00',
      last_open: '2026-07-31T23:00:00+00:00', missing_openings: 10,
      latest_price: null, latest_price_available: false },
    { product: 'ETH-USD', timeframe: '1h', rows: 8750, first_open: '2025-08-01T00:00:00+00:00',
      last_open: '2026-07-31T23:00:00+00:00', missing_openings: 10,
      latest_price: null, latest_price_available: false },
  ],
  benchmarks: [
    { version: 'v1', protocol_version: 'trading-lab.real-benchmark.v1',
      experiment_type: 'confirmatory_pending', confirmatory_result: false,
      corpus_content_hash: '688c250d', products: [
        { product: 'BTC-USD', rank_ic: '-0.003040593', mae: '0.00686238', rmse: '0.010178',
          observations: 7728, folds: 46, benchmark_spec_hash: 'dd7c474b',
          dataset_hash: 'f37c0b05', benchmark_results_hash: '0343014d' }]},
    { version: 'v2', protocol_version: 'trading-lab.real-benchmark.v2',
      experiment_type: 'exploratory', confirmatory_result: false,
      corpus_content_hash: '688c250d', products: [
        { product: 'BTC-USD', rank_ic: '-0.007970077', mae: '0.00649862', rmse: '0.00963739',
          observations: 7728, folds: 46, benchmark_spec_hash: 'ec4d19b5',
          dataset_hash: '4375db4f', benchmark_results_hash: 'e810c0e2' }]},
  ] as BenchmarkSummary[],
  capabilities,
};

export const system: SystemInfo = {
  api_version: 'trading-lab.app-api.v1',
  signal_engine: {
    protocol: 'trading-lab.signal-engine.v1', rule: 'static-symmetric-threshold-v1',
    spec_hash: overview.signal_spec_hash, frozen: true, optimized: false,
    long_threshold: '0.0025', short_threshold: '-0.0025',
    full_strength_excess: '0.01', prediction_horizon: 4, boundary_semantics: 'strict',
  },
  risk_engine: {
    protocol: 'trading-lab.risk-engine.v1', spec_hash: overview.risk_spec_hash,
    frozen: true, optimized: false, max_long_exposure: '0.25', max_short_exposure: '0.25',
    risk_scale: '1', volatility_scaling_enabled: false,
    strength_mapping_version: 'linear-strength-to-exposure-v1',
  },
  market_data: {
    available: true, corpus_id: 'coinbase_history_v1',
    corpus_spec_hash: '7a9de4d8', corpus_content_hash: '688c250d',
    products: ['BTC-USD', 'ETH-USD'], timeframe: '1h',
    point_in_time_revision_history: false,
  },
  benchmarks: {
    v1_available: true, v2_exploratory_available: true, v2_confirmatory_observed: false,
    confirmatory_holdout: {
      holdout_id: 'coinbase_confirmatory_2026q4', range_start: '2026-09-01T00:00:00Z',
      range_end: '2026-11-30T23:00:00Z', products: ['BTC-USD', 'ETH-USD'],
      timeframe: '1h', captured: false, single_use: true,
    },
  },
  capabilities,
};

export const markets: MarketsIndex = {
  timeframe: '1h', corpus_id: 'coinbase_history_v1', corpus_content_hash: '688c250d',
  products: [
    { product: 'BTC-USD', rows: 8750, first_open: '2025-08-01T00:00:00+00:00',
      last_open: '2026-07-31T23:00:00+00:00', missing_openings: 10 },
    { product: 'ETH-USD', rows: 8750, first_open: '2025-08-01T00:00:00+00:00',
      last_open: '2026-07-31T23:00:00+00:00', missing_openings: 10 },
  ],
};

/** A corpus missing one registered instrument, to prove the page says so. */
export const marketsMissingEth: MarketsIndex = {
  ...markets,
  products: markets.products.filter((item) => item.product === 'BTC-USD'),
};

function candle(index: number) {
  const base = 60000 + index;
  return {
    bar_open_at: new Date(Date.UTC(2026, 6, 1, index % 24)).toISOString(),
    open: String(base), high: String(base + 20), low: String(base - 20),
    close: String(base + 5), volume: '1.5',
  };
}

export const candlePage: CandlePage = {
  product: 'BTC-USD', timeframe: '1h',
  candles: Array.from({ length: 200 }, (_, index) => candle(index)),
  page: { returned: 200, limit: 200, has_more: true, next_cursor: 'abc' },
};

export const chart: ChartSeries = {
  product: 'BTC-USD',
  series: Array.from({ length: 120 }, (_, index) => candle(index)),
  metadata: {
    source_timeframe: '1h', aggregation: 'ohlc-bucket', bucket_size: 18,
    source_count: 8750, returned_count: 120, max_points: 500, aggregated: true,
  },
};

export const signals: SignalsView = {
  available: false,
  reason: 'no persisted signal run available',
  signal_spec: { ...system.signal_engine, optimized: false },
  decisions: [],
  page: { returned: 0, has_more: false, next_cursor: null },
};

export const risk: RiskView = {
  available: false,
  reason: 'no persisted position target run available',
  risk_spec: { ...system.risk_engine, risk_scale_rule_version: 'constant-unit-scale-v1' },
  targets: [],
  page: { returned: 0, has_more: false, next_cursor: null },
};

export const backtestsEmpty = {
  available: false,
  reason: 'no persisted economic backtest available',
  execution_spec: {
    protocol: 'trading-lab.execution.v1',
    execution_spec_hash: 'e'.repeat(64),
    fee_rate: '0.0010', slippage_rate: '0.0005', initial_equity: '100000',
    currency: 'USD',
    fill_policy: 'next-contiguous-bar-open-after-decision-v1',
    mark_policy: 'next-observable-open-v1',
    instrument_model: 'synthetic-linear-usd-notional-v1',
    cost_model: 'synthetic', optimized: false, exchange_account_specific: false,
  },
  signal_spec_hash: 's'.repeat(64),
  risk_spec_hash: 'r'.repeat(64),
  runs: [],
};

const metrics = {
  initial_equity: '100000', final_equity: '98750.5', net_return: '-0.0124950',
  gross_return: '-0.0031000', net_pnl: '-1249.5', gross_pnl: '-310.0',
  total_fees: '740.25', total_slippage_cost: '370.10',
  total_execution_cost: '1110.35', turnover_ratio: '7.4025',
  max_drawdown: '-0.0412', annualized_sharpe: '-0.31', periods_per_year: 8760,
  fill_count: 412, rebalance_count: 460, expired_target_count: 3,
  average_abs_exposure: '0.1837', exposure_time_fraction: '0.87',
};

export const backtests = {
  ...backtestsEmpty,
  available: true,
  reason: null,
  runs: [
    {
      version: 'v1', product: 'BTC-USD', experiment_type: 'exploratory',
      confirmatory: false, live_execution: false, cost_model: 'synthetic',
      source_benchmark_protocol: 'trading-lab.real-benchmark.v2',
      economic_backtest_spec_hash: 'a'.repeat(64),
      economic_results_hash: 'b'.repeat(64),
      window: { first_fill_at: '2025-09-08T02:00:00+00:00',
                last_fill_at: '2026-07-29T21:00:00+00:00',
                liquidation_at: '2026-07-29T22:00:00+00:00' },
      metrics,
    },
    {
      version: 'v1', product: 'ETH-USD', experiment_type: 'exploratory',
      confirmatory: false, live_execution: false, cost_model: 'synthetic',
      source_benchmark_protocol: 'trading-lab.real-benchmark.v2',
      economic_backtest_spec_hash: 'c'.repeat(64),
      economic_results_hash: 'd'.repeat(64),
      window: { first_fill_at: '2025-09-08T02:00:00+00:00',
                last_fill_at: '2026-07-29T21:00:00+00:00',
                liquidation_at: '2026-07-29T22:00:00+00:00' },
      metrics: { ...metrics, net_return: '0.0044', final_equity: '100440' },
    },
  ],
};

export const backtestEquity = {
  version: 'v1', product: 'BTC-USD',
  series: [
    { timestamp: '2025-09-08T02:00:00+00:00', equity: '100000', position_quantity: '0.1',
      target_exposure: '0.05', realized_exposure: '0.05', cumulative_fees: '5' },
    { timestamp: '2025-10-08T02:00:00+00:00', equity: '99200', position_quantity: '0.1',
      target_exposure: '0.05', realized_exposure: '0.04', cumulative_fees: '210' },
    { timestamp: '2026-07-29T22:00:00+00:00', equity: '98750.5', position_quantity: '0',
      target_exposure: '0', realized_exposure: '0', cumulative_fees: '740.25' },
  ],
  metadata: { source_count: 7728, returned_count: 3, max_points: 500,
              aggregation: 'bucket-extrema', aggregated: true,
              initial_equity: '100000' },
};

export const backtestFills = {
  version: 'v1', product: 'BTC-USD',
  fills: [
    { timestamp: '2025-09-08T02:00:00+00:00', side: 'buy', reference_price: '60000',
      fill_price: '60030', quantity_delta: '0.2083', notional: '12506',
      fee: '12.5', slippage_cost: '6.25', position_after: '0.2083',
      equity_after: '99981.25' },
  ],
  page: { returned: 1, has_more: true, total: 412, next_cursor: 'abc' },
};

const embargo = (product: string, embargoed = false) => ({
  product, protected_product: true, embargoed,
  window_active: embargoed, window_elapsed: false,
  start: '2026-09-01T00:00:00Z', end: '2026-11-30T23:00:00Z',
  closes_at: '2026-12-01T00:00:00Z',
  holdout_id: 'coinbase_confirmatory_2026q4', holdout_hash: 'h'.repeat(64),
  observed: false,
  reason: embargoed
    ? 'paper trading for ' + product + ' is disabled to preserve the confirmatory research holdout 2026-09-01T00:00:00Z..2026-11-30T23:00:00Z'
    : 'paper trading for ' + product + ' is allowed until the embargo boundary at 2026-09-01T00:00:00Z',
});

const paperExecution = {
  spec_hash: 'x'.repeat(64), fee_rate: '0.0010', slippage_rate: '0.0005',
  initial_equity: '100000', currency: 'USD',
  fill_price_policy: 'next-contiguous-bar-open-after-decision-v1',
  fill_observation_policy: 'recorded-when-the-fill-bar-closes-v1',
  terminal_liquidation: false, cost_model: 'synthetic',
  differs_from_backtest: ['a live session never liquidates a terminal position'],
};

export const paperStopped = {
  available: false, reason: 'no shadow session is running',
  shadow_mode: true, real_money: false, broker_connected: false,
  paper_model_spec_hash: 'm'.repeat(64), paper_model_optimized: false,
  signal_spec_hash: 's'.repeat(64), risk_spec_hash: 'r'.repeat(64),
  paper_execution: paperExecution,
  protected_holdout: { holdout_id: 'coinbase_confirmatory_2026q4',
    products: ['BTC-USD', 'ETH-USD'], start: '2026-09-01T00:00:00Z',
    end: '2026-11-30T23:00:00Z', holdout_hash: 'h'.repeat(64), observed: false },
  session: null, products: ['BTC-USD', 'ETH-USD'],
  embargo: { 'BTC-USD': embargo('BTC-USD'), 'ETH-USD': embargo('ETH-USD') },
};

export const paperRunning = {
  ...paperStopped, available: true, reason: null, events: 42,
  session: { session_id: 'paper-20260811T031900Z',
             products: ['BTC-USD', 'ETH-USD'], started_at: '2026-08-11T03:19:00+00:00' },
};

export const paperEmbargoed = {
  ...paperRunning,
  embargo: { 'BTC-USD': embargo('BTC-USD', true), 'ETH-USD': embargo('ETH-USD', true) },
};

export const paperProducts = {
  products: [
    {
      product: 'BTC-USD', available: true, reason: null, status: 'RUNNING',
      embargo: embargo('BTC-USD'),
      last_candle: { bar_open_at: '2026-08-11T02:00:00+00:00', open: '64000',
                     high: '64200', low: '63900', close: '64100', volume: '12.5' },
      last_prediction: { prediction: '0.0031', bar_open_at: '2026-08-11T02:00:00+00:00',
                         decision_available_at: '2026-08-11T03:00:00+00:00' },
      last_signal: { direction: 'LONG', strength: '0.24' },
      last_target: { target_exposure: '0.06', side: 'LONG' },
      last_fill: { side: 'buy', fill_price: '64032.00', quantity_delta: '0.09' },
      portfolio: { equity: '99871.20', position_quantity: '0.09',
                   cumulative_fees: '64.03', cumulative_slippage_cost: '32.02',
                   fill_count: '2' },
      last_event_at: '2026-08-11T03:00:05+00:00', gap_count: 0,
    },
    {
      product: 'ETH-USD', available: false,
      reason: 'the session has produced no event for this product yet',
      status: 'STARTING', embargo: embargo('ETH-USD'),
      last_candle: null, last_prediction: null, last_signal: null,
      last_target: null, last_fill: null, portfolio: null,
      last_event_at: null, gap_count: 0,
    },
  ],
};

export const paperEvents = {
  available: true, reason: null,
  events: [
    { event_id: 1, event_type: 'CANDLE_INGESTED', event_at: '2026-08-11T03:00:01+00:00',
      product: 'BTC-USD', natural_key: '2026-08-11T02:00:00+00:00', payload: {},
      event_hash: 'e'.repeat(64) },
    { event_id: 2, event_type: 'PREDICTION_CREATED',
      event_at: '2026-08-11T03:00:02+00:00', product: 'BTC-USD',
      natural_key: '2026-08-11T02:00:00+00:00', payload: {}, event_hash: 'f'.repeat(64) },
  ],
  page: { returned: 2, last_event_id: 2 },
};

export const paperEquity = {
  available: true, reason: null, product: 'BTC-USD',
  series: [
    { timestamp: '2026-08-11T01:00:00+00:00', equity: '100000',
      position_quantity: '0', cumulative_fees: '0' },
    { timestamp: '2026-08-11T02:00:00+00:00', equity: '99871.20',
      position_quantity: '0.09', cumulative_fees: '64.03' },
  ],
  metadata: { returned_count: 2, source_count: 2, max_points: 500 },
};

/* --- operations (Phase 5E) ---------------------------------------------- */

export const opsRuntime = {
  runtime_schema_version: 'trading-lab.runtime-layout.v1',
  layout: {
    runtime_schema_version: 'trading-lab.runtime-layout.v1',
    directories: { runtime: true, logs: true, exports: true, support: true, tmp: true },
    paper_database_present: true, ops_database_present: true, settings_present: true,
  },
  app: {
    state: 'RUNNING', reason: 'running', pid: 4242, host: '127.0.0.1', port: 8787,
    started_at: '2026-08-11T12:00:00Z', uptime_seconds: 5400, rss_bytes: 41943040,
  },
  paper_session: null,
  snapshots: {
    snapshot_every_events: 250, status: 'HEALTHY' as const,
    products: {
      'BTC-USD': { events_since_last_snapshot: 198, snapshot_due: false, has_snapshot: true },
      'ETH-USD': { events_since_last_snapshot: 12, snapshot_due: false, has_snapshot: true },
    },
  },
  real_money: false, broker_connected: false,
};

export const opsRecoveryClean = {
  last_shutdown_clean: true, recovery_performed: false,
  event_chain_verified: true, latest_snapshot_verified: true,
  status: 'HEALTHY' as const, error_code: null, events: 2930, sessions: 2,
};

export const opsRecoveryUnclean = {
  ...opsRecoveryClean, last_shutdown_clean: false, recovery_performed: true,
};

export const opsRecoveryBroken = {
  ...opsRecoveryClean, event_chain_verified: false, status: 'ERROR' as const,
  error_code: 'PAPER_EVENT_CHAIN_INVALID',
};

export const opsStorage = {
  paper_database_bytes: 5124096, ops_database_bytes: 16384, log_bytes: 581,
  export_bytes: 0, events: 5860, sessions: 2, snapshots: 4,
  log_cap_bytes: 52428800,
  paper_events_retention: 'append-only; never pruned automatically',
  database_warning: null,
};

export const opsHealth = {
  available: true,
  components: ['app_api', 'paper_engine', 'event_store', 'market_ingestion', 'model', 'holdout_guard'],
  states: ['HEALTHY', 'DEGRADED', 'ERROR', 'STOPPED', 'EMBARGOED'] as const,
  retention: 10000,
  latest: {
    app_api: { status: 'HEALTHY' as const, observed_at: '2026-08-11T12:00:00Z', error_code: null },
    market_ingestion: {
      status: 'DEGRADED' as const, observed_at: '2026-08-11T11:00:00Z',
      error_code: 'MARKET_NETWORK_UNAVAILABLE',
    },
    holdout_guard: { status: 'EMBARGOED' as const, observed_at: '2026-08-11T11:00:00Z', error_code: null },
  },
  records: [],
};

export const opsSettings = {
  schema_version: 'trading-lab.settings.v1',
  defaults: {
    theme: 'dark' as const, sidebar_collapsed: false, default_product: 'BTC-USD',
    default_chart_window: '30d', time_display: 'utc' as const,
    log_retention_preset: 'standard', launch_browser: true, paper_auto_start: false,
  },
  allowed_fields: ['default_chart_window', 'default_product', 'launch_browser',
    'log_retention_preset', 'paper_auto_start', 'sidebar_collapsed', 'theme', 'time_display'],
  forbidden_trading_fields: ['fee_rate', 'holdout_end', 'model_alpha', 'risk_cap',
    'signal_threshold', 'slippage_rate'],
  trading_contracts_immutable: true,
  options: {
    theme: ['dark', 'light', 'system'], time_display: ['utc', 'local'],
    default_chart_window: ['24h', '7d', '30d', '90d', 'all'],
    default_product: ['BTC-USD', 'ETH-USD'],
    log_retention_preset: ['small', 'standard', 'large'],
  },
  current: {
    theme: 'dark' as const, sidebar_collapsed: false, default_product: 'BTC-USD',
    default_chart_window: '30d', time_display: 'utc' as const,
    log_retention_preset: 'standard', launch_browser: true, paper_auto_start: false,
  },
};

/* --- instruments and providers (Phase 6A) ------------------------------- */

const btcInstrument = {
  metadata_version: 'trading-lab.instrument.v1',
  instrument_id: 'coinbase:BTC-USD', venue: 'coinbase', symbol: 'BTC-USD',
  asset_class: 'CRYPTO' as const, base_asset: 'BTC', quote_asset: 'USD',
  price_currency: 'USD', timezone: 'UTC', trading_calendar: 'CRYPTO_24_7',
  native_timeframes: ['1h', '1d'], quantity_precision: 8, price_precision: 2,
  display_name: 'Bitcoin / US Dollar',
  instrument_spec_hash:
    '492c167c1e66a37a377cff8b4e135841c5a13a7c60324ec9b5c8b1976bf5701f',
  providers: ['coinbase-public-v1'], tradable: true,
  legacy_product_id: 'BTC-USD' as string | null,
};

const ethInstrument = {
  ...btcInstrument,
  instrument_id: 'coinbase:ETH-USD', symbol: 'ETH-USD', base_asset: 'ETH',
  display_name: 'Ether / US Dollar',
  instrument_spec_hash:
    '2a9e1d1c922fbb9af68ad92f8f2638e951a2fef830afb6e514a29ebfab35d7f3',
  legacy_product_id: 'ETH-USD',
};

/** The four US markets this build describes but does not trade.
 *
 *  Their venue is xnas because that is where they list; the provider that
 *  serves them is a separate fact and never appears in the identity. Every
 *  one carries `tradable: false` and a null legacy id, which is what stops
 *  them reaching a picker that drives a backtest. */
const aaplInstrument = {
  metadata_version: 'trading-lab.instrument.v1',
  instrument_id: 'xnas:AAPL', venue: 'xnas', symbol: 'AAPL',
  asset_class: 'EQUITY' as const, base_asset: 'AAPL', quote_asset: 'USD',
  price_currency: 'USD', timezone: 'America/New_York',
  trading_calendar: 'US_EQUITY_REGULAR',
  native_timeframes: ['30m', '1d'], quantity_precision: 0, price_precision: 2,
  display_name: 'Apple Inc.',
  instrument_spec_hash:
    'b9bb3ddb9309e527a98dcaa8ee4704bbcb2068997cae2aff244cd21cb8f354fb',
  providers: ['massive-stocks-historical-v1'], tradable: false,
  legacy_product_id: null as string | null,
};

const qqqInstrument = {
  ...aaplInstrument,
  instrument_id: 'xnas:QQQ', symbol: 'QQQ', base_asset: 'QQQ',
  asset_class: 'ETF' as const, display_name: 'Invesco QQQ Trust, Series 1',
  instrument_spec_hash:
    'adaf3edbc85da84eb695445a4ef3a1a44460883c070c5bd0e58c33fc85cd559b',
};

export const instruments = {
  api_version: 'trading-lab.app-api.v1',
  count: 4,
  tradable_count: 2,
  asset_classes: [
    { asset_class: 'CRYPTO' as const, instruments: [btcInstrument, ethInstrument] },
    { asset_class: 'EQUITY' as const, instruments: [aaplInstrument] },
    { asset_class: 'ETF' as const, instruments: [qqqInstrument] },
  ],
  instruments: [btcInstrument, ethInstrument, aaplInstrument, qqqInstrument],
};

/** Only the two crypto markets, for the tests about an all-tradable registry. */
export const instrumentsTradableOnly = {
  ...instruments,
  count: 2,
  tradable_count: 2,
  asset_classes: [
    { asset_class: 'CRYPTO' as const, instruments: [btcInstrument, ethInstrument] },
  ],
  instruments: [btcInstrument, ethInstrument],
};

export const instrumentDetail = {
  ...btcInstrument,
  calendar: {
    schema_version: 'trading-lab.trading-calendar.v1',
    calendar_id: 'CRYPTO_24_7',
    description: 'continuous trading, no session boundaries or holidays',
    available: true, timeframe: '1h',
    bars_per_day: 24, annualization_periods: 8760,
  },
  provider_details: [],
};

export const equityInstrumentDetail = {
  ...aaplInstrument,
  calendar: {
    schema_version: 'trading-lab.equity-calendar.v1',
    calendar_id: 'US_EQUITY_REGULAR',
    description:
      'US equity regular session (NYSE/NASDAQ), holidays and early closes',
    available: true, timeframe: '30m',
    bars_per_day: 13, annualization_periods: 3263,
    spec: {
      calendar_provider: 'pandas_market_calendars',
      calendar_provider_version: '5.4.0',
      calendar_name: 'XNYS',
      timezone: 'America/New_York',
      session_type: 'REGULAR',
    },
    spec_hash:
      '1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314',
  },
  provider_details: [],
};

export const calendars = {
  api_version: 'trading-lab.app-api.v1',
  count: 2,
  calendars: [
    { ...instrumentDetail.calendar,
      instruments: ['coinbase:BTC-USD', 'coinbase:ETH-USD'] },
    { ...equityInstrumentDetail.calendar,
      instruments: ['xnas:AAPL', 'xnas:MSFT', 'xnas:NVDA', 'xnas:QQQ'] },
  ],
};

/** A Thanksgiving week: the holiday absent, the day after short. */
export const equitySessions = {
  api_version: 'trading-lab.app-api.v1',
  instrument_id: 'xnas:AAPL',
  timeframe: '30m',
  tradable: false,
  start: '2026-11-23',
  end: '2026-11-30',
  calendar: equityInstrumentDetail.calendar,
  continuous: false,
  session_count: 5,
  early_close_count: 1,
  sessions: [
    { session_date: '2026-11-23', open_at: '2026-11-23T14:30:00Z',
      close_at: '2026-11-23T21:00:00Z', session_type: 'REGULAR',
      duration_seconds: 23400, early_close: false, expected_bars: 13 },
    { session_date: '2026-11-24', open_at: '2026-11-24T14:30:00Z',
      close_at: '2026-11-24T21:00:00Z', session_type: 'REGULAR',
      duration_seconds: 23400, early_close: false, expected_bars: 13 },
    { session_date: '2026-11-25', open_at: '2026-11-25T14:30:00Z',
      close_at: '2026-11-25T21:00:00Z', session_type: 'REGULAR',
      duration_seconds: 23400, early_close: false, expected_bars: 13 },
    { session_date: '2026-11-27', open_at: '2026-11-27T14:30:00Z',
      close_at: '2026-11-27T18:00:00Z', session_type: 'REGULAR',
      duration_seconds: 12600, early_close: true, expected_bars: 7 },
    { session_date: '2026-11-30', open_at: '2026-11-30T14:30:00Z',
      close_at: '2026-11-30T21:00:00Z', session_type: 'REGULAR',
      duration_seconds: 23400, early_close: false, expected_bars: 13 },
  ],
};

export const cryptoSessions = {
  api_version: 'trading-lab.app-api.v1',
  instrument_id: 'coinbase:BTC-USD',
  timeframe: '1h',
  tradable: true,
  start: '2026-11-23',
  end: '2026-11-30',
  calendar: instrumentDetail.calendar,
  continuous: true,
  session_count: null,
  sessions: [],
};

export const providers = {
  api_version: 'trading-lab.app-api.v1',
  schema_version: 'trading-lab.instrument-registry.v1',
  count: 1,
  providers: [{
    schema_version: 'trading-lab.market-provider.v1',
    provider_id: 'coinbase-public-v1',
    display_name: 'Coinbase (public market data)',
    capabilities: {
      historical_bars: true, latest_closed_bar: true, realtime_ticks: false,
      order_book: false, corporate_actions: false,
      market_calendar: 'CRYPTO_24_7', data_freshness: 'LIVE',
      authenticated: false, private_account_data: false,
    },
    instruments: ['coinbase:BTC-USD', 'coinbase:ETH-USD'],
  }, {
    schema_version: 'trading-lab.massive-provider.v1',
    provider_id: 'massive-stocks-historical-v1',
    display_name: 'Massive (US stocks, historical)',
    capabilities: {
      historical_bars: true, latest_closed_bar: false, realtime_ticks: false,
      order_book: false, corporate_actions: true,
      market_calendar: 'US_EQUITY_REGULAR', data_freshness: 'END_OF_DAY',
      // A market-data key buys prices and nothing that belongs to an account.
      authenticated: true, private_account_data: false,
    },
    instruments: ['xnas:AAPL', 'xnas:MSFT', 'xnas:NVDA', 'xnas:QQQ'],
  }],
};

/** A registry whose equity is tradable, to prove grouping is not hardcoded.
 *
 *  The real seed equities are all reference-only, so they would never reach an
 *  <optgroup> in the picker. This variant makes one tradable purely to exercise
 *  the grouping path -- it describes no market this build actually has. */
const tradableAapl = {
  ...aaplInstrument, tradable: true, legacy_product_id: 'AAPL' as string | null,
};

export const instrumentsWithEquity = {
  api_version: 'trading-lab.app-api.v1',
  count: 3,
  tradable_count: 3,
  asset_classes: [
    { asset_class: 'CRYPTO' as const, instruments: [btcInstrument, ethInstrument] },
    { asset_class: 'EQUITY' as const, instruments: [tradableAapl] },
  ],
  instruments: [btcInstrument, ethInstrument, tradableAapl],
};

export { aaplInstrument, qqqInstrument };

/* --- portfolio (Phase 6B) ----------------------------------------------- */

export const portfolioContract = {
  protocol: 'trading-lab.portfolio.v1',
  portfolio_spec_hash:
    '32ec6c9f5f62c24bd18077dda79eefb30a334edcf26b811175f9b93584cdcebf',
  frozen: true, optimized: false, base_currency: 'USD',
  initial_equity: '100000', max_instrument_abs_exposure: '0.25',
  max_gross_exposure: '0.50', max_net_abs_exposure: '0.50',
  allocation_rule: 'proportional-gross-cap-v1',
  simultaneous_rebalance_rule: 'single-pretrade-equity-batch-v1',
  cash_model: 'shared-cash-v1', short_model: 'synthetic-linear-short-v1',
  gross_cap_rationale:
    "two instruments at RiskSpec V1's 25 % each; the structure the existing " +
    'rules already permit, not a fitted optimum',
};

export const portfolioEmpty = {
  api_version: 'trading-lab.app-api.v1', available: false,
  reason: 'no portfolio backtest has been run',
  portfolio: portfolioContract, instruments: ['BTC-USD', 'ETH-USD'],
  shared_capital: true, real_money: false, broker_connected: false,
  commercial_edge_established: false,
};

export const portfolioBacktestsEmpty = {
  api_version: 'trading-lab.app-api.v1', available: false,
  reason: 'no portfolio backtest has been run',
  portfolio: portfolioContract, runs: [],
};

/* --- shared paper portfolio (Phase 6C) ---------------------------------- */

const paperPortfolioContract = {
  api_version: 'trading-lab.app-api.v1', mode: 'SHARED_PORTFOLIO',
  shadow_mode: true, shared_capital: true, real_money: false,
  broker_connected: false, commercial_edge_established: false,
  portfolio_spec_hash:
    '32ec6c9f5f62c24bd18077dda79eefb30a334edcf26b811175f9b93584cdcebf',
  initial_equity: '100000', max_instrument_abs_exposure: '0.25',
  max_gross_exposure: '0.50',
  allocation_rule: 'proportional-gross-cap-v1',
  simultaneous_rebalance_rule: 'single-pretrade-equity-batch-v1',
  paper_execution: {
    spec_hash: 'bb944167fbcc', 
    fill_price_policy: 'next-contiguous-bar-open-after-decision-v1',
    fill_observation_policy: 'recorded-when-the-fill-bar-closes-v1',
  },
  protected_holdout: {
    holdout_id: 'coinbase_confirmatory_2026q4', start: '2026-09-01T00:00:00Z',
    end: '2026-11-30T23:00:00Z', holdout_hash: 'bf95ee8577bb', observed: false,
  },
  instruments: ['coinbase:BTC-USD', 'coinbase:ETH-USD'],
  embargo: {
    'coinbase:BTC-USD': { embargoed: false, reason: 'allowed' },
    'coinbase:ETH-USD': { embargoed: false, reason: 'allowed' },
  },
};

export const paperPortfolioEmpty = {
  ...paperPortfolioContract, available: false,
  reason: 'no shared portfolio session recorded',
  session_id: null, active_session: null,
};

export const paperPortfolioRunning = {
  ...paperPortfolioContract, available: true, reason: null,
  session_id: 'portfolio-20260812T000000Z',
  active_session: { session_id: 'portfolio-20260812T000000Z' },
  events: 512, chain: { verified: true, events: 512 }, snapshot_verified: true,
  fill_count: 4, rebalance_count: 9,
  state: {
    timestamp: '2026-08-12T03:00:00+00:00',
    cash: '74981.22', equity: '99872.55',
    gross_exposure: '0.2489', net_exposure: '0.2489',
    cumulative_fees: '61.40', cumulative_slippage_cost: '30.70',
    positions: [
      { instrument_id: 'coinbase:BTC-USD', quantity: '0.41230000',
        mark_price: '60300.00', market_value: '24861.69',
        target_exposure: '0.25', cumulative_fees: '38.20',
        cumulative_slippage_cost: '19.10', cumulative_gross_pnl: '-45.10' },
      { instrument_id: 'coinbase:ETH-USD', quantity: '0', mark_price: '3010.00',
        market_value: '0', target_exposure: '0', cumulative_fees: '23.20',
        cumulative_slippage_cost: '11.60', cumulative_gross_pnl: '12.40' },
    ],
  },
  pending_batches: {},
};

export const paperPortfolioPendingIdle = {
  api_version: 'trading-lab.app-api.v1', available: true, status: 'IDLE',
  pending_batches: {},
};

export const paperPortfolioPendingWaiting = {
  api_version: 'trading-lab.app-api.v1', available: true,
  status: 'WAITING_FOR_PORTFOLIO_BATCH',
  pending_batches: {
    '2026-08-12T03:00:00+00:00': {
      decision_at: '2026-08-12T03:00:00+00:00',
      required: ['coinbase:BTC-USD', 'coinbase:ETH-USD'],
      ready: ['coinbase:BTC-USD'], missing: ['coinbase:ETH-USD'],
      complete: false,
    },
  },
};

export const paperPortfolioEquity = {
  api_version: 'trading-lab.app-api.v1', available: true,
  series: Array.from({ length: 12 }, (_, index) => ({
    timestamp: new Date(Date.UTC(2026, 7, 12, index)).toISOString(),
    equity: String(100000 - index * 12), cash: String(75000 - index),
    gross_exposure: '0.2489', net_exposure: '0.2489',
  })),
  metadata: { source_count: 12, returned_count: 12, max_points: 500 },
};

export const paperLegacy = {
  api_version: 'trading-lab.app-api.v1', available: true,
  label: 'PRE-SHARED-PORTFOLIO', shared_capital: false,
  note: 'independent per-product accounts; their equity is not the history of '
      + 'the shared portfolio and is never added to it',
  session: null, sessions: 2, events: 5860,
};
