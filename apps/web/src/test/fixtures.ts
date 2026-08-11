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
  ],
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
