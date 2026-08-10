/** Payload fixtures mirroring real API responses (values copied from committed
 *  artefacts, so a shape change in Python shows up as a failing test here). */
import type { BenchmarkSummary, CandlePage, ChartSeries, MarketsIndex, Overview, RiskView, SignalsView, SystemInfo } from '../api/types';

export const capabilities = {
  market_history: true, signal_engine: true, position_target: true,
  economic_backtest: false, paper_trading: false, live_trading: false,
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
