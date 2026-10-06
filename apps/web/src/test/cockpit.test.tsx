/** The Beginner/Expert cockpit over a mocked API. All data below is SYNTHETIC (labelled by its
 *  provider id) and keeps the real response shapes. */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import { overview, system, markets, paperRunning } from './fixtures';

const hourly = (count: number, start = Date.UTC(2026, 6, 25)) => Array.from({ length: count }, (_, index) => {
  const close = 100 + index;
  return { bar_open_at: new Date(start + index * 3_600_000).toISOString(), open: String(close - 1),
    high: String(close + 1), low: String(close - 2), close: String(close), volume: '1' };
});
const CANDLES = hourly(168);
const LAST = CANDLES[CANDLES.length - 1]!.bar_open_at;
const decision = (index: number) => ({
  timestamp: CANDLES[index]!.bar_open_at, prediction: index % 2 ? '0.01' : '-0.01', direction: index % 2 ? 'LONG' : 'SHORT',
  strength: '0.5', signal_spec_hash: 's'.repeat(64), decision_hash: `d${index}`.padEnd(64, '0'),
  fold_index: 3, model_spec_hash: 'm'.repeat(64), fitted_hash: 'f'.repeat(64),
});
const DECISIONS = [167, 166, 165, 164].map(decision);
const signals = {
  available: true, product: 'BTC-USD', out_of_sample: 'walk-forward', order: 'newest_first',
  signal_spec: { ...system.signal_engine }, decisions: DECISIONS, page: { returned: 4, has_more: false, next_cursor: null, total: 4 },
  counts: { decisions: 4, targets: 4, folds: 1 }, window: { first: CANDLES[0]!.bar_open_at, last: LAST },
  verified_against: 'economic_backtest_v1',
  protocol: { benchmark_protocol: 'p', benchmark_spec_hash: 'b'.repeat(64), economic_backtest_spec_hash: 'e'.repeat(64), economic_results_hash: 'r'.repeat(64) },
  corpus: { corpus_content_hash: 'c'.repeat(64), corpus_spec_hash: 'c'.repeat(64), dataset_hash: 'd'.repeat(64) },
  signal_series_hash: 'x'.repeat(64), position_target_series_hash: 'y'.repeat(64),
};
const risk = {
  available: true, product: 'BTC-USD', risk_spec: { ...system.risk_engine, risk_scale_rule_version: 'v1' },
  targets: [{ timestamp: DECISIONS[0]!.timestamp, side: 'LONG', target_exposure: '0.125', signal_strength: '0.5', position_target_hash: 'p'.repeat(64) }],
  page: { returned: 1, has_more: false, next_cursor: null },
};
const quality = { rank_ic: '0.02', mae: '0.01', rmse: '0.02', observations: 120, predictions: 130, unscored_predictions: 10, label_horizon: 4, scoring_policy: 'w' };
const replay = {
  available: true, reason: null, confirmatory: false, optimized: false,
  window: { start: '2026-05-01T00:00:00+00:00', end: '2026-07-31T23:00:00+00:00', read_cutoff: '2026-08-01T00:00:00+00:00' },
  determinism: { verified: true, replay_count: 2, first_chain_head_hash: 'a', second_chain_head_hash: 'a' },
  limitations: ['historical window already spent'],
  products: ['BTC-USD', 'ETH-USD'].map((name) => ({
    product: name, result_hash: 'a'.repeat(64), hashes: {}, prediction_quality: quality,
    counts: { bars: 10, warmup_without_prediction: 3, fills: 1, signals: {}, targets: {}, gaps: 0, expired_targets: 1, pending_terminal_targets: 0 },
    metrics: { initial_equity: '100000', final_equity: '99000', net_return: '-0.01', net_pnl: '-1000', max_drawdown: '-0.02',
      annualized_sharpe: null, periods_per_year: 8760, total_fees: '5', total_slippage_cost: '4', total_execution_cost: '9' },
  })),
};
const fills = { product: 'BTC-USD', fills: [{ timestamp: CANDLES[100]!.bar_open_at, available_at: CANDLES[101]!.bar_open_at,
  decided_at: CANDLES[99]!.bar_open_at, side: 'BUY', quantity_delta: '0.5', fee: '1', slippage_cost: '1', equity_after: '99999' }],
page: { returned: 1, total: 1, has_more: false, next_cursor: null } };
const FOMC_BASE = { api_version: 'v1', source: 'fomc', provider_id: 'synthetic', spec_revision: 25, spec_hash: 'h'.repeat(64), schema_version: 'fomc-store-v5', read_only: true };
const AS_OF = '2026-08-01T00:00:00+00:00';
const fomcStatus = { ...FOMC_BASE, status: 'AVAILABLE', store: 'synthetic', suggested_as_of: AS_OF };
const fomcSnapshot = { ...FOMC_BASE, discovery: null, health: null,
  snapshot: { policy: 'POLICY', spec_hash: 'h'.repeat(64), mode: 'DURABLE_OBSERVED', T: AS_OF, H: 12, P: 10, read_state: 'FOMC_RESOLVED', identity: 'i'.repeat(64) },
  items: [{ sid: 'a'.repeat(64), state: 'CURRENT_REVISION', step: 1, revision: null, content_hash: null, live_available: true,
    title: 'Synthetic FOMC statement', official_statement_date: '2026-07-28', declared_release_at: CANDLES[60]!.bar_open_at,
    declared_release_text: null, declared_release_trust_verdict: null, observation_mode: 'HISTORICAL_BACKFILL',
    canonical_source_url: null, content_domain: 'CANONICAL', observations: 1 }] };

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function mockApi(overrides: Record<string, unknown | (() => Promise<Response>)> = {}) {
  const routes: Record<string, unknown> = {
    '/api/v1/health': { status: 'ok', api_version: 'v1', core_status: 'ready' },
    '/api/v1/overview': { ...overview, products: overview.products.map((item) => ({ ...item, last_open: LAST, first_open: CANDLES[0]!.bar_open_at })) },
    '/api/v1/system': system, '/api/v1/markets': markets,
    '/api/v1/signals': signals, '/api/v1/risk/targets': risk,
    '/api/v1/markets/BTC-USD': { product: 'BTC-USD', timeframe: '1h', candles: CANDLES, page: { returned: 168, has_more: false, next_cursor: null } },
    '/api/v1/markets/ETH-USD': { product: 'ETH-USD', timeframe: '1h', candles: CANDLES, page: { returned: 168, has_more: false, next_cursor: null } },
    '/api/v1/paper/replay': replay, '/api/v1/paper/replay/BTC-USD/fills': fills, '/api/v1/paper/replay/ETH-USD/fills': { ...fills, product: 'ETH-USD' },
    '/api/v1/sources/fomc': fomcStatus, '/api/v1/sources/fomc/snapshot': fomcSnapshot,
    '/api/v1/paper/status': paperRunning, '/api/v1/research/benchmarks': { benchmarks: [] },
    ...overrides,
  };
  return vi.fn((input: string) => {
    const bare = String(input).split('?')[0] ?? '';
    const route = routes[bare];
    if (typeof route === 'function') return (route as () => Promise<Response>)();
    return bare in routes ? jsonResponse(route) : jsonResponse({ error: `no such endpoint ${bare}` }, 400);
  });
}

function Probe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

function renderCockpit(path = '/cockpit', overrides?: Parameters<typeof mockApi>[0]) {
  const fetch = mockApi(overrides);
  vi.stubGlobal('fetch', fetch);
  render(<MemoryRouter initialEntries={[path]}><App /><Probe /></MemoryRouter>);
  return fetch;
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('Beginner cockpit', () => {
  it('shows the situation, the 4-hour projection with its reference price, and the paper result', async () => {
    renderCockpit();
    const prediction = await screen.findByLabelText('Prediction');
    expect(within(prediction).getByText(/Prediction · next 4 hours/)).toBeInTheDocument();
    // latest decision is bar 167 (close 267), prediction +1 % -> 269.67
    await waitFor(() => expect(within(prediction).getByText(/267 \(close of the decision bar\)/)).toBeInTheDocument());
    expect(within(prediction).getByText('269.67')).toBeInTheDocument();
    expect(within(prediction).getByLabelText('Probabilities and intervals')).toHaveTextContent('not provided');
    const paper = screen.getByLabelText('Paper result');
    expect(within(paper).getByText(/frozen replay/)).toBeInTheDocument();
    expect(screen.getByLabelText('Signal and exposure')).toHaveTextContent('12.50 %');
  });

  it('says "not provided" for TP/SL, with the reason, when nothing supplies them', async () => {
    renderCockpit();
    const card = await screen.findByLabelText('Take profit and stop loss');
    expect(card).toHaveTextContent('Not provided');
    expect(card).toHaveTextContent(/no levels are attached to this frozen reference/);
  });

  it('shows TP/SL with origin, method and gap handling when the policy provides them', async () => {
    const protection = { take_profit: '270', stop_loss: '255', policy_version: 'risk-policy-v2', origin: 'versioned risk policy',
      method: 'fixed 2 percent', gap_treatment: 'filled at the gap open', intrabar_ambiguity: 'stop first' };
    renderCockpit('/cockpit', { '/api/v1/risk/targets': { ...risk, targets: [{ ...risk.targets[0], protection }] } });
    const card = await screen.findByLabelText('Take profit and stop loss');
    await waitFor(() => expect(card).toHaveTextContent('risk-policy-v2'));
    expect(card).toHaveTextContent('versioned risk policy');
    expect(card).toHaveTextContent('fixed 2 percent');
    expect(card).toHaveTextContent('stop first');
  });

  it('keeps the four scores separate, each with period and sample size', async () => {
    renderCockpit();
    const scores = await screen.findByLabelText('Scores');
    for (const title of ['Data quality', 'Signal strength', 'Model performance', 'Risk and exposure']) {
      const card = within(scores).getByLabelText(`${title} score`);
      expect(within(card).getByText('Period')).toBeInTheDocument();
      expect(within(card).getByText('Sample')).toBeInTheDocument();
      expect(within(card).getByText('Definition')).toBeInTheDocument();
    }
    expect(within(scores).getByLabelText('Model performance score')).toHaveTextContent('120 scored labels');
  });

  it('labels events with their declared release time and claims no market link', async () => {
    renderCockpit();
    const events = await screen.findByLabelText('Relevant events');
    await waitFor(() => expect(events).toHaveTextContent('Synthetic FOMC statement'));
    expect(events).toHaveTextContent('declared release');
    expect(events).toHaveTextContent(/no event-to-price relationship is claimed/);
  });
});

describe('mode switch', () => {
  it('keeps product, period and model in the URL and moves to the Expert views', async () => {
    renderCockpit('/cockpit?product=ETH-USD&model=trading-lab.signal-engine.v1&start=2026-07-28&end=2026-07-31');
    await screen.findByLabelText('Prediction');
    await userEvent.click(screen.getByRole('button', { name: 'Expert' }));
    expect(await screen.findByLabelText('Features and outputs')).toBeInTheDocument();
    const url = new URL(`http://x${screen.getByTestId('location').textContent}`);
    expect(url.searchParams.get('mode')).toBe('expert');
    expect(url.searchParams.get('product')).toBe('ETH-USD');
    expect(url.searchParams.get('start')).toBe('2026-07-28');
    expect(url.searchParams.get('end')).toBe('2026-07-31');
    expect(url.searchParams.get('model')).toBe('trading-lab.signal-engine.v1');
    expect(screen.getByRole('button', { name: 'Expert' })).toHaveAttribute('aria-pressed', 'true');
    await userEvent.click(screen.getByRole('button', { name: 'Beginner' }));
    expect(await screen.findByLabelText('Prediction')).toBeInTheDocument();
    expect(screen.getByTestId('location').textContent).not.toContain('mode=');
    expect(screen.getByTestId('location').textContent).toContain('product=ETH-USD');
  });

  it('carries the selection through the navigation links', async () => {
    renderCockpit('/cockpit?mode=expert&product=ETH-USD');
    await screen.findByLabelText('Features and outputs');
    expect(screen.getByRole('link', { name: /Risk/ })).toHaveAttribute('href', '/risk?mode=expert&product=ETH-USD');
  });

  it('changes the product through the selection form and requests that product', async () => {
    const fetch = renderCockpit();
    await screen.findByLabelText('Prediction');
    await userEvent.selectOptions(screen.getByLabelText('Product'), 'ETH-USD');
    await waitFor(() => expect(screen.getByTestId('location').textContent).toContain('product=ETH-USD'));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).startsWith('/api/v1/signals') && String(url).includes('product=ETH-USD'))).toBe(true));
  });
});

describe('Expert cockpit', () => {
  it('shows exact outputs, provenance, models, costs and diagnostics, and says what is not served', async () => {
    renderCockpit('/cockpit?mode=expert');
    const outputs = await screen.findByLabelText('Features and outputs');
    await waitFor(() => expect(within(outputs).getAllByText('0.01').length).toBeGreaterThan(0));
    expect(outputs).toHaveTextContent(/does not serve the feature vector/);
    expect(screen.getByLabelText('Models and parameters')).toHaveTextContent('FROZEN · NOT OPTIMISED');
    expect(screen.getByLabelText('Snapshots and provenance')).toHaveTextContent('i'.repeat(16));
    const diagnostics = screen.getByLabelText('Diagnostics and exclusions');
    expect(diagnostics).toHaveTextContent('Warm-up bars without prediction');
    expect(diagnostics).toHaveTextContent(/Drift and market regimes.*not provided/);
    expect(screen.getByLabelText('Splits, baselines, metrics and costs')).toHaveTextContent(/Baseline comparisons are not served/);
    expect(screen.getAllByLabelText(/score$/)).toHaveLength(4);
  });

  it('lists executions of the period from the frozen replay', async () => {
    renderCockpit('/cockpit?mode=expert');
    const executions = await screen.findByLabelText('Decisions, executions and portfolio');
    await waitFor(() => expect(executions).toHaveTextContent('1 executions in the period'));
  });
});

describe('chart', () => {
  it('steps through marks with the keyboard and announces them', async () => {
    renderCockpit();
    const chart = await screen.findByRole('img', { name: /Timeline: 168 closes, 4 decisions, 1 executions, 1 events/ });
    chart.focus();
    await userEvent.keyboard('{Home}');
    await waitFor(() => expect(document.getElementById('cockpit-chart-readout')?.textContent).toMatch(/Decision|Execution|Event/));
    await userEvent.keyboard('{End}');
    expect(document.getElementById('cockpit-chart-readout')?.textContent).toMatch(/Decision SHORT|Decision LONG/);
  });

  it('refuses a period longer than one chart rather than sampling prices', async () => {
    renderCockpit('/cockpit?start=2026-01-01&end=2026-07-31');
    expect(await screen.findAllByText('Period too long for one chart')).not.toHaveLength(0);
  });
});

describe('loading, errors and empty runs', () => {
  it('shows a retryable error when the overview fails, without losing navigation', async () => {
    renderCockpit('/cockpit', { '/api/v1/overview': () => jsonResponse({ error: 'boom' }, 500) });
    expect(await screen.findByText('boom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Sections' })).toBeInTheDocument();
  });

  it('shows a loading state first', async () => {
    let release: (response: Response) => void = () => undefined;
    renderCockpit('/cockpit', { '/api/v1/overview': () => new Promise<Response>((resolve) => { release = resolve; }) });
    expect(screen.getAllByRole('status').some((node) => node.textContent?.includes('Loading cockpit'))).toBe(true);
    // settle the shared in-flight request so it cannot leak into the next test
    release(await jsonResponse({ ...overview, products: [] }));
    await screen.findByLabelText('Selection');
  });

  it('reports a failed price request in the chart only', async () => {
    renderCockpit('/cockpit', { '/api/v1/markets/BTC-USD': () => jsonResponse({ error: 'prices down' }, 500) });
    expect(await screen.findByText('prices down')).toBeInTheDocument();
    expect(screen.getByLabelText('Signal and exposure')).toBeInTheDocument();
  });

  it('shows empty predictions, not a made-up one, when no run is persisted', async () => {
    renderCockpit('/cockpit', { '/api/v1/signals': { ...signals, available: false, decisions: [], reason: 'no persisted signal run available' },
      '/api/v1/risk/targets': { ...risk, available: false, targets: [] } });
    expect(await screen.findByText('No prediction in this period')).toBeInTheDocument();
    expect(screen.getByLabelText('Signal and exposure')).toHaveTextContent('No exposure target in this period.');
  });
});
