/**
 * Signals and Risk with a persisted out-of-sample walk-forward run: the pages show the backend's
 * decisions and targets verbatim, with the OOS label and provenance, page back with the served
 * cursor, and keep the honest unavailable answer (with its reason) otherwise.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import * as fixtures from './fixtures';
import type { RiskView, SignalsView } from '../api/types';

const H = (c: string) => c.repeat(64);
const provenance = {
  product: 'BTC-USD', out_of_sample: 'walk_forward', order: 'newest_first',
  signal_series_hash: H('a'), position_target_series_hash: H('b'),
  counts: { decisions: 7728, targets: 7728, folds: 46 },
  window: { first: '2025-09-08T01:00:00+00:00', last: '2026-07-31T00:00:00+00:00' },
  verified_against: 'economic_backtest_v1',
  protocol: {
    benchmark_protocol: 'trading-lab.real-benchmark.v2', benchmark_spec_hash: H('c'),
    economic_backtest_spec_hash: H('d'), economic_results_hash: H('e'),
  },
  corpus: { corpus_content_hash: H('f'), corpus_spec_hash: H('1'), dataset_hash: H('2') },
};
const decision = (t: string, direction: 'LONG' | 'FLAT' | 'SHORT', strength: string, n: string) => ({
  timestamp: t, prediction: '0.0031', direction, strength, signal_spec_hash: H('7'),
  decision_hash: H(n), fold_index: 45, model_spec_hash: H('8'), fitted_hash: H('9'),
  out_of_sample: 'walk_forward',
});
const signalsPage1: SignalsView = {
  ...fixtures.signals, ...provenance, available: true, reason: undefined,
  decisions: [decision('2026-07-31T00:00:00+00:00', 'LONG', '0.2400', '3'),
              decision('2026-07-30T23:00:00+00:00', 'FLAT', '0', '4')],
  page: { returned: 2, has_more: true, next_cursor: 'cursor-1', total: 3 },
};
const signalsPage2: SignalsView = {
  ...signalsPage1,
  decisions: [decision('2026-07-30T22:00:00+00:00', 'SHORT', '0.5', '5')],
  page: { returned: 1, has_more: false, next_cursor: null, total: 3 },
};
const target = (t: string, side: string, exposure: string, n: string) => ({
  timestamp: t, side: side as 'LONG', target_exposure: exposure, signal_strength: '0.24',
  position_target_hash: H(n), raw_target_exposure: exposure, risk_scale: '1', out_of_sample: 'walk_forward',
});
const riskPage: RiskView = {
  ...fixtures.risk, ...provenance, available: true, reason: undefined,
  targets: [target('2026-07-31T00:00:00+00:00', 'LONG', '0.0600', '6')],
  page: { returned: 1, has_more: false, next_cursor: null, total: 1 },
};

function install(routes: Record<string, (url: URL) => unknown>) {
  routes = { '/api/v1/markets': () => fixtures.markets, ...routes };
  const fetch = vi.fn((input: string) => {
    const url = new URL(String(input), 'http://localhost');
    const route = routes[url.pathname];
    const payload = route ? route(url) : {};
    return Promise.resolve({
      ok: true, status: 200, headers: new Headers({ 'content-type': 'application/json' }),
      json: () => Promise.resolve(payload), text: () => Promise.resolve(JSON.stringify(payload)),
    } as Response);
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('signals with a persisted run', () => {
  it('shows the decisions verbatim with the out-of-sample label and pages back', async () => {
    const fetch = install({
      '/api/v1/signals': (url) => (url.searchParams.get('cursor') ? signalsPage2 : signalsPage1),
    });
    render(<MemoryRouter initialEntries={['/signals']}><App /></MemoryRouter>);
    expect(await screen.findByText(/Out-of-sample: walk_forward/)).toBeInTheDocument();
    expect(screen.getByText('0.2400')).toBeInTheDocument();
    expect(screen.getByText('FLAT')).toBeInTheDocument();
    expect(screen.getByText('economic_backtest_v1')).toBeInTheDocument();
    expect(fetch.mock.calls.map(([path]) => String(path))).toContain(
      '/api/v1/signals?limit=200',
    );
    await userEvent.click(screen.getByRole('button', { name: /Load older decisions/ }));
    await waitFor(() => expect(screen.getByText('SHORT')).toBeInTheDocument());
    expect(fetch.mock.calls.map(([path]) => String(path))).toContain(
      '/api/v1/signals?limit=200&cursor=cursor-1',
    );
    expect(screen.queryByRole('button', { name: /Load older decisions/ })).toBeNull();
  });

  it('keeps the unavailable answer and shows its reason', async () => {
    install({
      '/api/v1/signals': () => ({
        ...fixtures.signals, reason: 'persisted signal run refused: stored decisions differ from the replayed ones',
      }),
    });
    render(<MemoryRouter initialEntries={['/signals']}><App /></MemoryRouter>);
    expect(await screen.findByText(/No persisted signal run available/i)).toBeInTheDocument();
    expect(screen.getByText(/persisted signal run refused/)).toBeInTheDocument();
    expect(screen.queryByText(/Out-of-sample/)).toBeNull();
  });
});

describe('risk with a persisted run', () => {
  it('shows the targets verbatim with the out-of-sample label', async () => {
    install({ '/api/v1/risk/targets': () => riskPage });
    render(<MemoryRouter initialEntries={['/risk']}><App /></MemoryRouter>);
    expect(await screen.findByText(/Out-of-sample: walk_forward/)).toBeInTheDocument();
    expect(screen.getByText('0.0600')).toBeInTheDocument();
    expect(screen.getAllByText('Target exposure').length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/No persisted position target run available/i)).toBeNull();
  });

  it('switches product and keeps the unavailable answer when there is no run', async () => {
    const fetch = install({
      '/api/v1/risk/targets': (url) => (url.searchParams.get('product') === 'ETH-USD'
        ? { ...fixtures.risk, reason: 'no persisted position target run available' }
        : riskPage),
    });
    render(<MemoryRouter initialEntries={['/risk']}><App /></MemoryRouter>);
    await screen.findByText(/Out-of-sample: walk_forward/);
    await userEvent.selectOptions(await screen.findByRole('combobox'), 'ETH-USD');
    expect(await screen.findByText(/No persisted position target run available/i)).toBeInTheDocument();
    expect(fetch.mock.calls.map(([path]) => String(path))).toContain(
      '/api/v1/risk/targets?limit=200&product=ETH-USD',
    );
  });
});
