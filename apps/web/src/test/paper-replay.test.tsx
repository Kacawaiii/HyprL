import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { PaperPage } from '../pages/PaperPage';
import { PaperReplaySection } from '../pages/PaperReplaySection';
import { invalidate } from '../state/useQuery';
import type { PaperReplaySummary, PaperReplayProduct } from '../api/types';
import { paperPortfolioEmpty, paperPortfolioPendingIdle, paperLegacy } from './fixtures';

const product: PaperReplayProduct = {
  product: 'BTC-USD', result_hash: 'a'.repeat(64), hashes: { fitted_hash: 'b'.repeat(64) },
  prediction_quality: { rank_ic: '-0.123456789012345678901234567890', mae: '0.0089', rmse: '0.0123',
    observations: 10, predictions: 14, unscored_predictions: 4, label_horizon: 4, scoring_policy: 'window only' },
  counts: { bars: 14, warmup_without_prediction: 0, fills: 101, signals: { LONG: 2, FLAT: 10, SHORT: 2 },
    targets: { LONG: 2, FLAT: 10, SHORT: 2 }, gaps: 0, expired_targets: 0, pending_terminal_targets: 1 },
  metrics: { initial_equity: '100000', final_equity: '98765.43210987654321', net_return: '-0.01234567890123456789',
    net_pnl: '-1234.567890123456789', max_drawdown: '-0.0567890123456789', annualized_sharpe: '-1.23456789',
    periods_per_year: 8760, total_fees: '12.345', total_slippage_cost: '6.789', total_execution_cost: '19.134' },
};
const summary: PaperReplaySummary = {
  available: true, reason: null, products: [product, { ...product, product: 'ETH-USD' }],
  confirmatory: false, optimized: false,
  window: { start: '2026-05-01T00:00:00+00:00', end: '2026-07-31T23:00:00+00:00', read_cutoff: '2026-08-01T00:00:00+00:00' },
  determinism: { verified: true, replay_count: 2, first_chain_head_hash: 'a', second_chain_head_hash: 'a' },
  limitations: ['historical window already spent', 'synthetic costs; no terminal liquidation'],
};

function respond(data: unknown, status = 200) {
  return Promise.resolve({ ok: status === 200, status, text: () => Promise.resolve(JSON.stringify(data)) } as Response);
}

function mockFetch(replay: unknown = summary, status = 200) {
  return vi.fn((input: string) => {
    const url = String(input);
    if (url.includes('/replay/') && url.includes('/equity')) return respond({
      series: [{ timestamp: '2026-05-01T00:00:00+00:00', available_at: '2026-05-01T01:00:00+00:00', equity: '100000', drawdown: '0' }],
      page: { returned: 1, total: 1, has_more: false, next_cursor: null }, metadata: { source_count: 14 },
    });
    if (url.includes('/replay/') && url.includes('/fills')) {
      const next = url.includes('cursor=');
      return respond({ fills: [{ timestamp: next ? '2026-05-03T00:00:00+00:00' : '2026-05-02T00:00:00+00:00',
        available_at: '2026-05-02T01:00:00+00:00', decided_at: '2026-05-01T23:00:00+00:00',
        side: next ? 'sell' : 'buy', quantity_delta: '0.012345', fee: '1.2345', slippage_cost: '0.6789' }],
        page: { returned: 1, total: 101, has_more: !next, next_cursor: next ? null : 'next-page' } });
    }
    if (url === '/api/v1/paper/replay') return respond(replay, status);
    if (url === '/api/v1/paper/portfolio') return respond(paperPortfolioEmpty);
    if (url.includes('/portfolio/pending')) return respond(paperPortfolioPendingIdle);
    if (url.includes('/portfolio/equity')) return respond({ available: false, series: [], metadata: {} });
    if (url === '/api/v1/paper/legacy') return respond(paperLegacy);
    return respond({ error: 'unknown endpoint' }, 404);
  });
}

beforeEach(() => {
  invalidate();
  vi.stubGlobal('fetch', mockFetch());
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it('shows frozen OOS values rounded, with the exact value in the tooltip, on Paper, separate from the live session', async () => {
  render(<PaperPage />);
  expect(await screen.findByRole('heading', { name: 'Shared paper portfolio' })).toBeInTheDocument();
  const section = await screen.findByRole('region', { name: 'Replay hors échantillon (v2)' });
  expect(within(section).getByText(/non confirmatoire, non optimisée/)).toBeInTheDocument();
  const btc = await within(section).findByRole('article', { name: 'Replay BTC-USD' });
  // Rounded for reading, exact value in the tooltip.
  const shown: Array<[string, string]> = [
    ['-0.1235', product.prediction_quality.rank_ic!], ['98,765.43', product.metrics.final_equity!],
    ['-1.23 %', product.metrics.net_return!], ['-5.68 %', product.metrics.max_drawdown!],
    ['-1.235', product.metrics.annualized_sharpe!],
  ];
  for (const [text, exact] of shown) {
    expect(within(btc).getByText(text)).toHaveAttribute('title', exact);
  }
  expect(within(section).getByText(/2 replays identiques/)).toBeInTheDocument();
  expect(await within(btc).findByRole('img', { name: 'Replay equity BTC-USD' })).toBeInTheDocument();
  expect(within(section).getAllByRole('article')).toHaveLength(2);
});

it('pages replay fills without sending a control or live-trading request', async () => {
  const fetcher = mockFetch();
  vi.stubGlobal('fetch', fetcher);
  render(<PaperReplaySection />);
  const btc = await screen.findByRole('article', { name: 'Replay BTC-USD' });
  await userEvent.click(within(btc).getByText('Fills du replay'));
  const next = await within(btc).findByRole('button', { name: 'Suivant' });
  await waitFor(() => expect(next).toBeEnabled());
  await userEvent.click(next);
  expect(await within(btc).findByText('sell')).toBeInTheDocument();
  expect(fetcher.mock.calls.some(([url]) => url.includes('cursor=next-page'))).toBe(true);
  expect(fetcher.mock.calls.every(([url]) => url.startsWith('/api/v1/paper/replay'))).toBe(true);
});

it('reports missing replay evidence without fabricating numbers', async () => {
  vi.stubGlobal('fetch', mockFetch({ available: false, reason: 'not run', products: [] }));
  render(<PaperReplaySection />);
  expect(await screen.findByText('Replay indisponible')).toBeInTheDocument();
  expect(screen.queryByText('Rank IC')).not.toBeInTheDocument();
});

it('shows replay errors and keeps the section labelled', async () => {
  vi.stubGlobal('fetch', mockFetch({ error: 'replay digest mismatch' }, 409));
  render(<PaperReplaySection />);
  expect(await screen.findByRole('alert')).toHaveTextContent('replay digest mismatch');
  expect(screen.getByRole('heading', { name: 'Replay hors échantillon (v2)' })).toBeInTheDocument();
});
