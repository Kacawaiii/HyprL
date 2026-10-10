/** The Radar home and the Paper accounts: behaviour a reader sees, on synthetic snapshots. */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import { evolution, protectionDistances, rebasedCurves, analystSeries } from '../lib/radar';
import { paperSnapshot, radarHome } from './radarFixtures';

function respond(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function mock(routes: Record<string, unknown>) {
  return vi.fn((input: string) => {
    const path = String(input).split('?')[0] ?? '';
    if (path in routes) {
      const value = routes[path];
      return value instanceof Error ? respond({ error: value.message }, 503) : respond(value);
    }
    if (path === '/api/v1/health') return respond({ status: 'ok', api_version: 'v1', core_status: 'ready' });
    return respond({ error: 'not found' }, 404);
  });
}

async function firstArticle(): Promise<HTMLElement> {
  const articles = await screen.findAllByRole('article');
  return articles[0] as HTMLElement;
}

function renderAt(url: string) {
  return render(<MemoryRouter initialEntries={[url]}><App /></MemoryRouter>);
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('radar home', () => {
  beforeEach(() => vi.stubGlobal('fetch', mock({ '/api/v1/radar/home': radarHome })));

  it('is the index route and ranks events by importance', async () => {
    renderAt('/');
    const articles = await screen.findAllByRole('article');
    expect(articles).toHaveLength(2);
    expect(within(articles[0] as HTMLElement).getByRole('heading', { name: 'Synthetic oil supply shock' })).toBeInTheDocument();
    expect(within(articles[0] as HTMLElement).getByLabelText('Rank 1')).toBeInTheDocument();
    expect(within(articles[1] as HTMLElement).getByLabelText('Rank 2')).toBeInTheDocument();
  });

  it('shows five separate score badges, never one merged number', async () => {
    renderAt('/');
    const first = await firstArticle();
    const scores = within(first).getByLabelText(/five separate measures/i);
    for (const name of ['Event importance', 'Evidence strength', 'Model conviction', 'Predictive quality', 'After-cost performance']) {
      expect(within(scores).getByText(name)).toBeInTheDocument();
    }
    expect(within(scores).getAllByText(/\//).length).toBeGreaterThanOrEqual(2);
    expect(within(scores).getByText('82/100')).toBeInTheDocument();
    expect(within(scores).getByText('65/100')).toBeInTheDocument();
  });

  it('beginner reads watch, why, how long, risk, result', async () => {
    renderAt('/');
    const first = await firstArticle();
    for (const label of ['What to watch', 'Why', 'For how long', 'Risk', 'Result']) {
      expect(within(first).getByText(label)).toBeInTheDocument();
    }
    expect(within(first).getByText('A durable ceasefire.')).toBeInTheDocument();
    expect(within(first).queryByText('Provenance and availability')).not.toBeInTheDocument();
  });

  it('expert adds provenance, availability and run identity on the same data', async () => {
    renderAt('/?mode=expert');
    const first = await firstArticle();
    expect(within(first).getByText('Provenance and availability')).toBeInTheDocument();
    expect(within(first).getByText('Available at T')).toBeInTheDocument();
    expect(within(first).getAllByText(/trader:2026-10-10:synthetic/).length).toBeGreaterThan(0);
    expect(within(first).queryByText('What to watch')).not.toBeInTheDocument();
    expect(await screen.findByText('Model versions')).toBeInTheDocument();
  });

  it('lists each model per concerned asset with its probability, review and the evolution across runs', async () => {
    renderAt('/');
    const first = await firstArticle();
    const table = within(first).getByRole('table', { name: /1d horizon/ });
    const rows = within(table).getAllByRole('row');
    expect(within(rows[1] as HTMLElement).getByText('Claude')).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText('70 %')).toBeInTheDocument();
    expect(within(table).getByText('DOWNGRADE')).toBeInTheDocument();
    expect(within(first).getByText(/60 % → 70 % over 2 runs \(\+10 pts\)/)).toBeInTheDocument();
  });

  it('says plainly when no model covers an asset and when an outcome is realised', async () => {
    renderAt('/');
    const first = await firstArticle();
    expect(within(first).getByText(/No model run covers this asset/)).toBeInTheDocument();
    const outcome = within(first).getByRole('table', { name: 'Realised outcomes' });
    expect(within(outcome).getByText('+1.00 %')).toBeInTheDocument();
    expect(within(outcome).getByText('+0.40 %')).toBeInTheDocument();
  });

  it('filters to events whose assets a model covers', async () => {
    renderAt('/');
    await screen.findAllByRole('article');
    await userEvent.click(screen.getByLabelText(/only events whose assets a model covers/i));
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(1));
  });

  it('keeps links to the other pages carrying the selection', async () => {
    renderAt('/?mode=expert');
    const nav = await screen.findByRole('navigation', { name: 'Go deeper' });
    expect(within(nav).getByRole('link', { name: 'Paper accounts' })).toHaveAttribute('href', '/paper?mode=expert');
  });

  it('opens headline links in a new context without leaking the referrer', async () => {
    renderAt('/');
    const link = await screen.findByRole('link', { name: 'Synthetic oil supply shock' });
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('keeps the old overview reachable', async () => {
    vi.stubGlobal('fetch', mock({ '/api/v1/overview': new Error('x') }));
    renderAt('/overview');
    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Radar/ })).toBeInTheDocument();
  });
});

describe('radar home without a snapshot', () => {
  it('explains how to publish one instead of failing', async () => {
    vi.stubGlobal('fetch', mock({ '/api/v1/radar/home': new Error('radar snapshot unavailable') }));
    renderAt('/');
    expect(await screen.findByText('No radar snapshot is published yet')).toBeInTheDocument();
  });
});

describe('paper accounts', () => {
  beforeEach(() => vi.stubGlobal('fetch', mock({ '/api/v1/radar/paper': paperSnapshot })));

  it('shows the three accounts by suffix only, with the Claude book tagged', async () => {
    renderAt('/paper');
    expect(await screen.findByRole('article', { name: 'AI stocks' })).toBeInTheDocument();
    expect(screen.getByRole('article', { name: 'AI crypto' })).toBeInTheDocument();
    const book = screen.getByRole('article', { name: 'Claude book' });
    expect(within(book).getByText('account …CCCC')).toBeInTheDocument();
    const table = within(book).getByRole('table', { name: /Claude book positions/ });
    expect(within(table).getByText('momo_v0')).toBeInTheDocument();
    expect(within(table).getByText('sleeve')).toBeInTheDocument();
  });

  it('shows stop and target only where a policy defines them', async () => {
    renderAt('/paper');
    const book = await screen.findByRole('article', { name: 'Claude book' });
    const rows = within(book).getAllByRole('row');
    const bat = rows.find((row) => within(row).queryByText('BATUSD'));
    const btc = rows.find((row) => within(row).queryByText('BTCUSD'));
    expect(within(bat as HTMLElement).getByText(/0\.1124/)).toBeInTheDocument();
    expect(within(btc as HTMLElement).getByText('no policy')).toBeInTheDocument();
  });

  it('charts equity against SPY and BTC and lists the journal with reasons', async () => {
    renderAt('/paper');
    expect(await screen.findByRole('img', { name: /re-based to 100 against SPY and BTC/ })).toBeInTheDocument();
    const journal = screen.getByRole('list', { name: 'Trade journal' });
    expect(within(journal).getByText('BAT breaks its 20-day high')).toBeInTheDocument();
    expect(within(journal).getByText(/Invalidation: stop/)).toBeInTheDocument();
  });
});

describe('radar helpers', () => {
  it('describes an evolution and handles the empty and single cases', () => {
    expect(evolution([])).toMatch(/not covered/);
    expect(evolution([{ at: 'a', run_id: 'r', p: 0.6, view: 'UP' }])).toMatch(/only run/);
  });

  it('skips runs where an analyst is absent or has no probability', () => {
    const series = analystSeries([
      { run_id: 'r1', at: 'a', by_analyst: {} },
      { run_id: 'r2', at: 'b', by_analyst: { analyst_gpt: { view: 'UP', p_outperform: null, verdict: null, horizon: '1d' } } },
      { run_id: 'r3', at: 'c', by_analyst: { analyst_gpt: { view: 'UP', p_outperform: 0.7, verdict: null, horizon: '1d' } } },
    ], 'analyst_gpt');
    expect(series.map((s) => s.run_id)).toEqual(['r3']);
  });

  it('re-bases every curve to 100 at its own first point', () => {
    const curves = rebasedCurves(paperSnapshot);
    expect(curves.map((c) => c.key)).toEqual(['ia_actions', 'ia_crypto', 'claude_book', 'SPY', 'BTC']);
    for (const curve of curves) expect(curve.points[0]?.value).toBe(100);
    expect(curves.find((c) => c.key === 'ia_crypto')?.points[2]?.value).toBeCloseTo(100.5, 6);
  });

  it('measures stop and target distance from the last price and never invents one', () => {
    expect(protectionDistances(100, 90, 120)).toEqual({ stop: -0.1, target: 0.2 });
    expect(protectionDistances(100, null, null)).toEqual({ stop: null, target: null });
    expect(protectionDistances(null, 90, 120)).toEqual({ stop: null, target: null });
  });
});
