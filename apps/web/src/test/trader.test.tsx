/**
 * The Agent trader view over real-shaped responses: traderFixtures.json is produced by the trader API itself from its
 * SYNTHETIC runner (tests/trader_agent/export_views.py; a Python test fails if it drifts). Nothing here is a market result.
 *
 * The recurring assertions are about what the page must not claim: a pending label is not realized, an absent run shows no
 * numbers, the disclaimer is never hidden, and no request is anything but a GET.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import { joinLedger, paperLines, runState, scoredReturn, todayRows, rejections } from '../lib/trader';
import type { TraderLabels, TraderLedger, TraderToday } from '../api/traderTypes';
import raw from './traderFixtures.json';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const fx: any = raw;
type State = 'issued' | 'partial' | 'realized';

function reply(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function mock(state: State | 'empty', overrides: Record<string, unknown> = {}) {
  const e = fx.empty;
  const s = fx.shared;
  const st = state === 'empty' ? null : fx.states[state];
  return vi.fn((input: string, init?: RequestInit) => {
    expect(init?.method ?? 'GET').toBe('GET');
    const url = new URL(String(input), 'http://x');
    const leaf = url.pathname.replace('/api/v1/trader/', '');
    const after = Number(url.searchParams.get('after') ?? 0);
    if (url.pathname === '/api/v1/health') return reply({ core_status: 'ready' });
    if (leaf in overrides) {
      const o = overrides[leaf] as { __status?: number; error?: string } | null;
      return o && o.__status ? reply({ error: o.error }, o.__status) : reply(o);
    }
    if (state === 'empty') return reply(e[leaf]);
    if (leaf === 'ledger' || leaf === 'labels') {
      const page = leaf === 'ledger' ? s.ledger : st.labels;
      return reply(after > 0 ? { ...page, records: [], next_after: null } : page);
    }
    if (leaf === 'scorecard') return reply(st.scorecard);
    if (leaf === 'today') return reply(url.searchParams.get('date') === '2026-10-10' ? s.skipped_day : s.today);
    if (leaf === 'context') return reply(s.context);
    return leaf in s ? reply(s[leaf]) : reply({ error: 'no such endpoint' }, 404);
  });
}

function open(path: string, state: State | 'empty', overrides?: Record<string, unknown>) {
  const fetchMock = mock(state, overrides);
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
  return fetchMock;
}

beforeEach(() => invalidate());
afterEach(() => { vi.unstubAllGlobals(); });

describe('fixtures are what the API serves', () => {
  it('are synthetic and keep the real shapes', () => {
    expect(fx.synthetic).toBe(true);
    expect(fx.shared.ledger.records[0].payload.outputs.return).toBeNull();
    expect(fx.shared.today.runs.at(-1).payload.decision.views.length).toBeGreaterThan(40);
    expect(fx.states.issued.labels.records).toHaveLength(0);
  });
});

describe('Agent trader, Beginner', () => {
  it('always shows the paper-experiment disclaimer and the views of the day', async () => {
    open('/trader?date=2026-10-07', 'realized');
    expect(await screen.findByText(/paper experiment, not advice/)).toBeInTheDocument();
    expect(screen.getByText('PAPER ONLY · NO PROVEN EDGE')).toBeInTheDocument();
    expect(await screen.findByText('SYNTHETIC DEMO DATA')).toBeInTheDocument();
    const table = await screen.findByRole('table', { name: 'Views per asset' });
    expect(within(table).getAllByRole('row').length).toBe(1 + 10);
    const row = within(table).getAllByRole('row').find((r) => r.textContent?.startsWith('AAPL1d'));
    expect(row?.textContent).toMatch(/Up/);
    expect(row?.textContent).toMatch(/56 %/);
    expect(row?.textContent).toMatch(/probability it beats SPY/);
  });

  it('shows what the reviewer rejected or lowered, and disagreement as no view', async () => {
    open('/trader?date=2026-10-07', 'realized');
    const section = await screen.findByRole('region', { name: 'Rejected by the reviewer' });
    expect(within(section).getByText(/Rejected:/)).toBeInTheDocument();
    expect(within(section).getByText(/Lowered:/)).toBeInTheDocument();
    expect(section.textContent).toMatch(/does not support the claim/);
    const table = await screen.findByRole('table', { name: 'Views per asset' });
    const msft = within(table).getAllByRole('row').filter((r) => r.textContent?.startsWith('MSFT'));
    expect(msft.every((r) => /No view/.test(r.textContent ?? ''))).toBe(true);
  });

  it('separates finished paper results from waiting ones', async () => {
    open('/trader?date=2026-10-07', 'partial');
    const results = await screen.findByRole('table', { name: 'Paper results' });
    expect(results.textContent).toMatch(/waiting for the first outcome|\d+ ?%/);
    expect(await screen.findByText(/recorded predictions have a measured outcome/)).toBeInTheDocument();
  });

  it('says plainly that no run exists yet and shows no numbers', async () => {
    open('/trader?date=2026-10-07', 'empty');
    expect(await screen.findByText(/No run yet for this day/)).toBeInTheDocument();
    expect(screen.getByText('No views for this day')).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Views per asset' })).not.toBeInTheDocument();
    expect(screen.getByText(/paper experiment, not advice/)).toBeInTheDocument();
  });

  it('says a skipped day was skipped', async () => {
    open('/trader?date=2026-10-10', 'realized');
    expect(await screen.findByText(/Run skipped — 2026-10-10/)).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Views per asset' })).not.toBeInTheDocument();
  });

  it('keeps the page and the disclaimer when a request fails, with a retry', async () => {
    open('/trader?date=2026-10-07', 'realized', { today: { __status: 503, error: 'trader evidence unavailable' } });
    expect(await screen.findByText('trader evidence unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByText(/paper experiment, not advice/)).toBeInTheDocument();
  });
});

describe('Agent trader, Expert', () => {
  it('puts both analysts side by side with sources, counter-thesis, falsifier and reviewer verdicts', async () => {
    open('/trader?mode=expert&date=2026-10-07&product=XLK', 'realized');
    const section = await screen.findByRole('region', { name: 'XLK 1d analysts' });
    expect(within(section).getByRole('article', { name: 'Claude analyst' })).toBeInTheDocument();
    expect(within(section).getByRole('article', { name: 'GPT analyst' })).toBeInTheDocument();
    expect(within(section).getAllByText('Counter-thesis')).toHaveLength(2);
    expect(within(section).getAllByText('Falsifier')).toHaveLength(2);
    expect(section.textContent).toMatch(/REVIEWER DOWNGRADE/);
    const link = within(section).getAllByRole('link')[0];
    expect(link).toHaveAttribute('href', 'https://example.invalid/synthetic');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noreferrer'));
    expect(section.textContent).toMatch(/published 2026-10-07 11:00:48 UTC/);
  });

  it('labels ledger rows pending or realized without rewriting predictions', async () => {
    open('/trader?mode=expert&date=2026-10-07', 'partial');
    const ledger = await screen.findByRole('table', { name: 'Ledger' });
    await waitFor(() => expect(within(ledger).getAllByText(/^REALIZED/).length).toBeGreaterThan(0));
    expect(within(ledger).getAllByText('PENDING').length).toBeGreaterThan(0);
    expect(screen.getByText(/never rewrites its prediction/)).toBeInTheDocument();
  });

  it('shows only pending labels before any outcome exists', async () => {
    open('/trader?mode=expert&date=2026-10-07', 'issued');
    const ledger = await screen.findByRole('table', { name: 'Ledger' });
    expect(within(ledger).queryByText(/^REALIZED/)).not.toBeInTheDocument();
    expect(await screen.findByText(/0 realized/)).toBeInTheDocument();
  });

  it('shows baselines with sample sizes, kept vs rejected, calibration with no claim, and run health', async () => {
    open('/trader?mode=expert&date=2026-10-07', 'realized');
    const baselines = await screen.findByRole('table', { name: 'Baseline comparison' });
    for (const name of ['Consensus', 'always_up', 'momentum20', 'random_seeded', 'spy_relative_zero']) {
      expect(within(baselines).getByText(name)).toBeInTheDocument();
    }
    expect(within(baselines).getByText('Scored')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Kept vs rejected' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Calibration of consensus/ })).toBeInTheDocument();
    expect(screen.getByText(/No calibration is claimed/)).toBeInTheDocument();
    expect(screen.getByText('PENDING MINIMUM SAMPLE')).toBeInTheDocument();
    const health = await screen.findByRole('region', { name: 'Run health' });
    expect(await within(health).findByText('HEALTHY')).toBeInTheDocument();
    expect(within(health).getByText(/COMPLETE · 2026-10-10/)).toBeInTheDocument();
  });

  it('says the supervisor never wrote when there is no health file', async () => {
    open('/trader?mode=expert&date=2026-10-07', 'realized', { health: { schema: 'trader-health-view-v1', health: null, last_label: null, paused: false } });
    expect(await screen.findByText('never written')).toBeInTheDocument();
    expect(screen.getByText('never ran')).toBeInTheDocument();
  });

  it('keeps day and asset when the mode is switched', async () => {
    open('/trader?date=2026-10-07&product=MSFT', 'realized');
    await screen.findByRole('table', { name: 'Views per asset' });
    expect(screen.getByLabelText('Asset')).toHaveValue('MSFT');
    await userEvent.click(screen.getByRole('button', { name: 'Expert' }));
    expect(await screen.findByRole('region', { name: 'MSFT 1d analysts' })).toBeInTheDocument();
    expect(screen.getByLabelText('Day')).toHaveValue('2026-10-07');
    expect(screen.getByLabelText('Asset')).toHaveValue('MSFT');
  });

  it('plots a chart per asset with a table alternative and markers reachable by keyboard', async () => {
    open('/trader?mode=expert&date=2026-10-07&product=AAPL', 'realized');
    const chart = await screen.findByRole('group', { name: /AAPL: reference prices/ });
    await waitFor(() => expect(chart.querySelectorAll('polygon[tabindex="0"]').length).toBeGreaterThan(0));
    expect(screen.getAllByText('Table view').length).toBeGreaterThan(0);
  });
});

describe('trader derivations', () => {
  const ledger = fx.shared.ledger as TraderLedger;
  const labels = fx.states.realized.labels as TraderLabels;
  const today = fx.shared.today as TraderToday;
  const views = today.runs.at(-1)!.payload.decision!.views;

  it('joins labels by prediction hash and leaves unlabeled predictions pending', () => {
    const rows = joinLedger(ledger.records, labels.records.slice(0, 3));
    expect(rows.filter((r) => r.state === 'realized')).toHaveLength(3);
    expect(rows.filter((r) => r.state === 'pending').every((r) => r.scoredReturn === null)).toBe(true);
    expect(joinLedger(ledger.records, []).every((r) => r.state === 'pending')).toBe(true);
  });

  it('scores equities against SPY and crypto raw', () => {
    const equity = labels.records.find((l) => l.payload.product === 'AAPL')!.payload;
    expect(scoredReturn(equity)).toBe(equity.value.spy_relative_return);
    const crypto = labels.records.find((l) => l.payload.product === 'BTC-USD');
    if (crypto) expect(scoredReturn(crypto.payload)).toBe(crypto.payload.value.raw_return);
  });

  it('derives one consensus row per asset and horizon and the rejections', () => {
    expect(todayRows(views)).toHaveLength(10);
    expect(rejections(views).map((r) => r.verdict).sort()).toEqual(['DOWNGRADE', 'REJECT']);
  });

  it('classifies a day from its statuses', () => {
    expect(runState([])).toBe('none');
    expect(runState(['RUNNING'])).toBe('running');
    expect(runState(['RUNNING', 'COMPLETE'])).toBe('complete');
    expect(runState(['SKIPPED_HOLIDAY'])).toBe('skipped');
    expect(runState(['RUNNING', 'FAILED'])).toBe('failed');
    expect(runState(['PAUSED'])).toBe('paused');
  });

  it('sums finished cohorts without compounding and keeps pending ones apart', () => {
    const cohorts = fx.states.partial.scorecard.portfolio_cohorts;
    const lines = paperLines(cohorts);
    const line = lines.find((l) => l.analyst === 'consensus' && l.horizon === '1d')!;
    const expected = cohorts.filter((c: { analyst: string; horizon: string; state: string }) => c.analyst === 'consensus' && c.horizon === '1d' && c.state === 'COMPLETE')
      .reduce((s: number, c: { variants: { unhedged: { capital_return: number } } }) => s + c.variants.unhedged.capital_return, 0);
    expect(line.unhedged ?? 0).toBeCloseTo(expected, 12);
    expect(line.complete + line.pending).toBe(cohorts.filter((c: { analyst: string; horizon: string }) => c.analyst === 'consensus' && c.horizon === '1d').length);
  });
});
