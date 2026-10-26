/** The "Aujourd'hui" home on synthetic snapshots: what a beginner sees at a glance, and what it must never claim. */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import type { RadarHome } from '../api/radarTypes';
import { invalidate } from '../state/useQuery';
import { dollars, pctFr, parisTime } from '../lib/fr';
import { paperSnapshot, radarHome } from './radarFixtures';

const home: RadarHome = {
  ...radarHome,
  regime: {
    ...radarHome.regime,
    QQQ: { symbol: 'QQQ', last: 600, at: '2026-10-09T20:00:00Z', status: 'OBSERVED', returns_pct: { '1d': -0.73 } },
    VIX: { symbol: '^VIX', last: 15, at: '2026-10-09T20:00:00Z', status: 'OBSERVED', returns_pct: { '1d': 2.49 } },
  },
  calendar: [
    { at: '2026-10-13T11:00:00Z', date: '2026-10-13', time_known: true, title: 'Résultats JPMorgan (T3)', kind: 'earnings',
      verification: 'OFFICIEL', assets: ['JPM'], origin: 'calendar' },
    { at: null, date: '2026-10-28', time_known: false, title: 'Décision de la Fed', kind: 'fed', verification: 'FIL',
      assets: [], origin: 'calendar' },
    { at: '2026-10-14T12:30:00Z', date: '2026-10-14', time_known: true, title: 'Inflation US (CPI)', kind: 'cpi',
      verification: 'OFFICIEL', assets: [], origin: 'calendar' },
    { at: null, date: '2026-11-03', time_known: false, title: 'Fourth event, not shown', kind: 'election',
      verification: 'FIL', assets: [], origin: 'calendar' },
  ],
  ai_today: {
    run_id: 'trader:2026-10-10:synthetic', at: '2026-10-10T12:00:00Z', session: '2026-10-10',
    lines: [
      { account: 'ia_actions', asset: 'XOM', horizon: '1d', slot: 'am', direction: 'UP', size: 'full', analyst: 'analyst_claude', verdict: 'KEEP', status: 'executed' },
      { account: 'ia_crypto', asset: 'BTC-USD', horizon: '1d', slot: 'am', direction: 'DOWN', size: 'half', analyst: 'analyst_gpt', verdict: 'KEEP', status: 'planned' },
      { account: 'ia_actions', asset: 'XLE', horizon: '1d', slot: 'am', direction: 'UP', size: 'half', analyst: 'analyst_claude', verdict: 'DOWNGRADE', status: 'expired' },
    ],
  },
  system: { level: 'orange', checked_at: '2026-10-10T12:00:00Z',
    problems: [{ level: 'orange', text: 'Le trader IA a signalé 1 type d’incident en 24 h.', detail: 'MISSING_ANCHOR_PRICE' }] },
};

function respond(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function mount(radar: unknown = home, paper: unknown = paperSnapshot, path = '/') {
  const fetchMock = vi.fn((input: string, init?: RequestInit) => {
    expect(init?.method ?? 'GET').toBe('GET');
    const path = String(input).split('?')[0] ?? '';
    if (path === '/api/v1/radar/home') return respond(radar);
    if (path === '/api/v1/radar/paper') return respond(paper);
    if (path === '/api/v1/health') return respond({ status: 'ok', api_version: 'v1', core_status: 'ready' });
    return respond({ error: 'not found' }, 404);
  });
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
  return fetchMock;
}

beforeEach(() => { invalidate(); window.localStorage.clear(); });
afterEach(() => vi.unstubAllGlobals());

describe('Aujourd’hui', () => {
  it('shows three paper accounts with today, since start, S&P 500 gap and positions', async () => {
    mount();
    const tiles = await screen.findByRole('region', { name: 'Comptes' });
    const crypto = await within(tiles).findByRole('article', { name: 'IA crypto' });
    expect(within(crypto).getByText(dollars(100500))).toBeInTheDocument();
    // 20 $ today on a 100 480 $ base; +0.50 % since start; 0.50 − 0.47 = +0.03 point over the S&P 500.
    expect(within(crypto).getByText(pctFr(20 / 100480))).toBeInTheDocument();
    expect(within(crypto).getByText('+0,50 %')).toBeInTheDocument();
    expect(within(crypto).getByText('+0,03 %')).toBeInTheDocument();
    const book = within(tiles).getByRole('article', { name: 'HypR-A1 book' });
    // Without a previous-close equity the book's day change is unknown, never 0.
    expect(within(book).getAllByText('—').length).toBeGreaterThan(0);
    expect(within(book).getByText('2')).toBeInTheDocument();
    expect(screen.getByText(/argent fictif/)).toBeInTheDocument();
  });

  it('lists the five markets with today’s change, the VIX rise coloured as bad news', async () => {
    mount();
    const strip = await screen.findByRole('list', { name: 'Marchés aujourd’hui' });
    expect(within(strip).getAllByRole('listitem').map((li) => li.firstChild?.textContent))
      .toEqual(['S&P 500', 'Nasdaq', 'Bitcoin', 'Pétrole', 'Peur (VIX)']);
    expect(within(strip).getByText('−0,73 %')).toHaveAttribute('data-tone', 'down');
    expect(within(strip).getByText('+2,49 %')).toHaveAttribute('data-tone', 'down');
  });

  it('links the system light with its problem count to Système', async () => {
    mount();
    const light = await screen.findByRole('link', { name: /État du système : À surveiller, 1 problème/ });
    expect(light).toHaveAttribute('href', '/system');
    expect(light).toHaveTextContent('1 problème');
  });

  it('shows the next three scheduled events in Paris time, and an unconfirmed time as such', async () => {
    mount();
    const agenda = await screen.findByRole('list', { name: 'Prochains rendez-vous' });
    const items = within(agenda).getAllByRole('listitem');
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent('Résultats JPMorgan (T3)');
    expect(items[0]).toHaveTextContent(parisTime('2026-10-13T11:00:00Z'));
    expect(parisTime('2026-10-13T11:00:00Z')).toBe('13:00');
    expect(items[1]).toHaveTextContent('heure à confirmer');
    expect(items[1]).toHaveTextContent('presse');
    expect(agenda).not.toHaveTextContent('Fourth event');
  });

  it('lists news on one line each and opens the chain and five scores in a drawer only', async () => {
    mount();
    const list = await screen.findByRole('list', { name: '5 news qui comptent' });
    const lines = within(list).getAllByRole('button');
    expect(lines.length).toBeLessThanOrEqual(5);
    expect(lines[0]).toHaveTextContent('82');
    expect(lines[0]).toHaveTextContent('Synthetic oil supply shock');
    expect(within(list).queryByText('Fait sourcé')).not.toBeInTheDocument();
    await userEvent.click(lines[0] as HTMLElement);
    const dialog = await screen.findByRole('dialog', { name: 'Détail de la news' });
    expect(within(dialog).getByLabelText(/five separate measures/i)).toBeInTheDocument();
    expect(within(dialog).getAllByRole('listitem').some((li) => /Fait sourcé/.test(li.textContent ?? ''))).toBe(true);
    for (const step of ['Attentes du marché', 'Scénario', 'Condition d’entrée', 'Invalidation', 'Résultat après coûts']) {
      expect(within(dialog).getAllByText(step).length).toBeGreaterThan(0);
    }
    await userEvent.click(within(dialog).getByRole('button', { name: 'Fermer le détail' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  it('says what the AIs decided with direction, size and status, and dates an older session', async () => {
    mount();
    const lines = await screen.findByRole('list', { name: 'Décisions des IA' });
    const rows = within(lines).getAllByRole('listitem');
    expect(rows[0]).toHaveTextContent('XOM');
    expect(rows[0]).toHaveTextContent('hausse');
    expect(rows[0]).toHaveTextContent('taille normale');
    expect(rows[0]).toHaveTextContent('exécuté');
    expect(rows[1]).toHaveTextContent('demi-taille');
    expect(rows[1]).toHaveTextContent('prévu');
    expect(rows[2]).toHaveTextContent('expiré');
    expect(screen.getByRole('heading', { name: /Ce que font les IA \(décisions du/ })).toBeInTheDocument();
  });

  it('stays honest with an older snapshot that has no home blocks', async () => {
    mount({ ...radarHome });
    expect(await screen.findByText('Aucun rendez-vous daté dans le calendrier.')).toBeInTheDocument();
    expect(screen.getByText('Aucune décision des IA enregistrée.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /État du système : inconnu/ })).toHaveTextContent('État non vérifié');
  });

  it('never shows hashes or run identifiers to a beginner', async () => {
    mount();
    await screen.findByRole('list', { name: '5 news qui comptent' });
    const text = document.body.textContent ?? '';
    expect(text).not.toMatch(/[0-9a-f]{16,}/);
    expect(text).not.toContain('trader:2026');
  });
});

describe('Système', () => {
  it('shows the light and plain problems first, codes only in Expert mode', async () => {
    mount(home, paperSnapshot, '/system');
    const problems = await screen.findByRole('region', { name: 'Problèmes' });
    expect(within(problems).getByText('À surveiller.')).toBeInTheDocument();
    expect(problems).toHaveTextContent('1 type d’incident');
    expect(problems).not.toHaveTextContent('MISSING_ANCHOR_PRICE');
    expect(screen.getAllByRole('tab').map((t) => t.textContent)).toEqual(['Tâches', 'Sources', 'Application']);
  });

  it('adds the codes behind each problem in Expert mode', async () => {
    mount(home, paperSnapshot, '/system?mode=expert');
    const problems = await screen.findByRole('region', { name: 'Problèmes' });
    await waitFor(() => expect(problems).toHaveTextContent('MISSING_ANCHOR_PRICE'));
  });
});
