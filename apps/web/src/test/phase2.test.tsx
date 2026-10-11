import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { DecisionCard } from '../components/DecisionCard';
import { MarketDecisionPanel } from '../pages/MarketDecisionPanel';
import { RadarHealthPanel } from '../pages/RadarHealthPanel';
import { marketSelection } from '../lib/marketOverlays';
import { verification } from '../lib/decisions';
import { invalidate } from '../state/useQuery';
import { analysisSnapshot, feedDecision } from './phase2Fixtures';

beforeEach(() => {
  invalidate();
  vi.stubGlobal('fetch', vi.fn((input: string) => {
    const path = String(input).split('?')[0];
    const payload = path === '/api/v1/radar/analysis' ? analysisSnapshot : { error: 'unavailable' };
    const status = path === '/api/v1/radar/analysis' ? 200 : 503;
    return Promise.resolve({ ok: status === 200, status, text: async () => JSON.stringify(payload) } as Response);
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe('decision provenance', () => {
  it('shows the six steps and explicitly calls a feed headline unconfirmed', () => {
    render(<DecisionCard decision={feedDecision} />);
    for (const label of ['Fait sourcé', 'Attentes du marché', 'Scénario', 'Condition d’entrée', 'Invalidation', 'Résultat après coûts']) {
      expect(screen.getByRole('heading', { name: new RegExp(label) })).toBeInTheDocument();
    }
    expect(screen.getByText(/Présent dans le flux; aucune confirmation primaire/)).toBeInTheDocument();
    expect(screen.getByText(/2026-10-09 18:00/)).toBeInTheDocument();
    expect(screen.getByText('Margin below 10%')).toBeInTheDocument();
    expect(screen.getByText(/résultat réalisé après coûts non connu/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Examplewire' })).toHaveAttribute('rel', 'noreferrer noopener');
  });
  it('shows absent consensus and invalidation without inventing values', () => {
    render(<DecisionCard />);
    expect(screen.getByText('NON VERIFIE')).toBeInTheDocument();
    expect(screen.getByText('Consensus daté non consigné.')).toBeInTheDocument();
    expect(screen.getByText(/ferait abandonner le modèle n’est pas consigné/)).toBeInTheDocument();
  });
  it('identifies primary domains without trusting a publisher or lookalike hostname', () => {
    expect(verification('https://www.sec.gov/Archives/a.htm')).toBe('OFFICIEL');
    expect(verification('https://sec.gov.evil.invalid/a')).toBe('FIL');
    expect(verification('https://www.sec.gov/Archives/a.htm', false)).toBe('FIL');
    expect(verification('javascript:alert(1)')).toBe('NON VERIFIE');
  });
});

describe('markets overlays', () => {
  it('joins symbol aliases while filtering each AI horizon and keeping event timestamps', () => {
    const selected = marketSelection(analysisSnapshot, 'AAAUSD', '1d', 7, ['event', 'decision', 'outcome']);
    expect(selected.prices.map((p) => p.price)).toEqual([100, 102]);
    expect(selected.overlays.map((o) => o.id)).toEqual(['e', 'd1', 'net']);
    expect(selected.overlays[0]?.at).toBe('2026-10-10T10:05:00Z');
  });
  it('selects a marker by keyboard and lets the reader hide events or change horizon', async () => {
    render(<MarketDecisionPanel />);
    const chart = await screen.findByRole('group', { name: /prix et événements/ });
    const event = within(chart).getByRole('button', { name: /Événements radar/ });
    event.focus(); await userEvent.keyboard('{Enter}');
    expect(event).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('heading', { name: /Événements radar/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('checkbox', { name: 'Événements radar' }));
    expect(within(chart).queryByRole('button', { name: /Événements radar/ })).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Horizon' }), '5d');
    await waitFor(() => expect(screen.getByText('Five-session falsifier')).toBeInTheDocument());
    expect(within(chart).getByRole('button', { name: /REJECT/ })).toBeInTheDocument();
    expect(within(chart).queryByRole('button', { name: /Net result/ })).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Analyste' }), 'analyst_claude');
    expect(within(chart).queryByRole('button', { name: /REJECT/ })).not.toBeInTheDocument();
  });
});

it('shows net sample sizes, small-sample caution, seeded random baseline and missing ablation', async () => {
  render(<MemoryRouter initialEntries={['/news-value']}><App /></MemoryRouter>);
  await screen.findByRole('heading', { name: 'Est-ce que l’actualité aide ?' });
  const book = screen.getByRole('table', { name: 'Actualité versus technique après coûts' });
  expect(within(book).getAllByText('Trop tôt')).toHaveLength(2);
  const scores = screen.getByRole('table', { name: 'Scorecards et baselines avec effectifs' });
  const random = within(scores).getAllByRole('row').find((r) => within(r).queryByText('random'))!;
  expect(within(random).getByText('+0.400 %')).toBeInTheDocument();
  expect(screen.getByText('Trop tôt : comparaison appariée indisponible.')).toBeInTheDocument();
});

it('renders radar health independently when system research data is unavailable', async () => {
  render(<MemoryRouter initialEntries={['/system']}><App /></MemoryRouter>);
  await screen.findByRole('table', { name: 'Sources radar' });
  expect(screen.getByText('DAILY_BUDGET')).toBeInTheDocument();
  expect(screen.getByText(/OWNER_BUSY/)).toBeInTheDocument();
  expect(screen.getByText('claude-book-open.timer')).toBeInTheDocument();
  expect(screen.getAllByText('Périmé / inconnu').length).toBeGreaterThan(0);
});

it('explains that a missing phase 2 snapshot must be exported', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503, text: async () => '{"error":"unavailable"}' })));
  render(<RadarHealthPanel />);
  expect(await screen.findByText('Snapshot phase 2 indisponible')).toBeInTheDocument();
});
