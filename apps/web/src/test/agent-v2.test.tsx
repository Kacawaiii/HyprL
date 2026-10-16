import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AgentV2Page } from '../pages/AgentV2Page';
import { invalidate } from '../state/useQuery';

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());
const layer = { closed: 0, wins: 0, pnl: 0, expectancy_r: null, cluster_days: 0, evidence: 'UNPROVEN' };

it('shows missing quote evidence without inventing performance and only uses GET', async () => {
  vi.stubGlobal('fetch', vi.fn((_url: string, init?: RequestInit) => {
    expect(init?.method ?? 'GET').toBe('GET');
    return Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({
      schema: 'agent-v2-summary-v1', mode: 'SYNTHETIC', state: 'NO_QUOTES', positions: [], pending: 0,
      layers: { core: layer, news: layer }, supervisor_vetoes: { count: 0, codes: {}, counterfactual: 'UNMEASURED' },
      last_mark: null, last_blocked: null, events: 1, code_sha: 'a'.repeat(40), preregistration_hash: 'b'.repeat(64),
    })) } as Response);
  }));
  render(<MemoryRouter><AgentV2Page /></MemoryRouter>);
  expect(await screen.findByText(/Aucune quote reçue/)).toBeInTheDocument();
  expect(screen.getByText(/Données synthétiques/)).toBeInTheDocument();
  expect(screen.getAllByText('Pas de résultat')).toHaveLength(2);
  expect(screen.getByText(/Effet sur les résultats : non mesuré/)).toBeInTheDocument();
  expect(screen.getByText(/Aucun ordre courtier/)).toBeInTheDocument();
});

it('shows unavailable runtime as an error with a retry', async () => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 503,
    text: () => Promise.resolve('{"error":"v2 shadow runtime not configured"}') } as Response)));
  render(<MemoryRouter><AgentV2Page /></MemoryRouter>);
  expect(await screen.findByText(/v2 shadow runtime not configured/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Retry|Réessayer/ })).toBeInTheDocument();
});
