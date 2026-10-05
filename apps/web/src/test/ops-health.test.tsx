/** The operations health panel on System. It must show what the server observed and nothing more:
 *  a missing figure is "not observed", degraded is not hidden, and the panel only ever reads. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import * as fixtures from './fixtures';
import { degradedHealth, opsHealth } from './opsFixtures';

function reply(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function open(health: unknown, status = 200) {
  const fetchMock = vi.fn((input: string, init?: RequestInit) => {
    void init;
    const [path] = String(input).split('?');
    if (path === '/api/v1/ops/health') return reply(status === 200 ? health : { error: 'not found' }, status);
    if (path === '/api/v1/system') return reply(fixtures.system);
    if (path === '/api/v1/health') return reply({ core_status: 'ready' });
    return reply({ error: 'no such endpoint' }, 404);
  });
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter initialEntries={['/system']}><App /></MemoryRouter>);
  return fetchMock;
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('operations health panel', () => {
  it('shows versions, operations, freshness, budgets, workers and resources', async () => {
    const fetchMock = open(opsHealth);
    const panel = await screen.findByRole('region', { name: 'Operations health' });
    expect(await within(panel).findByTestId('ops-status')).toHaveTextContent('OBSERVED');
    expect(within(panel).getByText('API')).toBeInTheDocument();
    expect(within(panel).getAllByText('fomc rev 25', { exact: false }).length).toBeGreaterThan(1);
    const operations = within(panel).getByRole('table', { name: 'Last operations' });
    expect(within(operations).getByText('BACKUP_TARGET_EXISTS')).toBeInTheDocument();
    expect(within(operations).getByText('BLOCKED')).toBeInTheDocument();
    expect(within(panel).getByText(/archive attested 2h 0m ago/)).toBeInTheDocument();
    expect(within(panel).getByText('NONE REPORTED')).toBeInTheDocument();
    expect(within(panel).getByText('4 of 1000')).toBeInTheDocument();
    expect(within(panel).getByText('1 of 8')).toBeInTheDocument();
    expect(within(panel).getByText('2.0 MiB of 128 MiB')).toBeInTheDocument();
    expect(within(panel).getByText('40%')).toBeInTheDocument();
    expect(within(panel).getByText('20 GiB')).toBeInTheDocument();
    expect(within(panel).getByText(/not a guarantee of a live feed/)).toBeInTheDocument();
    const calls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/ops/health'));
    expect(calls.every(([, init]) => !init?.method || init.method === 'GET')).toBe(true);
  });

  it('keeps an unconfigured source and a missing EDGAR budget as not observed', async () => {
    open(opsHealth);
    const panel = await screen.findByRole('region', { name: 'Operations health' });
    expect(await within(panel).findByText('NOT_CONFIGURED')).toBeInTheDocument();
    expect(within(panel).getByText('EDGAR budget').nextSibling).toHaveTextContent('not observed');
  });

  it('shows degradation, error codes, an expired grant and unknown resources', async () => {
    open(degradedHealth);
    const panel = await screen.findByRole('region', { name: 'Operations health' });
    expect(await within(panel).findByTestId('ops-status')).toHaveTextContent('DEGRADED');
    const errors = within(panel).getByRole('list', { name: 'Reported errors' });
    expect(within(errors).getByText('EDGAR_STATUS_STALE')).toBeInTheDocument();
    expect(within(errors).getByText('FOMC_SOURCE_UNAVAILABLE')).toBeInTheDocument();
    expect(within(panel).getByText('GRANT EXPIRED')).toBeInTheDocument();
    expect(within(panel).getByText('TERMINATED')).toBeInTheDocument();
    expect(within(panel).getByText('Job budgets').nextSibling).toHaveTextContent('not observed');
    expect(within(panel).getByText('API memory').nextSibling).toHaveTextContent('—');
    expect(within(panel).queryByText('NONE REPORTED')).not.toBeInTheDocument();
  });

  it('says so when the server does not publish operations health, and keeps the page', async () => {
    open(null, 404);
    expect(await screen.findByText(/does not publish operations health/)).toBeInTheDocument();
    expect(await screen.findByText('Signal engine')).toBeInTheDocument();
  });
});
