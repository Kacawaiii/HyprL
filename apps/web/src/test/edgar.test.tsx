/**
 * The SEC EDGAR journey in the cockpit's Events page: switch source, read at an instant, the snapshot
 * identity, health per CIK, filings with their server-attested availability and acceptanceDateTime shown
 * as provenance only, a filing's revisions and absences, and the replay check.
 */
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';

const ACC = '0000320193-26-000071';
const AS_OF = '2026-06-17T15:12:33.800000+00:00';
const IDENTITY = 'cd655512de4269c0aa11bb22cc33dd44ee55ff6677889900aabbccddeeff0011';
const base = {
  api_version: 'trading-lab.app-api.v1', source: 'edgar', provider_id: 'sec_edgar_submissions_v1', spec_revision: 1,
  spec_hash: '98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce', schema_version: 'edgar-store-v1', read_only: true,
};
const header = {
  policy: 'EDGAR_FILINGS_AS_OF_V1', spec_hash: base.spec_hash, mode: 'DURABLE_OBSERVED', T: AS_OF, H: 38, P: 30,
  read_state: 'EDGAR_RESOLVED', identity: IDENTITY,
};
const routes: Record<string, unknown> = {
  '/api/v1/health': { status: 'ok', api_version: 'v1', core_status: 'ready' },
  '/api/v1/sources/fomc': { ...base, source: 'fomc', status: 'NOT_CONFIGURED', store: null },
  '/api/v1/sources/edgar': {
    ...base, status: 'AVAILABLE', store: '/stores/edgar', horizon: 38, suggested_as_of: AS_OF, watchlist: ['0000320193'],
    counts: { epochs: 1, responses: 12, revisions: 4, observations: 30, absences: 1 },
  },
  '/api/v1/sources/edgar/snapshot': {
    ...base, snapshot: header, watchlist: ['0000320193'],
    health: { '0000320193': { result_state: null, reason: 'NO_FAILURE', check_at: '2026-06-17T15:10:00+00:00', attempt: 35, record: 36 } },
    filings: [{
      accession_number: ACC, cik: '0000320193', form: '8-K', filing_date: '2026-06-16', report_date: '2026-06-16',
      items: '2.02,7.01,9.01', state: 'ABSENT_FROM_LISTING', revisions_seen: 2, observations: 6,
      first_available_at: '2026-06-17T13:12:32.800000+00:00', first_observed_at: '2026-06-17T13:01:00.400000+00:00',
      amendment_link: null, acceptance_datetime_text: '2026-06-16T16:31:05.000Z', entity_name: 'Example Corp',
    }],
  },
  '/api/v1/sources/edgar/replay': { ...base, snapshot: header, replay_identity: IDENTITY, identical: true, error: null },
};
const detail = {
  ...base, snapshot: header, filing: { accession_number: ACC, state: 'ABSENT_FROM_LISTING' },
  revisions: [
    { revision: 'a:111', committed_seq: 5, content_sha256: '1'.repeat(64), first_record: 4, fields: { items: '2.02,9.01' } },
    { revision: 'a:222', committed_seq: 13, content_sha256: '2'.repeat(64), first_record: 12, fields: { items: '2.02,7.01,9.01' } },
  ],
  observations: [{ record: 4, observed_at: '2026-06-17T13:01:00.400000+00:00', raw_sha256: '3'.repeat(64), position: 0, revision: 'a:111', entity_name: 'Example Corp' }],
  absences: [{ record: 28, observed_at: '2026-06-17T14:01:00+00:00', raw_sha256: '4'.repeat(64), filing_date: '2026-06-16', listing_oldest_filing_date: '2026-05-01' }],
};

function respond(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

beforeEach(() => {
  invalidate();
  vi.stubGlobal('fetch', vi.fn((input: string) => {
    const path = String(input);
    if (path.startsWith('/api/v1/sources/edgar/filings/')) return respond(detail);
    const bare = path.split('?')[0] ?? path;
    return bare in routes ? respond(routes[bare]) : respond({ error: 'no such endpoint' }, 400);
  }));
});
afterEach(() => vi.unstubAllGlobals());

async function openEdgar() {
  render(<MemoryRouter initialEntries={['/events']}><App /></MemoryRouter>);
  await userEvent.click(await screen.findByRole('tab', { name: 'SEC EDGAR filings' }));
}

describe('SEC EDGAR journey', () => {
  it('reads filings at the server-attested instant with acceptance time as provenance only', async () => {
    await openEdgar();
    expect(await screen.findByTestId('edgar-identity')).toHaveTextContent(IDENTITY);
    const filings = screen.getByRole('region', { name: 'Filings' });
    expect(within(filings).getByText('ABSENT_FROM_LISTING')).toBeInTheDocument();
    expect(within(filings).getByText('2026-06-17T13:12:32.800000+00:00')).toBeInTheDocument();  // availability
    expect(within(filings).getByText('2026-06-16T16:31:05.000Z')).toBeInTheDocument();  // provenance column
    expect(within(filings).getByRole('columnheader', { name: 'Acceptance (provenance)' })).toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'EDGAR source health' })).getByText('NO_FAILURE')).toBeInTheDocument();
  });

  it('opens a filing with its revisions and its absence', async () => {
    await openEdgar();
    await userEvent.click(await screen.findByRole('button', { name: `Open filing ${ACC}` }));
    const panel = await screen.findByRole('region', { name: 'Filing detail' });
    expect(within(panel).getByText('2.02,9.01')).toBeInTheDocument();
    expect(within(panel).getByText('2.02,7.01,9.01')).toBeInTheDocument();
    expect(within(panel).getByText('2026-05-01')).toBeInTheDocument();
  });

  it('verifies the replay and keeps FOMC as the default source', async () => {
    render(<MemoryRouter initialEntries={['/events']}><App /></MemoryRouter>);
    expect(await screen.findByText('No FOMC store configured')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'SEC EDGAR filings' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Verify replay' }));
    expect(await screen.findByText('Replay identical')).toBeInTheDocument();
  });

  it('hides the previous filings while a different instant is loading', async () => {
    await openEdgar();
    await screen.findByTestId('edgar-identity');
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));
    const asOf = screen.getByLabelText('As of');
    await userEvent.clear(asOf);
    await userEvent.type(asOf, '2026-01-01T00:00:00+00:00');
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect(screen.getByText(/Reading the store/)).toBeInTheDocument();
    expect(screen.queryByTestId('edgar-identity')).not.toBeInTheDocument();
    expect(screen.queryByText(ACC)).not.toBeInTheDocument();
  });
});
