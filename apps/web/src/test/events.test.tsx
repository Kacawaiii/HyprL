/**
 * The FOMC journey in the cockpit: a read at an instant, its identity, source health, items, an
 * item's revisions and provenance, and the verified replay. The page shows backend values verbatim
 * and never decides what was available.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import type { FomcItemDetail, FomcReplay, FomcSnapshotView, FomcSourceStatus } from '../api/types';

const SPEC = 'b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece';
const SID = '1f23e18c6920acad765737d4a21f44d07396a3b757d6bf7b1266c8c6aa433448';
const CONTENT = 'ff180415cfa9f351e0e0fed4d2126947eea0af5c07f9be1bc22d1dbbab81d71f';
const IDENTITY = 'd9a2b3ccb4dea00fcd68706a4489ed27324d46b06c6e8149a5d1b959e764b5fa';
const AS_OF = '2026-10-02T14:48:07.319660+00:00';
const base = {
  api_version: 'trading-lab.app-api.v1', source: 'fomc' as const, provider_id: 'federal_reserve_fomc_statements_v1',
  spec_revision: 25, spec_hash: SPEC, schema_version: 'fomc-store-v5', read_only: true as const,
};
const status: FomcSourceStatus = {
  ...base, status: 'AVAILABLE', store: '/archive/closure-copy', horizon: 622,
  first_durable_activity: '2026-10-02T13:20:05.978184+00:00', last_durable_activity: '2026-10-02T14:49:40.004237+00:00',
  suggested_as_of: AS_OF, counts: { epochs: 1, responses: 128, revisions: 6, observations: 15, cycles: 80 },
};
const header = {
  policy: 'FOMC_CURRENT_CONTENT_SELECTION_V1', spec_hash: SPEC, mode: 'DURABLE_OBSERVED', T: AS_OF, H: 622, P: 606,
  read_state: 'FOMC_RESOLVED', identity: IDENTITY,
};
const snapshot: FomcSnapshotView = {
  ...base, snapshot: header, discovery: { state: 'EVENTS_OBSERVED_ZERO', cycle_id: 600, B: 600 },
  health: {
    discovery_feed: { result_state: null, reason: 'NO_FAILURE', check: 602, check_at: '2026-10-02T14:47:46+00:00', outcome: 'FEED_CLASSIFIED' },
    primary_statement: { result_state: 'SOURCE_UNAVAILABLE', reason: 'DNS', check: 590, check_at: '2026-10-02T14:40:12+00:00', outcome: null },
  },
  items: [{
    sid: SID, state: 'CURRENT_REVISION', step: 4, revision: `${SID}:${CONTENT}`, content_hash: CONTENT, live_available: true,
    title: 'Federal Reserve issues FOMC statement', official_statement_date: '2026-06-17',
    declared_release_at: '2026-06-17T18:00:00+00:00', declared_release_text: 'For release at 2:00 p.m. EDT',
    declared_release_trust_verdict: 'TRUSTED_EXACT', observation_mode: 'HISTORICAL_BACKFILL',
    canonical_source_url: 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm',
    content_domain: 'CANONICAL', observations: 4,
  }],
};
const detail: FomcItemDetail = {
  ...base, snapshot: header, item: { sid: SID, state: 'CURRENT_REVISION' },
  revisions: [{
    revision_id: `${SID}:${CONTENT}`, committed_seq: 12, content_hash: CONTENT,
    first_raw_sha256: 'cdd75b6049e46b817412a95144057ce46a4d903920842b6321e603525e897d23', observation_mode: 'HISTORICAL_BACKFILL',
    content_identity: { identity: 'FOMC_CONTENT_IDENTITY_V2', canonicalizer: 'FOMC_CANON_V2', domain: 'CANONICAL', bytes_sha256: '0d4f' },
  }],
  observations: [{
    record: 10, attempt: 7, mode: 'HISTORICAL_BACKFILL', verdict: 'CLOCK_VERIFIED', status: 200,
    observed_at: '2026-10-02T13:21:35.651429+00:00', wall_at_receipt: '2026-10-02T13:21:35.651429+00:00',
    request_url: 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm',
    final_url: 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm',
    redirect_chain: ['https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm'],
    raw_sha256: 'cdd75b6049e46b817412a95144057ce46a4d903920842b6321e603525e897d23', byte_length: 81083,
    processing_outcome: 'NORMALIZED_REVISION_COMMITTED', revision: `${SID}:${CONTENT}`,
  }],
};
const replay: FomcReplay = { ...base, snapshot: header, replay_identity: IDENTITY, identical: true, error: null };

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve({
    ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)),
  } as Response);
}

function mockApi(overrides: Record<string, unknown> = {}) {
  const routes: Record<string, unknown> = {
    '/api/v1/health': { status: 'ok', api_version: 'v1', core_status: 'ready' },
    '/api/v1/sources/fomc': status,
    '/api/v1/sources/fomc/snapshot': snapshot,
    '/api/v1/sources/fomc/replay': replay,
    ...overrides,
  };
  return vi.fn((input: string) => {
    const path = String(input);
    if (path.startsWith('/api/v1/sources/fomc/items/')) return jsonResponse(routes.item ?? detail);
    const bare = path.split('?')[0] ?? path;
    return bare in routes ? jsonResponse(routes[bare]) : jsonResponse({ error: 'no such endpoint' }, 400);
  });
}

function renderEvents(overrides?: Record<string, unknown>) {
  const fetch = mockApi(overrides);
  vi.stubGlobal('fetch', fetch);
  render(<MemoryRouter initialEntries={['/events']}><App /></MemoryRouter>);
  return fetch;
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('FOMC events journey', () => {
  it('reads the store at the server-attested instant and shows the snapshot verbatim', async () => {
    const fetch = renderEvents();
    expect(await screen.findByTestId('snapshot-identity')).toHaveTextContent(IDENTITY);
    expect(screen.getByText('FOMC_RESOLVED')).toBeInTheDocument();
    expect(screen.getByText('622 · 606')).toBeInTheDocument();
    expect(screen.getByText('EVENTS_OBSERVED_ZERO')).toBeInTheDocument();
    const health = screen.getByRole('region', { name: 'Source health' });
    expect(within(health).getByText('SOURCE_UNAVAILABLE')).toBeInTheDocument();  // a failure state shown as shipped
    const items = screen.getByRole('region', { name: 'Items' });
    expect(within(items).getByText('For release at 2:00 p.m. EDT')).toBeInTheDocument();
    expect(within(items).getByText('CANONICAL')).toBeInTheDocument();
    const read = fetch.mock.calls.map(([path]) => String(path)).find((path) => path.includes('/snapshot'));
    expect(read).toBe(`/api/v1/sources/fomc/snapshot?as_of=${encodeURIComponent(AS_OF)}`);
  });

  it('opens an item with its revisions and observation provenance', async () => {
    renderEvents();
    await userEvent.click(await screen.findByRole('button', { name: `Open item ${SID.slice(0, 12)}` }));
    const panel = await screen.findByRole('region', { name: 'Item detail' });
    expect(within(panel).getByText('NORMALIZED_REVISION_COMMITTED')).toBeInTheDocument();
    expect(within(panel).getByText('81083')).toBeInTheDocument();
    expect(within(panel).getByText('CLOCK_VERIFIED')).toBeInTheDocument();
    expect(within(panel).getAllByText('HISTORICAL_BACKFILL').length).toBe(2);  // the revision and its observation
  });

  it('verifies the offline replay on demand', async () => {
    const fetch = renderEvents();
    await userEvent.click(await screen.findByRole('button', { name: 'Verify replay' }));
    expect(await screen.findByText('Replay identical')).toBeInTheDocument();
    expect(fetch.mock.calls.some(([path]) => String(path).startsWith('/api/v1/sources/fomc/replay?as_of='))).toBe(true);
  });

  it('sends a chosen read and refuses a malformed horizon without a request', async () => {
    const fetch = renderEvents();
    await screen.findByTestId('snapshot-identity');
    const horizon = screen.getByLabelText('Horizon');
    await userEvent.type(horizon, '12x');
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('whole number');
    const before = fetch.mock.calls.length;
    await userEvent.clear(horizon);
    await userEvent.type(horizon, '300');
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    await waitFor(() => expect(fetch.mock.calls.length).toBeGreaterThan(before));
    expect(fetch.mock.calls.map(([path]) => String(path))).toContain(
      `/api/v1/sources/fomc/snapshot?as_of=${encodeURIComponent(AS_OF)}&horizon=300`);
  });

  it('shows an unresolved read as nothing known, not a guess', async () => {
    renderEvents({
      '/api/v1/sources/fomc/snapshot': {
        ...snapshot, snapshot: { ...header, read_state: 'FOMC_CAUSAL_VISIBILITY_UNRESOLVED', P: 600 },
        discovery: null, health: null, items: [],
      },
    });
    expect(await screen.findByText('FOMC_CAUSAL_VISIBILITY_UNRESOLVED')).toBeInTheDocument();
    expect(screen.getByText(/nothing is shown as of this instant/)).toBeInTheDocument();
  });

  it('says plainly when no store is configured', async () => {
    renderEvents({ '/api/v1/sources/fomc': { ...base, status: 'NOT_CONFIGURED', store: null } });
    expect(await screen.findByText('No FOMC store configured')).toBeInTheDocument();
  });
});
