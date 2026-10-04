/** Timeline values and order are shipped by the API; missing sources remain explicit. */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import type { TimelineView } from '../api/types';
import { invalidate } from '../state/useQuery';

const T = '2026-06-17T19:01:35.400000+00:00';
const SID = '1'.repeat(64);
const ACC = '0000320193-26-000071';
const IDENTITY = '3'.repeat(64);
const FOMC_AVAILABLE = '2026-06-17T18:07:05.100000+00:00';
const EDGAR_AVAILABLE = '2026-06-17T18:20:32.800000+00:00';
const RELEASE = '2026-06-17T18:00:00+00:00';
const ACCEPTANCE = '2026-06-16T16:31:05.000Z';
const timeline: TimelineView = {
  api_version: 'trading-lab.app-api.v1', read_only: true, T, identity: IDENTITY,
  sources: {
    fomc: { read_state: 'FOMC_RESOLVED', identity: '4'.repeat(64), H: 300, P: 250 },
    edgar: { read_state: 'EDGAR_RESOLVED', identity: '5'.repeat(64), H: 38, P: 30 },
  },
  rows: [
    { source: 'fomc', id: SID, title: 'Federal Reserve issues FOMC statement', form: null,
      state: 'CURRENT_REVISION', revision: `${SID}:${'6'.repeat(64)}`, content_identity: '6'.repeat(64),
      available_at: FOMC_AVAILABLE,
      provenance: { declared_release_at: RELEASE, declared_release_text: 'For release at 2:00 p.m. EDT' } },
    { source: 'edgar', id: ACC, title: null, form: '8-K', state: 'ABSENT_FROM_LISTING',
      revision: `synthetic:${'7'.repeat(64)}`, content_identity: '7'.repeat(64), available_at: EDGAR_AVAILABLE,
      provenance: { acceptance_datetime_text: ACCEPTANCE } },
  ],
};

function respond(payload: unknown) {
  return Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

async function openTimeline(view = timeline, configured = true) {
  const fetch = vi.fn((input: string) => {
    const path = String(input).split('?')[0];
    if (path === '/api/v1/health') return respond({ status: 'ok', api_version: 'v1', core_status: 'ready' });
    if (path === '/api/v1/events/timeline') return respond(view);
    if (path === '/api/v1/sources/fomc/snapshot') return respond({
      snapshot: { read_state: 'FOMC_RESOLVED', identity: '4'.repeat(64), T, H: 300, P: 250 },
      items: [], health: null, discovery: null,
    });
    if (path === '/api/v1/sources/fomc' || path === '/api/v1/sources/edgar') {
      return respond({ status: configured ? 'AVAILABLE' : 'NOT_CONFIGURED', suggested_as_of: configured ? T : null });
    }
    return respond({ error: 'no such synthetic endpoint' });
  });
  vi.stubGlobal('fetch', fetch);
  render(<MemoryRouter initialEntries={['/events']}><App /></MemoryRouter>);
  await userEvent.click(await screen.findByRole('tab', { name: 'Timeline' }));
  return fetch;
}

beforeEach(() => invalidate());
afterEach(() => vi.unstubAllGlobals());

describe('official events timeline', () => {
  it('shows values verbatim in backend order with explicit provenance labels', async () => {
    const fetch = await openTimeline();
    expect(await screen.findByTestId('timeline-identity')).toHaveTextContent(IDENTITY);
    const panel = screen.getByRole('region', { name: 'Timeline' });
    const rows = within(panel).getAllByRole('row').slice(1);
    expect(within(rows[0]!).getByText('fomc')).toBeInTheDocument();
    expect(within(rows[1]!).getByText('edgar')).toBeInTheDocument(); // earlier acceptance never reorders
    for (const value of [SID, ACC, FOMC_AVAILABLE, EDGAR_AVAILABLE, RELEASE, ACCEPTANCE,
      'For release at 2:00 p.m. EDT', 'CURRENT_REVISION', 'ABSENT_FROM_LISTING', '8-K',
      timeline.rows[0]!.revision, timeline.rows[1]!.content_identity]) {
      expect(within(panel).getByText(value)).toBeInTheDocument();
    }
    for (const name of ['Declared release at (provenance)', 'Declared release text (provenance)', 'Acceptance (provenance)']) {
      expect(within(panel).getByRole('columnheader', { name })).toBeInTheDocument();
    }
    expect(screen.getByRole('status', { name: 'fomc read state' })).toHaveTextContent('FOMC_RESOLVED');
    expect(screen.getByRole('status', { name: 'edgar read state' })).toHaveTextContent('EDGAR_RESOLVED');
    expect(fetch.mock.calls.map(([path]) => path)).toContain(`/api/v1/events/timeline?as_of=${encodeURIComponent(T)}`);
  });

  it('shows an unresolved source banner alongside the resolved source rows', async () => {
    await openTimeline({ ...timeline,
      sources: { ...timeline.sources, fomc: { ...timeline.sources.fomc, read_state: 'FOMC_CAUSAL_VISIBILITY_UNRESOLVED', P: null } },
      rows: [timeline.rows[1]!],
    });
    await screen.findByTestId('timeline-identity');
    expect(screen.getByRole('status', { name: 'fomc read state' })).toHaveTextContent('FOMC_CAUSAL_VISIBILITY_UNRESOLVED');
    expect(screen.getByText('Nothing is confirmed for this source at this read.')).toBeInTheDocument();
    expect(screen.getByText(/Partial timeline/)).toBeInTheDocument();
    const panel = screen.getByRole('region', { name: 'Timeline' });
    expect(within(panel).getByText(ACC)).toBeInTheDocument();
    expect(within(panel).queryByText(SID)).not.toBeInTheDocument();
  });

  it('sends separate horizons and a chosen instant, validating before making a request', async () => {
    const fetch = await openTimeline();
    await screen.findByTestId('timeline-identity');
    const before = fetch.mock.calls.length;
    await userEvent.type(screen.getByLabelText('EDGAR horizon'), '12x');
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect(screen.getByRole('alert')).toHaveTextContent('whole number');
    expect(fetch.mock.calls.length).toBe(before);
    await userEvent.clear(screen.getByLabelText('EDGAR horizon'));
    await userEvent.type(screen.getByLabelText('EDGAR horizon'), '0');
    await userEvent.type(screen.getByLabelText('FOMC horizon'), '200');
    const chosen = '2026-06-17T18:30:00Z';
    await userEvent.clear(screen.getByLabelText('As of'));
    await userEvent.type(screen.getByLabelText('As of'), chosen);
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    await waitFor(() => expect(fetch.mock.calls.map(([path]) => path)).toContain(
      `/api/v1/events/timeline?as_of=${encodeURIComponent(chosen)}&fomc_horizon=200&edgar_horizon=0`));
  });

  it('keeps unconfigured and refused sources visible without claiming an empty confirmed timeline', async () => {
    const empty: TimelineView = { ...timeline, rows: [], sources: {
      fomc: { read_state: 'NOT_CONFIGURED', identity: null, H: null, P: null },
      edgar: { read_state: 'REJECTED', identity: null, H: null, P: null, reason: 'Synthetic incompatible store' },
    } };
    await openTimeline(empty, false);
    await userEvent.type(await screen.findByLabelText('As of'), T);
    await userEvent.click(screen.getByRole('button', { name: 'Read' }));
    await screen.findByTestId('timeline-identity');
    expect(screen.getByRole('status', { name: 'fomc read state' })).toHaveTextContent('NOT_CONFIGURED');
    expect(screen.getByRole('status', { name: 'edgar read state' })).toHaveTextContent('REJECTED');
    expect(screen.getByText('Synthetic incompatible store')).toBeInTheDocument();
    expect(screen.getByText(/unavailable sources are not confirmed empty/)).toBeInTheDocument();
    expect(screen.queryByText('No timeline event was known at this instant.')).not.toBeInTheDocument();
  });
});
