/**
 * The main path end to end through the real router and sidebar, on real-shaped responses:
 * snapshot → dataset → model → experiment → prediction → monitoring, then health and API docs.
 * A single mounted app: selection, token and navigation state must survive every hop. The whole
 * journey is read-only; no request other than GET is ever sent.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { App } from '../App';
import { clearLabToken } from '../state/labToken';
import { invalidate } from '../state/useQuery';
import { lab } from './labFixtures';
import { opsHealth } from './opsFixtures';
import * as fixtures from './fixtures';

const TOKEN = 't'.repeat(40);
const SPEC = 'b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece';
const IDENTITY = 'd9a2b3ccb4dea00fcd68706a4489ed27324d46b06c6e8149a5d1b959e764b5fa';
const AS_OF = '2026-10-02T14:48:07.319660+00:00';
const base = {
  api_version: 'trading-lab.app-api.v1', source: 'fomc', provider_id: 'federal_reserve_fomc_statements_v1',
  spec_revision: 25, spec_hash: SPEC, schema_version: 'fomc-store-v5', read_only: true,
};

function reply(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

function mount() {
  const jobs = lab.labJobs;
  const routes: Record<string, unknown> = {
    '/api/v1/health': { core_status: 'ready' },
    '/api/v1/system': fixtures.system,
    '/api/v1/ops/health': opsHealth,
    '/api/v1/sources/fomc': {
      ...base, status: 'AVAILABLE', horizon: 622, suggested_as_of: AS_OF,
      counts: { epochs: 1, responses: 128, revisions: 6, observations: 15, cycles: 80 },
    },
    '/api/v1/sources/fomc/snapshot': {
      ...base, discovery: null, health: null, items: [],
      snapshot: { policy: 'FOMC_CURRENT_CONTENT_SELECTION_V1', spec_hash: SPEC, mode: 'DURABLE_OBSERVED', T: AS_OF,
        H: 622, P: 606, read_state: 'FOMC_RESOLVED', identity: IDENTITY },
    },
    '/api/v1/lab/models': lab.labModels,
    '/api/v1/lab/jobs': jobs,
    '/api/v1/observability/predictions': lab.ledgerPage,
    '/api/v1/observability/references': lab.references,
    '/api/v1/observability/health': lab.health,
    '/api/v1/observability/monitoring': lab.monitoring,
  };
  const fetchMock = vi.fn((input: string, init?: RequestInit) => {
    void init;
    const [path] = String(input).split('?');
    const id = path?.match(/\/lab\/jobs\/([a-f0-9]{32})\/results$/)?.[1];
    if (id) {
      const job = jobs.jobs.find((item: { id: string }) => item.id === id);
      return reply(job?.kind === 'dataset' ? lab.datasetResult : lab.experimentResult);
    }
    if (path?.startsWith('/api/v1/observability/predictions/')) return reply(lab.predictionView);
    const payload = routes[path ?? ''];
    return payload === undefined ? reply({ error: 'no such endpoint' }, 404) : reply(payload);
  });
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter initialEntries={['/events']}><App /></MemoryRouter>);
  return fetchMock;
}

beforeEach(() => { invalidate(); clearLabToken(); });
afterEach(() => vi.unstubAllGlobals());

describe('main journey', () => {
  it('walks snapshot → dataset → model → prediction → monitoring → health → API docs, read-only', async () => {
    const fetchMock = mount();

    // Snapshot: the source read at its own attested instant and horizon.
    expect(await screen.findByTestId('snapshot-identity')).toHaveTextContent(IDENTITY);
    expect(screen.getByText('622 · 606')).toBeInTheDocument();

    // Dataset: admissible decisions and explicit exclusions.
    await userEvent.click(screen.getByRole('link', { name: 'Lab' }));
    await userEvent.type(await screen.findByLabelText(/Operator token \(/), TOKEN);
    await userEvent.click(screen.getByRole('button', { name: 'Unlock' }));
    expect(await screen.findByText(/182 of 240 candidate decisions are admissible/)).toBeInTheDocument();
    expect(within(screen.getByRole('table', { name: 'Exclusions by reason' })).getByText('PRICE_FEATURE_WARMUP_OR_GAP')).toBeInTheDocument();

    // Model: declared capabilities; the token carried over, no second prompt.
    await userEvent.click(screen.getByRole('link', { name: 'Models' }));
    const momentum = await screen.findByRole('article', { name: 'Model local-momentum-v1' });
    expect(within(momentum).getByText('TRAIN: YES')).toBeInTheDocument();

    // Experiment: negative result kept.
    await userEvent.click(screen.getByRole('link', { name: 'Experiments' }));
    expect((await screen.findAllByText('CRITERION NOT MET')).length).toBeGreaterThan(0);

    // Prediction: absent outputs are "not provided".
    await userEvent.click(screen.getByRole('link', { name: 'Predictions' }));
    await userEvent.click((await screen.findAllByRole('button', { name: 'Detail' }))[0]!);
    const detail = await screen.findByLabelText('Prediction detail');
    expect(within(detail).getByText('Target price').nextSibling).toHaveTextContent('not provided');

    // Monitoring: causes kept apart.
    await userEvent.click(screen.getByRole('link', { name: 'Monitoring' }));
    const diagnosis = await screen.findByLabelText('Diagnosis');
    expect(diagnosis.querySelectorAll('[data-category]').length).toBe(4);

    // Health panel and API docs.
    await userEvent.click(screen.getByRole('link', { name: 'System' }));
    const panel = await screen.findByRole('region', { name: 'Operations health' });
    expect(await within(panel).findByTestId('ops-status')).toHaveTextContent('OBSERVED');
    await userEvent.click(screen.getByRole('link', { name: 'API docs' }));
    expect(await screen.findByRole('table', { name: 'Main path' })).toBeInTheDocument();

    const methods = fetchMock.mock.calls.map(([, init]) => (init as RequestInit | undefined)?.method ?? 'GET');
    expect(methods.every((method) => method === 'GET')).toBe(true);
  }, 30000);
});
