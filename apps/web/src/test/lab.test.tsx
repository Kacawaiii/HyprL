/**
 * The Lab views over real-shaped responses (captured from the synthetic demo).
 *
 * The recurring assertions are about what the page must not claim: a write is sent only by an explicit
 * control (build, launch, cancel, monitor) with the bearer token, an absent output is "not provided",
 * pending labels are not realized, an unmeasured service is not "healthy".
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { App } from '../App';
import { clearLabToken } from '../state/labToken';
import { invalidate } from '../state/useQuery';
import { lab } from './labFixtures';

const TOKEN = 't'.repeat(40);

function reply(payload: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(payload)) } as Response);
}

type Init = RequestInit & { headers?: Record<string, string> };

function mock(overrides: Record<string, unknown> = {}) {
  const jobs = overrides['/api/v1/lab/jobs'] ?? lab.labJobs;
  const routes: Record<string, unknown> = {
    '/api/v1/lab/models': lab.labModels,
    '/api/v1/lab/jobs': jobs,
    '/api/v1/observability/predictions': lab.ledgerPage,
    '/api/v1/observability/references': lab.references,
    '/api/v1/observability/health': lab.health,
    '/api/v1/observability/monitoring': lab.monitoring,
    '/api/v1/research/hypotheses': lab.hypothesesPage,
    '/api/v1/research/comparison': lab.comparison,
    '/api/v1/research/proposals': lab.proposals,
    '/api/v1/health': { core_status: 'ready' },
    ...overrides,
  };
  return vi.fn((input: string, init?: Init) => {
    const [path] = String(input).split('?');
    const id = path?.match(/\/lab\/jobs\/([a-f0-9]{32})\/results$/)?.[1];
    if (id) {
      const job = (jobs as typeof lab.labJobs).jobs.find((item: { id: string }) => item.id === id);
      if (job?.kind === 'monitoring') return reply(monitoringResult(job.subject));
      return reply(job?.kind === 'dataset' ? lab.datasetResult : lab.experimentResult);
    }
    if (path?.startsWith('/api/v1/observability/predictions/')) return reply(lab.predictionView);
    if (path?.startsWith('/api/v1/research/hypotheses/')) return reply(lab.hypothesisDetail);
    const payload = routes[init?.method === 'POST' ? `POST ${path}` : path ?? ''];
    if (payload instanceof Error) return Promise.reject(payload);
    if (payload && typeof payload === 'object' && '__status' in payload) {
      const failure = payload as { __status: number; error: string };
      return reply({ error: failure.error }, failure.__status);
    }
    return payload === undefined ? reply({ error: 'no such endpoint' }, 404) : reply(payload);
  });
}

/**
 * A monitoring job's result. Constructed: its per-product view is the captured observability monitoring
 * response; the envelope follows scripts/trading_lab/platform/lab_monitoring.py.
 */
function monitoringResult(subject: string) {
  return {
    state: 'COMPLETE', result_hash: '9'.repeat(64), error_code: null,
    result: {
      schema: 'lab-experiment-monitoring-v1', experiment_job_id: subject, experiment_hash: '8'.repeat(64),
      model_id: 'local-momentum-v1', as_of: '2026-10-06T10:00:00Z', predictions: 80, reference_split: 'validation',
      monitored_split: 'test', ledger: { verified: true, records: 400, head_hash: '7'.repeat(64) }, synthetic: true,
      products: { 'BTC-USD': { reference_hash: '6'.repeat(64), monitoring_hash: '5'.repeat(64), view: lab.monitoring } },
      limitations: ['synthetic prices and labels; establishes no real edge'],
    },
  };
}

function withMonitoringJob(state: string) {
  const subject = lab.labJobs.jobs[0].id;
  return {
    ...lab.labJobs,
    jobs: [...lab.labJobs.jobs, { ...lab.labJobs.jobs[0], id: 'd'.repeat(32), kind: 'monitoring', subject, state,
      progress: state === 'COMPLETE' ? 1 : 0.5, result_hash: state === 'COMPLETE' ? '9'.repeat(64) : null }],
  };
}

function open(path: string, overrides?: Record<string, unknown>) {
  const fetchMock = mock(overrides);
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
  return fetchMock;
}

async function unlock() {
  await userEvent.type(await screen.findByLabelText(/Operator token \(/), TOKEN);
  await userEvent.click(screen.getByRole('button', { name: 'Unlock' }));
}

function calls(fetchMock: ReturnType<typeof mock>) {
  return fetchMock.mock.calls.map(([url, init]) => ({ url: String(url), init: init as Init | undefined }));
}

beforeEach(() => { invalidate(); clearLabToken(); });
afterEach(() => { vi.unstubAllGlobals(); });

describe('operator token gate', () => {
  it('asks for the token and sends no Model Lab request without one', async () => {
    const fetchMock = open('/lab/models');
    expect(await screen.findByText(/The model registry needs the operator token/)).toBeInTheDocument();
    expect(calls(fetchMock).some((call) => call.url.includes('/api/v1/lab/'))).toBe(false);
  });

  it('sends the token as a bearer header only, never in a URL, and can forget it', async () => {
    const fetchMock = open('/lab/models');
    await screen.findByText(/needs the operator token/);
    await unlock();
    expect(await screen.findByRole('article', { name: 'Model local-momentum-v1' })).toBeInTheDocument();
    const lab = calls(fetchMock).filter((call) => call.url.includes('/api/v1/lab/'));
    expect(lab.length).toBeGreaterThan(0);
    for (const call of lab) {
      expect(call.init?.headers?.Authorization).toBe(`Bearer ${TOKEN}`);
      expect(call.url).not.toContain(TOKEN);
    }
    await userEvent.click(screen.getByRole('button', { name: 'Forget token' }));
    expect(await screen.findByText(/needs the operator token/)).toBeInTheDocument();
    expect(screen.queryByRole('article', { name: 'Model local-momentum-v1' })).not.toBeInTheDocument();
  });

  it('explains a refused token', async () => {
    open('/lab/models', { '/api/v1/lab/models': { __status: 401, error: 'Model Lab requires operator authentication' } });
    await unlock();
    expect(await screen.findByText('Token refused')).toBeInTheDocument();
  });

  it('explains a server without Model Lab', async () => {
    open('/lab/models', { '/api/v1/lab/models': { __status: 503, error: 'Model Lab is not configured' } });
    await unlock();
    expect(await screen.findByText('Not configured on this server')).toBeInTheDocument();
  });
});

describe('models', () => {
  it('shows declared capabilities, absent outputs and the demonstration/frozen distinction', async () => {
    open('/lab/models');
    await unlock();
    const momentum = await screen.findByRole('article', { name: 'Model local-momentum-v1' });
    expect(within(momentum).getByText(/EXTERNAL ADAPTER/)).toBeInTheDocument();
    expect(within(momentum).getByText('TRAIN: YES')).toBeInTheDocument();
    expect(within(momentum).getByText(/target_price, class, probabilities, quantiles, scenarios/)).toBeInTheDocument();
    const frozen = screen.getByRole('article', { name: 'Model paper-ridge-v1' });
    expect(within(frozen).getByText(/FROZEN REFERENCE/)).toBeInTheDocument();
    expect(within(frozen).getByText('TRAIN: NO')).toBeInTheDocument();
    expect(screen.getByText(/never uploads or imports code/)).toBeInTheDocument();
    expect(screen.getByLabelText('Adapter registration').textContent).toContain('register_entry_point');
    expect(within(momentum).getByRole('link', { name: 'Use in an experiment' }).getAttribute('href')).toContain('model=local-momentum-v1');
    expect(within(frozen).queryByRole('link', { name: 'Use in an experiment' })).not.toBeInTheDocument();
  });

  it('keeps the selected model when switching Beginner/Expert', async () => {
    open('/lab/models');
    await unlock();
    const momentum = await screen.findByRole('article', { name: 'Model local-momentum-v1' });
    await userEvent.click(within(momentum).getByRole('button', { name: 'Select this model' }));
    expect(within(momentum).getByRole('button', { name: /Selected for/ })).toHaveAttribute('aria-pressed', 'true');
    expect(within(momentum).queryByText('Contract')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Expert' }));
    expect(await within(momentum).findByText('Contract')).toBeInTheDocument();
    expect(within(screen.getByRole('article', { name: 'Model local-momentum-v1' })).getByRole('button', { name: /Selected for/ })).toBeInTheDocument();
  });
});

describe('datasets', () => {
  it('sends nothing before the token and the click, and refuses an illegal configuration', async () => {
    const fetchMock = open('/lab/datasets');
    const command = await screen.findByLabelText('Prepared dataset command');
    expect(command.textContent).toContain('"synthetic":true');
    expect(command.textContent).toContain('$HYPRL_MODEL_LAB_TOKEN');
    expect(screen.getByRole('button', { name: 'Build dataset' })).toBeDisabled();
    expect(screen.getAllByTestId('waiting-authorization')[0]).toHaveTextContent('WAITING_AUTHORIZATION');
    expect(screen.getByRole('radio', { name: /Real prices/ })).toBeDisabled();
    await unlock();
    const bars = screen.getByLabelText(/Hourly bars/);
    await userEvent.clear(bars);
    await userEvent.type(bars, '5');
    expect(await screen.findByText(/Bars must be/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Build dataset' })).toBeDisabled();
    expect(screen.queryByLabelText('Prepared dataset command')).not.toBeInTheDocument();
    expect(calls(fetchMock).some((call) => call.init?.method === 'POST')).toBe(false);
  });

  it('builds a dataset with one authenticated POST of the chosen configuration', async () => {
    const created = { job_id: 'c'.repeat(32), state: 'QUEUED', synthetic: true };
    const fetchMock = open('/lab/datasets', { 'POST /api/v1/lab/datasets': created });
    await unlock();
    await userEvent.click(await screen.findByRole('button', { name: 'Build dataset' }));
    expect(await screen.findByText(/queued in an isolated worker/)).toBeInTheDocument();
    const posts = calls(fetchMock).filter((call) => call.init?.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0]!.url).toBe('/api/v1/lab/datasets');
    expect(posts[0]!.init?.headers?.Authorization).toBe(`Bearer ${TOKEN}`);
    const body = JSON.parse(String(posts[0]!.init?.body));
    expect(body.synthetic).toBe(true);
    expect(body.products).toEqual(['BTC-USD', 'ETH-USD']);
    expect(body.horizon_seconds).toBe(14400);
  });

  it('explains a page the listener refuses', async () => {
    open('/lab/datasets', { 'POST /api/v1/lab/datasets': { __status: 403, error: 'Model Lab accepts local clients and same-origin loopback pages only' } });
    await unlock();
    await userEvent.click(await screen.findByRole('button', { name: 'Build dataset' }));
    expect(await screen.findByText(/served by the lab listener itself on a loopback address/)).toBeInTheDocument();
  });

  it('links a completed dataset to the experiment configuration', async () => {
    open('/lab/datasets?mode=expert');
    await unlock();
    const next = await screen.findByRole('link', { name: /Next: configure an experiment/ });
    expect(next.getAttribute('href')).toContain('/lab/experiments');
    expect(next.getAttribute('href')).toContain('dataset=');
    expect(next.getAttribute('href')).toContain('mode=expert');
  });

  it('shows admissible decisions and exclusions of a built dataset', async () => {
    open('/lab/datasets');
    await unlock();
    expect(await screen.findByText(/182 of 240 candidate decisions are admissible/)).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Exclusions by reason' });
    expect(within(table).getByText('PRICE_FEATURE_WARMUP_OR_GAP')).toBeInTheDocument();
    expect(screen.getByText(/no event feature was selected/)).toBeInTheDocument();
  });
});

describe('experiments', () => {
  it('compares with baselines, keeps a negative result and sends no write by itself', async () => {
    const fetchMock = open('/lab/experiments');
    await unlock();
    expect(await screen.findAllByText('CRITERION NOT MET')).toHaveLength(2);
    const table = screen.getByRole('table', { name: 'BTC-USD test comparison' });
    expect(within(table).getByText('local-momentum-v1')).toBeInTheDocument();
    expect(within(table).getByText('ZERO')).toBeInTheDocument();
    expect(within(table).getByText('TRAIN_MEAN')).toBeInTheDocument();
    expect(screen.getByText(/not a trading result/)).toBeInTheDocument();
    expect(calls(fetchMock).some((call) => call.init?.method === 'POST')).toBe(false);
  });

  const running = {
    ...lab.labJobs,
    jobs: [{ ...lab.labJobs.jobs[0], state: 'RUNNING', progress: 0.4, result_hash: null }, ...lab.labJobs.jobs.slice(1)],
  };

  it('cancels a running job with one authenticated POST', async () => {
    const id = running.jobs[0]!.id;
    const fetchMock = open('/lab/experiments', {
      '/api/v1/lab/jobs': running,
      [`POST /api/v1/lab/jobs/${id}/cancel`]: { ...running.jobs[0], cancel_requested: true },
    });
    await unlock();
    const cancel = await screen.findByRole('button', { name: 'Cancel this job' });
    expect(screen.getAllByRole('progressbar', { name: 'Job progress' }).some((bar) => bar.getAttribute('aria-valuenow') === '40')).toBe(true);
    expect(screen.getAllByRole('table', { name: 'Job log' }).length).toBeGreaterThan(0);
    expect(calls(fetchMock).some((call) => call.init?.method === 'POST')).toBe(false);
    await userEvent.click(cancel);
    expect(await screen.findByText(/Cancellation recorded/)).toBeInTheDocument();
    const posts = calls(fetchMock).filter((call) => call.init?.method === 'POST');
    expect(posts.map((call) => call.url)).toEqual([`/api/v1/lab/jobs/${id}/cancel`]);
    expect(posts[0]!.init?.body).toBe('{}');
  });

  it('follows a running job by polling', async () => {
    const fetchMock = open('/lab/experiments', { '/api/v1/lab/jobs': running });
    await unlock();
    await screen.findByRole('button', { name: 'Cancel this job' });
    const count = () => calls(fetchMock).filter((call) => call.url.endsWith('/api/v1/lab/jobs')).length;
    const before = count();
    await waitFor(() => expect(count()).toBeGreaterThan(before), { timeout: 4500 });
  }, 8000);

  it('launches an experiment on the chosen dataset and model, and says real data waits for authorization', async () => {
    const launched = { job_id: 'e'.repeat(32), state: 'QUEUED', synthetic: true, fingerprint: 'a'.repeat(64) };
    const fetchMock = open('/lab/experiments?model=local-momentum-v1', { 'POST /api/v1/lab/experiments': launched });
    await unlock();
    await screen.findAllByText('CRITERION NOT MET');
    expect(screen.getAllByTestId('waiting-authorization')[0]).toHaveTextContent('WAITING_AUTHORIZATION');
    const button = await screen.findByRole('button', { name: 'Launch experiment' });
    await waitFor(() => expect(button).toBeEnabled());
    expect(screen.getByLabelText('Chosen model')).toHaveTextContent('EXTERNAL ADAPTER');
    await userEvent.click(button);
    expect(await screen.findByText(/recorded before any result/)).toBeInTheDocument();
    const posts = calls(fetchMock).filter((call) => call.init?.method === 'POST');
    expect(posts.map((call) => call.url)).toEqual(['/api/v1/lab/experiments']);
    expect(JSON.parse(String(posts[0]!.init?.body))).toEqual({
      dataset_hash: lab.datasetResult.result.dataset_hash, model_id: 'local-momentum-v1', embargo_seconds: 3600,
    });
  });

  it('refuses an out-of-range embargo before sending', async () => {
    const fetchMock = open('/lab/experiments');
    await unlock();
    const embargo = await screen.findByLabelText(/Embargo/);
    await userEvent.clear(embargo);
    await userEvent.type(embargo, '90000');
    expect(await screen.findByText(/embargo from 0 to 86400 seconds/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Launch experiment' })).toBeDisabled();
    expect(calls(fetchMock).some((call) => call.init?.method === 'POST')).toBe(false);
  });

  it('opens the monitoring of a completed experiment and lands on it', async () => {
    const queued = { job_id: 'd'.repeat(32), state: 'QUEUED', synthetic: true, subject: lab.labJobs.jobs[0]!.id };
    const fetchMock = open('/lab/experiments', { 'POST /api/v1/lab/monitoring': queued });
    await unlock();
    await screen.findAllByText('CRITERION NOT MET');
    await userEvent.click(screen.getByRole('button', { name: 'Open the monitoring of these predictions' }));
    expect(await screen.findByLabelText('Experiment monitoring')).toBeInTheDocument();
    const post = calls(fetchMock).find((call) => call.init?.method === 'POST');
    expect(post!.url).toBe('/api/v1/lab/monitoring');
    expect(JSON.parse(String(post!.init?.body))).toEqual({ experiment_job_id: lab.labJobs.jobs[0]!.id });
  });

  it('prepares a reproduction with the recorded dataset, model and embargo', async () => {
    open('/lab/experiments');
    await unlock();
    await screen.findAllByText('CRITERION NOT MET');
    await userEvent.click(screen.getByText('Reproduce'));
    const command = screen.getByLabelText('Reproduction command');
    expect(command.textContent).toContain(lab.experimentResult.result.manifest.dataset_hash);
    expect(command.textContent).toContain('"model_id":"local-momentum-v1"');
    expect(command.textContent).toContain('"embargo_seconds":3600');
  });

  it('shows runtime and source hashes in Expert mode only', async () => {
    open('/lab/experiments');
    await unlock();
    await screen.findAllByText('CRITERION NOT MET');
    expect(screen.queryByText('Runtime and source hashes')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Expert' }));
    expect(await screen.findByText('Runtime and source hashes')).toBeInTheDocument();
  });
});

describe('prediction ledger', () => {
  const pendingPage = (() => {
    const [first, ...rest] = lab.ledgerPage.records;
    const pending = {
      ...first, identity: 'f'.repeat(64),
      view: { ...first.view, label_state: 'PENDING', latest_label: null, label_versions: 0 },
    };
    return { ...lab.ledgerPage, records: [first, pending, ...rest], page: { returned: 3, has_more: true, next_cursor: 'CURSOR1' } };
  })();

  it('separates pending from realized labels and never fills a pending one', async () => {
    open('/lab/ledger', { '/api/v1/observability/predictions': pendingPage });
    expect(await screen.findByText(/1 pending/)).toBeInTheDocument();
    expect(screen.getAllByText('PENDING')).toHaveLength(1);
  });

  it('shows a prediction with absent outputs as not provided and its append-only label history', async () => {
    open('/lab/ledger');
    const [detail] = await screen.findAllByRole('button', { name: 'Detail' });
    await userEvent.click(detail!);
    const panel = await screen.findByLabelText('Prediction detail');
    expect(within(panel).getByText(/not recorded for this prediction/)).toBeInTheDocument();
    expect(within(panel).getByText('Target price').nextSibling).toHaveTextContent('not provided');
    expect(within(panel).getByRole('table', { name: 'Label history' })).toBeInTheDocument();
    expect(within(panel).getByText(/no calibrated method claimed/)).toBeInTheDocument();
  });

  it('continues with the original as_of and the cursor when loading more', async () => {
    const fetchMock = open('/lab/ledger', { '/api/v1/observability/predictions': pendingPage });
    await userEvent.click(await screen.findByRole('button', { name: 'Load more' }));
    await waitFor(() => expect(calls(fetchMock).filter((call) => call.url.includes('/observability/predictions?')).length).toBe(2));
    const [first, second] = calls(fetchMock).filter((call) => call.url.includes('/observability/predictions?'));
    const asOf = (url: string) => new URL(url, 'http://x').searchParams.get('as_of');
    expect(new URL(second!.url, 'http://x').searchParams.get('cursor')).toBe('CURSOR1');
    expect(asOf(second!.url)).toBe(asOf(first!.url));
  });

  it('shows the empty and error states', async () => {
    open('/lab/ledger', { '/api/v1/observability/predictions': { ...lab.ledgerPage, records: [], page: { returned: 0, has_more: false, next_cursor: null } } });
    expect(await screen.findByText('No prediction recorded')).toBeInTheDocument();
  });

  it('shows a failure with a retry', async () => {
    open('/lab/ledger', { '/api/v1/observability/predictions': { __status: 503, error: 'research observability store not configured' } });
    expect(await screen.findByText('Could not load')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });
});

describe('monitoring', () => {
  it('keeps the four causes apart and raises only what the evidence shows', async () => {
    open('/lab/monitoring');
    const section = await screen.findByLabelText('Diagnosis');
    const raised = lab.monitoring.classification.map((item: { category: string }) => item.category);
    for (const category of ['MISSING_DATA', 'TECHNICAL_DEGRADATION', 'DRIFT', 'PERFORMANCE_DROP']) {
      const row = section.querySelector(`[data-category="${category}"]`) as HTMLElement;
      expect(within(row).getByText(raised.includes(category) ? 'RAISED' : 'NOT RAISED')).toBeInTheDocument();
    }
  });

  it('states an unmeasured inference service as unknown, not healthy', async () => {
    open('/lab/monitoring');
    const inference = await screen.findByLabelText('Inference');
    expect(within(inference).getByText('NOT_OBSERVED')).toBeInTheDocument();
    expect(within(inference).getByText(/unknown, not healthy/)).toBeInTheDocument();
  });

  it('gives every edge claim its sample and method and no verdict', async () => {
    open('/lab/monitoring');
    const section = await screen.findByLabelText('Edge versus baselines');
    expect(within(section).getAllByText(/paired predictions \(paired-descriptive-mse-reduction-v1\)/).length).toBeGreaterThan(0);
    expect(within(section).getByText(/does not establish that an edge exists/)).toBeInTheDocument();
  });

  it('breaks performance down by product, period and regime with the regime definition', async () => {
    open('/lab/monitoring');
    await screen.findByLabelText('Performance breakdown');
    expect(screen.getByRole('table', { name: 'By product' })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'By period (decision month)' })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'By regime' })).toBeInTheDocument();
    expect(screen.getByText(/price-regimes-v1/)).toBeInTheDocument();
  });

  it('requests monitoring against the chosen reference with an explicit as_of', async () => {
    const fetchMock = open('/lab/monitoring');
    await screen.findByLabelText('Diagnosis');
    const call = calls(fetchMock).find((item) => item.url.includes('/observability/monitoring'));
    const params = new URL(call!.url, 'http://x').searchParams;
    expect(params.get('as_of')).toBeTruthy();
    expect(params.get('reference_hash')).toBe(lab.references.records[0].identity);
    expect(params.get('product')).toBe(lab.references.records[0].payload.selection.product);
  });

  it('shows the monitoring of a Model Lab experiment against its validation reference', async () => {
    open(`/lab/monitoring?monitor=${'d'.repeat(32)}`, { '/api/v1/lab/jobs': withMonitoringJob('COMPLETE') });
    await unlock();
    const section = await screen.findByLabelText('Experiment monitoring');
    expect(await within(section).findByText(/test split is monitored against a reference fixed on the validation split/)).toBeInTheDocument();
    expect(within(section).getByText(/chain verified/)).toBeInTheDocument();
    expect(within(section).getByText(/establishes no real edge/)).toBeInTheDocument();
    expect(within(section).getByRole('button', { name: 'BTC-USD' })).toHaveAttribute('aria-pressed', 'true');
    expect(within(section).getByLabelText('Diagnosis')).toBeInTheDocument();
  });

  it('follows a running monitoring job instead of concluding', async () => {
    open('/lab/monitoring', { '/api/v1/lab/jobs': withMonitoringJob('RUNNING') });
    await unlock();
    const section = await screen.findByLabelText('Experiment monitoring');
    expect(await within(section).findByRole('progressbar', { name: 'Job progress' })).toHaveAttribute('aria-valuenow', '50');
    expect(within(section).queryByLabelText('Diagnosis')).not.toBeInTheDocument();
  });

  it('says so when no reference exists', async () => {
    open('/lab/monitoring', { '/api/v1/observability/references': { ...lab.references, records: [] } });
    expect(await screen.findByText('No monitoring reference')).toBeInTheDocument();
  });
});

describe('hypotheses', () => {
  it('reports the comparison as waiting for data, with reasons and actions', async () => {
    open('/lab/hypotheses');
    const section = await screen.findByLabelText('Prices versus prices plus events');
    expect(await within(section).findByText('WAITING_DATA')).toBeInTheDocument();
    expect(within(section).getByText('NO_ATTESTED_EVENT_PRICE_OVERLAP')).toBeInTheDocument();
    expect(within(section).getByText(/0 external model calls/)).toBeInTheDocument();
  });

  it('shows the falsification condition and a trial history', async () => {
    open('/lab/hypotheses');
    const detail = await screen.findByLabelText('Hypothesis detail');
    expect(within(detail).getByText('Falsified if')).toBeInTheDocument();
    expect(within(detail).getByRole('table', { name: 'Trial history' })).toBeInTheDocument();
  });
});

describe('navigation', () => {
  it('reaches the Lab from the sidebar and lands on datasets', async () => {
    open('/');
    await userEvent.click(await screen.findByRole('link', { name: /Lab/ }));
    expect(await screen.findByRole('navigation', { name: 'Lab views' })).toBeInTheDocument();
    expect(await screen.findByLabelText('Dataset builder')).toBeInTheDocument();
  });

  it('carries the selection through the tabs', async () => {
    open('/lab/datasets?mode=expert&product=ETH-USD&model=local-momentum-v1');
    const tab = await screen.findByRole('link', { name: 'Predictions' });
    expect(tab.getAttribute('href')).toContain('mode=expert');
    expect(tab.getAttribute('href')).toContain('product=ETH-USD');
    expect(tab.getAttribute('href')).toContain('model=local-momentum-v1');
  });
});
