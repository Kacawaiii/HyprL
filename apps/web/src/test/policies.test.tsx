import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import fixture from './policyFixtures.json';

function response(body: unknown, status = 200) {
  return Promise.resolve({ ok: status < 400, status, text: () => Promise.resolve(JSON.stringify(body)) } as Response);
}

function Probe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

function page(path = '/policies?product=BTC-USD', view: unknown = fixture.view, status = 200) {
  const fetch = vi.fn((input: string) => {
    if (input.includes('/policies/definitions')) return response(fixture.definitions);
    if (input.includes('/policies/report')) return response(view, status);
    return response({ status: 'ok', api_version: 'v1', core_status: 'ready' });
  });
  vi.stubGlobal('fetch', fetch);
  render(<MemoryRouter initialEntries={[path]}><App /><Probe /></MemoryRouter>);
  return fetch;
}

beforeEach(() => invalidate());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); invalidate(); });

describe('Versioned calibration and separate paper protection', () => {
  it('shows model, probability method and each level origin in Beginner', async () => {
    page();
    expect(await screen.findByRole('heading', { name: 'Probability calibration · BTC-USD' })).toBeInTheDocument();
    expect(screen.getByText('SYNTHETIC DEMONSTRATION')).toBeInTheDocument();
    expect(screen.getByText('synthetic-binary-score-v1')).toBeInTheDocument();
    expect(screen.getByText('isotonic-pav-step-v1')).toBeInTheDocument();
    expect(screen.getAllByText(/origin STRATEGY · method synthetic-target-v1/)).toHaveLength(1);
    expect(screen.getAllByText(/origin POLICY · method fixed-entry-distance-v1/)).toHaveLength(5);
    expect(screen.getByRole('img', { name: /Reliability diagram/ })).toBeInTheDocument();
    expect(screen.getByText('AMBIGUOUS BAR')).toBeInTheDocument();
    expect(screen.getAllByText(/opening through stop fills at open/)).toHaveLength(3);
    expect(screen.getAllByText(/stop with NOT_OBSERVED/)).toHaveLength(3);
    expect(screen.getByText(/WAITING_AUTHORIZATION/)).toBeInTheDocument();
  });

  it('keeps product, period and model through mode and navigation and exposes Brier decomposition', async () => {
    page('/policies?product=ETH-USD&start=2026-06-01&end=2026-06-30&model=selected-model');
    await screen.findByRole('heading', { name: 'Probability calibration · ETH-USD' });
    await userEvent.click(screen.getByRole('button', { name: 'Expert' }));
    expect(await screen.findByText('binning_residual')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Training and held-out test provenance' })).toBeInTheDocument();
    expect(screen.getByTestId('location').textContent).toContain('model=selected-model');
    expect(screen.getByTestId('location').textContent).toContain('start=2026-06-01');
    await userEvent.selectOptions(screen.getByRole('combobox'), 'BTC-USD');
    expect(screen.getByRole('heading', { name: 'Probability calibration · BTC-USD' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Expert' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('link', { name: /Cockpit/ })).toHaveAttribute('href', expect.stringContaining('model=selected-model'));
  });

  it('preserves missing evidence and small-sample refusal without a false metric or chart', async () => {
    const view = structuredClone(fixture.view);
    view.report.calibrations[0]!.calibrated = { ...view.report.calibrations[0]!.calibrated,
      state: 'REFUSED_SMALL_SAMPLE', count: 8, bins: [], brier: null } as unknown as typeof view.report.calibrations[0]['calibrated'];
    page('/policies?product=BTC-USD', view);
    expect(await screen.findByText('Calibration diagnostics refused: sample too small')).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /Reliability diagram/ })).not.toBeInTheDocument();
    expect(screen.getByText(/calibrated Brier: unavailable/)).toBeInTheDocument();
  });

  it('shows no evidence for an unknown product without substituting another asset', async () => {
    page('/policies?product=AAPL');
    expect(await screen.findByText('No policy evidence for this product')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /Probability calibration/ })).not.toBeInTheDocument();
  });

  it('shows an unconfigured evidence state', async () => {
    page('/policies', { state: 'NOT_CONFIGURED', identity: null, report: null, policy_hash: fixture.view.policy_hash });
    expect(await screen.findByText('Policy evidence not configured')).toBeInTheDocument();
  });

  it('shows integrity errors and offers retry', async () => {
    const fetch = page('/policies', { error: 'policy evidence integrity failure' }, 409);
    expect(await screen.findByRole('alert')).toHaveTextContent('policy evidence integrity failure');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(fetch.mock.calls.filter((c) => c[0].includes('/policies/report')).length).toBe(2));
  });

  it('shows a loading state while evidence is pending', async () => {
    vi.stubGlobal('fetch', (input: string) => input.includes('/policies/report') ? new Promise(() => {}) : response(fixture.definitions));
    render(<MemoryRouter initialEntries={['/policies']}><App /></MemoryRouter>);
    expect(await screen.findByText('Loading policy evidence…')).toBeInTheDocument();
  });
});
