/**
 * Behavioural tests for the cockpit shell and pages.
 *
 * These assert what a user can see and do -- not pixels, and not snapshots
 * that break on every copy change. The most important ones check that the UI
 * displays backend values verbatim and never derives a trading decision.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { invalidate } from '../state/useQuery';
import * as fixtures from './fixtures';

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve({
    ok: status < 400,
    status,
    text: () => Promise.resolve(JSON.stringify(payload)),
  } as Response);
}

function mockApi(overrides: Record<string, unknown> = {}) {
  const routes: Record<string, unknown> = {
    '/api/v1/health': { status: 'ok', api_version: 'v1', core_status: 'ready' },
    '/api/v1/overview': fixtures.overview,
    '/api/v1/system': fixtures.system,
    '/api/v1/markets': fixtures.markets,
    '/api/v1/signals': fixtures.signals,
    '/api/v1/risk/targets': fixtures.risk,
    '/api/v1/research/benchmarks': { benchmarks: fixtures.overview.benchmarks },
    '/api/v1/backtests': fixtures.backtests,
    ...overrides,
  };
  return vi.fn((input: string) => {
    const path = String(input);
    // Dynamic routes first: `/api/v1/markets/BTC-USD/chart` also starts with
    // `/api/v1/markets`, and matching the index route first would hand the
    // chart a candle page.
    if (path.includes('/chart')) return jsonResponse(fixtures.chart);
    if (path.includes('/backtests/') && path.includes('/equity')) {
      return jsonResponse(fixtures.backtestEquity);
    }
    if (path.includes('/backtests/') && path.includes('/fills')) {
      return jsonResponse(fixtures.backtestFills);
    }
    if (/\/api\/v1\/markets\/[^/?]+(\?|$)/.test(path)) {
      return jsonResponse(fixtures.candlePage);
    }
    for (const [route, payload] of Object.entries(routes)) {
      const [base] = path.split('?');
      if (base === route) {
        if (payload instanceof Error) return Promise.reject(payload);
        return jsonResponse(payload);
      }
    }
    return jsonResponse({ error: 'no such endpoint' }, 404);
  });
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  invalidate();
  vi.stubGlobal('fetch', mockApi());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('app shell', () => {
  it('renders the brand and every section link', async () => {
    renderAt('/');
    expect(screen.getByText('HyprL')).toBeInTheDocument();
    for (const label of ['Overview', 'Markets', 'Signals', 'Risk', 'Research', 'System']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument();
    }
  });

  it('exposes the navigation toggle to assistive technology', async () => {
    renderAt('/');
    const toggle = screen.getByRole('button', { name: /toggle navigation/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
  });

  it('is reachable by keyboard', async () => {
    renderAt('/');
    await userEvent.tab();
    expect(document.activeElement).toBeInstanceOf(HTMLElement);
    expect(document.activeElement?.tagName).toMatch(/^(A|BUTTON)$/);
  });

  it('keeps the shell usable when a request fails', async () => {
    vi.stubGlobal('fetch', mockApi({ '/api/v1/overview': new Error('backend down') }));
    renderAt('/');
    expect(await screen.findByRole('alert')).toBeInTheDocument();
    // navigation survives a failed data request
    expect(screen.getByRole('link', { name: 'Markets' })).toBeInTheDocument();
  });
});

describe('overview', () => {
  it('shows corpus coverage and the frozen engine badges', async () => {
    renderAt('/');
    // BTC-USD appears both as a product card heading and in the benchmark
    // table, so the heading role is what disambiguates it.
    expect(await screen.findByRole('heading', { name: 'BTC-USD' })).toBeInTheDocument();
    expect(screen.getAllByText('8,750').length).toBeGreaterThan(0);
    expect(screen.getAllByText('FROZEN V1')).toHaveLength(2);
  });

  it('reports unavailable capabilities instead of promising them', async () => {
    renderAt('/');
    await screen.findByRole('heading', { name: 'BTC-USD' });
    // Overview badges three capabilities. Since Phase 5C the backtest engine
    // exists, so exactly the two trading capabilities remain unavailable.
    expect(screen.getAllByText('NOT AVAILABLE')).toHaveLength(2);
    expect(screen.getAllByText('AVAILABLE').length).toBeGreaterThanOrEqual(1);
    expect(fixtures.overview.capabilities.paper_trading).toBe(false);
    expect(fixtures.overview.capabilities.live_trading).toBe(false);
  });

  it('never invents a latest price', async () => {
    renderAt('/');
    await screen.findByRole('heading', { name: 'BTC-USD' });
    expect(screen.getAllByText('unavailable').length).toBeGreaterThan(0);
  });
});

describe('markets', () => {
  it('requests a bounded window rather than the whole corpus', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    await waitFor(() => expect(screen.getByLabelText('Product')).toBeInTheDocument());
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      const chartCall = calls.find((path) => path.includes('/chart'));
      const candleCall = calls.find((path) => path.includes('/markets/BTC-USD?'));
      expect(chartCall).toContain('max_points=500');
      expect(candleCall).toContain('limit=200');
      // a start/end window is always supplied for the default view
      expect(chartCall).toContain('start=');
    });
  });

  it('discloses that an aggregated series is not native 1h data', async () => {
    renderAt('/markets');
    expect(await screen.findByText(/aggregated 18×1h buckets/)).toBeInTheDocument();
  });
});

describe('signals', () => {
  it('reports an absent run rather than fabricating decisions', async () => {
    renderAt('/signals');
    expect(await screen.findByText(/No persisted signal run available/i)).toBeInTheDocument();
    expect(screen.getByText('static-symmetric-threshold-v1')).toBeInTheDocument();
  });

  it('displays the frozen thresholds from the backend', async () => {
    renderAt('/signals');
    expect(await screen.findByText(/> 0.0025/)).toBeInTheDocument();
    expect(screen.getByText(/< -0.0025/)).toBeInTheDocument();
  });
});

describe('risk', () => {
  it('renders the frozen risk contract', async () => {
    renderAt('/risk');
    expect(await screen.findByText('Risk contract V1')).toBeInTheDocument();
    expect(screen.getAllByText('0.25').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/Signal strength is not position size/)).toBeInTheDocument();
  });

  it('reports no persisted targets instead of inventing them', async () => {
    renderAt('/risk');
    expect(
      await screen.findByText(/No persisted position target run available/i),
    ).toBeInTheDocument();
  });
});

describe('research', () => {
  it('labels V2 as exploratory and the confirmatory run as not observed', async () => {
    renderAt('/research');
    expect(await screen.findByText('EXPLORATORY')).toBeInTheDocument();
    expect(screen.getAllByText(/CONFIRMATORY: NOT OBSERVED/).length).toBe(2);
  });

  it('shows the committed rank_ic verbatim, including negative values', async () => {
    renderAt('/research');
    expect(await screen.findByText('-0.003041')).toBeInTheDocument();
    expect(screen.getByText('-0.007970')).toBeInTheDocument();
  });
});

describe('system', () => {
  it('shows the engine hashes and reserved holdout', async () => {
    renderAt('/system');
    expect(await screen.findByText('Market corpus')).toBeInTheDocument();
    expect(screen.getByText('coinbase_history_v1')).toBeInTheDocument();
    expect(screen.getByText('2026-09-01 → 2026-11-30')).toBeInTheDocument();
    expect(screen.getByText('NOT OBSERVED')).toBeInTheDocument();
  });

  it('never renders a filesystem path or hostname', async () => {
    const { container } = renderAt('/system');
    await screen.findByText('Market corpus');
    expect(container.textContent).not.toMatch(/\/home\//);
    expect(container.textContent).not.toMatch(/localhost:\d/);
  });
});

describe('Backtests', () => {
  it('labels every economic result as exploratory and synthetic', async () => {
    renderAt('/backtests');
    expect(await screen.findByText('EXPLORATORY')).toBeInTheDocument();
    expect(screen.getByText('SYNTHETIC EXECUTION COSTS')).toBeInTheDocument();
    expect(screen.getByText('NOT LIVE TRADING')).toBeInTheDocument();
    expect(screen.getByText('NO CONFIRMED EDGE')).toBeInTheDocument();
  });

  it('shows the backend metrics verbatim without recomputing them', async () => {
    renderAt('/backtests');
    // -0.0124950 formatted, never derived from equity in the browser
    expect(await screen.findByText('-1.25 %')).toBeInTheDocument();
    expect(screen.getByText('-4.12 %')).toBeInTheDocument();   // max drawdown
    expect(screen.getByText('412')).toBeInTheDocument();        // fill count
  });

  it('says so plainly when no run has been persisted', async () => {
    vi.stubGlobal('fetch', mockApi({ '/api/v1/backtests': fixtures.backtestsEmpty }));
    renderAt('/backtests');
    expect(
      await screen.findByText(/Economic Backtest Engine ready/),
    ).toBeInTheDocument();
    expect(screen.getByText(/no persisted economic backtest available/)).toBeInTheDocument();
  });

  it('discloses that the equity curve was downsampled', async () => {
    renderAt('/backtests');
    expect(await screen.findByText(/3 of 7728 points/)).toBeInTheDocument();
    expect(screen.getByText(/drawdowns survive/)).toBeInTheDocument();
  });

  it('lets the reader switch product without leaving the page', async () => {
    renderAt('/backtests');
    const eth = await screen.findByRole('button', { name: 'ETH-USD' });
    await userEvent.click(eth);
    await waitFor(() => expect(eth).toHaveAttribute('aria-pressed', 'true'));
  });
});
