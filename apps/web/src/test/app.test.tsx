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
    '/api/v1/paper/status': fixtures.paperRunning,
    '/api/v1/paper/products': fixtures.paperProducts,
    '/api/v1/paper/events': fixtures.paperEvents,
    '/api/v1/ops/runtime': fixtures.opsRuntime,
    '/api/v1/ops/recovery': fixtures.opsRecoveryClean,
    '/api/v1/ops/storage': fixtures.opsStorage,
    '/api/v1/ops/health-history': fixtures.opsHealth,
    '/api/v1/ops/settings': fixtures.opsSettings,
    ...overrides,
  };
  return vi.fn((input: string) => {
    const path = String(input);
    // Dynamic routes first: `/api/v1/markets/BTC-USD/chart` also starts with
    // `/api/v1/markets`, and matching the index route first would hand the
    // chart a candle page.
    if (path.includes('/chart')) return jsonResponse(fixtures.chart);
    if (path.includes('/paper/') && path.includes('/equity')) {
      return jsonResponse(fixtures.paperEquity);
    }
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

function renderAt(path: string, overrides?: Record<string, unknown>) {
  if (overrides) vi.stubGlobal('fetch', mockApi(overrides));
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
    // Overview badges three capabilities. Since Phase 5D the backtest engine and
    // shadow trading both exist, so only LIVE trading remains unavailable --
    // and that one must never quietly flip to available.
    expect(screen.getAllByText('NOT AVAILABLE')).toHaveLength(1);
    expect(screen.getAllByText('AVAILABLE').length).toBeGreaterThanOrEqual(2);
    expect(fixtures.overview.capabilities.paper_trading).toBe(true);
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
    expect(screen.getByText('UNOBSERVED')).toBeInTheDocument();
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

describe('Paper', () => {
  it('says shadow mode and no real money before any number', async () => {
    renderAt('/paper');
    expect(await screen.findByText('SHADOW MODE')).toBeInTheDocument();
    expect(screen.getByText('NO REAL MONEY')).toBeInTheDocument();
    expect(screen.getByText('NO BROKER')).toBeInTheDocument();
    expect(screen.getByText('NO EXCHANGE ACCOUNT')).toBeInTheDocument();
  });

  it('offers no way to place or override a trade', async () => {
    renderAt('/paper');
    await screen.findByText('SHADOW MODE');
    const buttons = screen.queryAllByRole('button');
    for (const button of buttons) {
      const label = (button.textContent ?? '').toLowerCase();
      for (const word of ['buy', 'sell', 'execute', 'start', 'stop', 'order']) {
        expect(label).not.toContain(word);
      }
    }
  });

  it('shows the holdout window and its current state', async () => {
    renderAt('/paper');
    expect(await screen.findByText('Protected research holdout')).toBeInTheDocument();
    expect(screen.getByText('UNOBSERVED')).toBeInTheDocument();
    expect(screen.getByText(/allowed until the embargo boundary/)).toBeInTheDocument();
  });

  it('says the embargo is active during the window', async () => {
    vi.stubGlobal('fetch', mockApi({ '/api/v1/paper/status': fixtures.paperEmbargoed }));
    renderAt('/paper');
    expect(await screen.findByText('EMBARGO ACTIVE')).toBeInTheDocument();
    expect(
      screen.getByText(/disabled to preserve the confirmatory research holdout/),
    ).toBeInTheDocument();
  });

  it('tells the reader the cockpit cannot start a session', async () => {
    vi.stubGlobal('fetch', mockApi({ '/api/v1/paper/status': fixtures.paperStopped }));
    renderAt('/paper');
    expect(await screen.findByText('No shadow session is running')).toBeInTheDocument();
    expect(screen.getByText(/paper_shadow.sh start/)).toBeInTheDocument();
  });

  it('renders backend paper state verbatim', async () => {
    renderAt('/paper');
    await screen.findByText('SHADOW MODE');
    expect((await screen.findAllByText('BTC-USD')).length).toBeGreaterThan(0);
    expect(screen.getByText('LONG')).toBeInTheDocument();
    expect(screen.getByText('0.0031')).toBeInTheDocument();   // prediction, unmodified
    expect(screen.getByText('0.06')).toBeInTheDocument();     // target exposure
    expect(screen.getByText('99,871.2')).toBeInTheDocument(); // paper equity
  });

  it('states the paper fill latency instead of hiding it', async () => {
    renderAt('/paper');
    await screen.findByText('SHADOW MODE');
    expect(
      screen.getByText('recorded-when-the-fill-bar-closes-v1'),
    ).toBeInTheDocument();
    expect(screen.getByText(/never liquidates a terminal position/)).toBeInTheDocument();
  });
});

describe('system operations', () => {
  it('reports a healthy startup without alarming the user', async () => {
    renderAt('/system');
    expect(await screen.findByText('HEALTHY STARTUP')).toBeInTheDocument();
    expect(screen.queryByText(/RECOVERED AFTER UNCLEAN SHUTDOWN/)).toBeNull();
    expect(screen.getAllByText('VERIFIED').length).toBe(2);
  });

  it('states an unclean restart calmly once the chain still verifies', async () => {
    renderAt('/system', {
      '/api/v1/ops/recovery': fixtures.opsRecoveryUnclean,
    });
    expect(
      await screen.findByText('RECOVERED AFTER UNCLEAN SHUTDOWN'),
    ).toBeInTheDocument();
    // recovery succeeded, so the chain is still reported as sound
    expect(screen.getAllByText('VERIFIED').length).toBe(2);
  });

  it('surfaces a broken event chain as a product error with a next step', async () => {
    renderAt('/system', { '/api/v1/ops/recovery': fixtures.opsRecoveryBroken });
    expect(await screen.findByText('PAPER_EVENT_CHAIN_INVALID')).toBeInTheDocument();
    expect(screen.getByText('INVALID')).toBeInTheDocument();
    expect(screen.getByText(/Do not delete the log/)).toBeInTheDocument();
    // a code and a sentence, never a stack trace
    expect(screen.queryByText(/Traceback/)).toBeNull();
  });

  it('shows runtime storage in human units and states the retention rule', async () => {
    renderAt('/system');
    expect(await screen.findByText('Runtime storage')).toBeInTheDocument();
    expect(screen.getByText('4.9 MiB')).toBeInTheDocument();
    expect(
      screen.getByText(/append-only; never pruned automatically/),
    ).toBeInTheDocument();
  });

  it('shows snapshot pressure per product', async () => {
    renderAt('/system');
    expect(await screen.findByText('Snapshots')).toBeInTheDocument();
    expect(screen.getByText(/198 events since last/)).toBeInTheDocument();
  });

  it('paints an active embargo amber rather than red', async () => {
    renderAt('/system');
    expect(await screen.findByText('EMBARGOED')).toBeInTheDocument();
    const badge = screen.getByText('EMBARGOED');
    expect(badge.getAttribute('data-tone')).toBe('warn');
    // a degraded ingestion is amber too; only integrity failures are 'off'
    expect(screen.getByText('DEGRADED').getAttribute('data-tone')).toBe('warn');
  });

  it('keeps declaring the holdout unobserved and trading disabled', async () => {
    renderAt('/system');
    expect(await screen.findByText('UNOBSERVED')).toBeInTheDocument();
    expect(screen.getByText('NOT CONNECTED')).toBeInTheDocument();
    expect(screen.getByText('NO')).toBeInTheDocument();
  });

  it('never renders an absolute path or a process id-bearing path', async () => {
    const { container } = renderAt('/system');
    await screen.findByText('Runtime storage');
    expect(container.textContent).not.toMatch(/\/home\//);
    expect(container.textContent).not.toMatch(/\/var\/trading_lab/);
  });
});

describe('settings', () => {
  it('offers appearance and interface controls', async () => {
    renderAt('/settings');
    expect(await screen.findByLabelText('Theme')).toBeInTheDocument();
    expect(screen.getByLabelText('Timestamps')).toBeInTheDocument();
  });

  it('declares the trading contracts immutable', async () => {
    renderAt('/settings');
    expect(await screen.findByText('IMMUTABLE IN THIS BUILD')).toBeInTheDocument();
    expect(screen.getByText(/frozen/i)).toBeInTheDocument();
  });

  it('offers no control that could change a trading contract', async () => {
    const { container } = renderAt('/settings');
    await screen.findByText('IMMUTABLE IN THIS BUILD');
    const controls = [...container.querySelectorAll('input, select, textarea')];
    const names = controls
      .map((node) => `${node.getAttribute('aria-label') ?? ''} ${node.getAttribute('name') ?? ''}`)
      .join(' ')
      .toLowerCase();
    for (const forbidden of ['threshold', 'risk', 'fee', 'slippage', 'alpha', 'holdout', 'exposure']) {
      expect(names).not.toContain(forbidden);
    }
  });

  it('lists the fields the backend refuses, from the backend', async () => {
    renderAt('/settings');
    expect(await screen.findByText(/Fields the backend refuses/)).toBeInTheDocument();
    expect(screen.getByText('signal_threshold')).toBeInTheDocument();
    expect(screen.getByText('holdout_end')).toBeInTheDocument();
  });

  it('says operational settings are written from the command line', async () => {
    renderAt('/settings');
    expect(
      await screen.findByText(/hyprl.sh settings --set field=value/),
    ).toBeInTheDocument();
  });

  it('persists the theme choice for the next launch', async () => {
    renderAt('/settings');
    const select = await screen.findByLabelText('Theme');
    await userEvent.selectOptions(select, 'light');
    await waitFor(() => expect(localStorage.getItem('hyprl.theme')).toBe('light'));
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('is reachable from the sidebar', async () => {
    renderAt('/');
    expect(await screen.findByRole('link', { name: /Settings/ })).toBeInTheDocument();
  });
});

describe('startup behaviour', () => {
  it('stays usable when the operations endpoints fail', async () => {
    renderAt('/system', {
      '/api/v1/ops/runtime': new Error('ops unavailable'),
      '/api/v1/ops/storage': new Error('ops unavailable'),
      '/api/v1/ops/health-history': new Error('ops unavailable'),
    });
    // the research sections still render. A generous timeout: this page runs
    // five independent queries and three of them are rejecting.
    expect(
      await screen.findByText('Market corpus', {}, { timeout: 5000 }),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Settings/ })).toBeInTheDocument();
  });

  it('renders the shell immediately even while the runtime is still loading', () => {
    const pending = vi.fn(() => new Promise<Response>(() => undefined));
    vi.stubGlobal('fetch', pending);
    render(
      <MemoryRouter initialEntries={['/system']}>
        <App />
      </MemoryRouter>,
    );
    // navigation is available before any request resolves
    expect(screen.getByRole('link', { name: /Overview/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Settings/ })).toBeInTheDocument();
  });
});
