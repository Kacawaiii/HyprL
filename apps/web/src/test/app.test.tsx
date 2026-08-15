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
    '/api/v1/instruments': fixtures.instruments,
    '/api/v1/providers': fixtures.providers,
    '/api/v1/calendars': fixtures.calendars,
    '/api/v1/portfolio': fixtures.portfolioEmpty,
    '/api/v1/portfolio/backtests': fixtures.portfolioBacktestsEmpty,
    '/api/v1/paper/portfolio': fixtures.paperPortfolioRunning,
    '/api/v1/paper/portfolio/pending': fixtures.paperPortfolioPendingIdle,
    '/api/v1/paper/portfolio/equity': fixtures.paperPortfolioEquity,
    '/api/v1/paper/legacy': fixtures.paperLegacy,
    '/api/v1/research/equities/corpus': fixtures.researchCorpusAvailable,
    ...overrides,
  };
  return vi.fn((input: string) => {
    const path = String(input);
    // Dynamic routes first: `/api/v1/markets/BTC-USD/chart` also starts with
    // `/api/v1/markets`, and matching the index route first would hand the
    // chart a candle page.
    if (path.includes('/chart')) return jsonResponse(fixtures.chart);
    // The research bars route before anything else that matches /research.
    if (path.includes('/research/equities/') && path.includes('/bars')) {
      return jsonResponse(fixtures.researchBars);
    }
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
    // Instrument sub-routes before the detail route, for the same reason the
    // chart route comes before /markets: the prefixes overlap.
    if (path.includes('/api/v1/instruments/') && path.includes('/sessions')) {
      return jsonResponse(
        path.includes('xnas') ? fixtures.equitySessions : fixtures.cryptoSessions);
    }
    if (/\/api\/v1\/instruments\/[^/?]+(\?|$)/.test(path)) {
      return jsonResponse(
        path.includes('xnas')
          ? fixtures.equityInstrumentDetail
          : fixtures.instrumentDetail);
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
    await waitFor(() => expect(screen.getByLabelText('Instrument')).toBeInTheDocument());
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
    expect(screen.getByText('SHARED CAPITAL')).toBeInTheDocument();
    expect(screen.getByText('NO CONFIRMED EDGE')).toBeInTheDocument();
    expect(screen.getByText('NOT CONNECTED')).toBeInTheDocument();
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

  it('shows the reserved holdout as unobserved', async () => {
    renderAt('/paper');
    await screen.findByText('SHADOW MODE');
    expect(screen.getByText('UNOBSERVED')).toBeInTheDocument();
    expect(screen.getByText('2026-09-01 → 2026-11-30')).toBeInTheDocument();
  });

  it('states the paper fill latency instead of hiding it', async () => {
    renderAt('/paper');
    await screen.findByText('SHADOW MODE');
    expect(
      screen.getByText('recorded-when-the-fill-bar-closes-v1'),
    ).toBeInTheDocument();
  });

  it('shows one shared cash ledger and one equity', async () => {
    renderAt('/paper');
    await screen.findByText('Shared paper portfolio');
    expect(screen.getByText('99,872.55')).toBeInTheDocument();   // equity
    expect(screen.getByText('74,981.22')).toBeInTheDocument();   // shared cash
    expect(screen.getByText('SHARED_PORTFOLIO')).toBeInTheDocument();
  });

  it('renders one card per instrument from backend state', async () => {
    renderAt('/paper');
    await screen.findByText('Shared paper portfolio');
    expect(screen.getByText('coinbase:BTC-USD')).toBeInTheDocument();
    expect(screen.getByText('coinbase:ETH-USD')).toBeInTheDocument();
    expect(screen.getByText('24,861.69')).toBeInTheDocument();   // market value
  });

  it('says a batch is waiting rather than implying a trade happened', async () => {
    /* The runtime routinely holds one instrument's target while the other is
       outstanding, and a target exposure reads like a position. */
    renderAt('/paper', {
      '/api/v1/paper/portfolio/pending': fixtures.paperPortfolioPendingWaiting,
    });
    expect(
      await screen.findByText('WAITING FOR PORTFOLIO BATCH'),
    ).toBeInTheDocument();
    expect(screen.getByText('BTC-USD READY')).toBeInTheDocument();
    expect(screen.getByText('ETH-USD WAITING')).toBeInTheDocument();
    expect(screen.getByText(/Nothing has been traded for these timestamps/))
      .toBeInTheDocument();
  });

  it('says no batch is open when nothing is pending', async () => {
    renderAt('/paper');
    expect(
      await screen.findByText(/No batch is open/),
    ).toBeInTheDocument();
    expect(screen.queryByText('WAITING FOR PORTFOLIO BATCH')).toBeNull();
  });

  it('tells the reader the cockpit cannot start a session', async () => {
    renderAt('/paper', { '/api/v1/paper/portfolio': fixtures.paperPortfolioEmpty });
    expect(
      await screen.findByText('No shared portfolio session recorded'),
    ).toBeInTheDocument();
    expect(screen.getByText(/hyprl.sh paper start/)).toBeInTheDocument();
  });

  it('keeps the legacy per-product sessions apart and never sums them', async () => {
    renderAt('/paper');
    await screen.findByText('Shared paper portfolio');
    expect(screen.getByText('PRE-SHARED-PORTFOLIO')).toBeInTheDocument();
    expect(
      screen.getByText(/never added to it/),
    ).toBeInTheDocument();
    // the legacy event count must not appear as portfolio state
    expect(screen.queryByText('105,860')).toBeNull();
  });

  it('reports the portfolio event chain as verified', async () => {
    renderAt('/paper');
    await screen.findByText('Shared paper portfolio');
    expect(screen.getByText('VERIFIED')).toBeInTheDocument();
  });

  it('shows the gross cap and allocation rule of the frozen spec', async () => {
    renderAt('/paper');
    await screen.findByText('Shared paper portfolio');
    expect(
      screen.getByText(/proportional-gross-cap-v1/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/single-pretrade-equity-batch-v1/),
    ).toBeInTheDocument();
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

describe('instruments', () => {
  it('builds the market list from the registry, not from a literal', async () => {
    renderAt('/markets');
    const select = await screen.findByLabelText('Instrument');
    const options = [...select.querySelectorAll('option')].map((node) => node.textContent);
    expect(options).toEqual(['BTC-USD', 'ETH-USD']);
  });

  it('groups options by asset class', async () => {
    const { container } = renderAt('/markets');
    await screen.findByLabelText('Instrument');
    const groups = [...container.querySelectorAll('optgroup')].map((node) =>
      node.getAttribute('label'),
    );
    expect(groups).toEqual(['Crypto']);
  });

  it('shows a new asset class without a frontend change', async () => {
    /* The registry is the source of truth: adding equities server-side must
       reach the selector with no code edit here. */
    renderAt('/markets', { '/api/v1/instruments': fixtures.instrumentsWithEquity });
    const { container } = { container: document.body };
    await screen.findByLabelText('Instrument');
    await waitFor(() => {
      const groups = [...container.querySelectorAll('optgroup')].map((node) =>
        node.getAttribute('label'),
      );
      expect(groups).toEqual(['Crypto', 'Equities']);
    });
    expect(screen.getByText('AAPL')).toBeInTheDocument();
  });

  it('invents no product list when the registry cannot be read', async () => {
    renderAt('/markets', { '/api/v1/instruments': new Error('registry down') });
    expect(await screen.findByText('No instruments available')).toBeInTheDocument();
    expect(screen.queryByText('BTC-USD')).toBeNull();
    expect(screen.queryByText('ETH-USD')).toBeNull();
  });

  it('describes the selected instrument from backend metadata', async () => {
    renderAt('/markets');
    expect(await screen.findByText('Bitcoin / US Dollar')).toBeInTheDocument();
    expect(screen.getByText('coinbase:BTC-USD')).toBeInTheDocument();
    expect(screen.getByText('Crypto')).toBeInTheDocument();
    expect(screen.getByText('coinbase')).toBeInTheDocument();
    expect(screen.getByText('CRYPTO_24_7')).toBeInTheDocument();
    expect(screen.getByText('1h, 1d')).toBeInTheDocument();
  });

  it('labels the source as market data, never as live trading', async () => {
    renderAt('/markets');
    expect(await screen.findByText('PUBLIC MARKET DATA')).toBeInTheDocument();
    // The equity provider holds a key, so it is labelled differently -- and
    // still labelled as market data, because that is all a market-data key
    // buys. Both providers appear: one per instrument panel on the page.
    expect(await screen.findByText('KEYED MARKET DATA')).toBeInTheDocument();
    // No badge claiming live trading. The shell's "No live trading"
    // disclaimer is the opposite claim and must stay.
    expect(screen.queryByText('LIVE TRADING')).toBeNull();
    expect(screen.getByText('No live trading')).toBeInTheDocument();
    expect(screen.getAllByText(/no ticks/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/no order book/).length).toBeGreaterThan(0);
    // Every provider panel denies holding account data, keyed or not.
    expect(screen.getAllByText(/no account data/).length).toBe(2);
  });

  it('switches instrument and refetches only that instrument', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    const select = await screen.findByLabelText('Instrument');
    await userEvent.selectOptions(select, 'ETH-USD');
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(calls.some((path) => path.includes('ETH-USD'))).toBe(true);
    });
    expect(await screen.findByText('Ether / US Dollar')).toBeInTheDocument();
  });

  it('addresses instruments by the identifier the rest of the API uses', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    await screen.findByLabelText('Instrument');
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(calls.some((path) => path.includes('/markets/BTC-USD'))).toBe(true);
      // Never the canonical form on the legacy endpoints. The registry's own
      // /instruments/{id} route is addressed canonically and is excluded here
      // deliberately: that endpoint takes the canonical id and no other.
      const legacy = calls.filter((path) => !path.includes('/api/v1/instruments/'));
      expect(legacy.some((path) => path.includes('coinbase%3ABTC-USD'))).toBe(false);
      expect(legacy.some((path) => path.includes('coinbase:BTC-USD'))).toBe(false);
    });
  });

  it('requests the registry once and reuses it', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    await screen.findByLabelText('Instrument');
    await waitFor(() => {
      const registryCalls = fetchMock.mock.calls
        .map((call) => String(call[0]))
        .filter((path) => path.endsWith('/api/v1/instruments'));
      expect(registryCalls.length).toBe(1);
    });
  });
});

describe('no hardcoded market list', () => {
  it('ships no product literal in any page or component', async () => {
    /* The failure this prevents: a page keeps offering a market the registry
       dropped, or misses one it gained, and nothing fails until a user
       notices. */
    const modules = import.meta.glob('../{pages,components,layouts,api}/**/*.tsx', {
      query: '?raw', import: 'default', eager: true,
    }) as Record<string, string>;
    const offenders: string[] = [];
    for (const [path, source] of Object.entries(modules)) {
      // No exemption, including for the selector itself: it reads the
      // registry and therefore needs no symbol literal anywhere.
      if (/['"]BTC-USD['"]|['"]ETH-USD['"]/.test(source)) offenders.push(path);
    }
    expect(offenders).toEqual([]);
  });
    it('says so when a registered instrument has no captured history', async () => {
    /* A registry entry with no corpus rows used to render an empty page with
       no explanation. */
    renderAt('/markets', { '/api/v1/markets': fixtures.marketsMissingEth });
    const select = await screen.findByLabelText('Instrument');
    await userEvent.selectOptions(select, 'ETH-USD');
    expect(await screen.findByText('No market history')).toBeInTheDocument();
    expect(screen.getByText(/no captured history|no bars for it/)).toBeInTheDocument();
  });
});

describe('identity boundaries', () => {
  it('addresses instruments by an id the backend supplied, never a rebuilt one', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    const select = await screen.findByLabelText('Instrument');
    const values = [...select.querySelectorAll('option')].map((node) =>
      node.getAttribute('value'),
    );
    // exactly the ids the registry returned, character for character
    expect(values).toEqual(['BTC-USD', 'ETH-USD']);
  });

  it('shows the canonical identifier separately from the display label', async () => {
    renderAt('/markets');
    // label for humans, identifier for machines, not the same string
    expect(await screen.findByText('Bitcoin / US Dollar')).toBeInTheDocument();
    expect(screen.getByText('coinbase:BTC-USD')).toBeInTheDocument();
  });

  it('never reconstructs a business identity from a symbol', () => {
    /* The frontend must not own canonicalisation: two implementations of one
       rule disagree, and the browser's copy would be the wrong one. */
    const modules = import.meta.glob('../{pages,components,layouts,api}/**/*.{ts,tsx}', {
      query: '?raw', import: 'default', eager: true,
    }) as Record<string, string>;
    const offenders: string[] = [];
    for (const [path, source] of Object.entries(modules)) {
      // a symbol or instrument id being case-folded, stripped of separators,
      // or split apart to rebuild an identity
      if (/(symbol|instrument_id|product)\w*\s*\.\s*(toUpperCase|toLowerCase|replace|split|normalize)\(/.test(source)) {
        offenders.push(path);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('treats an unknown instrument as unavailable rather than guessing', async () => {
    renderAt('/markets', { '/api/v1/instruments': new Error('registry down') });
    expect(await screen.findByText('No instruments available')).toBeInTheDocument();
    const select = screen.getByLabelText('Instrument') as HTMLSelectElement;
    expect(select.disabled).toBe(true);
  });

  it('passes the selected id straight back to the API', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    renderAt('/markets');
    const select = await screen.findByLabelText('Instrument');
    await userEvent.selectOptions(select, 'ETH-USD');
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(calls.some((path) => path.includes('/markets/ETH-USD'))).toBe(true);
      // never a spelling the frontend invented
      expect(calls.some((path) => /markets\/(eth-usd|ETHUSD)/.test(path))).toBe(false);
    });
  });
});

describe('portfolio', () => {
  it('is reachable from the sidebar', async () => {
    renderAt('/');
    expect(await screen.findByRole('link', { name: /Portfolio/ })).toBeInTheDocument();
  });

  it('shows the frozen engine limits before any run exists', async () => {
    renderAt('/portfolio');
    expect(await screen.findByText('Portfolio engine')).toBeInTheDocument();
    expect(screen.getByText('READY')).toBeInTheDocument();
    expect(screen.getByText('proportional-gross-cap-v1')).toBeInTheDocument();
    expect(screen.getByText('single-pretrade-equity-batch-v1')).toBeInTheDocument();
    expect(screen.getByText('shared-cash-v1')).toBeInTheDocument();
  });

  it('states the caps as percentages of NAV', async () => {
    renderAt('/portfolio');
    await screen.findByText('Portfolio engine');
    expect(screen.getByText('25.0000 %')).toBeInTheDocument();
    expect(screen.getAllByText('50.0000 %').length).toBe(2);
  });

  it('invents no figures when nothing has been simulated', async () => {
    const { container } = renderAt('/portfolio');
    expect(
      await screen.findByText('No persisted portfolio run available'),
    ).toBeInTheDocument();
    // no equity, return or drawdown may appear before a run exists
    expect(screen.queryByText(/Final equity/)).toBeNull();
    expect(screen.queryByText(/Max drawdown/)).toBeNull();
    expect(container.textContent).not.toMatch(/Sharpe/);
  });

  it('declares shared capital and no confirmed edge', async () => {
    renderAt('/portfolio');
    await screen.findByText('Portfolio engine');
    expect(screen.getByText('SHARED CAPITAL')).toBeInTheDocument();
    expect(screen.getByText('SYNTHETIC EXECUTION COSTS')).toBeInTheDocument();
    expect(screen.getByText('NO CONFIRMED EDGE')).toBeInTheDocument();
  });

  it('says the limits were not optimised', async () => {
    renderAt('/portfolio');
    await screen.findByText('Portfolio engine');
    expect(screen.getByText('Optimized')).toBeInTheDocument();
    expect(screen.getByText(/not a fitted optimum/)).toBeInTheDocument();
  });

  it('stays usable when the portfolio endpoints fail', async () => {
    renderAt('/portfolio', { '/api/v1/portfolio': new Error('portfolio down') });
    expect(await screen.findByText('Could not load')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Overview/ })).toBeInTheDocument();
  });
});

/* --- US equities and real trading sessions (Phase 6D) ---------------------
 *
 * The distinction under test is "described" versus "traded". Four US markets
 * are registered so their identity, provider and sessions can be inspected;
 * none of them has a model, a signal or a paper session, and none may appear
 * anywhere that implies one. */

describe('reference markets', () => {
  it('keeps untradable markets out of the instrument picker', async () => {
    renderAt('/markets');
    const select = await screen.findByLabelText('Instrument');
    const options = [...select.querySelectorAll('option')].map((node) => node.value);
    expect(options).toEqual(['BTC-USD', 'ETH-USD']);
    // An "Equities" heading with nothing under it would read as a failure.
    const groups = [...document.body.querySelectorAll('optgroup')].map((node) =>
      node.getAttribute('label'),
    );
    expect(groups).toEqual(['Crypto']);
  });

  it('lists them in their own section, labelled as not traded', async () => {
    renderAt('/markets');
    expect(await screen.findByText(/Reference markets/)).toBeInTheDocument();
    expect(screen.getByText('NOT TRADED')).toBeInTheDocument();
    expect(screen.getAllByText('REFERENCE ONLY').length).toBeGreaterThan(0);
    expect(
      screen.getByText(/no model, no signal, no paper session/),
    ).toBeInTheDocument();
  });

  it('names the venue, not the provider, as the identity', async () => {
    renderAt('/markets');
    expect(await screen.findByText('xnas:AAPL')).toBeInTheDocument();
    // massive:AAPL is not a thing. The provider is named separately, as the
    // source of the bars rather than as part of the instrument.
    expect(screen.queryByText('massive:AAPL')).toBeNull();
    expect(
      await screen.findByText('Massive (US stocks, historical)'),
    ).toBeInTheDocument();
  });

  it('shows end-of-day freshness so nothing reads it as a live price', async () => {
    renderAt('/markets');
    expect(await screen.findByText('END OF DAY')).toBeInTheDocument();
  });
});

describe('trading sessions', () => {
  it('shows real sessions with the holiday and the weekend absent', async () => {
    renderAt('/markets');
    expect(await screen.findByText('2026-11-23')).toBeInTheDocument();
    expect(await screen.findByText('2026-11-27')).toBeInTheDocument();
    // Thanksgiving and the weekend after it are not rows, because no bar was
    // ever expected on them.
    expect(screen.queryByText('2026-11-26')).toBeNull();
    expect(screen.queryByText('2026-11-28')).toBeNull();
    expect(screen.queryByText('2026-11-29')).toBeNull();
  });

  it('marks the early close and gives it fewer bars', async () => {
    renderAt('/markets');
    expect(await screen.findByText('EARLY CLOSE')).toBeInTheDocument();
    const rows = [...document.body.querySelectorAll('tr')];
    const short = rows.find((row) => row.textContent?.includes('2026-11-27'));
    const full = rows.find((row) => row.textContent?.includes('2026-11-23'));
    expect(short?.textContent).toContain('7');
    expect(full?.textContent).toContain('13');
    // Not padded to match a regular session.
    expect(short?.textContent).not.toContain('13');
  });

  it('says a gap-free calendar is not the same as missing data', async () => {
    renderAt('/markets');
    expect(
      await screen.findByText(/not gaps in the data/),
    ).toBeInTheDocument();
  });

  it('computes no session arithmetic in the browser', async () => {
    /* Every number rendered here came from the API. If the page ever starts
       deriving bar counts itself, this fixture -- whose expected_bars are
       supplied, not derivable from the timestamps by the browser -- is what
       makes the divergence visible. */
    renderAt('/markets');
    await screen.findByText('2026-11-27');
    const rows = [...document.body.querySelectorAll('tr')];
    const short = rows.find((row) => row.textContent?.includes('2026-11-27'));
    expect(short?.textContent).toContain('3h30');
    expect(short?.textContent).toContain('14:30');
    expect(short?.textContent).toContain('18:00');
  });
});

describe('calendars', () => {
  it('shows the equity annualisation from the server, never 8760', async () => {
    renderAt('/markets');
    // 13 bars a session, 3263 periods a year. A crypto constant here would
    // inflate every equity Sharpe ratio by roughly 1.6x.
    expect(await screen.findByText(/13 × 30m per session/)).toBeInTheDocument();
    expect(await screen.findByText(/3,263 periods a year/)).toBeInTheDocument();
  });

  it('names the library and version that produced the schedule', async () => {
    renderAt('/markets');
    expect(
      await screen.findByText(/pandas_market_calendars 5\.4\.0/),
    ).toBeInTheDocument();
    expect(await screen.findByText(/REGULAR session/)).toBeInTheDocument();
  });

  it('shows the crypto calendar as continuous, with its own factor', async () => {
    renderAt('/markets');
    expect(await screen.findByText(/24 × 1h per session/)).toBeInTheDocument();
    expect(await screen.findByText(/8,760 periods a year/)).toBeInTheDocument();
  });
});

describe('registry loading state', () => {
  it('shows a loading state rather than a guessed default', async () => {
    const pending = vi.fn((input: string) =>
      String(input).includes('/instruments')
        ? new Promise<Response>(() => undefined)
        : jsonResponse(fixtures.markets),
    );
    vi.stubGlobal('fetch', pending);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByText('Loading…')).toBeInTheDocument();
  });
});
