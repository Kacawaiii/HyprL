/**
 * The Markets view of the local equity research corpus.
 *
 * These tests are mostly about what the UI must NOT say. A daily snapshot from
 * an unofficial source, charted next to a live crypto panel, is exactly the
 * screen where "LIVE" or "SPLIT-ADJUSTED" would slip in and be believed --
 * so the absence of those words is asserted as strictly as the presence of the
 * honest ones.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
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
    '/api/v1/system': fixtures.system,
    '/api/v1/overview': fixtures.overview,
    '/api/v1/markets': fixtures.markets,
    '/api/v1/instruments': fixtures.instruments,
    '/api/v1/providers': fixtures.providers,
    '/api/v1/calendars': fixtures.calendars,
    '/api/v1/research/equities/corpus': fixtures.researchCorpusAvailable,
    ...overrides,
  };
  return vi.fn((input: string) => {
    const path = String(input);
    if (path.includes('/research/equities/') && path.includes('/bars')) {
      return jsonResponse(fixtures.researchBars);
    }
    if (path.includes('/chart')) return jsonResponse(fixtures.chart);
    if (/\/api\/v1\/markets\/[^/?]+(\?|$)/.test(path)) {
      return jsonResponse(fixtures.candlePage);
    }
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

function renderMarkets(overrides?: Record<string, unknown>) {
  vi.stubGlobal('fetch', mockApi(overrides));
  return render(
    <MemoryRouter initialEntries={['/markets']}>
      <App />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  invalidate();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Words this corpus has not earned, checked over the whole rendered page. */
const FORBIDDEN = [
  /\bLIVE\b/, /\bREALTIME\b/, /\bREAL-TIME\b/, /\bOFFICIAL\b/,
  /\bLICENSED\b/, /SPLIT[- ]ADJUSTED/i, /TOTAL RETURN/i,
];

describe('local research corpus panel', () => {
  it('renders the corpus with its exact labels', async () => {
    renderMarkets();
    expect(await screen.findByText('LOCAL RESEARCH CORPUS')).toBeInTheDocument();
    expect(screen.getByText('HISTORICAL')).toBeInTheDocument();
    expect(screen.getByText('RAW')).toBeInTheDocument();
    expect(screen.getByText('REGULAR SESSION')).toBeInTheDocument();
    expect(screen.getByText('UNOFFICIAL SOURCE')).toBeInTheDocument();
  });

  it('states the timeframe as 1D and never as 30m', async () => {
    renderMarkets();
    expect(await screen.findByText('1D')).toBeInTheDocument();
    const body = document.body.textContent ?? '';
    expect(body).not.toMatch(/\b30m\b/);
    expect(body).not.toMatch(/30-minute/);
  });

  it('carries the source warning in plain words', async () => {
    renderMarkets();
    expect(
      await screen.findByText(/Unofficial research source\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/Local historical snapshot\./)).toBeInTheDocument();
    expect(screen.getByText(/Not a live market feed\./)).toBeInTheDocument();
  });

  it('claims nothing the corpus cannot support', async () => {
    renderMarkets();
    await screen.findByText('LOCAL RESEARCH CORPUS');
    const body = document.body.textContent ?? '';
    for (const pattern of FORBIDDEN) {
      // "Not a live market feed" is a denial, so the word may appear only in
      // lowercase prose; the badges and headings must never carry the claim.
      const badges = [...document.querySelectorAll('.badge')]
        .map((node) => node.textContent ?? '').join(' ');
      expect(badges).not.toMatch(pattern);
    }
    expect(body).not.toMatch(/SPLIT[- ]ADJUSTED/i);
    expect(body).not.toMatch(/TOTAL RETURN/i);
  });

  it('draws a chart from the local API', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByLabelText('Research instrument');
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(
        calls.some((path) => path.includes('/research/equities/') &&
          path.includes('/bars')),
      ).toBe(true);
    });
    expect(await screen.findByText(/daily sessions/)).toBeInTheDocument();
  });

  it('asks for a bounded window, never for everything', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByLabelText('Research instrument');
    await waitFor(() => {
      const bars = fetchMock.mock.calls
        .map((call) => String(call[0]))
        .filter((path) => path.includes('/bars'));
      expect(bars.length).toBeGreaterThan(0);
      for (const path of bars) {
        const limit = new URL(path, 'http://local').searchParams.get('limit');
        expect(limit).not.toBeNull();
        expect(Number(limit)).toBeLessThanOrEqual(1000);
      }
    });
  });

  it('lets the reader switch instrument within the corpus', async () => {
    const fetchMock = mockApi();
    vi.stubGlobal('fetch', fetchMock);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    const select = await screen.findByLabelText('Research instrument');
    const options = [...select.querySelectorAll('option')].map(
      (node) => node.getAttribute('value'));
    expect(options).toEqual(['xnas:AAPL', 'xnas:QQQ']);
    await userEvent.selectOptions(select, 'xnas:QQQ');
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(calls.some((path) => path.includes('xnas%3AQQQ'))).toBe(true);
    });
  });

  it('shows equities and ETFs without marking them tradable', async () => {
    renderMarkets();
    await screen.findByText('LOCAL RESEARCH CORPUS');
    expect(screen.getByText('NOT TRADED')).toBeInTheDocument();
    // The tradable picker still offers only the two crypto markets.
    const tradableSelect = await screen.findByLabelText('Instrument');
    const options = [...tradableSelect.querySelectorAll('option')].map(
      (node) => node.textContent);
    expect(options).toEqual(['BTC-USD', 'ETH-USD']);
  });

  it('offers no download, capture or repair action', async () => {
    renderMarkets();
    await screen.findByText('LOCAL RESEARCH CORPUS');
    for (const label of [/download/i, /capture/i, /fetch/i, /refresh corpus/i,
                         /install corpus/i, /repair/i]) {
      expect(screen.queryByRole('button', { name: label })).toBeNull();
    }
    const body = document.body.textContent ?? '';
    expect(body).not.toMatch(/download the corpus/i);
  });

  it('links to no raw provider payload', async () => {
    renderMarkets();
    await screen.findByText('LOCAL RESEARCH CORPUS');
    const links = [...document.querySelectorAll('a')].map(
      (node) => node.getAttribute('href') ?? '');
    for (const href of links) {
      expect(href).not.toMatch(/raw/);
      expect(href).not.toMatch(/yahoo/i);
      expect(href).not.toMatch(/finance\.yahoo\.com/);
    }
  });
});

describe('local research corpus absent', () => {
  it('says the corpus is not on this machine, and blames nothing', async () => {
    renderMarkets({
      '/api/v1/research/equities/corpus': fixtures.researchCorpusAbsent,
    });
    expect(
      await screen.findByText(
        'Local Yahoo research corpus not available on this machine.'),
    ).toBeInTheDocument();
    const body = document.body.textContent ?? '';
    // None of the words that would imply a failed network attempt.
    expect(body).not.toMatch(/error loading market/i);
    expect(body).not.toMatch(/download failed/i);
    expect(body).not.toMatch(/provider offline/i);
    expect(body).not.toMatch(/could not connect/i);
    expect(body).not.toMatch(/timed out/i);
  });

  it('requests no bars when there is no corpus', async () => {
    const fetchMock = mockApi({
      '/api/v1/research/equities/corpus': fixtures.researchCorpusAbsent,
    });
    vi.stubGlobal('fetch', fetchMock);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByText(
      'Local Yahoo research corpus not available on this machine.');
    const calls = fetchMock.mock.calls.map((call) => String(call[0]));
    expect(calls.some((path) => path.includes('/bars'))).toBe(false);
  });

  it('offers no button that would start a download', async () => {
    renderMarkets({
      '/api/v1/research/equities/corpus': fixtures.researchCorpusAbsent,
    });
    await screen.findByText(
      'Local Yahoo research corpus not available on this machine.');
    expect(screen.queryByRole('button', { name: /download/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /install/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /capture/i })).toBeNull();
  });
});

describe('local research corpus invalid', () => {
  it('renders a distinct invalid state and no chart', async () => {
    renderMarkets({
      '/api/v1/research/equities/corpus': fixtures.researchCorpusCorrupt,
    });
    expect(await screen.findByText('LOCAL CORPUS INVALID')).toBeInTheDocument();
    expect(screen.queryByText(/daily sessions/)).toBeNull();
    expect(screen.queryByLabelText('Research instrument')).toBeNull();
  });

  it('never falls back to the source', async () => {
    const fetchMock = mockApi({
      '/api/v1/research/equities/corpus': fixtures.researchCorpusCorrupt,
    });
    vi.stubGlobal('fetch', fetchMock);
    render(
      <MemoryRouter initialEntries={['/markets']}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByText('LOCAL CORPUS INVALID');
    const calls = fetchMock.mock.calls.map((call) => String(call[0]));
    expect(calls.some((path) => path.includes('/bars'))).toBe(false);
    expect(calls.some((path) => path.includes('yahoo'))).toBe(false);
  });

  it('names the atomicity rule rather than showing three good instruments',
    async () => {
      renderMarkets({
        '/api/v1/research/equities/corpus': fixtures.researchCorpusCorrupt,
      });
      await screen.findByText('LOCAL CORPUS INVALID');
      // Stated in the panel's own prose and echoed in the server's reason.
      expect(screen.getAllByText(/atomic/).length).toBeGreaterThan(0);
      // The three intact instruments are shown as diagnostics, never as a
      // usable subset: no chart, no picker.
      expect(screen.queryByText(/daily sessions/)).toBeNull();
    });
});

describe('no corpus data ships in the bundle', () => {
  it('embeds no canonical row in any module', async () => {
    /* The frontend fetches a window at runtime. A fixture is test-only, but a
       price literal in src/ would end up in the production bundle. */
    const modules = import.meta.glob('../{components,pages,api,state}/**/*.{ts,tsx}', {
      query: '?raw', import: 'default', eager: true,
    }) as Record<string, string>;
    for (const [name, source] of Object.entries(modules)) {
      expect(source, name).not.toMatch(/224\.3699951171875/);
      expect(source, name).not.toMatch(/xnas_AAPL\.jsonl/);
      expect(source, name).not.toMatch(/"bar_open_at":\s*"20/);
    }
  });
});
