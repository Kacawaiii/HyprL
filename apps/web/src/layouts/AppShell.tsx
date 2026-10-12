/** The cockpit frame: sidebar, topbar, scrolling content.
 *  It stays usable when a request fails -- navigation must never depend on data. */
import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation, useSearchParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { PerfOverlay } from '../components/PerfOverlay';
import { applyTheme, readTheme, writeTheme } from '../lib/theme';
import type { ThemePreference } from '../lib/theme';
import { carrySelection, parseSelection, writeSelection } from '../lib/cockpit';
import type { Mode } from '../lib/cockpit';

// Five entries a beginner needs. The July lab study (BTC/ETH research cockpit) is kept, but only in Expert mode and
// folded under one group so its frozen prices are never mistaken for today's market.
const NAV = [
  { to: '/', label: 'Aujourd’hui', icon: '◉', end: true },
  { to: '/radar', label: 'Actualités', icon: '◌', also: ['/events'] },
  { to: '/paper', label: 'Comptes', icon: '◐' },
  { to: '/trader', label: 'IA', icon: '◎', also: ['/news-value'] },
  { to: '/system', label: 'Système', icon: '⚙' },
];
const ARCHIVES = [
  { to: '/cockpit', label: 'Cockpit', icon: '◉' },
  { to: '/overview', label: 'Overview', icon: '◫' },
  { to: '/markets', label: 'Markets', icon: '◪' },
  { to: '/signals', label: 'Signals', icon: '⌁' },
  { to: '/risk', label: 'Risk', icon: '⚖' },
  { to: '/policies', label: 'Calibration / TP / SL', icon: '◇' },
  { to: '/backtests', label: 'Backtests', icon: '◷' },
  { to: '/portfolio', label: 'Portfolio', icon: '◈' },
  { to: '/research', label: 'Research', icon: '⌕' },
  { to: '/lab', label: 'Lab', icon: '⚗' },
];
const FOOTER = [
  { to: '/api-docs', label: 'API docs' },
  { to: '/settings', label: 'Settings' },
];

export function AppShell() {
  const [open, setOpen] = useState(false);
  // Read once, synchronously: resolving the theme after a request would
  // paint the wrong one first and flash.
  const [theme, setTheme] = useState<ThemePreference>(() => readTheme());
  const [params, setParams] = useSearchParams();
  const { pathname } = useLocation();
  const mode = parseSelection(params).mode;
  const inArchive = ARCHIVES.some((item) => pathname === item.to || pathname.startsWith(`${item.to}/`));
  const [archivesOpen, setArchivesOpen] = useState(inArchive);
  useEffect(() => { if (inArchive) setArchivesOpen(true); }, [inArchive]);
  // The mode, product, period and model ride in the URL; links carry them to every page.
  const carry = carrySelection(params);
  const setMode = (value: Mode) => setParams((current) => writeSelection(current, { mode: value }));
  const health = useQuery('health', (signal) => apiClient.getHealth(signal), {
    staleMs: 10_000,
  });

  useEffect(() => {
    applyTheme(theme);
    writeTheme(theme);
  }, [theme]);

  const state =
    health.status === 'loading' ? 'loading' : health.status === 'error' ? 'error' : 'ok';

  return (
    <div className="shell" data-open={open}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>HyprL</span>
        </div>
        <nav className="nav" aria-label="Sections">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={{ pathname: item.to, search: carry }}
              end={item.end}
              // /events and /news-value are tabs of Actualités and IA: their entry stays lit there.
              className={item.also?.includes(pathname) ? 'nav-link is-current' : 'nav-link'}
              onClick={() => setOpen(false)}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
          {mode === 'expert' && (
            <div className="nav-group">
              <button className="nav-group-toggle" aria-expanded={archivesOpen} onClick={() => setArchivesOpen((value) => !value)}>
                <span className="nav-icon" aria-hidden="true">{archivesOpen ? '▾' : '▸'}</span>
                <span>Archives (labo de juillet)</span>
              </button>
              {archivesOpen && ARCHIVES.map((item) => (
                <NavLink key={item.to} to={{ pathname: item.to, search: carry }} className="nav-link nav-link-sub"
                  onClick={() => setOpen(false)}>
                  <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </div>
          )}
        </nav>
        <div className="sidebar-footer">
          <div>Comptes papier, pas d’argent réel</div>
          <nav className="footer-links" aria-label="Outils">
            {FOOTER.map((item) => (
              <NavLink key={item.to} to={{ pathname: item.to, search: carry }} onClick={() => setOpen(false)}>{item.label}</NavLink>
            ))}
          </nav>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <button
            className="control"
            onClick={() => setOpen((value) => !value)}
            aria-label="Toggle navigation"
            aria-expanded={open}
          >
            ☰
          </button>
          <div className="topbar-spacer" />
          <div className="mode-switch" role="group" aria-label="Cockpit mode">
            {(['beginner', 'expert'] as const).map((value) => (
              <button
                key={value}
                className="control"
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
              >
                {value === 'beginner' ? 'Beginner' : 'Expert'}
              </button>
            ))}
          </div>
          <span className="status-pill">
            <span className="status-dot" data-state={state} aria-hidden="true" />
            {state === 'ok' ? `API ${health.data?.core_status ?? ''}` : `API ${state}`}
          </span>
          <button
            className="control"
            onClick={() => setTheme((value) => (value === 'light' ? 'dark' : 'light'))}
            aria-label="Toggle colour theme"
          >
            {theme === 'light' ? '☀' : '☾'}
          </button>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
      <PerfOverlay />
    </div>
  );
}
