/** The cockpit frame: sidebar, topbar, scrolling content.
 *  It stays usable when a request fails -- navigation must never depend on data. */
import { useEffect, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { PerfOverlay } from '../components/PerfOverlay';

const NAV = [
  { to: '/', label: 'Overview', icon: '◫', end: true },
  { to: '/markets', label: 'Markets', icon: '◪' },
  { to: '/signals', label: 'Signals', icon: '⌁' },
  { to: '/risk', label: 'Risk', icon: '⚖' },
  { to: '/research', label: 'Research', icon: '⌕' },
  { to: '/system', label: 'System', icon: '⚙' },
];

export function AppShell() {
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const health = useQuery('health', (signal) => apiClient.getHealth(signal), {
    staleMs: 10_000,
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
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
              to={item.to}
              end={item.end}
              className="nav-link"
              onClick={() => setOpen(false)}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div>Research cockpit</div>
          <div>No live trading</div>
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
          <span className="status-pill">
            <span className="status-dot" data-state={state} aria-hidden="true" />
            {state === 'ok' ? `API ${health.data?.core_status ?? ''}` : `API ${state}`}
          </span>
          <button
            className="control"
            onClick={() => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))}
            aria-label="Toggle colour theme"
          >
            {theme === 'dark' ? '☾' : '☀'}
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
