/** Appearance, interface and local operations. Nothing that trades.
 *
 *  The backend decides what is settable and says so in the payload; this page
 *  renders that answer rather than carrying its own list. If the two ever
 *  disagreed, the frontend's copy would be the wrong one, and a settings page
 *  that offers a control the server refuses is worse than no control at all.
 *
 *  Appearance is stored locally: it is per-browser, it must apply before any
 *  request completes, and a theme flicker while waiting on HTTP is a worse
 *  experience than a preference that does not follow you between machines.
 *  Operational settings live in the runtime file, where the CLI reads them. */
import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, ErrorState, LoadingState } from '../components/States';
import { applyTheme, readTheme, writeTheme } from '../lib/theme';
import type { ThemePreference } from '../lib/theme';

const SIDEBAR_KEY = 'hyprl.sidebar';
const TIME_KEY = 'hyprl.time_display';

export function SettingsPage() {
  const [theme, setTheme] = useState<ThemePreference>(() => readTheme());
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === 'true',
  );
  const [timeDisplay, setTimeDisplay] = useState(
    () => localStorage.getItem(TIME_KEY) ?? 'utc',
  );
  const { data, status, error, refetch } = useQuery('ops-settings', (signal) =>
    apiClient.getOpsSettings(signal),
  );

  useEffect(() => {
    writeTheme(theme);
    applyTheme(theme);
  }, [theme]);
  useEffect(() => localStorage.setItem(SIDEBAR_KEY, String(collapsed)), [collapsed]);
  useEffect(() => localStorage.setItem(TIME_KEY, timeDisplay), [timeDisplay]);

  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">Appearance</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv">
            <dt>Theme</dt>
            <dd>
              <select
                className="control"
                aria-label="Theme"
                value={theme}
                onChange={(event) => setTheme(event.target.value as ThemePreference)}
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="system">Match system</option>
              </select>
            </dd>
          </div>
          <div className="kv">
            <dt>Sidebar</dt>
            <dd>
              <label>
                <input
                  type="checkbox"
                  checked={collapsed}
                  onChange={(event) => setCollapsed(event.target.checked)}
                />{' '}
                Start collapsed
              </label>
            </dd>
          </div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Interface</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv">
            <dt>Timestamps</dt>
            <dd>
              <select
                className="control"
                aria-label="Timestamps"
                value={timeDisplay}
                onChange={(event) => setTimeDisplay(event.target.value)}
              >
                <option value="utc">UTC</option>
                <option value="local">Local time</option>
              </select>
            </dd>
          </div>
          <div className="kv">
            <dt className="muted">Stored</dt>
            <dd className="muted">In this browser only</dd>
          </div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Local operations</h2>
        {status === 'loading' && <LoadingState label="Loading settings" />}
        {status === 'error' && error && <ErrorState error={error} onRetry={refetch} />}
        {data && (
          <>
            <dl style={{ margin: 0 }}>
              <div className="kv">
                <dt>Default product</dt>
                <dd>{data.current.default_product}</dd>
              </div>
              <div className="kv">
                <dt>Default chart window</dt>
                <dd>{data.current.default_chart_window}</dd>
              </div>
              <div className="kv">
                <dt>Log retention</dt>
                <dd>{data.current.log_retention_preset}</dd>
              </div>
              <div className="kv">
                <dt>Open a browser on start</dt>
                <dd>{data.current.launch_browser ? 'Yes' : 'No'}</dd>
              </div>
              <div className="kv">
                <dt>Shadow auto-start</dt>
                <dd>{data.current.paper_auto_start ? 'Yes' : 'No'}</dd>
              </div>
            </dl>
            <p className="muted">
              These are written by the command line so that a browser cannot change how
              the application runs: <code>./scripts/hyprl.sh settings --set field=value</code>
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Trading contracts</h2>
        <p>
          <Badge tone="warn">IMMUTABLE IN THIS BUILD</Badge>
        </p>
        <p className="muted">
          Signal thresholds, risk caps, fees, slippage, model hyper-parameters, the
          execution policy and the reserved research window are frozen. Their hashes
          appear in committed results, so making one of them a setting would leave every
          recorded result describing software that no longer exists.
        </p>
        {data && (
          <details>
            <summary className="muted">Fields the backend refuses ({data.forbidden_trading_fields.length})</summary>
            <ul className="muted" style={{ columns: 2 }}>
              {data.forbidden_trading_fields.map((field) => (
                <li key={field}>
                  <code>{field}</code>
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>
    </div>
  );
}
