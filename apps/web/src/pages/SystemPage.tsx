import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, CapabilityBadge, ErrorState, Hash, LoadingState } from '../components/States';
import { ProductError } from '../components/ProductError';
import type { HealthState } from '../api/types';

/** Bytes for people. The exact figure is never the point on this page; the
 *  order of magnitude is. */
function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  if (value < 1024) return `${value} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[index]}`;
}

function duration(seconds: number | null | undefined): string {
  if (!seconds || seconds < 0) return '—';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m`;
  return `${Math.floor(seconds)}s`;
}

const TONE: Record<HealthState, 'ok' | 'warn' | 'off'> = {
  HEALTHY: 'ok',
  DEGRADED: 'warn',
  // An active embargo is the guard working. Amber, never red.
  EMBARGOED: 'warn',
  STOPPED: 'off',
  ERROR: 'off',
};

function StateBadge({ state }: { state: HealthState }) {
  return <Badge tone={TONE[state] ?? 'off'}>{state}</Badge>;
}

export function SystemPage() {
  const { data, status, error, refetch } = useQuery('system', (signal) =>
    apiClient.getSystem(signal),
  );
  // Independent queries: the operations views must render even when the
  // research artefacts are missing, and vice versa.
  const runtime = useQuery('ops-runtime', (signal) => apiClient.getOpsRuntime(signal), {
    staleMs: 5_000,
  });
  const recovery = useQuery('ops-recovery', (signal) => apiClient.getOpsRecovery(signal));
  const storage = useQuery('ops-storage', (signal) => apiClient.getOpsStorage(signal));
  const health = useQuery('ops-health', (signal) =>
    apiClient.getHealthHistory(undefined, 25, signal),
  );

  if (status === 'loading') return <LoadingState label="Loading system" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">Application</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>API</dt><dd>{data.api_version}</dd></div>
          <div className="kv">
            <dt>App</dt>
            <dd>
              {runtime.data ? (
                <StateBadge state={(runtime.data.app.state === 'RUNNING' ? 'HEALTHY' : 'STOPPED') as HealthState} />
              ) : (
                '—'
              )}
            </dd>
          </div>
          <div className="kv">
            <dt>Uptime</dt>
            <dd>{duration(runtime.data?.app.uptime_seconds)}</dd>
          </div>
          <div className="kv">
            <dt>Memory</dt>
            <dd>{bytes(runtime.data?.app.rss_bytes)}</dd>
          </div>
          <div className="kv">
            <dt>Real money</dt>
            <dd><Badge tone="off">NO</Badge></dd>
          </div>
          <div className="kv">
            <dt>Broker</dt>
            <dd><Badge tone="off">NOT CONNECTED</Badge></dd>
          </div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Startup</h2>
        {recovery.status === 'loading' && <LoadingState label="Checking runtime" />}
        {recovery.data && (
          <>
            <p>
              {recovery.data.last_shutdown_clean === null ? (
                <Badge tone="ok">FIRST RUN</Badge>
              ) : recovery.data.recovery_performed ? (
                /* Successful recovery is information, not an alarm. The system
                   did exactly what it was built to do. */
                <Badge tone="warn">RECOVERED AFTER UNCLEAN SHUTDOWN</Badge>
              ) : (
                <Badge tone="ok">HEALTHY STARTUP</Badge>
              )}
            </p>
            <dl style={{ margin: 0 }}>
              <div className="kv">
                <dt>Event chain</dt>
                <dd>
                  {recovery.data.event_chain_verified === null ? (
                    <span className="muted">no session recorded</span>
                  ) : (
                    <Badge tone={recovery.data.event_chain_verified ? 'ok' : 'off'}>
                      {recovery.data.event_chain_verified ? 'VERIFIED' : 'INVALID'}
                    </Badge>
                  )}
                </dd>
              </div>
              <div className="kv">
                <dt>Latest snapshot</dt>
                <dd>
                  {recovery.data.latest_snapshot_verified === null ? (
                    <span className="muted">none written yet</span>
                  ) : (
                    <Badge tone={recovery.data.latest_snapshot_verified ? 'ok' : 'off'}>
                      {recovery.data.latest_snapshot_verified ? 'VERIFIED' : 'INVALID'}
                    </Badge>
                  )}
                </dd>
              </div>
              <div className="kv"><dt>Events</dt><dd>{recovery.data.events.toLocaleString()}</dd></div>
              <div className="kv"><dt>Sessions</dt><dd>{recovery.data.sessions}</dd></div>
            </dl>
            {recovery.data.error_code && (
              <ProductError code={recovery.data.error_code} component="event_store" />
            )}
          </>
        )}
      </section>

      <section className="grid grid-2">
        <article className="card">
          <h2 className="card-title">Signal engine</h2>
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Protocol</dt><dd>{data.signal_engine.protocol}</dd></div>
            <div className="kv"><dt>Rule</dt><dd>{data.signal_engine.rule}</dd></div>
            <div className="kv"><dt>Frozen</dt><dd><Badge tone="ok">YES</Badge></dd></div>
            <div className="kv"><dt>Spec hash</dt><dd><Hash value={data.signal_engine.spec_hash} chars={20} /></dd></div>
          </dl>
        </article>
        <article className="card">
          <h2 className="card-title">Risk engine</h2>
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Protocol</dt><dd>{data.risk_engine.protocol}</dd></div>
            <div className="kv"><dt>Frozen</dt><dd><Badge tone="ok">YES</Badge></dd></div>
            <div className="kv"><dt>Spec hash</dt><dd><Hash value={data.risk_engine.spec_hash} chars={20} /></dd></div>
          </dl>
        </article>
      </section>

      <section className="card">
        <h2 className="card-title">Research protection</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv">
            <dt>V2 confirmatory holdout</dt>
            <dd><Badge tone="warn">UNOBSERVED</Badge></dd>
          </div>
          {data.benchmarks.confirmatory_holdout && (
            <div className="kv">
              <dt>Reserved window</dt>
              <dd>
                {data.benchmarks.confirmatory_holdout.range_start.slice(0, 10)} →{' '}
                {data.benchmarks.confirmatory_holdout.range_end.slice(0, 10)}
              </dd>
            </div>
          )}
          <div className="kv"><dt>Enforced</dt><dd><Badge tone="ok">YES</Badge></dd></div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Market corpus</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Corpus</dt><dd>{data.market_data.corpus_id ?? '—'}</dd></div>
          <div className="kv"><dt>Products</dt><dd>{data.market_data.products.join(', ')}</dd></div>
          <div className="kv"><dt>Timeframe</dt><dd>{data.market_data.timeframe}</dd></div>
          <div className="kv"><dt>Spec hash</dt><dd><Hash value={data.market_data.corpus_spec_hash} chars={20} /></dd></div>
          <div className="kv"><dt>Content hash</dt><dd><Hash value={data.market_data.corpus_content_hash} chars={20} /></dd></div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Runtime storage</h2>
        {storage.status === 'loading' && <LoadingState label="Measuring storage" />}
        {storage.data && (
          <>
            <dl style={{ margin: 0 }}>
              <div className="kv"><dt>Event log</dt><dd>{bytes(storage.data.paper_database_bytes)}</dd></div>
              <div className="kv"><dt>Events</dt><dd>{storage.data.events.toLocaleString()}</dd></div>
              <div className="kv"><dt>State snapshots</dt><dd>{storage.data.snapshots}</dd></div>
              <div className="kv"><dt>Operations database</dt><dd>{bytes(storage.data.ops_database_bytes)}</dd></div>
              <div className="kv">
                <dt>Logs</dt>
                <dd>{bytes(storage.data.log_bytes)} <span className="muted">of {bytes(storage.data.log_cap_bytes)} cap</span></dd>
              </div>
              <div className="kv"><dt>Exports</dt><dd>{bytes(storage.data.export_bytes)}</dd></div>
            </dl>
            <p className="muted">{storage.data.paper_events_retention}</p>
            {storage.data.database_warning && (
              <p className="muted">{storage.data.database_warning}</p>
            )}
          </>
        )}
      </section>

      {runtime.data && (
        <section className="card">
          <h2 className="card-title">Snapshots</h2>
          <dl style={{ margin: 0 }}>
            {Object.entries(runtime.data.snapshots.products).map(([product, item]) => (
              <div className="kv" key={product}>
                <dt>{product}</dt>
                <dd>
                  {item.events_since_last_snapshot.toLocaleString()} events since last
                  {item.snapshot_due ? ' · due' : ''}
                </dd>
              </div>
            ))}
            <div className="kv">
              <dt>Cadence</dt>
              <dd><StateBadge state={runtime.data.snapshots.status} /></dd>
            </div>
          </dl>
          {runtime.data.snapshots.error_code && (
            <ProductError code={runtime.data.snapshots.error_code} component="event_store" />
          )}
        </section>
      )}

      <section className="card">
        <h2 className="card-title">Health</h2>
        {health.status === 'loading' && <LoadingState label="Loading health" />}
        {health.data && !health.data.available && (
          <p className="muted">No health has been recorded on this machine yet.</p>
        )}
        {health.data?.latest && (
          <dl style={{ margin: 0 }}>
            {Object.entries(health.data.latest).map(([component, item]) => (
              <div className="kv" key={component}>
                <dt>{component.replace(/_/g, ' ')}</dt>
                <dd>
                  <StateBadge state={item.status} />{' '}
                  {item.error_code && <span className="muted">{item.error_code}</span>}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Capabilities</h2>
        <dl style={{ margin: 0 }}>
          {Object.entries(data.capabilities).map(([name, enabled]) => (
            <div className="kv" key={name}>
              <dt>{name.replace(/_/g, ' ')}</dt>
              <dd><CapabilityBadge enabled={enabled} /></dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}
