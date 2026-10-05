/** Operations health: versions, last operations, freshness, errors, budgets, workers, resources.
 *
 *  Source: GET /api/v1/ops/health (hyprl-ops-health-v1), read-only. The panel reports what the server
 *  observed. A missing value is shown as "not observed", never as zero, and archive age is never
 *  presented as a live-feed guarantee. */
import { ApiError, apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, ErrorState, Hash, LoadingState } from '../components/States';
import { bytes, duration } from '../lib/ops';
import type { OpsHealth, OpsRunningVersions } from '../api/types';

type Tone = 'ok' | 'warn' | 'off';

function tone(state: string | null | undefined): Tone {
  switch (state) {
    case 'OBSERVED': case 'RUNNING': case 'AVAILABLE': case 'COMPLETE': return 'ok';
    case 'NOT_OBSERVED': case 'NOT_CONFIGURED': case 'STOPPED': case 'UNKNOWN': return 'off';
    default: return 'warn';
  }
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="kv"><dt>{label}</dt><dd>{children}</dd></div>;
}

function Versions({ versions }: { versions: OpsRunningVersions | null }) {
  if (!versions) return <span className="muted">not observed</span>;
  return (
    <>
      <Hash value={versions.git_sha} chars={10} />{' '}
      {Object.entries(versions.specs).map(([name, spec]) => (
        <span key={name} className="muted">
          {name} rev {spec.revision ?? '—'} <Hash value={spec.hash} chars={8} />{' '}
        </span>
      ))}
    </>
  );
}

function Panel({ health }: { health: OpsHealth }) {
  const { workers, edgar_service: edgar, resources } = health;
  const budgets = workers.budgets;
  return (
    <>
      <p data-testid="ops-status">
        <Badge tone={health.status === 'OBSERVED' ? 'ok' : 'warn'}>{health.status}</Badge>{' '}
        <span className="muted">observed {health.observed_at}</span>
      </p>

      <h3>Versions</h3>
      <dl style={{ margin: 0 }}>
        <Row label="API">{health.running_versions.api ?? '—'}</Row>
        <Row label="Running code"><Versions versions={health.running_versions} /></Row>
        {Object.entries(health.services).map(([name, service]) => (
          <Row key={name} label={`${name} service`}>
            <Badge tone={tone(service.state)}>{service.state}</Badge>{' '}
            <Versions versions={service.running_versions} />
          </Row>
        ))}
      </dl>

      <h3>Freshness</h3>
      <dl style={{ margin: 0 }}>
        {Object.entries(health.sources).map(([name, source]) => (
          <Row key={name} label={`${name} source`}>
            <Badge tone={tone(source.state)}>{source.state}</Badge>{' '}
            {source.age_seconds != null && (
              <span className="muted">archive attested {duration(source.age_seconds)} ago</span>
            )}
            {source.read_state && <span className="muted"> · read {source.read_state}</span>}
          </Row>
        ))}
        <Row label="EDGAR capture">
          <Badge tone={tone(edgar.state)}>{edgar.state}</Badge>{' '}
          {edgar.freshness_limit_seconds != null && edgar.age_seconds != null && (
            <span className="muted">
              status {duration(edgar.age_seconds)} old, limit {edgar.freshness_limit_seconds}s
            </span>
          )}
        </Row>
      </dl>
      <p className="muted">
        Archive age is an attestation age. It is not a guarantee of a live feed; no request to an
        official source is made to produce this panel.
      </p>

      <h3>Errors</h3>
      {health.errors.length === 0 ? (
        <p><Badge tone="ok">NONE REPORTED</Badge></p>
      ) : (
        <ul aria-label="Reported errors">
          {health.errors.map((code) => <li key={code}><code>{code}</code></li>)}
        </ul>
      )}

      <h3>Last operations</h3>
      {health.last_operations.length === 0 ? (
        <p className="muted">No operation recorded.</p>
      ) : (
        <table className="data" aria-label="Last operations">
          <thead><tr><th>At</th><th>Action</th><th>Service</th><th>State</th><th>Code</th></tr></thead>
          <tbody>
            {[...health.last_operations].reverse().map((op, index) => (
              <tr key={`${op.at}-${index}`}>
                <td>{op.at ?? '—'}</td>
                <td>{op.action}</td>
                <td>{op.service}</td>
                <td><Badge tone={op.state === 'COMPLETE' ? 'ok' : op.state === 'BLOCKED' ? 'warn' : 'off'}>{op.state}</Badge></td>
                <td>{op.code ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Workers and budgets</h3>
      <dl style={{ margin: 0 }}>
        <Row label="Model Lab workers"><Badge tone={tone(workers.state)}>{workers.state}</Badge></Row>
        {workers.workers.map((worker, index) => (
          <Row key={index} label={`Worker ${worker.pid ?? 'pid unknown'}`}>
            {Math.round(worker.progress * 100)}%
          </Row>
        ))}
        {budgets ? (
          <>
            <Row label="Jobs">{budgets.jobs_used} of {budgets.jobs_limit}</Row>
            <Row label="Queue">{budgets.queue_used} of {budgets.queue_limit}</Row>
            <Row label="Artifacts">{bytes(budgets.artifact_bytes_used)} of {bytes(budgets.artifact_bytes_limit)}</Row>
          </>
        ) : (
          <Row label="Job budgets"><span className="muted">not observed</span></Row>
        )}
        {edgar.budgets ? (
          <>
            <Row label="EDGAR requests">
              {edgar.budgets.requests_used} of {edgar.budgets.requests_limit}
              {edgar.budgets.expired && <> <Badge tone="warn">GRANT EXPIRED</Badge></>}
              {edgar.budgets.terminated && <> <Badge tone="warn">TERMINATED</Badge></>}
            </Row>
            <Row label="Grant expiry">{edgar.budgets.not_after}</Row>
          </>
        ) : (
          <Row label="EDGAR budget"><span className="muted">not observed</span></Row>
        )}
      </dl>
      {workers.errors && workers.errors.length > 0 && (
        <p className="muted">Recent job error codes: {workers.errors.join(', ')}</p>
      )}

      <h3>Resources</h3>
      <dl style={{ margin: 0 }}>
        <Row label="API memory">{bytes(resources.api_rss_bytes)}</Row>
        <Row label="API CPU time">{duration(resources.api_cpu_seconds)}</Row>
        <Row label="API threads">{resources.api_threads}</Row>
        <Row label="Runtime disk free">{bytes(resources.runtime_disk_free_bytes)}</Row>
      </dl>
      <p className="muted">Scope: {resources.scope}.</p>

      <h3>Limits of this panel</h3>
      <ul>{health.limitations.map((text) => <li key={text}>{text}</li>)}</ul>
    </>
  );
}

export function OpsHealthPanel() {
  const { data, status, error, refetch } = useQuery('ops-health-v1', (signal) =>
    apiClient.getOpsHealth(signal), { staleMs: 5_000 });
  return (
    <section className="card" aria-label="Operations health">
      <h2 className="card-title">Operations health</h2>
      {status === 'loading' && <LoadingState label="Loading operations health" />}
      {status === 'error' && error && (
        error instanceof ApiError && error.status === 404 ? (
          <p className="muted">This server does not publish operations health.</p>
        ) : (
          <ErrorState error={error} onRetry={refetch} />
        )
      )}
      {data && <Panel health={data} />}
    </section>
  );
}
