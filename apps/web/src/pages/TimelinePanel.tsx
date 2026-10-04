/** One read at T, laid out in the server's availability order. Source timestamps stay provenance. */
import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, ErrorState, LoadingState } from '../components/States';
import { TimelineReadForm } from '../components/ReadForm';
import type { TimelineRead } from '../components/ReadForm';

function TimelineSnapshot({ read }: { read: TimelineRead }) {
  const timeline = useQuery(`timeline|${read.asOf}|${read.fomcHorizon ?? ''}|${read.edgarHorizon ?? ''}`, (signal) =>
    apiClient.getTimeline(read.asOf, read.fomcHorizon, read.edgarHorizon, signal));
  if (timeline.status === 'loading') return <LoadingState label="Reading official events" />;
  if (timeline.status === 'error' && timeline.error) return <ErrorState error={timeline.error} onRetry={timeline.refetch} />;
  const data = timeline.data;
  if (!data) return null;
  const complete = data.sources.fomc.read_state === 'FOMC_RESOLVED' && data.sources.edgar.read_state === 'EDGAR_RESOLVED';
  return (
    <>
      <section className="card" aria-label="Timeline read states">
        <h2 className="card-title">Source read states</h2>
        {Object.entries(data.sources).map(([source, state]) => {
          const resolved = state.read_state === 'FOMC_RESOLVED' || state.read_state === 'EDGAR_RESOLVED';
          return (
            <div key={source} role="status" aria-label={`${source} read state`}>
              <Badge tone={resolved ? 'ok' : 'warn'}>{source}</Badge> <Badge tone={resolved ? 'ok' : 'warn'}>{state.read_state}</Badge>
              {!resolved && <p>Nothing is confirmed for this source at this read.</p>}
              {state.reason && <p>{state.reason}</p>}
              <dl>
                <div className="kv"><dt>Horizon H · prefix P</dt><dd>{state.H ?? '—'} · {state.P ?? '—'}</dd></div>
                <div className="kv"><dt>Source snapshot identity</dt><dd><code>{state.identity ?? '—'}</code></dd></div>
              </dl>
            </div>
          );
        })}
      </section>
      <section className="card" aria-label="Timeline">
        <h2 className="card-title">Official events as of T ({data.rows.length})</h2>
        <dl>
          <div className="kv"><dt>As of (T)</dt><dd>{data.T}</dd></div>
          <div className="kv"><dt>Timeline identity</dt><dd><code data-testid="timeline-identity">{data.identity}</code></dd></div>
        </dl>
        {!complete && <p role="status">Partial timeline: rows are shown only for resolved sources. Check each source read state above.</p>}
        {data.rows.length === 0 ? (
          <p>{complete ? 'No timeline event was known at this instant.' : 'No rows can be shown for this read; unavailable sources are not confirmed empty.'}</p>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th>Source</th><th>Item id</th><th>Title / form</th><th>State</th><th>Revision</th><th>Content identity</th><th>Available at</th><th>Declared release at (provenance)</th><th>Declared release text (provenance)</th><th>Acceptance (provenance)</th></tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={`${row.source}:${row.id}`}>
                    <td><Badge tone="ok">{row.source}</Badge></td><td>{row.id}</td><td>{row.title ?? row.form ?? '—'}</td>
                    <td>{row.state}</td><td><code>{row.revision}</code></td><td><code>{row.content_identity}</code></td>
                    <td>{row.available_at}</td><td>{row.provenance.declared_release_at ?? '—'}</td>
                    <td>{row.provenance.declared_release_text ?? '—'}</td><td>{row.provenance.acceptance_datetime_text ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

export function TimelinePanel() {
  const fomc = useQuery('fomc-status', (signal) => apiClient.getFomcStatus(signal));
  const edgar = useQuery('edgar-status', (signal) => apiClient.getEdgarStatus(signal));
  const [read, setRead] = useState<TimelineRead | null>(null);
  // Choose a suggested input from the configured sources; only the timeline response decides visibility.
  const suggestions = [fomc.data?.suggested_as_of, edgar.data?.suggested_as_of].filter((value): value is string => !!value);
  suggestions.sort((a, b) => Date.parse(a) - Date.parse(b));
  const suggested = suggestions[0] ?? '';
  const loading = fomc.status === 'loading' || edgar.status === 'loading';
  useEffect(() => {
    if (!loading && suggested && read === null) setRead({ asOf: suggested });
  }, [loading, suggested, read]);
  if (loading) return <LoadingState label="Loading event sources" />;
  return (
    <div className="stack">
      {fomc.status === 'error' && fomc.error && <ErrorState error={fomc.error} onRetry={fomc.refetch} />}
      {edgar.status === 'error' && edgar.error && <ErrorState error={edgar.error} onRetry={edgar.refetch} />}
      {!read && <p>Choose an instant to read both sources, including their unconfigured or refused states.</p>}
      <TimelineReadForm initialAsOf={suggested} onRead={setRead} />
      {read && <TimelineSnapshot key={`${read.asOf}|${read.fomcHorizon ?? ''}|${read.edgarHorizon ?? ''}`} read={read} />}
    </div>
  );
}
