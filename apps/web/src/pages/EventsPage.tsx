import { SourceTimeline } from '../components/SourceTimeline';
import { SourcePager } from '../components/SourcePager';
import { useSourcePage } from '../state/useSourcePage';
/** Official event sources: what the FOMC store knew at an instant.
 *
 *  Every value here is a durable record or a derivation the Python reader already performs
 *  (events_as_of, verified replay, source health) and ships verbatim. The page chooses the read --
 *  an instant and, optionally, a commit horizon -- and lays the answer out; it never decides what
 *  was available. An unresolved read shows nothing as of that instant, never a guess. */
import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { ReadForm, readKey } from '../components/ReadForm';
import type { Read } from '../components/ReadForm';
import { EdgarPanel } from './EdgarPanel';
import type { FomcItemSummary, FomcSourceStatus } from '../api/types';

function StoreCard({ status }: { status: FomcSourceStatus }) {
  return (
    <section className="card">
      <h2 className="card-title">FOMC store</h2>
      <dl style={{ margin: 0 }}>
        <div className="kv"><dt>Status</dt><dd><Badge tone="ok">{status.status}</Badge> read-only</dd></div>
        <div className="kv"><dt>Store</dt><dd>{status.store}</dd></div>
        <div className="kv"><dt>Spec</dt><dd>revision {status.spec_revision} <Hash value={status.spec_hash} chars={16} /></dd></div>
        <div className="kv"><dt>Schema</dt><dd>{status.schema_version}</dd></div>
        <div className="kv"><dt>Horizon</dt><dd>{status.horizon}</dd></div>
        <div className="kv"><dt>Durable activity</dt><dd>{status.first_durable_activity} → {status.last_durable_activity}</dd></div>
        <div className="kv"><dt>Server-attested now (lower bound)</dt><dd>{status.suggested_as_of ?? '—'}</dd></div>
        {status.counts && (
          <div className="kv">
            <dt>Records</dt>
            <dd>
              {status.counts.responses} responses · {status.counts.revisions} revisions ·{' '}
              {status.counts.observations} observations · {status.counts.cycles} cycles · {status.counts.epochs} epochs
            </dd>
          </div>
        )}
      </dl>
    </section>
  );
}

function ItemDetail({ sid, read }: { sid: string; read: Read }) {
  const detail = useSourcePage((signal, page, horizon) =>
    apiClient.getFomcItem(sid, read.asOf, horizon, signal, page), read.horizon);
  if (detail.status === 'loading') return <LoadingState label="Loading item" />;
  if (detail.status === 'error' && detail.error) return <ErrorState error={detail.error} onRetry={detail.refetch} />;
  const data = detail.data;
  if (!data || !data.item) return <EmptyState title="Nothing known about this item as of this read" />;
  return (
    <section className="card" aria-label="Item detail">
      <h2 className="card-title">Item <Hash value={sid} chars={16} /> · {data.item.state}</h2>
      <h3 className="muted">Revisions</h3>
      <div className="table-scroll">
        <table className="data">
          <thead><tr><th>Revision</th><th>Committed</th><th>Mode</th><th>Content domain</th><th>First raw</th></tr></thead>
          <tbody>
            {data.revisions.map((revision) => (
              <tr key={revision.revision_id}>
                <td><Hash value={revision.content_hash} chars={16} /></td>
                <td>{revision.committed_seq}</td>
                <td>{revision.observation_mode}</td>
                <td>{revision.content_identity?.domain ?? '—'}</td>
                <td><Hash value={revision.first_raw_sha256} chars={16} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3 className="muted">Observations and provenance</h3>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr><th>Record</th><th>Mode</th><th>Clock</th><th>Observed at</th><th>Outcome</th><th>Raw</th><th>Bytes</th><th>URL</th><th>Revision</th></tr>
          </thead>
          <tbody>
            {data.observations.map((observation) => (
              <tr key={observation.record}>
                <td>{observation.record}</td>
                <td>{observation.mode}</td>
                <td>{observation.verdict}</td>
                <td>{observation.observed_at ?? '—'}</td>
                <td>{observation.processing_outcome ?? '—'}</td>
                <td><Hash value={observation.raw_sha256} chars={16} /></td>
                <td>{observation.byte_length ?? '—'}</td>
                <td>{observation.final_url ?? observation.request_url}</td>
                <td><Hash value={observation.revision?.split(':')[1] ?? null} chars={12} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourcePager {...detail} />
    </section>
  );
}

function Snapshot({ read }: { read: Read }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [verify, setVerify] = useState(false);
  const snap = useSourcePage((signal, page, horizon) =>
    apiClient.getFomcSnapshot(read.asOf, horizon, signal, page), read.horizon);
  const snapshotRead = { ...read, horizon: snap.horizon ?? read.horizon };
  const replay = useQuery(verify ? `fomc-replay|${readKey(snapshotRead)}` : null, (signal) =>
    apiClient.getFomcReplay(snapshotRead.asOf, snapshotRead.horizon, signal));
  useEffect(() => {
    setSelected(null);
    setVerify(false);
  }, [read.asOf, read.horizon]);

  if (snap.status === 'loading') return <LoadingState label="Reading the store" />;
  if (snap.status === 'error' && snap.error) return <ErrorState error={snap.error} onRetry={snap.refetch} />;
  const data = snap.data;
  if (!data) return null;
  const header = data.snapshot;
  const resolved = header.read_state === 'FOMC_RESOLVED';
  return (
    <>
      <section className="card" aria-label="Snapshot">
        <h2 className="card-title">Snapshot</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Read state</dt><dd><Badge tone={resolved ? 'ok' : 'warn'}>{header.read_state}</Badge></dd></div>
          <div className="kv"><dt>Identity</dt><dd><code data-testid="snapshot-identity">{header.identity}</code></dd></div>
          <div className="kv"><dt>As of (T)</dt><dd>{header.T}</dd></div>
          <div className="kv"><dt>Horizon H · prefix P</dt><dd>{header.H} · {header.P ?? '—'}</dd></div>
          <div className="kv"><dt>Policy</dt><dd>{header.policy} ({header.mode})</dd></div>
          <div className="kv"><dt>Discovery</dt><dd>{data.discovery?.state ?? '—'}</dd></div>
          <div className="kv">
            <dt>Offline replay</dt>
            <dd>
              {!verify && <button className="control" onClick={() => setVerify(true)}>Verify replay</button>}
              {verify && replay.status === 'loading' && 'Replaying…'}
              {verify && replay.status === 'error' && replay.error && <Badge tone="off">{replay.error.message}</Badge>}
              {verify && replay.data && (replay.data.identical
                ? <><Badge tone="ok">Replay identical</Badge> <Hash value={replay.data.replay_identity} chars={16} /></>
                : <Badge tone="off">{`Replay differs: ${replay.data.error ?? replay.data.replay_identity}`}</Badge>)}
            </dd>
          </div>
        </dl>
      </section>

      {data.health && (
        <section className="card" aria-label="Source health">
          <h2 className="card-title">Source health</h2>
          <table className="data">
            <thead><tr><th>Surface</th><th>Failure state</th><th>Reason</th><th>Checked at</th><th>Outcome</th></tr></thead>
            <tbody>
              {Object.entries(data.health).map(([surface, row]) => (
                <tr key={surface}>
                  <td>{surface}</td>
                  <td>{row.result_state ?? '—'}</td>
                  <td>{row.reason}</td>
                  <td>{row.check_at ?? '—'}</td>
                  <td>{row.outcome ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card" aria-label="Items">
        <h2 className="card-title">Items as of T ({data.pagination?.totals.items ?? data.items.length})</h2>
        {data.items.length === 0 ? (
          <div className="state">{resolved ? 'No item was known at this instant.' : 'The read is unresolved: nothing is shown as of this instant.'}</div>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th /><th>State</th><th>Title</th><th>Statement date</th><th>Declared release</th><th>Mode</th><th>Content</th><th>Observations</th><th>Revision</th></tr>
              </thead>
              <tbody>
                {data.items.map((item: FomcItemSummary) => (
                  <tr key={item.sid}>
                    <td><button className="control" aria-label={`Open item ${item.sid.slice(0, 12)}`} onClick={() => setSelected(item.sid)}>Open</button></td>
                    <td>{item.state}</td>
                    <td>{item.title ?? '—'}</td>
                    <td>{item.official_statement_date ?? '—'}</td>
                    <td>{item.declared_release_text ?? '—'}</td>
                    <td>{item.observation_mode ?? '—'}</td>
                    <td>{item.content_domain ?? '—'}</td>
                    <td>{item.observations}</td>
                    <td><Hash value={item.content_hash} chars={12} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <SourcePager {...snap} />
      <SourceTimeline source="fomc" read={snapshotRead} />

      {selected && <ItemDetail key={selected} sid={selected} read={snapshotRead} />}
    </>
  );
}

function FomcPanel() {
  const status = useQuery('fomc-status', (signal) => apiClient.getFomcStatus(signal));
  const [read, setRead] = useState<Read | null>(null);
  const suggested = status.data?.suggested_as_of ?? null;

  useEffect(() => {
    if (suggested && read === null) setRead({ asOf: suggested });
  }, [suggested, read]);

  if (status.status === 'loading') return <LoadingState label="Loading event sources" />;
  if (status.status === 'error' && status.error) return <ErrorState error={status.error} onRetry={status.refetch} />;
  const data = status.data;
  if (!data) return null;
  if (data.status === 'NOT_CONFIGURED') {
    return (
      <EmptyState
        title="No FOMC store configured"
        detail="Start the API with --fomc-store DIR (an archive or a copy of a capture store; it is opened read-only)."
      />
    );
  }
  if (data.status === 'REJECTED') {
    return (
      <section className="card">
        <h2 className="card-title">FOMC store refused</h2>
        <p><Badge tone="off">REJECTED</Badge> {data.reason}</p>
      </section>
    );
  }
  return (
    <div className="stack">
      <StoreCard status={data} />
      <ReadForm initialAsOf={suggested ?? ''} onRead={setRead} />
      {read && <Snapshot key={readKey(read)} read={read} />}
    </div>
  );
}

const SOURCES = [
  { id: 'fomc', label: 'FOMC statements' },
  { id: 'edgar', label: 'SEC EDGAR filings' },
] as const;

export function EventsPage() {
  const [source, setSource] = useState<'fomc' | 'edgar'>('fomc');
  return (
    <div className="stack">
      <div role="tablist" aria-label="Event source" className="kv">
        {SOURCES.map((item) => (
          <button
            key={item.id}
            role="tab"
            aria-selected={source === item.id}
            aria-pressed={source === item.id}
            className="control"
            onClick={() => setSource(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {source === 'fomc' ? <FomcPanel /> : <EdgarPanel />}
    </div>
  );
}
