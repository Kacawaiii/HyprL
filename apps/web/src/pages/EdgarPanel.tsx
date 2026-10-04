/** SEC EDGAR filings as the store knew them at an instant (offline slice: no capture is authorized).
 *  Every value is shipped by the Python reader; acceptanceDateTime is shown as provenance text only and a
 *  filing's availability is the server-attested one, never its acceptance time. */
import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { ReadForm, readKey } from '../components/ReadForm';
import type { Read } from '../components/ReadForm';

function FilingDetail({ accession, read }: { accession: string; read: Read }) {
  const detail = useQuery(`edgar-filing|${accession}|${readKey(read)}`, (signal) =>
    apiClient.getEdgarFiling(accession, read.asOf, read.horizon, signal));
  if (detail.status === 'loading') return <LoadingState label="Loading filing" />;
  if (detail.status === 'error' && detail.error) return <ErrorState error={detail.error} onRetry={detail.refetch} />;
  const data = detail.data;
  if (!data || !data.filing) return <EmptyState title="Nothing known about this filing as of this read" />;
  return (
    <section className="card" aria-label="Filing detail">
      <h2 className="card-title">Filing {accession} · {data.filing.state}</h2>
      <h3 className="muted">Revisions of the listed metadata</h3>
      <table className="data">
        <thead><tr><th>Revision</th><th>Committed</th><th>Items</th><th>First record</th></tr></thead>
        <tbody>
          {data.revisions.map((revision) => (
            <tr key={revision.revision}>
              <td><Hash value={revision.content_sha256} chars={16} /></td>
              <td>{revision.committed_seq}</td>
              <td>{String(revision.fields.items ?? '—')}</td>
              <td>{revision.first_record}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3 className="muted">Observations</h3>
      <table className="data">
        <thead><tr><th>Record</th><th>Observed at</th><th>Raw</th><th>Row</th><th>Revision</th></tr></thead>
        <tbody>
          {data.observations.map((observation) => (
            <tr key={observation.record}>
              <td>{observation.record}</td>
              <td>{observation.observed_at ?? 'clock unverified'}</td>
              <td><Hash value={observation.raw_sha256} chars={16} /></td>
              <td>{observation.position}</td>
              <td><Hash value={observation.revision.split(':')[1] ?? null} chars={12} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.absences.length > 0 && (
        <>
          <h3 className="muted">Absences from later listings</h3>
          <table className="data">
            <thead><tr><th>Record</th><th>Observed at</th><th>Filing date</th><th>Listing window from</th></tr></thead>
            <tbody>
              {data.absences.map((absence) => (
                <tr key={absence.record}>
                  <td>{absence.record}</td>
                  <td>{absence.observed_at ?? 'clock unverified'}</td>
                  <td>{absence.filing_date}</td>
                  <td>{absence.listing_oldest_filing_date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

function EdgarSnapshot({ read }: { read: Read }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [verify, setVerify] = useState(false);
  const snap = useQuery(`edgar-snapshot|${readKey(read)}`, (signal) =>
    apiClient.getEdgarSnapshot(read.asOf, read.horizon, signal));
  const replay = useQuery(verify ? `edgar-replay|${readKey(read)}` : null, (signal) =>
    apiClient.getEdgarReplay(read.asOf, read.horizon, signal));
  useEffect(() => {
    setSelected(null);
    setVerify(false);
  }, [read.asOf, read.horizon]);

  if (snap.status === 'loading') return <LoadingState label="Reading the store" />;
  if (snap.status === 'error' && snap.error) return <ErrorState error={snap.error} onRetry={snap.refetch} />;
  const data = snap.data;
  if (!data) return null;
  const header = data.snapshot;
  const resolved = header.read_state === 'EDGAR_RESOLVED';
  return (
    <>
      <section className="card" aria-label="EDGAR snapshot">
        <h2 className="card-title">Snapshot</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Read state</dt><dd><Badge tone={resolved ? 'ok' : 'warn'}>{header.read_state}</Badge></dd></div>
          <div className="kv"><dt>Identity</dt><dd><code data-testid="edgar-identity">{header.identity}</code></dd></div>
          <div className="kv"><dt>As of (T)</dt><dd>{header.T}</dd></div>
          <div className="kv"><dt>Horizon H · prefix P</dt><dd>{header.H} · {header.P ?? '—'}</dd></div>
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
        <section className="card" aria-label="EDGAR source health">
          <h2 className="card-title">Source health</h2>
          <table className="data">
            <thead><tr><th>CIK</th><th>Failure state</th><th>Reason</th><th>Checked at</th></tr></thead>
            <tbody>
              {Object.entries(data.health).map(([cik, row]) => (
                <tr key={cik}>
                  <td>{cik}</td>
                  <td>{row.result_state ?? '—'}</td>
                  <td>{row.reason}</td>
                  <td>{row.check_at ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card" aria-label="Filings">
        <h2 className="card-title">Filings as of T ({data.filings.length})</h2>
        {data.filings.length === 0 ? (
          <div className="state">{resolved ? 'No filing was known at this instant.' : 'The read is unresolved: nothing is shown as of this instant.'}</div>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th /><th>Accession</th><th>Form</th><th>Filed</th><th>State</th><th>Items</th><th>Revisions</th><th>Available at</th><th>Acceptance (provenance)</th></tr>
              </thead>
              <tbody>
                {data.filings.map((filing) => (
                  <tr key={filing.accession_number}>
                    <td><button className="control" aria-label={`Open filing ${filing.accession_number}`} onClick={() => setSelected(filing.accession_number)}>Open</button></td>
                    <td>{filing.accession_number}</td>
                    <td>{filing.form}</td>
                    <td>{filing.filing_date}</td>
                    <td>{filing.state}</td>
                    <td>{filing.items ?? '—'}</td>
                    <td>{filing.revisions_seen}</td>
                    <td>{filing.first_available_at ?? '—'}</td>
                    <td>{filing.acceptance_datetime_text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selected && <FilingDetail key={selected} accession={selected} read={read} />}
    </>
  );
}

export function EdgarPanel() {
  const status = useQuery('edgar-status', (signal) => apiClient.getEdgarStatus(signal));
  const [read, setRead] = useState<Read | null>(null);
  const suggested = status.data?.suggested_as_of ?? null;
  useEffect(() => {
    if (suggested && read === null) setRead({ asOf: suggested });
  }, [suggested, read]);

  if (status.status === 'loading') return <LoadingState label="Loading SEC EDGAR" />;
  if (status.status === 'error' && status.error) return <ErrorState error={status.error} onRetry={status.refetch} />;
  const data = status.data;
  if (!data) return null;
  if (data.status === 'NOT_CONFIGURED') {
    return <EmptyState title="No EDGAR store configured" detail="Start the API with --edgar-store DIR (opened read-only)." />;
  }
  if (data.status === 'REJECTED') {
    return (
      <section className="card">
        <h2 className="card-title">EDGAR store refused</h2>
        <p><Badge tone="off">REJECTED</Badge> {data.reason}</p>
      </section>
    );
  }
  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">SEC EDGAR store</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Status</dt><dd><Badge tone="ok">{data.status}</Badge> read-only · offline slice</dd></div>
          <div className="kv"><dt>Store</dt><dd>{data.store}</dd></div>
          <div className="kv"><dt>Spec</dt><dd>revision {data.spec_revision} <Hash value={data.spec_hash} chars={16} /></dd></div>
          <div className="kv"><dt>Watchlist</dt><dd>{(data.watchlist ?? []).join(', ') || '—'}</dd></div>
          <div className="kv"><dt>Horizon</dt><dd>{data.horizon}</dd></div>
          <div className="kv"><dt>Server-attested now (lower bound)</dt><dd>{data.suggested_as_of ?? '—'}</dd></div>
          {data.counts && (
            <div className="kv">
              <dt>Records</dt>
              <dd>
                {data.counts.responses} listings · {data.counts.revisions} revisions · {data.counts.observations} observations ·{' '}
                {data.counts.absences} absences
              </dd>
            </div>
          )}
        </dl>
      </section>
      <ReadForm initialAsOf={suggested ?? ''} onRead={setRead} />
      {read && <EdgarSnapshot key={readKey(read)} read={read} />}
    </div>
  );
}
