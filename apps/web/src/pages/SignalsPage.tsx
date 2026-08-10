import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { DataTable, type Column } from '../components/DataTable';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';

type Decision = {
  timestamp: string;
  prediction: string;
  direction: string;
  strength: string;
  decision_hash: string;
};

export function SignalsPage() {
  const { data, status, error, refetch } = useQuery('signals', (signal) =>
    apiClient.getSignals(200, signal),
  );

  if (status === 'loading') return <LoadingState label="Loading signals" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  const columns: Column<Decision>[] = [
    { key: 't', header: 'Timestamp', render: (row) => row.timestamp.replace('T', ' ').slice(0, 16) },
    { key: 'p', header: 'Prediction', render: (row) => row.prediction },
    // The direction is displayed exactly as the backend decided it. It is
    // never re-derived from the prediction here.
    { key: 'd', header: 'Direction', render: (row) => (
      <span className={row.direction === 'LONG' ? 'positive' : row.direction === 'SHORT' ? 'negative' : 'muted'}>
        {row.direction}
      </span>
    ) },
    { key: 's', header: 'Strength', render: (row) => row.strength },
    { key: 'h', header: 'Decision', render: (row) => <Hash value={row.decision_hash} chars={10} /> },
  ];

  const spec = data.signal_spec;

  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">Signal contract V1</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Rule</dt><dd>{spec.rule}</dd></div>
          <div className="kv"><dt>Long threshold</dt><dd className="positive">&gt; {spec.long_threshold}</dd></div>
          <div className="kv"><dt>Short threshold</dt><dd className="negative">&lt; {spec.short_threshold}</dd></div>
          <div className="kv"><dt>Full strength excess</dt><dd>{spec.full_strength_excess}</dd></div>
          <div className="kv"><dt>Boundary</dt><dd>{spec.boundary_semantics}</dd></div>
          <div className="kv"><dt>Horizon</dt><dd>{spec.prediction_horizon} bars</dd></div>
          <div className="kv"><dt>Optimized</dt><dd className="warning">{String(spec.optimized)}</dd></div>
          <div className="kv"><dt>Spec hash</dt><dd><Hash value={spec.spec_hash} /></dd></div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Decisions</h2>
        {data.available ? (
          <DataTable rows={data.decisions as Decision[]} columns={columns} height={420} />
        ) : (
          <EmptyState
            title="No persisted signal run available"
            detail="Signal decisions are generated on demand from predictions. None have been recorded to disk, and none are simulated here."
          />
        )}
      </section>
    </div>
  );
}
