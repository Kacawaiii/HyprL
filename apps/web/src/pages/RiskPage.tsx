import { useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { DataTable, type Column } from '../components/DataTable';
import { Num } from '../components/Num';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { ProductSelect, RunProvenanceCard, unavailableDetail, useOlderRows } from '../components/RunProvenance';

type Target = {
  timestamp: string;
  side: string;
  target_exposure: string;
  signal_strength: string;
  position_target_hash: string;
};

export function RiskPage() {
  const [product, setProduct] = useState<string | null>(null);
  const { data, status, error, refetch } = useQuery(`risk:${product ?? 'default'}`, (signal) =>
    apiClient.getRiskTargets(200, signal, { product: product ?? undefined }),
  );
  const more = useOlderRows(
    product ?? 'default', data,
    (cursor) => apiClient.getRiskTargets(200, undefined, { product: product ?? undefined, cursor }),
    (view) => view.targets as Target[],
  );

  if (status === 'loading') return <LoadingState label="Loading risk contract" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  const spec = data.risk_spec;

  // Side and exposure are displayed exactly as the backend decided them.
  const columns: Column<Target>[] = [
    { key: 't', header: 'Timestamp', render: (row) => row.timestamp.replace('T', ' ').slice(0, 16) },
    { key: 'side', header: 'Side', render: (row) => (
      <span className={row.side === 'LONG' ? 'positive' : row.side === 'SHORT' ? 'negative' : 'muted'}>
        {row.side}
      </span>
    ) },
    { key: 'e', header: 'Target exposure', render: (row) => <Num value={row.target_exposure} kind="percent" /> },
    { key: 's', header: 'Signal strength', render: (row) => <Num value={row.signal_strength} /> },
    { key: 'h', header: 'Target', render: (row) => <Hash value={row.position_target_hash} chars={10} /> },
  ];

  return (
    <div className="stack">
      <ProductSelect value={product} shown={data.product} onChange={setProduct} />
      <RunProvenanceCard run={data} title="Walk-forward run" />
      <section className="card">
        <h2 className="card-title">Risk contract V1</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Max long exposure</dt><dd>{spec.max_long_exposure}</dd></div>
          <div className="kv"><dt>Max short exposure</dt><dd>{spec.max_short_exposure}</dd></div>
          <div className="kv"><dt>Risk scale</dt><dd>{spec.risk_scale}</dd></div>
          <div className="kv"><dt>Volatility scaling</dt><dd className="muted">{String(spec.volatility_scaling_enabled)}</dd></div>
          <div className="kv"><dt>Mapping</dt><dd>{spec.strength_mapping_version}</dd></div>
          <div className="kv"><dt>Optimized</dt><dd className="warning">{String(spec.optimized)}</dd></div>
          <div className="kv"><dt>Spec hash</dt><dd><Hash value={spec.spec_hash} /></dd></div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Mapping</h2>
        <div className="row" style={{ gap: 16, flexWrap: 'wrap' }}>
          <div className="card" style={{ flex: 1, minWidth: 170 }}>
            <div className="card-title">Signal strength</div>
            <div className="metric">0 → 1</div>
            <div className="metric-sub">descriptive intensity</div>
          </div>
          <span aria-hidden="true" className="muted">→</span>
          <div className="card" style={{ flex: 1, minWidth: 170 }}>
            <div className="card-title">Risk mapping</div>
            <div className="metric">× {spec.max_long_exposure}</div>
            <div className="metric-sub">capped, scale {spec.risk_scale}</div>
          </div>
          <span aria-hidden="true" className="muted">→</span>
          <div className="card" style={{ flex: 1, minWidth: 170 }}>
            <div className="card-title">Target exposure</div>
            <div className="metric">±{spec.max_long_exposure}</div>
            <div className="metric-sub">fraction of NAV, not an order</div>
          </div>
        </div>
        <p className="metric-sub" style={{ marginTop: 12 }}>
          Signal strength is not position size, and a target exposure is not a trade
          quantity. Converting one into the other needs equity and price, which this
          layer never sees.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">Position targets</h2>
        {data.available ? (
          <>
            <DataTable
              rows={[...(data.targets as Target[]), ...more.older]}
              columns={columns}
              height={420}
            />
            {more.failure && <ErrorState error={more.failure} onRetry={more.loadOlder} />}
            {more.hasMore && (
              <button className="control" onClick={more.loadOlder} disabled={more.busy}>
                {more.busy ? 'Loading…' : 'Load older targets'}
              </button>
            )}
          </>
        ) : (
          <EmptyState
            title="No persisted position target run available"
            detail={unavailableDetail(data.reason, 'No persisted position target run available', 'None have been recorded, and none are fabricated here.')}
          />
        )}
      </section>
    </div>
  );
}
