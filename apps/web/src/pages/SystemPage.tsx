import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, CapabilityBadge, ErrorState, Hash, LoadingState } from '../components/States';

export function SystemPage() {
  const { data, status, error, refetch } = useQuery('system', (signal) =>
    apiClient.getSystem(signal),
  );

  if (status === 'loading') return <LoadingState label="Loading system" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  return (
    <div className="stack">
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
        <h2 className="card-title">Market corpus</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Corpus</dt><dd>{data.market_data.corpus_id ?? '—'}</dd></div>
          <div className="kv"><dt>Products</dt><dd>{data.market_data.products.join(', ')}</dd></div>
          <div className="kv"><dt>Timeframe</dt><dd>{data.market_data.timeframe}</dd></div>
          <div className="kv"><dt>Spec hash</dt><dd><Hash value={data.market_data.corpus_spec_hash} chars={20} /></dd></div>
          <div className="kv"><dt>Content hash</dt><dd><Hash value={data.market_data.corpus_content_hash} chars={20} /></dd></div>
          <div className="kv">
            <dt>Point-in-time revisions</dt>
            <dd className="muted">{String(data.market_data.point_in_time_revision_history)}</dd>
          </div>
        </dl>
      </section>

      <section className="card">
        <h2 className="card-title">Benchmarks</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>V1</dt><dd><CapabilityBadge enabled={data.benchmarks.v1_available} /></dd></div>
          <div className="kv"><dt>V2 exploratory</dt><dd><CapabilityBadge enabled={data.benchmarks.v2_exploratory_available} /></dd></div>
          <div className="kv">
            <dt>V2 confirmatory</dt>
            <dd><Badge tone="warn">NOT OBSERVED</Badge></dd>
          </div>
          {data.benchmarks.confirmatory_holdout && (
            <div className="kv">
              <dt>Reserved holdout</dt>
              <dd>
                {data.benchmarks.confirmatory_holdout.range_start.slice(0, 10)} →{' '}
                {data.benchmarks.confirmatory_holdout.range_end.slice(0, 10)}
              </dd>
            </div>
          )}
        </dl>
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
