import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, CapabilityBadge, ErrorState, Hash, LoadingState } from '../components/States';

export function OverviewPage() {
  const { data, status, error, refetch } = useQuery('overview', (signal) =>
    apiClient.getOverview(signal),
  );

  if (status === 'loading') return <LoadingState label="Loading overview" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  return (
    <div className="stack">
      <section className="grid grid-3">
        {data.products.map((product) => (
          <article className="card" key={product.product}>
            <h2 className="card-title">{product.product}</h2>
            <div className="metric">{product.rows.toLocaleString()}</div>
            <div className="metric-sub">hourly bars · {product.missing_openings} declared gaps</div>
            <dl style={{ margin: '12px 0 0' }}>
              <div className="kv"><dt>From</dt><dd>{product.first_open.slice(0, 10)}</dd></div>
              <div className="kv"><dt>To</dt><dd>{product.last_open.slice(0, 10)}</dd></div>
              <div className="kv">
                <dt>Latest price</dt>
                <dd className="muted">{product.latest_price ?? 'unavailable'}</dd>
              </div>
            </dl>
          </article>
        ))}
      </section>

      <section className="grid grid-3">
        <article className="card">
          <h2 className="card-title">Signal engine</h2>
          <div className="row"><Badge tone="ok">FROZEN V1</Badge></div>
          <p className="metric-sub" style={{ marginTop: 10 }}>
            Not optimized for profitability
          </p>
          <Hash value={data.signal_spec_hash} />
        </article>
        <article className="card">
          <h2 className="card-title">Risk engine</h2>
          <div className="row"><Badge tone="ok">FROZEN V1</Badge></div>
          <p className="metric-sub" style={{ marginTop: 10 }}>
            Target exposure only · no execution
          </p>
          <Hash value={data.risk_spec_hash} />
        </article>
        <article className="card">
          <h2 className="card-title">Capabilities</h2>
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Backtest</dt><dd><CapabilityBadge enabled={data.capabilities.economic_backtest} /></dd></div>
            <div className="kv"><dt>Paper trading</dt><dd><CapabilityBadge enabled={data.capabilities.paper_trading} /></dd></div>
            <div className="kv"><dt>Live trading</dt><dd><CapabilityBadge enabled={data.capabilities.live_trading} /></dd></div>
          </dl>
        </article>
      </section>

      <section className="card">
        <h2 className="card-title">Benchmarks</h2>
        <table className="data">
          <thead>
            <tr><th>Version</th><th>Type</th><th>Product</th><th>rank_ic</th><th>MAE</th><th>RMSE</th><th>Obs</th></tr>
          </thead>
          <tbody>
            {data.benchmarks.flatMap((benchmark) =>
              benchmark.products.map((entry) => (
                <tr key={`${benchmark.version}-${entry.product}`}>
                  <td>{benchmark.version.toUpperCase()}</td>
                  <td>
                    <Badge tone={benchmark.confirmatory_result ? 'ok' : 'warn'}>
                      {benchmark.experiment_type.toUpperCase()}
                    </Badge>
                  </td>
                  <td>{entry.product}</td>
                  <td className={entry.rank_ic.startsWith('-') ? 'negative' : 'positive'}>
                    {Number(entry.rank_ic).toFixed(6)}
                  </td>
                  <td>{Number(entry.mae).toFixed(6)}</td>
                  <td>{Number(entry.rmse).toFixed(6)}</td>
                  <td>{entry.observations.toLocaleString()}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
