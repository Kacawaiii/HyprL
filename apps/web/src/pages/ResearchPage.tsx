import { useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, ErrorState, Hash, LoadingState } from '../components/States';

function metric(value: string | null): string {
  return value === null ? '—' : Number(value).toFixed(6);
}

export function ResearchPage() {
  const [selected, setSelected] = useState<{ version: string; product: string } | null>(null);
  const summaries = useQuery('benchmarks', (signal) => apiClient.getBenchmarks(signal));
  const detail = useQuery(
    selected ? `benchmark:${selected.version}:${selected.product}` : null,
    (signal) => apiClient.getBenchmarkDetail(selected!.version, selected!.product, signal),
  );

  if (summaries.status === 'loading') return <LoadingState label="Loading benchmarks" />;
  if (summaries.status === 'error' && summaries.error) {
    return <ErrorState error={summaries.error} onRetry={summaries.refetch} />;
  }

  return (
    <div className="stack">
      {summaries.data?.benchmarks.map((benchmark) => (
        <section className="card" key={benchmark.version}>
          <div className="row" style={{ marginBottom: 12 }}>
            <h2 className="card-title" style={{ margin: 0 }}>
              Benchmark {benchmark.version.toUpperCase()}
            </h2>
            <Badge tone={benchmark.confirmatory_result ? 'ok' : 'warn'}>
              {benchmark.experiment_type.toUpperCase()}
            </Badge>
            <Badge tone="off">
              CONFIRMATORY: {benchmark.confirmatory_result ? 'YES' : 'NOT OBSERVED'}
            </Badge>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Product</th><th>rank_ic</th><th>MAE</th><th>RMSE</th>
                <th>Obs</th><th>Folds</th><th>Dataset</th><th />
              </tr>
            </thead>
            <tbody>
              {benchmark.products.map((entry) => (
                <tr key={entry.product}>
                  <td>{entry.product}</td>
                  <td className={entry.rank_ic.startsWith('-') ? 'negative' : 'positive'}>
                    {metric(entry.rank_ic)}
                  </td>
                  <td>{metric(entry.mae)}</td>
                  <td>{metric(entry.rmse)}</td>
                  <td>{entry.observations.toLocaleString()}</td>
                  <td>{entry.folds}</td>
                  <td><Hash value={entry.dataset_hash} chars={10} /></td>
                  <td>
                    <button
                      className="control"
                      onClick={() =>
                        setSelected({ version: benchmark.version, product: entry.product })
                      }
                    >
                      Periods
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}

      {selected && (
        <section className="card">
          <h2 className="card-title">
            {selected.version.toUpperCase()} · {selected.product} · quarters
          </h2>
          {detail.status === 'loading' && <LoadingState label="Loading detail" />}
          {detail.status === 'error' && detail.error && (
            <ErrorState error={detail.error} onRetry={detail.refetch} />
          )}
          {detail.data && (
            <>
              <table className="data">
                <thead>
                  <tr><th>Period</th><th>rank_ic</th><th>MAE</th><th>RMSE</th><th>Obs</th></tr>
                </thead>
                <tbody>
                  {detail.data.periods.map((period) => (
                    <tr key={period.start}>
                      <td>{period.start.slice(0, 10)} → {period.end.slice(0, 10)}</td>
                      <td className={period.rank_ic?.startsWith('-') ? 'negative' : 'positive'}>
                        {metric(period.rank_ic)}
                      </td>
                      <td>{metric(period.mae)}</td>
                      <td>{metric(period.rmse)}</td>
                      <td>{period.observations.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <h3 className="card-title" style={{ marginTop: 20 }}>Sensitivity scenarios</h3>
              <table className="data">
                <thead><tr><th>Scenario</th><th>rank_ic</th><th>MAE</th><th>RMSE</th></tr></thead>
                <tbody>
                  {detail.data.scenarios.map((scenario) => (
                    <tr key={scenario.scenario_id}>
                      <td>{scenario.scenario_id}</td>
                      <td className={scenario.rank_ic?.startsWith('-') ? 'negative' : 'positive'}>
                        {metric(scenario.rank_ic)}
                      </td>
                      <td>{metric(scenario.mae)}</td>
                      <td>{metric(scenario.rmse)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="metric-sub" style={{ marginTop: 12 }}>
                Results are read from committed artefacts. Nothing is recomputed in the
                browser, and there is no way to re-run a benchmark from this page.
              </p>
            </>
          )}
        </section>
      )}
    </div>
  );
}
