/** Monitoring: data gaps, technical degradation, drift and performance drop are kept apart, and every edge claim names its sample and method. */
import { useState } from 'react';
import { apiClient } from '../../api/client';
import type { MonitoringView as Monitoring, PerformanceBlock, ReferenceRow } from '../../api/labTypes';
import { Badge, EmptyState, Hash } from '../../components/States';
import { classificationExplanation, edgeSentence, formatCount, small } from '../../lib/lab';
import { useCockpit } from '../../state/useCockpit';
import { useQuery } from '../../state/useQuery';
import { QueryBoundary, Synthetic } from './shared';

const CATEGORIES = ['MISSING_DATA', 'TECHNICAL_DEGRADATION', 'DRIFT', 'PERFORMANCE_DROP'] as const;

function Diagnostics({ data }: { data: Monitoring }) {
  const raised = new Map(data.classification.map((item) => [item.category, item]));
  return (
    <section className="card" aria-label="Diagnosis">
      <h2 className="card-title">Diagnosis (four separate causes)</h2>
      <ul className="lab-list">
        {CATEGORIES.map((category) => {
          const item = raised.get(category);
          return (
            <li key={category} className="row" data-category={category}>
              <Badge tone={item ? 'warn' : 'ok'}>{item ? 'RAISED' : 'NOT RAISED'}</Badge>
              <strong>{category.replace('_', ' ')}</strong>
              <span className="muted">{item ? classificationExplanation(item) : 'No evidence of this cause in the sample.'}</span>
            </li>
          );
        })}
      </ul>
      <p className="lab-note" style={{ marginTop: 8 }}>
        {data.sample === 0
          ? 'No prediction fell in this selection: nothing can be concluded.'
          : `Based on ${formatCount(data.sample)} predictions, scope ${data.scope}. Thresholds are descriptive (method ${data.method.version}), not statistical tests.`}
      </p>
    </section>
  );
}

function Block({ title, block }: { title: string; block: PerformanceBlock }) {
  return (
    <tr>
      <td>{title}</td><td>{block.sample}</td><td>{block.pending}</td>
      <td>{block.model ? small(block.model.mae) : 'not available'}</td>
      <td>{Object.entries(block.baselines).map(([name, v]) => `${name} ${small(v.mae)}`).join(' · ') || 'none'}</td>
      <td>{block.edge.ZERO ? small(block.edge.ZERO.mse_reduction) : 'not available'}</td>
    </tr>
  );
}

function PerformanceTable({ label, groups }: { label: string; groups: Record<string, PerformanceBlock> }) {
  const entries = Object.entries(groups);
  return (
    <table className="data" aria-label={label}>
      <caption className="lab-note" style={{ textAlign: 'left' }}>{label}</caption>
      <thead><tr><th>Group</th><th>Labelled</th><th>Pending</th><th>Model MAE</th><th>Baseline MAE</th><th>MSE reduction vs ZERO</th></tr></thead>
      <tbody>
        {entries.length === 0 && <tr><td colSpan={6}>No labelled predictions.</td></tr>}
        {entries.map(([name, block]) => <Block key={name} title={name} block={block} />)}
      </tbody>
    </table>
  );
}

function Panels({ data, expert }: { data: Monitoring; expert: boolean }) {
  const freshness = data.freshness;
  const stale = freshness.seconds_since_last_decision !== null && freshness.seconds_since_last_decision > freshness.threshold_seconds;
  const entries = Object.entries(data.drift);
  return (
    <div className="stack">
      <Diagnostics data={data} />
      <div className="grid grid-3">
        <article className="card" aria-label="Freshness and inputs">
          <h3 className="card-title">Freshness and inputs</h3>
          <dl>
            <div className="kv"><dt>Last decision age</dt><dd>{freshness.seconds_since_last_decision === null ? 'no decision' : `${Math.round(freshness.seconds_since_last_decision / 3600)} h`}{stale ? ' (stale)' : ''}</dd></div>
            <div className="kv"><dt>Staleness threshold</dt><dd>{freshness.threshold_seconds / 3600} h</dd></div>
            <div className="kv"><dt>Decision gaps</dt><dd>{freshness.decision_gaps ?? 'unknown'}</dd></div>
            <div className="kv"><dt>Input gaps</dt><dd>{freshness.input_gaps} known, {freshness.input_gap_unknown_rows} unknown</dd></div>
            <div className="kv"><dt>Missing features</dt><dd>{data.input_quality.missing_features} of {data.input_quality.sample}</dd></div>
          </dl>
        </article>
        <article className="card" aria-label="Inference">
          <h3 className="card-title">Inference service</h3>
          <div className="metric">{data.inference.state}</div>
          <p className="metric-sub">
            {data.inference.state === 'NOT_OBSERVED'
              ? 'Latency, errors and availability were not measured for this population. This is unknown, not healthy.'
              : `${data.inference.attempts} attempts, ${data.inference.errors} errors, availability ${small(data.inference.availability)}.`}
          </p>
        </article>
        <article className="card" aria-label="Prediction distribution">
          <h3 className="card-title">Prediction distribution</h3>
          <dl>
            <div className="kv"><dt>Count</dt><dd>{data.distributions.prediction.count}</dd></div>
            <div className="kv"><dt>Mean / std</dt><dd>{small(data.distributions.prediction.mean)} / {small(data.distributions.prediction.std)}</dd></div>
            <div className="kv"><dt>Median / p95</dt><dd>{small(data.distributions.prediction.p50)} / {small(data.distributions.prediction.p95)}</dd></div>
          </dl>
        </article>
      </div>
      <section className="card" aria-label="Drift">
        <h3 className="card-title">Drift against the reference</h3>
        <table className="data">
          <thead><tr><th>Signal</th><th>State</th><th>PSI</th><th>KS</th><th>Reference / current</th></tr></thead>
          <tbody>
            {entries.map(([name, drift]) => (
              <tr key={name}>
                <td>{name}</td>
                <td><Badge tone={drift.state === 'STABLE' ? 'ok' : drift.state === 'INSUFFICIENT_SAMPLE' ? 'off' : 'warn'}>{drift.state}</Badge></td>
                <td>{small(drift.psi)}</td><td>{small(drift.ks_statistic)}</td><td>{drift.reference_count} / {drift.current_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="lab-note">PSI and KS are descriptive distances (thresholds in the method); there is no p-value.</p>
      </section>
      <section className="card" aria-label="Edge versus baselines">
        <h3 className="card-title">Edge against baselines</h3>
        <ul>
          {Object.keys(data.performance.edge).map((baseline) => <li key={baseline}>{edgeSentence(data.performance, baseline)}</li>)}
          {Object.keys(data.performance.edge).length === 0 && <li>No labelled pairs: no edge can be stated.</li>}
        </ul>
        <p className="lab-note">
          A measured difference on this sample does not establish that an edge exists, nor that it disappeared.
          {data.performance.pending > 0 ? ` ${data.performance.pending} labels are still pending and excluded.` : ''}
        </p>
      </section>
      <section className="card" aria-label="Performance breakdown">
        <PerformanceTable label="By product" groups={data.performance_by_product} />
        <PerformanceTable label="By period (decision month)" groups={data.performance_by_period} />
        <PerformanceTable label="By regime" groups={data.performance_by_regime} />
        <p className="lab-note">Regimes ({data.regime_definition.version}): {data.regime_definition.definition.join('; ')}.</p>
      </section>
      <section className="card" aria-label="Method and limits">
        <h3 className="card-title">Method and limits</h3>
        <ul>{[...data.method.limitations, ...data.limitations].map((limit) => <li key={limit}>{limit}</li>)}</ul>
        {expert && (
          <dl>
            <div className="kv"><dt>Method hash</dt><dd><Hash value={data.method_hash} chars={20} /></dd></div>
            <div className="kv"><dt>Regime hash</dt><dd><Hash value={data.regime_hash} chars={20} /></dd></div>
            <div className="kv"><dt>Reference</dt><dd><Hash value={data.reference_hash} chars={20} /></dd></div>
            <div className="kv"><dt>Population</dt><dd><Hash value={data.population_hash} chars={20} /></dd></div>
          </dl>
        )}
      </section>
    </div>
  );
}

function Selected({ reference, asOf, expert }: { reference: ReferenceRow; asOf: string; expert: boolean }) {
  const { product, model_id: modelId } = reference.payload.selection;
  const monitoring = useQuery(
    `monitoring:${reference.identity}:${asOf}`,
    (signal) => apiClient.getMonitoring({ asOf, product, modelId, referenceHash: reference.identity }, signal),
  );
  return (
    <QueryBoundary query={monitoring} label="Computing monitoring">
      {(data) => <Panels data={data} expert={expert} />}
    </QueryBoundary>
  );
}

export function MonitoringView() {
  const { selection, update } = useCockpit();
  const [asOf, setAsOf] = useState(() => new Date().toISOString());
  const [chosen, setChosen] = useState<string | null>(null);
  const references = useQuery('monitoring:references', (signal) => apiClient.getObservabilityReferences(50, signal));
  const health = useQuery('monitoring:health', (signal) => apiClient.getObservabilityHealth(signal));
  return (
    <div className="stack">
      <QueryBoundary query={health} label="Checking the observability store">
        {(data) => (
          <p className="lab-note">
            Observability store: {data.verified ? 'chain verified' : 'CHAIN NOT VERIFIED'}, {formatCount(data.records)} records, read-only,
            {' '}{data.network_requests} network requests. Method {data.method.version}.
          </p>
        )}
      </QueryBoundary>
      <QueryBoundary query={references} label="Loading monitoring references">
        {(data) => {
          const rows = data.records;
          const reference0 = rows[0];
          if (!reference0) return <EmptyState title="No monitoring reference" detail="Monitoring compares against a versioned reference population; none is recorded." />;
          const fromSelection = rows.find((row) => row.payload.selection.product === selection.product && row.payload.selection.model_id === selection.model);
          const reference = rows.find((row) => row.identity === chosen) ?? fromSelection ?? reference0;
          return (
            <>
              <form className="card cockpit-selection" aria-label="Monitoring selection" onSubmit={(event) => event.preventDefault()}>
                <label>Reference{' '}
                  <select className="control" value={reference.identity}
                    onChange={(event) => {
                      const next = rows.find((row) => row.identity === event.target.value);
                      setChosen(event.target.value);
                      if (next) update({ product: next.payload.selection.product, model: next.payload.selection.model_id });
                    }}>
                    {rows.map((row) => (
                      <option key={row.identity} value={row.identity}>
                        {row.payload.selection.model_id} · {row.payload.selection.product} · {row.payload.selection.split}
                      </option>
                    ))}
                  </select>
                </label>
                <span className="muted">As of {asOf.slice(0, 19).replace('T', ' ')} UTC</span>
                <button type="button" className="control" onClick={() => setAsOf(new Date().toISOString())}>Refresh</button>
                {reference.payload.synthetic && <Synthetic />}
              </form>
              <Selected key={reference.identity} reference={reference} asOf={asOf} expert={selection.mode === 'expert'} />
            </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
