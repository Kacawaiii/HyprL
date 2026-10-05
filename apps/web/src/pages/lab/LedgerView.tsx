/** Prediction ledger: every prediction with its pending or realized label. A label arriving later never rewrites the prediction. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../../api/client';
import type { LedgerRow, PageInfo } from '../../api/labTypes';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../../components/States';
import { horizonLabel } from '../../lib/lab';
import { formatRatio } from '../../lib/format';
import { useCockpit } from '../../state/useCockpit';
import { useQuery } from '../../state/useQuery';
import { QueryBoundary, Synthetic, Timestamp } from './shared';

const PAGE = 25;

function absent(value: unknown): string {
  return value === null || value === undefined ? 'not provided' : JSON.stringify(value);
}

function Detail({ identity, expert }: { identity: string; expert: boolean }) {
  const view = useQuery(`ledger:${identity}`, (signal) => apiClient.getObservabilityPrediction(identity, signal));
  return (
    <QueryBoundary query={view} label="Loading prediction">
      {(data) => {
        const prediction = data.prediction;
        return (
          <div className="stack" aria-label="Prediction detail">
            <div className="row" style={{ flexWrap: 'wrap' }}>
              <strong>{prediction.product} · <Timestamp value={prediction.decision_at} /></strong>
              <Badge tone={data.label_state === 'AVAILABLE' ? 'ok' : 'warn'}>LABEL {data.label_state}</Badge>
              {prediction.synthetic && <Synthetic />}
            </div>
            <dl>
              <div className="kv"><dt>Model</dt><dd>{prediction.model_id}</dd></div>
              <div className="kv"><dt>Horizon</dt><dd>{horizonLabel(prediction.horizon_seconds)}</dd></div>
              <div className="kv"><dt>Predicted return</dt><dd>{formatRatio(prediction.outputs.return, 4, 'not provided')}</dd></div>
              <div className="kv"><dt>Target price</dt><dd>{absent(prediction.outputs.target_price)}</dd></div>
              <div className="kv"><dt>Class / probabilities / quantiles / scenarios</dt>
                <dd>{[prediction.outputs.class, prediction.outputs.probabilities, prediction.outputs.quantiles, prediction.outputs.scenarios]
                  .every((value) => value === null) ? 'not provided' : 'provided'}</dd></div>
              <div className="kv"><dt>Signal / risk decision / proposed position</dt>
                <dd>{[prediction.signal, prediction.risk, prediction.proposed_position].every((v) => v === null) ? 'not recorded for this prediction' : 'recorded'}</dd></div>
              <div className="kv"><dt>Execution</dt><dd>{data.execution_state}</dd></div>
              <div className="kv"><dt>Uncertainty</dt><dd>{absent(prediction.uncertainty)} (no calibrated method claimed)</dd></div>
            </dl>
            <table className="data" aria-label="Label history">
              <caption className="lab-note" style={{ textAlign: 'left' }}>Label history (append-only)</caption>
              <thead><tr><th>Version</th><th>Realized value</th><th>Realized at</th><th>Available at</th><th>Recorded</th></tr></thead>
              <tbody>
                {data.labels.length === 0 && <tr><td colSpan={5}>Pending: the label is not realized or not yet available.</td></tr>}
                {data.labels.map((label) => (
                  <tr key={label.label_id}><td>{label.version}</td><td>{formatRatio(label.value)}</td>
                    <td><Timestamp value={label.realized_at} /></td><td><Timestamp value={label.available_at} /></td><td><Timestamp value={label.recorded_at} /></td></tr>
                ))}
              </tbody>
            </table>
            {data.inputs && (
              <>
                <p className="lab-note">
                  Inputs: snapshot coverage {data.inputs.snapshot.coverage.state}, input quality {data.inputs.input_quality.state}
                  {data.inputs.input_quality.gaps > 0 ? `, ${data.inputs.input_quality.gaps} gaps` : ''}.
                  Source states: {Object.entries(data.inputs.input_quality.source_states).map(([k, v]) => `${k} ${v}`).join(', ')}.
                  Events used: {prediction.event_ids.length === 0 ? 'none' : prediction.event_ids.length}.
                </p>
                {expert && (
                  <details open>
                    <summary>Exact features</summary>
                    <table className="data">
                      <thead><tr><th>Feature</th><th>Value</th></tr></thead>
                      <tbody>{data.inputs.features.map(([name, value]) => <tr key={name}><td>{name}</td><td>{value}</td></tr>)}</tbody>
                    </table>
                  </details>
                )}
              </>
            )}
            {expert && (
              <dl>
                <div className="kv"><dt>Prediction hash</dt><dd><Hash value={data.prediction_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Snapshot</dt><dd><Hash value={prediction.snapshot_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Features hash</dt><dd><Hash value={prediction.features_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Artifact</dt><dd><Hash value={prediction.artifact_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Model contract</dt><dd><Hash value={prediction.model_contract_hash} chars={20} /></dd></div>
              </dl>
            )}
          </div>
        );
      }}
    </QueryBoundary>
  );
}

export function LedgerView() {
  const { selection, update } = useCockpit();
  const expert = selection.mode === 'expert';
  // Products come from the market registry, plus any product already present in the ledger or the selection.
  const overview = useQuery('overview', (signal) => apiClient.getOverview(signal));
  const [rows, setRows] = useState<LedgerRow[]>([]);
  const [page, setPage] = useState<PageInfo | null>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState<Error | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  // Continuation requires the original as_of, so it is pinned per filter and reused for every page.
  const asOf = useRef('');
  const controller = useRef<AbortController | null>(null);

  const load = useCallback((cursor?: string) => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    if (!cursor) {
      asOf.current = new Date().toISOString();
      setRows([]);
      setPage(null);
    }
    setState('loading');
    apiClient.getObservabilityPredictions({
      product: selection.product ?? undefined, modelId: selection.model ?? undefined,
      limit: PAGE, cursor, asOf: asOf.current,
    }, next.signal).then((result) => {
      setRows((current) => (cursor ? [...current, ...result.records] : result.records));
      setPage(result.page);
      setState('ready');
    }).catch((failure: Error) => {
      if (next.signal.aborted) return;
      setError(failure);
      setState('error');
    });
  }, [selection.product, selection.model]);

  useEffect(() => {
    setOpen(null);
    load();
    return () => controller.current?.abort();
  }, [load]);

  const pending = rows.filter((row) => row.view.label_state !== 'AVAILABLE').length;
  const selected = rows.find((row) => row.identity === open);

  return (
    <div className="stack">
      <form className="card cockpit-selection" aria-label="Ledger filters" onSubmit={(event) => event.preventDefault()}>
        <label>Product{' '}
          <select className="control" value={selection.product ?? ''} onChange={(event) => update({ product: event.target.value || null })}>
            <option value="">All</option>
            {[...new Set([
              ...(overview.data?.products.map((item) => item.product) ?? []),
              ...rows.map((row) => row.payload.product),
              ...(selection.product ? [selection.product] : []),
            ])].map((product) => <option key={product} value={product}>{product}</option>)}
          </select>
        </label>
        <label>Model{' '}
          <input className="control" value={selection.model ?? ''} placeholder="all models" onChange={(event) => update({ model: event.target.value || null })} />
        </label>
        <span className="muted">
          {rows.length} loaded{page?.has_more ? ' (more available)' : ''}: {rows.length - pending} realized, {pending} pending
        </span>
      </form>
      {state === 'error' && error && <ErrorState error={error} onRetry={() => load()} />}
      {state === 'loading' && rows.length === 0 && <LoadingState label="Loading predictions" />}
      {state !== 'error' && !(state === 'loading' && rows.length === 0) && rows.length === 0 && (
        <EmptyState title="No prediction recorded" detail="The ledger is empty for this selection." />
      )}
      {rows.length > 0 && (
        <div className="grid grid-2">
          <section className="card" aria-label="Predictions">
            <table className="data">
              <thead>
                <tr><th>Decision</th><th>Product</th><th>Predicted</th><th>Realized</th>{expert && <th>Model</th>}<th /></tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.identity} aria-selected={open === row.identity}>
                    <td><Timestamp value={row.payload.decision_at} /></td>
                    <td>{row.payload.product}</td>
                    <td>{formatRatio(row.payload.outputs.return, 3, 'not provided')}</td>
                    <td>{row.view.latest_label ? formatRatio(row.view.latest_label.value, 3) : <Badge tone="warn">PENDING</Badge>}</td>
                    {expert && <td>{row.payload.model_id}</td>}
                    <td><button className="control" onClick={() => setOpen(row.identity)}>Detail</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {page?.has_more && (
              <button className="control" style={{ marginTop: 8 }} disabled={state === 'loading'}
                onClick={() => page.next_cursor && load(page.next_cursor)}>
                {state === 'loading' ? 'Loading…' : 'Load more'}
              </button>
            )}
          </section>
          <section className="card" aria-label="Selected prediction">
            {selected ? <Detail key={selected.identity} identity={selected.identity} expert={expert} />
              : <EmptyState title="Select a prediction" detail="Inputs, label history and hashes appear here." />}
          </section>
        </div>
      )}
    </div>
  );
}
