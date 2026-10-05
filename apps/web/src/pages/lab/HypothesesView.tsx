/** Hypothesis registry: what was predicted, how it can fail, the criteria fixed beforehand and every trial, including null results. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../../api/client';
import type { HypothesisRow, PageInfo } from '../../api/labTypes';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../../components/States';
import { useCockpit } from '../../state/useCockpit';
import { useQuery } from '../../state/useQuery';
import { QueryBoundary, Synthetic, Timestamp } from './shared';

function Readiness() {
  const comparison = useQuery('research:comparison', (signal) => apiClient.getResearchComparison(signal));
  const proposals = useQuery('research:proposals', (signal) => apiClient.getResearchProposals(signal));
  return (
    <section className="card" aria-label="Prices versus prices plus events">
      <h2 className="card-title">Prices only vs prices + events</h2>
      <QueryBoundary query={comparison} label="Loading comparison readiness">
        {(data) => (
          <div className="stack">
            <p>
              <Badge tone={data.state === 'WAITING_DATA' ? 'warn' : 'off'}>{data.state}</Badge>{' '}
              {data.paired_decisions} paired admissible decisions. Execution is {data.execution_enabled ? 'enabled' : 'disabled'}.
              Crypto is the primary objective ({data.crypto_role}); equities are {data.equity_role}. Studied windows stay {data.studied_windows}.
            </p>
            <div className="grid grid-2">
              <div><h3 className="card-title">Why it cannot run</h3><ul>{data.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>
              <div><h3 className="card-title">Next actions</h3><ul>{data.actions.map((action) => <li key={action}>{action}</li>)}</ul></div>
            </div>
            <p className="lab-note">Protocol <Hash value={data.protocol_hash} chars={16} /> · criteria <Hash value={data.criteria_hash} chars={16} /></p>
          </div>
        )}
      </QueryBoundary>
      <QueryBoundary query={proposals} label="Loading proposal engine">
        {(data) => (
          <p className="lab-note" style={{ marginTop: 8 }}>
            Proposal engine {data.engine}: at most {data.max_proposals} proposals, {data.max_trials_per_hypothesis} trials per hypothesis,
            {' '}{data.external_model_calls} external model calls, execution {data.execution_enabled ? 'enabled' : 'disabled'}. {data.preparation}.
          </p>
        )}
      </QueryBoundary>
    </section>
  );
}

function Detail({ identity, expert }: { identity: string; expert: boolean }) {
  const detail = useQuery(`research:hypothesis:${identity}`, (signal) => apiClient.getResearchHypothesis(identity, signal));
  return (
    <QueryBoundary query={detail} label="Loading hypothesis">
      {(data) => {
        const hypothesis = data.payload;
        return (
          <div className="stack" aria-label="Hypothesis detail">
            <div className="row" style={{ flexWrap: 'wrap' }}>
              <strong>{hypothesis.hypothesis_id}</strong>
              <Badge tone="off">{hypothesis.scope}</Badge>
              {hypothesis.synthetic && <Synthetic />}
            </div>
            <dl>
              <div className="kv"><dt>Statement</dt><dd>{hypothesis.statement}</dd></div>
              <div className="kv"><dt>Mechanism</dt><dd>{hypothesis.mechanism}</dd></div>
              <div className="kv"><dt>Falsified if</dt><dd>{hypothesis.falsification}</dd></div>
              <div className="kv"><dt>Target / horizon</dt><dd>{hypothesis.target} · {hypothesis.horizon_seconds / 3600} h</dd></div>
              <div className="kv"><dt>Sources</dt><dd>{hypothesis.sources.join(', ')}</dd></div>
              <div className="kv"><dt>Baselines</dt><dd>{hypothesis.baselines.join(', ')}</dd></div>
              <div className="kv"><dt>Criteria hash (frozen)</dt><dd><Hash value={data.criteria_hash} chars={20} /></dd></div>
            </dl>
            <table className="data" aria-label="Trial history">
              <caption className="lab-note" style={{ textAlign: 'left' }}>Trial history: nothing is removed, including negative and abandoned trials</caption>
              <thead><tr><th>Trial</th><th>State</th><th>Outcome</th><th>Recorded</th></tr></thead>
              <tbody>
                {data.trial_history.length === 0 && <tr><td colSpan={4}>No trial recorded for this hypothesis.</td></tr>}
                {data.trial_history.map((trial) => (
                  <tr key={trial.identity}><td>{trial.payload.trial_id}</td><td>{trial.payload.state}</td><td>{trial.payload.outcome}</td><td><Timestamp value={trial.recorded_at} /></td></tr>
                ))}
              </tbody>
            </table>
            {expert && (
              <details>
                <summary>Decision criteria, fixed before results</summary>
                <pre className="code-block">{JSON.stringify(hypothesis.decision_criteria, null, 2)}</pre>
              </details>
            )}
            {expert && (
              <details>
                <summary>Budgets</summary>
                <pre className="code-block">{JSON.stringify(hypothesis.budgets, null, 2)}</pre>
              </details>
            )}
          </div>
        );
      }}
    </QueryBoundary>
  );
}

export function HypothesesView() {
  const { selection } = useCockpit();
  const [rows, setRows] = useState<HypothesisRow[]>([]);
  const [page, setPage] = useState<PageInfo | null>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState<Error | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const asOf = useRef('');
  const controller = useRef<AbortController | null>(null);

  const load = useCallback((cursor?: string) => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    if (!cursor) asOf.current = new Date().toISOString();
    setState('loading');
    apiClient.getResearchHypotheses({ limit: 20, cursor, asOf: asOf.current }, next.signal).then((result) => {
      setRows((current) => (cursor ? [...current, ...result.records] : result.records));
      setPage(result.page);
      setState('ready');
    }).catch((failure: Error) => {
      if (next.signal.aborted) return;
      setError(failure);
      setState('error');
    });
  }, []);

  useEffect(() => {
    load();
    return () => controller.current?.abort();
  }, [load]);

  const selected = rows.find((row) => row.identity === open) ?? rows[0];
  return (
    <div className="stack">
      <Readiness />
      <section className="card" aria-label="Hypotheses">
        <h2 className="card-title">Hypotheses</h2>
        {state === 'error' && error && <ErrorState error={error} onRetry={() => load()} />}
        {state === 'loading' && rows.length === 0 && <LoadingState label="Loading hypotheses" />}
        {state === 'ready' && rows.length === 0 && <EmptyState title="No hypothesis registered" />}
        {rows.length > 0 && (
          <div className="grid grid-2">
            <div>
              <ul className="lab-list" aria-label="Hypothesis list">
                {rows.map((row) => (
                  <li key={row.identity}>
                    <button className="row-button" aria-current={selected?.identity === row.identity} onClick={() => setOpen(row.identity)}>
                      <span style={{ flex: 1 }}>{row.payload.hypothesis_id}</span><Badge tone="off">{row.payload.scope}</Badge>
                    </button>
                  </li>
                ))}
              </ul>
              {page?.has_more && (
                <button className="control" style={{ marginTop: 8 }} disabled={state === 'loading'}
                  onClick={() => page.next_cursor && load(page.next_cursor)}>Load more</button>
              )}
            </div>
            {selected && <Detail key={selected.identity} identity={selected.identity} expert={selection.mode === 'expert'} />}
          </div>
        )}
      </section>
    </div>
  );
}
