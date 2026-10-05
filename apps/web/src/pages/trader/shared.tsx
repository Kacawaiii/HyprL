import type { ReactNode } from 'react';
import { Badge, ErrorState, LoadingState } from '../../components/States';
import type { QueryResult } from '../../state/useQuery';
import { RUN_STATE_TEXT, type RunState } from '../../lib/trader';

export const pct = (p: number | null | undefined, digits = 0) => (p === null || p === undefined ? 'not available' : `${(p * 100).toFixed(digits)} %`);
export const signed = (p: number | null | undefined, digits = 2) =>
  p === null || p === undefined ? 'not available' : `${p >= 0 ? '+' : ''}${(p * 100).toFixed(digits)} %`;
export const when = (iso: string) => iso.replace('T', ' ').replace(/(\.\d+)?(Z|\+00:00)$/, ' UTC');

/** A query rendered with the cockpit's loading and error states; the rest of the page keeps working. */
export function Query<T>({ query, label, children }: { query: QueryResult<T>; label: string; children: (data: T) => ReactNode }) {
  if (query.status === 'loading' && query.data === undefined) return <LoadingState label={label} />;
  if (query.error && query.data === undefined) return <ErrorState error={query.error} onRetry={query.refetch} />;
  return <>{query.data === undefined ? null : children(query.data)}</>;
}

export function SyntheticBadge() {
  return <Badge tone="warn">SYNTHETIC DEMO DATA</Badge>;
}

export function RunBanner({ state, date }: { state: RunState; date: string }) {
  const text = RUN_STATE_TEXT[state];
  if (state === 'complete') return null;
  return (
    <div className="state" role="status" aria-label="Run state">
      <strong>{text.title} — {date}</strong>
      <span>{text.detail}</span>
    </div>
  );
}

export function Direction({ view }: { view: 'UP' | 'DOWN' | 'ABSTAIN' }) {
  const symbol = view === 'UP' ? '▲' : view === 'DOWN' ? '▼' : '◆';
  return <span><span aria-hidden="true">{symbol} </span>{view === 'UP' ? 'Up' : view === 'DOWN' ? 'Down' : 'No view'}</span>;
}
