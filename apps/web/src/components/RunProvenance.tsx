/** Shared by the Signals and Risk pages: which persisted run is shown, how it was
 *  verified, and how to page back through it. Everything here is displayed as the
 *  backend served it; nothing is decided or recomputed in the browser. */
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import type { Page, RunProvenance } from '../api/types';
import { Hash } from './States';

/** The products come from the markets registry; the first one is the server's default (null). */
export function ProductSelect({ value, shown, onChange }: {
  value: string | null; shown: string | undefined; onChange: (v: string) => void;
}) {
  const markets = useQuery('markets', (signal) => apiClient.getMarkets(signal));
  const products = markets.data?.products.map((item) => item.product) ?? [];
  if (products.length === 0) return null;
  return (
    <label className="row" style={{ gap: 8 }}>
      <span className="muted">Product</span>
      <select
        className="control"
        value={value ?? shown ?? products[0]}
        onChange={(event) => onChange(event.target.value)}
      >
        {products.map((product) => <option key={product} value={product}>{product}</option>)}
      </select>
    </label>
  );
}

export function RunProvenanceCard({ run, title }: { run: RunProvenance; title: string }) {
  if (!run.out_of_sample) return null;
  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      <p className="metric-sub" style={{ marginTop: 0 }}>
        <strong>Out-of-sample: {run.out_of_sample}</strong>. Every row comes from a model fitted only on
        data before it. Exploratory, not confirmatory, and not a live or paper result.
      </p>
      <dl style={{ margin: 0 }}>
        <div className="kv"><dt>Product</dt><dd>{run.product}</dd></div>
        <div className="kv"><dt>Rows</dt><dd>{run.counts?.decisions} over {run.counts?.folds} folds</dd></div>
        <div className="kv"><dt>Window</dt><dd>{run.window?.first} → {run.window?.last}</dd></div>
        <div className="kv"><dt>Verified against</dt><dd>{run.verified_against}</dd></div>
        <div className="kv"><dt>Signal series</dt><dd><Hash value={run.signal_series_hash ?? ''} /></dd></div>
        <div className="kv"><dt>Target series</dt><dd><Hash value={run.position_target_series_hash ?? ''} /></dd></div>
        <div className="kv"><dt>Protocol</dt><dd>{run.protocol?.benchmark_protocol}</dd></div>
        <div className="kv"><dt>Benchmark spec</dt><dd><Hash value={run.protocol?.benchmark_spec_hash ?? ''} /></dd></div>
        <div className="kv"><dt>Corpus</dt><dd><Hash value={run.corpus?.corpus_content_hash ?? ''} /></dd></div>
      </dl>
    </section>
  );
}

/** Older rows appended below the first page, newest first, following the served cursor. */
export function useOlderRows<V extends { page: Page }, R>(
  resetKey: string,
  first: V | undefined,
  fetchPage: (cursor: string) => Promise<V>,
  rowsOf: (view: V) => R[],
) {
  const [older, setOlder] = useState<R[]>([]);
  const [cursor, setCursor] = useState<string | null | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Error | undefined>();
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    setOlder([]);
    setCursor(undefined);
    setFailure(undefined);
    setBusy(false);
  }, [resetKey]);

  const next = cursor === undefined ? first?.page.next_cursor ?? null : cursor;

  const loadOlder = () => {
    if (!next || busy) return;
    const mine = generation.current;
    setBusy(true);
    fetchPage(next)
      .then((view) => {
        if (mine !== generation.current) return;
        setOlder((rows) => [...rows, ...rowsOf(view)]);
        setCursor(view.page.next_cursor);
        setBusy(false);
      })
      .catch((error: Error) => {
        if (mine !== generation.current) return;
        setFailure(error);
        setBusy(false);
      });
  };

  return { older, hasMore: next !== null, busy, failure, loadOlder };
}

/** The backend's reason when it says more than the title already does, else the standing note. */
export function unavailableDetail(reason: string | undefined, title: string, fallback: string): string {
  return reason && reason.toLowerCase() !== title.toLowerCase() ? reason : fallback;
}
