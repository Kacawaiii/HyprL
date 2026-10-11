/**
 * Agent trader: the prospective PAPER trader's views, outcomes and health.
 * Read-only. The page never starts a run, never sends an order and never invents a value the API did not serve.
 */
import { useSearchParams } from 'react-router-dom';
import { Badge, ErrorState } from '../components/States';
import { useCockpit } from '../state/useCockpit';
import { NOT_A_CLAIM, assetsOf, runState } from '../lib/trader';
import { BeginnerTrader } from './trader/BeginnerTrader';
import { ExpertTrader } from './trader/ExpertTrader';
import { RunBanner, SyntheticBadge } from './trader/shared';
import { useTraderData } from './trader/useTraderData';
import { TraderDecisionCards } from './trader/TraderDecisionCards';

const DAY = /^\d{4}-\d{2}-\d{2}$/;

export function TraderPage() {
  const { selection, update } = useCockpit();
  const [params, setParams] = useSearchParams();
  const raw = params.get('date');
  const requested = raw && DAY.test(raw) ? raw : null;
  const onDate = (value: string | null) => setParams((current) => {
    const next = new URLSearchParams(current);
    if (value) next.set('date', value); else next.delete('date');
    return next;
  });
  const onProduct = (product: string | null) => update({ product });
  const date = requested ?? new Date().toISOString().slice(0, 10);
  const data = useTraderData(date);
  const run = [...(data.today.data?.runs ?? [])].reverse().find((r) => r.payload.status === 'COMPLETE' || r.payload.status === 'DEGRADED')?.payload ?? null;
  const state = runState((data.today.data?.runs ?? []).map((r) => r.payload.status));
  const assets = assetsOf(run?.decision?.views ?? []);
  const known = Object.keys(data.series.data?.series ?? {}).filter((a) => assets.length === 0 || assets.includes(a));
  const choices = assets.length ? assets : known;
  const asset = selection.product && choices.includes(selection.product) ? selection.product : choices[0] ?? null;
  return (
    <div className="stack">
      <section className="card" aria-label="Agent trader">
        <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'space-between' }}>
          <h1 className="card-title" style={{ margin: 0 }}>Agent trader</h1>
          <div className="row" style={{ flexWrap: 'wrap' }}>
            <Badge tone="warn">PAPER ONLY · NO PROVEN EDGE</Badge>
            {(data.synthetic || run?.synthetic) && <SyntheticBadge />}
            <label>Day <input className="control" type="date" value={date} onChange={(e) => onDate(e.target.value || null)} /></label>
            <label>Asset <select className="control" value={asset ?? ''} onChange={(e) => onProduct(e.target.value || null)} disabled={choices.length === 0}>
              {choices.length === 0 && <option value="">none yet</option>}
              {choices.map((a) => <option key={a} value={a}>{a}</option>)}
            </select></label>
          </div>
        </div>
        <p role="note" style={{ margin: '8px 0 0' }}><strong>Read this first.</strong> {NOT_A_CLAIM}</p>
      </section>
      {data.today.error && data.today.data === undefined ? <ErrorState error={data.today.error} onRetry={data.today.refetch} /> : <RunBanner state={state} date={date} />}
      {run && <TraderDecisionCards run={run} rows={data.rows} asset={asset} />}
      {selection.mode === 'expert'
        ? <ExpertTrader data={data} run={run} asset={asset} />
        : <BeginnerTrader data={data} run={run} asset={asset} />}
    </div>
  );
}
