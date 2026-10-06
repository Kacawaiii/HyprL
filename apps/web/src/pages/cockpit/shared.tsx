import { CockpitChart } from '../../components/CockpitChart';
import { Link, useSearchParams } from 'react-router-dom';
import { EmptyState, ErrorState, LoadingState } from '../../components/States';
import { carrySelection, fillsInPeriod } from '../../lib/cockpit';
import type { CockpitData } from './useCockpitData';

export function ChartSection({ data }: { data: CockpitData }) {
  const { period, candles, candleList, decisions, projections, fills, events, tooLong } = data;
  if (!period) return <EmptyState title="No valid period" detail="Pick a start day on or before the end day, or clear the selection." />;
  if (tooLong) {
    return <EmptyState title="Period too long for one chart"
      detail="Pick at most 41 days (1000 hourly bars). The chart never samples or aggregates prices." />;
  }
  if (candles.status === 'loading') return <LoadingState label="Loading prices" />;
  if (candles.status === 'error' && candles.error) return <ErrorState error={candles.error} onRetry={candles.refetch} />;
  if (candleList.length === 0) return <EmptyState title="No prices in this period" />;
  const from = Date.parse(period.from);
  const to = Date.parse(period.to);
  const inside = decisions.filter((row) => Date.parse(row.timestamp) >= from && Date.parse(row.timestamp) < to);
  const periodFills = fillsInPeriod(fills.fills, period);
  return (
    <>
      <CockpitChart showProjection input={{
        period, candles: candleList, decisions: inside, fills: periodFills,
        projections: projections.filter((item) => Date.parse(item.decisionAt) >= from && Date.parse(item.decisionAt) < to),
        events,
      }} />
      <p className="metric-sub">
        {candleList.length} closes · {inside.length} decisions ({data.signals.data?.available ? 'walk-forward, out of sample' : 'no persisted run'}) ·{' '}
        {periodFills.length} executions (frozen paper replay{fills.complete ? '' : ', first pages only'}) ·{' '}
        {events.filter((item) => Date.parse(item.at) >= from && Date.parse(item.at) < to).length} events in period.
        {' '}The replay window and the signal run are separate studies; they overlap only where both are shown.
      </p>
    </>
  );
}

export function ProtectionCard({ data }: { data: CockpitData }) {
  const [params] = useSearchParams();
  const level = data.protection;
  return (
    <section className="card" aria-label="Take profit and stop loss">
      <h2 className="card-title">Take profit / stop loss</h2>
      {level ? (
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Take profit</dt><dd>{level.take_profit ?? 'not provided'}</dd></div>
          <div className="kv"><dt>Stop loss</dt><dd>{level.stop_loss ?? 'not provided'}</dd></div>
          <div className="kv"><dt>Origin</dt><dd>{level.origin}</dd></div>
          <div className="kv"><dt>Policy version</dt><dd>{level.policy_version}</dd></div>
          <div className="kv"><dt>Method</dt><dd>{level.method}</dd></div>
          <div className="kv"><dt>Gaps</dt><dd>{level.gap_treatment ?? 'not stated'}</dd></div>
          <div className="kv"><dt>Intrabar ambiguity</dt><dd>{level.intrabar_ambiguity ?? 'not stated'}</dd></div>
        </dl>
      ) : (
        <>
          <p><strong>Not provided.</strong></p>
          <p className="metric-sub">
            Neither the signal contract nor the risk contract V1 defines take-profit or stop-loss levels, and no
            levels are attached to this frozen reference. Separate policy evidence shows synthetic TP/SL
            with their origin, method and treatment of gaps and intrabar ambiguity.
          </p>
        </>
      )}
      <p><Link to={{ pathname: '/policies', search: carrySelection(params) }}>Calibration and TP/SL policy evidence →</Link></p>
    </section>
  );
}

export function UncertaintyNote({ data }: { data: CockpitData }) {
  return (
    <p className="metric-sub" aria-label="Probabilities and intervals">
      Probabilities and intervals: <strong>not provided</strong>. {data.model
        ? `The model declares ${data.model.outputs.join('; ')}; it does not declare ${data.model.notProvided.join(', ')}.`
        : ''}{' '}
      No calibration is claimed.
    </p>
  );
}
