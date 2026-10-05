import { Link } from 'react-router-dom';
import { ScoreCard } from '../../components/ScoreCard';
import { Badge, EmptyState, Hash } from '../../components/States';
import { Num } from '../../components/Num';
import { dataQualityScore, modelScore, riskScore, signalScore, inPeriod } from '../../lib/cockpit';
import { ChartSection, ProtectionCard, UncertaintyNote } from './shared';
import type { CockpitData } from './useCockpitData';

const SIDE_CLASS: Record<string, string> = { LONG: 'positive', SHORT: 'negative', FLAT: 'muted' };

export function BeginnerView({ data, carry }: { data: CockpitData; carry: string }) {
  const { summary, period, decisions, targets, replayProduct, replay, fomc, events } = data;
  const latest = period ? decisions.find((row) => inPeriod(row.timestamp, period)) : undefined;
  const target = period ? targets.find((row) => inPeriod(row.timestamp, period)) : undefined;
  const projection = latest ? data.projections.find((item) => item.decisionAt === latest.timestamp) : undefined;
  const metrics = replayProduct?.metrics;
  const spec = data.system.data?.risk_engine;
  const scores = [
    dataQualityScore(summary),
    signalScore(decisions, period),
    modelScore(replayProduct?.prediction_quality, replay.data?.window),
    riskScore(targets, period, spec?.max_long_exposure ?? null),
  ];
  const detail = (path: string, label: string) => <Link to={{ pathname: path, search: carry }}>{label} →</Link>;
  return (
    <div className="stack">
      <section className="card" aria-label="Product summary">
        <h2 className="card-title">{data.product ?? 'Product'} · situation</h2>
        {summary ? (
          <p>
            {summary.rows.toLocaleString()} hourly bars from {summary.first_open.slice(0, 10)} to {summary.last_open.slice(0, 10)},
            {' '}{summary.missing_openings} declared gaps. No live price feed: the latest price is{' '}
            <strong>{summary.latest_price ?? 'not available'}</strong>; everything below is a recorded study, not a live market.
          </p>
        ) : <EmptyState title="No product summary" />}
        <p className="row" style={{ flexWrap: 'wrap' }}>
          <Badge tone="warn">PAPER / SHADOW ONLY</Badge> <Badge tone="off">NO REAL MONEY</Badge>{' '}
          <Badge tone="off">NO CONFIRMED EDGE</Badge>
        </p>
      </section>

      <section className="card" aria-label="Prices, predictions, decisions, executions and events">
        <h2 className="card-title">Prices, prediction and decisions</h2>
        <ChartSection data={data} />
      </section>

      <section className="grid grid-2">
        <article className="card" aria-label="Prediction">
          <h2 className="card-title">Prediction · next {data.horizon} hours</h2>
          {latest ? (
            <>
              <div className="metric"><Num value={latest.prediction} kind="percent" digits={3} /></div>
              <div className="metric-sub">forecast price move over {data.horizon} hourly bars, decided {latest.timestamp.slice(0, 16).replace('T', ' ')}</div>
              {projection ? (
                <dl style={{ margin: '10px 0 0' }}>
                  <div className="kv"><dt>Reference price</dt><dd>{projection.referencePrice} (close of the decision bar)</dd></div>
                  <div className="kv"><dt>Projected price</dt><dd>{projection.projectedPrice.toFixed(2)}</dd></div>
                </dl>
              ) : <p className="metric-sub">No reference price: the decision bar is outside the loaded prices, so no projection is drawn.</p>}
              <p className="metric-sub">
                Simple explanation: the model turns recent price history into one number, the expected price change
                over the next hours. It is a point estimate with no stated confidence or range.
              </p>
            </>
          ) : <EmptyState title="No prediction in this period" detail="The persisted out-of-sample run has no decision here." />}
          <UncertaintyNote data={data} />
        </article>

        <article className="card" aria-label="Signal and exposure">
          <h2 className="card-title">Signal and proposed exposure</h2>
          {latest ? (
            <div className="metric"><span className={SIDE_CLASS[latest.direction]}>{latest.direction}</span></div>
          ) : <div className="metric muted">none</div>}
          {target ? (
            <dl style={{ margin: '10px 0 0' }}>
              <div className="kv"><dt>Proposed exposure</dt><dd><Num value={target.target_exposure} kind="percent" /> of NAV ({target.side})</dd></div>
              <div className="kv"><dt>Position cap</dt><dd>{spec?.max_long_exposure ?? '—'}</dd></div>
            </dl>
          ) : <p className="metric-sub">No exposure target in this period.</p>}
          <p className="metric-sub">A proposal for the paper study. It is not an order and nothing is sent to a broker.</p>
        </article>
      </section>

      <section className="grid grid-2">
        <ProtectionCard data={data} />
        <section className="card" aria-label="Paper result">
          <h2 className="card-title">Paper result · frozen replay</h2>
          {metrics ? (
            <dl style={{ margin: 0 }}>
              <div className="kv"><dt>Net return</dt><dd><Num value={metrics.net_return} kind="percent" /></dd></div>
              <div className="kv"><dt>Max drawdown</dt><dd><Num value={metrics.max_drawdown} kind="percent" /></dd></div>
              <div className="kv"><dt>Costs paid</dt><dd><Num value={metrics.total_execution_cost} kind="money" /></dd></div>
              <div className="kv"><dt>Window</dt><dd>{replay.data?.window?.start.slice(0, 10)} → {replay.data?.window?.end.slice(0, 10)}</dd></div>
            </dl>
          ) : <EmptyState title="No replay result for this product" />}
          <p className="metric-sub">Exploratory replay of a past window, with fees and slippage. Not a forecast of future results.</p>
        </section>
      </section>

      <section className="card" aria-label="Relevant events">
        <h2 className="card-title">Relevant events</h2>
        {events.length === 0 ? (
          <p className="muted">{fomc.status === 'error' ? 'The event store could not be read.' : 'No event is recorded in the available store snapshot.'}</p>
        ) : (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {events.slice(0, 5).map(({ item }) => (
              <li key={item.sid}>
                {item.declared_release_at?.slice(0, 10)} · {item.title ?? 'untitled'} <Badge tone="off">FOMC · declared release</Badge>
              </li>
            ))}
          </ul>
        )}
        <p className="metric-sub">
          Macro context from the FOMC store; its relevance to {data.product ?? 'this product'} is not measured here,
          and no event-to-price relationship is claimed. Company filings (SEC) do not apply to a crypto product.
        </p>
      </section>

      <section aria-label="Scores">
        <h2 className="card-title">Scores (separate by design)</h2>
        <div className="grid grid-3">{scores.map((score) => <ScoreCard key={score.key} score={score} />)}</div>
      </section>

      <section className="card" aria-label="Data and model health">
        <h2 className="card-title">Data and model health</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Prices</dt><dd>{summary ? `${summary.missing_openings} declared gaps` : '—'}</dd></div>
          <div className="kv"><dt>Signal run</dt><dd>{data.signals.data?.available ? <Badge tone="ok">PERSISTED · VERIFIED</Badge> : <Badge tone="warn">NONE</Badge>}</dd></div>
          <div className="kv"><dt>Replay determinism</dt><dd>{replay.data?.determinism?.verified ? <Badge tone="ok">VERIFIED</Badge> : <Badge tone="warn">NOT VERIFIED</Badge>}</dd></div>
          <div className="kv"><dt>Event store</dt><dd>{data.fomcStatus.data?.status ?? '—'}</dd></div>
          <div className="kv"><dt>Model spec</dt><dd><Hash value={data.system.data?.signal_engine.spec_hash} /></dd></div>
        </dl>
      </section>

      <nav className="row" style={{ flexWrap: 'wrap', gap: 16 }} aria-label="Detail">
        {detail('/signals', 'Signals detail')}
        {detail('/risk', 'Risk detail')}
        {detail('/paper', 'Paper detail')}
        {detail('/events', 'Events detail')}
        {detail('/system', 'System health')}
      </nav>
    </div>
  );
}
