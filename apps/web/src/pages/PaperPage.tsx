/**
 * Live / Paper — shadow trading.
 *
 * Everything here is simulated. There is no broker, no exchange account and no
 * money, and the page says so above the first number rather than in a footnote.
 *
 * There is deliberately no BUY, SELL, or EXECUTE control: the API is read-only
 * and the session is started from the command line. A cockpit that can start a
 * trading process is one XSS away from doing it by accident.
 */

import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { DataTable, type Column } from '../components/DataTable';
import { LineChart } from '../components/LineChart';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import type { EmbargoState, PaperEvent, PaperProductState } from '../api/types';

const money = (value: string | undefined) =>
  value === undefined ? '—' : Number(value).toLocaleString('en-US',
    { maximumFractionDigits: 2 });
const pct = (value: string | undefined) =>
  value === undefined ? '—' : `${(Number(value) * 100).toFixed(2)} %`;

function Badges() {
  return (
    <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
      <span className="badge" data-tone="warn">SHADOW MODE</span>
      <span className="badge" data-tone="warn">NO REAL MONEY</span>
      <span className="badge" data-tone="off">NO BROKER</span>
      <span className="badge" data-tone="off">NO EXCHANGE ACCOUNT</span>
    </div>
  );
}

function HoldoutBanner({ embargo }: { embargo: EmbargoState }) {
  const active = embargo.embargoed;
  return (
    <section className="card">
      <h2 className="card-title">Protected research holdout</h2>
      <div className="row" style={{ gap: 10, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <span className="badge" data-tone={active ? 'warn' : 'ok'}>
          {active ? 'EMBARGO ACTIVE' : 'UNOBSERVED'}
        </span>
        <span className="metric-sub">
          {embargo.start.slice(0, 10)} → {embargo.end.slice(0, 10)}
        </span>
      </div>
      <p className="metric-sub" style={{ marginTop: 10 }}>{embargo.reason}</p>
      <p className="metric-sub">
        The window is reserved for a single confirmatory evaluation of Benchmark V2.
        No candle inside it is requested, stored, predicted on, or shown here — the
        session stands down by itself at the boundary.
      </p>
    </section>
  );
}

function Field({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="kv"><dt>{label}</dt><dd className={tone}>{value}</dd></div>
  );
}

function ProductCard({ state }: { state: PaperProductState }) {
  const portfolio = state.portfolio ?? undefined;
  const prediction = state.last_prediction ?? undefined;
  const signal = state.last_signal ?? undefined;
  const target = state.last_target ?? undefined;
  const fill = state.last_fill ?? undefined;

  const equity = useQuery(
    state.available ? `paper-equity-${state.product}` : null,
    (signalToken) => apiClient.getPaperEquity(state.product, 500, signalToken),
  );
  const points = (equity.data?.series ?? []).map((point) => ({
    timestamp: point.timestamp, value: Number(point.equity),
  }));

  return (
    <section className="card">
      <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <h2 className="card-title">{state.product}</h2>
        <span className="badge"
              data-tone={state.status === 'RUNNING' ? 'ok'
                : state.status === 'EMBARGOED' ? 'warn' : 'off'}>
          {state.status}
        </span>
      </div>

      {!state.available ? (
        <EmptyState title="No shadow activity for this product"
                    detail={state.reason ?? 'Start a session from the command line.'} />
      ) : (
        <>
          <dl style={{ margin: '10px 0 0' }}>
            <Field label="Latest closed candle"
                   value={state.last_candle?.bar_open_at?.replace('T', ' ').slice(0, 16) ?? '—'} />
            <Field label="Latest prediction" value={prediction?.prediction ?? '—'} />
            <Field label="Signal" value={signal?.direction ?? '—'} />
            <Field label="Strength" value={signal?.strength ?? '—'} />
            <Field label="Target exposure" value={target?.target_exposure ?? '—'} />
            <Field label="Paper position" value={portfolio?.position_quantity ?? '—'} />
            <Field label="Paper equity" value={money(portfolio?.equity)} />
            <Field label="Cumulative fees" value={money(portfolio?.cumulative_fees)} />
            <Field label="Slippage cost"
                   value={money(portfolio?.cumulative_slippage_cost)} />
            <Field label="Fills" value={String(portfolio?.fill_count ?? 0)} />
            <Field label="Gaps seen" value={String(state.gap_count)} />
            <Field label="Last fill"
                   value={fill ? `${fill.side} @ ${Number(fill.fill_price).toFixed(2)}` : '—'} />
            <Field label="Last event" value={state.last_event_at ?? '—'} />
          </dl>
          {points.length > 1 ? (
            <div style={{ marginTop: 12 }}>
              <LineChart points={points} height={160} label={`${state.product} paper equity`} />
            </div>
          ) : (
            <p className="metric-sub" style={{ marginTop: 12 }}>
              Not enough marks yet to draw an equity curve.
            </p>
          )}
        </>
      )}
    </section>
  );
}

const EVENT_COLUMNS: Column<PaperEvent>[] = [
  { key: 'event_at', header: 'At', width: '1.6fr',
    render: (row) => row.event_at.replace('T', ' ').slice(0, 19) },
  { key: 'event_type', header: 'Event', width: '1.6fr', render: (row) => row.event_type },
  { key: 'product', header: 'Product', width: '1fr', render: (row) => row.product ?? '—' },
  { key: 'natural_key', header: 'Key', width: '1.6fr',
    render: (row) => (row.natural_key ?? '—').replace('T', ' ').slice(0, 16) },
];

export function PaperPage() {
  const status = useQuery('paper-status', (signal) => apiClient.getPaperStatus(signal));
  const products = useQuery('paper-products', (signal) =>
    apiClient.getPaperProducts(signal));
  const events = useQuery('paper-events', (signal) =>
    apiClient.getPaperEvents(undefined, 100, signal));

  if (status.status === 'loading') return <LoadingState label="Loading shadow session" />;
  if (status.status === 'error' && status.error) {
    return <ErrorState error={status.error} onRetry={status.refetch} />;
  }
  if (!status.data) return null;
  const data = status.data;
  const execution = data.paper_execution;
  const firstEmbargo = Object.values(data.embargo)[0];

  return (
    <div className="stack">
      <Badges />

      <section className="card">
        <h2 className="card-title">Session</h2>
        <dl style={{ margin: 0 }}>
          <Field label="State" value={data.available ? 'RUNNING' : 'STOPPED'} />
          <Field label="Reason" value={data.reason ?? '—'} />
          <Field label="Session id" value={data.session?.session_id ?? '—'} />
          <Field label="Started" value={data.session?.started_at ?? '—'} />
          <Field label="Events" value={String(data.events ?? 0)} />
          <Field label="Real money" value={String(data.real_money)} tone="warning" />
          <Field label="Broker connected" value={String(data.broker_connected)}
                 tone="warning" />
        </dl>
        {!data.available ? (
          <EmptyState
            title="No shadow session is running"
            detail="Start one from the command line: ./scripts/paper_shadow.sh start — the cockpit is read-only and cannot start it."
          />
        ) : null}
      </section>

      {firstEmbargo ? <HoldoutBanner embargo={firstEmbargo} /> : null}

      {(products.data?.products ?? []).map((state) => (
        <ProductCard key={state.product} state={state} />
      ))}

      <section className="card">
        <h2 className="card-title">Event feed</h2>
        <DataTable rows={events.data?.events ?? []} columns={EVENT_COLUMNS}
                   height={320} empty="No events recorded" />
        <p className="metric-sub">
          Showing the most recent {events.data?.page.returned ?? 0} events. The log is
          append-only and hash-chained; the full history is never loaded here.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">Paper execution contract</h2>
        <dl style={{ margin: 0 }}>
          <Field label="Fee" value={pct(execution.fee_rate)} />
          <Field label="Slippage" value={pct(execution.slippage_rate)} />
          <Field label="Initial equity"
                 value={`${money(execution.initial_equity)} ${execution.currency}`} />
          <Field label="Fill price" value={execution.fill_price_policy} />
          <Field label="Fill recorded" value={execution.fill_observation_policy} />
          <Field label="Terminal liquidation"
                 value={String(execution.terminal_liquidation)} />
          <Field label="Cost model" value={execution.cost_model} tone="warning" />
          <Field label="Model optimized" value={String(data.paper_model_optimized)}
                 tone="warning" />
          <div className="kv"><dt>Paper execution spec</dt>
            <dd><Hash value={execution.spec_hash} /></dd></div>
          <div className="kv"><dt>Shadow model spec</dt>
            <dd><Hash value={data.paper_model_spec_hash} /></dd></div>
          <div className="kv"><dt>Signal spec</dt>
            <dd><Hash value={data.signal_spec_hash} /></dd></div>
          <div className="kv"><dt>Risk spec</dt>
            <dd><Hash value={data.risk_spec_hash} /></dd></div>
        </dl>
        <ul className="metric-sub" style={{ marginTop: 10, paddingLeft: 18 }}>
          {execution.differs_from_backtest.map((line) => <li key={line}>{line}</li>)}
        </ul>
      </section>
    </div>
  );
}
