/**
 * Economic backtests.
 *
 * Every number here is simulated under a synthetic cost contract, driven by
 * predictions that were already observed once. The badges are not decoration:
 * a page that shows equity curves without saying what they are is how a
 * research artefact quietly becomes a claim.
 */

import { useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { DataTable, type Column } from '../components/DataTable';
import { LineChart } from '../components/LineChart';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import type {
  BacktestEquity, BacktestFill, BacktestFillPage, BacktestSummary,
} from '../api/types';

const percent = (value: string | null) =>
  value === null ? '—' : `${(Number(value) * 100).toFixed(2)} %`;
const money = (value: string) =>
  Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 });
const decimals = (value: string | null, digits = 3) =>
  value === null ? '—' : Number(value).toFixed(digits);

function Badges() {
  return (
    <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
      <span className="badge" data-tone="warn">EXPLORATORY</span>
      <span className="badge" data-tone="warn">SYNTHETIC EXECUTION COSTS</span>
      <span className="badge" data-tone="off">NOT LIVE TRADING</span>
      <span className="badge" data-tone="warn">NO CONFIRMED EDGE</span>
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card" style={{ flex: 1, minWidth: 150 }}>
      <div className="card-title">{label}</div>
      <div className="metric">{value}</div>
      {sub ? <div className="metric-sub">{sub}</div> : null}
    </div>
  );
}

const FILL_COLUMNS: Column<BacktestFill>[] = [
  { key: 'timestamp', header: 'Time', width: '2fr',
    render: (row) => row.timestamp.replace('T', ' ').slice(0, 16) },
  { key: 'side', header: 'Side', width: '0.7fr', render: (row) => row.side },
  { key: 'reference_price', header: 'Reference', width: '1fr',
    render: (row) => Number(row.reference_price).toFixed(2) },
  { key: 'fill_price', header: 'Fill', width: '1fr',
    render: (row) => Number(row.fill_price).toFixed(2) },
  { key: 'quantity_delta', header: 'Δ Qty', width: '1fr',
    render: (row) => Number(row.quantity_delta).toFixed(6) },
  { key: 'fee', header: 'Fee', width: '0.9fr',
    render: (row) => Number(row.fee).toFixed(2) },
  { key: 'equity_after', header: 'Equity', width: '1.1fr',
    render: (row) => money(row.equity_after) },
];

export function BacktestsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const { data, status, error, refetch } = useQuery('backtests', (signal) =>
    apiClient.getBacktests(signal),
  );

  const runs = data?.runs ?? [];
  const active: BacktestSummary | undefined =
    runs.find((run) => run.product === selected) ?? runs[0];

  // A null key means "do not fetch": the equity and fill requests only exist
  // once a run does, so an empty state never triggers a round trip.
  const equity = useQuery<BacktestEquity>(
    active ? `backtest-equity-${active.version}-${active.product}` : null,
    (signal) =>
      apiClient.getBacktestEquity(active!.version, active!.product, 500, signal),
  );
  const fills = useQuery<BacktestFillPage>(
    active ? `backtest-fills-${active.version}-${active.product}` : null,
    (signal) =>
      apiClient.getBacktestFills(active!.version, active!.product, { limit: 100 }, signal),
  );

  if (status === 'loading') return <LoadingState label="Loading economic backtests" />;
  if (status === 'error' && error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data) return null;

  const spec = data.execution_spec;

  const contract = (
    <section className="card">
      <h2 className="card-title">Execution contract V1</h2>
      <dl style={{ margin: 0 }}>
        <div className="kv"><dt>Fee rate</dt><dd>{percent(spec.fee_rate)} per fill</dd></div>
        <div className="kv"><dt>Slippage</dt><dd>{percent(spec.slippage_rate)} against the trade</dd></div>
        <div className="kv"><dt>Initial equity</dt><dd>{money(spec.initial_equity)} {spec.currency}</dd></div>
        <div className="kv"><dt>Fill policy</dt><dd>{spec.fill_policy}</dd></div>
        <div className="kv"><dt>Mark policy</dt><dd>{spec.mark_policy}</dd></div>
        <div className="kv"><dt>Instrument</dt><dd>{spec.instrument_model}</dd></div>
        <div className="kv"><dt>Cost model</dt><dd className="warning">{spec.cost_model}</dd></div>
        <div className="kv"><dt>Optimized</dt><dd className="warning">{String(spec.optimized)}</dd></div>
        <div className="kv"><dt>Account specific</dt><dd className="warning">{String(spec.exchange_account_specific)}</dd></div>
        <div className="kv"><dt>Spec hash</dt><dd><Hash value={spec.execution_spec_hash} /></dd></div>
      </dl>
      <p className="metric-sub" style={{ marginTop: 12 }}>
        10 bps of fee and 5 bps of slippage are a synthetic infrastructure contract.
        They are not any exchange&apos;s schedule, they were not tuned, and they do not
        describe a real account.
      </p>
    </section>
  );

  if (!data.available || !active) {
    return (
      <div className="stack">
        <Badges />
        {contract}
        <section className="card">
          <h2 className="card-title">Backtests</h2>
          <EmptyState
            title="Economic Backtest Engine ready — no persisted run available."
            detail={data.reason ?? 'Nothing has been recorded, and nothing is fabricated here.'}
          />
        </section>
      </div>
    );
  }

  const metrics = active.metrics;
  const equitySeries = (equity.data?.series ?? []).map((point) => ({
    timestamp: point.timestamp,
    value: Number(point.equity),
  }));
  let peak = Number.NEGATIVE_INFINITY;
  const drawdownSeries = equitySeries.map((point) => {
    peak = Math.max(peak, point.value);
    return { timestamp: point.timestamp, value: peak > 0 ? point.value / peak - 1 : 0 };
  });
  const exposureSeries = (equity.data?.series ?? []).map((point) => ({
    timestamp: point.timestamp,
    value: Number(point.realized_exposure),
  }));

  return (
    <div className="stack">
      <Badges />

      <section className="card">
        <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <h2 className="card-title">Simulated portfolio</h2>
          <div className="row" style={{ gap: 6 }}>
            {runs.map((run) => (
              <button
                key={`${run.version}-${run.product}`}
                type="button"
                className="control"
                aria-pressed={run.product === active.product}
                onClick={() => setSelected(run.product)}
              >
                {run.product}
              </button>
            ))}
          </div>
        </div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
          <Metric label="Initial equity" value={money(metrics.initial_equity)} sub="USD" />
          <Metric label="Final equity" value={money(metrics.final_equity)} sub="USD, net" />
          <Metric label="Net return" value={percent(metrics.net_return)} sub="after costs" />
          <Metric label="Gross return" value={percent(metrics.gross_return)} sub="costs removed" />
          <Metric label="Max drawdown" value={percent(metrics.max_drawdown)} sub="on net equity" />
          <Metric
            label="Sharpe"
            value={decimals(metrics.annualized_sharpe, 2)}
            sub={`descriptive, ${metrics.periods_per_year}/yr`}
          />
        </div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
          <Metric label="Total fees" value={money(metrics.total_fees)} sub="USD" />
          <Metric label="Slippage cost" value={money(metrics.total_slippage_cost)} sub="USD" />
          <Metric label="Turnover" value={decimals(metrics.turnover_ratio, 2)} sub="Σ notional / equity" />
          <Metric label="Fills" value={String(metrics.fill_count)} sub={`${metrics.rebalance_count} rebalances`} />
          <Metric label="Expired targets" value={String(metrics.expired_target_count)} sub="gap-stale" />
          <Metric label="Avg exposure" value={percent(metrics.average_abs_exposure)} sub="absolute, of NAV" />
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Equity curve</h2>
        {equity.status === 'loading' ? <LoadingState label="Loading equity" /> : (
          <>
            <LineChart
              points={equitySeries}
              baseline={Number(metrics.initial_equity)}
              fill
              label="Equity curve"
            />
            <p className="metric-sub">
              {equity.data?.metadata.returned_count ?? 0} of{' '}
              {equity.data?.metadata.source_count ?? 0} points
              {equity.data?.metadata.aggregated
                ? ' — bucketed keeping each bucket’s extrema, so drawdowns survive'
                : ''}
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Drawdown</h2>
        <LineChart points={drawdownSeries} baseline={0} height={160} label="Drawdown" />
      </section>

      <section className="card">
        <h2 className="card-title">Exposure</h2>
        <LineChart points={exposureSeries} baseline={0} height={160} label="Exposure" />
      </section>

      <section className="card">
        <h2 className="card-title">Recent fills</h2>
        <DataTable
          rows={fills.data?.fills ?? []}
          columns={FILL_COLUMNS}
          height={360}
          empty="No fills recorded"
        />
        <p className="metric-sub">
          Showing {fills.data?.page.returned ?? 0} of {fills.data?.page.total ?? 0} fills.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">Provenance</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Experiment</dt><dd className="warning">{active.experiment_type}</dd></div>
          <div className="kv"><dt>Confirmatory</dt><dd className="warning">{String(active.confirmatory)}</dd></div>
          <div className="kv"><dt>Live execution</dt><dd className="warning">{String(active.live_execution)}</dd></div>
          <div className="kv"><dt>Source predictions</dt><dd>{active.source_benchmark_protocol}</dd></div>
          <div className="kv"><dt>Signal spec</dt><dd><Hash value={data.signal_spec_hash} /></dd></div>
          <div className="kv"><dt>Risk spec</dt><dd><Hash value={data.risk_spec_hash} /></dd></div>
          <div className="kv"><dt>Backtest spec</dt><dd><Hash value={active.economic_backtest_spec_hash} /></dd></div>
          <div className="kv"><dt>Results hash</dt><dd><Hash value={active.economic_results_hash} /></dd></div>
        </dl>
      </section>
      {contract}
    </div>
  );
}
