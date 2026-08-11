/** BTC and ETH as one portfolio on one pot of cash.
 *
 *  Every number here is computed in Python and arrives as a Decimal string.
 *  This page formats and lays out; it never adds two returns together. That
 *  matters more than usual on this page, because the obvious thing a reader
 *  wants to do -- compare this against the two separate 5C accounts -- is the
 *  one comparison that needs care: those ran on 100 000 each, this runs on
 *  100 000 total. The page says so rather than leaving it to be assumed. */
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { DataTable, type Column } from '../components/DataTable';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { LineChart } from '../components/LineChart';
import type { InstrumentAttribution, PortfolioFill } from '../api/types';

function percent(value: string | undefined): string {
  if (value === undefined) return '—';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return value;
  return `${(parsed * 100).toFixed(4)} %`;
}

function money(value: string | undefined): string {
  if (value === undefined) return '—';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return value;
  return parsed.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function ratio(value: string | undefined, digits = 4): string {
  if (value === undefined) return '—';
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : value;
}

export function PortfolioPage() {
  const status = useQuery('portfolio', (signal) => apiClient.getPortfolio(signal));
  const runs = useQuery('portfolio-backtests', (signal) =>
    apiClient.getPortfolioBacktests(signal),
  );

  if (status.status === 'loading') return <LoadingState label="Loading portfolio" />;
  if (status.status === 'error' && status.error) {
    return <ErrorState error={status.error} onRetry={status.refetch} />;
  }
  if (!status.data) return null;

  const contract = status.data.portfolio;
  const available = runs.data?.available ?? false;
  const version = runs.data?.runs[0]?.version ?? 'v1';

  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">Portfolio engine</h2>
        <p>
          <Badge tone={available ? 'ok' : 'warn'}>{available ? 'RESULTS AVAILABLE' : 'READY'}</Badge>{' '}
          <Badge tone="warn">SHARED CAPITAL</Badge>{' '}
          <Badge tone="warn">SYNTHETIC EXECUTION COSTS</Badge>{' '}
          <Badge tone="off">NO CONFIRMED EDGE</Badge>
        </p>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Protocol</dt><dd>{contract.protocol}</dd></div>
          <div className="kv"><dt>Initial equity</dt><dd>{money(contract.initial_equity)} {contract.base_currency}</dd></div>
          <div className="kv"><dt>Per-instrument cap</dt><dd>{percent(contract.max_instrument_abs_exposure)}</dd></div>
          <div className="kv"><dt>Gross cap</dt><dd>{percent(contract.max_gross_exposure)}</dd></div>
          <div className="kv"><dt>Net cap</dt><dd>{percent(contract.max_net_abs_exposure)}</dd></div>
          <div className="kv"><dt>Allocation rule</dt><dd>{contract.allocation_rule}</dd></div>
          <div className="kv"><dt>Rebalance rule</dt><dd>{contract.simultaneous_rebalance_rule}</dd></div>
          <div className="kv"><dt>Cash model</dt><dd>{contract.cash_model}</dd></div>
          <div className="kv"><dt>Optimized</dt><dd><Badge tone="off">NO</Badge></dd></div>
          <div className="kv"><dt>Spec hash</dt><dd><Hash value={contract.portfolio_spec_hash} chars={20} /></dd></div>
        </dl>
        <p className="muted">{contract.gross_cap_rationale}</p>
      </section>

      {!available && (
        <section className="card">
          <EmptyState
            title="No persisted portfolio run available"
            detail={
              <>
                {runs.data?.reason ?? status.data.reason}. The engine and its
                limits are frozen above; nothing has been simulated against them
                yet.
              </>
            }
          />
        </section>
      )}

      {available && <PortfolioResult version={version} />}
    </div>
  );
}

function PortfolioResult({ version }: { version: string }) {
  const detail = useQuery(`portfolio-detail:${version}`, (signal) =>
    apiClient.getPortfolioDetail(version, signal),
  );
  const equity = useQuery(`portfolio-equity:${version}`, (signal) =>
    apiClient.getPortfolioEquity(version, 500, signal),
  );
  const attribution = useQuery(`portfolio-attribution:${version}`, (signal) =>
    apiClient.getPortfolioAttribution(version, signal),
  );
  const fills = useQuery(`portfolio-fills:${version}`, (signal) =>
    apiClient.getPortfolioFills(version, 100, signal),
  );

  const fillColumns: Column<PortfolioFill>[] = [
    { key: 'time', header: 'Timestamp (UTC)', render: (row) => row.timestamp.replace('T', ' ').slice(0, 16) },
    { key: 'instrument', header: 'Instrument', render: (row) => row.instrument_id },
    { key: 'side', header: 'Side', render: (row) => row.side },
    { key: 'reference', header: 'Reference', render: (row) => ratio(row.reference_price, 2) },
    { key: 'fill', header: 'Fill', render: (row) => ratio(row.fill_price, 2) },
    { key: 'delta', header: 'Δ quantity', render: (row) => ratio(row.quantity_delta, 6) },
    { key: 'fee', header: 'Fee', render: (row) => ratio(row.fee, 2) },
    { key: 'slip', header: 'Slippage', render: (row) => ratio(row.slippage_cost, 2) },
  ];

  return (
    <>
      {detail.status === 'loading' && <LoadingState label="Loading result" />}
      {detail.status === 'error' && detail.error && (
        <ErrorState error={detail.error} onRetry={detail.refetch} />
      )}
      {detail.data && (
        <>
          <section className="card">
            <h2 className="card-title">Portfolio result</h2>
            <p>
              <Badge tone="warn">{detail.data.experiment_type.toUpperCase()}</Badge>{' '}
              <Badge tone="off">CONFIRMATORY: NO</Badge>{' '}
              <Badge tone="off">COMMERCIAL EDGE: NOT ESTABLISHED</Badge>
            </p>
            <div className="grid grid-2">
              <dl style={{ margin: 0 }}>
                <div className="kv"><dt>Initial equity</dt><dd>{money(detail.data.metrics.initial_equity)}</dd></div>
                <div className="kv"><dt>Final equity</dt><dd>{money(detail.data.metrics.final_equity)}</dd></div>
                <div className="kv"><dt>Net return</dt><dd>{percent(detail.data.metrics.net_return)}</dd></div>
                <div className="kv"><dt>Gross return</dt><dd>{percent(detail.data.metrics.gross_return)}</dd></div>
                <div className="kv"><dt>Max drawdown</dt><dd>{percent(detail.data.metrics.max_drawdown)}</dd></div>
                <div className="kv"><dt>Sharpe (annualised)</dt><dd>{ratio(detail.data.metrics.annualized_sharpe)}</dd></div>
              </dl>
              <dl style={{ margin: 0 }}>
                <div className="kv"><dt>Fees</dt><dd>{money(detail.data.metrics.total_fees)}</dd></div>
                <div className="kv"><dt>Slippage</dt><dd>{money(detail.data.metrics.total_slippage_cost)}</dd></div>
                <div className="kv"><dt>Execution cost</dt><dd>{money(detail.data.metrics.total_execution_cost)}</dd></div>
                <div className="kv"><dt>Turnover</dt><dd>{ratio(detail.data.metrics.portfolio_turnover)}</dd></div>
                <div className="kv"><dt>Avg gross exposure</dt><dd>{percent(detail.data.metrics.average_gross_exposure)}</dd></div>
                <div className="kv"><dt>Avg |net| exposure</dt><dd>{percent(detail.data.metrics.average_abs_net_exposure)}</dd></div>
              </dl>
            </div>
            <p className="muted">
              {detail.data.metrics.fill_count} fills over{' '}
              {detail.data.metrics.rebalance_count} rebalances ·{' '}
              result <Hash value={detail.data.result_hash} chars={16} />
            </p>
            <p className="muted">
              Shared capital: one cash ledger of{' '}
              {money(detail.data.metrics.initial_equity)} across{' '}
              {detail.data.instruments.join(' and ')}. The separate
              single-product runs each used their own {money(detail.data.metrics.initial_equity)},
              so their percentages are not additive with this one.
            </p>
          </section>

          <section className="card">
            <h2 className="card-title">Portfolio equity</h2>
            {equity.status === 'loading' && <LoadingState label="Loading equity" />}
            {equity.data && (
              <>
                <LineChart
                  points={equity.data.series.map((row) => ({
                    timestamp: row.timestamp, value: Number(row.equity),
                  }))}
                  baseline={Number(equity.data.metadata.initial_equity)}
                />
                <p className="metric-sub" style={{ marginTop: 8 }}>
                  {equity.data.metadata.returned_count} of{' '}
                  {equity.data.metadata.source_count.toLocaleString()} marks
                  {equity.data.metadata.aggregated
                    ? ' · bucket extrema preserved (drawdowns survive downsampling)'
                    : ' · every mark'}
                </p>
              </>
            )}
          </section>

          <section className="card">
            <h2 className="card-title">Gross exposure</h2>
            {equity.data && (
              <LineChart
                points={equity.data.series.map((row) => ({
                  timestamp: row.timestamp, value: Number(row.gross_exposure),
                }))}
                baseline={0}
              />
            )}
          </section>
        </>
      )}

      <section className="card">
        <h2 className="card-title">Instrument allocation</h2>
        {attribution.status === 'loading' && <LoadingState label="Loading attribution" />}
        {attribution.data && (
          <>
            <div className="grid grid-2">
              {attribution.data.attribution.map((record: InstrumentAttribution) => (
                <article className="card" key={record.instrument_id}>
                  <h3 className="card-title">{record.instrument_id}</h3>
                  <dl style={{ margin: 0 }}>
                    <div className="kv"><dt>Gross contribution</dt><dd>{money(record.gross_pnl)}</dd></div>
                    <div className="kv"><dt>Net contribution</dt><dd>{money(record.net_pnl)}</dd></div>
                    <div className="kv"><dt>Fees</dt><dd>{money(record.fees)}</dd></div>
                    <div className="kv"><dt>Slippage</dt><dd>{money(record.slippage_cost)}</dd></div>
                    <div className="kv"><dt>Turnover</dt><dd>{ratio(record.turnover)}</dd></div>
                    <div className="kv"><dt>Avg |exposure|</dt><dd>{percent(record.average_abs_exposure)}</dd></div>
                    <div className="kv"><dt>Fills</dt><dd>{record.fill_count}</dd></div>
                  </dl>
                </article>
              ))}
            </div>
            <p className="muted">
              Gross contribution is quantity held times the price move between
              marks; execution costs attach to that instrument's own fills. The
              net contributions add up to the change in portfolio equity.
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Fills</h2>
        {fills.status === 'loading' && <LoadingState label="Loading fills" />}
        {fills.data && (
          <>
            <DataTable rows={fills.data.fills} columns={fillColumns} height={380} />
            <p className="metric-sub" style={{ marginTop: 8 }}>
              First {fills.data.page.returned} of {fills.data.page.total}
              {fills.data.page.has_more ? ' · more available via cursor' : ''}
            </p>
          </>
        )}
      </section>
    </>
  );
}
