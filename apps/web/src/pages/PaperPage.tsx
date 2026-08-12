/** The live shadow portfolio: BTC and ETH on one pot of capital.
 *
 *  Two things this page must never do.
 *
 *  It must never imply a trade has happened when a batch is still waiting.
 *  Live candles arrive in no defined order, so the runtime routinely holds one
 *  instrument's target while the other's is outstanding — and a reader looking
 *  at a target exposure could easily read it as a position. The pending panel
 *  says WAITING FOR PORTFOLIO BATCH in as many words.
 *
 *  It must never blend the legacy per-product sessions into this portfolio.
 *  Those ran on 100 000 each and never shared a dollar; their equity is not
 *  this portfolio's history. They live in a collapsed section, never summed,
 *  never charted together. */
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { LineChart } from '../components/LineChart';
import type { PaperPortfolioPosition, PendingBatch } from '../api/types';

function money(value: string | null | undefined): string {
  if (!value) return '—';
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : value;
}

function percent(value: string | null | undefined): string {
  if (!value) return '—';
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(4)} %` : value;
}

function ratio(value: string | null | undefined, digits = 6): string {
  if (!value) return '—';
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : value;
}

export function PaperPage() {
  const status = useQuery('paper-portfolio', (signal) =>
    apiClient.getPaperPortfolio(signal), { staleMs: 5_000 });
  const pending = useQuery('paper-portfolio-pending', (signal) =>
    apiClient.getPaperPortfolioPending(signal), { staleMs: 5_000 });
  const equity = useQuery('paper-portfolio-equity', (signal) =>
    apiClient.getPaperPortfolioEquity(500, signal));
  const legacy = useQuery('paper-legacy', (signal) => apiClient.getPaperLegacy(signal));

  if (status.status === 'loading') return <LoadingState label="Loading paper portfolio" />;
  if (status.status === 'error' && status.error) {
    return <ErrorState error={status.error} onRetry={status.refetch} />;
  }
  if (!status.data) return null;

  const data = status.data;
  const state = data.state ?? null;
  const embargoed = Object.values(data.embargo ?? {}).some((item) => item.embargoed);
  const batches = Object.values(pending.data?.pending_batches ?? {});

  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">Shared paper portfolio</h2>
        <p>
          <Badge tone="warn">SHADOW MODE</Badge>{' '}
          <Badge tone="warn">SHARED CAPITAL</Badge>{' '}
          <Badge tone="off">NO REAL MONEY</Badge>{' '}
          <Badge tone="off">NO CONFIRMED EDGE</Badge>
          {embargoed && <> <Badge tone="warn">EMBARGOED</Badge></>}
        </p>
        <div className="grid grid-2">
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Mode</dt><dd>{data.mode}</dd></div>
            <div className="kv"><dt>Equity</dt><dd>{money(state?.equity ?? (data.available ? null : data.initial_equity))}</dd></div>
            <div className="kv"><dt>Cash</dt><dd>{money(state?.cash ?? (data.available ? null : data.initial_equity))}</dd></div>
            <div className="kv"><dt>Gross exposure</dt><dd>{percent(state?.gross_exposure)}</dd></div>
            <div className="kv"><dt>Net exposure</dt><dd>{percent(state?.net_exposure)}</dd></div>
          </dl>
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Fees</dt><dd>{money(state?.cumulative_fees)}</dd></div>
            <div className="kv"><dt>Slippage</dt><dd>{money(state?.cumulative_slippage_cost)}</dd></div>
            <div className="kv"><dt>Fills</dt><dd>{data.fill_count ?? 0}</dd></div>
            <div className="kv"><dt>Rebalances</dt><dd>{data.rebalance_count ?? 0}</dd></div>
            <div className="kv">
              <dt>Event chain</dt>
              <dd>
                {data.chain
                  ? <Badge tone={data.chain.verified ? 'ok' : 'off'}>
                      {data.chain.verified ? 'VERIFIED' : 'INVALID'}
                    </Badge>
                  : <span className="muted">no session</span>}
              </dd>
            </div>
          </dl>
        </div>
        <p className="muted">
          Gross cap {percent(data.max_gross_exposure)} ·{' '}
          {data.allocation_rule} · {data.simultaneous_rebalance_rule} ·{' '}
          spec <Hash value={data.portfolio_spec_hash} chars={16} />
        </p>
      </section>

      {!data.available && (
        <section className="card">
          <EmptyState
            title="No shared portfolio session recorded"
            detail={
              <>
                {data.reason}. Start one with{' '}
                <code>./scripts/hyprl.sh paper start</code>. The engine and its
                limits are frozen above; nothing has been simulated live yet.
              </>
            }
          />
        </section>
      )}

      <section className="card">
        <h2 className="card-title">Portfolio batch</h2>
        {batches.length === 0 ? (
          <p className="muted">
            No batch is open. A rebalance happens only when every instrument has
            supplied a target for the same timestamp.
          </p>
        ) : (
          <>
            <p><Badge tone="warn">WAITING FOR PORTFOLIO BATCH</Badge></p>
            <dl style={{ margin: 0 }}>
              {batches.map((batch: PendingBatch) => (
                <div className="kv" key={batch.decision_at}>
                  <dt>{batch.decision_at.replace('T', ' ').slice(0, 16)}</dt>
                  <dd>
                    {batch.ready.map((name) => (
                      <span key={name}>
                        <Badge tone="ok">{name.split(':').pop()} READY</Badge>{' '}
                      </span>
                    ))}
                    {batch.missing.map((name) => (
                      <span key={name}>
                        <Badge tone="warn">{name.split(':').pop()} WAITING</Badge>{' '}
                      </span>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
            <p className="muted">
              Nothing has been traded for these timestamps. Arrival order does
              not decide the portfolio: the batch executes atomically once it is
              complete.
            </p>
          </>
        )}
      </section>

      <section className="grid grid-2">
        {(state?.positions ?? []).map((position: PaperPortfolioPosition) => (
          <article className="card" key={position.instrument_id}>
            <h3 className="card-title">{position.instrument_id}</h3>
            <dl style={{ margin: 0 }}>
              <div className="kv"><dt>Quantity</dt><dd>{ratio(position.quantity)}</dd></div>
              <div className="kv"><dt>Mark</dt><dd>{money(position.mark_price)}</dd></div>
              <div className="kv"><dt>Market value</dt><dd>{money(position.market_value)}</dd></div>
              <div className="kv"><dt>Target exposure</dt><dd>{percent(position.target_exposure)}</dd></div>
              <div className="kv"><dt>Fees</dt><dd>{money(position.cumulative_fees)}</dd></div>
              <div className="kv"><dt>Slippage</dt><dd>{money(position.cumulative_slippage_cost)}</dd></div>
              <div className="kv"><dt>Gross P&amp;L</dt><dd>{money(position.cumulative_gross_pnl)}</dd></div>
            </dl>
          </article>
        ))}
      </section>

      {equity.data?.available && equity.data.series.length > 0 && (
        <section className="card">
          <h2 className="card-title">Portfolio equity</h2>
          <LineChart
            points={equity.data.series.map((row) => ({
              timestamp: row.timestamp, value: Number(row.equity),
            }))}
            baseline={Number(data.initial_equity)}
          />
          <p className="metric-sub" style={{ marginTop: 8 }}>
            {equity.data.metadata.returned_count ?? equity.data.series.length} of{' '}
            {equity.data.metadata.source_count} recorded marks
          </p>
        </section>
      )}

      <section className="card">
        <h2 className="card-title">Execution and protection</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Fill price</dt><dd>{data.paper_execution.fill_price_policy}</dd></div>
          <div className="kv"><dt>Observation</dt><dd>{data.paper_execution.fill_observation_policy}</dd></div>
          <div className="kv">
            <dt>Reserved holdout</dt>
            <dd>
              {data.protected_holdout.start.slice(0, 10)} →{' '}
              {data.protected_holdout.end.slice(0, 10)}{' '}
              <Badge tone="warn">UNOBSERVED</Badge>
            </dd>
          </div>
          <div className="kv"><dt>Broker</dt><dd><Badge tone="off">NOT CONNECTED</Badge></dd></div>
        </dl>
      </section>

      {legacy.data?.available && (
        <section className="card">
          <details>
            <summary>
              Legacy individual paper sessions{' '}
              <Badge tone="off">PRE-SHARED-PORTFOLIO</Badge>
            </summary>
            <dl style={{ margin: 0, marginTop: 12 }}>
              <div className="kv"><dt>Sessions</dt><dd>{legacy.data.sessions}</dd></div>
              <div className="kv"><dt>Events</dt><dd>{legacy.data.events.toLocaleString()}</dd></div>
              <div className="kv"><dt>Shared capital</dt><dd><Badge tone="off">NO</Badge></dd></div>
            </dl>
            <p className="muted">{legacy.data.note}</p>
          </details>
        </section>
      )}
    </div>
  );
}
