/**
 * The three Alpaca paper accounts side by side: AI stocks, AI crypto and the Claude book (with its carre sleeve and
 * momo_v0 positions tagged). Positions show their stop and target only where a policy defined them; a position
 * without one says so instead of showing a made-up level.
 */
import { apiClient, ApiError } from '../../api/client';
import type { PaperAccount, PaperPosition } from '../../api/radarTypes';
import { Badge, EmptyState, ErrorState, LoadingState } from '../../components/States';
import { RebasedChart } from '../../components/RebasedChart';
import { instant, protectionDistances, rebasedCurves, signedNumber, signedPct } from '../../lib/radar';
import { useQuery } from '../../state/useQuery';
import { useCockpit } from '../../state/useCockpit';
import { DecisionCard } from '../../components/DecisionCard';

function money(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 2 });
}
function level(value: number | null) {
  return value === null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 6 });
}

function PositionRow({ position, expert }: { position: PaperPosition; expert: boolean }) {
  const distance = protectionDistances(position.last, position.stop, position.target);
  return (
    <tr data-tag={position.tag}>
      <td>{position.symbol}</td>
      <td><span className="chip" data-tag={position.tag}>{position.tag}</span></td>
      <td>{level(position.qty)}</td>
      <td>{level(position.entry)}</td>
      <td>{level(position.last)}</td>
      <td className={position.unrealized_pl !== null && position.unrealized_pl < 0 ? 'negative' : ''}>
        {signedNumber(position.unrealized_pl)} ({signedPct(position.unrealized_pct)})
      </td>
      <td>
        {position.protection === 'policy'
          ? <>{level(position.stop)} <span className="muted">({signedPct(distance.stop, 1)})</span></>
          : <span className="muted">no policy</span>}
      </td>
      <td>
        {position.protection === 'policy' && position.target !== null
          ? <>{level(position.target)} <span className="muted">({signedPct(distance.target, 1)})</span></>
          : <span className="muted">—</span>}
      </td>
      {expert && <td>{instant(position.opened_at)}</td>}
    </tr>
  );
}

function Account({ account, expert }: { account: PaperAccount; expert: boolean }) {
  const isBook = account.account === 'claude_book';
  return (
    <article className="card paper-account" aria-label={account.label} data-wide={account.positions.length > 0}>
      <header className="row paper-account-head">
        <h2 className="paper-account-title">{account.label}</h2>
        <span className="chip">account …{account.suffix}</span>
        <Badge tone="off">PAPER</Badge>
        {account.halted && <Badge tone="warn">HALTED</Badge>}
      </header>
      <dl className="kv-list">
        <div><dt>Equity</dt><dd>{money(account.equity)}</dd></div>
        <div><dt>Return since start</dt><dd>{signedPct(account.return_since_start)}</dd></div>
        {account.day_pnl !== undefined && <div><dt>Day P&amp;L</dt><dd>{money(account.day_pnl)}</dd></div>}
        {account.cash !== undefined && <div><dt>Cash</dt><dd>{money(account.cash)}</dd></div>}
        {expert && <div><dt>Peak</dt><dd>{money(account.peak)}</dd></div>}
      </dl>
      {account.observed_at && <p className="metric-sub">Dernière observation du compte : {instant(account.observed_at)}.</p>}
      {account.positions.length === 0 ? (
        <p className="muted">
          {isBook ? 'The book is flat.' : `No open position${account.open_lots === 0 ? '' : ` (${account.open_lots} lots reported)`}.`}
        </p>
      ) : (
        <div className="table-scroll">
          <table className="data" aria-label={`${account.label} positions`}>
            <thead>
              <tr><th>Symbol</th><th>Tag</th><th>Qty</th><th>Entry</th><th>Last</th><th>Unrealised</th><th>Stop</th><th>Target</th>
                {expert && <th>Opened</th>}</tr>
            </thead>
            <tbody>
              {account.positions.map((position) => <PositionRow key={position.symbol} position={position} expert={expert} />)}
            </tbody>
          </table>
        </div>
      )}
      {(account.open_orders?.length ?? 0) > 0 && (
        <details>
          <summary>{account.open_orders?.length} open order(s)</summary>
          <ul className="orders">
            {account.open_orders?.map((order, i) => (
              <li key={`${order.symbol}-${i}`}>
                {order.side} {order.qty} {order.symbol} {order.type}{order.limit !== null && ` @ ${order.limit}`} · {order.status}
                {order.legs.map((leg, j) => <span className="muted" key={j}> · {leg.type} {leg.limit ?? leg.stop}</span>)}
              </li>
            ))}
          </ul>
        </details>
      )}
      {account.positions.map((position) => <details className="decision-detail" key={`decision-${position.symbol}`}>
        <summary>{position.symbol} · {position.tag} : chaîne de décision</summary>
        <p className="metric-sub">Plan consigné pour cet actif; l’export n’attribue pas un lot exécuté à cette intention.</p>
        <DecisionCard decision={position.decision} />
      </details>)}
    </article>
  );
}

function Journal({ accounts }: { accounts: PaperAccount[] }) {
  const rows = accounts.flatMap((a) => a.journal.map((row) => ({ ...row, account: a.label })))
    .filter((row) => row.action !== 'submitted')
    .sort((a, b) => b.at.localeCompare(a.at)).slice(0, 15);
  if (rows.length === 0) return <p className="muted">No journal entry recorded.</p>;
  return (
    <ol className="journal" aria-label="Trade journal">
      {rows.map((row, i) => (
        <li key={`${row.at}-${i}`}>
          <div>
            <strong>{row.action.replace(/_/g, ' ')}</strong> {row.symbol ?? ''} <span className="muted">· {row.account}
              {row.engine && ` · ${row.engine}`} · {instant(row.at)}</span>
          </div>
          {row.reason && <p>{row.reason}</p>}
          {row.mechanism && <p className="muted">{row.mechanism}</p>}
          {row.invalidation && <p className="muted">Invalidation: {row.invalidation}</p>}
          <DecisionCard decision={row.decision} />
        </li>
      ))}
    </ol>
  );
}

export function PaperAccounts() {
  const { selection } = useCockpit();
  const expert = selection.mode === 'expert';
  const paper = useQuery('radar-paper', (signal) => apiClient.getRadarPaper(signal), { staleMs: 60_000 });
  if (paper.status === 'loading') return <LoadingState label="Loading paper accounts" />;
  if (paper.status === 'error' && paper.error) {
    if (paper.error instanceof ApiError && [404, 503].includes(paper.error.status)) {
      return <EmptyState title="No paper account snapshot is published yet"
        detail="The export job has not written paper.json for this runtime." />;
    }
    return <ErrorState error={paper.error} onRetry={paper.refetch} />;
  }
  const data = paper.data;
  if (!data) return null;
  const curves = rebasedCurves(data);
  return (
    <section className="stack paper-accounts" aria-label="Alpaca paper accounts">
      <div className="card">
        <h2 className="card-title">Alpaca paper accounts</h2>
        <p className="muted">
          Three separate paper accounts, as of {instant(data.generated_at)}. The Claude book is discretionary and is
          not part of any AI trader variant or baseline.
          {data.benchmarks.SPY.return_since_paper_start !== null && <> SPY since the paper start: {signedPct(data.benchmarks.SPY.return_since_paper_start)}.</>}
        </p>
      </div>
      <div className="grid grid-3 paper-grid">
        {data.accounts.map((account) => <Account key={account.account} account={account} expert={expert} />)}
      </div>
      <div className="card">
        <h2 className="card-title">Equity against SPY and BTC (re-based to 100)</h2>
        <RebasedChart series={curves} />
      </div>
      <div className="card">
        <h2 className="card-title">Trade journal</h2>
        <Journal accounts={data.accounts} />
      </div>
      {expert && data.limitations.length > 0 && (
        <ul className="muted limitations">{data.limitations.map((x) => <li key={x}>{x}</li>)}</ul>
      )}
    </section>
  );
}
