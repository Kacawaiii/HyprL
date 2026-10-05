import { EmptyState } from '../../components/States';
import { PriceChart } from '../../components/TraderCharts';
import type { Horizon } from '../../api/traderTypes';
import { assetChart, horizonText, ledgerCounts, paperLines, rejections, analystLabel, targetText, todayRows } from '../../lib/trader';
import { Direction, Query, pct, signed } from './shared';
import type { TraderData } from './useTraderData';
import type { RunSummary } from '../../api/traderTypes';

export function BeginnerTrader({ data, run, asset }: { data: TraderData; run: RunSummary | null; asset: string | null }) {
  const views = run?.decision?.views ?? [];
  const rows = todayRows(views);
  const rejected = rejections(views);
  const counts = ledgerCounts(data.rows);
  return (
    <>
      <section className="card" aria-label="Today's views">
        <h2 className="card-title">Today&apos;s views</h2>
        {rows.length === 0 ? (
          <EmptyState title="No views for this day" detail="See the state above: either no run happened or it produced no decision." />
        ) : (
          <div className="table-scroll">
            <table className="data" aria-label="Views per asset">
              <thead><tr><th>Asset</th><th>Horizon</th><th>View</th><th>Probability</th><th>Why</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={`${r.asset}/${r.horizon}`}>
                    <td>{r.asset}</td>
                    <td>{r.horizon} · {horizonText(r.horizon)}</td>
                    <td><Direction view={r.consensus.view} /></td>
                    <td>{r.consensus.view === 'ABSTAIN' ? '—' : <span title={targetText(r.asset)}>{pct(r.consensus.p_outperform)} <span className="metric-sub">{targetText(r.asset)}</span></span>}</td>
                    <td>{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="metric-sub">A view is shown only when both analysts, after review, agree. Probabilities are model judgments, not calibrated.</p>
      </section>

      <section className="card" aria-label="Rejected by the reviewer">
        <h2 className="card-title">What the reviewer rejected or lowered</h2>
        {rejected.length === 0 ? (
          <p className="metric-sub">{run ? 'Nothing was rejected or lowered in this run.' : 'No run to review.'}</p>
        ) : (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {rejected.map((r) => (
              <li key={`${r.analyst}/${r.asset}/${r.horizon}`}>
                <strong>{r.verdict === 'REJECT' ? 'Rejected' : 'Lowered'}:</strong> {analystLabel(r.analyst)} on {r.asset} ({r.horizon}) —{' '}
                {r.note} <span className="metric-sub">({r.code.replace(/_/g, ' ')}{r.to !== null ? `; ${pct(r.from)} → ${pct(r.to)}` : ''})</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card" aria-label="Paper positions and results">
        <h2 className="card-title">Paper positions and results</h2>
        <Query query={data.scorecard} label="Loading results">
          {(card) => {
            const lines = paperLines(card.portfolio_cohorts.filter((c) => c.analyst === 'consensus'));
            return lines.length === 0 ? (
              <p className="metric-sub">No paper cohort yet. Positions are proposed at each run and only measured when their horizon ends.</p>
            ) : (
              <table className="data" aria-label="Paper results">
                <thead><tr><th>Horizon</th><th>Finished days</th><th>Waiting days</th><th>Paper result (sum, after costs)</th></tr></thead>
                <tbody>
                  {lines.map((l) => (
                    <tr key={l.horizon}><td>{l.horizon}</td><td>{l.complete}</td><td>{l.pending}</td>
                      <td>{l.complete === 0 ? 'waiting for the first outcome' : signed(l.unhedged)}</td></tr>
                  ))}
                </tbody>
              </table>
            );
          }}
        </Query>
        {run?.portfolios && (
          <>
            <h3 className="card-title" style={{ marginTop: 12 }}>Positions proposed today (paper, not executed)</h3>
            {(['1d', '5d'] as Horizon[]).map((h) => {
              const weights = Object.entries(run.portfolios![h]?.unhedged ?? {});
              return (
                <p key={h} className="metric-sub">
                  {h}: {weights.length === 0 ? 'no position' : weights.map(([a, w]) => `${a} ${signed(w, 1)}`).join(' · ')}
                  {' '}of {Math.round((run.portfolios![h]?.capital_fraction ?? 1) * 100)} % of paper capital — {run.portfolios![h]?.execution === 'PENDING' ? 'waiting for the session open' : run.portfolios![h]?.execution}
                </p>
              );
            })}
          </>
        )}
        <p className="metric-sub">{counts.realized} of {counts.total} recorded predictions have a measured outcome; {counts.pending} are still waiting.</p>
      </section>

      <section className="card" aria-label="Price and decisions">
        <h2 className="card-title">Prices and decisions{asset ? ` — ${asset}` : ''}</h2>
        {asset && data.series.data
          ? <PriceChart chart={assetChart(asset, data.series.data.series[asset] ?? [], data.rows)} />
          : <EmptyState title="Nothing to plot yet" detail="Charts appear after the first run records a reference price." />}
      </section>
    </>
  );
}
