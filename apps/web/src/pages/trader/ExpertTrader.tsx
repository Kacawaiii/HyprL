import { useState } from 'react';
import { Badge, EmptyState, Hash } from '../../components/States';
import { CalibrationChart, PriceChart } from '../../components/TraderCharts';
import type { Horizon, RunSummary, ScoreEntry } from '../../api/traderTypes';
import {
  alertCounts, analystLabel, assetChart, assetsOf, baselineRows, isTaint, keptVsRejected, ledgerCounts, paperLines,
  runCounts, scoreKey, sideBySide, type ScoreKey,
} from '../../lib/trader';
import { Direction, Query, pct, signed, when } from './shared';
import type { TraderData } from './useTraderData';

const num = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? 'n/a' : v.toFixed(d));

function SideBySide({ run, asset }: { run: RunSummary; asset: string }) {
  const views = run.decision?.views ?? [];
  return (
    <div className="stack">
      {(['1d', '5d'] as Horizon[]).map((h) => {
        const pair = sideBySide(views, asset, h);
        const consensus = views.find((v) => v.analyst === 'consensus' && v.asset === asset && v.horizon === h);
        return (
          <section key={h} className="card" aria-label={`${asset} ${h} analysts`}>
            <h3 className="card-title">{asset} · {h} · consensus {consensus ? <Direction view={consensus.view} /> : 'none'}{consensus && consensus.view !== 'ABSTAIN' ? ` (${pct(consensus.p_outperform)})` : ''}</h3>
            <div className="grid grid-2">
              {pair.map((v) => (
                <article key={v.analyst} aria-label={analystLabel(v.analyst)}>
                  <h4 style={{ margin: '0 0 6px' }}>{analystLabel(v.analyst)}: <Direction view={v.raw_view?.view ?? v.view} /> {pct(v.raw_view?.p_outperform ?? v.p_outperform)}{' '}
                    <Badge tone={v.verdict === 'KEEP' ? 'ok' : 'warn'}>REVIEWER {v.verdict}</Badge></h4>
                  <dl style={{ margin: 0 }}>
                    <div className="kv"><dt>Confidence reason</dt><dd>{v.raw_view?.confidence_reason ?? 'not provided'}</dd></div>
                    <div className="kv"><dt>Priced in?</dt><dd>{v.raw_view?.priced_in_assessment ?? 'not provided'}</dd></div>
                    <div className="kv"><dt>Counter-thesis</dt><dd>{v.raw_view?.counter_thesis ?? 'not provided'}</dd></div>
                    <div className="kv"><dt>Falsifier</dt><dd>{v.raw_view?.falsifier ?? 'not provided'}</dd></div>
                    <div className="kv"><dt>Second-order</dt><dd>{v.raw_view?.second_order ?? 'not provided'}</dd></div>
                    <div className="kv"><dt>Reviewer</dt><dd>{v.review ? `${v.review.verdict} · ${v.review.reason_code}${v.review.adjusted_p !== null ? ` · p→${pct(v.review.adjusted_p)}` : ''} — ${v.review.note}` : 'not reviewed'}</dd></div>
                  </dl>
                  <ul className="metric-sub" aria-label="Sources" style={{ margin: '6px 0 0', paddingLeft: 16 }}>
                    {(v.raw_view?.catalysts ?? []).length === 0 && <li>No cited source.</li>}
                    {(v.raw_view?.catalysts ?? []).map((c) => (
                      <li key={c.url + c.published_at}>{c.fact} — <a href={c.url} rel="noreferrer noopener" target="_blank">{c.url}</a> · published {when(c.published_at)}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

function Entry({ name, e }: { name: string; e: ScoreEntry }) {
  return (
    <tr>
      <td>{analystLabel(name)}</td><td>{e.issued}</td><td>{e.realized}</td><td>{e.non_abstained}</td><td>{e.pending}</td>
      <td>{pct(e.hit_rate, 1)}</td><td>{num(e.brier)}</td><td>{num(e.climatology_brier)}</td><td>{signed(e.mean_unit_pnl_after_costs, 3)}</td><td>{e.days}</td>
    </tr>
  );
}

function Scoring({ data }: { data: TraderData }) {
  const [population, setPopulation] = useState<ScoreKey['population']>('equity_etf');
  const [horizon, setHorizon] = useState<Horizon>('1d');
  const target: ScoreKey['target'] = population === 'crypto' ? 'raw' : 'SPY_relative';
  return (
    <Query query={data.scorecard} label="Loading scorecard">
      {(card) => {
        const base = baselineRows(card, population, horizon, target);
        const consensus = card.scores[scoreKey({ analyst: 'consensus', population, horizon, target })];
        const kr = keptVsRejected(card, population, horizon, target);
        return (
          <>
            <section className="card" aria-label="Scorecard versus baselines">
              <h2 className="card-title">Scorecard vs baselines</h2>
              <div className="row" style={{ flexWrap: 'wrap' }}>
                <label>Population <select className="control" value={population} onChange={(e) => setPopulation(e.target.value as ScoreKey['population'])}>
                  <option value="equity_etf">Stocks and sector ETF (vs SPY)</option><option value="crypto">Crypto (raw direction)</option></select></label>
                <label>Horizon <select className="control" value={horizon} onChange={(e) => setHorizon(e.target.value as Horizon)}>
                  <option value="1d">1 session</option><option value="5d">5 sessions</option></select></label>
                <Badge tone="warn">{card.hypothesis_state.replace(/_/g, ' ')}</Badge>
              </div>
              {base.length === 0 ? <EmptyState title="No scored views in this population yet" /> : (
                <div className="table-scroll"><table className="data" aria-label="Baseline comparison">
                  <caption className="metric-sub" style={{ textAlign: 'left' }}>Target: {target}. Hit rate and Brier on non-abstained, realized views only; Brier lower is better; climatology = earlier observed frequency (starts at 0.50).</caption>
                  <thead><tr><th>Source</th><th>Issued</th><th>Realized</th><th>Scored</th><th>Pending</th><th>Hit rate</th><th>Brier</th><th>Climat. Brier</th><th>Mean P&amp;L / unit</th><th>Days</th></tr></thead>
                  <tbody>{base.map((b) => <Entry key={b.name} name={b.name} e={b.entry} />)}</tbody>
                </table></div>
              )}
              <ul className="metric-sub" style={{ paddingLeft: 16 }}>{card.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
            </section>

            <div className="grid grid-2">
              <section className="card" aria-label="Calibration">
                <h2 className="card-title">Calibration (consensus)</h2>
                {consensus ? <CalibrationChart bins={consensus.calibration_bins} label={`consensus ${population} ${horizon}`} /> : <EmptyState title="No consensus scores yet" />}
              </section>
              <section className="card" aria-label="Kept versus rejected">
                <h2 className="card-title">Reviewer: kept vs rejected accuracy</h2>
                <table className="data" aria-label="Kept vs rejected">
                  <thead><tr><th>Raw analyst views the reviewer…</th><th>Scored</th><th>Hit rate</th><th>False-positive rate</th><th>Pending</th></tr></thead>
                  <tbody>
                    {([['kept', kr.kept], ['downgraded', kr.downgraded], ['rejected', kr.rejected]] as const).map(([name, e]) => (
                      <tr key={name}><td>{name}</td><td>{e ? e.non_abstained : 0}</td><td>{e ? pct(e.hit_rate, 1) : 'n/a'}</td><td>{e ? pct(e.false_positive_rate, 1) : 'n/a'}</td><td>{e ? e.pending : 0}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="metric-sub">False positive = the stated direction turned out wrong. A useful reviewer keeps views with a higher hit rate than the ones it rejects; with samples this small the difference means nothing yet.</p>
              </section>
            </div>

            <section className="card" aria-label="Paper cohorts">
              <h2 className="card-title">Paper cohorts by source</h2>
              <table className="data" aria-label="Paper results by analyst">
                <thead><tr><th>Source</th><th>Horizon</th><th>Finished</th><th>Waiting</th><th>Sum, unhedged</th><th>Sum, SPY-hedged</th></tr></thead>
                <tbody>{paperLines(card.portfolio_cohorts).map((l) => (
                  <tr key={`${l.analyst}/${l.horizon}`}><td>{analystLabel(l.analyst)}</td><td>{l.horizon}</td><td>{l.complete}</td><td>{l.pending}</td><td>{l.complete ? signed(l.unhedged, 3) : 'pending'}</td><td>{l.complete ? signed(l.hedged, 3) : 'pending'}</td></tr>
                ))}</tbody>
              </table>
              <p className="metric-sub">Plain sum of finished cohorts&apos; returns on their capital fraction, after modelled costs; no compounding; 5-session cohorts overlap. Multiple testing: {card.multiple_testing.count} scored variants over {card.multiple_testing.attempted_runs} runs.</p>
            </section>
          </>
        );
      }}
    </Query>
  );
}

function Ledger({ data }: { data: TraderData }) {
  const [shown, setShown] = useState(25);
  const counts = ledgerCounts(data.rows);
  const rows = [...data.rows].reverse();
  return (
    <section className="card" aria-label="Prediction ledger">
      <h2 className="card-title">Ledger history</h2>
      <p className="metric-sub">{counts.total} predictions · {counts.realized} realized · {counts.pending} pending{data.ledger.complete ? '' : ' (first pages only)'}. A label arrives later and never rewrites its prediction.</p>
      {data.ledger.error && <p role="alert" className="negative">Ledger could not be fully loaded: {data.ledger.error.message}</p>}
      {rows.length === 0 ? <EmptyState title="The ledger is empty" detail="Predictions are appended once per run." /> : (
        <>
          <div className="table-scroll"><table className="data" aria-label="Ledger">
            <thead><tr><th>Decided</th><th>Source</th><th>Asset</th><th>Hor.</th><th>View</th><th>p</th><th>Weight</th><th>Label</th><th>Return (target)</th><th>Net / unit</th><th>Identity</th></tr></thead>
            <tbody>{rows.slice(0, shown).map((r) => {
              const p = r.prediction.payload;
              return (
                <tr key={r.prediction.identity}>
                  <td>{when(p.decision_at)}</td><td>{analystLabel(p.model_id.replace('trader:', ''))}</td><td>{p.product}</td><td>{p.signal.label_definition.horizon}</td>
                  <td><Direction view={p.outputs.class} /></td><td>{p.outputs.class === 'ABSTAIN' ? '—' : num(p.outputs.probabilities.outperform, 2)}</td><td>{signed(p.proposed_position.weight, 1)}</td>
                  <td><Badge tone={r.state === 'realized' ? 'ok' : 'warn'}>{r.state === 'realized' ? 'REALIZED' : 'PENDING'}</Badge></td>
                  <td>{r.scoredReturn === null ? 'pending' : signed(r.scoredReturn, 3)}</td><td>{r.label ? signed(r.label.value.net_unit_pnl, 3) : 'pending'}</td><td><Hash value={r.prediction.identity} /></td>
                </tr>
              );
            })}</tbody>
          </table></div>
          {shown < rows.length && <button className="control" onClick={() => setShown((n) => n + 25)}>Show 25 more ({rows.length - shown} left)</button>}
        </>
      )}
    </section>
  );
}

function RunHealth({ data }: { data: TraderData }) {
  return (
    <section className="card" aria-label="Run health">
      <h2 className="card-title">Run health</h2>
      <div className="grid grid-3">
        <Query query={data.health} label="Loading health">
          {(h) => (
            <dl style={{ margin: 0 }}>
              <div className="kv"><dt>Supervisor</dt><dd>{h.health ? <Badge tone={h.health.state === 'HEALTHY' ? 'ok' : 'warn'}>{h.health.state}</Badge> : 'never written'}</dd></div>
              <div className="kv"><dt>Last health check</dt><dd>{h.health ? when(h.health.at) : 'unknown'}</dd></div>
              <div className="kv"><dt>Last label job</dt><dd>{h.last_label ? `${h.last_label.state} · ${when(h.last_label.at)}` : 'never ran'}</dd></div>
              <div className="kv"><dt>Paused</dt><dd>{h.paused ? 'yes' : 'no'}</dd></div>
              <div className="kv"><dt>Calls used today</dt><dd>{h.health && Object.keys(h.health.budget_counts).length ? Object.entries(h.health.budget_counts).map(([k, v]) => `${k} ${v}`).join(', ') : 'none recorded'}</dd></div>
            </dl>
          )}
        </Query>
        <Query query={data.runs} label="Loading runs">
          {(r) => (
            <div>
              <strong>Run outcomes</strong>
              <p className="metric-sub">{runCounts(r.records).map(([s, n]) => `${s} ${n}`).join(' · ') || 'no run recorded'}</p>
              <ul className="metric-sub" aria-label="Recent runs" style={{ paddingLeft: 16 }}>
                {r.records.filter((x) => x.status !== 'RUNNING').slice(-5).reverse().map((x) => (
                  <li key={x.sequence}>{when(x.at)} — {x.status}{x.error ? ` (${x.error})` : ''}{x.synthetic ? ' · synthetic' : ''}</li>
                ))}
              </ul>
              <p className="metric-sub">Budget limits live in the private grant and are not served; only the counts above are.</p>
            </div>
          )}
        </Query>
        <Query query={data.alerts} label="Loading alerts">
          {(a) => (
            <div>
              <strong>Alerts</strong>
              {a.alerts.length === 0 ? <p className="metric-sub">No alert recorded.</p> : (
                <ul className="metric-sub" aria-label="Alert counts" style={{ paddingLeft: 16 }}>
                  {alertCounts(a.alerts).map(([code, n]) => <li key={code}>{code} × {n}{isTaint(code) ? ' — run rejected as tainted (a model used a tool it was not granted)' : ''}</li>)}
                </ul>
              )}
            </div>
          )}
        </Query>
      </div>
    </section>
  );
}

export function ExpertTrader({ data, run, asset }: { data: TraderData; run: RunSummary | null; asset: string | null }) {
  const assets = assetsOf(run?.decision?.views ?? []);
  const shown = asset && assets.includes(asset) ? asset : assets[0] ?? null;
  return (
    <>
      <section className="card" aria-label="Prices, decisions and outcomes">
        <h2 className="card-title">Prices, decisions and realized outcomes{shown ? ` — ${shown}` : ''}</h2>
        {shown && data.series.data
          ? <PriceChart chart={assetChart(shown, data.series.data.series[shown] ?? [], data.rows)} />
          : <EmptyState title="Nothing to plot yet" />}
      </section>
      {run?.decision && shown ? <SideBySide run={run} asset={shown} /> : <EmptyState title="No analyst output for this day" />}
      {run?.decision && (
        <section className="card" aria-label="Run provenance">
          <h2 className="card-title">Run provenance</h2>
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Decided</dt><dd>{when(run.decision.decision_at)}</dd></div>
            {Object.entries(run.decision.models).map(([role, m]) => <div key={role} className="kv"><dt>{role}</dt><dd>{m.model} · {m.reported_version}</dd></div>)}
            <div className="kv"><dt>Context</dt><dd><Hash value={run.decision.context_hash} /></dd></div>
            <div className="kv"><dt>Preregistration</dt><dd><Hash value={run.decision.preregistration_hash} /></dd></div>
            <div className="kv"><dt>Skills</dt><dd>{Object.entries(run.decision.skill_hashes).map(([k, v]) => <span key={k}>{k} <Hash value={v} /> </span>)}</dd></div>
          </dl>
          <Query query={data.context} label="Loading sources">
            {(c) => c.context && (
              <details><summary>{c.context.sources.length} data sources and {c.context.limitations.length} stated limitations</summary>
                <ul className="metric-sub">{c.context.sources.map((s) => <li key={s.digest}>{s.url} · received {when(s.received_at)} · <Hash value={s.digest} /></li>)}</ul>
                <ul className="metric-sub">{c.context.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
              </details>
            )}
          </Query>
        </section>
      )}
      <Scoring data={data} />
      <Ledger data={data} />
      <RunHealth data={data} />
    </>
  );
}
