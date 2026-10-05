import { Link } from 'react-router-dom';
import { DataTable } from '../../components/DataTable';
import type { Column } from '../../components/DataTable';
import { ScoreCard } from '../../components/ScoreCard';
import { SourceTimeline } from '../../components/SourceTimeline';
import { Badge, EmptyState, ErrorState, Hash } from '../../components/States';
import { RunProvenanceCard } from '../../components/RunProvenance';
import type { PaperReplayFill, SignalsView } from '../../api/types';
import {
  dataQualityScore, fillsInPeriod, inPeriod, modelScore, riskScore, signalScore,
} from '../../lib/cockpit';
import { ChartSection, ProtectionCard, UncertaintyNote } from './shared';
import type { CockpitData } from './useCockpitData';

type Decision = SignalsView['decisions'][number];

const decisionColumns: Column<Decision>[] = [
  { key: 't', header: 'Decision (bar open)', render: (row) => row.timestamp },
  { key: 'p', header: 'Raw prediction', render: (row) => row.prediction },
  { key: 'd', header: 'Direction', render: (row) => row.direction },
  { key: 's', header: 'Strength', render: (row) => row.strength },
  { key: 'f', header: 'Fold', render: (row) => row.fold_index ?? '' },
  { key: 'm', header: 'Model spec', render: (row) => <Hash value={row.model_spec_hash} chars={10} /> },
  { key: 'x', header: 'Fitted artefact', render: (row) => <Hash value={row.fitted_hash} chars={10} /> },
  { key: 'h', header: 'Decision hash', render: (row) => <Hash value={row.decision_hash} chars={10} /> },
];

const fillColumns: Column<PaperReplayFill>[] = [
  { key: 't', header: 'Executed', render: (row) => row.timestamp },
  { key: 'd', header: 'Decided', render: (row) => row.decided_at },
  { key: 'a', header: 'Available', render: (row) => row.available_at },
  { key: 's', header: 'Side', render: (row) => row.side },
  { key: 'q', header: 'Quantity Δ', render: (row) => row.quantity_delta },
  { key: 'f', header: 'Fee', render: (row) => row.fee },
  { key: 'l', header: 'Slippage', render: (row) => row.slippage_cost },
  { key: 'e', header: 'Equity after', render: (row) => row.equity_after },
];

export function ExpertView({ data, carry }: { data: CockpitData; carry: string }) {
  const { summary, period, decisions, targets, replayProduct, replay, system, fomcStatus, asOf } = data;
  const inside = period ? decisions.filter((row) => inPeriod(row.timestamp, period)) : [];
  const fills = fillsInPeriod(data.fills.fills, period);
  const signal = system.data?.signal_engine;
  const risk = system.data?.risk_engine;
  const execution = data.paperStatus.data?.paper_execution;
  const entries = (data.overview.data?.benchmarks ?? []).flatMap((benchmark) =>
    benchmark.products.filter((item) => item.product === data.product).map((item) => ({ benchmark, item })));
  const scores = [
    dataQualityScore(summary), signalScore(decisions, period),
    modelScore(replayProduct?.prediction_quality, replay.data?.window),
    riskScore(targets, period, risk?.max_long_exposure ?? null),
  ];
  const counts = replayProduct?.counts;
  return (
    <div className="stack">
      <section className="card" aria-label="Prices, predictions, decisions, executions and events">
        <h2 className="card-title">Timeline · {data.product}</h2>
        <ChartSection data={data} />
        <UncertaintyNote data={data} />
      </section>

      <section aria-label="Scores">
        <h2 className="card-title">Scores</h2>
        <div className="grid grid-3">{scores.map((score) => <ScoreCard key={score.key} score={score} />)}</div>
      </section>

      <section className="card" aria-label="Snapshots and provenance">
        <h2 className="card-title">Snapshots, provenance and timeline</h2>
        {data.signals.data && <RunProvenanceCard run={data.signals.data} title="Signal run" />}
        <dl style={{ margin: '12px 0 0' }}>
          <div className="kv"><dt>Event read T (as of)</dt><dd>{asOf ?? 'no read available'}</dd></div>
          <div className="kv"><dt>Snapshot identity</dt><dd><Hash value={data.fomc.data?.snapshot.identity} chars={16} /></dd></div>
          <div className="kv"><dt>Snapshot horizon H / policy</dt><dd>{data.fomc.data ? `${data.fomc.data.snapshot.H} · ${data.fomc.data.snapshot.policy}` : '—'}</dd></div>
          <div className="kv"><dt>Read state</dt><dd>{data.fomc.data?.snapshot.read_state ?? fomcStatus.data?.status ?? '—'}</dd></div>
        </dl>
        {asOf ? <SourceTimeline source="fomc" read={{ asOf }} /> : <p className="muted">No event timeline: the store is not configured.</p>}
        <p className="metric-sub">Event marks use the declared release time. Observation, ingestion and availability times are on the timeline.</p>
      </section>

      <section className="card" aria-label="Features and outputs">
        <h2 className="card-title">Exact features and outputs</h2>
        <p className="metric-sub">
          The API serves each decision's raw prediction, direction, strength and hashes. It does not serve the feature
          vector behind a prediction, so none is shown rather than reconstructed.
        </p>
        {data.signals.status === 'error' && data.signals.error && <ErrorState error={data.signals.error} onRetry={data.signals.refetch} />}
        {data.signals.data?.available ? (
          <>
            <DataTable rows={inside} columns={decisionColumns} height={320} />
            <p className="metric-sub">
              {inside.length} decisions in the period · {decisions.length} loaded of {data.signals.data.page.total ?? '?'} in the run.
            </p>
            {data.olderDecisions.failure && <ErrorState error={data.olderDecisions.failure} onRetry={data.olderDecisions.loadOlder} />}
            {data.olderDecisions.hasMore && (
              <button className="control" onClick={data.olderDecisions.loadOlder} disabled={data.olderDecisions.busy}>
                {data.olderDecisions.busy ? 'Loading…' : 'Load older decisions'}
              </button>
            )}
          </>
        ) : <EmptyState title="No persisted signal run" detail={data.signals.data?.reason} />}
      </section>

      <section className="card" aria-label="Models and parameters">
        <h2 className="card-title">Models, versions and parameters</h2>
        {data.model && signal && risk ? (
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Model</dt><dd>{data.model.id} <Badge tone="warn">FROZEN · NOT OPTIMISED</Badge></dd></div>
            <div className="kv"><dt>Horizon</dt><dd>{data.model.horizonBars} bars</dd></div>
            <div className="kv"><dt>Declared outputs</dt><dd>{data.model.outputs.join(' · ')}</dd></div>
            <div className="kv"><dt>Not provided</dt><dd>{data.model.notProvided.join(' · ')}</dd></div>
            <div className="kv"><dt>Signal spec</dt><dd><Hash value={signal.spec_hash} chars={16} /> · long &gt; {signal.long_threshold} · short &lt; {signal.short_threshold}</dd></div>
            <div className="kv"><dt>Risk spec</dt><dd><Hash value={risk.spec_hash} chars={16} /> · cap {risk.max_long_exposure} · scale {risk.risk_scale}</dd></div>
            <div className="kv"><dt>Paper model spec</dt><dd><Hash value={data.paperStatus.data?.paper_model_spec_hash} chars={16} /></dd></div>
          </dl>
        ) : <EmptyState title="Model contract unavailable" />}
      </section>

      <section className="card" aria-label="Splits, baselines, metrics and costs">
        <h2 className="card-title">Splits, baselines, metrics and costs</h2>
        <div className="table-scroll">
          <table className="data">
            <thead><tr><th>Benchmark</th><th>Type</th><th>Rank IC</th><th>MAE</th><th>RMSE</th><th>Obs</th><th>Folds</th><th>Results</th></tr></thead>
            <tbody>
              {entries.map(({ benchmark, item }) => (
                <tr key={benchmark.version}>
                  <td>{benchmark.version.toUpperCase()}</td><td>{benchmark.experiment_type}</td>
                  <td>{item.rank_ic}</td><td>{item.mae}</td><td>{item.rmse}</td>
                  <td>{item.observations}</td><td>{item.folds}</td><td><Hash value={item.benchmark_results_hash} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {entries.length === 0 && <p className="muted">No benchmark recorded for this product.</p>}
        <p className="metric-sub">
          Splits are walk-forward and temporal; the fold count above is the split count. Baseline comparisons are not
          served by the API for this run, so none is shown.
        </p>
        {execution && (
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Fee rate / slippage</dt><dd>{execution.fee_rate} / {execution.slippage_rate}</dd></div>
            <div className="kv"><dt>Fill price</dt><dd>{execution.fill_price_policy}</dd></div>
            <div className="kv"><dt>Cost model</dt><dd>{execution.cost_model}</dd></div>
          </dl>
        )}
        {replayProduct && (
          <dl style={{ margin: 0 }}>
            <div className="kv"><dt>Replay costs (fees + slippage)</dt><dd>{replayProduct.metrics.total_execution_cost}</dd></div>
            <div className="kv"><dt>Replay result hash</dt><dd><Hash value={replayProduct.result_hash} chars={16} /></dd></div>
          </dl>
        )}
      </section>

      <section className="card" aria-label="Decisions, executions and portfolio">
        <h2 className="card-title">Executions · frozen paper replay</h2>
        {data.fills.error && <ErrorState error={data.fills.error} />}
        <DataTable rows={fills} columns={fillColumns} height={260} />
        <p className="metric-sub">
          {fills.length} executions in the period · {data.fills.fills.length} loaded of {data.fills.total ?? '?'}.
          Portfolio detail: <Link to={{ pathname: '/paper', search: carry }}>Paper</Link> ·{' '}
          <Link to={{ pathname: '/portfolio', search: carry }}>Portfolio</Link>.
        </p>
      </section>

      <ProtectionCard data={data} />

      <section className="card" aria-label="Diagnostics and exclusions">
        <h2 className="card-title">Diagnostics and exclusions</h2>
        <dl style={{ margin: 0 }}>
          <div className="kv"><dt>Declared price gaps</dt><dd>{summary?.missing_openings ?? '—'}</dd></div>
          <div className="kv"><dt>Warm-up bars without prediction</dt><dd>{counts?.warmup_without_prediction ?? '—'}</dd></div>
          <div className="kv"><dt>Unscored predictions</dt><dd>{replayProduct?.prediction_quality.unscored_predictions ?? '—'}</dd></div>
          <div className="kv"><dt>Replay gaps</dt><dd>{counts?.gaps ?? '—'}</dd></div>
          <div className="kv"><dt>Expired targets</dt><dd>{counts?.expired_targets ?? '—'}</dd></div>
          <div className="kv"><dt>Terminal targets pending</dt><dd>{counts?.pending_terminal_targets ?? '—'}</dd></div>
          <div className="kv"><dt>Protected holdout</dt><dd>{data.paperStatus.data ? `${data.paperStatus.data.protected_holdout.start.slice(0, 10)} → ${data.paperStatus.data.protected_holdout.end.slice(0, 10)} · ${data.paperStatus.data.protected_holdout.observed ? 'OBSERVED' : 'unobserved'}` : '—'}</dd></div>
          <div className="kv"><dt>Drift and market regimes</dt><dd>not provided (no versioned reference or definition is served)</dd></div>
        </dl>
        {replay.data?.limitations && (
          <ul className="metric-sub">{replay.data.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        )}
      </section>
    </div>
  );
}
