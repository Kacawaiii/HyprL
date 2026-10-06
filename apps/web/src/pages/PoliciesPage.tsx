import { apiClient } from '../api/client';
import type { PolicyCalibration, ProtectionSimulation, Reliability } from '../api/policyTypes';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { useCockpit } from '../state/useCockpit';
import { useQuery } from '../state/useQuery';

function metric(value: number | null) { return value === null ? 'unavailable' : value.toFixed(6); }

function ReliabilityDiagram({ raw, calibrated }: { raw: Reliability; calibrated: Reliability }) {
  if (calibrated.state !== 'AVAILABLE') return <EmptyState title="Calibration diagnostics refused: sample too small" />;
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox="0 0 340 300" role="img" aria-label="Reliability diagram: predicted probability versus observed frequency"
        style={{ width: '100%', maxWidth: 420 }}>
        <line x1="50" y1="250" x2="290" y2="10" stroke="currentColor" strokeDasharray="4 4" />
        <path d="M50 10 V250 H290" fill="none" stroke="currentColor" />
        {[0, .5, 1].map((v) => <g key={v}><text x={50 + v * 240} y="270" textAnchor="middle" fill="currentColor">{v}</text>
          <text x="35" y={254 - v * 240} textAnchor="end" fill="currentColor">{v}</text></g>)}
        <text x="170" y="295" textAnchor="middle" fill="currentColor">Mean predicted probability</text>
        <text x="12" y="140" transform="rotate(-90 12 140)" textAnchor="middle" fill="currentColor">Observed frequency</text>
        {([['Raw', raw, '#bb6600'], ['Calibrated', calibrated, '#008878']] as const).map(([name, stats, color]) => (
          <g key={name}>{stats.bins.filter((b) => b.count > 0 && b.mean_probability !== null && b.observed_frequency !== null).map((b) => (
            <circle key={b.lower} cx={50 + b.mean_probability! * 240} cy={250 - b.observed_frequency! * 240}
              r={name === 'Raw' ? 5 : 4} fill={name === 'Raw' ? 'none' : color} stroke={color}>
              <title>{name}: n={b.count}, mean={b.mean_probability}, observed={b.observed_frequency}</title>
            </circle>
          ))}</g>
        ))}
      </svg>
      <figcaption>Hollow orange: raw · filled green: calibrated · dashed line: perfect reliability. Ten equal-width bins; empty bins omitted. n={calibrated.count} test decisions.</figcaption>
    </figure>
  );
}

function CalibrationCard({ item, expert }: { item: PolicyCalibration; expert: boolean }) {
  const sample = item.sample;
  return <section className="card" aria-label="Synthetic probability calibration">
    <h2 className="card-title">Probability calibration · {item.product}</h2>
    <p>Evidence model: <strong>{sample.origin}</strong>. Event: {sample.event}; horizon {sample.horizon_seconds / 3600} hours.</p>
    <p>At {sample.decision_at}: raw score {sample.raw_probability.toFixed(4)} → calibrated probability {sample.calibrated_probability.toFixed(4)}.</p>
    <p>Method: <strong>{sample.method}</strong>. Fitted on walk-forward training folds only. Test labels are used for diagnostics.</p>
    <p>Evaluation: {item.folds[0]?.test.first} → {item.folds.at(-1)?.test.last}, n={item.calibrated.count}, {item.folds.length} folds.</p>
    <ReliabilityDiagram raw={item.raw} calibrated={item.calibrated} />
    <p>Raw Brier: {metric(item.raw.brier)} · calibrated Brier: {metric(item.calibrated.brier)}. Lower is better; these synthetic results do not qualify a real model.</p>
    {expert && <>
      <dl>
        <div className="kv"><dt>Calibration identity / fold</dt><dd><Hash value={sample.calibration_hash} /> / {sample.fold_index}</dd></div>
        <div className="kv"><dt>Model artifact</dt><dd><Hash value={sample.artifact_hash} /></dd></div>
        <div className="kv"><dt>Prediction identity</dt><dd><Hash value={sample.prediction_hash} /></dd></div>
        <div className="kv"><dt>Diagnostic method</dt><dd>{item.calibrated.method}</dd></div>
        {(['reliability', 'resolution', 'uncertainty', 'binned_brier', 'binning_residual'] as const).map((key) =>
          <div className="kv" key={key}><dt>{key}</dt><dd>{metric(item.calibrated[key])}</dd></div>)}
      </dl>
      <p>Binned Brier = reliability − resolution + uncertainty. Unbinned calibrated Brier = binned Brier + binning residual.</p>
      <div className="table-wrap"><table><caption>Calibrated reliability bins</caption>
        <thead><tr><th>Bin</th><th>Count</th><th>Mean probability</th><th>Observed frequency</th></tr></thead>
        <tbody>{item.calibrated.bins.map((b) => <tr key={b.lower}><td>{b.lower}–{b.upper}</td><td>{b.count}</td>
          <td>{metric(b.mean_probability)}</td><td>{metric(b.observed_frequency)}</td></tr>)}</tbody>
      </table></div>
      <div className="table-wrap"><table><caption>Training and held-out test provenance</caption>
        <thead><tr><th>Fold</th><th>Train n (negative / positive)</th><th>Last label available</th><th>Fitted / validation start</th><th>Test n</th><th>Identity</th></tr></thead>
        <tbody>{item.folds.map((f, i) => <tr key={f.calibration_hash}><td>{i}</td><td>{f.artifact.train.count} ({f.artifact.train.class_counts.join(' / ')})</td>
          <td>{f.artifact.train.last_label_available_at}</td><td>{f.artifact.fitted_at} / {f.artifact.validation_start}</td><td>{f.test.count}</td><td><Hash value={f.calibration_hash} /></td></tr>)}</tbody>
      </table></div>
    </>}
  </section>;
}

function ProtectionCard({ simulation, expert }: { simulation: ProtectionSimulation; expert: boolean }) {
  const { plan, exit } = simulation;
  return <section className="card" aria-label={`Paper protection ${simulation.scenario}`}>
    <h2 className="card-title">{simulation.scenario} · {plan.side} · {plan.mode}</h2>
    <p>Entry fill {plan.entry_fill} at {plan.entry_at}; horizon {plan.horizon_seconds / 3600} hours. State: <strong>{simulation.state}</strong>.</p>
    <dl>{(['take_profit', 'stop_loss'] as const).map((key) => {
      const level = plan.levels[key];
      return <div key={key} className="kv"><dt>{key === 'take_profit' ? 'Take profit' : 'Stop loss'}</dt><dd>
        <strong>{level.value}</strong> · origin {level.origin} · method {level.method}
        <div>Version identity: <Hash value={level.source_hash} /> · provided {level.provided_at}</div>
      </dd></div>;
    })}</dl>
    <p>Gaps: {simulation.gap_method}.</p>
    <p>Intrabar ambiguity: {simulation.ambiguity_method}.</p>
    <p>Missing bars: {simulation.missing_bar_method}.</p>
    {exit ? <p>Exit {exit.reason}: reference {exit.reference_price}, fill {exit.fill_price}, net unit P&amp;L {exit.net_pnl}.
      {' '}Observed by {exit.available_at}. {exit.clock_method}.</p> : <p>No realized fill or P&amp;L.</p>}
    {simulation.ambiguous && <p><Badge tone="warn">AMBIGUOUS BAR</Badge> Target-first sensitivity: net unit P&amp;L {simulation.target_first_sensitivity?.net_pnl}.</p>}
    {expert && <dl>
      <div className="kv"><dt>Policy / plan / simulation</dt><dd><Hash value={simulation.policy_hash} /> / <Hash value={simulation.plan_hash} /> / <Hash value={simulation.result_hash} /></dd></div>
      <div className="kv"><dt>Source prediction</dt><dd><Hash value={plan.source_prediction_hash} /></dd></div>
      <div className="kv"><dt>Costs</dt><dd>{simulation.cost_method}; <Hash value={simulation.cost_hash} /></dd></div>
      <div className="kv"><dt>Fees / exit slippage</dt><dd>{exit?.fees ?? 'unavailable'} / {exit?.slippage_cost ?? 'unavailable'}</dd></div>
    </dl>}
  </section>;
}

export function PoliciesPage() {
  const { selection, update } = useCockpit();
  const definitions = useQuery('policies:definitions', (signal) => apiClient.getPolicyDefinitions(signal));
  const result = useQuery('policies:report', (signal) => apiClient.getPolicyReport(signal));
  const report = result.data?.report;
  const product = selection.product ?? report?.calibrations[0]?.product;
  const item = report?.calibrations.find((c) => c.product === product);
  const expert = selection.mode === 'expert';
  return <div className="stack">
    <section className="card">
      <h1>Calibration and TP/SL policies</h1>
      <Badge tone="warn">SYNTHETIC DEMONSTRATION</Badge>
      <p>Versioned evidence for a synthetic probability model and separate unit paper positions. Frozen model, backtest and replay references retain their original results.</p>
      <p>Real calibration: {definitions.data?.real_calibration ?? 'unavailable'}. Prediction intervals are not provided.</p>
      {definitions.status === 'error' && definitions.error && <ErrorState error={definitions.error} onRetry={definitions.refetch} />}
      {definitions.data && <p>Policy revision {definitions.data.spec.revision}: <Hash value={definitions.data.policy_hash} />.
        {' '}Minimum training sample {definitions.data.spec.calibration.min_train}, {definitions.data.spec.calibration.min_class} per class;
        {' '}minimum diagnostic sample {definitions.data.spec.calibration.min_evaluation}, {definitions.data.spec.calibration.min_evaluation_class} per class.</p>}
    </section>
    {result.status === 'loading' && <LoadingState label="Loading policy evidence" />}
    {result.status === 'error' && result.error && <ErrorState error={result.error} onRetry={result.refetch} />}
    {result.data?.state === 'NOT_CONFIGURED' && <EmptyState title="Policy evidence not configured" detail="Generate the offline synthetic policies demo, then serve it with --policy-root. The API never fits or simulates on a read." />}
    {report && <>
      <label>Evidence product{' '}<select className="control" value={product} onChange={(event) => update({ product: event.target.value })}>
        {!item && <option value={product}>{product} (no policy evidence)</option>}
        {report.calibrations.map((c) => <option key={c.product}>{c.product}</option>)}
      </select></label>
      {item ? <CalibrationCard item={item} expert={expert} /> : <EmptyState title="No policy evidence for this product" />}
      {report.simulations.filter((s) => s.plan.product === product).map((s) => <ProtectionCard key={s.result_hash} simulation={s} expert={expert} />)}
      <p>Report identity: <Hash value={result.data?.identity} />.</p>
      <ul>{report.limitations.map((text) => <li key={text}>{text}</li>)}</ul>
    </>}
  </div>;
}
