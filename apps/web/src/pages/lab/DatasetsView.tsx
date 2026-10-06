/** Dataset builder: choose products, period, target and horizon, build it in a worker, and see its admissible decisions and exclusions. */
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient } from '../../api/client';
import type { DatasetManifest, JobStatus } from '../../api/labTypes';
import { Badge, EmptyState, Hash } from '../../components/States';
import {
  DATASET_LIMITS, DEFAULT_DATASET_FORM, curlCommand, datasetRequest, formatCount, horizonLabel, isActive, validateDatasetForm,
} from '../../lib/lab';
import { carrySelection } from '../../lib/cockpit';
import type { DatasetForm } from '../../lib/lab';
import { useCockpit } from '../../state/useCockpit';
import { useLabToken } from '../../state/labToken';
import { useQuery } from '../../state/useQuery';
import { useLabAction } from '../../state/useLabAction';
import { ActionOutcome, JobProgress, LockedState, QueryBoundary, Synthetic, WaitingAuthorization } from './shared';

function exclusionsByReason(manifest: DatasetManifest): [string, number][] {
  const counts = new Map<string, number>();
  for (const item of manifest.exclusions) counts.set(item.reason, (counts.get(item.reason) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

const REASON_TEXT: Record<string, string> = {
  PRICE_FEATURE_WARMUP_OR_GAP: 'Not enough price history (indicator warm-up) or a gap in the prices.',
  LABEL_NOT_REALIZED_OR_GAP: 'The forward return was not yet realized at the end of the period, or a bar was missing.',
};

function Builder({ onCreated }: { onCreated: (jobId: string) => void }) {
  const { token } = useLabToken();
  const { selection } = useCockpit();
  const [form, setForm] = useState<DatasetForm>(DEFAULT_DATASET_FORM);
  const problems = useMemo(() => validateDatasetForm(form), [form]);
  const request = useMemo(() => datasetRequest(form), [form]);
  const build = useLabAction((body: Record<string, unknown>) => apiClient.createLabDataset(token, body),
    (data) => onCreated(data.job_id));
  const toggle = (product: string) => setForm((current) => ({
    ...current,
    products: current.products.includes(product)
      ? current.products.filter((item) => item !== product) : [...current.products, product],
  }));
  const blocked = problems.length > 0 || !token || build.state.status === 'sending';
  return (
    <section className="card" aria-label="Dataset builder">
      <h2 className="card-title">Build a dataset <Synthetic /></h2>
      <form className="lab-form" aria-label="Dataset configuration"
        onSubmit={(event) => { event.preventDefault(); if (!blocked) void build.run(request); }}>
        <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
          <legend className="lab-note">Data source</legend>
          <label style={{ flexDirection: 'row', alignItems: 'center' }}>
            <input type="radio" name="source" checked readOnly /> Synthetic prices (labelled SYNTHETIC)
          </label>
          <label style={{ flexDirection: 'row', alignItems: 'center' }}>
            <input type="radio" name="source" disabled aria-describedby="real-data-state" /> Real prices: WAITING_AUTHORIZATION
          </label>
        </fieldset>
        <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
          <legend className="lab-note">Products</legend>
          <div className="row">
            {DATASET_LIMITS.products.map((product) => (
              <label key={product} style={{ flexDirection: 'row', alignItems: 'center' }}>
                <input type="checkbox" checked={form.products.includes(product)} onChange={() => toggle(product)} />
                {product}
              </label>
            ))}
          </div>
        </fieldset>
        <label>Start (ISO instant)
          <input className="control" value={form.start} onChange={(event) => setForm({ ...form, start: event.target.value })} />
        </label>
        <label>Hourly bars ({DATASET_LIMITS.minBars}–{DATASET_LIMITS.maxBars})
          <input className="control" type="number" value={form.bars}
            onChange={(event) => setForm({ ...form, bars: Number(event.target.value) })} />
        </label>
        <label>Horizon (hours)
          <input className="control" type="number" value={form.horizonHours}
            onChange={(event) => setForm({ ...form, horizonHours: Number(event.target.value) })} />
        </label>
        <label>Seed
          <input className="control" type="number" value={form.seed}
            onChange={(event) => setForm({ ...form, seed: Number(event.target.value) })} />
        </label>
        <label>Target
          <input className="control" value="forward_return" readOnly aria-readonly="true" />
        </label>
        <button className="control" type="submit" disabled={blocked}>
          {build.state.status === 'sending' ? 'Building…' : 'Build dataset'}
        </button>
      </form>
      <div id="real-data-state"><WaitingAuthorization /></div>
      <p className="lab-note" style={{ marginTop: 8 }}>
        Period: {form.bars} hourly bars from {form.start.slice(0, 16)} UTC. Admissible decisions: a decision at T is kept only
        if every price it depends on was available by T and its label ({horizonLabel(form.horizonHours * 3600)} forward return)
        is kept outside the inputs. Warm-up, gaps, protected intervals and unresolved selected events are excluded with a
        reason, never filled. The registered models support exactly 4 h; other horizons build a dataset but no model will accept it.
      </p>
      {problems.length > 0 && <ul role="alert" className="negative">{problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>}
      {!token && <p className="lab-note">Enter the operator token above to build.</p>}
      <ActionOutcome state={build.state} done={(data) => (
        <p>Dataset job <Hash value={data.job_id} chars={8} /> queued in an isolated worker. It is followed below.</p>
      )} />
      {problems.length === 0 && (
        <details open={selection.mode === 'expert'}>
          <summary>Equivalent terminal command (reproduction)</summary>
          <pre className="code-block" aria-label="Prepared dataset command">{curlCommand('/api/v1/lab/datasets', request)}</pre>
        </details>
      )}
    </section>
  );
}

function DatasetDetail({ job, token }: { job: JobStatus; token: string }) {
  const { version } = useLabToken();
  const { selection } = useCockpit();
  const result = useQuery(`lab:dataset:${version}:${job.id}`, (signal) => apiClient.getLabDatasetResult(token, job.id, signal));
  return (
    <QueryBoundary query={result} label="Loading dataset manifest">
      {({ result: dataset }) => {
        const manifest = dataset.manifest;
        const reasons = exclusionsByReason(manifest);
        const total = manifest.counts.included + manifest.counts.excluded;
        return (
          <div className="stack">
            <div className="row" style={{ flexWrap: 'wrap' }}>
              <strong>{manifest.dataset_id}</strong>
              {manifest.synthetic && <Synthetic />}
              <Badge tone="off">TARGET {manifest.target}</Badge>
              <Badge tone="off">HORIZON {horizonLabel(manifest.horizon_seconds)}</Badge>
            </div>
            <p>
              {formatCount(manifest.counts.included)} of {formatCount(total)} candidate decisions are admissible
              ({formatCount(manifest.counts.excluded)} excluded) for {manifest.products.join(' and ')},
              {' '}{manifest.decision_start.slice(0, 16)} → {manifest.decision_end.slice(0, 16)} UTC.
              {selection.mode === 'beginner' && ' Excluded decisions are left out on purpose: using them would let the model peek at the future.'}
            </p>
            <table className="data" aria-label="Exclusions by reason">
              <thead><tr><th>Exclusion reason</th><th>Decisions</th><th>Meaning</th></tr></thead>
              <tbody>
                {reasons.map(([reason, count]) => (
                  <tr key={reason}><td>{reason}</td><td>{count}</td><td>{REASON_TEXT[reason] ?? 'See the dataset policy.'}</td></tr>
                ))}
                {reasons.length === 0 && <tr><td colSpan={3}>No exclusion recorded.</td></tr>}
              </tbody>
            </table>
            <p className="lab-note">
              {manifest.policies.event_columns.length === 0
                ? 'Price features only: no event feature was selected, so no event availability is claimed.'
                : `Event features: ${manifest.policies.event_columns.join(', ')}. An unavailable event excludes the decision; it is never filled with zero.`}
            </p>
            {selection.mode === 'expert' && (
              <dl>
                <div className="kv"><dt>Dataset hash</dt><dd><Hash value={dataset.dataset_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Features hash</dt><dd><Hash value={manifest.features_hash} chars={20} /></dd></div>
                <div className="kv"><dt>Snapshots bound</dt><dd>{formatCount(manifest.snapshot_hashes.length)}</dd></div>
                <div className="kv"><dt>Feature columns</dt><dd>{manifest.policies.columns.join(', ')}</dd></div>
                <div className="kv"><dt>Calendar</dt><dd>{manifest.policies.calendar}</dd></div>
                <div className="kv"><dt>Availability rule</dt><dd>{manifest.policies.availability}</dd></div>
                <div className="kv"><dt>Splits</dt><dd>{manifest.splits.state} ({manifest.splits.method}); assigned by each experiment</dd></div>
                <div className="kv"><dt>Per product</dt><dd>{Object.entries(manifest.counts.by_product).map(([p, n]) => `${p} ${n}`).join(' · ')}</dd></div>
              </dl>
            )}
          </div>
        );
      }}
    </QueryBoundary>
  );
}

function BuiltDatasets({ created }: { created: string | null }) {
  const { token, version } = useLabToken();
  const { params } = useCockpit();
  const [selected, setSelected] = useState<string | null>(null);
  const jobs = useQuery(token ? `lab:jobs:${version}` : null, (signal) => apiClient.getLabJobs(token, signal), { staleMs: 1_000 });
  const cancel = useLabAction((id: string) => apiClient.cancelLabJob(token, id), () => jobs.refetch());
  const anyActive = jobs.data?.jobs.some((job) => job.kind === 'dataset' && isActive(job)) ?? false;
  const { refetch } = jobs;
  useEffect(() => { if (created) refetch(); }, [created, refetch]);
  useEffect(() => {
    if (!anyActive) return undefined;
    const timer = setInterval(refetch, 1_500);
    return () => clearInterval(timer);
  }, [anyActive, refetch]);
  if (!token) return <LockedState what="Built datasets" />;
  return (
    <section className="card" aria-label="Built datasets">
      <h2 className="card-title">Built datasets</h2>
      <QueryBoundary query={jobs} label="Loading dataset jobs">
        {(data) => {
          const datasets = data.jobs.filter((job) => job.kind === 'dataset');
          if (datasets.length === 0) return <EmptyState title="No dataset yet" detail="Build one above." />;
          const active = datasets.find((job) => job.id === (selected ?? created))
            ?? datasets.find((job) => job.state === 'COMPLETE') ?? datasets[0];
          return (
            <div className="grid grid-2">
              <ul className="lab-list" aria-label="Dataset jobs">
                {datasets.map((job) => (
                  <li key={job.id}>
                    <button className="row-button" aria-current={active?.id === job.id} onClick={() => setSelected(job.id)}>
                      <Hash value={job.id} chars={8} /> <Badge tone={job.state === 'COMPLETE' ? 'ok' : isActive(job) ? 'warn' : 'off'}>{job.state}</Badge>
                    </button>
                  </li>
                ))}
              </ul>
              {active && active.state === 'COMPLETE' && (
                <div className="stack">
                  <DatasetDetail job={active} token={token} />
                  <Link className="control" to={{ pathname: '/lab/experiments', search: withDataset(params, active.id) }}>
                    Next: configure an experiment on this dataset
                  </Link>
                </div>
              )}
              {active && active.state !== 'COMPLETE' && (
                <div className="stack" aria-label="Dataset job">
                  <JobProgress job={active} />
                  {isActive(active) && (
                    <button className="control" onClick={() => void cancel.run(active.id)}
                      disabled={cancel.state.status === 'sending' || active.cancel_requested}>
                      {active.cancel_requested ? 'Cancellation requested' : 'Cancel this job'}
                    </button>
                  )}
                  <ActionOutcome state={cancel.state} done={(status) => <p>Cancellation recorded: {status.state}.</p>} />
                  {active.state === 'FAILED' && (
                    <p role="alert" className="negative">The worker failed ({active.error_code ?? 'no code'}); nothing was published. Adjust the configuration and build again.</p>
                  )}
                </div>
              )}
            </div>
          );
        }}
      </QueryBoundary>
    </section>
  );
}

/** The cockpit selection plus the dataset job the Experiments tab should preselect. */
function withDataset(params: URLSearchParams, jobId: string): string {
  const next = new URLSearchParams(carrySelection(params));
  next.set('dataset', jobId);
  return `?${next.toString()}`;
}

export function DatasetsView() {
  const [created, setCreated] = useState<string | null>(null);
  return (
    <div className="stack">
      <Builder onCreated={setCreated} />
      <BuiltDatasets created={created} />
    </div>
  );
}
