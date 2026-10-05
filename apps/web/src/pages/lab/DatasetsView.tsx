/** Dataset builder: choose products, period, target and horizon; see the admissible decisions and the exclusions of a built dataset. */
import { useMemo, useState } from 'react';
import { apiClient } from '../../api/client';
import type { DatasetManifest, JobStatus } from '../../api/labTypes';
import { Badge, EmptyState, Hash } from '../../components/States';
import {
  DATASET_LIMITS, DEFAULT_DATASET_FORM, curlCommand, datasetRequest, formatCount, horizonLabel, validateDatasetForm,
} from '../../lib/lab';
import type { DatasetForm } from '../../lib/lab';
import { useCockpit } from '../../state/useCockpit';
import { useLabToken } from '../../state/labToken';
import { useQuery } from '../../state/useQuery';
import { LockedState, QueryBoundary, Synthetic } from './shared';

function exclusionsByReason(manifest: DatasetManifest): [string, number][] {
  const counts = new Map<string, number>();
  for (const item of manifest.exclusions) counts.set(item.reason, (counts.get(item.reason) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

const REASON_TEXT: Record<string, string> = {
  PRICE_FEATURE_WARMUP_OR_GAP: 'Not enough price history (indicator warm-up) or a gap in the prices.',
  LABEL_NOT_REALIZED_OR_GAP: 'The forward return was not yet realized at the end of the period, or a bar was missing.',
};

function Builder() {
  const [form, setForm] = useState<DatasetForm>(DEFAULT_DATASET_FORM);
  const problems = useMemo(() => validateDatasetForm(form), [form]);
  const request = useMemo(() => datasetRequest(form), [form]);
  const toggle = (product: string) => setForm((current) => ({
    ...current,
    products: current.products.includes(product)
      ? current.products.filter((item) => item !== product) : [...current.products, product],
  }));
  return (
    <section className="card" aria-label="Dataset builder">
      <h2 className="card-title">Build a dataset <Synthetic /></h2>
      <form className="lab-form" onSubmit={(event) => event.preventDefault()}>
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
      </form>
      <p className="lab-note" style={{ marginTop: 8 }}>
        Admissible decisions: a decision at T is kept only if every price it depends on was available by T and its
        label ({horizonLabel(form.horizonHours * 3600)} forward return) is kept outside the inputs. Warm-up, gaps,
        protected intervals and unresolved selected events are excluded with a reason, never filled. The registered
        models support exactly 4 h; other horizons build a dataset but no model will accept it.
      </p>
      {problems.length > 0 ? (
        <ul role="alert" className="negative">{problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>
      ) : (
        <>
          <p className="lab-note" style={{ margin: '12px 0 4px' }}>
            Request prepared, <strong>not sent</strong>. The server refuses writes that come from a browser page, so run it from your terminal:
          </p>
          <pre className="code-block" aria-label="Prepared dataset command">{curlCommand('/api/v1/lab/datasets', request)}</pre>
        </>
      )}
    </section>
  );
}

function DatasetDetail({ job, token }: { job: JobStatus; token: string }) {
  const { token: current, version } = useLabToken();
  const { selection } = useCockpit();
  const result = useQuery(`lab:dataset:${version}:${job.id}`, (signal) => apiClient.getLabDatasetResult(token || current, job.id, signal));
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

function BuiltDatasets() {
  const { token, version } = useLabToken();
  const [selected, setSelected] = useState<string | null>(null);
  const jobs = useQuery(token ? `lab:jobs:${version}` : null, (signal) => apiClient.getLabJobs(token, signal), { staleMs: 5_000 });
  if (!token) return <LockedState what="Built datasets" />;
  return (
    <section className="card" aria-label="Built datasets">
      <h2 className="card-title">Built datasets</h2>
      <QueryBoundary query={jobs} label="Loading dataset jobs">
        {(data) => {
          const datasets = data.jobs.filter((job) => job.kind === 'dataset');
          if (datasets.length === 0) return <EmptyState title="No dataset yet" detail="Run the prepared command above." />;
          const active = datasets.find((job) => job.id === selected) ?? datasets.find((job) => job.state === 'COMPLETE');
          return (
            <div className="grid grid-2">
              <ul className="lab-list" aria-label="Dataset jobs">
                {datasets.map((job) => (
                  <li key={job.id}>
                    <button className="row-button" aria-current={active?.id === job.id} onClick={() => setSelected(job.id)}>
                      <Hash value={job.id} chars={8} /> <Badge tone={job.state === 'COMPLETE' ? 'ok' : 'warn'}>{job.state}</Badge>
                    </button>
                  </li>
                ))}
              </ul>
              {active?.state === 'COMPLETE'
                ? <DatasetDetail job={active} token={token} />
                : <EmptyState title="Select a completed dataset" detail="Its manifest, exclusions and fingerprints appear here." />}
            </div>
          );
        }}
      </QueryBoundary>
    </section>
  );
}

export function DatasetsView() {
  return (
    <div className="stack">
      <Builder />
      <BuiltDatasets />
    </div>
  );
}
