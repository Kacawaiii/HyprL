/** Experiments: prepare, follow a job (progress, logs, limits), compare with baselines, read artifacts and hashes, reproduce. */
import { useEffect, useState } from 'react';
import { apiClient } from '../../api/client';
import type { ExperimentResult, JobStatus, ModelDescriptor } from '../../api/labTypes';
import { Badge, EmptyState, Hash } from '../../components/States';
import {
  curlCommand, formatEpoch, isActive, progressPercent, testComparison, verdictSentence,
} from '../../lib/lab';
import { formatPercent, formatRatio } from '../../lib/format';
import { useCockpit } from '../../state/useCockpit';
import { useLabToken } from '../../state/labToken';
import { useQuery } from '../../state/useQuery';
import { LockedState, QueryBoundary, Synthetic } from './shared';

function Progress({ job }: { job: Pick<JobStatus, 'progress' | 'state'> }) {
  const percent = progressPercent(job);
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
      aria-label="Job progress" data-state={job.state}><span style={{ width: `${percent}%` }} /></div>
  );
}

function Prepare({ datasetJobs, models, token }: { datasetJobs: JobStatus[]; models: ModelDescriptor[]; token: string }) {
  const { version } = useLabToken();
  const trainable = models.filter((model) => model.contract.capabilities.includes('train'));
  const [datasetJob, setDatasetJob] = useState('');
  const [model, setModel] = useState('');
  const [embargo, setEmbargo] = useState(3600);
  const chosenJob = datasetJob || datasetJobs[0]?.id || '';
  const chosenModel = model || trainable[0]?.contract.model_id || '';
  // A dataset is named by the hash its job result carries.
  const dataset = useQuery(chosenJob ? `lab:dataset:${version}:${chosenJob}` : null,
    (signal) => apiClient.getLabDatasetResult(token, chosenJob, signal));
  const hash = dataset.data?.result.dataset_hash ?? '';
  const valid = /^[a-f0-9]{64}$/.test(hash) && chosenModel !== '' && Number.isInteger(embargo) && embargo >= 0;
  return (
    <section className="card" aria-label="Prepare an experiment">
      <h2 className="card-title">Configure an experiment <Synthetic /></h2>
      <form className="lab-form" onSubmit={(event) => event.preventDefault()}>
        <label>Dataset
          <select className="control" value={chosenJob} onChange={(event) => setDatasetJob(event.target.value)}>
            {datasetJobs.length === 0 && <option value="">no completed dataset</option>}
            {datasetJobs.map((job) => <option key={job.id} value={job.id}>job {job.id.slice(0, 8)}</option>)}
          </select>
        </label>
        <label>Model (can train)
          <select className="control" value={chosenModel} onChange={(event) => setModel(event.target.value)}>
            {trainable.map((item) => <option key={item.contract.model_id} value={item.contract.model_id}>{item.contract.model_id}</option>)}
          </select>
        </label>
        <label>Embargo (seconds)
          <input className="control" type="number" value={embargo} onChange={(event) => setEmbargo(Number(event.target.value))} />
        </label>
      </form>
      <p className="lab-note" style={{ marginTop: 8 }}>
        The criterion, baselines (ZERO, TRAIN_MEAN), purge/embargo and resource budgets are fixed by the server before any
        result exists; they are not editable here. Transformations are fitted on the training split only.
      </p>
      {valid ? (
        <pre className="code-block" aria-label="Prepared experiment command">
          {curlCommand('/api/v1/lab/experiments', { dataset_hash: hash, model_id: chosenModel, embargo_seconds: embargo })}
        </pre>
      ) : <p role="alert" className="negative">Select a completed dataset, a trainable model and a non-negative embargo.</p>}
    </section>
  );
}

function ResultPanel({ result, expert }: { result: ExperimentResult['result']; expert: boolean }) {
  const products = Object.keys(result.metrics);
  const verdicts = verdictSentence(result);
  const manifest = result.manifest;
  return (
    <div className="stack">
      {verdicts.map((verdict) => (
        <p key={verdict.product}><Badge tone={verdict.met ? 'ok' : 'warn'}>{verdict.met ? 'CRITERION MET' : 'CRITERION NOT MET'}</Badge> {verdict.text}</p>
      ))}
      {products.map((product) => (
        <table className="data" key={product} aria-label={`${product} test comparison`}>
          <caption className="lab-note" style={{ textAlign: 'left' }}>
            {product}, test split, same decisions for every row (lower error is better)
          </caption>
          <thead><tr><th>Predictor</th><th>MAE</th><th>RMSE</th><th>Decisions</th></tr></thead>
          <tbody>
            {testComparison(result, product).map((row) => (
              <tr key={row.name}><td>{row.name}</td><td>{formatRatio(row.mae)}</td><td>{formatRatio(row.rmse)}</td><td>{row.count}</td></tr>
            ))}
          </tbody>
        </table>
      ))}
      <table className="data" aria-label="Synthetic backtest">
        <caption className="lab-note" style={{ textAlign: 'left' }}>Economic backtest on synthetic prices, with costs (not a trading result)</caption>
        <thead><tr><th>Product</th><th>Net return</th><th>Gross return</th><th>Max drawdown</th><th>Costs</th><th>Fills</th></tr></thead>
        <tbody>
          {Object.entries(result.backtests).map(([product, backtest]) => (
            <tr key={product}>
              <td>{product}</td><td>{formatPercent(backtest.metrics.net_return)}</td><td>{formatPercent(backtest.metrics.gross_return)}</td>
              <td>{formatPercent(backtest.metrics.max_drawdown)}</td><td>{formatRatio(backtest.metrics.total_execution_cost)}</td>
              <td>{backtest.metrics.fill_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="lab-note">
        Shadow replay: {result.shadow.mode}, {result.shadow.events} events, chain {result.shadow.chain.verified ? 'verified' : 'NOT verified'},
        broker {result.shadow.broker_connected ? 'connected' : 'not connected'}. {result.limitations.join(' · ')}.
      </p>
      <details open={expert}>
        <summary>Hypothesis, splits and artifacts</summary>
        <dl>
          <div className="kv"><dt>Hypothesis</dt><dd>{manifest.hypothesis.statement}</dd></div>
          <div className="kv"><dt>Falsified if</dt><dd>{manifest.hypothesis.falsification}</dd></div>
          <div className="kv"><dt>Split method</dt><dd>{manifest.splits.method}; purge {manifest.splits.purge_seconds / 3600} h, embargo {manifest.splits.embargo_seconds / 3600} h</dd></div>
          {(['train', 'validation', 'test'] as const).map((role) => (
            <div className="kv" key={role}>
              <dt>{role}</dt>
              <dd>{manifest.splits.roles[role].rows} rows · {manifest.splits.roles[role].first.slice(0, 13)} → {manifest.splits.roles[role].last.slice(0, 13)}</dd>
            </div>
          ))}
          <div className="kv"><dt>Purged / embargoed</dt><dd>{manifest.splits.exclusions.length} decisions</dd></div>
          <div className="kv"><dt>Transformations</dt><dd>{manifest.parameters.transformations}</dd></div>
          <div className="kv"><dt>Result fingerprint</dt><dd><Hash value={result.fingerprint} chars={20} /></dd></div>
          <div className="kv"><dt>Dataset hash</dt><dd><Hash value={manifest.dataset_hash} chars={20} /></dd></div>
          <div className="kv"><dt>Model contract</dt><dd><Hash value={manifest.model_contract_hash} chars={20} /></dd></div>
          {Object.entries(manifest.artifacts).map(([name, value]) => (
            <div className="kv" key={name}><dt>{name}</dt><dd>{typeof value === 'string' ? <Hash value={value} chars={20} /> : `${Object.keys(value as object).length} entries`}</dd></div>
          ))}
        </dl>
      </details>
      {expert && (
        <details>
          <summary>Runtime and source hashes</summary>
          <dl>
            {Object.entries(manifest.parameters.runtime).filter(([key]) => key !== 'source_hashes').map(([key, value]) => (
              <div className="kv" key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>
            ))}
            {Object.entries(manifest.parameters.runtime.source_hashes ?? {}).map(([file, hash]) => (
              <div className="kv" key={file}><dt>{file}</dt><dd><Hash value={hash} chars={16} /></dd></div>
            ))}
          </dl>
        </details>
      )}
      <details>
        <summary>Reproduce</summary>
        <p className="lab-note">
          Submitting the same dataset hash, model and embargo produces a new job whose result fingerprint should equal
          <Hash value={result.fingerprint} chars={16} /> within the recorded numeric runtime. A different fingerprint is a finding, not a rounding detail.
        </p>
        <pre className="code-block" aria-label="Reproduction command">
          {curlCommand('/api/v1/lab/experiments', {
            dataset_hash: manifest.dataset_hash, model_id: manifest.parameters.model_id, embargo_seconds: manifest.splits.embargo_seconds,
          })}
        </pre>
      </details>
    </div>
  );
}

function JobDetail({ job, token, expert }: { job: JobStatus; token: string; expert: boolean }) {
  const { version } = useLabToken();
  const result = useQuery(
    job.state === 'COMPLETE' ? `lab:result:${version}:${job.id}` : null,
    (signal) => apiClient.getLabExperimentResult(token, job.id, signal),
  );
  return (
    <div className="stack">
      <div className="row" style={{ flexWrap: 'wrap' }}>
        <strong><Hash value={job.id} chars={12} /></strong>
        <Badge tone={job.state === 'COMPLETE' ? 'ok' : isActive(job) ? 'warn' : 'off'}>{job.state}</Badge>
        {job.error_code && <Badge tone="warn">{job.error_code}</Badge>}
        {job.cancel_requested && <Badge tone="warn">CANCEL REQUESTED</Badge>}
      </div>
      <Progress job={job} />
      <p className="lab-note">
        {progressPercent(job)} % · limits {job.limits.wall_seconds} s wall, {job.limits.cpu_seconds} s CPU, {job.limits.memory_mb} MB, {job.limits.output_mb} MB output
        {job.worker_pid !== null && expert ? ` · worker pid ${job.worker_pid}` : ''}
      </p>
      <table className="data" aria-label="Job log">
        <thead><tr><th>#</th><th>Event</th><th>Progress</th><th>At</th></tr></thead>
        <tbody>
          {job.logs.map((log) => (
            <tr key={log.sequence}><td>{log.sequence}</td><td>{log.code}</td><td>{Math.round(log.progress * 100)} %</td><td>{formatEpoch(log.at)}</td></tr>
          ))}
        </tbody>
      </table>
      {isActive(job) && (
        <>
          <p className="lab-note">A page cannot cancel a job (the server accepts no browser-originated write). Cancel from your terminal:</p>
          <pre className="code-block" aria-label="Cancel command">{curlCommand(`/api/v1/lab/jobs/${job.id}/cancel`, {})}</pre>
        </>
      )}
      {job.state === 'COMPLETE' && (
        <QueryBoundary query={result} label="Loading experiment result">
          {(data) => <ResultPanel result={data.result} expert={expert} />}
        </QueryBoundary>
      )}
    </div>
  );
}

export function ExperimentsView() {
  const { token, version } = useLabToken();
  const { selection } = useCockpit();
  const expert = selection.mode === 'expert';
  const [selected, setSelected] = useState<string | null>(null);
  const jobs = useQuery(token ? `lab:jobs:${version}` : null, (signal) => apiClient.getLabJobs(token, signal), { staleMs: 1_000 });
  const models = useQuery(token ? `lab:models:${version}` : null, (signal) => apiClient.getLabModels(token, signal));
  const anyActive = jobs.data?.jobs.some(isActive) ?? false;
  const { refetch } = jobs;
  // Follow running jobs: poll only while something is active, so an idle page makes no request.
  useEffect(() => {
    if (!anyActive) return undefined;
    const timer = setInterval(refetch, 2_000);
    return () => clearInterval(timer);
  }, [anyActive, refetch]);
  if (!token) return <LockedState what="Experiments" />;
  return (
    <div className="stack">
      <QueryBoundary query={jobs} label="Loading jobs">
        {(data) => {
          const experiments = data.jobs.filter((job) => job.kind === 'experiment');
          const active = experiments.find((job) => job.id === selected) ?? experiments[0];
          return (
            <>
              <Prepare datasetJobs={data.jobs.filter((job) => job.kind === 'dataset' && job.state === 'COMPLETE')} token={token}
                models={models.data?.models ?? []} />
              <section className="card" aria-label="Experiment jobs">
                <h2 className="card-title">Jobs <span className="muted">({data.worker_limit} worker at a time)</span></h2>
                {experiments.length === 0 ? <EmptyState title="No experiment yet" detail="Run the prepared command above." /> : (
                  <div className="grid grid-2">
                    <ul className="lab-list" aria-label="Experiment list">
                      {experiments.map((job) => (
                        <li key={job.id}>
                          <button className="row-button" aria-current={active?.id === job.id} onClick={() => setSelected(job.id)}>
                            <Hash value={job.id} chars={8} /> <Badge tone={job.state === 'COMPLETE' ? 'ok' : isActive(job) ? 'warn' : 'off'}>{job.state}</Badge>
                            <span style={{ flex: 1 }}><Progress job={job} /></span>
                          </button>
                        </li>
                      ))}
                    </ul>
                    {active && <JobDetail key={active.id} job={active} token={token} expert={expert} />}
                  </div>
                )}
              </section>
            </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
