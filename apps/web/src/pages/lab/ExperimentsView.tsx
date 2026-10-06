/** Experiments: configure and launch a job, follow it (progress, logs, limits), cancel it, compare with baselines, read artifacts and hashes, reproduce, open its monitoring. */
import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { apiClient } from '../../api/client';
import type { ExperimentResult, JobStatus, ModelDescriptor } from '../../api/labTypes';
import { Badge, EmptyState, Hash } from '../../components/States';
import { carrySelection } from '../../lib/cockpit';
import {
  curlCommand, horizonLabel, isActive, modelFit, modelRole, testComparison, verdictSentence,
} from '../../lib/lab';
import { formatPercent, formatRatio } from '../../lib/format';
import { useCockpit } from '../../state/useCockpit';
import { useLabToken } from '../../state/labToken';
import { useLabAction } from '../../state/useLabAction';
import { useQuery } from '../../state/useQuery';
import { ActionOutcome, JobProgress, LockedState, Progress, QueryBoundary, Synthetic, WaitingAuthorization } from './shared';

function Prepare({ datasetJobs, models, token, onLaunched }: {
  datasetJobs: JobStatus[]; models: ModelDescriptor[]; token: string; onLaunched: (jobId: string) => void;
}) {
  const { version } = useLabToken();
  const { selection } = useCockpit();
  const [search] = useSearchParams();
  const trainable = models.filter((model) => model.contract.capabilities.includes('train'));
  const preferredDataset = datasetJobs.find((job) => job.id === search.get('dataset'))?.id;
  const preferredModel = trainable.find((item) => item.contract.model_id === selection.model)?.contract.model_id;
  const [datasetJob, setDatasetJob] = useState('');
  const [model, setModel] = useState('');
  const [embargo, setEmbargo] = useState(3600);
  const chosenJob = datasetJob || preferredDataset || datasetJobs[0]?.id || '';
  const chosenModel = model || preferredModel || trainable[0]?.contract.model_id || '';
  // A dataset is named by the hash its job result carries.
  const dataset = useQuery(chosenJob ? `lab:dataset:${version}:${chosenJob}` : null,
    (signal) => apiClient.getLabDatasetResult(token, chosenJob, signal));
  const hash = dataset.data?.result.dataset_hash ?? '';
  const manifest = dataset.data?.result.manifest;
  const descriptor = models.find((item) => item.contract.model_id === chosenModel);
  const fit = descriptor && manifest ? modelFit(descriptor.contract, manifest) : [];
  const launch = useLabAction((body: Record<string, unknown>) => apiClient.createLabExperiment(token, body),
    (data) => onLaunched(data.job_id));
  const body = { dataset_hash: hash, model_id: chosenModel, embargo_seconds: embargo };
  const valid = /^[a-f0-9]{64}$/.test(hash) && chosenModel !== '' && Number.isInteger(embargo)
    && embargo >= 0 && embargo <= 86400 && fit.length === 0;
  return (
    <section className="card" aria-label="Prepare an experiment">
      <h2 className="card-title">Configure an experiment <Synthetic /></h2>
      <form className="lab-form" aria-label="Experiment configuration"
        onSubmit={(event) => { event.preventDefault(); if (valid && launch.state.status !== 'sending') void launch.run(body); }}>
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
        <label>Embargo (seconds, 0–86400)
          <input className="control" type="number" value={embargo} onChange={(event) => setEmbargo(Number(event.target.value))} />
        </label>
        <button className="control" type="submit" disabled={!valid || launch.state.status === 'sending'}>
          {launch.state.status === 'sending' ? 'Launching…' : 'Launch experiment'}
        </button>
      </form>
      {descriptor && (
        <p className="lab-note" style={{ marginTop: 8 }} aria-label="Chosen model">
          <Badge tone={modelRole(descriptor).tone}>{modelRole(descriptor).label}</Badge>{' '}
          declares {descriptor.contract.capabilities.join(', ').toUpperCase()} · horizons {descriptor.contract.horizons_seconds.map(horizonLabel).join(', ')}
          {' '}· outputs: return only, the others are not provided.
          {manifest && ` Dataset: ${manifest.products.join(' + ')}, ${horizonLabel(manifest.horizon_seconds)} ${manifest.target}, ${manifest.counts.included} admissible decisions.`}
        </p>
      )}
      <p className="lab-note" style={{ marginTop: 8 }}>
        The criterion, baselines (ZERO, TRAIN_MEAN), purge/embargo and resource budgets are fixed by the server before any
        result exists; they are not editable here. Transformations are fitted on the training split only.
      </p>
      <WaitingAuthorization />
      {fit.length > 0 && <ul role="alert" className="negative">{fit.map((problem) => <li key={problem}>{problem}</li>)}</ul>}
      {!valid && fit.length === 0 && (
        <p role="alert" className="negative">Select a completed dataset, a trainable model and an embargo from 0 to 86400 seconds.</p>
      )}
      <ActionOutcome state={launch.state} done={(data) => (
        <p>Experiment job <Hash value={data.job_id} chars={8} /> queued; prepared manifest <Hash value={data.fingerprint ?? ''} chars={12} /> recorded before any result.</p>
      )} />
      {valid && (
        <details open={selection.mode === 'expert'}>
          <summary>Equivalent terminal command (reproduction)</summary>
          <pre className="code-block" aria-label="Prepared experiment command">{curlCommand('/api/v1/lab/experiments', body)}</pre>
        </details>
      )}
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
    </div>
  );
}

function Reproduce({ manifest, fingerprint, token, onLaunched }: {
  manifest: ExperimentResult['result']['manifest']; fingerprint: string; token: string; onLaunched: (jobId: string) => void;
}) {
  const body = { dataset_hash: manifest.dataset_hash, model_id: manifest.parameters.model_id, embargo_seconds: manifest.splits.embargo_seconds };
  const again = useLabAction((payload: Record<string, unknown>) => apiClient.createLabExperiment(token, payload), (data) => onLaunched(data.job_id));
  return (
    <details>
      <summary>Reproduce</summary>
      <p className="lab-note">
        Submitting the same dataset hash, model and embargo produces a new job whose result fingerprint should equal
        <Hash value={fingerprint} chars={16} /> within the recorded numeric runtime. A different fingerprint is a finding, not a rounding detail.
        Every previous run is kept.
      </p>
      <button className="control" onClick={() => void again.run(body)} disabled={again.state.status === 'sending'}>Run the reproduction</button>
      <ActionOutcome state={again.state} done={(data) => <p>Reproduction job <Hash value={data.job_id} chars={8} /> queued.</p>} />
      <pre className="code-block" aria-label="Reproduction command">{curlCommand('/api/v1/lab/experiments', body)}</pre>
    </details>
  );
}

function OpenMonitoring({ job, monitoring, token }: { job: JobStatus; monitoring: JobStatus | undefined; token: string }) {
  const navigate = useNavigate();
  const { params } = useCockpit();
  const target = (id: string) => {
    const next = new URLSearchParams(carrySelection(params));
    next.set('monitor', id);
    return { pathname: '/lab/monitoring', search: `?${next.toString()}` };
  };
  const open = useLabAction((id: string) => apiClient.createLabMonitoring(token, id), (data) => navigate(target(data.job_id)));
  return (
    <section aria-label="Monitoring of this experiment" className="stack">
      {monitoring ? (
        <Link className="control" to={target(monitoring.id)}>Open the monitoring of these predictions ({monitoring.state})</Link>
      ) : (
        <button className="control" onClick={() => void open.run(job.id)} disabled={open.state.status === 'sending'}>
          Open the monitoring of these predictions
        </button>
      )}
      <p className="lab-note">
        Imports this experiment&apos;s exact predictions into a private ledger (no refit), fixes a reference on the validation split
        and monitors the test split against it, in a worker.
      </p>
      <ActionOutcome state={open.state} done={() => <p>Monitoring job queued.</p>} />
    </section>
  );
}

function JobDetail({ job, token, expert, monitoring, onLaunched }: {
  job: JobStatus; token: string; expert: boolean; monitoring: JobStatus | undefined; onLaunched: (jobId: string) => void;
}) {
  const { version } = useLabToken();
  const result = useQuery(
    job.state === 'COMPLETE' ? `lab:result:${version}:${job.id}` : null,
    (signal) => apiClient.getLabExperimentResult(token, job.id, signal),
  );
  const cancel = useLabAction((id: string) => apiClient.cancelLabJob(token, id), () => onLaunched(job.id));
  return (
    <div className="stack" aria-label="Experiment job">
      <JobProgress job={job} expert={expert} />
      {isActive(job) && (
        <>
          <button className="control" onClick={() => void cancel.run(job.id)} disabled={cancel.state.status === 'sending' || job.cancel_requested}>
            {job.cancel_requested ? 'Cancellation requested' : 'Cancel this job'}
          </button>
          <p className="lab-note">Cancelling interrupts the worker; a cancelled job never publishes a result. Its prepared configuration is kept.</p>
        </>
      )}
      <ActionOutcome state={cancel.state} done={(status) => <p>Cancellation recorded: {status.state}.</p>} />
      {job.state === 'FAILED' && (
        <p role="alert" className="negative">The worker failed ({job.error_code ?? 'no code'}); no result was published. The prepared manifest is kept.</p>
      )}
      {job.state === 'COMPLETE' && (
        <QueryBoundary query={result} label="Loading experiment result">
          {(data) => (
            <>
              <ResultPanel result={data.result} expert={expert} />
              <Reproduce manifest={data.result.manifest} fingerprint={data.result.fingerprint} token={token} onLaunched={onLaunched} />
              <OpenMonitoring job={job} monitoring={monitoring} token={token} />
            </>
          )}
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
  const launched = (jobId: string) => { setSelected(jobId); refetch(); };
  // Follow running jobs: poll only while something is active, so an idle page makes no request.
  useEffect(() => {
    if (!anyActive) return undefined;
    const timer = setInterval(refetch, 1_500);
    return () => clearInterval(timer);
  }, [anyActive, refetch]);
  if (!token) return <LockedState what="Experiments" />;
  return (
    <div className="stack">
      <QueryBoundary query={jobs} label="Loading jobs">
        {(data) => {
          const experiments = data.jobs.filter((job) => job.kind === 'experiment');
          const active = experiments.find((job) => job.id === selected) ?? experiments[0];
          const monitoring = active && data.jobs.find((job) => job.kind === 'monitoring' && job.subject === active.id
            && job.state !== 'CANCELLED' && job.state !== 'FAILED');
          return (
            <>
              <Prepare datasetJobs={data.jobs.filter((job) => job.kind === 'dataset' && job.state === 'COMPLETE')} token={token}
                models={models.data?.models ?? []} onLaunched={launched} />
              <section className="card" aria-label="Experiment jobs">
                <h2 className="card-title">Jobs <span className="muted">({data.worker_limit} worker at a time)</span></h2>
                {experiments.length === 0 ? <EmptyState title="No experiment yet" detail="Launch one above." /> : (
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
                    {active && <JobDetail key={active.id} job={active} token={token} expert={expert} monitoring={monitoring} onLaunched={launched} />}
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
