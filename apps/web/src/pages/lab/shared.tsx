/** Pieces shared by the Lab views: the token gate, a paged-section helper and the mode flag. */
import { useState, type ReactNode } from 'react';
import { Badge, ErrorState, Hash, LoadingState } from '../../components/States';
import type { JobStatus } from '../../api/labTypes';
import { formatEpoch, isActive, progressPercent } from '../../lib/lab';
import { ApiError } from '../../api/client';
import { clearLabToken, setLabToken, useLabToken } from '../../state/labToken';
import type { QueryResult } from '../../state/useQuery';
import type { ActionState } from '../../state/useLabAction';

/** The operator token form. The token stays in memory; the field is a password input so it is not shown or autofilled. */
export function TokenGate() {
  const { token } = useLabToken();
  const [draft, setDraft] = useState('');
  if (token) {
    return (
      <div className="row" role="status">
        <span className="badge" data-tone="ok">OPERATOR TOKEN SET (memory only)</span>
        <button className="control" onClick={() => clearLabToken()}>Forget token</button>
      </div>
    );
  }
  return (
    <form
      className="lab-form"
      aria-label="Operator token entry"
      onSubmit={(event) => { event.preventDefault(); if (draft.trim()) { setLabToken(draft); setDraft(''); } }}
    >
      <label>Operator token (HYPRL_MODEL_LAB_TOKEN)
        <input className="control" type="password" autoComplete="off" value={draft}
          onChange={(event) => setDraft(event.target.value)} />
      </label>
      <button className="control" type="submit" disabled={!draft.trim()}>Unlock</button>
      <p className="lab-note">Kept in this tab&apos;s memory only; a reload forgets it. It authorises the lab&apos;s reads and its synthetic job controls.</p>
    </form>
  );
}

export function LockedState({ what }: { what: string }) {
  return (
    <div className="state" role="status">
      <strong>{what} needs the operator token</strong>
      <span>The Model Lab endpoints are authenticated. Enter the token above to read them and run synthetic jobs.</span>
    </div>
  );
}

/** Loading / error / ready for a query; 401, 503 get a specific explanation instead of a raw message. */
export function QueryBoundary<T>({ query, label, children }: {
  query: QueryResult<T>; label: string; children: (data: T) => ReactNode;
}) {
  if (query.status === 'loading' && query.data === undefined) return <LoadingState label={label} />;
  if (query.status === 'error' && query.error) {
    const status = query.error instanceof ApiError ? query.error.status : 0;
    if (status === 503) {
      return (
        <div className="state" role="status">
          <strong>Not configured on this server</strong>
          <span>{query.error.message}. Start the API with the matching option (see the Lab documentation).</span>
        </div>
      );
    }
    if (status === 401) {
      return (
        <div className="state" role="alert">
          <strong className="negative">Token refused</strong>
          <span>The operator token was not accepted. Forget it and enter it again.</span>
          <button className="control" onClick={() => clearLabToken()}>Forget token</button>
        </div>
      );
    }
    return <ErrorState error={query.error} onRetry={query.refetch} />;
  }
  if (query.data === undefined) return <LoadingState label={label} />;
  return <>{children(query.data)}</>;
}

export function Synthetic() {
  return <span className="badge" data-tone="warn">SYNTHETIC</span>;
}

export function Timestamp({ value }: { value: string }) {
  return <time dateTime={value} title={value}>{value.replace('T', ' ').slice(0, 16)}</time>;
}

/** The outcome of a job control, announced politely (sent) or assertively (refused). */
export function ActionOutcome<T>({ state, done }: { state: ActionState<T>; done: (data: T) => ReactNode }) {
  if (state.status === 'idle') return null;
  if (state.status === 'sending') return <p role="status" className="lab-note">Sending to the local lab listener…</p>;
  if (state.status === 'error') return <p role="alert" className="negative">{state.message}</p>;
  return <div role="status">{done(state.data)}</div>;
}

/** Real-data training is not available from this page, and the page says so instead of hiding the option. */
export function WaitingAuthorization() {
  return (
    <p className="lab-note" data-testid="waiting-authorization">
      <span className="badge" data-tone="warn">WAITING_AUTHORIZATION</span>{' '}
      Real-data datasets and training are not available: they need an operator authorization that does not exist yet.
      This page builds and trains on clearly labelled synthetic data only; the authorized frozen replay stays read-only
      (Predictions and Monitoring tabs).
    </p>
  );
}

export function Progress({ job }: { job: Pick<JobStatus, 'progress' | 'state'> }) {
  const percent = progressPercent(job);
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
      aria-label="Job progress" data-state={job.state}><span style={{ width: `${percent}%` }} /></div>
  );
}

/** A job as the worker reports it: state, progress, resource limits and its structured log (codes only, no text). */
export function JobProgress({ job, expert = false }: { job: JobStatus; expert?: boolean }) {
  return (
    <div className="stack">
      <div className="row" style={{ flexWrap: 'wrap' }}>
        <strong><Hash value={job.id} chars={12} /></strong>
        <Badge tone={job.state === 'COMPLETE' ? 'ok' : isActive(job) ? 'warn' : 'off'}>{job.state}</Badge>
        {job.error_code && <Badge tone="warn">{job.error_code}</Badge>}
        {job.cancel_requested && <Badge tone="warn">CANCEL REQUESTED</Badge>}
      </div>
      <Progress job={job} />
      <p className="lab-note" aria-live="polite">
        {progressPercent(job)} % · limits {job.limits.wall_seconds} s wall, {job.limits.cpu_seconds} s CPU, {job.limits.memory_mb} MB, {job.limits.output_mb} MB output
        {job.worker_pid !== null && expert ? ` · worker pid ${job.worker_pid}` : ''}
      </p>
      <div className="table-scroll">
        <table className="data" aria-label="Job log">
          <thead><tr><th>#</th><th>Event</th><th>Progress</th><th>At</th></tr></thead>
          <tbody>
            {job.logs.map((log) => (
              <tr key={log.sequence}><td>{log.sequence}</td><td>{log.code}</td><td>{Math.round(log.progress * 100)} %</td><td>{formatEpoch(log.at)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
