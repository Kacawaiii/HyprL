/** Pieces shared by the Lab views: the token gate, a paged-section helper and the mode flag. */
import { useState, type ReactNode } from 'react';
import { ErrorState, LoadingState } from '../../components/States';
import { ApiError } from '../../api/client';
import { clearLabToken, setLabToken, useLabToken } from '../../state/labToken';
import type { QueryResult } from '../../state/useQuery';

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
      <p className="lab-note">Kept in this tab&apos;s memory only; a reload forgets it. It authorises reading, never writing.</p>
    </form>
  );
}

export function LockedState({ what }: { what: string }) {
  return (
    <div className="state" role="status">
      <strong>{what} needs the operator token</strong>
      <span>The Model Lab endpoints are authenticated. Enter the token above to read them.</span>
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
