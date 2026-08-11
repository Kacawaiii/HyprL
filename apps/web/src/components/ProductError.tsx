/** A product error surface: a code, a component, a sentence, a next step.
 *
 *  Not a stack trace. A traceback tells a user nothing they can act on and
 *  tells anyone they forward it to more about the machine than they meant to
 *  share. What helps is a stable code they can search and quote, and one
 *  concrete suggestion.
 *
 *  The copyable diagnostic is assembled from these same four fields. It never
 *  includes a path, an environment variable or a response body. */
import { useState } from 'react';

export type ErrorCode =
  | 'PAPER_EVENT_CHAIN_INVALID'
  | 'PAPER_SNAPSHOT_INVALID'
  | 'PAPER_SNAPSHOT_OVERDUE'
  | 'PAPER_MODEL_HASH_MISMATCH'
  | 'PAPER_RUNTIME_ABSENT'
  | 'PAPER_RUNTIME_EMPTY'
  | 'MARKET_NETWORK_UNAVAILABLE'
  | 'RESEARCH_HOLDOUT_EMBARGO_ACTIVE'
  | 'APP_PORT_IN_USE'
  | 'APP_PID_FOREIGN'
  | 'APP_NOT_RESPONDING';

interface Explanation {
  message: string;
  recovery: string;
  tone: 'warn' | 'off';
}

/** Sentences live here, not in the components that raise them, so the same
 *  condition reads the same way everywhere it surfaces. */
const EXPLANATIONS: Record<ErrorCode, Explanation> = {
  PAPER_EVENT_CHAIN_INVALID: {
    message: 'The shadow event log no longer verifies against its own hashes.',
    recovery: 'Run ./scripts/hyprl.sh doctor and export the runtime before doing anything else. Do not delete the log.',
    tone: 'off',
  },
  PAPER_SNAPSHOT_INVALID: {
    message: 'A state snapshot does not match the events it claims to summarise.',
    recovery: 'Run ./scripts/hyprl.sh doctor. Restart replays from the log, so the session is still usable.',
    tone: 'off',
  },
  PAPER_SNAPSHOT_OVERDUE: {
    message: 'Snapshots have fallen behind the event log.',
    recovery: 'Nothing is lost — a restart replays further than usual. Report it if it persists.',
    tone: 'warn',
  },
  PAPER_MODEL_HASH_MISMATCH: {
    message: 'A frozen shadow model artifact does not match its recorded hash.',
    recovery: 'The artifact has changed on disk. Restore it from the commit that froze it.',
    tone: 'off',
  },
  PAPER_RUNTIME_ABSENT: {
    message: 'No shadow session has been recorded on this machine yet.',
    recovery: 'Start one with ./scripts/hyprl.sh paper start.',
    tone: 'warn',
  },
  PAPER_RUNTIME_EMPTY: {
    message: 'The runtime database exists but holds no session.',
    recovery: 'Start a session with ./scripts/hyprl.sh paper start.',
    tone: 'warn',
  },
  MARKET_NETWORK_UNAVAILABLE: {
    message: 'The public market endpoint could not be reached.',
    recovery: 'Recorded data is unaffected. Ingestion resumes when the network returns.',
    tone: 'warn',
  },
  RESEARCH_HOLDOUT_EMBARGO_ACTIVE: {
    message: 'The reserved confirmatory holdout window is in force.',
    recovery: 'This is the guard working as designed. Shadow trading resumes after the window closes.',
    tone: 'warn',
  },
  APP_PORT_IN_USE: {
    message: 'The application port is held by another process.',
    recovery: 'Stop whatever holds it, or start with --port on a different one.',
    tone: 'off',
  },
  APP_PID_FOREIGN: {
    message: 'The recorded process id now belongs to something else.',
    recovery: 'HyprL refused to signal it. The stale record has been cleared; start again.',
    tone: 'off',
  },
  APP_NOT_RESPONDING: {
    message: 'The application started but did not answer in time.',
    recovery: 'Check ./scripts/hyprl.sh logs, then restart.',
    tone: 'warn',
  },
};

export function explain(code: string | null | undefined): Explanation | null {
  if (!code) return null;
  return EXPLANATIONS[code as ErrorCode] ?? null;
}

export function ProductError({
  code,
  component,
  observedAt,
}: {
  code: string;
  component: string;
  observedAt?: string | null;
}) {
  const [copied, setCopied] = useState(false);
  const explanation = explain(code);

  const diagnostic = [
    `code: ${code}`,
    `component: ${component}`,
    observedAt ? `observed: ${observedAt}` : null,
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <div className="state" role="alert" data-tone={explanation?.tone ?? 'warn'}>
      <strong className={explanation?.tone === 'off' ? 'negative' : undefined}>{code}</strong>
      <span>{explanation?.message ?? 'An operational condition was reported.'}</span>
      {explanation && <span className="muted">{explanation.recovery}</span>}
      <span className="muted">
        {component}
        {observedAt ? ` · ${observedAt}` : ''}
      </span>
      <button
        className="control"
        onClick={() => {
          void navigator.clipboard?.writeText(diagnostic).then(
            () => setCopied(true),
            () => setCopied(false),
          );
        }}
      >
        {copied ? 'Copied' : 'Copy diagnostic'}
      </button>
    </div>
  );
}
