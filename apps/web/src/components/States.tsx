/** Loading, error and empty states. Every page uses these, so no view can
 *  render a blank screen or throw a component tree away on a failed request. */
import type { ReactNode } from 'react';

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <div className="skeleton" style={{ width: 180, height: 10 }} />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="state" role="alert">
      <strong className="negative">Could not load</strong>
      <span>{error.message}</span>
      {onRetry && (
        <button className="control" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: ReactNode }) {
  return (
    <div className="state">
      <strong>{title}</strong>
      {detail && <span>{detail}</span>}
    </div>
  );
}

export function Badge({ tone, children }: { tone: 'ok' | 'off' | 'warn'; children: ReactNode }) {
  return <span className="badge" data-tone={tone}>{children}</span>;
}

/** A capability the product either has or does not. Never "coming soon". */
export function CapabilityBadge({ enabled }: { enabled: boolean }) {
  return <Badge tone={enabled ? 'ok' : 'off'}>{enabled ? 'AVAILABLE' : 'NOT AVAILABLE'}</Badge>;
}

export function Hash({ value, chars = 12 }: { value: string | null | undefined; chars?: number }) {
  if (!value) return <span className="hash">—</span>;
  return (
    <span className="hash" title={value}>
      {value.slice(0, chars)}…
    </span>
  );
}
