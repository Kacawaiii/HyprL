import { useState } from 'react';
import { apiClient } from '../api/client';
import type { Read } from './ReadForm';
import { useSourcePage } from '../state/useSourcePage';
import { ErrorState, LoadingState } from './States';
import { SourcePager } from './SourcePager';

function TimelinePage({ source, read }: { source: 'fomc' | 'edgar'; read: Read }) {
  const timeline = useSourcePage((signal, page, horizon) =>
    apiClient.getSourceTimeline(source, read.asOf, horizon, signal, page), read.horizon);
  if (timeline.status === 'loading') return <LoadingState label="Reading timeline" />;
  if (timeline.error) return <ErrorState error={timeline.error} onRetry={timeline.refetch} />;
  return <>
    <table className="data">
      <thead><tr><th>Commit</th><th>Activity</th><th>Record</th><th>Observed / checked at</th></tr></thead>
      <tbody>{timeline.data?.rows.map((row, index) => <tr key={index}>
        <td>{row.committed_seq}</td><td>{row.kind}</td><td>{String(row.body.record ?? '—')}</td>
        <td>{String(row.body.observed_at ?? row.body.check_at ?? '—')}</td>
      </tr>)}</tbody>
    </table>
    <SourcePager {...timeline} />
  </>;
}

export function SourceTimeline({ source, read }: { source: 'fomc' | 'edgar'; read: Read }) {
  const [open, setOpen] = useState(false);
  return <section className="card" aria-label="Source timeline">
    <h2 className="card-title">Timeline</h2>
    <button className="control" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
      {open ? 'Hide timeline' : 'Show timeline'}
    </button>
    {open && <TimelinePage source={source} read={read} />}
  </section>;
}
