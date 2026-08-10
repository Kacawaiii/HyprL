/** Dev-only request instrumentation. Local, in-memory, nothing leaves the machine.
 *  Toggled with the `?perf` search param so it never ships noise to a normal run. */
import { useEffect, useState } from 'react';
import { onRequest, type RequestTiming } from '../api/client';

export function PerfOverlay() {
  const [timings, setTimings] = useState<RequestTiming[]>([]);
  const enabled =
    import.meta.env.DEV && typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).has('perf');

  useEffect(() => {
    if (!enabled) return;
    return onRequest((timing) =>
      setTimings((previous) => [timing, ...previous].slice(0, 6)),
    );
  }, [enabled]);

  if (!enabled || timings.length === 0) return null;

  return (
    <div className="perf" aria-hidden="true">
      {timings.map((timing, index) => (
        <div key={index}>
          {timing.ms}ms · {(timing.bytes / 1024).toFixed(1)}kB · {timing.status} ·{' '}
          {timing.path.replace('/api/v1', '')}
        </div>
      ))}
    </div>
  );
}
