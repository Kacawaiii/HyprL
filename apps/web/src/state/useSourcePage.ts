import { useEffect, useRef, useState } from 'react';
import type { FomcSnapshotHeader, SourcePagination } from '../api/types';

export type SourcePageOptions = { limit: number; cursor?: string };

/** Keep one displayed page and pin subsequent pages to the first read's H. */
export function useSourcePage<T extends { snapshot: FomcSnapshotHeader; pagination?: SourcePagination }>(
  fetcher: (signal: AbortSignal, page: SourcePageOptions, horizon?: number) => Promise<T>,
  horizon?: number,
) {
  const [cursors, setCursors] = useState<(string | undefined)[]>([undefined]);
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ data?: T; error?: Error; status: 'loading' | 'success' | 'error' }>({ status: 'loading' });
  const pinned = useRef(horizon);
  const load = useRef(fetcher);
  load.current = fetcher;
  const cursor = cursors[cursors.length - 1];
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });
    load.current(controller.signal, { limit: 200, cursor }, pinned.current).then((data) => {
      if (controller.signal.aborted) return;
      pinned.current = data.snapshot.H;
      setState({ status: 'success', data });
    }).catch((error: Error) => {
      if (!controller.signal.aborted) setState({ status: 'error', error });
    });
    return () => controller.abort();
  }, [cursor, retry]);
  return { ...state, refetch: () => setRetry((value) => value + 1),
    horizon: pinned.current,
    page: cursors.length,
    previous: cursors.length > 1 ? () => setCursors((value) => value.slice(0, -1)) : undefined,
    next: state.data?.pagination?.next_cursor ? () => {
      const next = state.data?.pagination?.next_cursor;
      if (next) setCursors((value) => value[value.length - 1] === next ? value : [...value, next]);
    } : undefined };
}
