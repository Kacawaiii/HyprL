/**
 * A small server-state cache.
 *
 * Deliberately not a global store holding every candle ever fetched: server
 * data and UI state have different lifetimes, and conflating them is how a
 * cockpit ends up re-rendering the world because a sidebar toggled.
 *
 * What it does provide is the part that actually matters: deduplication by
 * key, a cache so navigation does not refetch, cancellation on unmount, and
 * explicit invalidation.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

type Entry = { data: unknown; at: number };

const cache = new Map<string, Entry>();
const inflight = new Map<string, Promise<unknown>>();

export function invalidate(prefix?: string): void {
  if (!prefix) {
    cache.clear();
    return;
  }
  for (const key of [...cache.keys()]) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
}

export type QueryStatus = 'loading' | 'success' | 'error';

export interface QueryResult<T> {
  data: T | undefined;
  error: Error | undefined;
  status: QueryStatus;
  refetch: () => void;
}

export function useQuery<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
  options: { staleMs?: number } = {},
): QueryResult<T> {
  const staleMs = options.staleMs ?? 30_000;
  const [, force] = useState(0);
  const [state, setState] = useState<{ data?: T; error?: Error; status: QueryStatus }>(
    () => {
      if (key) {
        const hit = cache.get(key);
        if (hit) return { data: hit.data as T, status: 'success' };
      }
      return { status: 'loading' };
    },
  );
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(
    (controller: AbortController, ignoreCache: boolean) => {
      if (!key) return;
      const hit = cache.get(key);
      if (!ignoreCache && hit && Date.now() - hit.at < staleMs) {
        setState({ data: hit.data as T, status: 'success' });
        return;
      }
      setState((previous) =>
        previous.data === undefined ? { status: 'loading' } : { ...previous },
      );
      let promise = ignoreCache ? undefined : inflight.get(key);
      if (!promise) {
        promise = fetcherRef.current(controller.signal).then((data) => {
          cache.set(key, { data, at: Date.now() });
          inflight.delete(key);
          return data;
        }).catch((error) => {
          inflight.delete(key);
          throw error;
        });
        inflight.set(key, promise);
      }
      promise
        .then((data) => {
          if (!controller.signal.aborted) {
            setState({ data: data as T, status: 'success' });
          }
        })
        .catch((error: Error) => {
          if (!controller.signal.aborted) setState({ error, status: 'error' });
        });
    },
    [key, staleMs],
  );

  useEffect(() => {
    if (!key) return;
    const controller = new AbortController();
    run(controller, false);
    return () => controller.abort();
  }, [key, run]);

  const refetch = useCallback(() => {
    if (!key) return;
    cache.delete(key);
    const controller = new AbortController();
    run(controller, true);
    force((value) => value + 1);
  }, [key, run]);

  return { data: state.data, error: state.error, status: state.status, refetch };
}
