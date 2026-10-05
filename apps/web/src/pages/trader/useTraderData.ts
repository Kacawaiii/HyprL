import { useEffect, useMemo, useState } from 'react';
import { apiClient } from '../../api/client';
import type { Chained, LabelPayload, PredictionPayload } from '../../api/traderTypes';
import { joinLedger } from '../../lib/trader';
import { useQuery } from '../../state/useQuery';

const PAGE = 200;
const MAX_PAGES = 10;

interface Paged<T> { records: T[]; complete: boolean; loading: boolean; error: Error | undefined }

/** Follows `next_after` up to a fixed ceiling so a long ledger never becomes an unbounded fetch. */
function usePagedAll<T>(name: string, fetchPage: (after: number, signal: AbortSignal) => Promise<{ records: T[]; next_after: number | null }>): Paged<T> {
  const [state, setState] = useState<Paged<T>>({ records: [], complete: false, loading: true, error: undefined });
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      const records: T[] = [];
      let after = 0;
      try {
        for (let page = 0; page < MAX_PAGES; page += 1) {
          const result = await fetchPage(after, controller.signal);
          records.push(...result.records);
          if (result.next_after === null) {
            if (!cancelled) setState({ records, complete: true, loading: false, error: undefined });
            return;
          }
          after = result.next_after;
        }
        if (!cancelled) setState({ records, complete: false, loading: false, error: undefined });
      } catch (error) {
        if (!cancelled) setState({ records, complete: false, loading: false, error: error as Error });
      }
    })();
    return () => { cancelled = true; controller.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);
  return state;
}

/** Everything the page reads. Each endpoint fails on its own: one error never blanks the others. */
export function useTraderData(date: string | undefined) {
  const runs = useQuery('trader:runs', (signal) => apiClient.getTraderRuns(signal), { staleMs: 10_000 });
  const synthetic = runs.data?.records.some((r) => r.synthetic) ?? false;
  const today = useQuery(`trader:today:${date ?? ''}`, (signal) => apiClient.getTraderToday(date, signal), { staleMs: 10_000 });
  const context = useQuery(`trader:context:${date ?? ''}`, (signal) => apiClient.getTraderContext(date, signal), { staleMs: 10_000 });
  const scorecard = useQuery(runs.data ? `trader:score:${synthetic}` : null, (signal) => apiClient.getTraderScorecard(synthetic, signal), { staleMs: 10_000 });
  const series = useQuery('trader:series', (signal) => apiClient.getTraderSeries(signal), { staleMs: 10_000 });
  const alerts = useQuery('trader:alerts', (signal) => apiClient.getTraderAlerts(signal), { staleMs: 10_000 });
  const health = useQuery('trader:health', (signal) => apiClient.getTraderHealth(signal), { staleMs: 10_000 });
  const ledger = usePagedAll<Chained<PredictionPayload>>('ledger', (after, signal) => apiClient.getTraderLedger(after, PAGE, signal));
  const labels = usePagedAll<Chained<LabelPayload>>('labels', (after, signal) => apiClient.getTraderLabels(after, PAGE, signal));
  const rows = useMemo(() => joinLedger(ledger.records, labels.records), [ledger.records, labels.records]);
  return { runs, synthetic, today, context, scorecard, series, alerts, health, ledger, labels, rows };
}

export type TraderData = ReturnType<typeof useTraderData>;
