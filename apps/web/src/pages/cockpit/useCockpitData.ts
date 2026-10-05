/** Every request the cockpit makes, for one selection. All of it is data the API already serves. */
import { useEffect, useMemo, useState } from 'react';
import { apiClient } from '../../api/client';
import type { Candle, FomcItemSummary, PaperReplayFill, PaperReplayFills } from '../../api/types';
import {
  MAX_PERIOD_BARS, declaredCapabilities, indexCandles, periodBars, project, resolvePeriod,
} from '../../lib/cockpit';
import type { CockpitSelection } from '../../lib/cockpit';
import { useQuery } from '../../state/useQuery';
import { useOlderRows } from '../../components/RunProvenance';
import type { TimelineEvent } from '../../lib/timeline';

const MAX_FILL_PAGES = 10;

/** Replay fills are served oldest first, so reach the period by following the cursor (bounded). */
function useReplayFills(product: string | null, until: string | null) {
  const [state, setState] = useState<{ fills: PaperReplayFill[]; total: number | null; complete: boolean;
    error?: Error; loading: boolean }>({ fills: [], total: null, complete: false, loading: false });
  useEffect(() => {
    if (!product || !until) return;
    const controller = new AbortController();
    setState({ fills: [], total: null, complete: false, loading: true });
    (async () => {
      let cursor: string | undefined;
      const fills: PaperReplayFill[] = [];
      let page: PaperReplayFills | undefined;
      for (let index = 0; index < MAX_FILL_PAGES; index += 1) {
        page = await apiClient.getPaperReplayFills(product, cursor, controller.signal);
        fills.push(...page.fills);
        const last = page.fills[page.fills.length - 1];
        if (!page.page.has_more || !page.page.next_cursor || (last && Date.parse(last.timestamp) >= Date.parse(until))) break;
        cursor = page.page.next_cursor;
      }
      if (controller.signal.aborted) return;
      setState({ fills, total: page?.page.total ?? null, complete: !page?.page.has_more, loading: false });
    })().catch((error: Error) => {
      if (!controller.signal.aborted) setState({ fills: [], total: null, complete: false, error, loading: false });
    });
    return () => controller.abort();
  }, [product, until]);
  return state;
}

export function useCockpitData(selection: CockpitSelection) {
  const overview = useQuery('overview', (signal) => apiClient.getOverview(signal));
  const system = useQuery('system', (signal) => apiClient.getSystem(signal));
  const product = selection.product ?? overview.data?.products[0]?.product ?? null;
  const summary = overview.data?.products.find((item) => item.product === product);

  const signals = useQuery(product ? `signals:${product}` : null, (signal) =>
    apiClient.getSignals(200, signal, { product: product ?? undefined }));
  const risk = useQuery(product ? `risk:${product}` : null, (signal) =>
    apiClient.getRiskTargets(200, signal, { product: product ?? undefined }));
  const olderDecisions = useOlderRows(product ?? '', signals.data,
    (cursor) => apiClient.getSignals(200, undefined, { product: product ?? undefined, cursor }),
    (view) => view.decisions);
  const olderTargets = useOlderRows(product ?? '', risk.data,
    (cursor) => apiClient.getRiskTargets(200, undefined, { product: product ?? undefined, cursor }),
    (view) => view.targets);

  // The default period is the latest week the signal run covers, else the latest week of prices.
  const coverage = signals.data?.available && signals.data.window
    ? signals.data.window : { first: summary?.first_open ?? null, last: summary?.last_open ?? null };
  const { start, end } = selection;
  const first = coverage.first;
  const last = coverage.last;
  const period = useMemo(() => resolvePeriod({ start, end }, { first, last }), [start, end, first, last]);
  const tooLong = period ? periodBars(period) > MAX_PERIOD_BARS : false;

  const candles = useQuery(product && period && !tooLong ? `candles:${product}:${period.from}:${period.to}` : null,
    (signal) => apiClient.getCandles(product ?? '', {
      start: period?.from, end: new Date(Date.parse(period?.to ?? '') - 1000).toISOString(), limit: MAX_PERIOD_BARS,
    }, signal));
  const replay = useQuery('paper-replay', (signal) => apiClient.getPaperReplay(signal));
  const fills = useReplayFills(product, period && !tooLong ? period.to : null);
  const fomcStatus = useQuery('fomc-status', (signal) => apiClient.getFomcStatus(signal));
  const asOf = fomcStatus.data?.status === 'AVAILABLE' ? fomcStatus.data.suggested_as_of ?? null : null;
  const fomc = useQuery(asOf ? `fomc-snapshot:${asOf}` : null, (signal) =>
    apiClient.getFomcSnapshot(asOf ?? '', undefined, signal, { limit: 200 }));
  const paperStatus = useQuery('paper-status', (signal) => apiClient.getPaperStatus(signal));
  const benchmarks = useQuery('benchmarks', (signal) => apiClient.getBenchmarks(signal));

  const decisions = useMemo(
    () => [...(signals.data?.decisions ?? []), ...olderDecisions.older], [signals.data, olderDecisions.older]);
  const targets = useMemo(
    () => [...(risk.data?.targets ?? []), ...olderTargets.older], [risk.data, olderTargets.older]);
  const candleList: Candle[] = useMemo(() => candles.data?.candles ?? [], [candles.data]);
  const horizon = system.data?.signal_engine.prediction_horizon ?? 4;
  const model = system.data ? declaredCapabilities(system.data.signal_engine) : null;

  const projections = useMemo(() => {
    const index = indexCandles(candleList);
    return decisions.flatMap((row) => {
      const item = project(row, index, horizon);
      return item ? [item] : [];
    });
  }, [decisions, candleList, horizon]);

  // Items are shown at their DECLARED release time; observation and availability are not that.
  const events: Array<TimelineEvent & { item: FomcItemSummary }> = (fomc.data?.items ?? []).flatMap((item) =>
    item.declared_release_at
      ? [{ at: item.declared_release_at, label: item.title ?? item.sid.slice(0, 12), kind: 'declared release', item }] : []);

  const replayProduct = replay.data?.products.find((item) => item.product === product);
  const protection = targets[0]?.protection;

  return {
    overview, system, product, summary, signals, risk, olderDecisions, olderTargets, decisions, targets,
    period, tooLong, candles, candleList, replay, replayProduct, fills, fomcStatus, fomc, events,
    paperStatus, benchmarks, projections, horizon, model, protection, asOf,
  };
}

export type CockpitData = ReturnType<typeof useCockpitData>;
