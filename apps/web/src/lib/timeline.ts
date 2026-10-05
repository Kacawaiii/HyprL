/**
 * Geometry of the cockpit timeline: prices, predictions, decisions, executions and events on one
 * shared time axis. Pure, so the alignment can be tested: every coordinate lies inside the box, and
 * two layers at the same instant share an x.
 */
import type { Candle, PaperReplayFill } from '../api/types';
import type { Period, Projection } from './cockpit';

export interface TimelineDecision {
  timestamp: string;
  direction: string;
  strength: string;
  prediction: string;
}

export interface TimelineEvent {
  at: string;
  label: string;
  /** What `at` is: a declared release is not an observation, and neither is an availability. */
  kind: string;
}

export interface TimelineInput {
  period: Period;
  candles: readonly Candle[];
  decisions: readonly TimelineDecision[];
  projections: readonly Projection[];
  fills: readonly PaperReplayFill[];
  events: readonly TimelineEvent[];
}

export interface Box { width: number; height: number }

export interface Point { x: number; y: number }

export interface TimelineLayout {
  box: { left: number; right: number; top: number; bottom: number };
  price: Point[];
  decisions: Array<Point & { item: TimelineDecision; projection: Projection | null }>;
  projections: Array<{ from: Point; to: Point; projection: Projection }>;
  fills: Array<Point & { item: PaperReplayFill }>;
  events: Array<{ x: number; item: TimelineEvent }>;
  low: number;
  high: number;
}

export function layoutTimeline(input: TimelineInput, size: Box): TimelineLayout {
  const left = 52;
  const right = Math.max(left + 1, size.width - 12);
  const top = 12;
  const bottom = Math.max(top + 1, size.height - 22);
  const t0 = Date.parse(input.period.from);
  // The axis reaches past the period only as far as a projection's endpoint, so none is clipped.
  const t1 = Math.max(
    Date.parse(input.period.to),
    ...input.projections.map((item) => Date.parse(item.projectedAt)),
  );
  const span = Math.max(1, t1 - t0);
  const x = (iso: string) => left + ((Date.parse(iso) - t0) / span) * (right - left);

  const closes = input.candles.map((candle) => Number(candle.close)).filter(Number.isFinite);
  const projected = input.projections.map((item) => item.projectedPrice).filter(Number.isFinite);
  const values = [...closes, ...projected];
  let low = values.length ? Math.min(...values) : 0;
  let high = values.length ? Math.max(...values) : 1;
  if (low === high) { low -= 1; high += 1; }
  const pad = (high - low) * 0.06;
  low -= pad; high += pad;
  const y = (value: number) => bottom - ((value - low) / (high - low)) * (bottom - top);

  const byOpen = new Map(input.candles.map((candle) => [Date.parse(candle.bar_open_at), candle]));
  const projectionAt = new Map(input.projections.map((item) => [item.decisionAt, item]));
  return {
    box: { left, right, top, bottom },
    low, high,
    price: input.candles.map((candle) => ({
      // A candle's close is known at the end of its bar, so that is where the line puts it.
      x: x(new Date(Date.parse(candle.bar_open_at) + 3_600_000).toISOString()),
      y: y(Number(candle.close)),
    })),
    decisions: input.decisions.flatMap((item) => {
      const bar = byOpen.get(Date.parse(item.timestamp));
      if (!bar) return [];
      return [{ item, x: x(new Date(Date.parse(item.timestamp) + 3_600_000).toISOString()),
        y: y(Number(bar.close)), projection: projectionAt.get(item.timestamp) ?? null }];
    }),
    projections: input.projections.map((projection) => ({
      projection,
      from: { x: x(projection.referenceAt), y: y(Number(projection.referencePrice)) },
      to: { x: x(projection.projectedAt), y: y(projection.projectedPrice) },
    })),
    fills: input.fills.flatMap((item) => {
      const bar = byOpen.get(Date.parse(item.timestamp));
      return bar ? [{ item, x: x(item.timestamp), y: y(Number(bar.open)) }] : [];
    }),
    events: input.events
      .filter((item) => Date.parse(item.at) >= t0 && Date.parse(item.at) <= t1)
      .map((item) => ({ item, x: x(item.at) })),
  };
}

/** Everything the keyboard can step through, in time order. */
export interface TimelineStop { at: string; text: string }

export function timelineStops(layout: TimelineLayout): TimelineStop[] {
  const stops: TimelineStop[] = [
    ...layout.decisions.map(({ item, projection }) => ({
      at: item.timestamp,
      text: `Decision ${item.direction}, strength ${item.strength}, prediction ${item.prediction}` +
        (projection ? `, projects ${projection.projectedPrice.toFixed(2)} from reference ${projection.referencePrice}` : ', no reference price'),
    })),
    ...layout.fills.map(({ item }) => ({
      at: item.timestamp,
      text: `Execution ${item.side}, quantity ${item.quantity_delta}, fee ${item.fee}`,
    })),
    ...layout.events.map(({ item }) => ({ at: item.at, text: `Event (${item.kind}) ${item.label}` })),
  ];
  return stops.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
}
