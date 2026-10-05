import { describe, expect, it } from 'vitest';
import type { Candle } from '../api/types';
import {
  carrySelection, dataQualityScore, indexCandles, inPeriod, parseSelection, project, resolvePeriod,
  riskScore, signalScore, writeSelection,
} from '../lib/cockpit';
import { layoutTimeline, timelineStops } from '../lib/timeline';

const bar = (hour: number, close = 100 + hour): Candle => ({
  bar_open_at: new Date(Date.UTC(2026, 6, 31, hour)).toISOString(), open: String(close - 1),
  high: String(close + 1), low: String(close - 2), close: String(close), volume: '1',
});

describe('selection in the URL', () => {
  it('defaults to Beginner and ignores malformed values', () => {
    const selection = parseSelection(new URLSearchParams('mode=weird&start=yesterday&end=2026-02-31x'));
    expect(selection).toEqual({ mode: 'beginner', product: null, model: null, start: null, end: null });
  });

  it('keeps product, period and model when only the mode changes', () => {
    const before = new URLSearchParams('product=ETH-USD&model=m1&start=2026-07-01&end=2026-07-05&page=3');
    const after = writeSelection(before, { mode: 'expert' });
    expect(parseSelection(after)).toEqual({
      mode: 'expert', product: 'ETH-USD', model: 'm1', start: '2026-07-01', end: '2026-07-05',
    });
    expect(after.get('page')).toBe('3');
    expect(writeSelection(after, { mode: 'beginner' }).has('mode')).toBe(false);
  });

  it('carries only the selection to another page', () => {
    expect(carrySelection(new URLSearchParams('mode=expert&product=BTC-USD&cursor=abc')))
      .toBe('?mode=expert&product=BTC-USD');
    expect(carrySelection(new URLSearchParams('cursor=abc'))).toBe('');
  });
});

describe('periods', () => {
  it('defaults to the latest week of the covered window', () => {
    const period = resolvePeriod({ start: null, end: null }, { first: '2026-07-01T00:00:00+00:00', last: '2026-07-31T23:00:00+00:00' });
    expect(period).toEqual({ from: '2026-07-25T00:00:00.000Z', to: '2026-08-01T00:00:00.000Z', source: 'default' });
  });

  it('treats the selected end day as inclusive and rejects a reversed range', () => {
    const period = resolvePeriod({ start: '2026-07-01', end: '2026-07-02' }, { first: null, last: null });
    expect(period?.to).toBe('2026-07-03T00:00:00.000Z');
    expect(inPeriod('2026-07-02T23:00:00Z', period!)).toBe(true);
    expect(inPeriod('2026-07-03T00:00:00Z', period!)).toBe(false);
    expect(resolvePeriod({ start: '2026-07-05', end: '2026-07-01' }, { first: null, last: null })).toBeNull();
  });
});

describe('4-bar projection', () => {
  it('uses the decision bar close as the reference and the model formula for the target', () => {
    const projection = project({ timestamp: bar(3).bar_open_at, prediction: '0.01' }, indexCandles([bar(3, 200)]), 4);
    expect(projection?.referencePrice).toBe('200');
    expect(projection?.projectedPrice).toBeCloseTo(202);
    expect(projection?.referenceAt).toBe('2026-07-31T04:00:00.000Z');
    expect(projection?.projectedAt).toBe('2026-07-31T08:00:00.000Z');
  });

  it('invents nothing when the decision bar has no price', () => {
    expect(project({ timestamp: bar(9).bar_open_at, prediction: '0.01' }, indexCandles([bar(3)]), 4)).toBeNull();
    expect(project({ timestamp: bar(3).bar_open_at, prediction: 'nan' }, indexCandles([bar(3)]), 4)).toBeNull();
  });
});

describe('timeline alignment', () => {
  const period = { from: '2026-07-31T00:00:00.000Z', to: '2026-08-01T00:00:00.000Z', source: 'selected' as const };
  const candles = Array.from({ length: 24 }, (_, hour) => bar(hour));
  const decision = { timestamp: bar(10).bar_open_at, direction: 'LONG', strength: '0.5', prediction: '0.02' };
  const projection = project(decision, indexCandles(candles), 4)!;
  const fill = { timestamp: bar(11).bar_open_at, available_at: bar(12).bar_open_at, decided_at: bar(10).bar_open_at,
    side: 'BUY', quantity_delta: '1', fee: '0.1', slippage_cost: '0.1', equity_after: '1' };
  const layout = layoutTimeline({
    period, candles, decisions: [decision], projections: [projection], fills: [fill],
    events: [{ at: '2026-07-31T11:00:00.000Z', label: 'statement', kind: 'declared release' }],
  }, { width: 900, height: 300 });

  it('keeps every mark inside the box', () => {
    const { left, right, top, bottom } = layout.box;
    const points = [...layout.price, ...layout.decisions, ...layout.fills,
      ...layout.projections.flatMap((item) => [item.from, item.to])];
    for (const point of points) {
      expect(point.x).toBeGreaterThanOrEqual(left - 1e-9);
      expect(point.x).toBeLessThanOrEqual(right + 1e-9);
      expect(point.y).toBeGreaterThanOrEqual(top - 1e-9);
      expect(point.y).toBeLessThanOrEqual(bottom + 1e-9);
    }
  });

  it('puts layers at the same instant on the same x', () => {
    const eventAtFill = layoutTimeline({ period, candles, decisions: [], projections: [], fills: [fill],
      events: [{ at: fill.timestamp, label: 'x', kind: 'declared release' }] }, { width: 900, height: 300 });
    expect(eventAtFill.events[0]?.x).toBeCloseTo(eventAtFill.fills[0]!.x);
    // the decision sits at the end of its bar, where the projection starts
    expect(layout.decisions[0]?.x).toBeCloseTo(layout.projections[0]!.from.x);
  });

  it('steps through decisions, executions and events in time order', () => {
    const stops = timelineStops(layout);
    expect(stops.map((stop) => stop.at)).toEqual([...stops.map((stop) => stop.at)].sort());
    expect(stops).toHaveLength(3);
    expect(stops[0]?.text).toMatch(/Decision LONG.*reference 110/);
  });
});

describe('scores', () => {
  const period = { from: '2026-07-31T00:00:00.000Z', to: '2026-08-01T00:00:00.000Z', source: 'selected' as const };

  it('states a definition-ready sample for every score and never fills a missing one', () => {
    const quality = dataQualityScore({ rows: 90, missing_openings: 10, first_open: '2026-01-01T00:00', last_open: '2026-04-01T00:00' });
    expect(quality.value).toBe('90.00 %');
    expect(quality.sample).toBe(100);
    expect(dataQualityScore(undefined).value).toBeNull();
    const signal = signalScore([], period);
    expect(signal.value).toBeNull();
    expect(signal.sample).toBe(0);
    expect(riskScore([], null, '0.25').period).toBe('not available');
  });

  it('counts only rows inside the period', () => {
    const rows = [
      { timestamp: '2026-07-31T10:00:00Z', direction: 'LONG', strength: '0.4' },
      { timestamp: '2026-07-20T10:00:00Z', direction: 'SHORT', strength: '0.9' },
    ] as Parameters<typeof signalScore>[0];
    const score = signalScore(rows, period);
    expect(score.sample).toBe(1);
    expect(score.value).toBe('0.4');
  });
});
