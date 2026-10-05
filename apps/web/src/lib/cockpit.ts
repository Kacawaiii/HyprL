/**
 * Cockpit state and the few pure derivations the cockpit needs.
 *
 * The selection (mode, product, period, model) lives in the URL, so a mode switch, a reload and a
 * shared link all keep it. Everything here is pure: the pages show backend values verbatim, and the
 * only arithmetic is the one the model's own contract defines (the 4-bar projection), written once
 * with its formula so it can be tested and displayed.
 */
import type { Candle, PaperReplayFill, SignalsView } from '../api/types';

export type Mode = 'beginner' | 'expert';

export interface CockpitSelection {
  mode: Mode;
  product: string | null;
  model: string | null;
  /** Inclusive calendar days (YYYY-MM-DD); null means "the default period for this product". */
  start: string | null;
  end: string | null;
}

const DAY = /^\d{4}-\d{2}-\d{2}$/;
const KEYS = ['mode', 'product', 'model', 'start', 'end'] as const;

export function parseSelection(params: URLSearchParams): CockpitSelection {
  const day = (key: string) => {
    const value = params.get(key);
    return value && DAY.test(value) && !Number.isNaN(Date.parse(`${value}T00:00:00Z`)) ? value : null;
  };
  const text = (key: string) => params.get(key)?.trim() || null;
  return {
    mode: params.get('mode') === 'expert' ? 'expert' : 'beginner',
    product: text('product'),
    model: text('model'),
    start: day('start'),
    end: day('end'),
  };
}

/** Apply a patch to the selection, leaving every unrelated parameter (page cursors, etc.) alone. */
export function writeSelection(params: URLSearchParams, patch: Partial<CockpitSelection>): URLSearchParams {
  const next = new URLSearchParams(params);
  for (const key of KEYS) {
    if (!(key in patch)) continue;
    const value = patch[key];
    if (value === null || value === undefined || (key === 'mode' && value === 'beginner')) next.delete(key);
    else next.set(key, value);
  }
  return next;
}

/** The selection keys only, used to carry the context to another page of the cockpit. */
export function carrySelection(params: URLSearchParams): string {
  const carried = new URLSearchParams();
  for (const key of KEYS) {
    const value = params.get(key);
    if (value) carried.set(key, value);
  }
  const text = carried.toString();
  return text ? `?${text}` : '';
}

const HOUR_MS = 3_600_000;

export interface Period {
  /** Inclusive lower bound, ISO instant. */
  from: string;
  /** Exclusive upper bound, ISO instant. */
  to: string;
  source: 'selected' | 'default';
}

/** Default period: the last `days` days of the window the data covers; `end` is a day, inclusive. */
export function resolvePeriod(
  selection: Pick<CockpitSelection, 'start' | 'end'>,
  coverage: { first: string | null; last: string | null },
  days = 7,
): Period | null {
  if (selection.start && selection.end) {
    const from = `${selection.start}T00:00:00.000Z`;
    const to = new Date(Date.parse(`${selection.end}T00:00:00Z`) + 24 * HOUR_MS).toISOString();
    return Date.parse(to) > Date.parse(from) ? { from, to, source: 'selected' } : null;
  }
  if (!coverage.last) return null;
  const to = new Date(Date.parse(coverage.last) + HOUR_MS).toISOString();
  const from = new Date(Date.parse(to) - days * 24 * HOUR_MS).toISOString();
  const floor = coverage.first && Date.parse(coverage.first) > Date.parse(from) ? coverage.first : from;
  return { from: floor, to, source: 'default' };
}

export function inPeriod(timestamp: string, period: Period): boolean {
  const at = Date.parse(timestamp);
  return at >= Date.parse(period.from) && at < Date.parse(period.to);
}

/** Longest period the chart asks for: the candle endpoint's page ceiling at the hourly timeframe. */
export const MAX_PERIOD_BARS = 1000;

export function periodBars(period: Period): number {
  return Math.ceil((Date.parse(period.to) - Date.parse(period.from)) / HOUR_MS);
}

/**
 * The model's 4-bar projection (SIGNAL_SPEC_V1.prediction_horizon): `prediction` is the forward
 * return close[T+h]/close[T] - 1, so the projected price is reference * (1 + prediction), with
 * reference = the close of the decision bar T. When that bar is not in the loaded candles there is
 * no reference price and no projection is invented.
 */
export interface Projection {
  decisionAt: string;
  referenceAt: string;
  referencePrice: string;
  projectedAt: string;
  projectedPrice: number;
  prediction: string;
  horizonBars: number;
  /** Exclusive end of the decision bar: when its close could first be known. */
  formula: string;
}

export function project(
  decision: { timestamp: string; prediction: string },
  candles: ReadonlyMap<string, Candle>,
  horizonBars: number,
): Projection | null {
  const bar = candles.get(Date.parse(decision.timestamp).toString());
  const prediction = Number(decision.prediction);
  if (!bar || !Number.isFinite(prediction)) return null;
  const reference = Number(bar.close);
  if (!Number.isFinite(reference)) return null;
  const referenceAt = new Date(Date.parse(bar.bar_open_at) + HOUR_MS).toISOString();
  return {
    decisionAt: decision.timestamp,
    referenceAt,
    referencePrice: bar.close,
    projectedAt: new Date(Date.parse(referenceAt) + horizonBars * HOUR_MS).toISOString(),
    projectedPrice: reference * (1 + prediction),
    prediction: decision.prediction,
    horizonBars,
    formula: `close(T) × (1 + prediction), T + ${horizonBars} bars`,
  };
}

export function indexCandles(candles: readonly Candle[]): Map<string, Candle> {
  return new Map(candles.map((candle) => [Date.parse(candle.bar_open_at).toString(), candle]));
}

/** What the model's declared outputs are, from the signal contract (nothing more is claimed). */
export interface ModelCapabilities {
  id: string;
  horizonBars: number;
  outputs: string[];
  notProvided: string[];
}

export function declaredCapabilities(signal: { protocol: string; prediction_horizon: number }): ModelCapabilities {
  return {
    id: signal.protocol,
    horizonBars: signal.prediction_horizon,
    outputs: ['point forward return (decimal string)', 'direction (LONG / FLAT / SHORT)', 'strength (0 to 1)'],
    notProvided: ['class probabilities', 'quantiles or intervals', 'scenarios', 'target price (derived here, see formula)'],
  };
}

/* --- scores: four separate quantities, each with definition, period and sample size --- */

export interface Score {
  key: 'data' | 'signal' | 'model' | 'risk';
  title: string;
  value: string | null;
  detail: string | null;
  definition: string;
  period: string;
  sample: number | null;
  sampleUnit: string;
}

export function dataQualityScore(
  product: { rows: number; missing_openings: number; first_open: string; last_open: string } | undefined,
): Score {
  const expected = product ? product.rows + product.missing_openings : 0;
  return {
    key: 'data', title: 'Data quality',
    value: product && expected > 0 ? `${((product.rows / expected) * 100).toFixed(2)} %` : null,
    detail: product ? `${product.missing_openings} declared gaps` : null,
    definition: 'Hourly bars present divided by hourly bars expected over the covered window; declared gaps count as missing.',
    period: product ? `${product.first_open.slice(0, 10)} → ${product.last_open.slice(0, 10)}` : 'not available',
    sample: product ? expected : null, sampleUnit: 'expected bars',
  };
}

export function signalScore(
  decisions: SignalsView['decisions'], period: Period | null,
): Score {
  const inside = period ? decisions.filter((row) => inPeriod(row.timestamp, period)) : [];
  const latest = inside[0];
  return {
    key: 'signal', title: 'Signal strength',
    value: latest ? latest.strength : null,
    detail: latest ? `${latest.direction} at ${latest.timestamp.slice(0, 16).replace('T', ' ')}` : null,
    definition: 'Descriptive intensity of the latest decision in the period, 0 to 1, as the signal contract defines it. Not a probability and not a position size.',
    period: period ? `${period.from.slice(0, 10)} → ${period.to.slice(0, 10)}` : 'not available',
    sample: inside.length, sampleUnit: 'decisions',
  };
}

export function modelScore(quality: {
  rank_ic: string | null; observations: number; label_horizon: number;
} | undefined, window: { start: string; end: string } | undefined): Score {
  return {
    key: 'model', title: 'Model performance',
    value: quality?.rank_ic ?? null,
    detail: quality ? `rank IC at a ${quality.label_horizon}-bar label horizon` : null,
    definition: 'Rank information coefficient between predictions and realised forward returns, on the frozen paper replay. Exploratory; not a confirmed edge.',
    period: window ? `${window.start.slice(0, 10)} → ${window.end.slice(0, 10)}` : 'not available',
    sample: quality?.observations ?? null, sampleUnit: 'scored labels',
  };
}

export function riskScore(
  targets: Array<{ timestamp: string; side: string; target_exposure: string }>,
  period: Period | null, cap: string | null,
): Score {
  const inside = period ? targets.filter((row) => inPeriod(row.timestamp, period)) : [];
  const latest = inside[0];
  return {
    key: 'risk', title: 'Risk and exposure',
    value: latest ? latest.target_exposure : null,
    detail: latest ? `${latest.side}${cap ? ` · cap ${cap}` : ''}` : null,
    definition: 'Latest target exposure in the period as a fraction of NAV, after the risk contract caps it. A target, not an order.',
    period: period ? `${period.from.slice(0, 10)} → ${period.to.slice(0, 10)}` : 'not available',
    sample: inside.length, sampleUnit: 'targets',
  };
}

/** Executions of the frozen replay that fall inside the period. */
export function fillsInPeriod(fills: readonly PaperReplayFill[], period: Period | null): PaperReplayFill[] {
  return period ? fills.filter((fill) => inPeriod(fill.timestamp, period)) : [];
}
