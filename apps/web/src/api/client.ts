/**
 * The single door to the backend.
 *
 * Components never call `fetch` themselves. One client means one place for
 * cancellation, timeouts, error shape and instrumentation -- and it makes the
 * "no business logic in the browser" rule easy to hold, because every value
 * the UI shows can be traced back to a response field.
 */

import type {
  BacktestEquity, BacktestFillPage, BacktestSummary, BacktestsIndex,
  PaperEquity, PaperEventsPage, PaperProductState, PaperStatus,
  BenchmarkDetail, BenchmarkSummary, CandlePage, ChartSeries, Health,
  MarketsIndex, Overview, RiskView, SignalsView,
} from './types';

const DEFAULT_TIMEOUT_MS = 15_000;

/** Mirrors the server ceilings so the UI cannot compose an illegal request. */
export const MAX_PAGE_SIZE = 1000;
export const MAX_CHART_POINTS = 2000;
export const MAX_EQUITY_POINTS = 2000;
export const MAX_FILL_PAGE = 1000;

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export interface RequestTiming {
  path: string;
  ms: number;
  status: number;
  bytes: number;
}

const listeners = new Set<(timing: RequestTiming) => void>();

/** Dev-only instrumentation. Local, in-memory, no telemetry leaves the machine. */
export function onRequest(listener: (timing: RequestTiming) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function query(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      search.set(key, String(value));
    }
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', () => controller.abort(), { once: true });
  }
  const started = performance.now();
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    const text = await response.text();
    if (listeners.size > 0) {
      const timing: RequestTiming = {
        path,
        ms: Math.round(performance.now() - started),
        status: response.status,
        bytes: text.length,
      };
      listeners.forEach((listener) => listener(timing));
    }
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const parsed = JSON.parse(text) as { error?: string };
        if (parsed.error) message = parsed.error;
      } catch {
        /* a non-JSON error body is still an error; keep the generic message */
      }
      throw new ApiError(message, response.status);
    }
    return JSON.parse(text) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if ((error as Error)?.name === 'AbortError') {
      throw new ApiError('Request cancelled or timed out', 0);
    }
    throw new ApiError((error as Error)?.message ?? 'Network error', 0);
  } finally {
    clearTimeout(timeout);
  }
}

export interface CandleQuery {
  start?: string;
  end?: string;
  limit?: number;
  cursor?: string;
}

export const apiClient = {
  getHealth: (signal?: AbortSignal) => request<Health>('/api/v1/health', signal),
  getSystem: (signal?: AbortSignal) =>
    request<import('./types').SystemInfo>('/api/v1/system', signal),
  getOverview: (signal?: AbortSignal) => request<Overview>('/api/v1/overview', signal),
  getMarkets: (signal?: AbortSignal) => request<MarketsIndex>('/api/v1/markets', signal),
  getCandles: (product: string, options: CandleQuery = {}, signal?: AbortSignal) =>
    request<CandlePage>(
      `/api/v1/markets/${encodeURIComponent(product)}${query({ ...options })}`,
      signal,
    ),
  getChart: (
    product: string,
    options: { start?: string; end?: string; maxPoints?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<ChartSeries>(
      `/api/v1/markets/${encodeURIComponent(product)}/chart${query({
        start: options.start,
        end: options.end,
        max_points: options.maxPoints,
      })}`,
      signal,
    ),
  getSignals: (limit?: number, signal?: AbortSignal) =>
    request<SignalsView>(`/api/v1/signals${query({ limit })}`, signal),
  getRiskTargets: (limit?: number, signal?: AbortSignal) =>
    request<RiskView>(`/api/v1/risk/targets${query({ limit })}`, signal),
  getBacktests: (signal?: AbortSignal) =>
    request<BacktestsIndex>('/api/v1/backtests', signal),
  getBacktestDetail: (version: string, product: string, signal?: AbortSignal) =>
    request<BacktestSummary & { equity_points: number; expired_targets: number }>(
      `/api/v1/backtests/${encodeURIComponent(version)}/${encodeURIComponent(product)}`,
      signal,
    ),
  getBacktestEquity: (
    version: string, product: string, maxPoints?: number, signal?: AbortSignal,
  ) =>
    request<BacktestEquity>(
      `/api/v1/backtests/${encodeURIComponent(version)}/${encodeURIComponent(product)}/equity${query({ max_points: maxPoints })}`,
      signal,
    ),
  getBacktestFills: (
    version: string, product: string,
    options: { limit?: number; cursor?: string } = {}, signal?: AbortSignal,
  ) =>
    request<BacktestFillPage>(
      `/api/v1/backtests/${encodeURIComponent(version)}/${encodeURIComponent(product)}/fills${query({ ...options })}`,
      signal,
    ),
  getPaperStatus: (signal?: AbortSignal) =>
    request<PaperStatus>('/api/v1/paper/status', signal),
  getPaperProducts: (signal?: AbortSignal) =>
    request<{ products: PaperProductState[] }>('/api/v1/paper/products', signal),
  getPaperEvents: (product?: string, limit?: number, signal?: AbortSignal) =>
    request<PaperEventsPage>(
      product
        ? `/api/v1/paper/${encodeURIComponent(product)}/events${query({ limit })}`
        : `/api/v1/paper/events${query({ limit })}`,
      signal,
    ),
  getPaperEquity: (product: string, maxPoints?: number, signal?: AbortSignal) =>
    request<PaperEquity>(
      `/api/v1/paper/${encodeURIComponent(product)}/equity${query({ max_points: maxPoints })}`,
      signal,
    ),
  getBenchmarks: (signal?: AbortSignal) =>
    request<{ benchmarks: BenchmarkSummary[] }>('/api/v1/research/benchmarks', signal),
  getBenchmarkDetail: (version: string, product: string, signal?: AbortSignal) =>
    request<BenchmarkDetail>(
      `/api/v1/research/benchmarks/${encodeURIComponent(version)}/${encodeURIComponent(product)}`,
      signal,
    ),
};

export type ApiClient = typeof apiClient;
