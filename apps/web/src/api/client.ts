import type { SourceTimeline } from './types';
import type { PolicyDefinitions, PolicyReport } from './policyTypes';
import type {
  ComparisonReadiness, DatasetResult, ExperimentResult, HypothesisDetail, HypothesisRow, JobStatus,
  LabJobs, LabModels, LabMonitoringResult, LabSubmitted, LedgerRow, MonitoringView, ObservabilityHealth, PredictionView, ProposalCatalogue,
  ReferenceRow, ResearchPage, TrialRow,
} from './labTypes';
import type { SourcePageOptions } from '../state/useSourcePage';
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
  BenchmarkDetail, BenchmarkSummary, CalendarsIndex, CandlePage, ChartSeries,
  Health, HealthHistory, InstrumentDetail, InstrumentSessions,
  InstrumentsIndex, MarketsIndex,
  OpsHealth, OpsRecovery, OpsRuntime, OpsSettings, OpsStorage, Overview,
  PortfolioAttribution, PortfolioBacktestsIndex, PortfolioDetail,
  PaperLegacy, PaperPortfolioEquity, PaperPortfolioPending,
  PaperPortfolioStatus, PortfolioEquity, PortfolioFillPage, PortfolioStatus,
  ProvidersIndex, ResearchBarPage, ResearchCorpusStatus, RiskView, SignalsView,
  FomcItemDetail, FomcReplay, FomcSnapshotView, FomcSourceStatus,
  EdgarFilingDetail, EdgarReplay, EdgarSnapshotView, EdgarSourceStatus,
} from './types';

import type {
  TraderAlerts, TraderContext, TraderHealth, TraderLabels, TraderLedger, TraderRuns, TraderScorecard, TraderSeries, TraderToday,
} from './traderTypes';

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

/** Which persisted run to read, and where to resume (newest first). */
export interface RunQuery {
  product?: string;
  cursor?: string;
}

async function request<T>(
  path: string, signal?: AbortSignal, extra?: Record<string, string>, body?: Record<string, unknown>,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', () => controller.abort(), { once: true });
  }
  const started = performance.now();
  try {
    // The only writes are the Model Lab job controls; every other call is a GET.
    const response = await fetch(path, body === undefined ? {
      signal: controller.signal,
      headers: { Accept: 'application/json', ...extra },
    } : {
      method: 'POST',
      signal: controller.signal,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', ...extra },
      body: JSON.stringify(body),
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

function bearer(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export interface CandleQuery {
  start?: string;
  end?: string;
  limit?: number;
  cursor?: string;
}

export const apiClient = {
  getPolicyDefinitions: (signal?: AbortSignal) => request<PolicyDefinitions>('/api/v1/policies/definitions', signal),
  getPolicyReport: (signal?: AbortSignal) => request<PolicyReport>('/api/v1/policies/report', signal),
  /** The paper trader (read only, bounded). A dry-run runtime holds synthetic records; its scorecard needs `synthetic`. */
  getTraderToday: (date: string | undefined, signal?: AbortSignal) =>
    request<TraderToday>(`/api/v1/trader/today${query({ date })}`, signal),
  getTraderRuns: (signal?: AbortSignal) => request<TraderRuns>(`/api/v1/trader/runs${query({ limit: 100 })}`, signal),
  getTraderLedger: (after: number, limit: number, signal?: AbortSignal) =>
    request<TraderLedger>(`/api/v1/trader/ledger${query({ after, limit })}`, signal),
  getTraderLabels: (after: number, limit: number, signal?: AbortSignal) =>
    request<TraderLabels>(`/api/v1/trader/labels${query({ after, limit })}`, signal),
  getTraderScorecard: (synthetic: boolean, signal?: AbortSignal) =>
    request<TraderScorecard>(`/api/v1/trader/scorecard${query({ synthetic: synthetic ? 1 : undefined })}`, signal),
  getTraderAlerts: (signal?: AbortSignal) => request<TraderAlerts>(`/api/v1/trader/alerts${query({ limit: 200 })}`, signal),
  getTraderContext: (date: string | undefined, signal?: AbortSignal) =>
    request<TraderContext>(`/api/v1/trader/context${query({ date })}`, signal),
  getTraderSeries: (signal?: AbortSignal) => request<TraderSeries>(`/api/v1/trader/series${query({ limit: 200 })}`, signal),
  getTraderHealth: (signal?: AbortSignal) => request<TraderHealth>('/api/v1/trader/health', signal),
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
  getSignals: (limit?: number, signal?: AbortSignal, run: RunQuery = {}) =>
    request<SignalsView>(`/api/v1/signals${query({ limit, ...run })}`, signal),
  getRiskTargets: (limit?: number, signal?: AbortSignal, run: RunQuery = {}) =>
    request<RiskView>(`/api/v1/risk/targets${query({ limit, ...run })}`, signal),
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
  getPaperReplay: (signal?: AbortSignal) =>
    request<import('./types').PaperReplaySummary>('/api/v1/paper/replay', signal),
  getPaperReplayEquity: (product: string, signal?: AbortSignal) =>
    request<import('./types').PaperReplayEquity>(
      `/api/v1/paper/replay/${encodeURIComponent(product)}/equity?limit=500`, signal),
  getPaperReplayFills: (product: string, cursor?: string, signal?: AbortSignal) =>
    request<import('./types').PaperReplayFills>(
      `/api/v1/paper/replay/${encodeURIComponent(product)}/fills${query({ limit: 100, cursor })}`, signal),
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
  getHealthHistory: (component?: string, limit?: number, signal?: AbortSignal) =>
    request<HealthHistory>(
      `/api/v1/ops/health-history${query({ component, limit })}`, signal),
  getOpsHealth: (signal?: AbortSignal) =>
    request<OpsHealth>('/api/v1/ops/health', signal),
  getOpsRuntime: (signal?: AbortSignal) =>
    request<OpsRuntime>('/api/v1/ops/runtime', signal),
  getOpsRecovery: (signal?: AbortSignal) =>
    request<OpsRecovery>('/api/v1/ops/recovery', signal),
  getOpsStorage: (signal?: AbortSignal) =>
    request<OpsStorage>('/api/v1/ops/storage', signal),
  getOpsSettings: (signal?: AbortSignal) =>
    request<OpsSettings>('/api/v1/ops/settings', signal),
  getInstruments: (signal?: AbortSignal) =>
    request<InstrumentsIndex>('/api/v1/instruments', signal),
  getInstrument: (instrumentId: string, signal?: AbortSignal) =>
    request<InstrumentDetail>(
      `/api/v1/instruments/${encodeURIComponent(instrumentId)}`, signal),
  /** Whether the local equity research corpus binds on this machine.
   *  Status only. There is deliberately no companion call that fetches,
   *  repairs or refreshes it -- capture is a command-line act. */
  getResearchCorpus: (signal?: AbortSignal) =>
    request<ResearchCorpusStatus>('/api/v1/research/equities/corpus', signal),
  /** A bounded page of canonical daily bars from the local corpus. */
  getResearchBars: (
    instrumentId: string,
    options: { start?: string; end?: string; limit?: number; cursor?: string } = {},
    signal?: AbortSignal,
  ) =>
    request<ResearchBarPage>(
      `/api/v1/research/equities/${encodeURIComponent(instrumentId)}/bars${query({
        start: options.start,
        end: options.end,
        limit: options.limit,
        cursor: options.cursor,
      })}`,
      signal,
    ),
  getProviders: (signal?: AbortSignal) =>
    request<ProvidersIndex>('/api/v1/providers', signal),
  getCalendars: (signal?: AbortSignal) =>
    request<CalendarsIndex>('/api/v1/calendars', signal),
  /** Real sessions over a bounded window. The server refuses anything wider
   *  than a year, so the range is passed through rather than clamped here. */
  getInstrumentSessions: (
    instrumentId: string,
    params: { start?: string; end?: string; timeframe?: string } = {},
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams();
    if (params.start) query.set('start', params.start);
    if (params.end) query.set('end', params.end);
    if (params.timeframe) query.set('timeframe', params.timeframe);
    const suffix = query.toString() ? `?${query}` : '';
    return request<InstrumentSessions>(
      `/api/v1/instruments/${encodeURIComponent(instrumentId)}/sessions${suffix}`,
      signal,
    );
  },
  getPortfolio: (signal?: AbortSignal) =>
    request<PortfolioStatus>('/api/v1/portfolio', signal),
  getPortfolioBacktests: (signal?: AbortSignal) =>
    request<PortfolioBacktestsIndex>('/api/v1/portfolio/backtests', signal),
  getPortfolioDetail: (version: string, signal?: AbortSignal) =>
    request<PortfolioDetail>(
      `/api/v1/portfolio/backtests/${encodeURIComponent(version)}`, signal),
  getPortfolioEquity: (version: string, maxPoints?: number, signal?: AbortSignal) =>
    request<PortfolioEquity>(
      `/api/v1/portfolio/backtests/${encodeURIComponent(version)}/equity${query({ max_points: maxPoints })}`,
      signal),
  getPortfolioFills: (version: string, limit?: number, signal?: AbortSignal) =>
    request<PortfolioFillPage>(
      `/api/v1/portfolio/backtests/${encodeURIComponent(version)}/fills${query({ limit })}`,
      signal),
  getPaperPortfolio: (signal?: AbortSignal) =>
    request<PaperPortfolioStatus>('/api/v1/paper/portfolio', signal),
  getPaperPortfolioPending: (signal?: AbortSignal) =>
    request<PaperPortfolioPending>('/api/v1/paper/portfolio/pending', signal),
  getPaperPortfolioEquity: (maxPoints?: number, signal?: AbortSignal) =>
    request<PaperPortfolioEquity>(
      `/api/v1/paper/portfolio/equity${query({ max_points: maxPoints })}`, signal),
  getPaperLegacy: (signal?: AbortSignal) =>
    request<PaperLegacy>('/api/v1/paper/legacy', signal),
  getPortfolioAttribution: (version: string, signal?: AbortSignal) =>
    request<PortfolioAttribution>(
      `/api/v1/portfolio/backtests/${encodeURIComponent(version)}/attribution`, signal),
  /** The FOMC event store, read-only: status, a read at (as_of, horizon), an item, a verified replay. */
  getSourceTimeline: (source: 'fomc' | 'edgar', asOf: string, horizon?: number,
    signal?: AbortSignal, page?: SourcePageOptions) => request<SourceTimeline>(
      `/api/v1/sources/${source}/timeline${query({ as_of: asOf, horizon, ...page })}`, signal),
  getFomcStatus: (signal?: AbortSignal) =>
    request<FomcSourceStatus>('/api/v1/sources/fomc', signal),
  getFomcSnapshot: (asOf: string, horizon?: number, signal?: AbortSignal, page?: SourcePageOptions) =>
    request<FomcSnapshotView>(
      `/api/v1/sources/fomc/snapshot${query({ as_of: asOf, horizon, ...page })}`, signal),
  getFomcItem: (sid: string, asOf: string, horizon?: number, signal?: AbortSignal, page?: SourcePageOptions) =>
    request<FomcItemDetail>(
      `/api/v1/sources/fomc/items/${encodeURIComponent(sid)}${query({ as_of: asOf, horizon, ...page })}`, signal),
  getFomcReplay: (asOf: string, horizon?: number, signal?: AbortSignal) =>
    request<FomcReplay>(
      `/api/v1/sources/fomc/replay${query({ as_of: asOf, horizon })}`, signal),
  /** The SEC EDGAR store (offline slice), read-only: status, a read, a filing, a verified replay. */
  getEdgarStatus: (signal?: AbortSignal) =>
    request<EdgarSourceStatus>('/api/v1/sources/edgar', signal),
  getEdgarSnapshot: (asOf: string, horizon?: number, signal?: AbortSignal, page?: SourcePageOptions) =>
    request<EdgarSnapshotView>(
      `/api/v1/sources/edgar/snapshot${query({ as_of: asOf, horizon, ...page })}`, signal),
  getEdgarFiling: (accession: string, asOf: string, horizon?: number, signal?: AbortSignal, page?: SourcePageOptions) =>
    request<EdgarFilingDetail>(
      `/api/v1/sources/edgar/filings/${encodeURIComponent(accession)}${query({ as_of: asOf, horizon, ...page })}`, signal),
  getEdgarReplay: (asOf: string, horizon?: number, signal?: AbortSignal) =>
    request<EdgarReplay>(
      `/api/v1/sources/edgar/replay${query({ as_of: asOf, horizon })}`, signal),
  /** Model Lab (operator token in a bearer header). The listener admits this page only from its own
   *  loopback origin; the POSTs below are its synthetic job controls, nothing else writes. */
  getLabModels: (token: string, signal?: AbortSignal) =>
    request<LabModels>('/api/v1/lab/models', signal, bearer(token)),
  getLabJobs: (token: string, signal?: AbortSignal) =>
    request<LabJobs>('/api/v1/lab/jobs', signal, bearer(token)),
  getLabJob: (token: string, id: string, signal?: AbortSignal) =>
    request<JobStatus>(`/api/v1/lab/jobs/${encodeURIComponent(id)}`, signal, bearer(token)),
  getLabDatasetResult: (token: string, id: string, signal?: AbortSignal) =>
    request<DatasetResult>(`/api/v1/lab/jobs/${encodeURIComponent(id)}/results`, signal, bearer(token)),
  getLabExperimentResult: (token: string, id: string, signal?: AbortSignal) =>
    request<ExperimentResult>(`/api/v1/lab/jobs/${encodeURIComponent(id)}/results`, signal, bearer(token)),
  getLabMonitoringResult: (token: string, id: string, signal?: AbortSignal) =>
    request<LabMonitoringResult>(`/api/v1/lab/jobs/${encodeURIComponent(id)}/results`, signal, bearer(token)),
  createLabDataset: (token: string, body: Record<string, unknown>) =>
    request<LabSubmitted>('/api/v1/lab/datasets', undefined, bearer(token), body),
  createLabExperiment: (token: string, body: Record<string, unknown>) =>
    request<LabSubmitted>('/api/v1/lab/experiments', undefined, bearer(token), body),
  createLabMonitoring: (token: string, experimentJobId: string) =>
    request<LabSubmitted>('/api/v1/lab/monitoring', undefined, bearer(token), { experiment_job_id: experimentJobId }),
  cancelLabJob: (token: string, id: string) =>
    request<JobStatus>(`/api/v1/lab/jobs/${encodeURIComponent(id)}/cancel`, undefined, bearer(token), {}),
  /** Research registry and observability: read-only, same posture as the other source views. */
  getResearchHypotheses: (options: { limit?: number; cursor?: string; asOf?: string } = {}, signal?: AbortSignal) =>
    request<ResearchPage<HypothesisRow>>(
      `/api/v1/research/hypotheses${query({ limit: options.limit, cursor: options.cursor, as_of: options.asOf })}`, signal),
  getResearchHypothesis: (identity: string, signal?: AbortSignal) =>
    request<HypothesisDetail>(`/api/v1/research/hypotheses/${encodeURIComponent(identity)}`, signal),
  getResearchExperiments: (options: { limit?: number; cursor?: string; asOf?: string } = {}, signal?: AbortSignal) =>
    request<ResearchPage<TrialRow>>(
      `/api/v1/research/experiments${query({ limit: options.limit, cursor: options.cursor, as_of: options.asOf })}`, signal),
  getResearchComparison: (signal?: AbortSignal) =>
    request<ComparisonReadiness>('/api/v1/research/comparison', signal),
  getResearchProposals: (signal?: AbortSignal) =>
    request<ProposalCatalogue>('/api/v1/research/proposals', signal),
  getObservabilityPredictions: (
    options: { product?: string; modelId?: string; start?: string; end?: string; limit?: number; cursor?: string; asOf?: string } = {},
    signal?: AbortSignal,
  ) =>
    request<ResearchPage<LedgerRow>>(`/api/v1/observability/predictions${query({
      product: options.product, model_id: options.modelId, start: options.start, end: options.end,
      limit: options.limit, cursor: options.cursor, as_of: options.asOf,
    })}`, signal),
  getObservabilityPrediction: (identity: string, signal?: AbortSignal) =>
    request<PredictionView>(`/api/v1/observability/predictions/${encodeURIComponent(identity)}`, signal),
  getObservabilityReferences: (limit?: number, signal?: AbortSignal) =>
    request<ResearchPage<ReferenceRow>>(`/api/v1/observability/references${query({ limit })}`, signal),
  getObservabilityHealth: (signal?: AbortSignal) =>
    request<ObservabilityHealth>('/api/v1/observability/health', signal),
  getMonitoring: (
    options: { asOf: string; product?: string; modelId?: string; start?: string; end?: string; referenceHash?: string },
    signal?: AbortSignal,
  ) =>
    request<MonitoringView>(`/api/v1/observability/monitoring${query({
      as_of: options.asOf, product: options.product, model_id: options.modelId,
      start: options.start, end: options.end, reference_hash: options.referenceHash,
    })}`, signal),
};

export type ApiClient = typeof apiClient;
