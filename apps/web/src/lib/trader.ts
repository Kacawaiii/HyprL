/**
 * Pure derivations for the Agent trader view.
 *
 * Nothing here estimates anything: it joins records the API already serves (predictions with their
 * separately appended labels), groups them for reading, and states each definition next to its value.
 * Accuracy and calibration numbers come from the scorecard verbatim.
 */
import type {
  Chained, Cohort, DecisionView, Horizon, LabelPayload, PredictionPayload, RunRow, ScoreEntry, SeriesPoint, TraderScorecard,
} from '../api/traderTypes';

export const NOT_A_CLAIM =
  'This is a paper experiment, not advice and not a recommendation. No real order is ever sent. ' +
  'No edge has been shown: the probabilities are uncalibrated model judgments and the sample is small.';

export const ANALYSTS = ['analyst_claude', 'analyst_gpt'] as const;

export function analystLabel(name: string): string {
  return ({ analyst_claude: 'Claude analyst', analyst_gpt: 'GPT analyst', reviewer_claude: 'Reviewed Claude',
    reviewer_gpt: 'Reviewed GPT', consensus: 'Consensus' } as Record<string, string>)[name] ?? name;
}

export function horizonText(horizon: Horizon): string {
  return horizon === '1d' ? 'next session close' : '5 sessions ahead';
}

export function directionText(view: DecisionView['view']): string {
  return view === 'UP' ? 'Up' : view === 'DOWN' ? 'Down' : 'No view';
}

/** What a probability means for this asset: equities are scored against SPY, crypto in raw direction. */
export function targetText(asset: string): string {
  return asset.endsWith('-USD') ? 'probability the price ends higher' : 'probability it beats SPY';
}

export interface TodayRow {
  asset: string; horizon: Horizon; consensus: DecisionView; reason: string;
}

/** One row per asset and horizon: the consensus, with the first available analyst reason as the one-liner. */
export function todayRows(views: DecisionView[]): TodayRow[] {
  const rows: TodayRow[] = [];
  for (const consensus of views.filter((v) => v.analyst === 'consensus')) {
    const analysts = views.filter((v) => ANALYSTS.includes(v.analyst as never) && v.asset === consensus.asset && v.horizon === consensus.horizon);
    const reason = consensus.view === 'ABSTAIN'
      ? 'The two reviewed analysts do not agree on a direction, or one abstained: no view is taken.'
      : analysts.find((v) => v.raw_view?.view === consensus.view)?.raw_view?.confidence_reason ?? 'No reason recorded.';
    rows.push({ asset: consensus.asset, horizon: consensus.horizon, consensus, reason });
  }
  return rows.sort((a, b) => a.asset.localeCompare(b.asset) || a.horizon.localeCompare(b.horizon));
}

export interface Rejection { analyst: string; asset: string; horizon: Horizon; verdict: 'REJECT' | 'DOWNGRADE'; code: string; note: string; from: number; to: number | null }

export function rejections(views: DecisionView[]): Rejection[] {
  return views
    .filter((v) => ANALYSTS.includes(v.analyst as never) && v.review && (v.verdict === 'REJECT' || v.verdict === 'DOWNGRADE'))
    .map((v) => ({ analyst: v.analyst, asset: v.asset, horizon: v.horizon, verdict: v.verdict as 'REJECT' | 'DOWNGRADE',
      code: v.review!.reason_code, note: v.review!.note, from: v.p_outperform, to: v.review!.adjusted_p }));
}

/** The side-by-side unit for the expert: both analysts and the reviewer verdict for one asset/horizon. */
export function sideBySide(views: DecisionView[], asset: string, horizon: Horizon): DecisionView[] {
  return ANALYSTS.map((name) => views.find((v) => v.analyst === name && v.asset === asset && v.horizon === horizon))
    .filter((v): v is DecisionView => Boolean(v));
}

export function assetsOf(views: DecisionView[]): string[] {
  return [...new Set(views.map((v) => v.asset))].sort();
}

export type LabelState = 'realized' | 'pending';
export interface LedgerRow {
  prediction: Chained<PredictionPayload>; label: LabelPayload | null; state: LabelState;
  /** The realized return on the scorecard's target, as served; null while pending. Right or wrong is the scorecard's job, not the browser's. */
  scoredReturn: number | null;
}

/** Realized outcome on the scorecard's target: equities vs SPY, crypto raw. */
export function scoredReturn(label: LabelPayload): number {
  return label.value.spy_relative_return ?? label.value.raw_return;
}

/** Predictions joined to their labels by the prediction hash. A label never changes a prediction. */
export function joinLedger(predictions: Array<Chained<PredictionPayload>>, labels: Array<Chained<LabelPayload>>): LedgerRow[] {
  const byHash = new Map(labels.map((l) => [l.payload.prediction_hash, l.payload]));
  return predictions.map((prediction) => {
    const label = byHash.get(prediction.identity) ?? null;
    return { prediction, label, state: label ? 'realized' : 'pending', scoredReturn: label ? scoredReturn(label) : null };
  });
}

export function ledgerCounts(rows: LedgerRow[]) {
  const realized = rows.filter((r) => r.state === 'realized').length;
  return { total: rows.length, realized, pending: rows.length - realized };
}

export function priceOf(label: LabelPayload, side: 0 | 1): number | null {
  const prices = label.value.raw_prices;
  const pair = Array.isArray(prices) ? prices : prices[label.product];
  return pair ? pair[side] ?? null : null;
}

export interface ChartPoint { at: number; price: number }
export interface ChartDecision { at: number; price: number; view: DecisionView['view']; horizon: Horizon; p: number; realized: boolean }
export interface ChartOutcome { from: ChartPoint; to: ChartPoint; horizon: Horizon; net: number }
export interface AssetChart { asset: string; line: ChartPoint[]; decisions: ChartDecision[]; outcomes: ChartOutcome[] }

/** Prices, decisions and realized outcomes on one time axis. Only timestamped values are plotted. */
export function assetChart(asset: string, series: SeriesPoint[], rows: LedgerRow[]): AssetChart {
  const line = series.map((p) => ({ at: Date.parse(p.price_at), price: p.close })).sort((a, b) => a.at - b.at);
  const priceAt = (t: number) => [...line].reverse().find((p) => p.at <= t)?.price ?? null;
  const decisions: ChartDecision[] = [];
  const outcomes: ChartOutcome[] = [];
  for (const row of rows.filter((r) => r.prediction.payload.product === asset && r.prediction.payload.model_id === 'trader:consensus')) {
    const t = Date.parse(row.prediction.payload.decision_at);
    const price = priceAt(t);
    if (price !== null) {
      decisions.push({ at: t, price, view: row.prediction.payload.outputs.class, horizon: row.prediction.payload.signal.label_definition.horizon,
        p: row.prediction.payload.outputs.probabilities.outperform, realized: row.state === 'realized' });
    }
    if (row.label) {
      const open = priceOf(row.label, 0);
      const close = priceOf(row.label, 1);
      if (open !== null && close !== null) {
        outcomes.push({ from: { at: Date.parse(row.label.provenance.definition.entry_at), price: open },
          to: { at: Date.parse(row.label.provenance.definition.exit_at), price: close },
          horizon: row.label.provenance.definition.horizon, net: row.label.value.net_unit_pnl });
      }
    }
  }
  return { asset, line, decisions, outcomes };
}

export type RunState = 'none' | 'running' | 'complete' | 'degraded' | 'skipped' | 'failed' | 'paused' | 'not_started';

export function runState(statuses: string[]): RunState {
  if (statuses.length === 0) return 'none';
  if (statuses.includes('COMPLETE')) return 'complete';
  if (statuses.includes('DEGRADED')) return 'degraded';
  const last = statuses[statuses.length - 1] ?? '';
  if (last === 'RUNNING') return 'running';
  if (last === 'PAUSED') return 'paused';
  if (last === 'NOT_STARTED') return 'not_started';
  if (last.startsWith('SKIPPED')) return 'skipped';
  return 'failed';
}

export const RUN_STATE_TEXT: Record<RunState, { title: string; detail: string }> = {
  none: { title: 'No run yet for this day', detail: 'The trader runs once per US trading day at 12:00 UTC. Nothing is shown until a run completes; no value is filled in.' },
  running: { title: 'Run in progress or interrupted', detail: 'A run started but no decision was recorded. A run that never completes issues no predictions.' },
  complete: { title: 'Run complete', detail: '' },
  degraded: { title: 'Run degraded', detail: 'One analyst produced no valid output. Its views are marked MISSING with the error code, the consensus abstains for every asset, and the other analyst and the reviewer are scored as usual.' },
  skipped: { title: 'Run skipped', detail: 'The run was skipped on purpose (a market holiday or an exhausted quota). No predictions exist for this day.' },
  failed: { title: 'Run failed', detail: 'The run did not produce a valid decision (see run health for the reason). Failures are never retried into a prediction.' },
  paused: { title: 'Trader paused', detail: 'An operator paused the trader. No run happens until it is resumed.' },
  not_started: { title: 'Trader not started yet', detail: 'The prospective period has not begun.' },
};

export interface ScoreKey { analyst: string; population: 'crypto' | 'equity_etf'; horizon: Horizon; target: 'raw' | 'SPY_relative' }
export const scoreKey = (k: ScoreKey) => `${k.analyst}/${k.population}/${k.horizon}/${k.target}`;

export const BASELINES = ['always_up', 'momentum20', 'random_seeded', 'spy_relative_zero'] as const;

export interface BaselineRow { name: string; entry: ScoreEntry }
export function baselineRows(card: TraderScorecard, population: ScoreKey['population'], horizon: Horizon, target: ScoreKey['target']): BaselineRow[] {
  return ['consensus', ...BASELINES].flatMap((name) => {
    const entry = card.scores[scoreKey({ analyst: name, population, horizon, target })];
    return entry ? [{ name, entry }] : [];
  });
}

/** Kept vs rejected raw analyst views: a rejection that was right shows a low hit rate in the rejected group. */
export function keptVsRejected(card: TraderScorecard, population: ScoreKey['population'], horizon: Horizon, target: ScoreKey['target']) {
  const at = (analyst: string) => card.scores[scoreKey({ analyst, population, horizon, target })] ?? null;
  return { kept: at('reviewer_kept'), rejected: at('reviewer_rejected'), downgraded: at('reviewer_downgraded') };
}

export interface PaperLine { analyst: string; horizon: Horizon; complete: number; pending: number; unhedged: number | null; hedged: number | null }

/** Paper P&L per analyst and horizon: the plain sum of completed cohorts' returns on their capital fraction (no compounding). */
export function paperLines(cohorts: Cohort[]): PaperLine[] {
  const map = new Map<string, PaperLine>();
  for (const c of cohorts) {
    const key = `${c.analyst}/${c.horizon}`;
    const line = map.get(key) ?? { analyst: c.analyst, horizon: c.horizon, complete: 0, pending: 0, unhedged: null, hedged: null };
    if (c.state === 'COMPLETE' && c.variants.unhedged.capital_return !== null) {
      line.complete += 1;
      line.unhedged = (line.unhedged ?? 0) + c.variants.unhedged.capital_return;
      line.hedged = (line.hedged ?? 0) + (c.variants.spy_hedged.capital_return ?? 0);
    } else {
      line.pending += 1;
    }
    map.set(key, line);
  }
  return [...map.values()].sort((a, b) => a.analyst.localeCompare(b.analyst) || a.horizon.localeCompare(b.horizon));
}

export function alertCounts(alerts: Array<{ code: string }>): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const a of alerts) counts.set(a.code, (counts.get(a.code) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

export function runCounts(runs: RunRow[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const r of runs.filter((x) => x.status !== 'RUNNING' || !runs.some((y) => y.run_id === x.run_id && y.status !== 'RUNNING'))) {
    counts.set(r.status, (counts.get(r.status) ?? 0) + 1);
  }
  return [...counts.entries()];
}

const TAINT = new Set(['TAINTED_RUN']);
export const isTaint = (code: string) => TAINT.has(code);
