/**
 * Pure derivations for the Radar home. Values come from the snapshot verbatim; the only arithmetic here is
 * re-basing curves to a common start (display) and picking which stored view to show for a horizon.
 */
import type { AnticipationView, PaperSnapshot, TimelinePoint } from '../api/radarTypes';

export const ANALYST_ORDER = ['analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt', 'consensus'] as const;

const LABEL: Record<string, string> = {
  analyst_claude: 'Claude', analyst_gpt: 'GPT', reviewer_claude: 'Reviewer (Claude)', reviewer_gpt: 'Reviewer (GPT)',
  consensus: 'Consensus',
};
export function analystLabel(analyst: string): string {
  return LABEL[analyst] ?? analyst;
}

/** What a stored view means in words; ABSTAIN is a statement of no edge, never a hidden 50 %. */
export function viewWords(view: string | null | undefined): string {
  if (view === 'UP') return 'expects to outperform';
  if (view === 'DOWN') return 'expects to underperform';
  if (view === 'ABSTAIN') return 'no view';
  return 'unknown';
}

/** Probability as a whole percentage, or an explicit dash. */
export function pct(p: number | null | undefined, digits = 0): string {
  return p === null || p === undefined || !Number.isFinite(p) ? '—' : `${(p * 100).toFixed(digits)} %`;
}

/** A return already expressed as a fraction (0.012) shown with its sign. */
export function signedPct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const text = (value * 100).toFixed(digits);
  return `${value > 0 ? '+' : ''}${text} %`;
}

export function signedNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
}

/** `2026-10-10T11:30:23.4Z` -> `2026-10-10 11:30 UTC`. Instants are shown in UTC so T is unambiguous. */
export function instant(value: string | null | undefined): string {
  if (!value) return '—';
  const match = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(value);
  return match ? `${match[1]} ${match[2]} UTC` : value;
}

export function host(url: string | null | undefined): string {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, '') : '';
  } catch {
    return '';
  }
}

/** One view per analyst for a horizon, in the fixed reading order. */
export function viewsForHorizon(views: AnticipationView[], horizon: string): AnticipationView[] {
  const picked = new Map<string, AnticipationView>();
  for (const view of views) if (view.horizon === horizon) picked.set(view.analyst, view);
  return ANALYST_ORDER.filter((name) => picked.has(name)).map((name) => picked.get(name) as AnticipationView);
}

export function horizonsOf(views: AnticipationView[]): string[] {
  return [...new Set(views.map((view) => view.horizon))].sort();
}

export interface SeriesPoint { at: string; run_id: string; p: number; view: string }

export function analystSeries(timeline: TimelinePoint[], analyst: string): SeriesPoint[] {
  const out: SeriesPoint[] = [];
  for (const point of timeline) {
    const entry = point.by_analyst[analyst];
    if (entry && entry.p_outperform !== null) out.push({ at: point.at, run_id: point.run_id, p: entry.p_outperform, view: entry.view });
  }
  return out;
}

/** "0.60 -> 0.70 over 2 runs"; the evolution in one sentence for a reader who skips the chart. */
export function evolution(series: SeriesPoint[]): string {
  if (series.length === 0) return 'not covered by a run yet';
  const first = series[0];
  const last = series[series.length - 1];
  if (!first || !last) return 'not covered by a run yet';
  if (series.length === 1) return `${pct(first.p)} at the only run so far`;
  const delta = Math.round((last.p - first.p) * 100);
  const direction = delta === 0 ? 'unchanged' : `${delta > 0 ? '+' : ''}${delta} pts`;
  return `${pct(first.p)} → ${pct(last.p)} over ${series.length} runs (${direction})`;
}

export interface RebasedSeries { key: string; label: string; points: Array<{ at: string; value: number }> }

/** Account equity, SPY and BTC each re-based to 100 at their own first recorded point so one scale compares them. */
export function rebasedCurves(paper: PaperSnapshot): RebasedSeries[] {
  const names = new Map<string, string>(paper.accounts.map((account) => [account.account, account.label]));
  const series: RebasedSeries[] = [];
  const build = (key: string, label: string, read: (row: PaperSnapshot['curve'][number]) => number | null | undefined) => {
    const rows = paper.curve.map((row) => ({ at: row.at, raw: read(row) })).filter(
      (row): row is { at: string; raw: number } => typeof row.raw === 'number' && Number.isFinite(row.raw) && row.raw > 0);
    if (rows.length === 0) return;
    const base = rows[0]?.raw ?? 1;
    series.push({ key, label, points: rows.map((row) => ({ at: row.at, value: (row.raw / base) * 100 })) });
  };
  for (const [key, label] of names) build(key, label, (row) => row.equity[key]);
  build('SPY', 'SPY', (row) => row.spy);
  build('BTC', 'BTC', (row) => row.btc);
  return series;
}

/** Stop and target distances from the last price, for a position that has a policy; null otherwise. */
export function protectionDistances(last: number | null, stop: number | null, target: number | null) {
  if (last === null || !last) return { stop: null, target: null };
  return {
    stop: stop === null ? null : (stop - last) / last,
    target: target === null ? null : (target - last) / last,
  };
}
