import type { AnalysisSnapshot, DecisionChain } from '../api/analysisTypes';
import type { ScoreEntry } from '../api/traderTypes';
import { blankDecision } from '../lib/decisions';

export const feedDecision: DecisionChain = { ...blankDecision(), fact: 'Synthetic earnings headline', verification: 'FIL',
  sources: [{ url: 'https://news.example.invalid/a', publisher: 'Examplewire', published_at: '2026-10-10T10:00:00Z', verification: 'FIL' }],
  expectations: 'Synthetic consensus EPS 1.00', expectation_at: '2026-10-09T18:00:00Z', priced_in: 'Partly priced in',
  scenario: 'Margin expands', entry_condition: 'Wait for a breakout', invalidation: 'Margin below 10%' };
const score: ScoreEntry = { issued: 10, realized: 4, non_abstained: 3, pending: 6, hit_rate: .666666, brier: .2,
  climatology_brier: .25, false_positive_rate: .1, calibration_bins: [], ic: .1, mean_unit_pnl_after_costs: .004,
  abstention_rate: .25, days: 2, ties: 0 };
export const analysisSnapshot: AnalysisSnapshot = {
  schema: 'cockpit-analysis-v1', generated_at: '2026-10-11T00:30:00Z',
  prices: { 'AAA-USD': [
    { at: '2026-10-09T20:00:00Z', price: 100, basis: 'reference_close' },
    { at: '2026-10-10T20:00:00Z', price: 102, basis: 'label_price' },
  ] },
  overlays: [
    { id: 'e', kind: 'event', asset: 'AAA/USD', at: '2026-10-10T10:05:00Z', horizon: null, label: 'Synthetic headline', decision: feedDecision },
    { id: 'd1', kind: 'decision', asset: 'AAA-USD', at: '2026-10-10T12:00:00Z', horizon: '1d', analyst: 'analyst_claude', verdict: 'KEEP', label: 'UP', decision: feedDecision },
    { id: 'd5', kind: 'decision', asset: 'AAA-USD', at: '2026-10-10T12:00:00Z', horizon: '5d', analyst: 'analyst_gpt', verdict: 'REJECT', label: 'DOWN', decision: { ...feedDecision, invalidation: 'Five-session falsifier' } },
    { id: 'net', kind: 'outcome', asset: 'AAA-USD', at: '2026-10-10T20:00:00Z', horizon: '1d', analyst: 'analyst_claude', label: 'Net result', net_return: .004, decision: null },
    { id: 'book', kind: 'book_entry', asset: 'AAAUSD', at: '2026-10-10T14:00:00Z', horizon: null, label: 'Claude book technical', price: 101, price_basis: 'intent_limit', stop: 95, decision: feedDecision },
  ], book_trades: [],
  news: { groups: [
    { group: 'news', count: 3, closed: 2, scored: 1, r_samples: 1, hit_rate: 1, average_r: 2, pnl_after_costs: 100, state: 'TOO_EARLY' },
    { group: 'technical', count: 2, closed: 0, scored: 0, r_samples: 0, hit_rate: null, average_r: null, pnl_after_costs: null, state: 'TOO_EARLY' },
  ], minimum_n: 30, tag_counts: { news: 3, technical: 2 },
  ai_scores: { 'consensus/equity_etf/1d/SPY_relative': score, 'random_seeded/equity_etf/1d/SPY_relative': score },
  context_comparison: 'UNAVAILABLE', note: 'Aucune variante appariée avec/sans actualité identifiée.' },
  system: { radar_at: '2026-10-10T20:30:00Z', budget_date: '2026-10-11', budgets_used: { rss: 2 },
    sources: [
      { source: 'rss:example', status: 'LIVE', reason: null, checked_at: '2026-10-01T10:00:00Z', items: 2 },
      { source: 'sec', status: 'BLOCKED', reason: 'DAILY_BUDGET', checked_at: '2026-10-10T20:30:00Z', items: null },
    ], trader: { health: null, failures: [{ at: '2026-10-10T19:00:00Z', code: 'OWNER_BUSY' }] },
    units: [{ unit: 'claude-book-open.timer', state: 'active', result: null, last_run: 'Fri 2026-10-09 13:00:00 UTC', next_run: 'Mon 2026-10-12 13:00:00 UTC' }] },
  limitations: ['Reference prices only.'],
};
