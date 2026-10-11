/** Offline, sanitized observations. Missing values are never estimated by the UI. */
export type Verification = 'OFFICIEL' | 'FIL' | 'NON VERIFIE';
export interface DecisionSource {
  url: string | null; publisher: string | null; published_at: string | null; verification: Verification;
}
export interface DecisionResult {
  pnl_after_costs: number | null; r_after_costs: number | null; net_return: number | null;
  costs: number | null; at: string | null; basis: string;
}
export interface DecisionChain {
  fact: string | null; sources: DecisionSource[]; verification: Verification;
  expectations: string | null; expectation_at: string | null; priced_in: string | null;
  scenario: string | null; entry_condition: string | null; invalidation: string | null; result: DecisionResult | null;
}
export interface MarketObservation { at: string; price: number; basis: string }
export type OverlayKind = 'event' | 'decision' | 'outcome' | 'book_entry' | 'book_exit';
export interface MarketOverlay {
  id: string; kind: OverlayKind; asset: string; at: string | null; horizon: string | null; label: string;
  analyst?: string; verdict?: string; price?: number | null; price_basis?: string | null;
  stop?: number | null; net_return?: number | null; decision: DecisionChain | null;
}
export interface BookTrade {
  id: string; symbol: string; at: string; tag: string; group: string; state: string; closed_at: string | null;
  risk_usd: number | null; entry: number | null; stop: number | null; target: number | null; decision: DecisionChain;
}
export interface NewsGroup {
  group: string; count: number; closed: number; scored: number; r_samples: number; hit_rate: number | null;
  average_r: number | null; pnl_after_costs: number | null; state: string;
}
export interface UnitObservation {
  unit: string; state: string | null; result: string | null; last_run: string | null; next_run: string | null; exit_status?: string;
}
export interface AnalysisSnapshot {
  schema: 'cockpit-analysis-v1'; generated_at: string; prices: Record<string, MarketObservation[]>;
  overlays: MarketOverlay[]; book_trades: BookTrade[];
  news: {
    groups: NewsGroup[]; by_tag?: NewsGroup[]; minimum_n: number; tag_counts: Record<string, number>;
    ai_scores: Record<string, import('./traderTypes').ScoreEntry>;
    context_comparison: string; note: string;
  };
  system: {
    radar_at: string | null;
    sources: Array<{ source: string; status: string; reason: string | null; checked_at: string | null; items: number | null }>;
    budget_date: string; budgets_used: Record<string, number>;
    trader: {
      health?: { state: string; at: string; budget_counts: Record<string, number> } | null;
      last_label?: { state: string; at: string } | null; paused?: boolean;
      failures?: Array<{ at: string; code: string }>;
    };
    units: UnitObservation[];
  };
  limitations: string[];
}
