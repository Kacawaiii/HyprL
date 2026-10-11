/** Shapes served by /api/v1/radar/* (sanitized snapshots written by scripts/radar/cockpit_export.py). Null means unknown, never invented. */

export type RadarAnalyst = 'analyst_claude' | 'analyst_gpt' | 'reviewer_claude' | 'reviewer_gpt' | 'consensus';

export interface AnticipationView {
  decision?: import('./analysisTypes').DecisionChain;
  analyst: string; horizon: string; view: 'UP' | 'DOWN' | 'ABSTAIN' | string; p_outperform: number | null;
  verdict: string | null; reason: string | null; falsifier: string | null; review_note: string | null;
  catalysts: Array<{ url: string; published_at: string | null }>;
}
export interface TimelinePoint {
  run_id: string; at: string;
  by_analyst: Record<string, { view: string; p_outperform: number | null; verdict: string | null; horizon: string }>;
}
export interface Anticipation {
  state: 'COVERED' | 'NOT_COVERED';
  latest: { run_id: string; at: string; views: AnticipationView[] } | null;
  timeline: TimelinePoint[];
}
export interface Outcome {
  model_id: string; run_id: string; horizon: string; view: string | null; realized_at: string | null;
  available_at: string | null; raw_return: number | null; spy_relative_return: number | null;
  net_unit_pnl: number | null; cost_roundtrip: number | null;
}
export interface RadarAsset {
  symbol: string; role: string | null; mechanism: string | null; direction: string | null;
  priced_in: null | { return_pct: number | null; move_atr: number | null; baseline_at: string; price_at: string; note: string | null };
  anticipation: Anticipation; outcomes: Outcome[];
}
export interface ScoreEntry {
  issued: number | null; realized: number | null; hit_rate: number | null; brier: number | null;
  climatology_brier: number | null; ic: number | null; mean_unit_pnl_after_costs: number | null; days: number | null;
}
export interface EventBadges {
  event_importance: { value: number | null; components: Record<string, number> | null };
  evidence_strength: { value: number | null; status: string | null; publishers: number; primary: boolean };
  model_conviction: { label: string | null; by_analyst: Record<string, number> | null; basis: string };
  predictive_quality: { by_analyst: Record<string, ScoreEntry> | null; basis: string };
  after_cost_performance: {
    event_labels: null | { n: number; mean_net_unit_pnl: number };
    paper_accounts: Array<{ account: string; suffix: string; return_since_start: number | null }>;
    basis: string;
  };
}
export interface RadarStory {
  publisher: string | null; url: string; headline: string | null; published_at: string | null;
  received_at: string | null; primary: boolean;
}
export interface RadarEvent {
  decision?: import('./analysisTypes').DecisionChain;
  id: string; rank: number; headline: string; link: string | null; source: string | null;
  published_at: string | null; available_at: string | null; novelty: string | null;
  themes: string[]; countries: string[]; badges: EventBadges;
  what_changed: {
    expectations: string | null; changed_expectations: string | null; summary: string | null; impact: string | null;
    horizon: string | null; priced_in: string | null; invalidation: string | null; source: 'llm_scenario' | 'rules_only';
  };
  assets: RadarAsset[]; stories: RadarStory[];
  provenance: { priced_in_status: string | null; independence: string | null; retail_hype: number | null };
}
export interface RegimeItem {
  symbol: string; last: number | null; at: string; status: string; returns_pct: Record<string, number | null>;
}
export interface RadarHome {
  schema: 'cockpit-radar-home-v1'; generated_at: string;
  radar: {
    date: string; slot: string; cutoff: string; status: string; report_hash: string; total_events: number;
    shown_events: number; sources_by_status: Record<string, number>; limitations: string[];
  };
  trader: {
    runs: number; labels: number; latest_run: string | null; latest_run_at: string | null;
    models: Record<string, { model: string; cli_version: string; reported_version: string }>;
    preregistration_hash: string | null; predictive_quality: Record<string, ScoreEntry>; hypothesis_state: string | null;
  };
  regime: Record<string, RegimeItem>; events: RadarEvent[];
}

export interface PaperPosition {
  decision?: import('./analysisTypes').DecisionChain;
  symbol: string; asset_class: string | null; tag: string; qty: number | null; entry: number | null; last: number | null;
  unrealized_pl: number | null; unrealized_pct: number | null; stop: number | null; target: number | null;
  protection: 'policy' | 'none_defined'; opened_at: string | null;
  reason: null | { event: string | null; mechanism: string | null; priced_in: string | null; scenario: string | null; invalidation: string | null };
}
export interface PaperOrder {
  symbol: string; side: string; qty: number | null; type: string; limit: number | null; status: string;
  submitted: string | null; legs: Array<{ type: string; limit: number | null; stop: number | null }>;
}
export interface JournalRow {
  decision?: import('./analysisTypes').DecisionChain;
  at: string; action: string; symbol: string | null; qty: number | null; limit: number | null; stop: number | null;
  target: number | null; engine: string | null; reason: string | null; mechanism: string | null; invalidation: string | null;
}
export interface PaperAccount {
  observed_at?: string;
  account: string; suffix: string; label: string; equity: number | null; return_since_start: number | null;
  start_equity?: number | null; cash?: number | null; peak: number | null; pnl?: number | null; day_pnl?: number | null;
  open_lots?: number; halted?: boolean; planned_orders?: number;
  positions: PaperPosition[]; open_orders?: PaperOrder[]; tags?: string[]; journal: JournalRow[];
}
export interface PaperSnapshot {
  schema: 'cockpit-paper-v1'; generated_at: string; report_at: string | null; accounts: PaperAccount[];
  benchmarks: { SPY: { price: number | null; return_since_paper_start: number | null; base_at: string | null }; BTC: { price: number | null; at: string | null } };
  curve: Array<{ at: string; equity: Record<string, number>; spy: number | null; btc: number | null }>;
  limitations: string[]; notes: string[];
}
