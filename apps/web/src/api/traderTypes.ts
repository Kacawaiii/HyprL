/** Shapes served by /api/v1/trader/* (scripts/trading_lab/app_api/trader.py). Optional model outputs are null, never invented. */

export type ViewDirection = 'UP' | 'DOWN' | 'ABSTAIN';
export type Verdict = 'KEEP' | 'DOWNGRADE' | 'REJECT' | 'ABSTAIN';
export type Horizon = '1d' | '5d';

export interface Catalyst { url: string; published_at: string; fact: string }

export interface RawView {
  asset: string; horizon: Horizon; view: ViewDirection; p_outperform: number;
  confidence_reason: string; catalysts: Catalyst[]; priced_in_assessment: string; second_order: string;
  counter_thesis: string; falsifier: string; event_risk: unknown[];
}

export interface Review {
  analyst: string; asset: string; horizon: Horizon; verdict: Verdict; reason_code: string;
  adjusted_p: number | null; note: string;
}

/** `analyst` is analyst_claude | analyst_gpt (raw), reviewer_claude | reviewer_gpt (after review) or consensus. */
export interface DecisionView {
  analyst: string; asset: string; horizon: Horizon; view: ViewDirection; p_outperform: number;
  verdict: Verdict; raw_view: RawView | null; review: Review | null;
}

export interface Chained<T> { sequence: number; identity: string; recorded_at: string; chain_hash: string; payload: T }

export interface PortfolioVariant { weights?: Record<string, number> }
export interface HorizonPortfolio {
  capital_fraction: number; entry: string; execution: string; cohort_policy: string;
  unhedged: Record<string, number>; spy_hedged: Record<string, number>;
}

export interface RunSummary {
  schema: 'trader-run-v1'; run_id: string; status: string; at: string; synthetic: boolean; error?: string;
  budget_counts?: Record<string, number>; exclusions?: unknown[];
  portfolios?: Record<Horizon, HorizonPortfolio>;
  reviewed_analyst_portfolios?: Record<string, Record<Horizon, HorizonPortfolio>>;
  decision?: {
    run_id: string; session: string; decision_at: string; context_hash: string; synthetic: boolean;
    models: Record<string, { model: string; cli_version: string; reported_version: string }>;
    skill_hashes: Record<string, string>; preregistration_hash: string; views: DecisionView[];
  };
}

export interface TraderToday { schema: 'trader-today-v1'; date: string; runs: Array<Chained<RunSummary>> }

export interface RunRow {
  sequence: number; recorded_at: string; run_id: string; status: string; at: string; synthetic: boolean;
  error?: string; budget_counts?: Record<string, number>;
}
export interface TraderRuns { schema: 'trader-runs-page-v1'; records: RunRow[]; next_after: number | null }

export interface PredictionPayload {
  prediction_id: string; model_id: string; product: string; decision_at: string; horizon_seconds: number;
  outputs: { class: ViewDirection; probabilities: { outperform: number }; return: null; target_price: null; quantiles: null; scenarios: null };
  signal: { view: DecisionView; session: string; label_definition: { entry_at: string; exit_at: string; horizon: Horizon; targets: string[] }; run_id: string };
  risk: { mode: string; verdict: Verdict };
  proposed_position: { weight: number; entry_at: string; capital_usd: number; capital_fraction: number; variant: string };
  uncertainty: { method: string; calibration_claim: boolean };
  costs: { half_spread_bps: number; spread_method: string };
  synthetic: boolean;
}
export interface TraderLedger { schema: 'trader-ledger-page-v1'; records: Array<Chained<PredictionPayload>>; next_after: number | null; label_state: string }

export interface LabelValue {
  raw_return: number; spy_relative_return: number | null; spy_return: number | null; net_unit_pnl: number;
  cost_roundtrip: number; raw_prices: Record<string, [number, number]> | [number, number]; spread_method: string; synthetic: boolean;
}
export interface LabelPayload {
  label_id: string; prediction_id: string; prediction_hash: string; product: string; horizon_seconds: number;
  realized_at: string; available_at: string; recorded_at: string; value: LabelValue;
  provenance: { sources: Array<{ url: string; received_at: string; digest: string }>; definition: { entry_at: string; exit_at: string; horizon: Horizon } };
}
export interface TraderLabels { schema: 'trader-labels-page-v1'; records: Array<Chained<LabelPayload>>; next_after: number | null }

export interface CalibrationBin { low: number; high: number; n: number; mean_p: number | null; frequency: number | null }
export interface ScoreEntry {
  issued: number; realized: number; non_abstained: number; pending: number; hit_rate: number | null;
  false_positive_rate: number | null; brier: number | null; climatology_brier: number | null;
  calibration_bins: CalibrationBin[]; ic: number | null; mean_unit_pnl_after_costs: number | null;
  abstention_rate: number | null; days: number; ties: number;
}
export interface CohortVariant { weights: Record<string, number>; return_after_costs: number | null; capital_return: number | null }
export interface Cohort {
  session: string; horizon: Horizon; analyst: string; capital_fraction: number; state: 'COMPLETE' | 'PENDING';
  variants: { unhedged: CohortVariant; spy_hedged: CohortVariant };
}
export interface TraderScorecard {
  schema: 'trader-scorecard-v1'; synthetic: boolean; scores: Record<string, ScoreEntry>; hypothesis_state: string;
  portfolio_cohorts: Cohort[]; limitations: string[];
  multiple_testing: { attempted_runs: number; count: number; model_configuration_count: number };
}

export interface TraderAlerts { schema: 'trader-alerts-v1'; alerts: Array<{ schema: string; at: string; code: string; role: string | null }> }

export interface PriceContext {
  asset: string; adjustment: string; last_price_at: string; recent_closes: number[]; state: string;
  returns: Record<string, number>; vol_20d: number;
}
export interface TraderContext {
  schema: 'trader-context-view-v1'; date: string;
  context: null | {
    decision_time: string; price_cutoff: string; synthetic: boolean; universe: string[];
    prices: Record<string, PriceContext>; exclusions: unknown[]; limitations: string[];
    sources: Array<{ url: string; received_at: string; digest: string }>;
  };
}

export interface SeriesPoint { decision_time: string; price_at: string; close: number; adjustment: string | null; synthetic: boolean }
export interface TraderSeries { schema: 'trader-series-v1'; series: Record<string, SeriesPoint[]>; note: string }

export interface TraderHealth {
  schema: 'trader-health-view-v1';
  health: null | { schema: string; at: string; state: string; budget_counts: Record<string, number> };
  last_label: null | { at: string; state: string };
  paused: boolean;
}
