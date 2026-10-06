export interface Reliability {
  state: 'AVAILABLE' | 'REFUSED_SMALL_SAMPLE';
  method: string;
  count: number;
  bins: { lower: number; upper: number; count: number; mean_probability: number | null; observed_frequency: number | null }[];
  brier: number | null;
  binned_brier: number | null;
  reliability: number | null;
  resolution: number | null;
  uncertainty: number | null;
  binning_residual: number | null;
}

export interface PolicyCalibration {
  product: string;
  synthetic: true;
  raw: Reliability;
  calibrated: Reliability;
  sample: {
    prediction_hash: string; decision_at: string; origin: string; method: string;
    raw_probability: number; calibrated_probability: number; event: string;
    horizon_seconds: number; calibration_hash: string; artifact_hash: string; fold_index: number;
  };
  folds: {
    calibration_hash: string;
    artifact: {
      method: string; fitted_at: string; validation_start: string;
      train: { count: number; first: string; last: string; last_label_available_at: string; population_hash: string; class_counts: number[] };
    };
    test: { first: string; last: string; count: number; population_hash: string };
    raw: Reliability; calibrated: Reliability;
  }[];
}

export interface ProtectionFill {
  reason: string; reference_price: string; fill_price: string; available_at: string;
  gross_pnl: string; net_pnl: string; fees: string; slippage_cost: string; net_return: string;
  clock_method: string;
}

export interface ProtectionSimulation {
  scenario: string; state: 'CLOSED' | 'PENDING' | 'NOT_OBSERVED'; ambiguous: boolean;
  result_hash: string; plan_hash: string; policy_hash: string; cost_hash: string;
  cost_method: string; gap_method: string; ambiguity_method: string; missing_bar_method: string;
  as_of: string; bars_observed: number;
  exit: ProtectionFill | null; target_first_sensitivity: ProtectionFill | null;
  plan: {
    product: string; side: string; mode: string; entry_at: string; entry_fill: string;
    horizon_seconds: number; source_prediction_hash: string;
    levels: Record<'take_profit' | 'stop_loss', {
      value: string; origin: 'STRATEGY' | 'POLICY'; method: string; source_hash: string; provided_at: string;
    }>;
  };
}

export interface PolicyReport {
  state: 'AVAILABLE' | 'NOT_CONFIGURED'; policy_hash: string; identity: string | null; selection_hash?: string;
  report: null | { synthetic: true; calibrations: PolicyCalibration[]; simulations: ProtectionSimulation[]; limitations: string[] };
}

export interface PolicyDefinitions {
  policy_hash: string; real_calibration: string;
  spec: {
    revision: number;
    calibration: { method: string; min_train: number; min_class: number; min_evaluation: number; min_evaluation_class: number };
    risk: { method: string; gap: string; intrabar: string; missing_bars: string };
  };
}
