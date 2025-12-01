//! Batch evaluation of multiple strategy configs on the same dataset.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::backtest::run_backtest;
use crate::core::{BacktestConfig, BacktestReport, Candle};

/// Evaluate a batch of configs on a single asset.
/// Suitable for being called from HyprL's supersearch engine.
pub fn evaluate_batch(
    candles: &[Candle],
    signal: &[f64],
    configs: &[BacktestConfig],
) -> Vec<BacktestReport> {
    configs
        .par_iter()
        .map(|cfg| run_backtest(candles, signal, cfg))
        .collect()
}

/// Hard constraints mirroring HyprL's Python search engine filters.
#[derive(Debug, Clone)]
pub struct SearchConstraint {
    pub min_trades: usize,
    pub min_profit_factor: f64,
    pub min_sharpe: f64,
    pub max_drawdown: f64,
    pub max_risk_of_ruin: f64,
    pub min_expectancy: f64,
    pub min_robustness: f64,
    pub max_maxdd_p95: f64,
    pub min_pnl_p05: f64,
    pub min_pnl_p50: f64,
    pub min_pnl_p95: f64,
}

impl Default for SearchConstraint {
    fn default() -> Self {
        Self {
            min_trades: 50,
            min_profit_factor: 1.2,
            min_sharpe: 0.8,
            max_drawdown: 0.35,
            max_risk_of_ruin: 0.1,
            min_expectancy: 0.0,
            min_robustness: 0.0,
            max_maxdd_p95: 0.35,
            min_pnl_p05: f64::NEG_INFINITY,
            min_pnl_p50: f64::NEG_INFINITY,
            min_pnl_p95: f64::NEG_INFINITY,
        }
    }
}

impl SearchConstraint {
    pub fn passes(&self, report: &BacktestReport) -> bool {
        let metrics = &report.metrics;
        if report.n_trades < self.min_trades {
            return false;
        }
        if !metrics.profit_factor.is_finite()
            || metrics.profit_factor < self.min_profit_factor
        {
            return false;
        }
        if self.min_sharpe.is_finite() {
            if !metrics.sharpe.is_finite() || metrics.sharpe < self.min_sharpe {
                return false;
            }
        }
        if self.max_drawdown.is_finite() && self.max_drawdown > 0.0 {
            if metrics.max_drawdown.abs() > self.max_drawdown {
                return false;
            }
        }
        if self.max_maxdd_p95.is_finite() && self.max_maxdd_p95 > 0.0 {
            if metrics.maxdd_p95 > self.max_maxdd_p95 {
                return false;
            }
        }
        if !metrics.risk_of_ruin.is_finite() || metrics.risk_of_ruin > self.max_risk_of_ruin {
            return false;
        }
        if metrics.expectancy < self.min_expectancy {
            return false;
        }
        if self.min_robustness.is_finite() && metrics.robustness_score < self.min_robustness {
            return false;
        }
        if self.min_pnl_p05.is_finite() && metrics.pnl_p05 < self.min_pnl_p05 {
            return false;
        }
        if self.min_pnl_p50.is_finite() && metrics.pnl_p50 < self.min_pnl_p50 {
            return false;
        }
        if self.min_pnl_p95.is_finite() && metrics.pnl_p95 < self.min_pnl_p95 {
            return false;
        }
        true
    }
}

fn score_tuple(report: &BacktestReport) -> [f64; 4] {
    let metrics = &report.metrics;
    let pf = if metrics.profit_factor.is_finite() {
        metrics.profit_factor
    } else {
        0.0
    };
    let sharpe = if metrics.sharpe.is_finite() {
        metrics.sharpe
    } else {
        f64::NEG_INFINITY
    };
    let ror = if metrics.risk_of_ruin.is_finite() {
        metrics.risk_of_ruin
    } else {
        1.0
    };
    let dd = metrics.max_drawdown.abs().max(metrics.maxdd_p95.abs());
    let expectancy = metrics.expectancy;
    let robustness = if metrics.robustness_score.is_finite() {
        metrics.robustness_score
    } else {
        0.0
    };
    [
        -pf + ror * 2.0,
        -sharpe + ror,
        dd + ror * 100.0 - expectancy * 100.0,
        -robustness,
    ]
}

fn compare_scores(lhs: &[f64; 4], rhs: &[f64; 4]) -> Ordering {
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        match l.partial_cmp(r) {
            Some(Ordering::Less) => return Ordering::Less,
            Some(Ordering::Greater) => return Ordering::Greater,
            _ => continue,
        }
    }
    Ordering::Equal
}

/// Native grid-search runner with hard constraints.
pub fn run_native_search(
    candles: &[Candle],
    signal: &[f64],
    configs: &[BacktestConfig],
    constraints: &SearchConstraint,
    top_k: usize,
) -> Vec<BacktestReport> {
    if candles.is_empty() || signal.is_empty() || configs.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let reports: Vec<_> = configs
        .par_iter()
        .map(|cfg| run_backtest(candles, signal, cfg))
        .collect();

    let mut survivors: Vec<_> = reports
        .into_iter()
        .filter(|report| constraints.passes(report))
        .collect();

    survivors.sort_by(|a, b| {
        let sa = score_tuple(a);
        let sb = score_tuple(b);
        compare_scores(&sa, &sb)
    });

    if survivors.len() > top_k {
        survivors.truncate(top_k);
    }
    survivors
}
