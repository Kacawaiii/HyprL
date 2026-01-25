//! Multi-symbol and multi-timeframe backtesting.
//!
//! Optimized for processing millions of bars across multiple assets.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::backtest::run_backtest;
use crate::core::{BacktestConfig, BacktestReport, Candle};
use crate::data::{load_parquet, DataConfig};

/// Result of a multi-symbol backtest.
#[derive(Debug, Clone)]
pub struct MultiSymbolReport {
    /// Individual reports per symbol.
    pub reports: HashMap<String, BacktestReport>,
    /// Aggregated metrics.
    pub aggregate: AggregateMetrics,
    /// Total processing time in milliseconds.
    pub processing_time_ms: u64,
}

/// Aggregated metrics across multiple symbols.
#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    pub total_trades: usize,
    pub avg_sharpe: f64,
    pub avg_profit_factor: f64,
    pub avg_win_rate: f64,
    pub worst_drawdown: f64,
    pub best_return: f64,
    pub worst_return: f64,
    pub total_return: f64,
}

/// Run backtests on multiple symbols in parallel.
pub fn run_multi_symbol_backtest(
    data: &HashMap<String, Vec<Candle>>,
    signals: &HashMap<String, Vec<f64>>,
    config: &BacktestConfig,
) -> MultiSymbolReport {
    let start = std::time::Instant::now();

    let reports: HashMap<String, BacktestReport> = data
        .par_iter()
        .filter_map(|(symbol, candles)| {
            signals.get(symbol).map(|signal| {
                let report = run_backtest(candles, signal, config);
                (symbol.clone(), report)
            })
        })
        .collect();

    let aggregate = compute_aggregate_metrics(&reports);
    let processing_time_ms = start.elapsed().as_millis() as u64;

    MultiSymbolReport {
        reports,
        aggregate,
        processing_time_ms,
    }
}

/// Load and backtest multiple Parquet files.
pub fn run_multi_file_backtest(
    paths: &[&Path],
    signal_generator: impl Fn(&[Candle]) -> Vec<f64> + Sync,
    config: &BacktestConfig,
    data_config: &DataConfig,
) -> MultiSymbolReport {
    let start = std::time::Instant::now();

    let reports: HashMap<String, BacktestReport> = paths
        .par_iter()
        .filter_map(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            match load_parquet(path, data_config) {
                Ok(candles) => {
                    let signal = signal_generator(&candles);
                    let report = run_backtest(&candles, &signal, config);
                    Some((name, report))
                }
                Err(e) => {
                    eprintln!("Error loading {:?}: {}", path, e);
                    None
                }
            }
        })
        .collect();

    let aggregate = compute_aggregate_metrics(&reports);
    let processing_time_ms = start.elapsed().as_millis() as u64;

    MultiSymbolReport {
        reports,
        aggregate,
        processing_time_ms,
    }
}

/// Grid search across multiple symbols.
pub struct MultiSymbolGridSearch {
    pub data: HashMap<String, Vec<Candle>>,
    pub signals: HashMap<String, Vec<f64>>,
}

impl MultiSymbolGridSearch {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            signals: HashMap::new(),
        }
    }

    pub fn add_symbol(&mut self, name: String, candles: Vec<Candle>, signal: Vec<f64>) {
        self.data.insert(name.clone(), candles);
        self.signals.insert(name, signal);
    }

    /// Run grid search and return best configs per symbol.
    pub fn run_grid_search(
        &self,
        configs: &[BacktestConfig],
        min_profit_factor: f64,
        min_sharpe: f64,
    ) -> HashMap<String, Vec<(BacktestConfig, BacktestReport)>> {
        self.data
            .par_iter()
            .filter_map(|(symbol, candles)| {
                self.signals.get(symbol).map(|signal| {
                    let results: Vec<_> = configs
                        .iter()
                        .map(|cfg| {
                            let report = run_backtest(candles, signal, cfg);
                            (cfg.clone(), report)
                        })
                        .filter(|(_, report)| {
                            report.metrics.profit_factor >= min_profit_factor
                                && report.metrics.sharpe >= min_sharpe
                        })
                        .collect();
                    (symbol.clone(), results)
                })
            })
            .collect()
    }

    /// Find configs that work well across ALL symbols.
    pub fn find_robust_configs(
        &self,
        configs: &[BacktestConfig],
        min_profit_factor: f64,
        min_sharpe: f64,
        min_symbols_passing: usize,
    ) -> Vec<(BacktestConfig, f64)> {
        let _symbol_count = self.data.len();

        configs
            .par_iter()
            .filter_map(|cfg| {
                let results: Vec<_> = self
                    .data
                    .iter()
                    .filter_map(|(symbol, candles)| {
                        self.signals.get(symbol).map(|signal| {
                            run_backtest(candles, signal, cfg)
                        })
                    })
                    .collect();

                let passing: Vec<_> = results
                    .iter()
                    .filter(|r| {
                        r.metrics.profit_factor >= min_profit_factor
                            && r.metrics.sharpe >= min_sharpe
                    })
                    .collect();

                if passing.len() >= min_symbols_passing {
                    let avg_sharpe: f64 =
                        passing.iter().map(|r| r.metrics.sharpe).sum::<f64>() / passing.len() as f64;
                    Some((cfg.clone(), avg_sharpe))
                } else {
                    None
                }
            })
            .collect()
    }
}

fn compute_aggregate_metrics(reports: &HashMap<String, BacktestReport>) -> AggregateMetrics {
    if reports.is_empty() {
        return AggregateMetrics::default();
    }

    let n = reports.len() as f64;
    let mut agg = AggregateMetrics::default();

    let mut sharpe_sum = 0.0;
    let mut pf_sum = 0.0;
    let mut wr_sum = 0.0;

    for report in reports.values() {
        agg.total_trades += report.n_trades;

        let m = &report.metrics;
        sharpe_sum += m.sharpe;
        pf_sum += m.profit_factor;
        wr_sum += m.win_rate;

        if m.max_drawdown.abs() > agg.worst_drawdown.abs() {
            agg.worst_drawdown = m.max_drawdown;
        }
        if m.total_return > agg.best_return {
            agg.best_return = m.total_return;
        }
        if m.total_return < agg.worst_return || agg.worst_return == 0.0 {
            agg.worst_return = m.total_return;
        }
        agg.total_return += m.total_return;
    }

    agg.avg_sharpe = sharpe_sum / n;
    agg.avg_profit_factor = pf_sum / n;
    agg.avg_win_rate = wr_sum / n;

    agg
}

/// Walk-forward optimization across multiple symbols.
pub struct WalkForwardOptimizer {
    pub train_ratio: f64,
    pub n_folds: usize,
}

impl Default for WalkForwardOptimizer {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            n_folds: 5,
        }
    }
}

impl WalkForwardOptimizer {
    /// Run walk-forward optimization on a single symbol.
    pub fn optimize_single(
        &self,
        candles: &[Candle],
        signal: &[f64],
        configs: &[BacktestConfig],
    ) -> Vec<WalkForwardResult> {
        let n = candles.len();
        let fold_size = n / self.n_folds;

        (0..self.n_folds)
            .into_par_iter()
            .map(|fold| {
                let test_start = fold * fold_size;
                let test_end = if fold == self.n_folds - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                // Training data: everything before test period
                let train_end = test_start;
                let train_start = if train_end > fold_size {
                    train_end - fold_size
                } else {
                    0
                };

                if train_end <= train_start || test_end <= test_start {
                    return WalkForwardResult {
                        fold,
                        best_config: configs[0].clone(),
                        train_sharpe: 0.0,
                        test_sharpe: 0.0,
                        test_return: 0.0,
                        test_trades: 0,
                    };
                }

                let train_candles = &candles[train_start..train_end];
                let train_signal = &signal[train_start..train_end];

                // Find best config on training data
                let best_result = configs
                    .iter()
                    .map(|cfg| {
                        let report = run_backtest(train_candles, train_signal, cfg);
                        (cfg.clone(), report)
                    })
                    .max_by(|a, b| {
                        a.1.metrics
                            .sharpe
                            .partial_cmp(&b.1.metrics.sharpe)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                let (best_config, train_report) = match best_result {
                    Some(r) => r,
                    None => return WalkForwardResult {
                        fold,
                        best_config: configs[0].clone(),
                        train_sharpe: 0.0,
                        test_sharpe: 0.0,
                        test_return: 0.0,
                        test_trades: 0,
                    },
                };

                // Test on out-of-sample data
                let test_candles = &candles[test_start..test_end];
                let test_signal = &signal[test_start..test_end];
                let test_report = run_backtest(test_candles, test_signal, &best_config);

                WalkForwardResult {
                    fold,
                    best_config,
                    train_sharpe: train_report.metrics.sharpe,
                    test_sharpe: test_report.metrics.sharpe,
                    test_return: test_report.metrics.total_return,
                    test_trades: test_report.n_trades,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    pub fold: usize,
    pub best_config: BacktestConfig,
    pub train_sharpe: f64,
    pub test_sharpe: f64,
    pub test_return: f64,
    pub test_trades: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                ts: i as i64 * 1000,
                open: 100.0 + (i as f64 * 0.01),
                high: 101.0 + (i as f64 * 0.01),
                low: 99.0 + (i as f64 * 0.01),
                close: 100.5 + (i as f64 * 0.01),
                volume: 1000.0,
            })
            .collect()
    }

    fn make_config() -> BacktestConfig {
        BacktestConfig {
            risk_pct: 0.01,
            commission_pct: 0.001,
            slippage_pct: 0.001,
            max_leverage: 1.0,
            params: vec![],
            allow_short: true,
            label: None,
            atr_window: 14,
            atr_mult_stop: 2.0,
            atr_mult_tp: 4.0,
            use_atr_position_sizing: true,
            trailing_activation_r: 1.0,
            trailing_distance_r: 0.5,
        }
    }

    #[test]
    fn test_multi_symbol_backtest() {
        let mut data = HashMap::new();
        let mut signals = HashMap::new();

        for sym in ["EURUSD", "GBPUSD", "USDJPY"] {
            let candles = make_candles(1000);
            let signal: Vec<f64> = candles
                .iter()
                .enumerate()
                .map(|(i, _)| if i % 10 < 5 { 1.0 } else { -1.0 })
                .collect();
            data.insert(sym.to_string(), candles);
            signals.insert(sym.to_string(), signal);
        }

        let config = make_config();
        let report = run_multi_symbol_backtest(&data, &signals, &config);

        assert_eq!(report.reports.len(), 3);
        assert!(report.aggregate.total_trades > 0);
    }

    #[test]
    fn test_walk_forward() {
        let candles = make_candles(1000);
        let signal: Vec<f64> = candles
            .iter()
            .enumerate()
            .map(|(i, _)| if i % 10 < 5 { 1.0 } else { 0.0 })
            .collect();

        let configs = vec![make_config()];
        let optimizer = WalkForwardOptimizer {
            train_ratio: 0.7,
            n_folds: 3,
        };

        let results = optimizer.optimize_single(&candles, &signal, &configs);
        assert_eq!(results.len(), 3);
    }
}
