//! Risk / performance metrics.
//!
//! These functions keep the Rust supercalc metrics aligned with HyprL's
//! Python Risk Layer conventions. The implementations intentionally mirror
//! the formulas in `hyprl.risk.metrics` (risk of ruin, bootstrap summaries)
//! and `_sortino_ratio` inside the Python backtest runner so that downstream
//! consumers receive numerically comparable values.

use std::cmp::Ordering;

use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::core::{EquityPoint, PerformanceMetrics};

const MS_PER_YEAR: f64 = 365.25_f64 * 24.0 * 3600.0 * 1000.0;
const DEFAULT_BOOTSTRAP_RUNS: usize = 512;
const DEFAULT_BOOTSTRAP_SEED: u64 = 42;
const MIN_RISK_PCT: f64 = 1e-6;

/// Summary of bootstrap simulations (mirrors `BootstrapSummary` on the Python side).
#[derive(Debug, Clone, Copy)]
pub struct BootstrapStats {
    pub maxdd_p05: f64,
    pub maxdd_p95: f64,
    pub maxdd_p99: f64,
    pub pnl_p05: f64,
    pub pnl_p50: f64,
    pub pnl_p95: f64,
}

impl Default for BootstrapStats {
    fn default() -> Self {
        Self {
            maxdd_p05: 0.0,
            maxdd_p95: 0.0,
            maxdd_p99: 0.0,
            pnl_p05: 0.0,
            pnl_p50: 0.0,
            pnl_p95: 0.0,
        }
    }
}

/// Convert equity levels into simple returns.
pub fn equity_to_returns(equity: &[EquityPoint]) -> Vec<f64> {
    if equity.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(equity.len() - 1);
    for i in 1..equity.len() {
        let prev = equity[i - 1].equity;
        let curr = equity[i].equity;
        if prev > 0.0 {
            out.push(curr / prev - 1.0);
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Total return = last / first - 1.
pub fn total_return(equity: &[EquityPoint]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let first = equity.first().unwrap().equity;
    let last = equity.last().unwrap().equity;
    if first > 0.0 {
        last / first - 1.0
    } else {
        0.0
    }
}

/// CAGR derived from timestamps (milliseconds since epoch).
pub fn cagr(equity: &[EquityPoint]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let first = equity.first().unwrap();
    let last = equity.last().unwrap();
    let dt_ms = (last.ts - first.ts) as f64;
    if dt_ms <= 0.0 {
        return 0.0;
    }
    let years = dt_ms / MS_PER_YEAR;
    if years <= 0.0 {
        return 0.0;
    }
    let tr = total_return(equity);
    if tr <= -1.0 {
        return -1.0;
    }
    (1.0 + tr).powf(1.0 / years) - 1.0
}

fn duration_years(equity: &[EquityPoint]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let first = equity.first().unwrap().ts as f64;
    let last = equity.last().unwrap().ts as f64;
    let dt = last - first;
    if dt <= 0.0 {
        0.0
    } else {
        dt / MS_PER_YEAR
    }
}

fn bars_per_year(equity: &[EquityPoint]) -> f64 {
    let years = duration_years(equity);
    if years <= 0.0 {
        0.0
    } else {
        (equity.len().saturating_sub(1)) as f64 / years
    }
}

/// Maximum drawdown (negative number) and its duration (in bars).
pub fn max_drawdown(equity: &[EquityPoint]) -> (f64, i64) {
    if equity.is_empty() {
        return (0.0, 0);
    }

    let mut peak = equity[0].equity;
    let mut max_dd = 0.0_f64;
    let mut peak_index = 0_i64;
    let mut max_duration = 0_i64;

    for (idx, point) in equity.iter().enumerate() {
        if point.equity > peak {
            peak = point.equity;
            peak_index = idx as i64;
        }
        if peak > 0.0 {
            let dd = (point.equity / peak) - 1.0;
            if dd < max_dd {
                max_dd = dd;
                max_duration = (idx as i64) - peak_index;
            }
        }
    }

    (max_dd, max_duration)
}

/// Simple (non-annualized) Sharpe ratio.
pub fn sharpe_ratio(returns: &[f64], risk_free: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &r in returns {
        let excess = r - risk_free;
        sum += excess;
        sum_sq += excess * excess;
    }
    let n = returns.len() as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    let std = if var <= 0.0 { 0.0 } else { var.sqrt() };
    if std > 0.0 {
        mean / std
    } else {
        0.0
    }
}

fn sortino_ratio_from_returns(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.len() < 2 || bars_per_year <= 0.0 {
        return f64::NAN;
    }
    let target = 0.0;
    let mut excess = Vec::with_capacity(returns.len());
    for &value in returns {
        excess.push(value - target);
    }
    let mean_excess = excess.iter().sum::<f64>() / excess.len() as f64;
    let downside: Vec<f64> = excess.into_iter().filter(|v| *v < 0.0).collect();
    if downside.len() < 2 {
        return f64::NAN;
    }
    let down_mean = downside.iter().sum::<f64>() / downside.len() as f64;
    let variance = downside
        .iter()
        .map(|v| {
            let delta = *v - down_mean;
            delta * delta
        })
        .sum::<f64>()
        / ((downside.len() - 1) as f64);
    if variance <= 0.0 {
        return f64::NAN;
    }
    let downside_std = variance.sqrt();
    let annualized_return = mean_excess * bars_per_year;
    let annualized_downside = downside_std * bars_per_year.sqrt();
    if annualized_downside == 0.0 {
        f64::NAN
    } else {
        annualized_return / annualized_downside
    }
}

/// Annualized Sortino ratio derived from the equity curve.
pub fn sortino_ratio(equity: &[EquityPoint]) -> f64 {
    let returns = equity_to_returns(equity);
    let bars = bars_per_year(equity);
    sortino_ratio_from_returns(&returns, bars)
}

/// Profit factor = gross_profit / |gross_loss|.
pub fn profit_factor(returns: &[f64]) -> f64 {
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    for &r in returns {
        if r > 0.0 {
            gross_profit += r;
        } else if r < 0.0 {
            gross_loss += r;
        }
    }

    if gross_loss == 0.0 {
        if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        gross_profit / gross_loss.abs()
    }
}

/// Win rate and expectancy computed from per-bar returns.
pub fn win_rate_and_expectancy(returns: &[f64]) -> (f64, f64) {
    if returns.is_empty() {
        return (0.0, 0.0);
    }

    let mut wins = 0.0;
    let mut total = 0.0;

    for &r in returns {
        total += r;
        if r > 0.0 {
            wins += 1.0;
        }
    }

    let n = returns.len() as f64;
    let win_rate = if n > 0.0 { wins / n } else { 0.0 };
    let expectancy = if n > 0.0 { total / n } else { 0.0 };
    (win_rate, expectancy)
}

fn safe_ratio(num: f64, denom: f64) -> f64 {
    if !num.is_finite() || !denom.is_finite() || denom.abs() <= f64::EPSILON {
        f64::NAN
    } else {
        num / denom
    }
}

fn ratio_component(ratio: f64) -> f64 {
    if !ratio.is_finite() {
        0.5
    } else {
        (ratio.clamp(0.0, 2.0)) / 2.0
    }
}

fn inverse_component(ratio: f64) -> f64 {
    if !ratio.is_finite() {
        0.5
    } else {
        (2.0 - ratio.clamp(0.0, 2.0)).clamp(0.0, 1.0)
    }
}

fn win_component(delta: f64) -> f64 {
    let scaled = ((delta + 0.2) / 0.4).clamp(0.0, 1.0);
    if !scaled.is_finite() {
        0.5
    } else {
        scaled
    }
}

/// Approximate risk of ruin following the Python risk layer conventions.
pub fn risk_of_ruin(trade_returns: &[f64], risk_pct: f64) -> f64 {
    if trade_returns.is_empty() {
        return 1.0;
    }
    let mut wins = 0.0;
    let mut total = 0.0;
    let mut win_sum = 0.0;
    let mut loss_sum = 0.0;
    let mut win_count = 0.0;
    let mut loss_count = 0.0;

    for &r in trade_returns {
        total += 1.0;
        if r > 0.0 {
            wins += 1.0;
            win_sum += r;
            win_count += 1.0;
        } else if r < 0.0 {
            loss_sum += r;
            loss_count += 1.0;
        }
    }

    if total == 0.0 {
        return 1.0;
    }
    let p = wins / total;
    let q = 1.0 - p;
    let avg_win = if win_count > 0.0 { win_sum / win_count } else { 0.0 };
    let avg_loss = if loss_count > 0.0 { loss_sum / loss_count } else { 0.0 };

    if risk_pct <= 0.0 || !risk_pct.is_finite() {
        return 1.0;
    }

    let mut ratio = if avg_loss >= 0.0 {
        if avg_win > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else if avg_win <= 0.0 {
        0.0
    } else {
        avg_win / avg_loss.abs()
    };

    if ratio <= 0.0 || p <= 0.0 {
        return 1.0;
    }

    if !ratio.is_finite() {
        if avg_win > 0.0 && avg_loss == 0.0 {
            return 0.0;
        }
        ratio = 1e6;
    }

    if avg_loss == 0.0 && avg_win > 0.0 {
        return 0.0;
    }

    let edge = p * ratio - q;
    if edge <= 0.0 {
        return 1.0;
    }

    let risk_per_trade = risk_pct.max(MIN_RISK_PCT);
    let capital_units = (1.0 / risk_per_trade).max(1.0);
    let mut base = q / (p * ratio);
    if !base.is_finite() || base <= 0.0 {
        base = 0.0;
    }
    base = base.clamp(0.0, 0.999_999);
    base.powf(capital_units).clamp(0.0, 1.0)
}

fn quantile(mut data: Vec<f64>, q: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    });
    let n = data.len();
    if n == 1 {
        return data[0];
    }
    let clamped_q = q.clamp(0.0, 1.0);
    let pos = clamped_q * ((n - 1) as f64);
    let idx = pos.floor() as usize;
    let frac = pos - (idx as f64);
    if idx + 1 < n {
        data[idx] + (data[idx + 1] - data[idx]) * frac
    } else {
        data[idx]
    }
}

/// Bootstrap Monte Carlo metrics consistent with the Python implementation.
pub fn bootstrap_metrics(trade_returns: &[f64], n_runs: usize) -> BootstrapStats {
    if trade_returns.is_empty() || n_runs == 0 {
        return BootstrapStats::default();
    }
    let mut rng = Pcg64::seed_from_u64(DEFAULT_BOOTSTRAP_SEED);
    let dist = Uniform::new(0, trade_returns.len());
    let mut maxdds = Vec::with_capacity(n_runs);
    let mut pnl = Vec::with_capacity(n_runs);

    for _ in 0..n_runs {
        let mut equity = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        for _ in 0..trade_returns.len() {
            let idx = rng.sample(dist);
            let r = trade_returns[idx];
            equity *= 1.0 + r;
            if equity <= 0.0 {
                equity = MIN_RISK_PCT;
            }
            if equity > peak {
                peak = equity;
            }
            if peak > 0.0 {
                let dd = 1.0 - (equity / peak);
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }
        maxdds.push(max_dd);
        pnl.push(equity - 1.0);
    }

    BootstrapStats {
        maxdd_p05: quantile(maxdds.clone(), 0.05),
        maxdd_p95: quantile(maxdds.clone(), 0.95),
        maxdd_p99: quantile(maxdds, 0.99),
        pnl_p05: quantile(pnl.clone(), 0.05),
        pnl_p50: quantile(pnl.clone(), 0.5),
        pnl_p95: quantile(pnl, 0.95),
    }
}

/// Composite robustness score emulating `compute_robustness_score` in Python.
pub fn robustness_score(metrics: &PerformanceMetrics, boot: &BootstrapStats) -> f64 {
    let pf_ratio = safe_ratio(1.0 + boot.pnl_p05, 1.0 + metrics.total_return);
    let sharpe_ratio = if metrics.sharpe.is_finite() && metrics.sharpe.abs() > f64::EPSILON {
        safe_ratio(metrics.sharpe.abs(), metrics.sharpe.abs())
    } else {
        f64::NAN
    };
    let dd_ratio = safe_ratio(boot.maxdd_p95.abs(), metrics.max_drawdown.abs().max(1e-9));
    let pnl_spread = (boot.pnl_p95 - boot.pnl_p05).abs();
    let equity_vol_ratio = if metrics.expectancy.abs() > 1e-9 {
        safe_ratio(pnl_spread, metrics.expectancy.abs())
    } else {
        f64::NAN
    };
    let winrate_delta = (metrics.win_rate - 0.5).clamp(-0.2, 0.2);

    let pf_component = ratio_component(pf_ratio);
    let sharpe_component = ratio_component(sharpe_ratio);
    let dd_component = inverse_component(dd_ratio);
    let vol_component = inverse_component(equity_vol_ratio);
    let win_component = win_component(winrate_delta);

    (0.3 * pf_component
        + 0.3 * sharpe_component
        + 0.2 * dd_component
        + 0.1 * vol_component
        + 0.1 * win_component)
        .clamp(0.0, 1.0)
}

/// Build full PerformanceMetrics from an equity curve and strategy risk settings.
pub fn metrics_from_equity(equity: &[EquityPoint], risk_pct: f64) -> PerformanceMetrics {
    let returns = equity_to_returns(equity);
    let tr = total_return(equity);
    let cagr_val = cagr(equity);
    let (max_dd, max_dd_duration) = max_drawdown(equity);
    let sharpe = sharpe_ratio(&returns, 0.0);
    let sortino = sortino_ratio(equity);
    let pf = profit_factor(&returns);
    let (win_rate, expectancy) = win_rate_and_expectancy(&returns);
    let ror = risk_of_ruin(&returns, if risk_pct.is_finite() { risk_pct } else { MIN_RISK_PCT });
    let boot = bootstrap_metrics(&returns, DEFAULT_BOOTSTRAP_RUNS);

    let mut metrics = PerformanceMetrics {
        total_return: tr,
        cagr: cagr_val,
        sharpe,
        sortino,
        calmar: if max_dd < 0.0 {
            cagr_val / max_dd.abs()
        } else {
            0.0
        },
        max_drawdown: max_dd,
        max_drawdown_duration: max_dd_duration,
        profit_factor: pf,
        win_rate,
        expectancy,
        risk_of_ruin: ror,
        maxdd_p05: boot.maxdd_p05,
        maxdd_p95: boot.maxdd_p95,
        pnl_p05: boot.pnl_p05,
        pnl_p50: boot.pnl_p50,
        pnl_p95: boot.pnl_p95,
        robustness_score: 0.0,
    };

    metrics.robustness_score = robustness_score(&metrics, &boot);
    metrics
}
