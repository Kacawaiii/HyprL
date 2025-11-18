//! Performance metrics calculation
//! 
//! This module provides implementations of trading performance metrics
//! including Sharpe ratio, Sortino ratio, Risk of Ruin, and bootstrap analysis.

use std::f64;

/// Calculate Sharpe ratio from returns
pub fn sharpe_ratio(returns: &[f64]) -> Option<f64> {
    if returns.len() < 2 {
        return None;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    
    if variance <= 0.0 {
        return None;
    }
    
    let std_dev = variance.sqrt();
    let sharpe = (mean / std_dev) * (returns.len() as f64).sqrt();
    
    Some(sharpe)
}

/// Calculate Sortino ratio (using downside deviation)
pub fn sortino_ratio(returns: &[f64], target_return: f64) -> Option<f64> {
    if returns.len() < 2 {
        return None;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    
    // Calculate downside deviation (only negative deviations from target)
    let downside_returns: Vec<f64> = returns.iter()
        .map(|&r| (r - target_return).min(0.0))
        .collect();
    
    let downside_variance = downside_returns.iter()
        .map(|&r| r.powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    
    if downside_variance <= 0.0 {
        return None;
    }
    
    let downside_dev = downside_variance.sqrt();
    let sortino = (mean - target_return) / downside_dev * (returns.len() as f64).sqrt();
    
    Some(sortino)
}

/// Calculate maximum drawdown from equity curve
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    
    let mut peak: f64 = equity_curve[0];
    let mut max_dd: f64 = 0.0;
    
    for &equity in equity_curve.iter() {
        peak = peak.max(equity);
        if peak > 0.0 {
            let dd: f64 = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }
    }
    
    max_dd
}

/// Calculate win rate from trade results
pub fn win_rate(trade_pnls: &[f64]) -> f64 {
    if trade_pnls.is_empty() {
        return 0.0;
    }
    
    let wins = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).count();
    wins as f64 / trade_pnls.len() as f64
}

/// Calculate profit factor (gross profit / gross loss)
pub fn profit_factor(trade_pnls: &[f64]) -> Option<f64> {
    let gross_profit: f64 = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).sum();
    let gross_loss: f64 = trade_pnls.iter().filter(|&&pnl| pnl < 0.0).map(|&pnl| -pnl).sum();
    
    if gross_loss == 0.0 {
        return None;
    }
    
    Some(gross_profit / gross_loss)
}

/// Calculate expectancy (average trade result)
pub fn expectancy(trade_pnls: &[f64]) -> f64 {
    if trade_pnls.is_empty() {
        return 0.0;
    }
    
    trade_pnls.iter().sum::<f64>() / trade_pnls.len() as f64
}

/// Risk of Ruin calculation using simplified formula
/// Assumes equal bet sizing
pub fn risk_of_ruin(trade_returns: &[f64], risk_pct: f64) -> f64 {
    if trade_returns.is_empty() || risk_pct <= 0.0 || risk_pct >= 1.0 {
        return 0.0;
    }
    
    let win_rate = win_rate(trade_returns);
    let avg_win: f64 = trade_returns.iter()
        .filter(|&&r| r > 0.0)
        .sum::<f64>() / trade_returns.iter().filter(|&&r| r > 0.0).count().max(1) as f64;
    let avg_loss: f64 = trade_returns.iter()
        .filter(|&&r| r < 0.0)
        .sum::<f64>() / trade_returns.iter().filter(|&&r| r < 0.0).count().max(1) as f64;
    
    if avg_loss >= 0.0 || avg_win <= 0.0 {
        return 0.0;
    }
    
    let avg_win_loss_ratio = avg_win / avg_loss.abs();
    
    // Simplified RoR formula
    if win_rate >= 0.5 && avg_win_loss_ratio >= 1.0 {
        return 0.0; // Positive expectancy
    }
    
    // Basic RoR approximation
    let ror = ((1.0 - win_rate) / win_rate).powf(risk_pct * 100.0);
    ror.min(1.0)
}

/// Bootstrap percentile calculation
#[allow(dead_code)]
pub fn bootstrap_percentiles(values: &[f64], n_samples: usize, percentiles: &[f64]) -> Vec<Vec<f64>> {
    if values.is_empty() || n_samples == 0 {
        return vec![vec![]; percentiles.len()];
    }
    
    // Simple bootstrap: resample with replacement
    let mut bootstrap_means = Vec::with_capacity(n_samples);
    let len = values.len();
    
    // Use a simple pseudo-random generator (not cryptographically secure)
    let mut seed = 42u64;
    
    for _ in 0..n_samples {
        let mut sample_sum = 0.0;
        for _ in 0..len {
            // Simple LCG random number generator
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let idx = (seed % len as u64) as usize;
            sample_sum += values[idx];
        }
        bootstrap_means.push(sample_sum / len as f64);
    }
    
    // Sort bootstrap means
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate percentiles
    let mut result = Vec::with_capacity(percentiles.len());
    for &p in percentiles {
        let idx = ((p / 100.0) * (n_samples - 1) as f64) as usize;
        result.push(vec![bootstrap_means[idx.min(n_samples - 1)]]);
    }
    
    result
}

/// Calculate robustness score based on multiple metrics
pub fn robustness_score(
    sharpe: Option<f64>,
    profit_factor: Option<f64>,
    win_rate: f64,
    max_dd: f64,
) -> f64 {
    let sharpe_score = sharpe.unwrap_or(0.0).max(0.0).min(3.0) / 3.0;
    let pf_score = (profit_factor.unwrap_or(1.0).max(1.0).min(3.0) - 1.0) / 2.0;
    let wr_score = win_rate.max(0.0).min(1.0);
    let dd_score = 1.0 - max_dd.max(0.0).min(1.0);
    
    // Weighted average
    sharpe_score * 0.3 + pf_score * 0.3 + wr_score * 0.2 + dd_score * 0.2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sharpe = sharpe_ratio(&returns);
        assert!(sharpe.is_some());
        assert!(sharpe.unwrap() > 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sortino = sortino_ratio(&returns, 0.0);
        assert!(sortino.is_some());
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 100.0];
        let dd = max_drawdown(&equity);
        assert!(dd > 0.0);
        assert!(dd < 0.2); // Should be around 16.67%
    }

    #[test]
    fn test_win_rate() {
        let trades = vec![10.0, -5.0, 15.0, -3.0, 20.0];
        let wr = win_rate(&trades);
        assert_eq!(wr, 0.6); // 3 wins out of 5
    }

    #[test]
    fn test_profit_factor() {
        let trades = vec![10.0, -5.0, 15.0, -3.0];
        let pf = profit_factor(&trades);
        assert!(pf.is_some());
        assert!(pf.unwrap() > 1.0); // Profitable
    }
}
