//! Backtest engine implementation
//!
//! This module provides a native Rust backtest engine with ATR-based position sizing,
//! stop-loss/take-profit management, and comprehensive trade simulation.

use std::f64;

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Long,
    Short,
}

impl Direction {
    pub fn from_str(s: &str) -> Option<Direction> {
        match s.to_lowercase().as_str() {
            "long" => Some(Direction::Long),
            "short" => Some(Direction::Short),
            _ => None,
        }
    }
    
    pub fn to_str(&self) -> &'static str {
        match self {
            Direction::Long => "long",
            Direction::Short => "short",
        }
    }
}

/// Risk configuration for position sizing
#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub balance: f64,
    pub risk_pct: f64,
    pub atr_multiplier: f64,
    pub reward_multiple: f64,
    pub min_position_size: i64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        RiskConfig {
            balance: 100_000.0,
            risk_pct: 0.01,
            atr_multiplier: 2.0,
            reward_multiple: 2.0,
            min_position_size: 1,
        }
    }
}

/// Trade plan with entry, stop, and target prices
#[derive(Debug, Clone)]
pub struct RiskOutcome {
    pub direction: Direction,
    pub entry_price: f64,
    pub position_size: i64,
    pub stop_price: f64,
    pub take_profit_price: f64,
    pub risk_amount: f64,
    pub rr_multiple: f64,
}

/// Compute stop loss price based on ATR
pub fn compute_stop_price(
    entry_price: f64,
    atr: f64,
    direction: Direction,
    atr_multiplier: f64,
) -> f64 {
    if atr <= 0.0 || atr_multiplier <= 0.0 || entry_price <= 0.0 {
        return entry_price;
    }
    
    let distance = atr * atr_multiplier;
    match direction {
        Direction::Long => (entry_price - distance).max(0.0),
        Direction::Short => entry_price + distance,
    }
}

/// Compute take profit price based on ATR and reward multiple
pub fn compute_take_profit_price(
    entry_price: f64,
    atr: f64,
    direction: Direction,
    atr_multiplier: f64,
    reward_multiple: f64,
) -> f64 {
    if atr <= 0.0 || atr_multiplier <= 0.0 || reward_multiple <= 0.0 || entry_price <= 0.0 {
        return entry_price;
    }
    
    let distance = atr * atr_multiplier * reward_multiple;
    match direction {
        Direction::Long => entry_price + distance,
        Direction::Short => (entry_price - distance).max(0.0),
    }
}

/// Compute position size based on risk budget
pub fn compute_position_size(
    balance: f64,
    risk_pct: f64,
    entry_price: f64,
    stop_price: f64,
    min_position_size: i64,
) -> i64 {
    if balance <= 0.0 || risk_pct <= 0.0 || entry_price <= 0.0 {
        return 0;
    }
    
    let per_unit_risk = (entry_price - stop_price).abs();
    if per_unit_risk <= 0.0 {
        return 0;
    }
    
    let risk_budget = balance * risk_pct;
    let size = (risk_budget / per_unit_risk).floor() as i64;
    
    if size < min_position_size {
        return 0;
    }
    
    size
}

/// Plan a trade with risk management
pub fn plan_trade(
    entry_price: f64,
    atr: f64,
    direction: Direction,
    config: &RiskConfig,
) -> RiskOutcome {
    if atr <= 0.0 || entry_price <= 0.0 || !atr.is_finite() || !entry_price.is_finite() {
        return RiskOutcome {
            direction,
            entry_price,
            position_size: 0,
            stop_price: entry_price,
            take_profit_price: entry_price,
            risk_amount: 0.0,
            rr_multiple: 0.0,
        };
    }
    
    let stop_price = compute_stop_price(entry_price, atr, direction, config.atr_multiplier);
    let take_profit_price = compute_take_profit_price(
        entry_price,
        atr,
        direction,
        config.atr_multiplier,
        config.reward_multiple,
    );
    
    let position_size = compute_position_size(
        config.balance,
        config.risk_pct,
        entry_price,
        stop_price,
        config.min_position_size,
    );
    
    let per_unit_risk = (entry_price - stop_price).abs();
    let risk_amount = position_size as f64 * per_unit_risk;
    let rr_multiple = if per_unit_risk > 0.0 {
        config.reward_multiple
    } else {
        0.0
    };
    
    if position_size == 0 {
        return RiskOutcome {
            direction,
            entry_price,
            position_size: 0,
            stop_price: entry_price,
            take_profit_price: entry_price,
            risk_amount: 0.0,
            rr_multiple: 0.0,
        };
    }
    
    RiskOutcome {
        direction,
        entry_price,
        position_size,
        stop_price,
        take_profit_price,
        risk_amount,
        rr_multiple,
    }
}

/// Simulate trade execution bar-by-bar
pub fn simulate_trade(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    risk: &RiskOutcome,
) -> (f64, usize) {
    let n = highs.len().min(lows.len()).min(closes.len());
    
    if n == 0 {
        return (risk.entry_price, 0);
    }
    
    // Check each bar for stop or take profit hit
    for i in 0..n {
        let high = highs[i];
        let low = lows[i];
        
        let (hit_stop, hit_take) = match risk.direction {
            Direction::Long => (low <= risk.stop_price, high >= risk.take_profit_price),
            Direction::Short => (high >= risk.stop_price, low <= risk.take_profit_price),
        };
        
        if hit_stop {
            return (risk.stop_price, i);
        }
        if hit_take {
            return (risk.take_profit_price, i);
        }
    }
    
    // Exit at last close if no stop/TP hit
    (closes[n - 1], n - 1)
}

/// Compute trade P&L
pub fn compute_trade_pnl(risk: &RiskOutcome, entry_price: f64, exit_price: f64) -> f64 {
    let delta = match risk.direction {
        Direction::Long => exit_price - entry_price,
        Direction::Short => entry_price - exit_price,
    };
    delta * risk.position_size as f64
}

/// Single trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub direction: Direction,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position_size: i64,
    pub pnl: f64,
    pub return_pct: f64,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub final_balance: f64,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
    pub n_trades: usize,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: Option<f64>,
}

impl BacktestResult {
    pub fn empty(initial_balance: f64) -> Self {
        BacktestResult {
            final_balance: initial_balance,
            equity_curve: vec![initial_balance],
            trades: Vec::new(),
            n_trades: 0,
            win_rate: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_from_str() {
        assert_eq!(Direction::from_str("long"), Some(Direction::Long));
        assert_eq!(Direction::from_str("Long"), Some(Direction::Long));
        assert_eq!(Direction::from_str("SHORT"), Some(Direction::Short));
        assert_eq!(Direction::from_str("invalid"), None);
    }

    #[test]
    fn test_compute_stop_price_long() {
        let stop = compute_stop_price(100.0, 2.0, Direction::Long, 2.0);
        assert_eq!(stop, 96.0); // 100 - (2 * 2)
    }

    #[test]
    fn test_compute_stop_price_short() {
        let stop = compute_stop_price(100.0, 2.0, Direction::Short, 2.0);
        assert_eq!(stop, 104.0); // 100 + (2 * 2)
    }

    #[test]
    fn test_compute_take_profit_long() {
        let tp = compute_take_profit_price(100.0, 2.0, Direction::Long, 2.0, 2.0);
        assert_eq!(tp, 108.0); // 100 + (2 * 2 * 2)
    }

    #[test]
    fn test_compute_position_size() {
        let size = compute_position_size(10000.0, 0.01, 100.0, 98.0, 1);
        // Risk budget = 10000 * 0.01 = 100
        // Per unit risk = 100 - 98 = 2
        // Size = 100 / 2 = 50
        assert_eq!(size, 50);
    }

    #[test]
    fn test_simulate_trade_hit_stop() {
        let highs = vec![105.0, 106.0, 95.0];
        let lows = vec![99.0, 100.0, 90.0];
        let closes = vec![102.0, 103.0, 92.0];
        
        let risk = RiskOutcome {
            direction: Direction::Long,
            entry_price: 100.0,
            position_size: 10,
            stop_price: 96.0,
            take_profit_price: 110.0,
            risk_amount: 40.0,
            rr_multiple: 2.0,
        };
        
        let (exit_price, exit_idx) = simulate_trade(&highs, &lows, &closes, &risk);
        assert_eq!(exit_price, 96.0);
        assert_eq!(exit_idx, 2);
    }

    #[test]
    fn test_compute_trade_pnl_long_win() {
        let risk = RiskOutcome {
            direction: Direction::Long,
            entry_price: 100.0,
            position_size: 10,
            stop_price: 96.0,
            take_profit_price: 110.0,
            risk_amount: 40.0,
            rr_multiple: 2.0,
        };
        
        let pnl = compute_trade_pnl(&risk, 100.0, 110.0);
        assert_eq!(pnl, 100.0); // (110 - 100) * 10
    }
}
