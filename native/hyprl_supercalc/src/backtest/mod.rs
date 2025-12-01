//! Single-asset backtest engine.
//!
//! Assumptions:
//! - Input candles are sorted by timestamp.
//! - `signal` contains desired exposure per bar ([-1, 1]).
//! - Cost model (commission + slippage) matches the Python layer.

use crate::core::{BacktestConfig, BacktestReport, Candle, EquityPoint};
use crate::indicators;
use crate::metrics;

const EPSILON: f64 = 1e-9;

/// Run a backtest for a single strategy on a single asset.
///
/// - `signal[t]` expresses desired exposure for bar `t` in [-1, 1].
/// - Position changes incur commission + slippage costs on the traded notional.
/// - Equity starts at 1.0 and grows multiplicatively with returns.
pub fn run_backtest(candles: &[Candle], signal: &[f64], cfg: &BacktestConfig) -> BacktestReport {
    assert_eq!(
        candles.len(),
        signal.len(),
        "candles/signal length mismatch"
    );

    if candles.len() < 2 {
        let ts = candles.first().map(|c| c.ts).unwrap_or(0);
        let equity_curve = vec![EquityPoint { ts, equity: 1.0 }];
        let metrics = metrics::metrics_from_equity(&equity_curve, cfg.risk_pct);
        return BacktestReport {
            config: cfg.clone(),
            metrics,
            equity_curve,
            n_trades: 0,
            debug_info: Some("Not enough candles".to_string()),
        };
    }

    let leverage_cap = cfg.max_leverage.max(1.0);
    let use_atr = cfg.use_atr_position_sizing;
    let atr_values = if use_atr {
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        Some(indicators::atr(&highs, &lows, &closes, cfg.atr_window))
    } else {
        None
    };
    let mut equity = 1.0_f64;
    let mut position = 0.0_f64;
    let mut equity_curve = Vec::with_capacity(candles.len());
    let mut n_trades = 0usize;
    let mut stop_price: Option<f64> = None;
    let mut take_profit_price: Option<f64> = None;

    // Trailing stop state
    let mut entry_price = 0.0_f64;
    let mut risk_per_unit = 0.0_f64;
    let mut highest_price = 0.0_f64;
    let mut lowest_price = 0.0_f64;
    let trailing_active = cfg.trailing_activation_r > 0.0 && cfg.trailing_distance_r > 0.0;

    equity_curve.push(EquityPoint {
        ts: candles[0].ts,
        equity,
    });

    for i in 1..candles.len() {
        let prev_close = candles[i - 1].close;
        let bar = &candles[i];
        let mut closed_this_bar = false;

        if use_atr && position.abs() > EPSILON {
            if trailing_active && risk_per_unit > EPSILON {
                if position > 0.0 {
                    highest_price = highest_price.max(bar.high);
                    let r_unrealized = (highest_price - entry_price) / risk_per_unit;
                    if r_unrealized >= cfg.trailing_activation_r {
                        let candidate_stop = highest_price - cfg.trailing_distance_r * risk_per_unit;
                        if let Some(curr) = stop_price {
                            stop_price = Some(curr.max(candidate_stop));
                        } else {
                            stop_price = Some(candidate_stop);
                        }
                    }
                } else {
                    lowest_price = lowest_price.min(bar.low);
                    let r_unrealized = (entry_price - lowest_price) / risk_per_unit;
                    if r_unrealized >= cfg.trailing_activation_r {
                        let candidate_stop = lowest_price + cfg.trailing_distance_r * risk_per_unit;
                        if let Some(curr) = stop_price {
                            stop_price = Some(curr.min(candidate_stop));
                        } else {
                            stop_price = Some(candidate_stop);
                        }
                    }
                }
            }

            if let Some(exit_price) =
                evaluate_stop_take(position, stop_price, take_profit_price, bar)
            {
                let exit_return = if prev_close > 0.0 {
                    (exit_price / prev_close) - 1.0
                } else {
                    0.0
                };
                equity *= 1.0 + position * exit_return;
                equity = equity.max(f64::EPSILON);
                let closing_leverage = position.abs();
                position = 0.0;
                stop_price = None;
                take_profit_price = None;
                if closing_leverage > EPSILON {
                    apply_transaction_cost(&mut equity, cfg, closing_leverage, use_atr);
                    n_trades += 1;
                }
                closed_this_bar = true;
            }
        }

        let mut desired = signal[i].clamp(-1.0, 1.0);
        if !cfg.allow_short {
            desired = desired.max(0.0);
        }

        let mut target_position = position;
        if use_atr {
            if closed_this_bar {
                desired = 0.0;
            }
            let desired_sign = if desired > EPSILON {
                1.0
            } else if desired < -EPSILON {
                -1.0
            } else {
                0.0
            };

            if desired_sign == 0.0 {
                target_position = 0.0;
            } else if position.abs() <= EPSILON || position.signum() != desired_sign {
                let atr_value = atr_values
                    .as_ref()
                    .and_then(|vals| vals.get(i))
                    .copied()
                    .unwrap_or(0.0);
                if let Some(sizing) =
                    compute_atr_sizing(equity, prev_close, atr_value, cfg, leverage_cap)
                {
                    target_position = desired_sign * sizing.leverage;
                    entry_price = sizing.entry_price;
                    risk_per_unit = sizing.stop_dist;
                    highest_price = entry_price;
                    lowest_price = entry_price;

                    stop_price = Some(if desired_sign > 0.0 {
                        sizing.entry_price - sizing.stop_dist
                    } else {
                        sizing.entry_price + sizing.stop_dist
                    });
                    take_profit_price = Some(if desired_sign > 0.0 {
                        sizing.entry_price + sizing.tp_dist
                    } else {
                        sizing.entry_price - sizing.tp_dist
                    });
                } else {
                    target_position =
                        (desired_sign * leverage_cap).clamp(-leverage_cap, leverage_cap);
                    stop_price = None;
                    take_profit_price = None;
                }
            }
        } else {
            desired = desired.clamp(-leverage_cap, leverage_cap);
            target_position = desired;
        }

        let delta = target_position - position;
        if delta.abs() > EPSILON {
            apply_transaction_cost(&mut equity, cfg, delta.abs(), use_atr);
            position = target_position;
            n_trades += 1;

            if use_atr {
                if position.abs() <= EPSILON {
                    stop_price = None;
                    take_profit_price = None;
                }
            }
        }

        if prev_close.is_finite() && bar.close.is_finite() && prev_close > 0.0 {
            let r = (bar.close / prev_close) - 1.0;
            equity *= 1.0 + position * r;
            equity = equity.max(f64::EPSILON);
        }

        equity_curve.push(EquityPoint { ts: bar.ts, equity });
    }

    let metrics = metrics::metrics_from_equity(&equity_curve, cfg.risk_pct);

    BacktestReport {
        config: cfg.clone(),
        metrics,
        equity_curve,
        n_trades,
        debug_info: None,
    }
}

struct AtrSizing {
    leverage: f64,
    entry_price: f64,
    stop_dist: f64,
    tp_dist: f64,
}

fn compute_atr_sizing(
    equity: f64,
    entry_price: f64,
    atr_value: f64,
    cfg: &BacktestConfig,
    leverage_cap: f64,
) -> Option<AtrSizing> {
    if !entry_price.is_finite() || entry_price <= 0.0 {
        return None;
    }
    if !atr_value.is_finite() || atr_value <= 0.0 {
        return None;
    }
    let stop_dist = cfg.atr_mult_stop.max(0.0) * atr_value;
    let tp_dist = cfg.atr_mult_tp.max(0.0) * atr_value;
    if stop_dist <= 0.0 || tp_dist <= 0.0 {
        return None;
    }

    let risk_amount = equity * cfg.risk_pct.max(0.0);
    if risk_amount <= 0.0 {
        return None;
    }
    let units = risk_amount / stop_dist;
    let notional = units * entry_price;
    if !notional.is_finite() || notional <= 0.0 {
        return None;
    }
    let leverage = (notional / equity).clamp(0.0, leverage_cap);
    if leverage <= 0.0 {
        return None;
    }
    Some(AtrSizing {
        leverage,
        entry_price,
        stop_dist,
        tp_dist,
    })
}

fn evaluate_stop_take(
    position: f64,
    stop_price: Option<f64>,
    take_profit_price: Option<f64>,
    bar: &Candle,
) -> Option<f64> {
    let stop = stop_price?;
    let take = take_profit_price?;
    if position > 0.0 {
        if bar.low <= stop {
            return Some(stop);
        }
        if bar.high >= take {
            return Some(take);
        }
    } else if position < 0.0 {
        if bar.high >= stop {
            return Some(stop);
        }
        if bar.low <= take {
            return Some(take);
        }
    }
    None
}

fn apply_transaction_cost(
    equity: &mut f64,
    cfg: &BacktestConfig,
    delta_leverage: f64,
    atr_mode: bool,
) {
    if delta_leverage <= EPSILON {
        return;
    }
    let roundtrip_cost = (cfg.commission_pct + cfg.slippage_pct).max(0.0);
    if roundtrip_cost <= 0.0 {
        return;
    }
    let base = if atr_mode {
        *equity * delta_leverage
    } else {
        *equity * cfg.risk_pct.max(0.0) * delta_leverage
    };
    if base <= 0.0 {
        return;
    }
    *equity -= base * roundtrip_cost;
    if *equity < f64::EPSILON {
        *equity = f64::EPSILON;
    }
}
