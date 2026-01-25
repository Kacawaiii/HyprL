//! Core types shared across the supercalc engine.

use serde::{Deserialize, Serialize};

/// Single OHLCV candle.
/// Assumption: timestamps are already sorted and adjusted on the Python side.
/// Prices may already be adjusted for splits/dividends.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    pub ts: i64, // epoch milliseconds or nanoseconds – must match HyprL convention
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Backtest configuration for a single strategy on a single asset.
///
/// This struct is intentionally generic; HyprL will map its own
/// strategy/config structures onto this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Position sizing parameters (e.g., risk_pct, fixed_fraction, etc.).
    pub risk_pct: f64,
    /// Commission per side (fraction of notional, e.g. 0.0005).
    pub commission_pct: f64,
    /// Slippage per side (fraction of notional, e.g. 0.0005).
    pub slippage_pct: f64,
    /// Maximum allowed position leverage (if applicable).
    pub max_leverage: f64,
    /// Placeholder for strategy-specific numeric parameters (thresholds, etc.).
    pub params: Vec<f64>,
    /// Whether shorting is allowed.
    pub allow_short: bool,
    /// Optional label for debugging / tracking.
    pub label: Option<String>,
    /// ATR lookback window (Wilder smoothing, default 14)
    pub atr_window: usize,
    /// Stop-loss distance multiplier expressed in ATR units (default 2.0)
    pub atr_mult_stop: f64,
    /// Take-profit distance multiplier in ATR units (default 4.0)
    pub atr_mult_tp: f64,
    /// Toggle ATR-based position sizing + stop/take logic
    pub use_atr_position_sizing: bool,
    /// Trailing stop activation threshold in R-multiples (0.0 = disabled)
    pub trailing_activation_r: f64,
    /// Trailing stop distance in R-multiples
    pub trailing_distance_r: f64,
    // TODO(supercalc): extend with explicit fields once the HyprL–Rust schema is finalized.
}

/// Single point on the equity curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EquityPoint {
    pub ts: i64,
    pub equity: f64,
}

/// Summary metrics for risk/performance.
/// Must match the semantics of HyprL's Python Risk Layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: i64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub expectancy: f64,
    pub risk_of_ruin: f64,
    pub maxdd_p05: f64,
    pub maxdd_p95: f64,
    pub pnl_p05: f64,
    pub pnl_p50: f64,
    pub pnl_p95: f64,
    pub robustness_score: f64,
}

/// Full backtest report for a single strategy on a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub config: BacktestConfig,
    pub metrics: PerformanceMetrics,
    /// Equity curve, chronologically sorted.
    pub equity_curve: Vec<EquityPoint>,
    /// Number of trades taken.
    pub n_trades: usize,
    /// Additional debug / diagnostic info if needed.
    pub debug_info: Option<String>,
    /// Optional per-trade outcomes for downstream logging.
    pub trades: Vec<TradeOutcome>,
}

/// Per-trade outcome (minimal contract for Python logging/gates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOutcome {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub exit_reason: String,
}
