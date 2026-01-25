//! hyprl_supercalc – native compute engine for HyprL
//!
//! Roles:
//! - Read prepared OHLCV / feature data (typically from Polars DataFrames).
//! - Compute indicators and run backtests quickly.
//! - Emit metrics consistent with the Python Risk Layer (PF, Sharpe, DD, RoR, expectancy, etc.).
//! - Expose a minimal, stable PyO3 API for HyprL's orchestrator.

pub mod backtest;
pub mod batch;
pub mod core;
pub mod ffi;
pub mod indicators;
pub mod metrics;

// Re-export some key types for convenience
pub use crate::core::{BacktestConfig, BacktestReport, Candle};
pub use crate::indicators::{compute_indicators, IndicatorSet};
