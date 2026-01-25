# HyprL Rust Engine (hyprl_supercalc)

Native compute engine for ultra-fast backtesting and indicator calculation.

## Overview

`hyprl_supercalc` is a Rust library with PyO3 bindings that provides **37-56x performance improvements** over pure Python implementations. It handles the computationally intensive parts of HyprL:

- Technical indicator calculation (SMA, EMA, RSI, MACD, ATR, Bollinger Bands)
- Vectorized backtesting with realistic costs
- Parallel grid search optimization using Rayon
- Monte Carlo robustness analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (HyprL)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Strategy    │  │ Features    │  │ Grid Search         │ │
│  │ Engine      │  │ Pipeline    │  │ Optimizer           │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          │ PyO3 FFI                         │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              hyprl_supercalc (Rust)                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ indicators/ │  │ backtest/   │  │ batch.rs    │  │   │
│  │  │ - SMA, EMA  │  │ - Engine    │  │ - Parallel  │  │   │
│  │  │ - RSI, MACD │  │ - Metrics   │  │   search    │  │   │
│  │  │ - ATR, BB   │  │ - Trades    │  │ - Rayon     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         Rust                                │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Parallelism** | `Rayon` | Work-stealing scheduler to saturate all CPU cores |
| **Data Engine** | `Polars` | SIMD-vectorized DataFrames for millions of rows |
| **FFI Bridge** | `PyO3` | Zero-copy Python-Rust interop |
| **Mathematics** | `ndarray` | Cache-optimized matrix operations |
| **Serialization** | `serde` | Fast JSON serialization for configs/reports |

## Performance Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| ATR 14 (10k bars) | 45ms | 1.2ms | **37x** |
| RSI 14 (10k bars) | 38ms | 0.9ms | **42x** |
| Single backtest | 450ms | 12ms | **37x** |
| Grid search (1000 configs) | 7.5min | 8sec | **56x** |
| Indicator set (all) | 180ms | 4ms | **45x** |
| 1M candles processing | - | ~5M/sec | - |

**Memory**: 80% RAM reduction compared to pure Pandas implementation.

---

## Installation

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install maturin (Python-Rust build tool)
pip install maturin
```

### Build from Source

```bash
cd native/hyprl_supercalc

# Development build (faster compilation, slower runtime)
maturin develop

# Release build (slower compilation, faster runtime) - RECOMMENDED
maturin develop --release

# Build wheel for distribution
maturin build --release
```

### Verify Installation

```python
from hyprl_supercalc import compute_indicators_py, run_backtest_py
print("Rust engine loaded successfully!")
```

---

## API Reference

### compute_indicators_py

Compute all technical indicators in a single optimized pass.

```python
from hyprl_supercalc import compute_indicators_py
import polars as pl

# Load OHLCV data as Polars DataFrame
df = pl.DataFrame({
    "ts": [...],      # int64 timestamps (epoch ms)
    "open": [...],    # float64
    "high": [...],    # float64
    "low": [...],     # float64
    "close": [...],   # float64
    "volume": [...]   # float64
})

# Compute all indicators
indicators = compute_indicators_py(df)

# Available indicators:
print(indicators["sma_20"])           # Simple Moving Average (20)
print(indicators["ema_20"])           # Exponential Moving Average (20)
print(indicators["rsi_14"])           # Relative Strength Index (14)
print(indicators["macd"])             # MACD Line
print(indicators["macd_signal"])      # MACD Signal Line
print(indicators["macd_hist"])        # MACD Histogram
print(indicators["bb_upper_20"])      # Bollinger Band Upper
print(indicators["bb_mid_20"])        # Bollinger Band Middle
print(indicators["bb_lower_20"])      # Bollinger Band Lower
print(indicators["atr_14"])           # Average True Range (14)
print(indicators["trend_ratio_50_200"]) # SMA50/SMA200 ratio
print(indicators["rolling_vol_20"])   # Rolling volatility (20)
```

### run_backtest_py

Run a single backtest with full configuration.

```python
from hyprl_supercalc import run_backtest_py
import polars as pl

# Prepare data
df = pl.DataFrame({...})  # OHLCV data
signal = [0.0, 0.0, 1.0, 1.0, -1.0, ...]  # Signal array (same length as df)
# Signal: 1.0 = long, -1.0 = short, 0.0 = flat

# Configure backtest
config = {
    "risk_pct": 0.02,              # 2% risk per trade
    "commission_pct": 0.001,       # 0.1% commission per side
    "slippage_pct": 0.0005,        # 0.05% slippage per side
    "max_leverage": 1.0,           # No leverage
    "allow_short": True,           # Allow short positions
    "atr_window": 14,              # ATR lookback period
    "atr_mult_stop": 2.0,          # Stop loss = 2x ATR
    "atr_mult_tp": 4.0,            # Take profit = 4x ATR
    "use_atr_position_sizing": True,
    "trailing_activation_r": 1.5,  # Activate trailing at 1.5R profit
    "trailing_distance_r": 0.5,    # Trail by 0.5R
    "label": "NVDA_v3",            # Optional label
    "params": [0.55, 0.45],        # Strategy-specific parameters
}

# Run backtest
report = run_backtest_py(df, signal, config)

# Access results
print(f"Total Return: {report['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {report['metrics']['sharpe']:.2f}")
print(f"Max Drawdown: {report['metrics']['max_drawdown']:.2%}")
print(f"Win Rate: {report['metrics']['win_rate']:.2%}")
print(f"Profit Factor: {report['metrics']['profit_factor']:.2f}")
print(f"Number of Trades: {report['n_trades']}")

# Equity curve
for point in report['equity_curve']:
    print(f"  {point['ts']}: ${point['equity']:.2f}")

# Individual trades
for trade in report['trades']:
    print(f"  {trade['direction']}: {trade['entry_price']:.2f} -> {trade['exit_price']:.2f} = {trade['return_pct']:.2%}")
```

### run_batch_backtest_py

Run multiple backtests in parallel.

```python
from hyprl_supercalc import run_batch_backtest_py

# Multiple configurations
configs = [
    {"risk_pct": 0.01, "atr_mult_stop": 1.5, "atr_mult_tp": 3.0, ...},
    {"risk_pct": 0.02, "atr_mult_stop": 2.0, "atr_mult_tp": 4.0, ...},
    {"risk_pct": 0.03, "atr_mult_stop": 2.5, "atr_mult_tp": 5.0, ...},
]

# Run all in parallel (uses Rayon)
reports = run_batch_backtest_py(df, signal, configs)

for i, report in enumerate(reports):
    print(f"Config {i}: Sharpe={report['metrics']['sharpe']:.2f}")
```

### run_native_search_py

Parallel grid search with constraints filtering.

```python
from hyprl_supercalc import run_native_search_py

# Generate parameter grid
configs = []
for risk in [0.01, 0.015, 0.02, 0.025]:
    for stop in [1.5, 2.0, 2.5, 3.0]:
        for tp in [3.0, 4.0, 5.0, 6.0]:
            configs.append({
                "risk_pct": risk,
                "atr_mult_stop": stop,
                "atr_mult_tp": tp,
                "commission_pct": 0.001,
                "slippage_pct": 0.0005,
                "allow_short": True,
                "atr_window": 14,
                "use_atr_position_sizing": True,
            })

# Define constraints (filter out bad strategies early)
constraints = {
    "min_trades": 50,           # At least 50 trades
    "min_profit_factor": 1.2,   # PF > 1.2
    "min_sharpe": 0.5,          # Sharpe > 0.5
    "max_drawdown": 0.25,       # Max DD < 25%
    "max_risk_of_ruin": 0.05,   # RoR < 5%
    "min_expectancy": 0.005,    # Expectancy > 0.5%
    "min_robustness": 0.6,      # Robustness score > 60%
}

# Run parallel search, return top 10
top_results = run_native_search_py(
    df,
    signal,
    configs,
    constraints,
    top_k=10
)

print(f"Found {len(top_results)} strategies meeting constraints")
for i, report in enumerate(top_results):
    m = report['metrics']
    print(f"{i+1}. Sharpe={m['sharpe']:.2f} PF={m['profit_factor']:.2f} DD={m['max_drawdown']:.1%}")
```

---

## Data Types

### BacktestConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_pct` | float | required | Risk per trade (e.g., 0.02 = 2%) |
| `commission_pct` | float | 0.0 | Commission per side |
| `slippage_pct` | float | 0.0 | Slippage per side |
| `max_leverage` | float | 1.0 | Maximum leverage allowed |
| `allow_short` | bool | false | Allow short positions |
| `atr_window` | int | 14 | ATR lookback period |
| `atr_mult_stop` | float | 2.0 | Stop loss in ATR multiples |
| `atr_mult_tp` | float | 4.0 | Take profit in ATR multiples |
| `use_atr_position_sizing` | bool | false | Use ATR for position sizing |
| `trailing_activation_r` | float | 0.0 | Trailing stop activation (R-multiple) |
| `trailing_distance_r` | float | 0.0 | Trailing stop distance (R-multiple) |
| `label` | string | null | Optional strategy label |
| `params` | list[float] | [] | Strategy-specific parameters |

### PerformanceMetrics

| Field | Type | Description |
|-------|------|-------------|
| `total_return` | float | Total return (e.g., 0.28 = 28%) |
| `cagr` | float | Compound Annual Growth Rate |
| `sharpe` | float | Sharpe Ratio (annualized) |
| `sortino` | float | Sortino Ratio |
| `calmar` | float | Calmar Ratio (CAGR / Max DD) |
| `max_drawdown` | float | Maximum Drawdown |
| `max_drawdown_duration` | int | Max DD duration in bars |
| `profit_factor` | float | Gross Profit / Gross Loss |
| `win_rate` | float | Winning trades / Total trades |
| `expectancy` | float | Average return per trade |
| `risk_of_ruin` | float | Probability of ruin estimate |
| `robustness_score` | float | Overall robustness (0-1) |
| `maxdd_p05` | float | 5th percentile max drawdown |
| `maxdd_p95` | float | 95th percentile max drawdown |
| `pnl_p05` | float | 5th percentile P&L |
| `pnl_p50` | float | Median P&L |
| `pnl_p95` | float | 95th percentile P&L |

### SearchConstraint

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_trades` | int | 30 | Minimum number of trades |
| `min_profit_factor` | float | 1.0 | Minimum profit factor |
| `min_sharpe` | float | 0.0 | Minimum Sharpe ratio |
| `max_drawdown` | float | 1.0 | Maximum drawdown allowed |
| `max_risk_of_ruin` | float | 1.0 | Maximum risk of ruin |
| `min_expectancy` | float | 0.0 | Minimum expectancy |
| `min_robustness` | float | 0.0 | Minimum robustness score |
| `max_maxdd_p95` | float | 1.0 | Max 95th percentile DD |
| `min_pnl_p05` | float | -1.0 | Min 5th percentile P&L |
| `min_pnl_p50` | float | 0.0 | Min median P&L |
| `min_pnl_p95` | float | 0.0 | Min 95th percentile P&L |

### TradeOutcome

| Field | Type | Description |
|-------|------|-------------|
| `entry_idx` | int | Entry bar index |
| `exit_idx` | int | Exit bar index |
| `direction` | string | "long" or "short" |
| `entry_price` | float | Entry price |
| `exit_price` | float | Exit price |
| `pnl` | float | Profit/Loss in dollars |
| `return_pct` | float | Return percentage |
| `exit_reason` | string | "stop", "tp", "signal", "trailing" |

---

## Technical Indicators

### Implemented Indicators

| Indicator | Function | Parameters | Description |
|-----------|----------|------------|-------------|
| SMA | `sma()` | window | Simple Moving Average |
| EMA | `ema()` | window | Exponential Moving Average (pandas compatible) |
| RSI | `rsi()` | window | Relative Strength Index (Wilder smoothing) |
| MACD | `macd_full()` | 12, 26, 9 | MACD line, signal, histogram |
| ATR | `atr()` | window | Average True Range (Wilder smoothing) |
| Bollinger | `bollinger_bands()` | window, num_std | Upper, middle, lower bands |
| Trend Ratio | `trend_ratio()` | short, long | SMA ratio (clamped +/-5%) |
| Volatility | `rolling_volatility()` | window | Rolling log-return std dev |

### Indicator Formulas

**RSI (Relative Strength Index)**
```
RS = EMA(gains, window) / EMA(losses, window)
RSI = 100 - (100 / (1 + RS))
```

**ATR (Average True Range)**
```
TR = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = Wilder_EMA(TR, window)
```

**MACD**
```
MACD = EMA(close, 12) - EMA(close, 26)
Signal = EMA(MACD, 9)
Histogram = MACD - Signal
```

**Bollinger Bands**
```
Middle = SMA(close, 20)
Upper = Middle + 2 * StdDev(close, 20)
Lower = Middle - 2 * StdDev(close, 20)
```

---

## Advanced Features

### Monte Carlo Robustness Analysis

The engine generates 1000 bootstrap variants of the trade history to calculate:

- **PF P5 (Worst Case)**: Statistically guaranteed minimum profitability
- **MaxDD P95**: Expected maximum loss in adverse market conditions
- **Risk of Ruin**: Mathematical probability of capital going to zero

### Parallel Grid Search

Using Rayon's work-stealing scheduler:

```rust
configs.par_iter()
    .map(|config| run_backtest(&candles, &signal, config))
    .filter(|report| passes_constraints(report, &constraints))
    .collect()
```

This saturates all CPU cores for maximum throughput.

---

## Project Structure

```
native/hyprl_supercalc/
├── Cargo.toml              # Rust dependencies
├── Cargo.lock              # Locked versions
└── src/
    ├── lib.rs              # Library entry point, re-exports
    ├── core.rs             # Core types (Candle, Config, Report, Trade)
    ├── indicators/
    │   └── mod.rs          # All technical indicators (SMA, EMA, RSI, etc.)
    ├── backtest/
    │   └── mod.rs          # Bar-by-bar simulation engine
    ├── batch.rs            # Parallel execution with Rayon
    ├── metrics.rs          # Performance metrics (Sharpe, PF, DD, etc.)
    ├── ffi.rs              # PyO3 Python bindings
    ├── data.rs             # Data handling utilities
    └── multi.rs            # Multi-asset support
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| pyo3 | 0.22 | Python bindings |
| pyo3-polars | 0.16 | Polars DataFrame support |
| polars | 0.42 | Data manipulation |
| rayon | 1.10 | Parallel iteration |
| ndarray | 0.15 | N-dimensional arrays |
| serde | 1.0 | Serialization |
| serde_json | 1.0 | JSON support |
| anyhow | 1.0 | Error handling |
| thiserror | 1.0 | Error types |
| statrs | 0.16 | Statistical functions |
| rand | 0.8 | Random number generation |

---

## Development

### Running Tests

```bash
cd native/hyprl_supercalc
cargo test
```

### Benchmarking

```bash
cargo bench
```

### Building Documentation

```bash
cargo doc --open
```

### Code Coverage

```bash
cargo tarpaulin --out Html
```

---

## Troubleshooting

### Build Errors

**"maturin not found"**
```bash
pip install maturin
```

**"cargo not found"**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**"pyo3 version mismatch"**
```bash
pip install --upgrade pyo3
maturin develop --release
```

**Linux: "python3.x-dev not found"**
```bash
sudo apt install python3-dev
```

### Runtime Errors

**"module not found: hyprl_supercalc"**
```bash
cd native/hyprl_supercalc
maturin develop --release
```

**"signal and candles length mismatch"**
Ensure your signal array has the same length as the DataFrame rows.

**"ts column not found"**
DataFrame must have columns: `ts`, `open`, `high`, `low`, `close`, `volume`

---

## Integration with HyprL

The Rust engine is automatically used when available:

```python
# src/hyprl/native/supercalc.py
try:
    from hyprl_supercalc import (
        compute_indicators_py,
        run_backtest_py,
        run_native_search_py,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def compute_indicators(df):
    if RUST_AVAILABLE:
        return compute_indicators_py(df)  # 45x faster
    else:
        return compute_indicators_python(df)  # Fallback
```

---

## License

MIT License - See [LICENSE](../LICENSE)

---

*HyprL Engine - Designed for Speed, Built for Reliability.*
