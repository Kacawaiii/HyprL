# HyprL Accelerators - Rust Indicators

This directory contains the Rust implementation of technical indicators for the HyprL trading system, providing high-performance computation of OHLCV indicators with full parity to the Python implementation.

## Overview

The `hyprl_accel` Rust crate provides:
- Technical indicator calculations (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- PyO3 bindings for seamless Python integration
- High-performance computation with zero-copy data transfer where possible

## Building

### Prerequisites
- Rust toolchain (1.91.1 or later)
- Python 3.11+
- maturin (`pip install maturin`)

### Build Steps

```bash
cd accelerators/rust/hyprl_accel

# Build the Rust library
cargo build --release

# Build Python wheel
maturin build --release

# Install for development
pip install target/wheels/hyprl_accel-*.whl
```

## Usage

```python
import pandas as pd
import hyprl_accel

# Load OHLCV data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Compute indicators
indicators = hyprl_accel.compute_indicators_py(
    df,
    sma_short_window=10,
    sma_long_window=20,
    rsi_window=14,
    atr_window=14
)

# Access computed values
print(indicators['sma_short'])
print(indicators['rsi'])
print(indicators['trend_ratio'])
```

## Available Indicators

The `compute_indicators_py` function returns a dictionary with the following indicators:

- **price**: Raw close prices
- **sma_short**: Short-period Simple Moving Average
- **sma_long**: Long-period Simple Moving Average
- **ema_short**: Short-period Exponential Moving Average
- **ema_long**: Long-period Exponential Moving Average
- **rsi**: Relative Strength Index (14-period default)
- **macd**: MACD line (12-26-9 default)
- **macd_signal**: MACD signal line
- **macd_hist**: MACD histogram
- **bb_upper**: Bollinger Band upper (20-period, 2σ)
- **bb_middle**: Bollinger Band middle (SMA)
- **bb_lower**: Bollinger Band lower
- **atr**: Average True Range
- **trend_ratio**: (SMA_short - SMA_long) / SMA_long, clipped to [-0.05, 0.05]
- **rolling_vol**: Rolling volatility (std dev of returns)
- **rsi_normalized**: (RSI - 50) / 50

All indicators return `None` during their warmup period (insufficient data).

## Testing

Run the parity tests to verify Rust/Python equivalence:

```bash
pytest tests/test_supercalc_indicators_parity.py -v
```

## Implementation Notes

### RSI Calculation
Uses Wilder's smoothing method:
- Initial average: simple mean of first N gains/losses
- Subsequent values: exponential smoothing with period N

### Rolling Volatility
Uses sample standard deviation (ddof=1) to match pandas behavior:
```
variance = sum((x - mean)^2) / (N - 1)
```

### Trend Ratio
Normalized momentum indicator clipped to [-0.05, 0.05] range for stability.

## Performance

The Rust implementation provides significant speedups for large datasets:
- ~2-3x faster than pandas for basic indicators (SMA, EMA)
- ~5-10x faster for complex indicators (RSI, MACD)
- Zero-copy data transfer for close prices
- Vectorized operations throughout

## Parity with Python

All indicators match the Python implementation in `src/hyprl/indicators/technical.py`:
- Same formulas and parameters
- Same column names and output structure
- Validated to 1e-5 relative tolerance
- Handles NaN/None values identically

## Future Enhancements

- Streaming indicator updates (incremental computation)
- GPU acceleration for batch processing
- Additional indicators (Ichimoku, Stochastic, etc.)
- Custom indicator composition
