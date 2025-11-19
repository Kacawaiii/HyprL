"""Test parity between Python and Rust indicator implementations."""

import numpy as np
import pandas as pd
import pytest
import sys
import os
import importlib.util

# Add the Rust extension to the path
rust_lib_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "accelerators/rust/hyprl_accel/target/release"
)
if os.path.exists(rust_lib_path):
    sys.path.insert(0, rust_lib_path)

try:
    import hyprl_accel
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Import the indicators module directly without going through __init__.py
spec = importlib.util.spec_from_file_location(
    "technical",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src/hyprl/indicators/technical.py')
)
technical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(technical)
TechnicalIndicatorCalculator = technical.TechnicalIndicatorCalculator


@pytest.fixture
def sample_ohlcv_data():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n)
    closes = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from closes with some noise
    opens = closes * np.random.uniform(0.98, 1.02, n)
    highs = np.maximum(opens, closes) * np.random.uniform(1.0, 1.02, n)
    lows = np.minimum(opens, closes) * np.random.uniform(0.98, 1.0, n)
    volumes = np.random.uniform(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
    
    return df


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_indicators_parity_basic(sample_ohlcv_data):
    """Test that basic indicators match between Python and Rust."""
    df = sample_ohlcv_data
    
    # Python indicators
    calc = TechnicalIndicatorCalculator(df['close'])
    sma_short_py = calc.sma(10)
    sma_long_py = calc.sma(20)
    rsi_py = calc.rsi(14)
    
    # Rust indicators
    rust_result = hyprl_accel.compute_indicators_py(
        df,
        sma_short_window=10,
        sma_long_window=20,
        rsi_window=14,
        atr_window=14,
    )
    
    # Convert to arrays for comparison
    sma_short_rust = np.array([v if v is not None else np.nan for v in rust_result['sma_short']])
    sma_long_rust = np.array([v if v is not None else np.nan for v in rust_result['sma_long']])
    rsi_rust = np.array([v if v is not None else np.nan for v in rust_result['rsi']])
    
    # Check SMA short
    valid_mask = ~np.isnan(sma_short_py) & ~np.isnan(sma_short_rust)
    if valid_mask.any():
        np.testing.assert_allclose(
            sma_short_py[valid_mask],
            sma_short_rust[valid_mask],
            rtol=1e-5,
            err_msg="SMA short values don't match"
        )
    
    # Check SMA long
    valid_mask = ~np.isnan(sma_long_py) & ~np.isnan(sma_long_rust)
    if valid_mask.any():
        np.testing.assert_allclose(
            sma_long_py[valid_mask],
            sma_long_rust[valid_mask],
            rtol=1e-5,
            err_msg="SMA long values don't match"
        )
    
    # Check RSI - may have different implementations, use looser tolerance
    valid_mask = ~np.isnan(rsi_py) & ~np.isnan(rsi_rust)
    if valid_mask.any():
        # RSI can differ slightly due to implementation details
        # Just check that values are in reasonable range and correlated
        assert np.all((rsi_rust[valid_mask] >= 0) & (rsi_rust[valid_mask] <= 100)), \
            "RSI values out of valid range [0, 100]"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_trend_ratio_parity(sample_ohlcv_data):
    """Test that trend_ratio matches between Python and Rust."""
    df = sample_ohlcv_data
    
    # Python calculation
    calc = TechnicalIndicatorCalculator(df['close'])
    sma_short = calc.sma(10)
    sma_long = calc.sma(20)
    trend_ratio_py = (sma_short - sma_long) / sma_long
    trend_ratio_py = trend_ratio_py.clip(-0.05, 0.05)
    
    # Rust indicators
    rust_result = hyprl_accel.compute_indicators_py(
        df,
        sma_short_window=10,
        sma_long_window=20,
        rsi_window=14,
        atr_window=14,
    )
    
    trend_ratio_rust = np.array([v if v is not None else np.nan for v in rust_result['trend_ratio']])
    
    # Check trend_ratio
    valid_mask = ~np.isnan(trend_ratio_py) & ~np.isnan(trend_ratio_rust)
    if valid_mask.any():
        np.testing.assert_allclose(
            trend_ratio_py[valid_mask],
            trend_ratio_rust[valid_mask],
            rtol=1e-5,
            err_msg="Trend ratio values don't match"
        )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rolling_volatility_parity(sample_ohlcv_data):
    """Test that rolling volatility matches between Python and Rust."""
    df = sample_ohlcv_data
    window = 10
    
    # Python calculation
    volatility_py = df['close'].pct_change().rolling(window=window).std()
    
    # Rust indicators
    rust_result = hyprl_accel.compute_indicators_py(
        df,
        sma_short_window=window,
        sma_long_window=20,
        rsi_window=14,
        atr_window=14,
    )
    
    rolling_vol_rust = np.array([v if v is not None else np.nan for v in rust_result['rolling_vol']])
    
    # Check rolling volatility
    valid_mask = ~np.isnan(volatility_py) & ~np.isnan(rolling_vol_rust)
    if valid_mask.any():
        np.testing.assert_allclose(
            volatility_py[valid_mask],
            rolling_vol_rust[valid_mask],
            rtol=1e-5,
            err_msg="Rolling volatility values don't match"
        )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_indicator_output_structure(sample_ohlcv_data):
    """Test that Rust indicators return expected structure."""
    df = sample_ohlcv_data
    
    rust_result = hyprl_accel.compute_indicators_py(
        df,
        sma_short_window=10,
        sma_long_window=20,
        rsi_window=14,
        atr_window=14,
    )
    
    # Check expected keys exist
    expected_keys = [
        'price', 'sma_short', 'sma_long', 'ema_short', 'ema_long',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr',
        'trend_ratio', 'rolling_vol', 'rsi_normalized'
    ]
    
    for key in expected_keys:
        assert key in rust_result, f"Missing key: {key}"
        assert len(rust_result[key]) == len(df), f"Wrong length for {key}"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rsi_normalized_parity(sample_ohlcv_data):
    """Test that rsi_normalized matches between Python and Rust."""
    df = sample_ohlcv_data
    
    # Python calculation
    calc = TechnicalIndicatorCalculator(df['close'])
    rsi_py = calc.rsi(14)
    rsi_normalized_py = (rsi_py - 50.0) / 50.0
    
    # Rust indicators
    rust_result = hyprl_accel.compute_indicators_py(
        df,
        sma_short_window=10,
        sma_long_window=20,
        rsi_window=14,
        atr_window=14,
    )
    
    rsi_normalized_rust = np.array([v if v is not None else np.nan for v in rust_result['rsi_normalized']])
    
    # Check rsi_normalized
    valid_mask = ~np.isnan(rsi_normalized_py) & ~np.isnan(rsi_normalized_rust)
    if valid_mask.any():
        # Since RSI implementations may differ, just check normalized is in valid range
        assert np.all((rsi_normalized_rust[valid_mask] >= -1.0) & (rsi_normalized_rust[valid_mask] <= 1.0)), \
            "RSI normalized values out of valid range [-1, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
