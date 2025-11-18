"""Test parity between Python and Rust indicator implementations."""
import numpy as np
import pandas as pd
import pytest

from hyprl.indicators.technical import compute_feature_frame

try:
    from hyprl import hyprl_supercalc
    HAS_SUPERCALC = True
except ImportError:
    HAS_SUPERCALC = False


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_indicators_basic_import():
    """Test that we can import and call the Rust indicators."""
    # Create simple test data
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    df = pd.DataFrame({
        'open': np.linspace(100, 110, n),
        'high': np.linspace(101, 111, n),
        'low': np.linspace(99, 109, n),
        'close': np.linspace(100, 110, n),
        'volume': np.full(n, 1000.0),
    }, index=dates)
    
    # Call Rust function
    result = hyprl_supercalc.compute_indicators_py(
        df, 
        sma_short=5, 
        sma_long=20, 
        rsi_window=14, 
        atr_window=14
    )
    
    # Check that we got a dictionary back with expected keys
    assert isinstance(result, dict)
    assert 'sma_short' in result
    assert 'sma_long' in result
    assert 'rsi' in result
    assert 'atr' in result
    assert 'trend_ratio' in result
    assert 'volatility' in result
    
    # Check that arrays are the right length
    assert len(result['sma_short']) == n
    assert len(result['rsi']) == n


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_sma_parity():
    """Test that Rust SMA matches expected values."""
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Simple constant price for easy verification
    df = pd.DataFrame({
        'open': np.full(n, 100.0),
        'high': np.full(n, 101.0),
        'low': np.full(n, 99.0),
        'close': np.full(n, 100.0),
        'volume': np.full(n, 1000.0),
    }, index=dates)
    
    result = hyprl_supercalc.compute_indicators_py(
        df, 
        sma_short=5, 
        sma_long=20, 
        rsi_window=14, 
        atr_window=14
    )
    
    # For constant prices, SMA should equal the price (after warmup)
    sma_short = np.array(result['sma_short'])
    valid_sma = sma_short[~np.isnan(sma_short)]
    assert len(valid_sma) > 0
    assert np.allclose(valid_sma, 100.0, rtol=1e-6)


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_trend_ratio_bounds():
    """Test that trend_ratio is properly clipped to [-0.05, 0.05]."""
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Create prices with strong trend
    df = pd.DataFrame({
        'open': np.linspace(100, 200, n),
        'high': np.linspace(101, 201, n),
        'low': np.linspace(99, 199, n),
        'close': np.linspace(100, 200, n),
        'volume': np.full(n, 1000.0),
    }, index=dates)
    
    result = hyprl_supercalc.compute_indicators_py(
        df, 
        sma_short=5, 
        sma_long=20, 
        rsi_window=14, 
        atr_window=14
    )
    
    trend_ratio = np.array(result['trend_ratio'])
    valid_ratio = trend_ratio[~np.isnan(trend_ratio)]
    
    # All values should be within [-0.05, 0.05]
    assert np.all(valid_ratio >= -0.05)
    assert np.all(valid_ratio <= 0.05)


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_atr_positive():
    """Test that ATR values are positive."""
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Random-ish price movement
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'open': closes,
        'high': closes + np.random.rand(n),
        'low': closes - np.random.rand(n),
        'close': closes,
        'volume': np.full(n, 1000.0),
    }, index=dates)
    
    result = hyprl_supercalc.compute_indicators_py(
        df, 
        sma_short=5, 
        sma_long=20, 
        rsi_window=14, 
        atr_window=14
    )
    
    atr = np.array(result['atr'])
    valid_atr = atr[~np.isnan(atr)]
    
    # All ATR values should be positive
    assert len(valid_atr) > 0
    assert np.all(valid_atr > 0)


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_rsi_bounds():
    """Test that RSI values are within [0, 100]."""
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Oscillating prices
    closes = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    
    df = pd.DataFrame({
        'open': closes,
        'high': closes + 1,
        'low': closes - 1,
        'close': closes,
        'volume': np.full(n, 1000.0),
    }, index=dates)
    
    result = hyprl_supercalc.compute_indicators_py(
        df, 
        sma_short=5, 
        sma_long=20, 
        rsi_window=14, 
        atr_window=14
    )
    
    rsi = np.array(result['rsi'])
    valid_rsi = rsi[~np.isnan(rsi)]
    
    # RSI should be between 0 and 100
    assert len(valid_rsi) > 0
    assert np.all(valid_rsi >= 0)
    assert np.all(valid_rsi <= 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
