"""Test parity between Python and Rust backtest implementations."""
import numpy as np
import pandas as pd
import pytest

from hyprl.risk.manager import (
    compute_position_size,
    compute_stop_price,
    compute_take_profit_price,
)

try:
    from hyprl import hyprl_supercalc
    HAS_SUPERCALC = True
except ImportError:
    HAS_SUPERCALC = False


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_position_size_parity():
    """Test position sizing matches Python implementation."""
    balance = 10000.0
    risk_pct = 0.01
    entry_price = 100.0
    stop_price = 98.0
    min_size = 1
    
    # Python implementation
    python_size = compute_position_size(balance, risk_pct, entry_price, stop_price, min_size)
    
    # Rust implementation
    rust_size = hyprl_supercalc.compute_position_size_py(
        balance, risk_pct, entry_price, stop_price, min_size
    )
    
    assert python_size == rust_size


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_stop_price_parity_long():
    """Test stop price calculation for long trades."""
    entry = 100.0
    atr = 2.0
    multiplier = 2.0
    
    # Python implementation
    python_stop = compute_stop_price(entry, atr, "long", multiplier)
    
    # Rust implementation
    rust_stop = hyprl_supercalc.compute_stop_price_py(entry, atr, "long", multiplier)
    
    assert abs(python_stop - rust_stop) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_stop_price_parity_short():
    """Test stop price calculation for short trades."""
    entry = 100.0
    atr = 2.0
    multiplier = 2.0
    
    # Python implementation
    python_stop = compute_stop_price(entry, atr, "short", multiplier)
    
    # Rust implementation
    rust_stop = hyprl_supercalc.compute_stop_price_py(entry, atr, "short", multiplier)
    
    assert abs(python_stop - rust_stop) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_take_profit_parity_long():
    """Test take profit calculation for long trades."""
    entry = 100.0
    atr = 2.0
    multiplier = 2.0
    reward = 2.0
    
    # Python implementation
    python_tp = compute_take_profit_price(entry, atr, "long", multiplier, reward)
    
    # Rust implementation
    rust_tp = hyprl_supercalc.compute_take_profit_price_py(
        entry, atr, "long", multiplier, reward
    )
    
    assert abs(python_tp - rust_tp) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_take_profit_parity_short():
    """Test take profit calculation for short trades."""
    entry = 100.0
    atr = 2.0
    multiplier = 2.0
    reward = 2.0
    
    # Python implementation
    python_tp = compute_take_profit_price(entry, atr, "short", multiplier, reward)
    
    # Rust implementation
    rust_tp = hyprl_supercalc.compute_take_profit_price_py(
        entry, atr, "short", multiplier, reward
    )
    
    assert abs(python_tp - rust_tp) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_simulate_trade_hit_take_profit():
    """Test trade simulation hitting take profit."""
    highs = [105.0, 106.0, 112.0]
    lows = [99.0, 100.0, 108.0]
    closes = [102.0, 103.0, 110.0]
    
    exit_price, exit_idx, pnl = hyprl_supercalc.simulate_trade_py(
        highs, lows, closes, "long", 100.0, 96.0, 110.0, 10
    )
    
    # Should hit TP at 110.0 on bar 2
    assert abs(exit_price - 110.0) < 1e-9
    assert exit_idx == 2
    # PnL = (110 - 100) * 10 = 100
    assert abs(pnl - 100.0) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_simulate_trade_hit_stop_loss():
    """Test trade simulation hitting stop loss."""
    highs = [105.0, 106.0, 97.0]
    lows = [99.0, 100.0, 94.0]
    closes = [102.0, 103.0, 95.0]
    
    exit_price, exit_idx, pnl = hyprl_supercalc.simulate_trade_py(
        highs, lows, closes, "long", 100.0, 96.0, 110.0, 10
    )
    
    # Should hit stop at 96.0 on bar 2
    assert abs(exit_price - 96.0) < 1e-9
    assert exit_idx == 2
    # PnL = (96 - 100) * 10 = -40
    assert abs(pnl - (-40.0)) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_simulate_trade_exit_at_close():
    """Test trade simulation exiting at last close when no stop/TP hit."""
    highs = [105.0, 106.0, 107.0]
    lows = [99.0, 100.0, 101.0]
    closes = [102.0, 103.0, 104.0]
    
    exit_price, exit_idx, pnl = hyprl_supercalc.simulate_trade_py(
        highs, lows, closes, "long", 100.0, 96.0, 110.0, 10
    )
    
    # Should exit at last close
    assert abs(exit_price - 104.0) < 1e-9
    assert exit_idx == 2
    # PnL = (104 - 100) * 10 = 40
    assert abs(pnl - 40.0) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_simulate_trade_short():
    """Test short trade simulation."""
    highs = [102.0, 101.0, 95.0]
    lows = [99.0, 97.0, 90.0]
    closes = [100.0, 98.0, 92.0]
    
    exit_price, exit_idx, pnl = hyprl_supercalc.simulate_trade_py(
        highs, lows, closes, "short", 100.0, 104.0, 92.0, 10
    )
    
    # Should hit TP at 92.0 on bar 2
    assert abs(exit_price - 92.0) < 1e-9
    assert exit_idx == 2
    # PnL = (100 - 92) * 10 = 80
    assert abs(pnl - 80.0) < 1e-9


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_position_size_zero_when_budget_too_small():
    """Test that position size is zero when risk budget is too small."""
    balance = 100.0
    risk_pct = 0.01
    entry_price = 100.0
    stop_price = 95.0  # Large stop, risk per unit = 5
    min_size = 10  # Minimum size too large
    
    size = hyprl_supercalc.compute_position_size_py(
        balance, risk_pct, entry_price, stop_price, min_size
    )
    
    # Risk budget = 100 * 0.01 = 1
    # Size = 1 / 5 = 0.2, floor to 0
    # But 0 < min_size (10), so return 0
    assert size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
