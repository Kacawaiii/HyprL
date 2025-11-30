"""
Tests for portfolio-level exposure management.
"""
from __future__ import annotations

import pytest

from hyprl.risk.exposure import ExposureManager, PositionInfo


def test_exposure_manager_basic():
    """Test basic exposure calculation and cap enforcement."""
    manager = ExposureManager(max_total_exposure=0.10)  # 10% cap
    balance = 10000.0
    
    # Start with no positions
    assert manager.get_current_exposure_pct(balance) == 0.0
    
    # Add position A: $800 notional (8%)
    allowed, _ = manager.can_open_trade('AAPL', 10, 80.0, balance)
    assert allowed, "First trade should be allowed"
    manager.add_position('AAPL', 10, 80.0)
    
    assert abs(manager.get_current_exposure_pct(balance) - 0.08) < 0.001
    
    # Add position B: $150 notional (1.5%) - total 9.5%, OK
    allowed, _ = manager.can_open_trade('MSFT', 5, 30.0, balance)
    assert allowed, "Second trade should be allowed (total 9.5% < 10%)"
    manager.add_position('MSFT', 5, 30.0)
    
    current = manager.get_current_exposure_pct(balance)
    assert abs(current - 0.095) < 0.001, f"Expected 9.5%, got {current:.2%}"
    
    # Try to add position C: $200 notional (2%) - total would be 11.5%, REJECT
    allowed, reason = manager.can_open_trade('GOOGL', 4, 50.0, balance)
    assert not allowed, "Third trade should be rejected (exceeds cap)"
    assert "exposure cap exceeded" in reason


def test_exposure_manager_position_close():
    """Test that closing positions reduces exposure."""
    manager = ExposureManager(max_total_exposure=0.10)
    balance = 10000.0
    
    # Open two positions
    manager.add_position('AAPL', 10, 80.0)   # $800 (8%)
    manager.add_position('MSFT', 5, 30.0)    # $150 (1.5%)
    
    initial_exposure = manager.get_current_exposure_pct(balance)
    assert abs(initial_exposure - 0.095) < 0.001  # 9.5%
    
    # Close position A
    manager.remove_position('AAPL')
    
    new_exposure = manager.get_current_exposure_pct(balance)
    assert abs(new_exposure - 0.015) < 0.001, f"Expected 1.5%, got {new_exposure:.2%}"
    
    # Now a larger trade should be allowed
    allowed, _ = manager.can_open_trade('GOOGL', 15, 50.0, balance)  # $750 = 7.5%
    assert allowed, "Trade should be allowed after closing position A (total would be 9%)"


def test_exposure_manager_multiple_trades_rejection():
    """Test that multiple small trades are correctly rejected when they exceed cap."""
    manager = ExposureManager(max_total_exposure=0.10)
    balance = 10000.0
    
    # Open 5 positions of $200 each (2% each)
    for i in range(5):
        ticker = f'STOCK{i}'
        allowed, _ = manager.can_open_trade(ticker, 2, 100.0, balance)
        assert allowed, f"Trade {i} should be allowed (total {(i+1)*2}%)"
        manager.add_position(ticker, 2, 100.0)
    
    # Total exposure is now 10% (at cap)
    exposure = manager.get_current_exposure_pct(balance)
    assert abs(exposure - 0.10) < 0.001, f"Expected 10%, got {exposure:.2%}"
    
    # Try to open one more - should be rejected
    allowed, reason = manager.can_open_trade('STOCK6', 2, 100.0, balance)
    assert not allowed, "6th trade should be rejected (would exceed 10%)"
    assert "exposure cap exceeded" in reason


def test_exposure_manager_short_positions():
    """Test that short positions (negative qty) are handled correctly."""
    manager = ExposureManager(max_total_exposure=0.10)
    balance = 10000.0
    
    # Open short position: -10 shares at $50 = $500 notional (5%)
    allowed, _ = manager.can_open_trade('AAPL', -10, 50.0, balance)
    assert allowed
    manager.add_position('AAPL', -10, 50.0)
    
    # Notional should be absolute value
    pos = manager.get_position('AAPL')
    assert pos is not None
    assert pos.notional == 500.0, "Short position notional should be positive"
    
    exposure = manager.get_current_exposure_pct(balance)
    assert abs(exposure - 0.05) < 0.001, f"Expected 5%, got {exposure:.2%}"


def test_exposure_manager_zero_balance():
    """Test graceful handling of zero or negative balance."""
    manager = ExposureManager(max_total_exposure=0.10)
    
    # Zero balance
    allowed, reason = manager.can_open_trade('AAPL', 10, 50.0, 0.0)
    assert not allowed
    assert "invalid balance" in reason
    
    # Negative balance
    allowed, reason = manager.can_open_trade('AAPL', 10, 50.0, -1000.0)
    assert not allowed
    assert "invalid balance" in reason


def test_exposure_manager_invalid_cap():
    """Test that invalid max_total_exposure raises ValueError."""
    with pytest.raises(ValueError, match="max_total_exposure must be in"):
        ExposureManager(max_total_exposure=0.0)
    
    with pytest.raises(ValueError, match="max_total_exposure must be in"):
        ExposureManager(max_total_exposure=-0.1)
    
    with pytest.raises(ValueError, match="max_total_exposure must be in"):
        ExposureManager(max_total_exposure=1.5)


def test_exposure_manager_reset():
    """Test that reset() clears all positions."""
    manager = ExposureManager(max_total_exposure=0.10)
    balance = 10000.0
    
    # Add some positions
    manager.add_position('AAPL', 10, 100.0)
    manager.add_position('MSFT', 5, 200.0)
    
    assert manager.get_current_exposure_pct(balance) > 0
    
    # Reset
    manager.reset()
    
    assert manager.get_current_exposure_pct(balance) == 0.0
    assert manager.get_position('AAPL') is None
    assert manager.get_position('MSFT') is None


def test_position_info_notional():
    """Test PositionInfo.notional property."""
    # Long position
    pos_long = PositionInfo('AAPL', 10, 50.0)
    assert pos_long.notional == 500.0
    
    # Short position (notional should be positive)
    pos_short = PositionInfo('MSFT', -5, 100.0)
    assert pos_short.notional == 500.0
