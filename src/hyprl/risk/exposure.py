"""
Portfolio-level exposure management.

This module implements aggregate exposure caps across multiple open positions
to prevent excessive risk concentration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PositionInfo:
    """Information about an open position.
    
    Attributes:
        ticker: Symbol of the position
        qty: Position size (positive for long, negative for short)
        entry_price: Price at which position was entered
    """
    ticker: str
    qty: float
    entry_price: float
    
    @property
    def notional(self) -> float:
        """Returns the notional value (qty * price)."""
        return abs(self.qty * self.entry_price)


class ExposureManager:
    """
    Manages aggregate exposure across multiple open positions.
    
    Purpose:
    --------
    Prevents excessive risk concentration by capping total exposure as a
    percentage of account equity. This is a safety layer on top of per-trade
    ATR-based sizing.
    
    Example:
    --------
    If max_total_exposure=0.10 (10%) and balance=10000:
    - Open position A: $800 notional (8%)
    - Open position B: $150 notional (1.5%)
    - Current total: 9.5%
    - New trade C: $100 notional (1%) → ALLOWED (total would be 10.5%, slightly over but close)
    - New trade D: $200 notional (2%) → REJECTED (total would be 11.5% > 10%)
    
    Note: This is a HARD constraint. If a trade would exceed the cap, it is rejected entirely
    (not scaled down). This ensures predictable risk behavior.
    """
    
    def __init__(self, max_total_exposure: float = 0.10):
        """
        Args:
            max_total_exposure: Maximum allowed exposure as fraction of equity
                                (default: 0.10 = 10%)
        """
        if max_total_exposure <= 0.0 or max_total_exposure > 1.0:
            raise ValueError(f"max_total_exposure must be in (0, 1], got {max_total_exposure}")
        
        self.max_total_exposure = max_total_exposure
        self.open_positions: Dict[str, PositionInfo] = {}
    
    def get_current_exposure_pct(self, balance: float) -> float:
        """Calculate current total exposure as percentage of balance.
        
        Args:
            balance: Current account balance
        
        Returns:
            float: Exposure percentage (e.g., 0.08 = 8%)
        """
        if balance <= 0:
            return 0.0
        total_notional = sum(pos.notional for pos in self.open_positions.values())
        return total_notional / balance
    
    def can_open_trade(
        self,
        ticker: str,
        qty: float,
        price: float,
        balance: float
    ) -> tuple[bool, str]:
        """
        Check if a new trade can be opened without exceeding exposure cap.
        
        Args:
            ticker: Ticker symbol
            qty: Position size (shares/contracts)
            price: Entry price
            balance: Current account balance
        
        Returns:
            tuple: (allowed: bool, reason: str)
                   - (True, "OK") if trade allowed
                   - (False, "exposure cap exceeded: ...") if rejected
        """
        if balance <= 0:
            return False, "invalid balance"
        
        current_exposure_pct = self.get_current_exposure_pct(balance)
        new_notional = abs(qty * price)
        new_exposure_pct = new_notional / balance
        total_exposure_pct = current_exposure_pct + new_exposure_pct
        
        if total_exposure_pct > self.max_total_exposure:
            return False, (
                f"exposure cap exceeded: current={current_exposure_pct:.2%}, "
                f"new={new_exposure_pct:.2%}, total={total_exposure_pct:.2%} "
                f"> max={self.max_total_exposure:.2%}"
            )
        
        return True, "OK"
    
    def add_position(self, ticker: str, qty: float, price: float):
        """Register a newly opened position.
        
        Args:
            ticker: Ticker symbol
            qty: Position size
            price: Entry price
        """
        self.open_positions[ticker] = PositionInfo(ticker, qty, price)
    
    def remove_position(self, ticker: str):
        """Remove a closed position.
        
        Args:
            ticker: Ticker symbol to remove
        """
        self.open_positions.pop(ticker, None)
    
    def get_position(self, ticker: str) -> PositionInfo | None:
        """Get info about an open position, or None if not open.
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            PositionInfo or None
        """
        return self.open_positions.get(ticker)
    
    def reset(self):
        """Clear all open positions. Useful for starting fresh or testing."""
        self.open_positions.clear()
