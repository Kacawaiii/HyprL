# src/hyprl/risk/dynamic_allocation.py
"""
Dynamic Allocation Module for Strategy 2.0

Implements inverse-volatility (risk parity) allocation:
- Lower vol ticker → higher weight
- Higher vol ticker → lower weight
- Result: equal risk contribution per ticker

Usage:
    from hyprl.risk.dynamic_allocation import DynamicAllocator
    
    allocator = DynamicAllocator(['NVDA', 'MSFT', 'QQQ'])
    weights = allocator.rebalance(prices, current_date)
    size = allocator.get_position_size(ticker, capital, risk_pct, atr, price)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Configuration for dynamic allocation."""
    
    # Volatility calculation window
    vol_window: int = 20
    
    # Weight constraints (avoid extreme concentration)
    min_weight: float = 0.10  # 10% minimum
    max_weight: float = 0.50  # 50% maximum
    
    # Target portfolio volatility (annualized)
    target_vol: float = 0.15  # 15%
    
    # Rebalancing frequency
    rebalance_days: int = 5  # Every 5 days
    
    # Smoothing (avoid abrupt changes)
    smoothing_factor: float = 0.3  # 30% new, 70% old
    
    # Method: 'equal', 'inverse_vol', 'custom'
    method: str = 'inverse_vol'


@dataclass
class AllocationSnapshot:
    """Snapshot of allocation state."""
    timestamp: datetime
    volatilities: Dict[str, float]
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    method: str
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'volatilities': self.volatilities,
            'old_weights': self.old_weights,
            'new_weights': self.new_weights,
            'method': self.method
        }


class DynamicAllocator:
    """
    Allocates capital between tickers based on inverse volatility.
    Simplified risk parity approach.
    """
    
    def __init__(self, 
                 tickers: List[str], 
                 config: Optional[AllocationConfig] = None):
        self.tickers = tickers
        self.config = config or AllocationConfig()
        
        # Initialize equal weights
        n = len(tickers)
        self._current_weights = {t: 1.0 / n for t in tickers}
        self._last_rebalance: Optional[datetime] = None
        self._history: List[AllocationSnapshot] = []
    
    def compute_volatilities(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Compute annualized volatility for each ticker.
        
        Args:
            prices: Dict ticker -> Series of close prices
        
        Returns:
            Dict ticker -> annualized volatility
        """
        vols = {}
        for ticker in self.tickers:
            if ticker not in prices or len(prices[ticker]) < self.config.vol_window:
                # Default vol if insufficient data
                vols[ticker] = 0.30
                logger.warning(f"[ALLOC] {ticker}: insufficient data, using default vol 30%")
                continue
            
            returns = prices[ticker].pct_change().dropna()
            vol = returns.tail(self.config.vol_window).std() * np.sqrt(252)
            vols[ticker] = max(vol, 0.05)  # Floor at 5% to avoid div by zero
        
        return vols
    
    def compute_inverse_vol_weights(self, vols: Dict[str, float]) -> Dict[str, float]:
        """
        Compute weights inversely proportional to volatility.
        
        Args:
            vols: Dict ticker -> volatility
        
        Returns:
            Dict ticker -> normalized weight
        """
        # Inverse vol
        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        
        # Normalize
        total_inv_vol = sum(inv_vols.values())
        raw_weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}
        
        # Apply min/max constraints
        weights = {}
        for ticker, w in raw_weights.items():
            weights[ticker] = np.clip(w, self.config.min_weight, self.config.max_weight)
        
        # Re-normalize after clipping
        total = sum(weights.values())
        weights = {t: w / total for t, w in weights.items()}
        
        return weights
    
    def compute_equal_weights(self) -> Dict[str, float]:
        """Simple equal weighting."""
        n = len(self.tickers)
        return {t: 1.0 / n for t in self.tickers}
    
    def compute_target_weights(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Compute target weights based on configured method.
        """
        if self.config.method == 'equal':
            new_weights = self.compute_equal_weights()
        elif self.config.method == 'inverse_vol':
            vols = self.compute_volatilities(prices)
            new_weights = self.compute_inverse_vol_weights(vols)
        else:
            # Default to current weights
            new_weights = self._current_weights.copy()
        
        # Apply smoothing
        smoothed = {}
        for ticker in self.tickers:
            old_w = self._current_weights.get(ticker, new_weights[ticker])
            new_w = new_weights[ticker]
            smoothed[ticker] = (
                self.config.smoothing_factor * new_w + 
                (1 - self.config.smoothing_factor) * old_w
            )
        
        # Re-normalize
        total = sum(smoothed.values())
        smoothed = {t: w / total for t, w in smoothed.items()}
        
        return smoothed
    
    def should_rebalance(self, current_date: datetime) -> bool:
        """Check if rebalancing is due."""
        if self._last_rebalance is None:
            return True
        
        days_since = (current_date - self._last_rebalance).days
        return days_since >= self.config.rebalance_days
    
    def rebalance(self, 
                  prices: Dict[str, pd.Series], 
                  current_date: datetime) -> Dict[str, float]:
        """
        Execute rebalancing if due.
        
        Returns:
            Dict ticker -> new weight
        """
        if not self.should_rebalance(current_date):
            return self._current_weights
        
        vols = self.compute_volatilities(prices)
        new_weights = self.compute_target_weights(prices)
        
        # Log
        snapshot = AllocationSnapshot(
            timestamp=current_date,
            volatilities=vols.copy(),
            old_weights=self._current_weights.copy(),
            new_weights=new_weights.copy(),
            method=self.config.method
        )
        self._history.append(snapshot)
        
        # Update state
        self._current_weights = new_weights
        self._last_rebalance = current_date
        
        logger.info(f"[ALLOC] Rebalanced: {new_weights}")
        
        return new_weights
    
    def get_weight(self, ticker: str) -> float:
        """Get current weight for a ticker."""
        return self._current_weights.get(ticker, 1.0 / len(self.tickers))
    
    def get_position_size(self, 
                          ticker: str, 
                          total_capital: float,
                          risk_pct: float,
                          atr: float,
                          price: float) -> int:
        """
        Compute position size adjusted by allocation weight.
        
        Args:
            ticker: Symbol
            total_capital: Total portfolio capital
            risk_pct: Risk per trade (e.g., 0.01 = 1%)
            atr: Current ATR for stop calculation
            price: Current price
        
        Returns:
            Number of shares to buy
        """
        weight = self.get_weight(ticker)
        
        # Capital allocated to this ticker
        allocated_capital = total_capital * weight
        
        # Risk amount
        risk_amount = allocated_capital * risk_pct
        
        # Stop distance (1.5 ATR default)
        stop_distance = 1.5 * atr
        
        # Position size
        if stop_distance > 0:
            shares = risk_amount / stop_distance
        else:
            shares = risk_amount / (price * 0.02)  # Fallback 2%
        
        return max(int(shares), 0)
    
    def get_notional_limit(self, ticker: str, total_capital: float) -> float:
        """Get max notional for a ticker based on weight."""
        weight = self.get_weight(ticker)
        return total_capital * weight
    
    @property
    def current_weights(self) -> Dict[str, float]:
        """Current allocation weights."""
        return self._current_weights.copy()
    
    @property
    def last_rebalance(self) -> Optional[datetime]:
        """Last rebalance timestamp."""
        return self._last_rebalance
    
    @property
    def history(self) -> List[AllocationSnapshot]:
        """Rebalancing history."""
        return self._history.copy()
    
    def get_summary(self) -> dict:
        """Get allocation summary."""
        return {
            'method': self.config.method,
            'weights': self._current_weights.copy(),
            'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
            'history_length': len(self._history)
        }
    
    def __repr__(self) -> str:
        weights_str = ', '.join(f"{t}={w:.1%}" for t, w in self._current_weights.items())
        return f"DynamicAllocator({self.config.method}): {weights_str}"


# Convenience function
def compute_risk_parity_weights(prices: Dict[str, pd.Series],
                                tickers: List[str],
                                vol_window: int = 20,
                                min_weight: float = 0.10,
                                max_weight: float = 0.50) -> Dict[str, float]:
    """
    One-shot risk parity weight calculation.
    
    Returns:
        Dict ticker -> weight
    """
    config = AllocationConfig(
        vol_window=vol_window,
        min_weight=min_weight,
        max_weight=max_weight,
        method='inverse_vol'
    )
    allocator = DynamicAllocator(tickers, config)
    return allocator.compute_target_weights(prices)
