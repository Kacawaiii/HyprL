# src/hyprl/regime/classifier.py
"""
Market Regime Classifier for Strategy 2.0

Detects market state (CALM/VOLATILE/CRASH) to adapt:
- risk_pct multiplier
- threshold adjustments
- enabled tickers
- max daily trades
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    CALM = "calm"
    VOLATILE = "volatile"
    CRASH = "crash"


@dataclass
class RegimeConfig:
    """Configuration for regime classification."""
    
    # Volatility thresholds (annualized)
    vol_calm_max: float = 0.15        # <15% = calm
    vol_volatile_max: float = 0.30    # 15-30% = volatile
    # Above = crash
    
    # VIX proxy thresholds
    vix_calm_max: float = 18
    vix_volatile_max: float = 28
    
    # Trend thresholds (20-day return)
    trend_crash_threshold: float = -0.10  # -10% = crash signal
    
    # Lookback periods
    vol_window: int = 20
    correlation_window: int = 30
    
    # Signal weights for classification
    vol_weight: float = 0.5
    vix_weight: float = 0.3
    trend_weight: float = 0.2
    
    # Override thresholds
    severe_dd_threshold: float = -0.08  # -8% recent DD = crash boost


@dataclass
class RegimeAdjustments:
    """Adjustments to apply based on regime."""
    risk_multiplier: float
    threshold_tighten: float
    max_positions: int
    enabled_tickers: List[str]
    max_daily_trades: int
    
    def to_dict(self) -> dict:
        return {
            'risk_multiplier': self.risk_multiplier,
            'threshold_tighten': self.threshold_tighten,
            'max_positions': self.max_positions,
            'enabled_tickers': self.enabled_tickers,
            'max_daily_trades': self.max_daily_trades
        }


# Default adjustments per regime
DEFAULT_ADJUSTMENTS: Dict[MarketRegime, RegimeAdjustments] = {
    MarketRegime.CALM: RegimeAdjustments(
        risk_multiplier=1.0,
        threshold_tighten=0.0,
        max_positions=6,
        enabled_tickers=['NVDA', 'MSFT', 'QQQ'],
        max_daily_trades=50
    ),
    MarketRegime.VOLATILE: RegimeAdjustments(
        risk_multiplier=0.5,
        threshold_tighten=0.05,
        max_positions=4,
        enabled_tickers=['NVDA', 'MSFT', 'QQQ'],
        max_daily_trades=30
    ),
    MarketRegime.CRASH: RegimeAdjustments(
        risk_multiplier=0.25,
        threshold_tighten=0.10,
        max_positions=2,
        enabled_tickers=['QQQ'],  # Only ETF, not single stocks
        max_daily_trades=10
    )
}


@dataclass
class RegimeFeatures:
    """Features used for regime classification."""
    realized_vol: float
    vix: float
    trend_20d: float
    vol_of_vol: float
    recent_max_dd: float
    return_dispersion: float
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    
    def to_dict(self) -> dict:
        return {
            'realized_vol': self.realized_vol,
            'vix': self.vix,
            'trend_20d': self.trend_20d,
            'vol_of_vol': self.vol_of_vol,
            'recent_max_dd': self.recent_max_dd,
            'return_dispersion': self.return_dispersion,
            'timestamp': self.timestamp.isoformat()
        }


class RegimeClassifier:
    """
    Classifies market regime in real-time.
    Used to adapt risk_pct, thresholds, and filters.
    """
    
    def __init__(self, 
                 config: Optional[RegimeConfig] = None,
                 custom_adjustments: Optional[Dict[MarketRegime, RegimeAdjustments]] = None):
        self.config = config or RegimeConfig()
        self.adjustments = custom_adjustments or DEFAULT_ADJUSTMENTS
        self._history: List[dict] = []
        self._current_regime: MarketRegime = MarketRegime.CALM
        self._current_features: Optional[RegimeFeatures] = None
    
    def compute_features(self, 
                         spy_prices: pd.Series,
                         vix_prices: Optional[pd.Series] = None) -> RegimeFeatures:
        """
        Compute regime features from SPY (market proxy).
        
        Args:
            spy_prices: Close prices for SPY (at least 30 days)
            vix_prices: VIX index (optional, otherwise uses realized vol proxy)
        
        Returns:
            RegimeFeatures object
        """
        returns = spy_prices.pct_change().dropna()
        
        # 1. Realized Volatility (annualized)
        if len(returns) >= self.config.vol_window:
            realized_vol = returns.tail(self.config.vol_window).std() * np.sqrt(252)
        else:
            realized_vol = 0.15  # Default
        
        # 2. VIX or proxy
        if vix_prices is not None and len(vix_prices) > 0:
            vix = float(vix_prices.iloc[-1])
        else:
            # Proxy: realized vol * 100 (rough approximation)
            vix = realized_vol * 100
        
        # 3. Trend (20-day return)
        if len(spy_prices) >= 20:
            trend_20d = (spy_prices.iloc[-1] / spy_prices.iloc[-20]) - 1
        else:
            trend_20d = 0.0
        
        # 4. Volatility of Volatility
        if len(returns) >= 30:
            rolling_vol = returns.rolling(5).std()
            vol_of_vol = rolling_vol.tail(20).std()
        else:
            vol_of_vol = 0.0
        
        # 5. Recent Max Drawdown (20 days)
        if len(spy_prices) >= 20:
            recent_prices = spy_prices.tail(20)
            rolling_max = recent_prices.cummax()
            drawdown = (recent_prices / rolling_max) - 1
            recent_max_dd = float(drawdown.min())
        else:
            recent_max_dd = 0.0
        
        # 6. Return dispersion
        return_dispersion = float(returns.tail(10).std()) if len(returns) >= 10 else 0.0
        
        return RegimeFeatures(
            realized_vol=float(realized_vol),
            vix=float(vix),
            trend_20d=float(trend_20d),
            vol_of_vol=float(vol_of_vol),
            recent_max_dd=float(recent_max_dd),
            return_dispersion=float(return_dispersion)
        )
    
    def classify(self, features: RegimeFeatures) -> MarketRegime:
        """
        Classify regime based on computed features.
        
        Returns:
            MarketRegime enum
        """
        scores = {
            MarketRegime.CALM: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CRASH: 0.0
        }
        
        # Score based on realized vol
        vol = features.realized_vol
        if vol < self.config.vol_calm_max:
            scores[MarketRegime.CALM] += self.config.vol_weight
        elif vol < self.config.vol_volatile_max:
            scores[MarketRegime.VOLATILE] += self.config.vol_weight
        else:
            scores[MarketRegime.CRASH] += self.config.vol_weight
        
        # Score based on VIX
        vix = features.vix
        if vix < self.config.vix_calm_max:
            scores[MarketRegime.CALM] += self.config.vix_weight
        elif vix < self.config.vix_volatile_max:
            scores[MarketRegime.VOLATILE] += self.config.vix_weight
        else:
            scores[MarketRegime.CRASH] += self.config.vix_weight
        
        # Score based on trend
        trend = features.trend_20d
        if trend > 0:
            scores[MarketRegime.CALM] += self.config.trend_weight
        elif trend > self.config.trend_crash_threshold:
            scores[MarketRegime.VOLATILE] += self.config.trend_weight
        else:
            scores[MarketRegime.CRASH] += self.config.trend_weight
        
        # Override: crash if severe recent drawdown
        if features.recent_max_dd < self.config.severe_dd_threshold:
            scores[MarketRegime.CRASH] += 0.3
        
        # Return regime with highest score
        regime = max(scores, key=scores.get)
        
        # Log for audit
        self._history.append({
            'timestamp': features.timestamp.isoformat(),
            'features': features.to_dict(),
            'scores': {k.value: v for k, v in scores.items()},
            'regime': regime.value
        })
        
        self._current_regime = regime
        self._current_features = features
        
        logger.info(f"[REGIME] {regime.value.upper()} | vol={features.realized_vol:.1%} vix={features.vix:.1f} trend={features.trend_20d:.1%}")
        
        return regime
    
    def get_adjustments(self, regime: Optional[MarketRegime] = None) -> RegimeAdjustments:
        """
        Get adjustments for a regime.
        
        Args:
            regime: MarketRegime (uses current if None)
        
        Returns:
            RegimeAdjustments object
        """
        regime = regime or self._current_regime
        return self.adjustments[regime]
    
    def detect_and_adjust(self, 
                          spy_prices: pd.Series,
                          vix_prices: Optional[pd.Series] = None) -> Tuple[MarketRegime, RegimeAdjustments]:
        """
        One-shot: compute features, classify, return adjustments.
        
        Returns:
            (regime, adjustments) tuple
        """
        features = self.compute_features(spy_prices, vix_prices)
        regime = self.classify(features)
        adjustments = self.get_adjustments(regime)
        return regime, adjustments
    
    @property
    def current_regime(self) -> MarketRegime:
        """Current classified regime."""
        return self._current_regime
    
    @property
    def current_features(self) -> Optional[RegimeFeatures]:
        """Current regime features."""
        return self._current_features
    
    @property
    def history(self) -> List[dict]:
        """Classification history."""
        return self._history.copy()
    
    def get_summary(self) -> str:
        """One-liner summary for logs."""
        if self._current_features is None:
            return "[REGIME] Not initialized"
        
        f = self._current_features
        adj = self.get_adjustments()
        return (
            f"[REGIME] {self._current_regime.value.upper()} | "
            f"vol={f.realized_vol:.1%} vix={f.vix:.1f} trend={f.trend_20d:.1%} | "
            f"risk_mult={adj.risk_multiplier} thresh_adj={adj.threshold_tighten}"
        )


def apply_regime_to_signal(signal: dict, 
                           regime: MarketRegime, 
                           adjustments: RegimeAdjustments) -> dict:
    """
    Apply regime adjustments to a signal.
    
    Args:
        signal: Signal dict with decision, risk_pct, threshold, symbol, etc.
        regime: Current regime
        adjustments: Adjustments to apply
    
    Returns:
        Modified signal dict
    """
    # Store originals
    signal['risk_pct_original'] = signal.get('risk_pct', 0.01)
    signal['threshold_original'] = signal.get('threshold', 0.5)
    
    # Apply risk multiplier
    signal['risk_pct'] = signal['risk_pct_original'] * adjustments.risk_multiplier
    
    # Apply threshold tightening
    signal['threshold'] = signal['threshold_original'] + adjustments.threshold_tighten
    
    # Check if ticker is enabled
    ticker = signal.get('symbol', '')
    if ticker not in adjustments.enabled_tickers:
        signal['decision'] = 'HOLD'
        signal['reject_reason'] = f'ticker_disabled_in_{regime.value}'
    
    # Tag regime info
    signal['regime'] = regime.value
    signal['regime_risk_multiplier'] = adjustments.risk_multiplier
    signal['regime_threshold_adjust'] = adjustments.threshold_tighten
    
    return signal


# Convenience function for quick regime check
def get_current_regime(spy_prices: pd.Series,
                       vix_prices: Optional[pd.Series] = None,
                       config: Optional[RegimeConfig] = None) -> Tuple[MarketRegime, dict]:
    """
    Quick one-shot regime detection.
    
    Returns:
        (regime, adjustments_dict)
    """
    classifier = RegimeClassifier(config)
    regime, adjustments = classifier.detect_and_adjust(spy_prices, vix_prices)
    return regime, adjustments.to_dict()
