# src/hyprl/regime/__init__.py
"""
Market Regime Detection Module

Provides real-time classification of market conditions:
- CALM: Low volatility, stable trend
- VOLATILE: Elevated volatility, uncertain direction
- CRASH: Extreme volatility, severe drawdown

Usage:
    from hyprl.regime import RegimeClassifier, MarketRegime
    
    classifier = RegimeClassifier()
    regime, adjustments = classifier.detect_and_adjust(spy_prices)
"""

from .classifier import (
    MarketRegime,
    RegimeConfig,
    RegimeAdjustments,
    RegimeFeatures,
    RegimeClassifier,
    apply_regime_to_signal,
    get_current_regime,
    DEFAULT_ADJUSTMENTS
)

__all__ = [
    'MarketRegime',
    'RegimeConfig',
    'RegimeAdjustments',
    'RegimeFeatures',
    'RegimeClassifier',
    'apply_regime_to_signal',
    'get_current_regime',
    'DEFAULT_ADJUSTMENTS'
]
