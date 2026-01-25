"""
Market Regime Detection for HyprL
==================================
Détecte si le marché est:
- TRENDING (suivre la tendance)
- RANGING (mean reversion)
- VOLATILE (réduire sizing ou ne pas trader)

Ajuste automatiquement la stratégie.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float  # 0-1
    adx: float
    volatility_percentile: float
    trend_direction: float  # -1 to 1
    recommendation: str


class RegimeDetector:
    """Detect market regime from price data."""

    def __init__(
        self,
        adx_trending_threshold: float = 25,
        adx_strong_threshold: float = 40,
        volatility_high_percentile: float = 80,
        lookback_period: int = 50
    ):
        self.adx_trending = adx_trending_threshold
        self.adx_strong = adx_strong_threshold
        self.vol_high_pct = volatility_high_percentile
        self.lookback = lookback_period

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect market regime from OHLCV DataFrame.
        
        Expects columns: open, high, low, close, volume
        """
        if len(df) < self.lookback:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0,
                adx=0,
                volatility_percentile=50,
                trend_direction=0,
                recommendation="Insufficient data"
            )

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Calculate ADX for trend strength
        adx = self._calculate_adx(df)
        
        # Calculate volatility percentile
        vol_pct = self._calculate_volatility_percentile(df)
        
        # Calculate trend direction
        trend = self._calculate_trend_direction(df)

        # Determine regime
        regime, confidence, recommendation = self._classify_regime(adx, vol_pct, trend)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            adx=adx,
            volatility_percentile=vol_pct,
            trend_direction=trend,
            recommendation=recommendation
        )

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility as percentile of historical."""
        returns = df['close'].pct_change()
        current_vol = returns.tail(20).std()
        historical_vol = returns.rolling(20).std()
        
        percentile = (historical_vol < current_vol).sum() / len(historical_vol) * 100
        return float(percentile)

    def _calculate_trend_direction(self, df: pd.DataFrame) -> float:
        """Calculate trend direction from -1 (strong down) to 1 (strong up)."""
        close = df['close']
        
        # Multiple timeframe trend
        sma_short = close.rolling(10).mean()
        sma_medium = close.rolling(20).mean()
        sma_long = close.rolling(50).mean()

        # Current vs SMAs
        price = close.iloc[-1]
        
        score = 0
        if price > sma_short.iloc[-1]:
            score += 0.33
        else:
            score -= 0.33
            
        if price > sma_medium.iloc[-1]:
            score += 0.33
        else:
            score -= 0.33
            
        if price > sma_long.iloc[-1]:
            score += 0.34
        else:
            score -= 0.34

        return score

    def _classify_regime(
        self, 
        adx: float, 
        vol_pct: float, 
        trend: float
    ) -> Tuple[MarketRegime, float, str]:
        """Classify regime based on indicators."""
        
        # High volatility = volatile regime
        if vol_pct > self.vol_high_pct:
            return (
                MarketRegime.VOLATILE,
                min(1.0, vol_pct / 100),
                "Réduire taille positions ou attendre"
            )

        # Strong ADX = trending
        if adx > self.adx_trending:
            confidence = min(1.0, adx / self.adx_strong)
            
            if trend > 0.3:
                return (
                    MarketRegime.TRENDING_UP,
                    confidence,
                    "Favoriser les LONG, suivre la tendance"
                )
            elif trend < -0.3:
                return (
                    MarketRegime.TRENDING_DOWN,
                    confidence,
                    "Favoriser les SHORT, suivre la tendance"
                )

        # Low ADX = ranging
        if adx < self.adx_trending:
            return (
                MarketRegime.RANGING,
                min(1.0, (self.adx_trending - adx) / self.adx_trending),
                "Mean reversion, acheter bas / vendre haut"
            )

        return (
            MarketRegime.UNKNOWN,
            0.5,
            "Régime mixte, prudence"
        )

    def get_strategy_adjustment(self, regime: RegimeState) -> Dict[str, Any]:
        """Get recommended strategy adjustments for current regime."""
        
        adjustments = {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "position_size_multiplier": 1.0,
            "favor_direction": None,
            "use_tighter_stops": False,
            "recommendation": regime.recommendation
        }

        if regime.regime == MarketRegime.VOLATILE:
            adjustments["position_size_multiplier"] = 0.5
            adjustments["use_tighter_stops"] = True

        elif regime.regime == MarketRegime.TRENDING_UP:
            adjustments["favor_direction"] = "long"
            adjustments["position_size_multiplier"] = 1.0 + (regime.confidence * 0.25)

        elif regime.regime == MarketRegime.TRENDING_DOWN:
            adjustments["favor_direction"] = "short"
            adjustments["position_size_multiplier"] = 1.0 + (regime.confidence * 0.25)

        elif regime.regime == MarketRegime.RANGING:
            adjustments["position_size_multiplier"] = 0.75  # Smaller in chop

        return adjustments


# Global instance
_detector: Optional[RegimeDetector] = None

def get_regime_detector() -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector


def detect_regime(df: pd.DataFrame) -> RegimeState:
    """Convenience function to detect regime."""
    return get_regime_detector().detect(df)
