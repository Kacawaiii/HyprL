"""Market Regime Detection and Risk Adjustment.

Uses VIX and other market indicators to adjust trading behavior.
Works for any portfolio of stocks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import os


class MarketRegime(Enum):
    """Market regime classification."""
    CALM = "calm"           # VIX < 15 - Normal trading
    ELEVATED = "elevated"   # VIX 15-25 - Cautious
    HIGH_VOL = "high_vol"   # VIX 25-35 - Reduce size
    EXTREME = "extreme"     # VIX > 35 - Defensive mode
    UNKNOWN = "unknown"


@dataclass
class RegimeConfig:
    """Configuration for market regime detection."""
    # VIX thresholds
    vix_calm_max: float = 15.0
    vix_elevated_max: float = 25.0
    vix_high_max: float = 35.0

    # Size multipliers per regime
    size_mult_calm: float = 1.0
    size_mult_elevated: float = 0.8
    size_mult_high_vol: float = 0.5
    size_mult_extreme: float = 0.0  # No new trades

    # Threshold adjustments (make model more conservative)
    threshold_adj_calm: float = 0.0
    threshold_adj_elevated: float = 0.02
    threshold_adj_high_vol: float = 0.05
    threshold_adj_extreme: float = 0.10

    # Additional filters
    spy_sma_period: int = 20  # SPY below 20 SMA = bearish
    require_spy_above_sma: bool = False

    # Cache settings
    cache_seconds: int = 300  # Refresh every 5 minutes


@dataclass
class RegimeState:
    """Current market regime state."""
    regime: MarketRegime
    vix_value: float
    size_multiplier: float
    threshold_adjustment: float
    spy_price: Optional[float] = None
    spy_sma: Optional[float] = None
    spy_above_sma: Optional[bool] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""


class MarketRegimeDetector:
    """Detects market regime and provides risk adjustments."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._cache: Optional[RegimeState] = None
        self._cache_time: Optional[datetime] = None

    def _fetch_vix(self) -> Optional[float]:
        """Fetch current VIX value."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass

        # Fallback: try Alpaca if available
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            api_key = os.environ.get("APCA_API_KEY_ID")
            secret_key = os.environ.get("APCA_API_SECRET_KEY")

            if api_key and secret_key:
                client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
                # VIX not available on Alpaca, skip
        except Exception:
            pass

        return None

    def _fetch_spy_data(self) -> tuple[Optional[float], Optional[float]]:
        """Fetch SPY price and SMA."""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")
            if not hist.empty and len(hist) >= self.config.spy_sma_period:
                price = float(hist["Close"].iloc[-1])
                sma = float(hist["Close"].tail(self.config.spy_sma_period).mean())
                return price, sma
        except Exception:
            pass
        return None, None

    def _classify_regime(self, vix: float) -> MarketRegime:
        """Classify market regime based on VIX."""
        if vix < self.config.vix_calm_max:
            return MarketRegime.CALM
        elif vix < self.config.vix_elevated_max:
            return MarketRegime.ELEVATED
        elif vix < self.config.vix_high_max:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.EXTREME

    def _get_adjustments(self, regime: MarketRegime) -> tuple[float, float]:
        """Get size multiplier and threshold adjustment for regime."""
        adjustments = {
            MarketRegime.CALM: (self.config.size_mult_calm, self.config.threshold_adj_calm),
            MarketRegime.ELEVATED: (self.config.size_mult_elevated, self.config.threshold_adj_elevated),
            MarketRegime.HIGH_VOL: (self.config.size_mult_high_vol, self.config.threshold_adj_high_vol),
            MarketRegime.EXTREME: (self.config.size_mult_extreme, self.config.threshold_adj_extreme),
            MarketRegime.UNKNOWN: (0.5, 0.05),  # Conservative default
        }
        return adjustments.get(regime, (1.0, 0.0))

    def get_regime(self, force_refresh: bool = False) -> RegimeState:
        """Get current market regime with caching."""
        now = datetime.now(timezone.utc)

        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            age = (now - self._cache_time).total_seconds()
            if age < self.config.cache_seconds:
                return self._cache

        # Fetch VIX
        vix = self._fetch_vix()

        if vix is None:
            state = RegimeState(
                regime=MarketRegime.UNKNOWN,
                vix_value=0.0,
                size_multiplier=0.5,
                threshold_adjustment=0.05,
                reason="vix_fetch_failed",
                timestamp=now,
            )
            self._cache = state
            self._cache_time = now
            return state

        # Classify regime
        regime = self._classify_regime(vix)
        size_mult, threshold_adj = self._get_adjustments(regime)

        # Optional SPY filter
        spy_price, spy_sma = None, None
        spy_above_sma = None

        if self.config.require_spy_above_sma:
            spy_price, spy_sma = self._fetch_spy_data()
            if spy_price and spy_sma:
                spy_above_sma = spy_price > spy_sma
                if not spy_above_sma:
                    # SPY below SMA, be more conservative
                    size_mult *= 0.7
                    threshold_adj += 0.02

        reason_parts = [f"vix={vix:.1f}", regime.value]
        if spy_above_sma is not None:
            reason_parts.append(f"spy_above_sma={spy_above_sma}")

        state = RegimeState(
            regime=regime,
            vix_value=vix,
            size_multiplier=size_mult,
            threshold_adjustment=threshold_adj,
            spy_price=spy_price,
            spy_sma=spy_sma,
            spy_above_sma=spy_above_sma,
            reason=", ".join(reason_parts),
            timestamp=now,
        )

        self._cache = state
        self._cache_time = now
        return state

    def should_trade(self, regime_state: Optional[RegimeState] = None) -> tuple[bool, str]:
        """Check if trading is allowed in current regime."""
        if regime_state is None:
            regime_state = self.get_regime()

        if regime_state.regime == MarketRegime.EXTREME:
            return False, f"extreme_vix_{regime_state.vix_value:.1f}"

        if regime_state.size_multiplier <= 0:
            return False, "size_mult_zero"

        return True, ""

    def adjust_signal(
        self,
        probability: float,
        base_size: float,
        threshold: float,
        regime_state: Optional[RegimeState] = None,
    ) -> tuple[float, float, bool, str]:
        """Adjust signal based on market regime.

        Returns:
            (adjusted_size, adjusted_threshold, should_trade, reason)
        """
        if regime_state is None:
            regime_state = self.get_regime()

        can_trade, reason = self.should_trade(regime_state)

        if not can_trade:
            return 0.0, threshold, False, reason

        adjusted_size = base_size * regime_state.size_multiplier
        adjusted_threshold = threshold + regime_state.threshold_adjustment

        # Check if probability still meets adjusted threshold
        should_trade = probability >= adjusted_threshold

        return adjusted_size, adjusted_threshold, should_trade, regime_state.reason


# Singleton instance for easy access
_detector: Optional[MarketRegimeDetector] = None


def get_market_regime(config: Optional[RegimeConfig] = None) -> RegimeState:
    """Get current market regime (uses cached singleton)."""
    global _detector
    if _detector is None:
        _detector = MarketRegimeDetector(config)
    return _detector.get_regime()


def adjust_for_regime(
    probability: float,
    base_size: float,
    threshold: float,
    config: Optional[RegimeConfig] = None,
) -> tuple[float, float, bool, str]:
    """Adjust signal for market regime (convenience function)."""
    global _detector
    if _detector is None:
        _detector = MarketRegimeDetector(config)
    return _detector.adjust_signal(probability, base_size, threshold)
