"""
Macro Regime Detection Module - Adapt strategy based on VIX and market conditions.

Detects risk-on/risk-off regimes and adjusts position sizing and thresholds.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime based on VIX and macro indicators."""
    RISK_ON = "risk_on"       # VIX < 15, SPY trending up
    NEUTRAL = "neutral"       # VIX 15-20, mixed signals
    CAUTIOUS = "cautious"     # VIX 20-25, uncertainty
    RISK_OFF = "risk_off"     # VIX > 25, panic mode
    CRISIS = "crisis"         # VIX > 35, extreme fear


@dataclass
class RegimeConfig:
    """Trading parameters per regime."""
    regime: MarketRegime
    position_size_mult: float     # Multiplier on position size
    threshold_adjustment: float   # Add to probability thresholds
    allow_shorts: bool
    max_positions: int
    description: str


# Configuration per regime
REGIME_CONFIGS = {
    MarketRegime.RISK_ON: RegimeConfig(
        regime=MarketRegime.RISK_ON,
        position_size_mult=1.0,
        threshold_adjustment=0.0,
        allow_shorts=True,
        max_positions=3,
        description="Low volatility, bullish trend - full size",
    ),
    MarketRegime.NEUTRAL: RegimeConfig(
        regime=MarketRegime.NEUTRAL,
        position_size_mult=0.8,
        threshold_adjustment=0.02,
        allow_shorts=True,
        max_positions=2,
        description="Moderate volatility - reduce size 20%",
    ),
    MarketRegime.CAUTIOUS: RegimeConfig(
        regime=MarketRegime.CAUTIOUS,
        position_size_mult=0.5,
        threshold_adjustment=0.05,
        allow_shorts=True,
        max_positions=2,
        description="Elevated volatility - reduce size 50%",
    ),
    MarketRegime.RISK_OFF: RegimeConfig(
        regime=MarketRegime.RISK_OFF,
        position_size_mult=0.3,
        threshold_adjustment=0.10,
        allow_shorts=False,
        max_positions=1,
        description="High volatility - minimal size, no shorts",
    ),
    MarketRegime.CRISIS: RegimeConfig(
        regime=MarketRegime.CRISIS,
        position_size_mult=0.0,
        threshold_adjustment=0.20,
        allow_shorts=False,
        max_positions=0,
        description="Crisis mode - no new trades",
    ),
}


class RegimeDetector:
    """Detect current market regime based on VIX and macro indicators."""

    # VIX thresholds for regime classification
    VIX_CRISIS = 35
    VIX_RISK_OFF = 25
    VIX_CAUTIOUS = 20
    VIX_NEUTRAL = 15

    # VIX spike threshold (% change)
    VIX_SPIKE_THRESHOLD = 20

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Args:
            cache_ttl_seconds: How long to cache regime detection (default 5 min)
        """
        self.cache_ttl = cache_ttl_seconds
        self._cache: Optional[dict] = None
        self._cache_time: Optional[datetime] = None

    def _fetch_vix_data(self) -> dict:
        """Fetch VIX data from yfinance."""
        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y")

            if hist.empty:
                logger.warning("VIX data empty, using defaults")
                return {"vix_level": 20, "vix_percentile_52w": 50, "vix_change_1d": 0}

            current = float(hist["Close"].iloc[-1])
            percentile = float((hist["Close"] < current).mean() * 100)

            # 1-day change
            if len(hist) > 1:
                prev = float(hist["Close"].iloc[-2])
                change_1d = (current / prev - 1) * 100
            else:
                change_1d = 0

            # Term structure (VIX vs VIX3M if available)
            term_structure = 0
            try:
                vix3m = yf.Ticker("^VIX3M")
                vix3m_hist = vix3m.history(period="5d")
                if not vix3m_hist.empty:
                    vix3m_current = float(vix3m_hist["Close"].iloc[-1])
                    term_structure = current - vix3m_current
            except Exception:
                pass

            return {
                "vix_level": round(current, 2),
                "vix_percentile_52w": round(percentile, 1),
                "vix_change_1d": round(change_1d, 2),
                "vix_term_structure": round(term_structure, 2),
            }

        except Exception as e:
            logger.warning(f"Failed to fetch VIX data: {e}")
            return {
                "vix_level": 20,
                "vix_percentile_52w": 50,
                "vix_change_1d": 0,
                "vix_term_structure": 0,
            }

    def _fetch_spy_data(self) -> dict:
        """Fetch SPY trend data."""
        try:
            import yfinance as yf

            spy = yf.Ticker("SPY")
            hist = spy.history(period="1y")

            if hist.empty:
                return {"spy_trend_20d": 0, "spy_above_200ma": True}

            current = float(hist["Close"].iloc[-1])
            ma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
            ma200 = float(hist["Close"].rolling(200).mean().iloc[-1])

            trend_20d = (current / ma20 - 1) * 100

            return {
                "spy_level": round(current, 2),
                "spy_trend_20d": round(trend_20d, 2),
                "spy_above_200ma": current > ma200,
                "spy_ma200": round(ma200, 2),
            }

        except Exception as e:
            logger.warning(f"Failed to fetch SPY data: {e}")
            return {
                "spy_level": 0,
                "spy_trend_20d": 0,
                "spy_above_200ma": True,
                "spy_ma200": 0,
            }

    def _fetch_rates_data(self) -> dict:
        """Fetch interest rate data (10-year yield)."""
        try:
            import yfinance as yf

            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1mo")

            if hist.empty:
                return {"rate_10y": 4.0, "rate_10y_change_5d": 0}

            current = float(hist["Close"].iloc[-1])
            if len(hist) >= 6:
                change_5d = current - float(hist["Close"].iloc[-6])
            else:
                change_5d = 0

            return {
                "rate_10y": round(current, 2),
                "rate_10y_change_5d": round(change_5d, 3),
            }

        except Exception as e:
            logger.warning(f"Failed to fetch rates data: {e}")
            return {"rate_10y": 4.0, "rate_10y_change_5d": 0}

    def detect_regime(self, force_refresh: bool = False) -> tuple[MarketRegime, dict]:
        """
        Detect current market regime.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            (regime: MarketRegime, macro_data: dict)
        """
        # Check cache
        if not force_refresh and self._cache_time:
            elapsed = (datetime.now() - self._cache_time).seconds
            if elapsed < self.cache_ttl and self._cache:
                return self._cache["regime"], self._cache["data"]

        # Fetch all data
        vix = self._fetch_vix_data()
        spy = self._fetch_spy_data()
        rates = self._fetch_rates_data()

        all_data = {**vix, **spy, **rates, "timestamp": datetime.now().isoformat()}

        # Determine regime from VIX level
        vix_level = vix["vix_level"]

        if vix_level >= self.VIX_CRISIS:
            regime = MarketRegime.CRISIS
        elif vix_level >= self.VIX_RISK_OFF:
            regime = MarketRegime.RISK_OFF
        elif vix_level >= self.VIX_CAUTIOUS:
            regime = MarketRegime.CAUTIOUS
        elif vix_level >= self.VIX_NEUTRAL:
            regime = MarketRegime.NEUTRAL
        else:
            regime = MarketRegime.RISK_ON

        # Adjust for VIX spike (upgrade risk level)
        if vix["vix_change_1d"] > self.VIX_SPIKE_THRESHOLD:
            regime_order = [
                MarketRegime.RISK_ON,
                MarketRegime.NEUTRAL,
                MarketRegime.CAUTIOUS,
                MarketRegime.RISK_OFF,
                MarketRegime.CRISIS,
            ]
            idx = regime_order.index(regime)
            if idx < len(regime_order) - 1:
                regime = regime_order[idx + 1]
                logger.info(f"VIX spike detected ({vix['vix_change_1d']:.1f}%), upgrading regime to {regime.value}")

        # Adjust for SPY below 200MA (downgrade from risk_on)
        if not spy.get("spy_above_200ma", True) and regime == MarketRegime.RISK_ON:
            regime = MarketRegime.NEUTRAL
            logger.info("SPY below 200MA, downgrading from RISK_ON to NEUTRAL")

        # Cache results
        self._cache = {"regime": regime, "data": all_data}
        self._cache_time = datetime.now()

        logger.info(f"Regime detected: {regime.value} (VIX={vix_level})")
        return regime, all_data

    def get_regime_config(self, regime: MarketRegime = None) -> RegimeConfig:
        """Get trading configuration for current or specified regime."""
        if regime is None:
            regime, _ = self.detect_regime()
        return REGIME_CONFIGS[regime]

    def should_skip_trading(self) -> tuple[bool, str]:
        """Check if trading should be skipped entirely."""
        regime, data = self.detect_regime()

        if regime == MarketRegime.CRISIS:
            return True, f"crisis_mode_vix_{data['vix_level']}"

        return False, ""

    def adjust_parameters(
        self,
        long_threshold: float,
        short_threshold: float,
        position_size: float,
    ) -> dict:
        """
        Adjust trading parameters based on current regime.

        Returns:
            Dict with adjusted parameters
        """
        regime, data = self.detect_regime()
        config = REGIME_CONFIGS[regime]

        return {
            "regime": regime.value,
            "original_long_threshold": long_threshold,
            "original_short_threshold": short_threshold,
            "original_position_size": position_size,
            "adjusted_long_threshold": long_threshold + config.threshold_adjustment,
            "adjusted_short_threshold": short_threshold + config.threshold_adjustment,
            "adjusted_position_size": position_size * config.position_size_mult,
            "allow_shorts": config.allow_shorts,
            "max_positions": config.max_positions,
            "threshold_adjustment": config.threshold_adjustment,
            "size_multiplier": config.position_size_mult,
            "macro_data": data,
        }


# Singleton
_detector_instance: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    """Get or create singleton RegimeDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RegimeDetector()
    return _detector_instance
