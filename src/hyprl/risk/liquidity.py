"""
Liquidity-Adjusted Sizing Module - Adapt position size to market conditions.

Reduces size during low liquidity periods (open/close) and wide spreads.
"""

from datetime import datetime
from typing import Optional
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class LiquidityManager:
    """Manage position sizing based on liquidity factors."""

    # Maximum % of daily volume we want to be
    MAX_VOLUME_PCT = 0.01  # 1% of daily volume

    # Minimum position size (below this, skip the trade)
    MIN_POSITION_VALUE = 500  # $500 minimum

    def __init__(
        self,
        max_volume_pct: float = 0.01,
        min_position_value: float = 500,
    ):
        """
        Args:
            max_volume_pct: Max percentage of daily volume for our order
            min_position_value: Minimum position value in dollars
        """
        self.max_volume_pct = max_volume_pct
        self.min_position_value = min_position_value
        self._volume_cache: dict[str, tuple[float, datetime]] = {}
        self._spread_cache: dict[str, tuple[float, datetime]] = {}

    def get_time_multiplier(self, dt: datetime = None) -> tuple[float, str]:
        """
        Get position size multiplier based on time of day.

        First and last 30 minutes have wider spreads and more volatility.

        Returns:
            (multiplier: float, reason: str)
        """
        if dt is None:
            et = ZoneInfo("America/New_York")
            dt = datetime.now(et)
        elif dt.tzinfo is None:
            et = ZoneInfo("America/New_York")
            dt = dt.replace(tzinfo=et)

        hour, minute = dt.hour, dt.minute

        # Pre-market or after-hours
        if hour < 9 or (hour == 9 and minute < 30):
            return 0.5, "pre_market"
        if hour >= 16:
            return 0.5, "after_hours"

        # First 30 minutes (9:30-10:00)
        if hour == 9:
            return 0.5, "market_open_30min"

        # 10:00-10:30 - transitioning
        if hour == 10 and minute < 30:
            return 0.7, "market_open_transition"

        # Last 30 minutes (15:30-16:00)
        if hour == 15 and minute >= 30:
            return 0.5, "market_close_30min"

        # Last hour before close (15:00-15:30)
        if hour == 15:
            return 0.8, "approaching_close"

        # Normal trading hours
        return 1.0, "normal_hours"

    def get_volume_limit(self, symbol: str, avg_daily_volume: int) -> int:
        """
        Get maximum shares based on volume limit.

        We don't want to be more than X% of daily volume to avoid
        market impact and slippage.

        Args:
            symbol: Stock symbol
            avg_daily_volume: Average daily volume (shares)

        Returns:
            Maximum shares we should trade
        """
        if avg_daily_volume <= 0:
            logger.warning(f"Invalid avg_daily_volume for {symbol}: {avg_daily_volume}")
            return 100  # Conservative default

        return int(avg_daily_volume * self.max_volume_pct)

    def get_spread_multiplier(
        self,
        current_spread_pct: float,
        avg_spread_pct: float,
    ) -> tuple[float, str]:
        """
        Reduce size if spread is wider than usual.

        Args:
            current_spread_pct: Current bid-ask spread as percentage
            avg_spread_pct: Average spread as percentage

        Returns:
            (multiplier: float, reason: str)
        """
        if avg_spread_pct <= 0:
            return 1.0, "no_spread_data"

        spread_ratio = current_spread_pct / avg_spread_pct

        if spread_ratio > 3.0:
            return 0.3, "spread_3x_normal"
        elif spread_ratio > 2.0:
            return 0.5, "spread_2x_normal"
        elif spread_ratio > 1.5:
            return 0.7, "spread_1.5x_normal"

        return 1.0, "spread_normal"

    def adjust_position_size(
        self,
        requested_qty: int,
        symbol: str,
        price: float,
        avg_daily_volume: int = 0,
        current_spread_pct: float = 0,
        avg_spread_pct: float = 0,
    ) -> tuple[int, dict]:
        """
        Adjust position size based on all liquidity factors.

        Args:
            requested_qty: Originally requested quantity
            symbol: Stock symbol
            price: Current price (for notional calculation)
            avg_daily_volume: Average daily volume
            current_spread_pct: Current spread as percentage
            avg_spread_pct: Average spread as percentage

        Returns:
            (adjusted_qty: int, factors: dict)
        """
        factors = {
            "symbol": symbol,
            "original_qty": requested_qty,
            "original_notional": requested_qty * price,
        }

        adjusted = requested_qty

        # 1. Time of day adjustment
        time_mult, time_reason = self.get_time_multiplier()
        adjusted = int(adjusted * time_mult)
        factors["time_multiplier"] = time_mult
        factors["time_reason"] = time_reason

        # 2. Volume limit
        if avg_daily_volume > 0:
            volume_limit = self.get_volume_limit(symbol, avg_daily_volume)
            if adjusted > volume_limit:
                factors["volume_limited"] = True
                factors["volume_limit"] = volume_limit
                adjusted = volume_limit
            else:
                factors["volume_limited"] = False
                factors["volume_limit"] = volume_limit

        # 3. Spread adjustment
        if current_spread_pct > 0 and avg_spread_pct > 0:
            spread_mult, spread_reason = self.get_spread_multiplier(
                current_spread_pct, avg_spread_pct
            )
            adjusted = int(adjusted * spread_mult)
            factors["spread_multiplier"] = spread_mult
            factors["spread_reason"] = spread_reason
            factors["current_spread_pct"] = current_spread_pct
            factors["avg_spread_pct"] = avg_spread_pct

        # Ensure minimum
        adjusted = max(adjusted, 1)

        # Calculate final values
        factors["final_qty"] = adjusted
        factors["final_notional"] = adjusted * price
        factors["total_reduction_pct"] = round(
            (1 - adjusted / requested_qty) * 100, 1
        ) if requested_qty > 0 else 0

        # Check if below minimum
        if adjusted * price < self.min_position_value:
            factors["below_minimum"] = True
            factors["minimum_notional"] = self.min_position_value
        else:
            factors["below_minimum"] = False

        return adjusted, factors

    def should_skip_trade(
        self,
        adjusted_qty: int,
        price: float,
        factors: dict,
    ) -> tuple[bool, str]:
        """
        Determine if trade should be skipped due to liquidity issues.

        Returns:
            (skip: bool, reason: str)
        """
        notional = adjusted_qty * price

        # Below minimum size
        if notional < self.min_position_value:
            return True, f"below_minimum_notional_{notional:.0f}"

        # Time of day restriction (optional - could just reduce size)
        # if factors.get("time_reason") in ["pre_market", "after_hours"]:
        #     return True, factors["time_reason"]

        # Spread too wide
        if factors.get("spread_multiplier", 1.0) < 0.5:
            return True, f"spread_too_wide_{factors.get('spread_reason', 'unknown')}"

        return False, ""


# Helper functions for fetching volume/spread data
def get_avg_daily_volume(symbol: str, days: int = 20) -> int:
    """Fetch average daily volume for a symbol."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")

        if hist.empty:
            return 0

        return int(hist["Volume"].mean())

    except Exception as e:
        logger.warning(f"Failed to fetch volume for {symbol}: {e}")
        return 0


def get_avg_spread_pct(symbol: str, days: int = 5) -> float:
    """
    Estimate average spread for a symbol.

    Note: yfinance doesn't provide bid/ask data, so we estimate
    from high-low range as a proxy.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")

        if hist.empty:
            return 0.1  # Default 0.1%

        # Use intraday range as proxy for spread
        # This is a rough estimate - real spread data would be better
        avg_range_pct = ((hist["High"] - hist["Low"]) / hist["Close"]).mean() * 100
        # Spread is typically much smaller than daily range
        estimated_spread = avg_range_pct * 0.05  # ~5% of daily range

        return max(0.01, estimated_spread)  # Minimum 0.01%

    except Exception as e:
        logger.warning(f"Failed to estimate spread for {symbol}: {e}")
        return 0.1


# Singleton
_liquidity_instance: Optional[LiquidityManager] = None


def get_liquidity_manager() -> LiquidityManager:
    """Get or create singleton LiquidityManager."""
    global _liquidity_instance
    if _liquidity_instance is None:
        _liquidity_instance = LiquidityManager()
    return _liquidity_instance
