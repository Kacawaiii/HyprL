"""
Market Hours Guard for HyprL.
Prevents trading outside regular market hours.
"""

from datetime import datetime, time, timedelta
from typing import Optional, Tuple
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Regular market hours (9:30 AM - 4:00 PM ET)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Pre-market hours (4:00 AM - 9:30 AM ET) - RISKY
PRE_MARKET_OPEN = time(4, 0)

# After-hours (4:00 PM - 8:00 PM ET) - RISKY
AFTER_HOURS_CLOSE = time(20, 0)

# US Market holidays 2026 (update annually)
US_HOLIDAYS_2026 = [
    datetime(2026, 1, 1),   # New Year's Day
    datetime(2026, 1, 19),  # MLK Day
    datetime(2026, 2, 16),  # Presidents Day
    datetime(2026, 4, 3),   # Good Friday
    datetime(2026, 5, 25),  # Memorial Day
    datetime(2026, 7, 3),   # Independence Day (observed)
    datetime(2026, 9, 7),   # Labor Day
    datetime(2026, 11, 26), # Thanksgiving
    datetime(2026, 12, 25), # Christmas
]


class MarketHoursGuard:
    """
    Guards against trading outside market hours.

    CRITICAL: Trading outside regular hours has wider spreads,
    lower liquidity, and higher slippage.
    """

    def __init__(
        self,
        allow_premarket: bool = False,
        allow_afterhours: bool = False,
        buffer_minutes: int = 5
    ):
        """
        Initialize market hours guard.

        Args:
            allow_premarket: Allow trading 4:00-9:30 AM ET (risky)
            allow_afterhours: Allow trading 4:00-8:00 PM ET (risky)
            buffer_minutes: Don't trade within N minutes of open/close
        """
        self.allow_premarket = allow_premarket
        self.allow_afterhours = allow_afterhours
        self.buffer_minutes = buffer_minutes

    def is_market_open(self, check_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if market is currently open.

        Args:
            check_time: Time to check (default: now)

        Returns:
            Tuple of (is_open, reason)
        """
        if check_time is None:
            check_time = datetime.now(ET)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=ET)
        else:
            check_time = check_time.astimezone(ET)

        current_time = check_time.time()
        current_date = check_time.date()

        # Check weekend
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Market closed (weekend)"

        # Check holiday
        for holiday in US_HOLIDAYS_2026:
            if current_date == holiday.date():
                return False, f"Market closed (holiday: {holiday.strftime('%b %d')})"

        # Add buffer
        open_with_buffer = datetime.combine(current_date, MARKET_OPEN) + timedelta(minutes=self.buffer_minutes)
        close_with_buffer = datetime.combine(current_date, MARKET_CLOSE) - timedelta(minutes=self.buffer_minutes)

        check_datetime = datetime.combine(current_date, current_time)

        # Check regular hours
        if open_with_buffer.time() <= current_time <= close_with_buffer.time():
            return True, "Market open (regular hours)"

        # Check pre-market
        if self.allow_premarket and PRE_MARKET_OPEN <= current_time < MARKET_OPEN:
            return True, "Market open (pre-market - RISKY)"

        # Check after-hours
        if self.allow_afterhours and MARKET_CLOSE < current_time <= AFTER_HOURS_CLOSE:
            return True, "Market open (after-hours - RISKY)"

        # Market is closed
        if current_time < MARKET_OPEN:
            return False, f"Market closed (opens at 9:30 AM ET / 15:30 Paris)"
        else:
            return False, f"Market closed (closed at 4:00 PM ET / 22:00 Paris)"

    def can_trade(self, check_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if trading is allowed now.

        Same as is_market_open but with clearer naming for use in guards.
        """
        return self.is_market_open(check_time)

    def get_next_open(self, from_time: Optional[datetime] = None) -> datetime:
        """Get the next market open time."""
        if from_time is None:
            from_time = datetime.now(ET)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=ET)

        current_date = from_time.date()
        current_time = from_time.time()

        # If before today's open, return today's open
        if from_time.weekday() < 5 and current_time < MARKET_OPEN:
            next_open = datetime.combine(current_date, MARKET_OPEN)
            # Check if holiday
            is_holiday = any(h.date() == current_date for h in US_HOLIDAYS_2026)
            if not is_holiday:
                return next_open.replace(tzinfo=ET)

        # Otherwise, find next trading day
        next_day = current_date + timedelta(days=1)
        while True:
            # Skip weekends
            if next_day.weekday() >= 5:
                next_day += timedelta(days=1)
                continue
            # Skip holidays
            if any(h.date() == next_day for h in US_HOLIDAYS_2026):
                next_day += timedelta(days=1)
                continue
            break

        return datetime.combine(next_day, MARKET_OPEN).replace(tzinfo=ET)

    def time_until_open(self, from_time: Optional[datetime] = None) -> timedelta:
        """Get time until market opens."""
        if from_time is None:
            from_time = datetime.now(ET)

        is_open, _ = self.is_market_open(from_time)
        if is_open:
            return timedelta(0)

        next_open = self.get_next_open(from_time)
        return next_open - from_time

    def time_until_close(self, from_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Get time until market closes (None if closed)."""
        if from_time is None:
            from_time = datetime.now(ET)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=ET)

        is_open, _ = self.is_market_open(from_time)
        if not is_open:
            return None

        close_time = datetime.combine(from_time.date(), MARKET_CLOSE).replace(tzinfo=ET)
        return close_time - from_time


# Singleton instance for easy import
market_hours = MarketHoursGuard(
    allow_premarket=False,
    allow_afterhours=False,
    buffer_minutes=5
)


def is_market_open() -> Tuple[bool, str]:
    """Quick check if market is open."""
    return market_hours.is_market_open()


def can_trade() -> Tuple[bool, str]:
    """Quick check if trading is allowed."""
    return market_hours.can_trade()
