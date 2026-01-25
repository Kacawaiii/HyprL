"""
Event Calendar Module - Track earnings, FOMC, OpEx dates.

Blocks trades during high-risk event windows to avoid surprise gaps.
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# FOMC 2026 meeting dates (decision day is the second day)
FOMC_DATES_2026 = [
    "2026-01-28", "2026-01-29",
    "2026-03-17", "2026-03-18",
    "2026-05-05", "2026-05-06",
    "2026-06-16", "2026-06-17",
    "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16",
    "2026-11-03", "2026-11-04",
    "2026-12-15", "2026-12-16",
]


class EventCalendar:
    """Track market events that affect trading decisions."""

    def __init__(self, earnings_blackout_days: int = 3, post_earnings_days: int = 1):
        """
        Args:
            earnings_blackout_days: Days before earnings to block trades
            post_earnings_days: Days after earnings to block trades
        """
        self.earnings_blackout_days = earnings_blackout_days
        self.post_earnings_days = post_earnings_days
        self._earnings_cache: dict[str, tuple[Optional[datetime], datetime]] = {}

    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Fetch next earnings date for symbol using yfinance."""
        # Check cache first (valid for 6 hours)
        if symbol in self._earnings_cache:
            cached_date, cached_at = self._earnings_cache[symbol]
            if datetime.now() - cached_at < timedelta(hours=6):
                return cached_date

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            if cal is not None and not cal.empty:
                # calendar is a DataFrame with columns like 'Earnings Date'
                if 'Earnings Date' in cal.columns:
                    earnings_dates = cal['Earnings Date']
                    if len(earnings_dates) > 0:
                        earnings_date = earnings_dates.iloc[0]
                        if hasattr(earnings_date, 'to_pydatetime'):
                            earnings_date = earnings_date.to_pydatetime()
                        elif isinstance(earnings_date, str):
                            earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d")
                        self._earnings_cache[symbol] = (earnings_date, datetime.now())
                        return earnings_date
        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {symbol}: {e}")

        self._earnings_cache[symbol] = (None, datetime.now())
        return None

    def days_to_earnings(self, symbol: str) -> int:
        """Return days until next earnings, or 999 if unknown."""
        earnings = self.get_earnings_date(symbol)
        if earnings is None:
            return 999

        now = datetime.now()
        if hasattr(earnings, 'tzinfo') and earnings.tzinfo is not None:
            earnings = earnings.replace(tzinfo=None)

        delta = (earnings - now).days
        return max(delta, -30)  # Cap at -30 for past earnings

    def days_from_earnings(self, symbol: str) -> int:
        """Return days since last earnings, or 999 if unknown."""
        # This would need historical earnings data
        # For now, return 999 (no restriction)
        return 999

    def is_fomc_day(self, date: datetime = None) -> bool:
        """Check if date is a FOMC decision day."""
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        return date_str in FOMC_DATES_2026

    def is_fomc_week(self, date: datetime = None) -> bool:
        """Check if date is within 2 days of a FOMC meeting."""
        date = date or datetime.now()
        for fomc_str in FOMC_DATES_2026:
            fomc_date = datetime.strptime(fomc_str, "%Y-%m-%d")
            if abs((date - fomc_date).days) <= 2:
                return True
        return False

    @staticmethod
    def get_opex_date(year: int, month: int) -> datetime:
        """Get options expiration (3rd Friday) for given month."""
        first_day = datetime(year, month, 1)
        # Find first Friday (weekday 4)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        # Third Friday
        return first_friday + timedelta(weeks=2)

    def is_opex_day(self, date: datetime = None) -> bool:
        """Check if date is options expiration day (3rd Friday)."""
        date = date or datetime.now()
        opex = self.get_opex_date(date.year, date.month)
        return date.date() == opex.date()

    def is_opex_week(self, date: datetime = None) -> bool:
        """Check if date is within 2 days of options expiration."""
        date = date or datetime.now()
        opex = self.get_opex_date(date.year, date.month)
        return abs((date - opex).days) <= 2

    def get_event_data(self, symbol: str) -> dict:
        """Get all event data for a symbol."""
        now = datetime.now()
        return {
            "days_to_earnings": self.days_to_earnings(symbol),
            "days_from_earnings": self.days_from_earnings(symbol),
            "is_fomc_day": self.is_fomc_day(now),
            "is_fomc_week": self.is_fomc_week(now),
            "is_opex_day": self.is_opex_day(now),
            "is_opex_week": self.is_opex_week(now),
        }

    def should_skip_event_risk(self, symbol: str, event_data: dict = None) -> tuple[bool, str]:
        """
        Check if trade should be skipped due to event risk.

        Returns:
            (skip: bool, reason: str)
        """
        if event_data is None:
            event_data = self.get_event_data(symbol)

        # Earnings blackout: N days before
        days_to = event_data.get("days_to_earnings", 999)
        if 0 < days_to <= self.earnings_blackout_days:
            return True, f"earnings_blackout_{days_to}d"

        # Post-earnings blackout
        days_from = event_data.get("days_from_earnings", 999)
        if 0 <= days_from <= self.post_earnings_days:
            return True, f"post_earnings_blackout_{days_from}d"

        # FOMC day: no new positions
        if event_data.get("is_fomc_day", False):
            return True, "fomc_day_blackout"

        # OpEx day: high volatility
        if event_data.get("is_opex_day", False):
            return True, "opex_day_blackout"

        return False, ""


# Singleton for convenience
_calendar_instance: Optional[EventCalendar] = None


def get_event_calendar() -> EventCalendar:
    """Get or create singleton EventCalendar instance."""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = EventCalendar()
    return _calendar_instance
