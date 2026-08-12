"""When a market is open, and how many bars a year that is.

Two numbers in this project quietly assume crypto: 24 bars a day, and 8760
periods a year for annualising a Sharpe ratio. Both are correct for a market
that never closes and both are wrong for every other asset class -- an equity
session is about 6.5 hours, roughly 1638 hourly bars a year, and annualising
one with 8760 overstates it by more than twice.

So the number moves out of the arithmetic and onto the calendar. A future
equity calendar returns its own, and no formula has to be found and edited.

``Crypto247Calendar`` is implemented here in full because it needs nothing but
the standard library. The US equity calendar is not, and never will be: real
sessions need a real holiday database, so it lives in ``equity_calendar`` behind
the optional ``[equities]`` extra and is imported only when someone asks for it
by name. What is *not* here is a stub returning 24/7 with a comment promising to
fix it later -- a wrong calendar that runs is far more dangerous than a missing
one that raises, because it produces plausible numbers nobody re-derives.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from scripts.trading_lab.instruments import InstrumentError, Timeframe

TRADING_CALENDAR_SCHEMA_VERSION = "trading-lab.trading-calendar.v1"

CRYPTO_24_7 = "CRYPTO_24_7"
US_EQUITY_REGULAR = "US_EQUITY_REGULAR"

DAYS_PER_YEAR = 365


class TradingCalendarError(RuntimeError):
    """Raised when a calendar cannot answer for the market it was asked about."""


class TradingCalendar:
    """The interface. Subclasses answer for one market convention.

    Deliberately not an ABC with a metaclass: this needs to stay importable in
    a core install with nothing but the standard library, and the guard rails
    that matter here are the tests, not the type system.
    """

    calendar_id: str = "ABSTRACT"
    description: str = ""

    def is_session_open(self, moment: object) -> bool:
        raise NotImplementedError

    def next_expected_bar_open(self, moment: object, timeframe: object) -> datetime:
        raise NotImplementedError

    def bars_per_day(self, timeframe: object) -> int:
        raise NotImplementedError

    def annualization_periods(self, timeframe: object) -> int:
        raise NotImplementedError

    def payload(self) -> dict:
        return {
            "schema_version": TRADING_CALENDAR_SCHEMA_VERSION,
            "calendar_id": self.calendar_id,
            "description": self.description,
        }


def _parse_moment(value: object, *, field: str = "moment") -> datetime:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise TradingCalendarError(f"{field} {value!r} is not a timestamp") from error
    else:
        raise TradingCalendarError(f"{field} must be a datetime or ISO-8601 string")
    if moment.tzinfo is None:
        raise TradingCalendarError(f"{field} must carry a timezone")
    return moment.astimezone(timezone.utc)


@dataclass(frozen=True)
class Crypto247Calendar(TradingCalendar):
    """A market with no sessions, no holidays and no close."""

    calendar_id: str = CRYPTO_24_7
    description: str = "continuous trading, no session boundaries or holidays"

    def is_session_open(self, moment: object) -> bool:
        _parse_moment(moment)                        # still validated, always open
        return True

    def next_expected_bar_open(self, moment: object, timeframe: object) -> datetime:
        """The next grid opening strictly after ``moment``.

        Anchored to the Unix epoch rather than to the argument, so two callers
        asking at different instants agree on where the grid falls.
        """
        frame = Timeframe.parse(timeframe)
        duration = frame.duration
        if duration <= timedelta(0):                 # pragma: no cover - refused earlier
            raise TradingCalendarError("timeframe duration must be positive")
        instant = _parse_moment(moment)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        elapsed = (instant - epoch) // duration
        candidate = epoch + duration * elapsed
        return candidate if candidate > instant else candidate + duration

    def bars_per_day(self, timeframe: object) -> int:
        frame = Timeframe.parse(timeframe)
        day = timedelta(days=1)
        if day % frame.duration:
            raise TradingCalendarError(
                f"{frame.label} does not divide a day evenly; a bar count would "
                "be a rounding, not a fact")
        return day // frame.duration

    def annualization_periods(self, timeframe: object) -> int:
        """Bars per year: 8760 for 1h. What a Sharpe ratio is scaled by."""
        return self.bars_per_day(timeframe) * DAYS_PER_YEAR


CRYPTO_247_CALENDAR = Crypto247Calendar()

_CALENDARS = {CRYPTO_24_7: CRYPTO_247_CALENDAR}

# Calendars that exist but cost a dependency to build. Named here so that
# get_calendar() can tell "not implemented" apart from "implemented, extra not
# installed" -- two very different answers -- without importing anything.
_DEFERRED_CALENDARS = {US_EQUITY_REGULAR: "scripts.trading_lab.equity_calendar"}


def _load_deferred(calendar_id: str) -> TradingCalendar:
    """Import an optional calendar on demand.

    Deliberately inside the function. At module scope this import would drag
    pandas into every crypto process that ever asks what time it is, and the
    core install would stop working.
    """
    import importlib

    module = importlib.import_module(_DEFERRED_CALENDARS[calendar_id])
    return module.get_us_equity_calendar()


def get_calendar(calendar_id: str) -> TradingCalendar:
    """Resolve a calendar, or refuse. Never falls back to 24/7.

    An unknown calendar returning the crypto one "for now" is how an equity
    strategy ends up annualised by 8760 and looking twice as good as it is.
    """
    if calendar_id in _CALENDARS:
        return _CALENDARS[calendar_id]
    if calendar_id in _DEFERRED_CALENDARS:
        return _load_deferred(calendar_id)
    raise TradingCalendarError(
        f"no calendar implemented for {calendar_id!r}; implemented: "
        f"{known_calendars()}. A market whose sessions are unknown must not "
        "borrow another market's.")


def known_calendars() -> tuple[str, ...]:
    """Every calendar that can be resolved, installed or not.

    An optional extra that is missing is a deployment fact, not a reason to
    pretend the calendar does not exist.
    """
    return tuple(sorted({*_CALENDARS, *_DEFERRED_CALENDARS}))


def calendar_available(calendar_id: str) -> bool:
    """Whether this process can actually build the calendar right now."""
    try:
        get_calendar(calendar_id)
    except (TradingCalendarError, ImportError):
        return False
    return True


__all__ = [
    "CRYPTO_24_7", "CRYPTO_247_CALENDAR", "Crypto247Calendar", "InstrumentError",
    "TRADING_CALENDAR_SCHEMA_VERSION", "TradingCalendar", "TradingCalendarError",
    "US_EQUITY_REGULAR", "calendar_available", "get_calendar", "known_calendars",
]
