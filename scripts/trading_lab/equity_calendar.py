"""US equity regular sessions, from a real calendar rather than from guesses.

The crypto calendar answers "always open" and that is the whole truth about
it. A US equity calendar is the opposite: Thanksgiving moves, the day after it
closes at 13:00 Eastern, Juneteenth became a holiday in 2021, Good Friday is
closed while the bond market sometimes is not, and the UTC offset of the open
changes twice a year. Every one of those facts has an effective date, and a
hand-written table of them is wrong the moment one of them changes.

So none of it is written here. ``pandas_market_calendars`` supplies the
schedule and this module adapts it to the ``TradingCalendar`` interface the
platform already has. What *is* written here is the identity: which library,
which version, which calendar, which session type produced a given schedule --
because a series captured under one set of rules and a series captured under
another are not the same series, and only a recorded hash makes that visible.

**No fallback.** An unimplemented calendar raises. A US equity instrument that
quietly borrowed the crypto calendar would be annualised by 8760 instead of
about 3263, and its Sharpe ratio would come out roughly 1.6x too large --
plausible, wrong, and invisible.

This module is an optional extra. The crypto core must import without pandas,
which is why nothing imports it at module scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from scripts.trading_lab.instruments import Timeframe
from scripts.trading_lab.trading_calendar import (
    US_EQUITY_REGULAR,
    TradingCalendar,
    TradingCalendarError,
)

EQUITY_CALENDAR_SCHEMA_VERSION = "trading-lab.equity-calendar.v1"

# The identity of the rules, not a preference. Pinned exactly: a different
# version can move a holiday, and a series captured under each is a different
# series even when the code is identical.
CALENDAR_PROVIDER = "pandas_market_calendars"
CALENDAR_PROVIDER_VERSION = "5.4.0"

# XNYS is the registered MIC for the US equity regular session in this library.
# NASDAQ resolves to the same rules -- verified, not assumed: both produce
# identical 2026 schedules, 251 sessions apiece. A NASDAQ-listed instrument
# therefore uses this calendar, and the venue stays XNAS on the instrument.
US_EQUITY_CALENDAR_NAME = "XNYS"

SESSION_REGULAR = "REGULAR"

# V1 covers the regular session only. Pre-market and after-hours have their own
# liquidity, their own halts and their own reference prices; claiming them here
# would mean claiming bars this platform has never validated.
SUPPORTED_SESSION_TYPES = (SESSION_REGULAR,)


class EquityCalendarError(TradingCalendarError):
    """Raised when the equity calendar cannot answer safely."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def require_market_calendars():
    """Import the calendar library, or say exactly which extra is missing."""
    try:
        import pandas_market_calendars as market_calendars
    except ImportError as error:                     # pragma: no cover - env
        raise EquityCalendarError(
            "US equity sessions need the optional [equities] extra: "
            f"pip install 'pandas_market_calendars=={CALENDAR_PROVIDER_VERSION}'. "
            "The crypto core does not require it."
        ) from error
    installed = getattr(market_calendars, "__version__", None)
    if installed != CALENDAR_PROVIDER_VERSION:
        raise EquityCalendarError(
            f"{CALENDAR_PROVIDER} {installed!r} is installed but this build "
            f"pins {CALENDAR_PROVIDER_VERSION}. A different version can move a "
            "holiday, which silently changes every schedule derived from it.")
    return market_calendars


@dataclass(frozen=True)
class USEquityRegularCalendarSpec:
    """Which rules produced a schedule. Hashed so a series can name them."""

    calendar_provider: str = CALENDAR_PROVIDER
    calendar_provider_version: str = CALENDAR_PROVIDER_VERSION
    calendar_name: str = US_EQUITY_CALENDAR_NAME
    calendar_id: str = US_EQUITY_REGULAR
    timezone: str = "America/New_York"
    session_type: str = SESSION_REGULAR

    def __post_init__(self):
        if self.session_type not in SUPPORTED_SESSION_TYPES:
            raise EquityCalendarError(
                f"session type {self.session_type!r} is not implemented; "
                f"supported: {list(SUPPORTED_SESSION_TYPES)}")

    def canonical(self) -> dict:
        return {
            "schema_version": EQUITY_CALENDAR_SCHEMA_VERSION,
            "calendar_provider": self.calendar_provider,
            "calendar_provider_version": self.calendar_provider_version,
            "calendar_name": self.calendar_name,
            "calendar_id": self.calendar_id,
            "timezone": self.timezone,
            "session_type": self.session_type,
        }

    @property
    def spec_hash(self) -> str:
        return _sha256(self.canonical())


US_EQUITY_REGULAR_SPEC = USEquityRegularCalendarSpec()


@dataclass(frozen=True)
class TradingSession:
    """One real session, in UTC, with the exchange's own local date."""

    session_date: str
    open_at: datetime
    close_at: datetime
    session_type: str = SESSION_REGULAR

    @property
    def duration(self) -> timedelta:
        return self.close_at - self.open_at

    @property
    def early_close(self) -> bool:
        """Shorter than a full regular session. Decided by the calendar."""
        return self.duration < timedelta(hours=6, minutes=30)

    def payload(self) -> dict:
        return {
            "session_date": self.session_date,
            "open_at": _iso(self.open_at),
            "close_at": _iso(self.close_at),
            "session_type": self.session_type,
            "duration_seconds": int(self.duration.total_seconds()),
            "early_close": self.early_close,
        }


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_utc(value: object, *, field: str = "timestamp") -> datetime:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise EquityCalendarError(f"{field} {value!r} is not a timestamp") from error
    else:
        raise EquityCalendarError(f"{field} must be a datetime or ISO-8601 string")
    if moment.tzinfo is None:
        # A naive timestamp near an open is ambiguous by exactly the amount
        # that decides whether a bar is inside the session.
        raise EquityCalendarError(f"{field} must carry a timezone")
    return moment.astimezone(timezone.utc)


class USEquityRegularCalendar(TradingCalendar):
    """The US equity regular session, adapted from pandas_market_calendars.

    Holds a schedule window in memory. Queries outside it extend the window
    rather than guessing, and a query for a year nobody asked about costs one
    library call rather than a wrong answer.
    """

    calendar_id = US_EQUITY_REGULAR
    description = "US equity regular session (NYSE/NASDAQ), holidays and early closes"

    def __init__(self, spec: USEquityRegularCalendarSpec = US_EQUITY_REGULAR_SPEC):
        self.spec = spec
        market_calendars = require_market_calendars()
        self._calendar = market_calendars.get_calendar(spec.calendar_name)
        self._sessions: dict[str, TradingSession] = {}
        self._loaded: tuple[date, date] | None = None

    # --- schedule ---------------------------------------------------------

    def _load(self, first: date, last: date) -> None:
        if self._loaded and self._loaded[0] <= first and last <= self._loaded[1]:
            return
        low = min(first, self._loaded[0]) if self._loaded else first
        high = max(last, self._loaded[1]) if self._loaded else last
        schedule = self._calendar.schedule(
            start_date=low.isoformat(), end_date=high.isoformat())
        sessions = {}
        for day, row in schedule.iterrows():
            session_date = str(day)[:10]
            sessions[session_date] = TradingSession(
                session_date=session_date,
                open_at=row["market_open"].to_pydatetime().astimezone(timezone.utc),
                close_at=row["market_close"].to_pydatetime().astimezone(timezone.utc),
                session_type=self.spec.session_type)
        self._sessions = sessions
        self._loaded = (low, high)

    def _window_for(self, moment: datetime, *, span: int = 10) -> None:
        day = moment.date()
        self._load(day - timedelta(days=span), day + timedelta(days=span))

    def sessions_between(self, start: object, end: object) -> tuple[TradingSession, ...]:
        """Every real session in a range. The only way to enumerate them."""
        first = _as_utc(start, field="start").date()
        last = _as_utc(end, field="end").date()
        if last < first:
            raise EquityCalendarError("end precedes start")
        self._load(first, last)
        return tuple(session for key, session in sorted(self._sessions.items())
                     if first.isoformat() <= key <= last.isoformat())

    # --- interface --------------------------------------------------------

    def is_trading_session(self, day: object) -> bool:
        if isinstance(day, str):
            session_date = day[:10]
            parsed = date.fromisoformat(session_date)
        elif isinstance(day, datetime):
            parsed = _as_utc(day).date()
            session_date = parsed.isoformat()
        elif isinstance(day, date):
            parsed, session_date = day, day.isoformat()
        else:
            raise EquityCalendarError("a session date must be a date or ISO string")
        self._load(parsed - timedelta(days=5), parsed + timedelta(days=5))
        return session_date in self._sessions

    def session_for(self, moment: object) -> TradingSession | None:
        """The session containing this instant, or None outside one.

        Outside means overnight, a weekend, a holiday, or after an early
        close -- all ordinary states of a market that is not open.
        """
        instant = _as_utc(moment)
        self._window_for(instant)
        for session in self._sessions.values():
            if session.open_at <= instant < session.close_at:
                return session
        return None

    def session_on(self, day: object) -> TradingSession | None:
        session_date = (day[:10] if isinstance(day, str)
                        else _as_utc(day).date().isoformat()
                        if isinstance(day, datetime) else day.isoformat())
        parsed = date.fromisoformat(session_date)
        self._load(parsed - timedelta(days=5), parsed + timedelta(days=5))
        return self._sessions.get(session_date)

    def is_market_open(self, moment: object) -> bool:
        return self.session_for(moment) is not None

    def is_session_open(self, moment: object) -> bool:
        return self.is_market_open(moment)

    def is_regular_session(self, moment: object) -> bool:
        session = self.session_for(moment)
        return session is not None and session.session_type == SESSION_REGULAR

    def session_open(self, day: object) -> datetime:
        session = self.session_on(day)
        if session is None:
            raise EquityCalendarError(f"{day} is not a trading session")
        return session.open_at

    def session_close(self, day: object) -> datetime:
        session = self.session_on(day)
        if session is None:
            raise EquityCalendarError(f"{day} is not a trading session")
        return session.close_at

    def next_session_open(self, moment: object) -> datetime:
        instant = _as_utc(moment)
        for span in (10, 40, 120):
            self._window_for(instant, span=span)
            upcoming = sorted(session.open_at for session in self._sessions.values()
                              if session.open_at > instant)
            if upcoming:
                return upcoming[0]
        raise EquityCalendarError(       # pragma: no cover - 120 days of closure
            f"no session opens within 120 days of {_iso(instant)}")

    def next_session_close(self, moment: object) -> datetime:
        instant = _as_utc(moment)
        for span in (10, 40, 120):
            self._window_for(instant, span=span)
            upcoming = sorted(session.close_at for session in self._sessions.values()
                              if session.close_at > instant)
            if upcoming:
                return upcoming[0]
        raise EquityCalendarError(       # pragma: no cover
            f"no session closes within 120 days of {_iso(instant)}")

    def next_expected_bar_open(self, moment: object, timeframe: object) -> datetime:
        """The next bar opening on the session grid, never a clock tick.

        Outside a session this is the next session's open, not `moment + 30m`:
        the crypto rule of "always one interval later" is exactly what an
        equity calendar exists to refuse.
        """
        frame = Timeframe.parse(timeframe)
        instant = _as_utc(moment)
        session = self.session_for(instant)
        if session is not None:
            for opening in self.expected_bar_opens(session, frame):
                if opening > instant:
                    return opening
        return self.next_session_open(instant)

    def expected_bar_opens(self, session: TradingSession,
                           timeframe: object) -> tuple[datetime, ...]:
        """Every bar opening a session should contain.

        A bar may never cross the close, so an early close simply yields
        fewer bars -- 7 instead of 13 for a 30-minute grid. Nothing is
        synthesised to make the count match a normal day.

        A daily bar is the exception, and not really an exception at all: for
        an equity, a daily bar *is* the session. It is 6h30 of trading, not 24
        hours of clock, and every vendor publishes it that way. Measuring it
        against a 24-hour interval would make it never fit and report zero
        daily bars a year.
        """
        frame = Timeframe.parse(timeframe)
        duration = frame.duration
        if duration <= timedelta(0):                 # pragma: no cover
            raise EquityCalendarError("timeframe duration must be positive")
        if duration >= timedelta(days=1):
            return (session.open_at,)
        openings, cursor = [], session.open_at
        while cursor + duration <= session.close_at:
            openings.append(cursor)
            cursor = cursor + duration
        return tuple(openings)

    def bar_close_for(self, session: TradingSession, bar_open_at: datetime,
                      timeframe: object) -> datetime:
        """Where a bar opening at this instant ends.

        A daily bar ends at the session close, which is what makes it shorter
        on an early close -- and correct, rather than a 24-hour bar covering
        eighteen hours the market was shut.
        """
        frame = Timeframe.parse(timeframe)
        if frame.duration >= timedelta(days=1):
            return session.close_at
        return bar_open_at + frame.duration

    def bars_per_day(self, timeframe: object) -> int:
        """Bars in a *full* regular session. Early closes have fewer."""
        frame = Timeframe.parse(timeframe)
        if frame.duration >= timedelta(days=1):
            return 1
        full = timedelta(hours=6, minutes=30)
        if full % frame.duration:
            raise EquityCalendarError(
                f"{frame.label} does not divide a 6h30 regular session evenly; "
                "a bar count would be a rounding, not a fact")
        return int(full // frame.duration)

    def sessions_per_year(self, year: int) -> int:
        return len(self.sessions_between(f"{year}-01-01T00:00:00Z",
                                         f"{year}-12-31T23:59:59Z"))

    def annualization_periods(self, timeframe: object, *, year: int = 2026) -> int:
        """Real bars per year: sessions actually in the calendar times bars each.

        Not 252 by convention and emphatically not 8760. Counted from the
        schedule, so a year with an extra holiday reports one session fewer.
        """
        return self.sessions_per_year(year) * self.bars_per_day(timeframe)

    def payload(self) -> dict:
        return {
            "schema_version": EQUITY_CALENDAR_SCHEMA_VERSION,
            "calendar_id": self.calendar_id,
            "description": self.description,
            "spec": self.spec.canonical(),
            "spec_hash": self.spec.spec_hash,
        }


# --- the expected grid, and what a gap actually means ----------------------


@dataclass(frozen=True)
class ExpectedBarGrid:
    """Every bar the calendar says should exist over a range.

    Gap detection compares against this and nothing else. Overnight, a
    weekend, a holiday and the hours after an early close are not gaps: no bar
    was ever expected there. Reusing the crypto rule -- "the next bar is one
    interval later, always" -- would report a gap every single evening.
    """

    calendar_spec_hash: str
    timeframe: str
    openings: tuple[datetime, ...]

    @property
    def count(self) -> int:
        return len(self.openings)

    def contains(self, moment: object) -> bool:
        return _as_utc(moment) in self.openings

    def missing(self, observed) -> tuple[str, ...]:
        """Expected openings with no observed bar. Never forward-filled."""
        seen = {_as_utc(item, field="observed bar") for item in observed}
        return tuple(_iso(opening) for opening in self.openings
                     if opening not in seen)

    def unexpected(self, observed) -> tuple[str, ...]:
        """Observed bars the calendar never expected: outside a session, on a
        holiday, after an early close, or off the grid entirely."""
        expected = set(self.openings)
        return tuple(_iso(_as_utc(item, field="observed bar")) for item in observed
                     if _as_utc(item, field="observed bar") not in expected)


def build_expected_grid(calendar: USEquityRegularCalendar, *, start: object,
                        end: object, timeframe: object) -> ExpectedBarGrid:
    frame = Timeframe.parse(timeframe)
    openings: list[datetime] = []
    for session in calendar.sessions_between(start, end):
        openings.extend(calendar.expected_bar_opens(session, frame))
    return ExpectedBarGrid(calendar_spec_hash=calendar.spec.spec_hash,
                           timeframe=frame.label, openings=tuple(openings))


# --- causal availability ---------------------------------------------------


@dataclass(frozen=True)
class BarAvailability:
    """When a completed bar could first have been used for a decision.

    The same rule as crypto and for the same reason: a bar's high, low and
    close do not exist until it ends, so anything derived from them is
    available at the close and never at the open. Stated as its own object
    because an equity bar's close is decided by the session, not by adding an
    interval to the open.
    """

    bar_open_at: datetime
    bar_close_at: datetime

    @property
    def available_at(self) -> datetime:
        return self.bar_close_at

    def usable_for_decision_at(self, moment: object) -> bool:
        return _as_utc(moment) >= self.bar_close_at

    def payload(self) -> dict:
        return {"bar_open_at": _iso(self.bar_open_at),
                "bar_close_at": _iso(self.bar_close_at),
                "available_at": _iso(self.available_at)}


def bar_availability(calendar: USEquityRegularCalendar, *, bar_open_at: object,
                     timeframe: object) -> BarAvailability:
    """When this bar ends, and therefore when it could first inform a decision.

    The grid membership check below is what keeps an early close correct, and
    it is the only thing that needs to. ``expected_bar_opens`` never emits an
    opening whose bar would outlive the session, so the bar always fits and
    clamping the end to the close here would be a branch that can never run --
    a safety net with nothing under it, and worse than none, because it would
    silently truncate a bar rather than reject a grid that had started
    producing partial ones.
    """
    frame = Timeframe.parse(timeframe)
    opening = _as_utc(bar_open_at, field="bar_open_at")
    session = calendar.session_for(opening)
    if session is None:
        raise EquityCalendarError(
            f"{_iso(opening)} does not fall inside a regular session")
    if opening not in calendar.expected_bar_opens(session, frame):
        raise EquityCalendarError(
            f"{_iso(opening)} is not on the {frame.label} grid for session "
            f"{session.session_date}")
    closing = calendar.bar_close_for(session, opening, frame)
    if closing > session.close_at:                   # pragma: no cover - grid
        raise EquityCalendarError(
            f"a {frame.label} bar opening at {_iso(opening)} would end after "
            f"session {session.session_date} closed; the bar grid is wrong")
    return BarAvailability(bar_open_at=opening, bar_close_at=closing)


def get_us_equity_calendar(
        spec: USEquityRegularCalendarSpec = US_EQUITY_REGULAR_SPEC
) -> USEquityRegularCalendar:
    return USEquityRegularCalendar(spec)


__all__ = [
    "BarAvailability", "CALENDAR_PROVIDER", "CALENDAR_PROVIDER_VERSION",
    "EQUITY_CALENDAR_SCHEMA_VERSION", "EquityCalendarError", "ExpectedBarGrid",
    "SESSION_REGULAR", "SUPPORTED_SESSION_TYPES", "TradingSession",
    "US_EQUITY_CALENDAR_NAME", "US_EQUITY_REGULAR", "US_EQUITY_REGULAR_SPEC",
    "USEquityRegularCalendar", "USEquityRegularCalendarSpec",
    "bar_availability", "build_expected_grid", "get_us_equity_calendar",
    "require_market_calendars",
]
