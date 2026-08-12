"""The US equity calendar, checked against what a market actually does.

The tempting way to test this is to write down the 2026 holidays and assert the
calendar matches. That test passes forever and proves nothing: it compares the
calendar to a second hand-written calendar, and when the real one is wrong both
are wrong together.

So almost nothing here is a hardcoded date. The assertions are properties that
must hold for any correct US equity calendar in any year: a session never
crosses midnight in New York, no session falls on a Saturday, the UTC open
shifts by exactly one hour across a DST boundary, an early close is strictly
shorter than a regular one, a 30-minute grid never emits a bar that outlives
its session, and the number of sessions in a year lands in the range every US
equity year has landed in for decades.

The few concrete dates that do appear are the ones whose *semantics* are the
point -- the day after Thanksgiving must be short, Christmas must be closed --
and they are asserted as properties of that day, not as a table to diff.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip(
    "pandas_market_calendars",
    reason="US equity sessions need the optional [equities] extra")

from scripts.trading_lab.equity_calendar import (  # noqa: E402
    CALENDAR_PROVIDER_VERSION, EquityCalendarError, US_EQUITY_REGULAR_SPEC,
    USEquityRegularCalendarSpec, bar_availability, build_expected_grid,
    get_us_equity_calendar)

NEW_YORK = ZoneInfo("America/New_York")

REGULAR_SESSION = timedelta(hours=6, minutes=30)


@pytest.fixture(scope="module")
def calendar():
    return get_us_equity_calendar()


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# --- identity --------------------------------------------------------------


def test_the_calendar_records_which_rules_produced_it():
    """A schedule without a provenance is a schedule nobody can reproduce."""
    spec = US_EQUITY_REGULAR_SPEC
    payload = spec.canonical()
    assert payload["calendar_provider"] == "pandas_market_calendars"
    assert payload["calendar_provider_version"] == CALENDAR_PROVIDER_VERSION
    assert payload["timezone"] == "America/New_York"
    assert payload["session_type"] == "REGULAR"
    assert len(spec.spec_hash) == 64


def test_a_different_library_version_is_a_different_calendar():
    """A holiday moving between releases must not be invisible.

    The version is in the hash precisely so that a corpus captured under one
    release cannot be compared like-for-like with one captured under another.
    """
    other = USEquityRegularCalendarSpec(calendar_provider_version="9.9.9")
    assert other.spec_hash != US_EQUITY_REGULAR_SPEC.spec_hash


def test_a_session_type_the_platform_has_not_validated_is_refused():
    """Pre-market has its own liquidity and its own reference prices."""
    for session_type in ("PRE_MARKET", "AFTER_HOURS", "EXTENDED", ""):
        with pytest.raises(EquityCalendarError):
            USEquityRegularCalendarSpec(session_type=session_type)


def test_an_installed_version_that_is_not_the_pinned_one_is_refused(monkeypatch):
    import pandas_market_calendars

    from scripts.trading_lab import equity_calendar

    monkeypatch.setattr(pandas_market_calendars, "__version__", "0.0.1",
                        raising=False)
    with pytest.raises(EquityCalendarError) as error:
        equity_calendar.require_market_calendars()
    assert "pins" in str(error.value)


# --- session shape, as properties -----------------------------------------


def test_every_session_in_a_year_is_a_weekday_that_opens_before_it_closes(calendar):
    sessions = calendar.sessions_between("2026-01-01T00:00:00Z",
                                         "2026-12-31T23:59:59Z")
    assert sessions
    for session in sessions:
        assert session.close_at > session.open_at
        local_open = session.open_at.astimezone(NEW_YORK)
        local_close = session.close_at.astimezone(NEW_YORK)
        # A regular session opens and closes on the same New York day. In UTC
        # it does not always: an 16:00 EST close is 21:00 UTC, and a calendar
        # reasoning in UTC dates would put some closes on the wrong day.
        assert local_open.date() == local_close.date()
        assert local_open.date().isoformat() == session.session_date
        assert local_open.weekday() < 5, session.session_date
        assert session.duration <= REGULAR_SESSION


def test_a_year_holds_the_number_of_sessions_a_us_equity_year_holds(calendar):
    """About 252. Never 365, never 260, and it is counted, not assumed."""
    for year in (2024, 2025, 2026, 2027):
        count = calendar.sessions_per_year(year)
        assert 248 <= count <= 254, (year, count)
        weekdays = sum(1 for offset in range((date(year, 12, 31)
                                              - date(year, 1, 1)).days + 1)
                       if (date(year, 1, 1) + timedelta(days=offset)).weekday() < 5)
        # Holidays only ever remove sessions from the weekday count.
        assert count < weekdays


def test_no_session_exists_on_a_weekend(calendar):
    for saturday in ("2026-01-17", "2026-07-04", "2026-11-28"):
        assert not calendar.is_trading_session(saturday)
        assert calendar.session_on(saturday) is None


# --- daylight saving -------------------------------------------------------


def test_the_utc_open_moves_by_exactly_one_hour_across_dst(calendar):
    """The local open never moves; the UTC one does. Both must be true.

    A calendar that stored 14:30 UTC as "the open" would be right in January
    and an hour wrong from March, and every summer bar would be misaligned
    by exactly one bar on a 30-minute grid.
    """
    winter = calendar.session_on("2026-01-15")
    summer = calendar.session_on("2026-07-15")
    assert winter is not None and summer is not None

    local = {session.open_at.astimezone(NEW_YORK).strftime("%H:%M")
             for session in (winter, summer)}
    assert local == {"09:30"}

    utc_opens = {session.open_at.strftime("%H:%M")
                 for session in (winter, summer)}
    assert len(utc_opens) == 2
    difference = abs(winter.open_at.hour - summer.open_at.hour)
    assert difference == 1
    assert winter.duration == summer.duration == REGULAR_SESSION


def test_the_session_length_is_unchanged_by_dst(calendar):
    """The clock changing must not create or destroy half an hour of trading."""
    march = calendar.sessions_between("2026-03-05T00:00:00Z",
                                      "2026-03-15T23:59:59Z")
    november = calendar.sessions_between("2026-10-28T00:00:00Z",
                                         "2026-11-07T23:59:59Z")
    for session in (*march, *november):
        assert session.duration == REGULAR_SESSION, session.session_date


# --- holidays and early closes, by semantics ------------------------------


def test_christmas_and_thanksgiving_are_closed_and_the_day_after_is_short(calendar):
    """Asserted as properties of those days, not as a copied holiday table."""
    thanksgiving = _fourth_thursday(2026, 11)
    assert not calendar.is_trading_session(thanksgiving)

    day_after = thanksgiving + timedelta(days=1)
    short = calendar.session_on(day_after.isoformat())
    assert short is not None, "the day after Thanksgiving is a trading day"
    assert short.early_close
    assert short.duration < REGULAR_SESSION

    # Christmas, whichever weekday it falls on.
    christmas = date(2026, 12, 25)
    if christmas.weekday() < 5:
        assert not calendar.is_trading_session(christmas)


def test_juneteenth_is_closed_because_the_library_knows_it_is(calendar):
    """A holiday added in 2021. Hand-written tables are exactly where it is missed."""
    for year in (2023, 2024, 2026):
        observed = _observed(date(year, 6, 19))
        assert not calendar.is_trading_session(observed), year
    # And it was an ordinary session before it was a holiday.
    assert calendar.is_trading_session(_observed(date(2019, 6, 19)))


def test_an_early_close_is_strictly_shorter_and_still_a_real_session(calendar):
    early = [session for session
             in calendar.sessions_between("2026-01-01T00:00:00Z",
                                          "2026-12-31T23:59:59Z")
             if session.early_close]
    assert early, "a US equity year has at least one early close"
    for session in early:
        assert session.duration < REGULAR_SESSION
        assert session.duration > timedelta(0)
        assert calendar.is_market_open(session.open_at)
        # Still open at its own open; shut at its own close, which is what
        # makes it early rather than merely unusual.
        assert not calendar.is_market_open(session.close_at)
        assert not calendar.is_market_open(
            session.open_at + REGULAR_SESSION - timedelta(minutes=1))


def _fourth_thursday(year: int, month: int) -> date:
    thursdays = [date(year, month, day)
                 for day in range(1, 31)
                 if date(year, month, day).weekday() == 3]
    return thursdays[3]


def _observed(day: date) -> date:
    """US market holiday observation: Saturday -> Friday, Sunday -> Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


# --- open and closed -------------------------------------------------------


def test_the_market_is_shut_overnight_on_weekends_and_on_holidays(calendar):
    session = calendar.session_on("2026-01-15")
    assert calendar.is_market_open(session.open_at)
    assert calendar.is_market_open(session.close_at - timedelta(minutes=1))
    # The close is exclusive: at 16:00:00 the market is shut.
    assert not calendar.is_market_open(session.close_at)
    assert not calendar.is_market_open(session.open_at - timedelta(minutes=1))
    for shut in ("2026-01-17T18:00:00Z", "2026-12-25T15:00:00Z",
                 "2026-01-15T03:00:00Z"):
        assert not calendar.is_market_open(shut)


def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc(calendar):
    """The assumption would be wrong by exactly the offset that decides a session."""
    with pytest.raises(EquityCalendarError):
        calendar.is_market_open(datetime(2026, 1, 15, 15, 0))
    with pytest.raises(EquityCalendarError):
        calendar.is_market_open("2026-01-15T15:00:00")


def test_the_next_open_skips_the_weekend_and_the_holiday_behind_it(calendar):
    """Friday evening plus a Monday holiday lands on Tuesday, not on Saturday."""
    friday_evening = _utc("2026-01-16T21:05:00Z")
    following = calendar.next_session_open(friday_evening)
    assert following.astimezone(NEW_YORK).weekday() == 1     # Tuesday
    assert not calendar.is_trading_session(
        (friday_evening + timedelta(days=3)).date())          # MLK Monday
    assert calendar.is_market_open(following)


# --- the 30 minute grid ----------------------------------------------------


def test_a_regular_session_holds_exactly_thirteen_thirty_minute_bars(calendar):
    session = calendar.session_on("2026-01-15")
    openings = calendar.expected_bar_opens(session, "30m")
    assert len(openings) == 13
    assert openings[0] == session.open_at
    assert openings[-1] + timedelta(minutes=30) == session.close_at
    assert calendar.bars_per_day("30m") == 13
    # Contiguous, with no invented gap between them.
    for earlier, later in zip(openings, openings[1:]):
        assert later - earlier == timedelta(minutes=30)


def test_no_bar_outlives_the_session_that_contains_it(calendar):
    """The property that makes an early close correct rather than special-cased."""
    for day in calendar.sessions_between("2026-11-20T00:00:00Z",
                                         "2026-12-31T23:59:59Z"):
        openings = calendar.expected_bar_opens(day, "30m")
        assert openings
        for opening in openings:
            assert opening >= day.open_at
            assert opening + timedelta(minutes=30) <= day.close_at


@pytest.mark.parametrize("label,duration", [
    ("2h", timedelta(hours=2)), ("45m", timedelta(minutes=45)),
    ("4h", timedelta(hours=4)), ("15m", timedelta(minutes=15)),
])
def test_no_partial_bar_is_emitted_on_a_grid_that_does_not_divide(
        calendar, label, duration):
    """A trailing stub bar would end after the market shut.

    30m divides a 6h30 session exactly, which makes it the one timeframe that
    cannot expose this: "stop when the next bar would overrun" and "stop when
    the cursor reaches the close" agree for it and disagree for every other.
    So the property is checked on grids that do not divide, where a bar
    reported as complete would in fact be a partial one covering hours the
    market was shut.
    """
    for day in calendar.sessions_between("2026-11-23T00:00:00Z",
                                         "2026-11-30T23:59:59Z"):
        openings = calendar.expected_bar_opens(day, label)
        for opening in openings:
            assert opening >= day.open_at
            assert opening + duration <= day.close_at, (
                f"{label} bar at {opening} runs past the {day.session_date} close")
        # The leftover time at the end of the session is real and is simply
        # not covered by a bar, rather than being covered by a fake one.
        assert len(openings) == day.duration // duration


def test_an_early_close_yields_fewer_bars_and_none_are_synthesised(calendar):
    early = next(session for session
                 in calendar.sessions_between("2026-11-01T00:00:00Z",
                                              "2026-12-31T23:59:59Z")
                 if session.early_close)
    openings = calendar.expected_bar_opens(early, "30m")
    assert len(openings) < 13
    assert len(openings) == early.duration // timedelta(minutes=30)


def test_a_daily_bar_is_the_session_not_twenty_four_hours(calendar):
    """An equity daily bar is 6h30 of trading. Measured as 24h it never fits.

    The naive rule -- a bar must end before the close, a day is 24 hours,
    6h30 < 24h -- reports zero daily bars a year for every US equity. A daily
    bar is the session, which is also why it is shorter on an early close.
    """
    assert calendar.bars_per_day("1d") == 1
    regular = calendar.session_on("2026-01-15")
    daily = calendar.expected_bar_opens(regular, "1d")
    assert daily == (regular.open_at,)

    availability = bar_availability(calendar, bar_open_at=daily[0],
                                    timeframe="1d")
    assert availability.available_at == regular.close_at
    assert availability.available_at - availability.bar_open_at == REGULAR_SESSION

    early = next(session for session
                 in calendar.sessions_between("2026-11-01T00:00:00Z",
                                              "2026-12-31T23:59:59Z")
                 if session.early_close)
    short = bar_availability(
        calendar, bar_open_at=calendar.expected_bar_opens(early, "1d")[0],
        timeframe="1d")
    assert short.available_at == early.close_at
    assert short.available_at - short.bar_open_at < REGULAR_SESSION

    # A year of daily bars is a year of sessions, not 365 and not 252 by
    # convention.
    assert calendar.annualization_periods("1d", year=2026) == (
        calendar.sessions_per_year(2026))


def test_an_intraday_grid_longer_than_the_session_yields_nothing(calendar):
    """8h does not fit in 6h30, and a bar claiming eight hours would lie.

    Distinct from the daily case: a daily bar is *defined* as the session, an
    8h bar is defined as eight hours, and eight hours of this market do not
    exist in one day.
    """
    regular = calendar.session_on("2026-01-15")
    assert calendar.expected_bar_opens(regular, "8h") == ()
    assert calendar.expected_bar_opens(regular, "12h") == ()


def test_a_timeframe_that_does_not_divide_a_session_is_refused(calendar):
    """6h30 holds no whole number of hourly bars, so there is no honest count."""
    for label in ("1h", "2h", "4h"):
        with pytest.raises(EquityCalendarError):
            calendar.bars_per_day(label)


def test_the_next_expected_bar_is_never_simply_thirty_minutes_later(calendar):
    """The crypto rule, applied here, would put a bar in the middle of the night."""
    session = calendar.session_on("2026-01-15")
    inside = session.open_at + timedelta(minutes=5)
    assert calendar.next_expected_bar_open(inside, "30m") == (
        session.open_at + timedelta(minutes=30))

    after_close = session.close_at + timedelta(minutes=5)
    following = calendar.next_expected_bar_open(after_close, "30m")
    assert following != after_close + timedelta(minutes=30)
    assert following > session.close_at + timedelta(hours=12)
    assert calendar.is_market_open(following)


# --- annualisation ---------------------------------------------------------


def test_annualisation_comes_from_the_calendar_and_is_not_8760(calendar):
    """8760 would overstate a 30m equity Sharpe by roughly 1.6x."""
    periods = calendar.annualization_periods("30m", year=2026)
    assert periods == calendar.sessions_per_year(2026) * 13
    assert 3200 <= periods <= 3310
    assert periods != 8760
    assert periods != 365 * 13

    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    crypto = CRYPTO_247_CALENDAR.annualization_periods("1h")
    assert crypto == 8760
    # The whole reason the number lives on the calendar rather than in a
    # formula: the same call gives a different, correct answer per market.
    assert periods < crypto


def test_the_two_calendars_disagree_about_bars_per_day(calendar):
    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    assert CRYPTO_247_CALENDAR.bars_per_day("30m") == 48
    assert calendar.bars_per_day("30m") == 13


# --- expected grid and gaps ------------------------------------------------


def test_an_overnight_or_a_holiday_is_not_a_gap(calendar):
    """The point of the whole module: a shut market is not missing data."""
    grid = build_expected_grid(calendar, start="2026-01-14T00:00:00Z",
                               end="2026-01-21T23:59:59Z", timeframe="30m")
    # Eight calendar days, five sessions: the weekend and MLK Monday are not
    # in the grid at all. Derived from the calendar rather than written down,
    # so the assertion tracks the schedule instead of a guess about it.
    sessions = calendar.sessions_between("2026-01-14T00:00:00Z",
                                         "2026-01-21T23:59:59Z")
    assert len(sessions) == 5
    assert grid.count == 13 * len(sessions)
    assert grid.missing(grid.openings) == ()
    dates = {session.session_date for session in sessions}
    assert {"2026-01-17", "2026-01-18", "2026-01-19"}.isdisjoint(dates)

    for shut in ("2026-01-15T03:00:00Z", "2026-01-17T15:00:00Z",
                 "2026-01-19T15:00:00Z", "2026-01-15T21:30:00Z"):
        assert not grid.contains(shut)


def test_a_bar_the_calendar_never_expected_is_reported_as_unexpected(calendar):
    grid = build_expected_grid(calendar, start="2026-01-15T00:00:00Z",
                               end="2026-01-15T23:59:59Z", timeframe="30m")
    observed = [*grid.openings, "2026-01-15T02:00:00Z", "2026-01-17T15:00:00Z"]
    unexpected = grid.unexpected(observed)
    assert len(unexpected) == 2
    assert grid.missing(observed) == ()


def test_a_genuinely_missing_bar_inside_a_session_is_still_a_gap(calendar):
    """Nothing above may soften this: a bar absent mid-session is real."""
    grid = build_expected_grid(calendar, start="2026-01-15T00:00:00Z",
                               end="2026-01-15T23:59:59Z", timeframe="30m")
    observed = [opening for index, opening in enumerate(grid.openings)
                if index != 5]
    missing = grid.missing(observed)
    assert len(missing) == 1
    assert missing[0].startswith("2026-01-15T")


def test_the_grid_names_the_calendar_that_produced_it(calendar):
    grid = build_expected_grid(calendar, start="2026-01-15T00:00:00Z",
                               end="2026-01-16T23:59:59Z", timeframe="30m")
    assert grid.calendar_spec_hash == calendar.spec.spec_hash
    assert grid.timeframe == "30m"


# --- causality -------------------------------------------------------------


def test_a_bar_is_usable_only_once_it_has_closed(calendar):
    session = calendar.session_on("2026-01-15")
    opening = calendar.expected_bar_opens(session, "30m")[0]
    availability = bar_availability(calendar, bar_open_at=opening,
                                    timeframe="30m")
    assert availability.available_at == opening + timedelta(minutes=30)
    assert not availability.usable_for_decision_at(opening)
    assert not availability.usable_for_decision_at(
        opening + timedelta(minutes=29, seconds=59))
    assert availability.usable_for_decision_at(availability.available_at)


def test_the_last_bar_of_an_early_close_ends_when_the_market_does(calendar):
    """Exactly at the close, not thirty minutes after it opened."""
    early = next(session for session
                 in calendar.sessions_between("2026-11-01T00:00:00Z",
                                              "2026-12-31T23:59:59Z")
                 if session.early_close)
    last = calendar.expected_bar_opens(early, "30m")[-1]
    availability = bar_availability(calendar, bar_open_at=last, timeframe="30m")
    assert availability.available_at == early.close_at


@pytest.mark.parametrize("label", ["30m", "2h", "45m", "15m"])
def test_no_bar_is_ever_available_after_its_session_closed(calendar, label):
    """Availability is bounded by the close on every grid, not just 30m.

    A bar reported as available at 18:30 on a session that shut at 18:00 would
    hand a decision three half-hours of data that did not exist yet.
    """
    for day in calendar.sessions_between("2026-11-23T00:00:00Z",
                                         "2026-11-30T23:59:59Z"):
        for opening in calendar.expected_bar_opens(day, label):
            availability = bar_availability(calendar, bar_open_at=opening,
                                            timeframe=label)
            assert availability.available_at <= day.close_at
            assert availability.available_at > opening


def test_a_bar_that_is_not_on_the_grid_is_refused(calendar):
    session = calendar.session_on("2026-01-15")
    for offset in (timedelta(minutes=17), timedelta(minutes=-30),
                   timedelta(hours=12)):
        with pytest.raises(EquityCalendarError):
            bar_availability(calendar, bar_open_at=session.open_at + offset,
                             timeframe="30m")


def test_a_bar_on_a_day_the_market_was_shut_is_refused(calendar):
    with pytest.raises(EquityCalendarError):
        bar_availability(calendar, bar_open_at=_utc("2026-12-25T14:30:00Z"),
                         timeframe="30m")


# --- resolution ------------------------------------------------------------


def test_the_equity_calendar_resolves_through_the_shared_registry():
    from scripts.trading_lab.trading_calendar import (
        US_EQUITY_REGULAR, get_calendar, known_calendars)

    assert US_EQUITY_REGULAR in known_calendars()
    resolved = get_calendar(US_EQUITY_REGULAR)
    assert resolved.calendar_id == US_EQUITY_REGULAR
    assert resolved.payload()["spec_hash"] == US_EQUITY_REGULAR_SPEC.spec_hash


def test_every_registered_equity_uses_the_equity_calendar():
    from scripts.trading_lab.instrument_registry import EQUITY_INSTRUMENTS_V1
    from scripts.trading_lab.trading_calendar import (
        CRYPTO_24_7, US_EQUITY_REGULAR)

    specs = EQUITY_INSTRUMENTS_V1.all()
    assert specs
    for spec in specs:
        assert spec.trading_calendar == US_EQUITY_REGULAR
        assert spec.trading_calendar != CRYPTO_24_7
        assert spec.timezone == "America/New_York"
        assert "30m" in spec.native_timeframes
        # An equity annualised on an hourly grid is the specific error the
        # calendar exists to prevent, so the grid is not even offered.
        assert "1h" not in spec.native_timeframes
