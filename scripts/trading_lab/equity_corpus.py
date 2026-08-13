"""The US equity corpus contract: what is asked for, and what is accepted.

Everything in this module is decided *before* a single byte is fetched, and
that ordering is the point. A range chosen after looking at the data is a
curated range; a validation rule relaxed after a capture failed is not a
validation rule. So the spec is frozen and hashed first, the acceptance rules
are written first, and the capture runner in ``capture_us_equity_corpus`` is
only allowed to fill in the blanks.

Three identities, kept apart:

* ``corpus_spec_hash`` -- what was requested: instruments, provider, calendar,
  timeframe, session, adjustment policy, range, and the policies that decide
  what counts as a gap. It does not move when the data does.
* ``instrument_content_hash`` -- the canonical bars of one instrument.
* ``corpus_content_hash`` -- all four, in canonical order.

Re-running the same request against a provider that has changed its mind
produces the same spec hash and a different content hash. That difference is
the whole signal an audit is looking for, and it only exists because the two
are computed from different things.

**What "adjusted" means here.** Corpus V1 is SPLIT_ADJUSTED: historical prices
restated in current share terms. Dividends are *not* applied. This is not
total return, and a series built from it understates a total return by every
dividend paid in the window. The distinction is in the spec hash so it cannot
be lost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext

from scripts.trading_lab.equity_market import (
    ADJUSTMENT_SPLIT_ADJUSTED, EquityMarketError, require_adjustment_policy)
from scripts.trading_lab.instruments import InstrumentId, Timeframe
from scripts.trading_lab.massive_provider import MASSIVE_STOCKS_HISTORICAL_V1
from scripts.trading_lab.trading_calendar import US_EQUITY_REGULAR

EQUITY_CORPUS_SCHEMA_VERSION = "trading-lab.us-equity-corpus.v1"
CANONICAL_SCHEMA_VERSION = "trading-lab.us-equity-bar.v1"
CAPTURE_PROTOCOL_VERSION = "massive-stocks-aggregates-paged-v1"

CORPUS_ID = "massive_us_equity_v1"
CORPUS_ROOT = "data/equities/massive_us_equity_v1"

# Frozen before any capture. Moving a range after seeing performance is how a
# dataset gets quietly curated, and nothing downstream could tell.
CORPUS_INSTRUMENTS = ("xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ")
CORPUS_TIMEFRAME = "30m"
CORPUS_SESSION_TYPE = "REGULAR"
CORPUS_ADJUSTMENT = ADJUSTMENT_SPLIT_ADJUSTED
CORPUS_PROVIDER = MASSIVE_STOCKS_HISTORICAL_V1
CORPUS_CALENDAR = US_EQUITY_REGULAR

# Two years of sessions, ending before the month this was captured in. Nothing
# after 2026-07-31 belongs in V1: a different range is a different corpus, and
# it needs a V2 spec rather than a quiet extension.
CORPUS_RANGE_START = "2024-08-01"
CORPUS_RANGE_END = "2026-07-31"

# Named so the hash records them. A future corpus that forward-fills or that
# counts overnight as a gap is a different corpus, not the same one with a
# different report.
GAP_POLICY = "expected-bar-grid-v1:no-fill,no-interpolation,session-only"
CORPORATE_ACTION_POLICY = (
    "provider-adjusted:splits-only,dividends-recorded-not-applied")

# Bounds on a single capture run. A provider that pages forever, or that
# returns the same page twice, stops the run rather than filling a disk.
MAX_PAGES_PER_REQUEST = 200
MAX_ROWS_TOTAL = 2_000_000

# Money math happens at the project's economic precision, inside an explicit
# context, so a caller's ambient Decimal settings can never change a hash.
ECONOMIC_PRECISION = 34


class EquityCorpusError(RuntimeError):
    """Raised when a corpus is specified, captured or verified unsafely."""


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: object, *, field_name: str = "timestamp") -> datetime:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise EquityCorpusError(
                f"{field_name} {value!r} is not a timestamp") from error
    else:
        raise EquityCorpusError(
            f"{field_name} must be a datetime or ISO-8601 string")
    if moment.tzinfo is None:
        raise EquityCorpusError(
            f"{field_name} must carry a timezone; a naive equity timestamp is "
            "ambiguous by exactly the offset that decides its session")
    return moment.astimezone(timezone.utc)


def parse_day(value: object, *, field_name: str = "date") -> date:
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, TypeError) as error:
        raise EquityCorpusError(
            f"{field_name} {value!r} is not a date") from error


# --- the specification -----------------------------------------------------


@dataclass(frozen=True)
class USEquityCorpusV1:
    """What Corpus V1 asks for. Hashed before anything is fetched.

    The calendar's own spec hash is included, and so is the pinned library
    version it came from: a provider release that moves a holiday changes the
    expected bar grid, which changes what counts as a gap, which changes the
    meaning of the whole dataset. That must not be invisible.
    """

    corpus_id: str = CORPUS_ID
    instruments: tuple[str, ...] = CORPUS_INSTRUMENTS
    provider_id: str = CORPUS_PROVIDER
    timeframe: str = CORPUS_TIMEFRAME
    session_type: str = CORPUS_SESSION_TYPE
    adjustment_policy: str = CORPUS_ADJUSTMENT
    calendar_id: str = CORPUS_CALENDAR
    requested_start: str = CORPUS_RANGE_START
    requested_end: str = CORPUS_RANGE_END
    gap_policy: str = GAP_POLICY
    corporate_action_policy: str = CORPORATE_ACTION_POLICY
    capture_protocol_version: str = CAPTURE_PROTOCOL_VERSION
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION
    schema_version: str = EQUITY_CORPUS_SCHEMA_VERSION

    def __post_init__(self):
        if not self.instruments:
            raise EquityCorpusError("a corpus must name at least one instrument")
        object.__setattr__(self, "instruments", tuple(
            InstrumentId.coerce(item).canonical for item in self.instruments))
        if len(set(self.instruments)) != len(self.instruments):
            raise EquityCorpusError("an instrument is named twice")
        require_adjustment_policy(self.adjustment_policy)
        Timeframe.parse(self.timeframe)
        if parse_day(self.requested_end) <= parse_day(self.requested_start):
            raise EquityCorpusError("a corpus window must end after it starts")

    # --- calendar binding -------------------------------------------------

    def calendar(self):
        """The calendar this corpus is defined against. Never a default."""
        from scripts.trading_lab.trading_calendar import get_calendar

        return get_calendar(self.calendar_id)

    def calendar_identity(self) -> dict:
        """The rules, their version, and their hash -- all three in the spec."""
        from scripts.trading_lab.equity_calendar import (
            CALENDAR_PROVIDER, CALENDAR_PROVIDER_VERSION, US_EQUITY_REGULAR_SPEC)

        return {
            "calendar_id": self.calendar_id,
            "calendar_spec_hash": US_EQUITY_REGULAR_SPEC.spec_hash,
            "calendar_dependency": CALENDAR_PROVIDER,
            "calendar_dependency_version": CALENDAR_PROVIDER_VERSION,
        }

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "instruments": list(self.instruments),
            "provider_id": self.provider_id,
            "timeframe": Timeframe.parse(self.timeframe).label,
            "session_type": self.session_type,
            "adjustment_policy": self.adjustment_policy,
            **self.calendar_identity(),
            "requested_range": {"start": self.requested_start,
                                "end": self.requested_end},
            "canonical_schema_version": self.canonical_schema_version,
            "capture_protocol_version": self.capture_protocol_version,
            "gap_policy": self.gap_policy,
            "corporate_action_policy": self.corporate_action_policy,
        }

    @property
    def corpus_spec_hash(self) -> str:
        return sha256_canonical(self.canonical())

    def payload(self) -> dict:
        return {**self.canonical(), "corpus_spec_hash": self.corpus_spec_hash}

    # --- the expected grid ------------------------------------------------

    def sessions(self):
        """Every real session in the requested range, from the calendar."""
        calendar = self.calendar()
        return calendar.sessions_between(
            f"{self.requested_start}T00:00:00Z",
            f"{self.requested_end}T23:59:59Z")

    def expected_bar_opens(self) -> tuple[datetime, ...]:
        """Every bar the calendar says should exist. The only source of truth.

        Not 13 x weekdays. Holidays remove sessions and early closes remove
        bars, and both come from the calendar rather than from arithmetic.
        """
        calendar = self.calendar()
        openings: list[datetime] = []
        for session in self.sessions():
            openings.extend(calendar.expected_bar_opens(session, self.timeframe))
        return tuple(openings)

    def session_index(self) -> dict:
        """Bar opening -> the session that contains it.

        Built once and shared: it is what turns "is this bar legal" from a
        scan over five hundred sessions into a dictionary lookup, and it is
        also what supplies each row its session date and type.
        """
        calendar = self.calendar()
        index: dict[datetime, object] = {}
        for session in self.sessions():
            for opening in calendar.expected_bar_opens(session, self.timeframe):
                index[opening] = session
        return index


CORPUS_SPEC_V1 = USEquityCorpusV1()


# --- request identity ------------------------------------------------------


@dataclass(frozen=True)
class RequestIdentity:
    """What a request asks for, deterministically.

    Deliberately excludes the wall clock. Two capture runs a week apart that
    ask the same question have the same request identity, which is what makes
    a duplicate detectable and a re-capture comparable. The capture *time* is
    metadata attached alongside, never part of the identity.
    """

    provider_id: str
    instrument_id: str
    timeframe: str
    adjustment_policy: str
    start: str
    end: str
    page: int
    cursor: str = ""

    def canonical(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "instrument_id": self.instrument_id,
            "timeframe": self.timeframe,
            "adjustment_policy": self.adjustment_policy,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "cursor": self.cursor,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_canonical(self.canonical())

    @property
    def slug(self) -> str:
        """A filesystem name that is stable and says what it holds."""
        symbol = self.instrument_id.split(":")[-1]
        return f"{symbol}_page_{self.page:04d}_{self.identity_hash[:12]}"


# --- canonical bars --------------------------------------------------------

CANONICAL_FIELDS = (
    "instrument_id", "provider_id", "bar_open_at", "bar_close_at",
    "session_date", "session_type", "timeframe", "open", "high", "low",
    "close", "volume", "adjustment_policy", "source_raw_hash",
    "source_record_identity",
)


def _decimal(value: object, *, field_name: str) -> Decimal:
    """Prices as Decimal, never from a float.

    A float price has already lost digits before it reaches this line, and no
    amount of care downstream puts them back.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise EquityCorpusError(
            f"{field_name} arrived as a float; prices must be strings or "
            "Decimal so the printed value is the stored value")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise EquityCorpusError(
            f"{field_name} {value!r} is not a number") from error


@dataclass(frozen=True)
class CanonicalBar:
    """One accepted bar. Every field that gives it meaning travels with it."""

    instrument_id: str
    provider_id: str
    bar_open_at: datetime
    bar_close_at: datetime
    session_date: str
    session_type: str
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    adjustment_policy: str
    source_raw_hash: str
    source_record_identity: str

    def row(self) -> dict:
        """The canonical serialised form, in a fixed key order.

        Decimals become strings: a JSON number would be parsed back as a float
        by most readers, and the hash would then depend on the reader.
        """
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return {
                "instrument_id": self.instrument_id,
                "provider_id": self.provider_id,
                "bar_open_at": iso(self.bar_open_at),
                "bar_close_at": iso(self.bar_close_at),
                "session_date": self.session_date,
                "session_type": self.session_type,
                "timeframe": self.timeframe,
                "open": str(self.open),
                "high": str(self.high),
                "low": str(self.low),
                "close": str(self.close),
                "volume": str(self.volume),
                "adjustment_policy": self.adjustment_policy,
                "source_raw_hash": self.source_raw_hash,
                "source_record_identity": self.source_record_identity,
            }

    @property
    def canonical_key(self) -> tuple[str, str]:
        """What makes two rows the same bar: the market and the opening."""
        return (self.instrument_id, iso(self.bar_open_at))


def validate_ohlc(bar: CanonicalBar) -> CanonicalBar:
    """Every arithmetic invariant a bar must satisfy, checked once.

    Fail-closed throughout. A bar whose high is below its open is not a bar
    with a small error in it -- it is a row this platform cannot interpret,
    and accepting it would put an impossible price into a dataset that later
    looks perfectly ordinary.
    """
    low, high = bar.low, bar.high
    for name in ("open", "high", "low", "close"):
        value = getattr(bar, name)
        if value <= 0:
            raise EquityCorpusError(
                f"{bar.instrument_id} {iso(bar.bar_open_at)}: {name} {value} "
                "is not a positive price")
    if high < low:
        raise EquityCorpusError(
            f"{bar.instrument_id} {iso(bar.bar_open_at)}: high {high} is below "
            f"low {low}")
    if high < max(bar.open, bar.close):
        raise EquityCorpusError(
            f"{bar.instrument_id} {iso(bar.bar_open_at)}: high {high} is below "
            f"open/close")
    if low > min(bar.open, bar.close):
        raise EquityCorpusError(
            f"{bar.instrument_id} {iso(bar.bar_open_at)}: low {low} is above "
            f"open/close")
    if bar.volume < 0:
        raise EquityCorpusError(
            f"{bar.instrument_id} {iso(bar.bar_open_at)}: volume "
            f"{bar.volume} is negative")
    if bar.bar_close_at <= bar.bar_open_at:
        raise EquityCorpusError(
            f"{bar.instrument_id} {iso(bar.bar_open_at)}: bar closes at or "
            "before it opens")
    return bar


def build_canonical_bar(*, spec: USEquityCorpusV1, instrument_id: str,
                        session, bar_open_at: datetime, row: dict,
                        source_raw_hash: str,
                        source_record_identity: str) -> CanonicalBar:
    """Turn one accepted provider row into a canonical bar, or refuse it."""
    calendar = spec.calendar()
    frame = Timeframe.parse(spec.timeframe)
    bar = CanonicalBar(
        instrument_id=instrument_id,
        provider_id=spec.provider_id,
        bar_open_at=bar_open_at,
        bar_close_at=calendar.bar_close_for(session, bar_open_at, frame),
        session_date=session.session_date,
        session_type=session.session_type,
        timeframe=frame.label,
        open=_decimal(row["open"], field_name="open"),
        high=_decimal(row["high"], field_name="high"),
        low=_decimal(row["low"], field_name="low"),
        close=_decimal(row["close"], field_name="close"),
        volume=_decimal(row["volume"], field_name="volume"),
        adjustment_policy=spec.adjustment_policy,
        source_raw_hash=source_raw_hash,
        source_record_identity=source_record_identity)
    return validate_ohlc(bar)


def canonical_order(bars):
    """Instrument, then opening. The only order a hash is computed over."""
    return sorted(bars, key=lambda bar: bar.canonical_key)


def serialise_rows(bars) -> str:
    """JSONL, one canonical row per line, key order fixed."""
    return "".join(f"{canonical_json(bar.row())}\n" for bar in bars)


def instrument_content_hash(bars) -> str:
    """The bars of one instrument, in canonical order."""
    return sha256_canonical([bar.row() for bar in canonical_order(bars)])


def corpus_content_hash(by_instrument: dict) -> str:
    """All instruments, in canonical order, independent of filesystem order."""
    return sha256_canonical([
        {"instrument_id": instrument_id,
         "instrument_content_hash": instrument_content_hash(bars),
         "rows": len(bars)}
        for instrument_id, bars in sorted(by_instrument.items())
    ])


# --- session acceptance ----------------------------------------------------


def accept_bar_opening(spec: USEquityCorpusV1, session_index: dict,
                       bar_open_at: datetime):
    """The session that legitimises this opening, or a refusal.

    A provider may return more than was asked for -- pre-market, after-hours,
    a holiday it thinks was a session, a bar past an early close. HyprL does
    not accept a row simply because it exists. Membership of the expected bar
    grid is the entire test.
    """
    session = session_index.get(bar_open_at)
    if session is None:
        raise EquityCorpusError(
            f"{iso(bar_open_at)} is not on the {spec.timeframe} expected bar "
            f"grid for the {spec.calendar_id} calendar: it falls outside a "
            "regular session, on a holiday or weekend, or after an early close")
    return session


# --- gap audit -------------------------------------------------------------


@dataclass(frozen=True)
class GapAudit:
    """What the calendar expected against what actually arrived.

    Only one classification exists: MISSING_EXPECTED_BAR. Overnight, weekends,
    holidays and the hours after an early close are not gaps and never appear
    here -- no bar was ever expected at those times, so nothing is missing.
    Nothing is filled or interpolated; a gap is reported, not repaired.
    """

    instrument_id: str
    expected: int
    observed: int
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    duplicates: tuple[str, ...]

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    def payload(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "expected_bars": self.expected,
            "observed_bars": self.observed,
            "missing_expected_bars": len(self.missing),
            "extra_bars": len(self.extra),
            "duplicate_bars": len(self.duplicates),
            "classification": "MISSING_EXPECTED_BAR",
            "missing": list(self.missing),
            "extra": list(self.extra),
            "duplicates": list(self.duplicates),
        }


def audit_gaps(spec: USEquityCorpusV1, instrument_id: str, bars,
               expected_openings=None) -> GapAudit:
    expected = set(expected_openings if expected_openings is not None
                   else spec.expected_bar_opens())
    observed: list[datetime] = [bar.bar_open_at for bar in bars]
    seen, duplicates = set(), []
    for opening in observed:
        if opening in seen:
            duplicates.append(iso(opening))
        seen.add(opening)
    return GapAudit(
        instrument_id=instrument_id,
        expected=len(expected),
        observed=len(observed),
        missing=tuple(sorted(iso(item) for item in expected - seen)),
        extra=tuple(sorted(iso(item) for item in seen - expected)),
        duplicates=tuple(sorted(duplicates)))


def overlapping_missing_intervals(audits) -> tuple[str, ...]:
    """Openings missing from every instrument at once.

    Reported descriptively and named for exactly what it is. Several
    instruments missing the same interval is consistent with a provider
    outage and also with several other things, and this function does not
    know which. Calling it "provider outage" would be a conclusion the data
    does not support -- the same discipline the Coinbase capture used.
    """
    if not audits:
        return ()
    common = set(audits[0].missing)
    for audit in audits[1:]:
        common &= set(audit.missing)
    return tuple(sorted(common))


# --- corporate actions -----------------------------------------------------


@dataclass(frozen=True)
class SplitRecord:
    """A split as captured, kept for provenance rather than for arithmetic.

    Corpus V1 stores prices the provider already adjusted. These records exist
    so an auditor can see *which* actions the adjustment reflects, and so a
    later phase can check a suspicious price discontinuity against a real
    corporate event instead of guessing.
    """

    instrument_id: str
    effective_date: str
    ratio_numerator: int
    ratio_denominator: int
    provider_id: str
    source_raw_hash: str

    def __post_init__(self):
        parse_day(self.effective_date, field_name="effective_date")
        for name in ("ratio_numerator", "ratio_denominator"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise EquityCorpusError(f"{name} must be a positive integer")

    @property
    def ratio(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return Decimal(self.ratio_numerator) / Decimal(self.ratio_denominator)

    def row(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "effective_date": self.effective_date,
            "ratio_numerator": self.ratio_numerator,
            "ratio_denominator": self.ratio_denominator,
            "ratio": str(self.ratio),
            "label": f"{self.ratio_numerator}-for-{self.ratio_denominator}",
            "provider_id": self.provider_id,
            "source_raw_hash": self.source_raw_hash,
            "applied_by_provider": True,
            "recomputed_by_hyprl": False,
        }


def corporate_actions_hash(records) -> str:
    return sha256_canonical([
        record.row() for record in
        sorted(records, key=lambda item: (item.instrument_id,
                                          item.effective_date))])


__all__ = [
    "CANONICAL_FIELDS", "CANONICAL_SCHEMA_VERSION", "CAPTURE_PROTOCOL_VERSION",
    "CORPORATE_ACTION_POLICY", "CORPUS_ADJUSTMENT", "CORPUS_CALENDAR",
    "CORPUS_ID", "CORPUS_INSTRUMENTS", "CORPUS_PROVIDER", "CORPUS_RANGE_END",
    "CORPUS_RANGE_START", "CORPUS_ROOT", "CORPUS_SESSION_TYPE",
    "CORPUS_SPEC_V1", "CORPUS_TIMEFRAME", "CanonicalBar",
    "EQUITY_CORPUS_SCHEMA_VERSION", "ECONOMIC_PRECISION", "EquityCorpusError",
    "GAP_POLICY", "GapAudit", "MAX_PAGES_PER_REQUEST", "MAX_ROWS_TOTAL",
    "RequestIdentity", "SplitRecord", "USEquityCorpusV1", "accept_bar_opening",
    "audit_gaps", "build_canonical_bar", "canonical_json", "canonical_order",
    "corporate_actions_hash", "corpus_content_hash", "instrument_content_hash",
    "iso", "overlapping_missing_intervals", "parse_day", "parse_utc",
    "serialise_rows", "sha256_bytes", "sha256_canonical", "validate_ohlc",
]
