"""What an equity bar means, which is not what a crypto bar means.

A BTC candle is a fact about a continuous market: the price was what it was,
and it will still have been that a year from now. An equity bar is a fact about
a company as much as a market, and the company changes shape. A 4-for-1 split
divides every historical price by four overnight. A dividend steps the price
down by roughly its amount. A ticker gets reused: the AAPL of 1985 is not a
different company, but plenty of tickers have been.

None of that is a data quality problem to be smoothed over. It is the data.
What breaks a backtest is not the split -- it is a series that is *silently*
half raw and half adjusted, because both halves look completely normal.

So the adjustment policy is part of the series' identity and part of its hash.
RAW and SPLIT_ADJUSTED produce different spec hashes, so a model trained under
one and evaluated under the other cannot be mistaken for a like-for-like
comparison: the hashes simply do not match.

Nothing here fetches anything. This module defines what a bar and a corporate
action *are*; ``massive_provider`` deals with where they come from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

from scripts.trading_lab.instruments import InstrumentError, InstrumentId, Timeframe

EQUITY_MARKET_SCHEMA_VERSION = "trading-lab.equity-market.v1"

# --- adjustment policy -----------------------------------------------------

# Prices exactly as printed on the day. A split shows up as a discontinuity,
# because that is what happened.
ADJUSTMENT_RAW = "RAW"

# Historical prices restated in today's share terms, so a split is invisible.
# Dividends are NOT applied: total-return adjustment is a third policy, and
# calling it SPLIT_ADJUSTED would be a lie by omission.
ADJUSTMENT_SPLIT_ADJUSTED = "SPLIT_ADJUSTED"

ADJUSTMENT_POLICIES = (ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED)

# Explicitly not implemented. Named so that a caller asking for it gets a
# refusal rather than silently receiving split-adjusted prices and believing
# dividends were handled.
ADJUSTMENT_TOTAL_RETURN = "TOTAL_RETURN"

class EquityMarketError(RuntimeError):
    """Raised when equity market data is malformed or semantically unsafe."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def require_adjustment_policy(policy: object) -> str:
    if policy == ADJUSTMENT_TOTAL_RETURN:
        raise EquityMarketError(
            "TOTAL_RETURN adjustment is not implemented. Split adjustment is "
            "not a substitute: it leaves every dividend out of the return.")
    if policy not in ADJUSTMENT_POLICIES:
        raise EquityMarketError(
            f"unknown adjustment policy {policy!r}; supported: "
            f"{list(ADJUSTMENT_POLICIES)}")
    return str(policy)


def _decimal(value: object, *, field_name: str) -> Decimal:
    """Prices as Decimal, from strings. A float price has already lost digits."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise EquityMarketError(
            f"{field_name} arrived as a float; prices must be strings or "
            "Decimal so the printed value is the stored value")
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise EquityMarketError(f"{field_name} {value!r} is not a number") from error
    # Decimal happily accepts "nan", "inf" and "Infinity". A non-finite price
    # satisfies every ordering invariant below by accident, because each IEEE
    # comparison against it answers False -- so it would validate, serialise as
    # "Infinity" and hash into an artefact that later looks entirely ordinary.
    if not converted.is_finite():
        raise EquityMarketError(f"{field_name} is not a finite number")
    return converted


def _utc(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise EquityMarketError(
                f"{field_name} {value!r} is not a timestamp") from error
    else:
        raise EquityMarketError(f"{field_name} must be a datetime or ISO-8601 string")
    if moment.tzinfo is None:
        raise EquityMarketError(
            f"{field_name} must carry a timezone; a naive equity timestamp is "
            "ambiguous by exactly the offset that decides its session")
    return moment.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EquityMarketBar:
    """One equity bar, carrying the policy it was produced under.

    The policy travels with the bar rather than sitting in a config somewhere,
    because the failure this prevents is a series assembled from two sources
    with different policies. Row by row, both look valid.
    """

    instrument_id: InstrumentId
    timeframe: str
    provider_id: str
    adjustment_policy: str
    bar_open_at: datetime
    bar_close_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    session_date: str = ""
    _hash: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "adjustment_policy",
                           require_adjustment_policy(self.adjustment_policy))
        object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe).label)
        if self.bar_close_at <= self.bar_open_at:
            raise EquityMarketError(
                f"bar closes at {_iso(self.bar_close_at)}, at or before it opens")
        # Finiteness before any comparison. NaN makes a Decimal comparison
        # raise decimal.InvalidOperation -- an untyped error from a library
        # the caller never named -- and Infinity answers False to every
        # ordering test, so it passes each guard in turn. A bar can also be
        # constructed directly rather than through _decimal, so the check
        # cannot live only at the conversion boundary.
        for name in ("open", "high", "low", "close", "volume"):
            value = getattr(self, name)
            if not value.is_finite():
                raise EquityMarketError(f"{name} is not a finite number")
        low, high = self.low, self.high
        if high < low:
            raise EquityMarketError(f"high {high} is below low {low}")
        for name in ("open", "close"):
            value = getattr(self, name)
            if not low <= value <= high:
                raise EquityMarketError(
                    f"{name} {value} falls outside the bar's low-high range "
                    f"[{low}, {high}]")
        if self.volume < 0:
            raise EquityMarketError(f"volume {self.volume} is negative")
        if low < 0:
            raise EquityMarketError(f"low {low} is negative")
        object.__setattr__(self, "_hash", _sha256(self.canonical()))

    @property
    def bar_hash(self) -> str:
        return self._hash

    @property
    def available_at(self) -> datetime:
        """When this bar could first have informed a decision: at its close."""
        return self.bar_close_at

    def belongs_to(self, instrument: object) -> bool:
        try:
            return InstrumentId.coerce(instrument) == self.instrument_id
        except InstrumentError:
            return False

    def require_instrument(self, instrument: object) -> "EquityMarketBar":
        if not self.belongs_to(instrument):
            wanted = InstrumentId.coerce(instrument).canonical
            raise EquityMarketError(
                f"this bar belongs to {self.instrument_id.canonical}, not "
                f"{wanted}; refusing to mix instruments in one series")
        return self

    def canonical(self) -> dict:
        return {
            "schema_version": EQUITY_MARKET_SCHEMA_VERSION,
            "instrument_id": self.instrument_id.canonical,
            "timeframe": self.timeframe,
            "provider_id": self.provider_id,
            "adjustment_policy": self.adjustment_policy,
            "bar_open_at": _iso(self.bar_open_at),
            "bar_close_at": _iso(self.bar_close_at),
            "session_date": self.session_date,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }

    def payload(self) -> dict:
        return {**self.canonical(), "bar_hash": self.bar_hash,
                "available_at": _iso(self.available_at)}


def build_equity_bar(*, instrument, timeframe, provider_id, adjustment_policy,
                     bar_open_at, bar_close_at, open, high, low, close, volume,
                     session_date: str = "") -> EquityMarketBar:
    """Construct a bar from raw values, converting and validating once."""
    return EquityMarketBar(
        instrument_id=InstrumentId.coerce(instrument),
        timeframe=timeframe,
        provider_id=str(provider_id),
        adjustment_policy=adjustment_policy,
        bar_open_at=_utc(bar_open_at, field_name="bar_open_at"),
        bar_close_at=_utc(bar_close_at, field_name="bar_close_at"),
        open=_decimal(open, field_name="open"),
        high=_decimal(high, field_name="high"),
        low=_decimal(low, field_name="low"),
        close=_decimal(close, field_name="close"),
        volume=_decimal(volume, field_name="volume"),
        session_date=session_date)


def require_single_policy(bars) -> str:
    """Refuse a series whose rows disagree about how they were adjusted.

    This is the check the whole module exists for. A series that is raw before
    a split and adjusted after it shows a return of several hundred percent on
    one bar, and every risk number computed from it is wrong.
    """
    policies = {bar.adjustment_policy for bar in bars}
    if not policies:
        raise EquityMarketError("an empty series declares no adjustment policy")
    if len(policies) > 1:
        raise EquityMarketError(
            f"a series must use one adjustment policy; found {sorted(policies)}. "
            "Mixing them produces a fake return at every corporate action.")
    return policies.pop()


# --- corporate actions -----------------------------------------------------


@dataclass(frozen=True)
class StockSplit:
    """A split, as a ratio and an effective date.

    ``ratio`` is new shares per old share: 4 for a 4-for-1. Stored as a
    Decimal fraction rather than a float because a 3-for-2 is 1.5 exactly and
    a 7-for-3 is not, and the difference compounds across a history.

    The effective date is the first session on which the *new* share count
    applies. A bar on that date is already adjusted; the bar before it is not.
    Off-by-one here silently moves a large fake return by one day.
    """

    instrument_id: InstrumentId
    effective_date: str
    ratio_numerator: int
    ratio_denominator: int
    source: str = ""

    def __post_init__(self):
        try:
            date.fromisoformat(self.effective_date)
        except (ValueError, TypeError) as error:
            raise EquityMarketError(
                f"effective_date {self.effective_date!r} is not a date") from error
        for name in ("ratio_numerator", "ratio_denominator"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise EquityMarketError(f"{name} must be a positive integer")

    @property
    def ratio(self) -> Decimal:
        return Decimal(self.ratio_numerator) / Decimal(self.ratio_denominator)

    @property
    def label(self) -> str:
        return f"{self.ratio_numerator}-for-{self.ratio_denominator}"

    def adjust_price(self, price: object) -> Decimal:
        """A pre-split price restated in post-split shares: divided by the ratio."""
        return _decimal(price, field_name="price") / self.ratio

    def adjust_volume(self, volume: object) -> Decimal:
        """Volume moves the other way: more shares, so a larger count."""
        return _decimal(volume, field_name="volume") * self.ratio

    def applies_to(self, session_date: str) -> bool:
        """Whether a bar on this session predates the split and needs adjusting."""
        return str(session_date)[:10] < self.effective_date

    def payload(self) -> dict:
        return {
            "instrument_id": self.instrument_id.canonical,
            "effective_date": self.effective_date,
            "ratio_numerator": self.ratio_numerator,
            "ratio_denominator": self.ratio_denominator,
            "ratio": str(self.ratio),
            "label": self.label,
            "source": self.source,
        }


@dataclass(frozen=True)
class CashDividend:
    """A dividend, recorded and deliberately not applied.

    Present so that a corpus can *state* that a dividend occurred in its
    window. SPLIT_ADJUSTED prices ignore it, which means a price return
    understates the total return by the dividend. Recording it makes that
    understatement visible instead of unknown.
    """

    instrument_id: InstrumentId
    ex_date: str
    amount: Decimal
    currency: str = "USD"
    source: str = ""

    def __post_init__(self):
        try:
            date.fromisoformat(self.ex_date)
        except (ValueError, TypeError) as error:
            raise EquityMarketError(
                f"ex_date {self.ex_date!r} is not a date") from error
        object.__setattr__(self, "amount", _decimal(self.amount, field_name="amount"))
        if self.amount <= 0:
            raise EquityMarketError("a dividend amount must be positive")

    def payload(self) -> dict:
        return {
            "instrument_id": self.instrument_id.canonical,
            "ex_date": self.ex_date,
            "amount": str(self.amount),
            "currency": self.currency,
            "applied_to_prices": False,
            "source": self.source,
        }


def apply_splits(bars, splits) -> tuple:
    """Restate a RAW series in current shares. Never applied twice.

    Requires RAW input and returns SPLIT_ADJUSTED output, so calling it on an
    already-adjusted series raises rather than dividing by the ratio a second
    time -- a mistake that produces a perfectly smooth, completely wrong price
    history.
    """
    rows = tuple(bars)
    if not rows:
        return ()
    policy = require_single_policy(rows)
    if policy != ADJUSTMENT_RAW:
        raise EquityMarketError(
            f"apply_splits needs a {ADJUSTMENT_RAW} series; this one is already "
            f"{policy}. Adjusting twice divides every historical price twice.")
    ordered = sorted(splits, key=lambda item: item.effective_date)
    adjusted = []
    for bar in rows:
        session = bar.session_date or _iso(bar.bar_open_at)[:10]
        price_factor, volume_factor = Decimal(1), Decimal(1)
        for split in ordered:
            if split.instrument_id != bar.instrument_id:
                raise EquityMarketError(
                    f"split for {split.instrument_id.canonical} cannot adjust a "
                    f"{bar.instrument_id.canonical} bar")
            if split.applies_to(session):
                price_factor *= split.ratio
                volume_factor *= split.ratio
        adjusted.append(EquityMarketBar(
            instrument_id=bar.instrument_id, timeframe=bar.timeframe,
            provider_id=bar.provider_id,
            adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED,
            bar_open_at=bar.bar_open_at, bar_close_at=bar.bar_close_at,
            open=bar.open / price_factor, high=bar.high / price_factor,
            low=bar.low / price_factor, close=bar.close / price_factor,
            volume=bar.volume * volume_factor, session_date=bar.session_date))
    return tuple(adjusted)


# --- corpus identity -------------------------------------------------------


@dataclass(frozen=True)
class EquityCorpusSpec:
    """What a captured equity dataset would be, hashed before anything is captured.

    Declaring the shape first means the corpus cannot later be described by
    whatever it happens to contain. Two corpora differing only in adjustment
    policy have different hashes here, which is the entire point: they are not
    interchangeable and no downstream comparison should treat them as such.

    Phase 6D defines this spec. It captures nothing.
    """

    instruments: tuple[str, ...]
    timeframe: str
    adjustment_policy: str
    provider_id: str
    calendar_id: str
    calendar_spec_hash: str
    start: str
    end: str
    schema_version: str = EQUITY_MARKET_SCHEMA_VERSION

    def __post_init__(self):
        if not self.instruments:
            raise EquityMarketError("a corpus must name at least one instrument")
        object.__setattr__(self, "instruments", tuple(
            InstrumentId.coerce(item).canonical for item in self.instruments))
        object.__setattr__(self, "adjustment_policy",
                           require_adjustment_policy(self.adjustment_policy))
        object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe).label)
        if self.end <= self.start:
            raise EquityMarketError("a corpus window must end after it starts")

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "instruments": sorted(self.instruments),
            "timeframe": self.timeframe,
            "adjustment_policy": self.adjustment_policy,
            "provider_id": self.provider_id,
            "calendar_id": self.calendar_id,
            "calendar_spec_hash": self.calendar_spec_hash,
            "start": self.start,
            "end": self.end,
        }

    @property
    def corpus_spec_hash(self) -> str:
        return _sha256(self.canonical())

    def payload(self) -> dict:
        return {**self.canonical(),
                "corpus_spec_hash": self.corpus_spec_hash,
                "captured": False,
                "capture_note": "Phase 6D defines this corpus; it captures nothing."}


__all__ = [
    "ADJUSTMENT_POLICIES", "ADJUSTMENT_RAW", "ADJUSTMENT_SPLIT_ADJUSTED",
    "ADJUSTMENT_TOTAL_RETURN", "CashDividend", "EQUITY_MARKET_SCHEMA_VERSION",
    "EquityCorpusSpec", "EquityMarketBar", "EquityMarketError",
    "StockSplit", "apply_splits", "build_equity_bar",
    "require_adjustment_policy", "require_single_policy",
]
