"""What a market is, stated once, so nothing has to guess from a string.

Until now a market was a free-form string: ``"BTC-USD"`` passed from a config
to an adapter to a guard, compared with ``==`` and ``in`` along the way. That
works exactly as long as every caller spells it the same way, and it fails
silently the first time one does not.

It already failed. The protected-holdout guard tested ``product in
self.products`` against raw text, so ``"btc-usd"``, ``"BTCUSD"``,
``"BTC/USD"``, ``" BTC-USD"`` and ``"coinbase:BTC-USD"`` all reported *not
protected* and sailed straight past the embargo. Twelve of thirteen spellings
of a protected instrument bypassed it. Nothing in the codebase happened to
send those spellings, so nothing noticed.

That is what this module exists to make impossible. There is one canonical
identity, one place that produces it, and every comparison happens on the
canonical form. User input is parsed, canonicalized and validated *before* it
reaches a guard -- never compared as raw text.

Canonical form
--------------
``venue:SYMBOL`` -- venue lowercased, symbol uppercased, base and quote joined
by a single ``-``::

    coinbase:BTC-USD

Chosen over a bare symbol because two venues can list the same ticker and mean
different instruments, and over a tuple because an identity that must be
stored, logged, keyed and compared should have one obvious text form.

Normalisation deliberately accepts more than it emits: ``/``, ``_`` and no
separator at all are folded onto ``-``, surrounding whitespace is stripped,
and case is ignored. Being liberal in what is *recognised* is what makes the
guard safe -- every spelling of a protected instrument lands on the protected
identity rather than slipping past as an unknown one.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import timedelta

INSTRUMENT_SCHEMA_VERSION = "trading-lab.instrument.v1"

# Closed on purpose. Naming an asset class here is a claim that the identity
# layer models it -- not that HyprL can trade it. Futures and options are
# absent because their identity needs expiry, strike and multiplier, and
# inventing half of that now would be a promise the code does not keep.
ASSET_CLASSES = ("CRYPTO", "EQUITY", "ETF", "INDEX", "FX")

# Suffixes used to split a separator-less symbol such as "BTCUSD". Longest
# first, so "USDT" is tried before "USD" and BTCUSDT does not become BTC-USDT
# by way of BTCUS-DT.
KNOWN_QUOTES = ("USDT", "USDC", "USD", "EUR", "GBP", "JPY", "BTC", "ETH")

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,16}(?:-[A-Z0-9]{1,16})?$")
_VENUE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ALLOWED_SEPARATORS = str.maketrans({"/": "-", "_": "-", "\\": "-", ".": "-"})

MAX_IDENTIFIER_CHARS = 64


class InstrumentError(ValueError):
    """Raised when a market identity cannot be established."""


def _clean(raw: object, *, field_name: str) -> str:
    """Strip a value down to comparable text, or refuse it.

    Unicode normalisation runs first: without it a full-width or otherwise
    decomposed character could produce a string that looks like a known symbol
    to a human and compares unequal to it in Python.
    """
    if not isinstance(raw, str):
        raise InstrumentError(f"{field_name} must be a string, got {type(raw).__name__}")
    if len(raw) > MAX_IDENTIFIER_CHARS:
        raise InstrumentError(f"{field_name} is longer than {MAX_IDENTIFIER_CHARS} characters")
    text = unicodedata.normalize("NFKC", raw).strip()
    if not text:
        raise InstrumentError(f"{field_name} is empty")
    # Control characters, including the NUL and newline that a naive strip on
    # a different code path might leave behind.
    if any(unicodedata.category(character) == "Cc" for character in text):
        raise InstrumentError(f"{field_name} contains a control character")
    return text


def normalize_symbol(raw: object) -> str:
    """Fold any accepted spelling of a symbol onto its canonical form.

    ``btc/usd``, ``BTC_USD``, ``BTCUSD`` and `` BTC-USD `` all become
    ``BTC-USD``. A symbol that cannot be read as base/quote is refused rather
    than passed through: a value nobody can canonicalise must not travel on to
    a comparison that would silently call it "some other instrument".
    """
    text = _clean(raw, field_name="symbol").translate(_ALLOWED_SEPARATORS).upper()
    # A dangling or doubled separator is refused, not repaired. Quietly
    # turning "BTC-" into "BTC" would invent an identity from input that
    # nobody can show was meant -- and normalisation that changes meaning is
    # worse than normalisation that refuses.
    if text.startswith("-") or text.endswith("-") or "--" in text:
        raise InstrumentError(
            f"symbol {raw!r} has a dangling separator; write it as BASE-QUOTE")
    if "-" not in text:
        for quote in KNOWN_QUOTES:
            if text.endswith(quote) and len(text) > len(quote):
                text = f"{text[:-len(quote)]}-{quote}"
                break
    if not _SYMBOL_PATTERN.match(text):
        raise InstrumentError(f"symbol {raw!r} is not a usable market symbol")
    return text


def normalize_venue(raw: object) -> str:
    text = _clean(raw, field_name="venue").lower()
    if not _VENUE_PATTERN.match(text):
        raise InstrumentError(f"venue {raw!r} is not a usable venue name")
    return text


@dataclass(frozen=True)
class InstrumentId:
    """A market, named once. Immutable, hashable, comparable only as itself."""

    venue: str
    symbol: str

    def __post_init__(self):
        # Normalising inside the constructor means no code path can hold an
        # InstrumentId that is not canonical -- including one built directly.
        object.__setattr__(self, "venue", normalize_venue(self.venue))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))

    @property
    def canonical(self) -> str:
        return f"{self.venue}:{self.symbol}"

    @property
    def base(self) -> str:
        return self.symbol.split("-")[0]

    @property
    def quote(self) -> str:
        parts = self.symbol.split("-")
        return parts[1] if len(parts) > 1 else ""

    def __str__(self) -> str:
        return self.canonical

    @classmethod
    def parse(cls, raw: object, *, default_venue: str | None = None) -> "InstrumentId":
        """Read any accepted spelling into the one canonical identity.

        Accepts ``coinbase:BTC-USD`` and, when a default venue is supplied, a
        bare ``BTC-USD``. Refusing a bare symbol without a default is what
        stops an identity from being invented out of half the information.
        """
        text = _clean(raw, field_name="instrument")
        if ":" in text:
            venue, _, symbol = text.partition(":")
            if not symbol.strip():
                raise InstrumentError(f"instrument {raw!r} has no symbol")
            return cls(venue=venue, symbol=symbol)
        if default_venue is None:
            raise InstrumentError(
                f"instrument {raw!r} names no venue; write it as 'venue:SYMBOL' "
                "or supply a default venue")
        return cls(venue=default_venue, symbol=text)

    @classmethod
    def coerce(cls, value: object, *, default_venue: str | None = None) -> "InstrumentId":
        """Accept an id or anything parseable into one."""
        if isinstance(value, cls):
            return value
        return cls.parse(value, default_venue=default_venue)


def same_symbol(left: object, right: object) -> bool:
    """Do two spellings name the same market symbol, whatever the venue?

    Used by the research embargo, which is deliberately venue-blind: a
    protected symbol quoted against another venue is still refused. Being
    conservative costs nothing here -- there is one venue -- and being liberal
    would reopen the exact bypass this module was written to close.
    """
    try:
        return normalize_symbol(_strip_venue(left)) == normalize_symbol(_strip_venue(right))
    except InstrumentError:
        return False


def _strip_venue(value: object) -> object:
    if isinstance(value, InstrumentId):
        return value.symbol
    if isinstance(value, str) and ":" in value:
        return value.partition(":")[2]
    return value


@dataclass(frozen=True)
class Timeframe:
    """A bar duration with a name, instead of a string used as arithmetic."""

    unit: str
    count: int

    _UNITS = {"m": timedelta(minutes=1), "h": timedelta(hours=1),
              "d": timedelta(days=1)}

    def __post_init__(self):
        if self.unit not in self._UNITS:
            raise InstrumentError(
                f"unknown timeframe unit {self.unit!r}; expected one of "
                f"{sorted(self._UNITS)}")
        if not isinstance(self.count, int) or self.count < 1 or self.count > 1000:
            raise InstrumentError("timeframe count must sit in 1..1000")

    @property
    def duration(self) -> timedelta:
        return self._UNITS[self.unit] * self.count

    @property
    def label(self) -> str:
        """The legacy spelling: '1h'. Kept identical so nothing rehashes."""
        return f"{self.count}{self.unit}"

    def __str__(self) -> str:
        return self.label

    @classmethod
    def parse(cls, raw: object) -> "Timeframe":
        if isinstance(raw, cls):
            return raw
        text = _clean(raw, field_name="timeframe").lower()
        match = re.match(r"^(\d{1,4})([mhd])$", text)
        if not match:
            raise InstrumentError(f"timeframe {raw!r} is not understood; expected e.g. '1h'")
        return cls(unit=match.group(2), count=int(match.group(1)))


TIMEFRAME_1H = Timeframe(unit="h", count=1)
TIMEFRAME_1D = Timeframe(unit="d", count=1)


@dataclass(frozen=True)
class InstrumentSpec:
    """Everything the platform knows about one market, and nothing about how
    to trade it.

    This is identity and metadata only. No fee, no threshold, no exposure cap:
    those live in the frozen trading specs, and mixing them in here would make
    an instrument's hash change whenever a cost assumption did.
    """

    instrument_id: InstrumentId
    asset_class: str
    base_asset: str
    quote_asset: str
    price_currency: str
    timezone: str
    trading_calendar: str
    native_timeframes: tuple[str, ...]
    quantity_precision: int
    price_precision: int
    display_name: str = ""
    metadata_version: str = INSTRUMENT_SCHEMA_VERSION
    _hash: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.asset_class not in ASSET_CLASSES:
            raise InstrumentError(
                f"unknown asset class {self.asset_class!r}; expected one of "
                f"{list(ASSET_CLASSES)}")
        for name in ("base_asset", "quote_asset", "price_currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InstrumentError(f"{name} must be a non-empty string")
        if not self.native_timeframes:
            raise InstrumentError("an instrument must declare at least one timeframe")
        for label in self.native_timeframes:
            Timeframe.parse(label)                   # refuse an unusable label early
        for name in ("quantity_precision", "price_precision"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0 or value > 18:
                raise InstrumentError(f"{name} must be an integer in 0..18")
        object.__setattr__(self, "_hash", _sha256(_canonical(self.canonical())))

    @property
    def instrument_spec_hash(self) -> str:
        return self._hash

    @property
    def canonical_id(self) -> str:
        return self.instrument_id.canonical

    def supports(self, timeframe: object) -> bool:
        return Timeframe.parse(timeframe).label in self.native_timeframes

    def canonical(self) -> dict:
        return {
            "metadata_version": self.metadata_version,
            "instrument_id": self.instrument_id.canonical,
            "venue": self.instrument_id.venue,
            "symbol": self.instrument_id.symbol,
            "asset_class": self.asset_class,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "price_currency": self.price_currency,
            "timezone": self.timezone,
            "trading_calendar": self.trading_calendar,
            "native_timeframes": list(self.native_timeframes),
            "quantity_precision": self.quantity_precision,
            "price_precision": self.price_precision,
        }

    def payload(self) -> dict:
        """The canonical description plus its hash, for APIs and manifests."""
        return {**self.canonical(),
                "display_name": self.display_name or self.instrument_id.symbol,
                "instrument_spec_hash": self.instrument_spec_hash}

    # --- legacy bridge ---------------------------------------------------

    @property
    def legacy_product_id(self) -> str:
        """The string the Phase 1-5 code and every committed artefact use.

        A bridge, not a second identity: committed results, benchmark specs and
        the recorded event log all carry 'BTC-USD', and rewriting them to say
        'coinbase:BTC-USD' would change hashes that exist precisely so they
        cannot change.
        """
        return self.instrument_id.symbol

    @property
    def legacy_asset(self) -> str:
        """The 'BTC/USD' spelling used by MarketBar."""
        return f"{self.base_asset}/{self.quote_asset}"


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
