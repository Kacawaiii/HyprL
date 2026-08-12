"""Where bars come from, separated from what they are bars of.

An instrument and a data source are different facts that happen to coincide
today: BTC-USD is listed on Coinbase and Coinbase is also where the candles
come from. The moment a second source appears -- the same equity from two
vendors, or Coinbase quotes replayed from a local capture -- code that
conflated them has to be untangled under pressure.

Two rules keep this honest.

**Capabilities are declared, not assumed.** A provider states exactly what it
can do. Nothing here says ``realtime_ticks`` or ``order_book`` is available,
because HyprL implements neither, and a capability flag that overstates the
truth is worse than no flag: callers branch on it.

**Market data only.** There is no ``place_order``, no ``account``, no
``balance`` and no ``wallet`` on this interface, and no authenticated
transport behind it. That is not an omission to be filled in later by
extending this class -- an execution venue is a different interface with a
different threat model, and it does not exist in this project.

The Coinbase provider is a thin adapter over the Phase 1-5D code. It does not
re-parse, re-validate or re-canonicalise anything: the response parser, the
decimal rules and the holdout guard all stay where they are, with one
implementation each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from scripts.trading_lab.instruments import (
    InstrumentError, InstrumentId, InstrumentSpec, Timeframe)
from scripts.trading_lab.trading_calendar import CRYPTO_24_7

MARKET_PROVIDER_SCHEMA_VERSION = "trading-lab.market-provider.v1"

COINBASE_PUBLIC_V1 = "coinbase-public-v1"

# How current a provider's data is. Not a quality rating -- a statement about
# what a caller is allowed to conclude from the most recent row. An end-of-day
# source can be excellent and still be useless for a decision at 15:00, and the
# only thing that stops it being used that way is this flag being checked.
FRESHNESS_UNSPECIFIED = "UNSPECIFIED"
FRESHNESS_END_OF_DAY = "END_OF_DAY"
FRESHNESS_DELAYED = "DELAYED"
FRESHNESS_LIVE = "LIVE"
FRESHNESS_LEVELS = (FRESHNESS_UNSPECIFIED, FRESHNESS_END_OF_DAY,
                    FRESHNESS_DELAYED, FRESHNESS_LIVE)


class MarketProviderError(RuntimeError):
    """Raised when a provider cannot serve a request."""


class UnsupportedCapabilityError(MarketProviderError):
    """Raised when a caller asks for something the provider does not do."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a source can actually deliver. Every field defaults to False.

    Defaulting to False matters: a capability added to this dataclass later is
    absent for every existing provider until someone deliberately turns it on.
    The alternative -- defaulting to True -- would silently promise a new
    ability on behalf of code written before it existed.
    """

    historical_bars: bool = False
    latest_closed_bar: bool = False
    realtime_ticks: bool = False
    order_book: bool = False
    corporate_actions: bool = False
    market_calendar: str = ""
    data_freshness: str = FRESHNESS_UNSPECIFIED
    authenticated: bool = False
    private_account_data: bool = False

    def __post_init__(self):
        if self.private_account_data and not self.authenticated:
            raise MarketProviderError(
                "private account data without authentication is not a thing a "
                "provider can offer")
        if self.data_freshness not in FRESHNESS_LEVELS:
            raise MarketProviderError(
                f"unknown data_freshness {self.data_freshness!r}; expected one "
                f"of {list(FRESHNESS_LEVELS)}")
        if self.data_freshness == FRESHNESS_LIVE and not (
                self.latest_closed_bar or self.realtime_ticks):
            raise MarketProviderError(
                "a provider cannot claim LIVE freshness while serving neither "
                "a latest closed bar nor ticks")

    def payload(self) -> dict:
        return {
            "historical_bars": self.historical_bars,
            "latest_closed_bar": self.latest_closed_bar,
            "realtime_ticks": self.realtime_ticks,
            "order_book": self.order_book,
            "corporate_actions": self.corporate_actions,
            "market_calendar": self.market_calendar,
            "data_freshness": self.data_freshness,
            "authenticated": self.authenticated,
            "private_account_data": self.private_account_data,
        }


class MarketDataProvider:
    """The interface. Market data in; nothing else, in either direction."""

    provider_id: str = "abstract"
    display_name: str = ""
    capabilities: ProviderCapabilities = ProviderCapabilities()
    # Instruments this provider can serve, by canonical id. Empty means none:
    # a provider must say what it covers rather than being asked to try.
    supported_instruments: tuple[str, ...] = ()

    def supports(self, instrument: object) -> bool:
        try:
            identity = InstrumentId.coerce(instrument)
        except InstrumentError:
            return False
        return identity.canonical in self.supported_instruments

    def require_supported(self, instrument: object) -> InstrumentId:
        identity = InstrumentId.coerce(instrument)
        if identity.canonical not in self.supported_instruments:
            raise MarketProviderError(
                f"{self.provider_id} does not serve {identity.canonical}; it "
                f"serves {list(self.supported_instruments)}")
        return identity

    def get_historical_bars(self, instrument, timeframe, *, start, end):
        raise UnsupportedCapabilityError(
            f"{self.provider_id} does not provide historical bars")

    def get_latest_closed_bars(self, instrument, timeframe, *, now=None,
                               since=None, max_bars=None):
        raise UnsupportedCapabilityError(
            f"{self.provider_id} does not provide a latest closed bar")

    def payload(self) -> dict:
        return {
            "schema_version": MARKET_PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "display_name": self.display_name or self.provider_id,
            "capabilities": self.capabilities.payload(),
            "instruments": list(self.supported_instruments),
        }


COINBASE_PUBLIC_CAPABILITIES = ProviderCapabilities(
    historical_bars=True,
    latest_closed_bar=True,
    # HyprL has no tick or book consumer. Declaring either would be a claim
    # about this project, not about the exchange.
    realtime_ticks=False,
    order_book=False,
    corporate_actions=False,
    market_calendar=CRYPTO_24_7,
    # Serves the most recent *closed* bar as soon as it closes. Live in the
    # only sense this project uses the word: never a forming bar.
    data_freshness=FRESHNESS_LIVE,
    # The public candles endpoint. No key, no header, no cookie -- and
    # therefore, structurally, no account data.
    authenticated=False,
    private_account_data=False,
)


class CoinbasePublicMarketDataProvider(MarketDataProvider):
    """Public Coinbase candles, through the existing Phase 5D transport.

    Every guard the shadow engine relies on stays exactly where it was: the
    request-level embargo check, the raw-payload refusal and the per-bar check
    all live in ``live_market`` and are reached through it. Reimplementing any
    of them here would create a second copy that could drift out of agreement
    with the first, and the first is the one that was audited.
    """

    provider_id = COINBASE_PUBLIC_V1
    display_name = "Coinbase (public market data)"
    capabilities = COINBASE_PUBLIC_CAPABILITIES

    def __init__(self, instruments=()):
        self.supported_instruments = tuple(
            InstrumentId.coerce(item).canonical for item in instruments)

    def get_historical_bars(self, instrument, timeframe, *, start, end):
        """Bars over a closed range, canonicalised by the existing adapter."""
        from scripts.trading_lab.live_market import fetch_closed_candles

        identity = self.require_supported(instrument)
        frame = Timeframe.parse(timeframe)
        return fetch_closed_candles(
            identity.symbol, start=_iso(start), end=_iso(end),
            timeframe=frame.label)

    def get_latest_closed_bars(self, instrument, timeframe, *, now=None,
                               since=None, max_bars=None):
        """Whatever has closed since ``since``. Never a forming bar."""
        from scripts.trading_lab.live_market import (
            MAX_LIVE_CANDLES_PER_POLL, poll_closed_candles)

        identity = self.require_supported(instrument)
        frame = Timeframe.parse(timeframe)
        moment = now or datetime.now(timezone.utc)
        return poll_closed_candles(
            identity.symbol, now=moment, since=since, timeframe=frame.label,
            max_candles=min(int(max_bars or MAX_LIVE_CANDLES_PER_POLL),
                            MAX_LIVE_CANDLES_PER_POLL))


def _iso(value) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


@dataclass(frozen=True)
class InstrumentBar:
    """A bar bound to the instrument it came from.

    A raw bar is six numbers and a timestamp; nothing about it says which
    market produced it. Two dicts of BTC and ETH candles are structurally
    identical, so a mixed-up list is caught by no type and no schema -- it
    just produces a plausible, wrong backtest. Binding the identity to the row
    makes the mistake raise instead.
    """

    instrument_id: InstrumentId
    timeframe: str
    provider_id: str
    bar: dict = field(repr=False)

    @property
    def bar_open_at(self) -> str:
        return str(self.bar.get("bar_open_at", ""))

    def belongs_to(self, instrument: object) -> bool:
        try:
            return InstrumentId.coerce(instrument) == self.instrument_id
        except InstrumentError:
            return False

    def require_instrument(self, instrument: object) -> "InstrumentBar":
        if not self.belongs_to(instrument):
            wanted = InstrumentId.coerce(instrument).canonical
            raise MarketProviderError(
                f"this bar belongs to {self.instrument_id.canonical}, not {wanted}; "
                "refusing to mix instruments in one series")
        return self


def bind_bars(instrument, timeframe, provider_id, rows):
    """Attach an identity to each row of an otherwise anonymous list."""
    identity = InstrumentId.coerce(instrument)
    label = Timeframe.parse(timeframe).label
    return tuple(InstrumentBar(instrument_id=identity, timeframe=label,
                               provider_id=provider_id, bar=dict(row))
                 for row in rows)


def require_single_instrument(bars, instrument=None) -> InstrumentId:
    """Refuse a series that mixes markets. Returns the one identity found."""
    identities = {bar.instrument_id for bar in bars}
    if not identities:
        if instrument is None:
            raise MarketProviderError("an empty series names no instrument")
        return InstrumentId.coerce(instrument)
    if len(identities) > 1:
        names = sorted(item.canonical for item in identities)
        raise MarketProviderError(
            f"a series must cover one instrument; found {names}")
    found = identities.pop()
    if instrument is not None and InstrumentId.coerce(instrument) != found:
        raise MarketProviderError(
            f"series covers {found.canonical}, not "
            f"{InstrumentId.coerce(instrument).canonical}")
    return found
