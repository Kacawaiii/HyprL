"""The closed set of markets this build knows about.

Two registries, both immutable and both fail-closed. An unknown instrument
raises; it is never treated as "probably fine" and never invented from a
string. That is the whole point of having a registry rather than passing
symbols around: there is one list, it is checked, and code downstream can stop
asking whether the thing it was handed is real.

**No plugin loading.** Nothing here scans a directory, imports by name, or
reads a registry file from disk. A market that appears because a file appeared
is a market nobody reviewed, and the same mechanism would let a protected
instrument be re-registered under a different identity.

**Known is not tradable.** Two registries answer two different questions.
``INSTRUMENTS_V1`` is what this build trades: BTC-USD and ETH-USD, exactly the
products Phase 1-5 used, and adding to it would silently enlist a market in the
shared paper portfolio, the holdout guard and every backtest. ``CATALOGUE_V1``
is what this build can *describe*, which now also includes four US equities.
They are registered so their sessions, calendar and provider can be inspected;
they are not tradable, have no model, no signal and no paper runtime, and
``is_tradable`` is the one function that says which is which.

Keeping them apart is not tidiness. The shared portfolio builds its batch from
the tradable registry and waits until every instrument in it has supplied a
target; an equity in that list would mean a batch that never completes and a
paper runtime that stops trading without ever raising.
"""

from __future__ import annotations

from scripts.trading_lab.instruments import (
    InstrumentError, InstrumentId, InstrumentSpec)
from scripts.trading_lab.market_providers import (
    COINBASE_PUBLIC_V1, CoinbasePublicMarketDataProvider, MarketProviderError)
from scripts.trading_lab.trading_calendar import CRYPTO_24_7, US_EQUITY_REGULAR

INSTRUMENT_REGISTRY_SCHEMA_VERSION = "trading-lab.instrument-registry.v1"

DEFAULT_VENUE = "coinbase"


class RegistryError(RuntimeError):
    """Raised when a registry is built or queried unsafely."""


class UnknownInstrumentError(RegistryError):
    """Raised when an instrument is not registered. Never a silent default."""


class InstrumentRegistry:
    """An immutable, duplicate-free set of instrument specifications."""

    def __init__(self, specs=()):
        by_id: dict[str, InstrumentSpec] = {}
        for spec in specs:
            if not isinstance(spec, InstrumentSpec):
                raise RegistryError("an instrument registry holds InstrumentSpec objects")
            key = spec.instrument_id.canonical
            legacy = spec.legacy_product_id
            # One check, not two. A separate canonical-id test would be dead
            # code: legacy_product_id is the symbol, so any canonical
            # collision is already a legacy collision -- while the reverse is
            # not true, because two venues can list the same symbol and the
            # committed artefacts and the hash-chained event log are keyed by
            # that bare string alone.
            clash = next((existing for existing in by_id.values()
                          if existing.instrument_id.canonical == key
                          or existing.legacy_product_id == legacy), None)
            if clash is not None:
                raise RegistryError(
                    f"{key} collides with the already registered "
                    f"{clash.instrument_id.canonical} (both resolve to the "
                    f"legacy product id {legacy!r}); two specifications for one "
                    "market would make every lookup order-dependent")
            by_id[key] = spec
        self._by_id = dict(by_id)
        self._by_legacy = {spec.legacy_product_id: spec for spec in by_id.values()}

    # --- lookup ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, value: object) -> bool:
        try:
            self.resolve(value)
        except RegistryError:
            return False
        return True

    def all(self) -> tuple[InstrumentSpec, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def legacy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_legacy))

    def resolve(self, value: object) -> InstrumentSpec:
        """Any accepted spelling to the one registered specification, or raise.

        Accepts a spec, an id, a canonical string, and a bare legacy symbol.
        The bare form is what every Phase 1-5 caller passes, so refusing it
        would mean rewriting call sites purely to satisfy a naming change.
        """
        if isinstance(value, InstrumentSpec):
            found = self._by_id.get(value.instrument_id.canonical)
            if found is None:
                raise UnknownInstrumentError(
                    f"{value.instrument_id.canonical} is not registered")
            return found
        try:
            identity = InstrumentId.coerce(value, default_venue=DEFAULT_VENUE)
        except InstrumentError as error:
            raise UnknownInstrumentError(
                f"{value!r} is not a usable instrument identity: {error}") from error
        found = self._by_id.get(identity.canonical)
        if found is None:
            raise UnknownInstrumentError(
                f"{identity.canonical} is not registered; registered: "
                f"{list(self.ids())}")
        return found

    def get(self, value: object, default=None):
        try:
            return self.resolve(value)
        except RegistryError:
            return default

    def by_asset_class(self) -> dict:
        grouped: dict[str, list] = {}
        for spec in self.all():
            grouped.setdefault(spec.asset_class, []).append(spec)
        return grouped

    def payload(self) -> dict:
        return {
            "schema_version": INSTRUMENT_REGISTRY_SCHEMA_VERSION,
            "count": len(self._by_id),
            "instruments": [spec.payload() for spec in self.all()],
        }


class ProviderRegistry:
    """An immutable, duplicate-free set of market data providers."""

    def __init__(self, providers=()):
        by_id: dict[str, object] = {}
        for provider in providers:
            key = getattr(provider, "provider_id", None)
            if not key:
                raise RegistryError("a provider must declare a provider_id")
            if key in by_id:
                raise RegistryError(f"provider {key!r} is already registered")
            by_id[key] = provider
        self._by_id = dict(by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, provider_id: object) -> bool:
        return provider_id in self._by_id

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def all(self) -> tuple:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def resolve(self, provider_id: object):
        if provider_id not in self._by_id:
            raise RegistryError(
                f"provider {provider_id!r} is not registered; registered: "
                f"{list(self.ids())}")
        return self._by_id[provider_id]

    def for_instrument(self, instrument: object) -> tuple:
        """Every provider that says it serves this market. Possibly none."""
        return tuple(provider for provider in self.all()
                     if provider.supports(instrument))

    def payload(self) -> dict:
        return {
            "schema_version": INSTRUMENT_REGISTRY_SCHEMA_VERSION,
            "count": len(self._by_id),
            "providers": [provider.payload() for provider in self.all()],
        }


def _crypto_spec(symbol: str, base: str, display: str) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=InstrumentId(venue=DEFAULT_VENUE, symbol=symbol),
        asset_class="CRYPTO",
        base_asset=base,
        quote_asset="USD",
        price_currency="USD",
        timezone="UTC",
        trading_calendar=CRYPTO_24_7,
        # 1h is what the corpus, the benchmarks and the shadow engine use. 1d
        # is listed because the Coinbase adapter already accepts it; nothing
        # else is claimed.
        native_timeframes=("1h", "1d"),
        quantity_precision=8,
        price_precision=2,
        display_name=display,
    )


BTC_USD = _crypto_spec("BTC-USD", "BTC", "Bitcoin / US Dollar")
ETH_USD = _crypto_spec("ETH-USD", "ETH", "Ether / US Dollar")

# The tradable universe. Unchanged since Phase 6A, and the reason equities
# below are not in it.
INSTRUMENTS_V1 = InstrumentRegistry((BTC_USD, ETH_USD))


def _equity_spec(venue: str, symbol: str, asset_class: str,
                 display: str) -> InstrumentSpec:
    """One US equity or ETF, described and not traded.

    The venue and asset class here are not guesses -- each is asserted against
    a recorded vendor reference payload in the test suite, so a wrong venue
    fails a test rather than quietly naming a different instrument. They are
    written out literally rather than read from that fixture at import time
    because a registry that builds itself from a file on disk is a registry a
    replaced file can extend, which is precisely what this module refuses to be.
    """
    return InstrumentSpec(
        instrument_id=InstrumentId(venue=venue, symbol=symbol),
        asset_class=asset_class,
        base_asset=symbol,
        quote_asset="USD",
        price_currency="USD",
        # The exchange's own clock. Every session boundary, DST shift and early
        # close is expressed in it; UTC would be a conversion, not the rule.
        timezone="America/New_York",
        trading_calendar=US_EQUITY_REGULAR,
        # 30m because a US session holds thirteen of them exactly. 1h is
        # deliberately absent: a 6h30 session does not contain a whole number
        # of hourly bars, so an hourly grid here would be a rounding.
        native_timeframes=("30m", "1d"),
        # On-exchange US equities trade in whole shares. Fractional shares are
        # a broker construct, and this build has no broker.
        quantity_precision=0,
        price_precision=2,
        display_name=display,
    )


AAPL = _equity_spec("xnas", "AAPL", "EQUITY", "Apple Inc.")
MSFT = _equity_spec("xnas", "MSFT", "EQUITY", "Microsoft Corporation")
NVDA = _equity_spec("xnas", "NVDA", "EQUITY", "NVIDIA Corporation")
QQQ = _equity_spec("xnas", "QQQ", "ETF", "Invesco QQQ Trust, Series 1")

EQUITY_INSTRUMENTS_V1 = InstrumentRegistry((AAPL, MSFT, NVDA, QQQ))

# Everything this build can describe. Browsing, metadata, sessions, calendars.
# Never the source of a trading universe.
CATALOGUE_V1 = InstrumentRegistry((BTC_USD, ETH_USD, AAPL, MSFT, NVDA, QQQ))


def _massive_provider():
    """The equity provider, offline.

    Built with no transport, so the registered provider cannot make a request
    even if something asks it to. A transport is injected by a caller that has
    decided to go to the network; nothing in this module has decided that.
    """
    from scripts.trading_lab.massive_provider import MassiveStocksHistoricalProvider

    return MassiveStocksHistoricalProvider(
        instruments=tuple(spec.instrument_id
                          for spec in EQUITY_INSTRUMENTS_V1.all()))


PROVIDERS_V1 = ProviderRegistry((
    CoinbasePublicMarketDataProvider(
        instruments=(BTC_USD.instrument_id, ETH_USD.instrument_id)),
    _massive_provider(),
))


def is_tradable(value: object, *,
                registry: InstrumentRegistry = INSTRUMENTS_V1) -> bool:
    """Whether this build may trade the instrument, as opposed to describe it.

    Fail-closed: anything unresolvable is not tradable. An unknown instrument
    is never given the benefit of the doubt.
    """
    return registry.get(value) is not None


def resolve_instrument(value: object, *,
                       registry: InstrumentRegistry = INSTRUMENTS_V1) -> InstrumentSpec:
    return registry.resolve(value)


def resolve_known_instrument(value: object) -> InstrumentSpec:
    """Resolve against the full catalogue, tradable or not."""
    return CATALOGUE_V1.resolve(value)


def legacy_product_ids(*, registry: InstrumentRegistry = INSTRUMENTS_V1) -> tuple[str, ...]:
    """The strings Phase 1-5 code and committed artefacts use."""
    return registry.legacy_ids()


__all__ = [
    "AAPL", "BTC_USD", "CATALOGUE_V1", "DEFAULT_VENUE",
    "EQUITY_INSTRUMENTS_V1", "ETH_USD", "INSTRUMENTS_V1",
    "INSTRUMENT_REGISTRY_SCHEMA_VERSION", "InstrumentRegistry", "MSFT",
    "MarketProviderError", "NVDA", "PROVIDERS_V1", "ProviderRegistry", "QQQ",
    "RegistryError", "UnknownInstrumentError", "COINBASE_PUBLIC_V1",
    "is_tradable", "legacy_product_ids", "resolve_instrument",
    "resolve_known_instrument",
]
