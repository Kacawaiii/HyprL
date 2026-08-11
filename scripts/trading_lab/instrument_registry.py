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

Registered today: BTC-USD and ETH-USD on Coinbase, served by the public
candles provider. Exactly the two products Phase 1-5 already used -- this
phase changes how they are named, not which ones exist.
"""

from __future__ import annotations

from scripts.trading_lab.instruments import (
    InstrumentError, InstrumentId, InstrumentSpec)
from scripts.trading_lab.market_providers import (
    COINBASE_PUBLIC_V1, CoinbasePublicMarketDataProvider, MarketProviderError)
from scripts.trading_lab.trading_calendar import CRYPTO_24_7

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

INSTRUMENTS_V1 = InstrumentRegistry((BTC_USD, ETH_USD))

PROVIDERS_V1 = ProviderRegistry((
    CoinbasePublicMarketDataProvider(
        instruments=(BTC_USD.instrument_id, ETH_USD.instrument_id)),
))


def resolve_instrument(value: object, *,
                       registry: InstrumentRegistry = INSTRUMENTS_V1) -> InstrumentSpec:
    return registry.resolve(value)


def legacy_product_ids(*, registry: InstrumentRegistry = INSTRUMENTS_V1) -> tuple[str, ...]:
    """The strings Phase 1-5 code and committed artefacts use."""
    return registry.legacy_ids()


__all__ = [
    "BTC_USD", "DEFAULT_VENUE", "ETH_USD", "INSTRUMENTS_V1",
    "INSTRUMENT_REGISTRY_SCHEMA_VERSION", "InstrumentRegistry",
    "MarketProviderError", "PROVIDERS_V1", "ProviderRegistry", "RegistryError",
    "UnknownInstrumentError", "COINBASE_PUBLIC_V1", "legacy_product_ids",
    "resolve_instrument",
]
