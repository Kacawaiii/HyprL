"""Phase 6A: instrument identity, registry, providers and calendars.

Not marked `ml`. This is the identity layer the safety guard depends on, so
it has to be provable in a core install with no model stack -- the same
argument that moved the holdout window into a dependency-free module in 5D.

The normalisation tests are written as bypass attempts rather than as
round-trips. A canonicaliser that only ever sees well-formed input is not
being tested; it is being demonstrated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.trading_lab.instruments import (
    ASSET_CLASSES, InstrumentError, InstrumentId, InstrumentSpec, Timeframe,
    normalize_symbol, normalize_venue, same_symbol)


# --- identity --------------------------------------------------------------


def test_an_instrument_is_named_by_venue_and_symbol():
    identity = InstrumentId(venue="coinbase", symbol="BTC-USD")
    assert identity.canonical == "coinbase:BTC-USD"
    assert identity.base == "BTC" and identity.quote == "USD"
    assert str(identity) == "coinbase:BTC-USD"


@pytest.mark.parametrize("spelling", [
    "BTC-USD", "btc-usd", "BTC-usd", "bTc-UsD",
    "BTC/USD", "btc/usd", "BTC_USD", "BTC.USD", "BTCUSD", "btcusd",
    " BTC-USD", "BTC-USD ", "\tBTC-USD\n", "  btc/usd  ",
])
def test_every_accepted_spelling_folds_onto_one_symbol(spelling):
    assert normalize_symbol(spelling) == "BTC-USD"


@pytest.mark.parametrize("spelling", [
    "coinbase:BTC-USD", "COINBASE:BTC-USD", "Coinbase:btc/usd",
    " coinbase : BTC_USD ", "coinbase:btcusd",
])
def test_every_accepted_spelling_parses_to_one_identity(spelling):
    assert InstrumentId.parse(spelling).canonical == "coinbase:BTC-USD"


def test_two_venues_listing_one_ticker_are_different_instruments():
    """The reason identity is a pair and not a bare symbol."""
    assert InstrumentId(venue="coinbase", symbol="BTC-USD") != \
        InstrumentId(venue="kraken", symbol="BTC-USD")
    assert InstrumentId(venue="coinbase", symbol="BTC-USD") == \
        InstrumentId(venue="COINBASE", symbol="btc/usd")


def test_an_identity_built_directly_is_still_canonical():
    """No code path may hold a non-canonical id, including the constructor."""
    identity = InstrumentId(venue="  COINBASE ", symbol=" btc_usd ")
    assert identity.venue == "coinbase" and identity.symbol == "BTC-USD"


def test_an_identity_is_immutable_and_hashable():
    identity = InstrumentId(venue="coinbase", symbol="BTC-USD")
    assert {identity: 1}[InstrumentId(venue="coinbase", symbol="btcusd")] == 1
    with pytest.raises(Exception):
        identity.symbol = "ETH-USD"


@pytest.mark.parametrize("bad", [
    "", "   ", "-", "---", "/", ":", "BTC-", "-USD", None, 42, b"BTC-USD",
    "BTC-USD-EXTRA-PART", "A" * 100, "BTC\x00USD", "BTC\nUSD", "BTC–USD",
])
def test_an_unusable_symbol_is_refused_rather_than_passed_through(bad):
    """A value nobody can canonicalise must not travel on to a comparison."""
    with pytest.raises(InstrumentError):
        normalize_symbol(bad)


def test_a_bare_symbol_without_a_venue_is_refused():
    """Half an identity must not be completed by guesswork."""
    with pytest.raises(InstrumentError):
        InstrumentId.parse("BTC-USD")
    assert InstrumentId.parse("BTC-USD", default_venue="coinbase").canonical \
        == "coinbase:BTC-USD"


@pytest.mark.parametrize("bad", ["", "  ", "COIN BASE", "coin:base", None, 7])
def test_an_unusable_venue_is_refused(bad):
    with pytest.raises(InstrumentError):
        normalize_venue(bad)


def test_a_longest_first_quote_split_does_not_mangle_usdt():
    assert normalize_symbol("BTCUSDT") == "BTC-USDT"
    assert normalize_symbol("ETHUSDC") == "ETH-USDC"
    assert normalize_symbol("BTCUSD") == "BTC-USD"


def test_symbol_comparison_ignores_venue():
    assert same_symbol("coinbase:BTC-USD", "BTC-USD")
    assert same_symbol("nasdaq:btc/usd", "BTCUSD")
    assert not same_symbol("BTC-USD", "ETH-USD")
    assert not same_symbol("nonsense", "BTC-USD")


# --- specification ---------------------------------------------------------


def _spec(**overrides) -> InstrumentSpec:
    payload = dict(
        instrument_id=InstrumentId(venue="coinbase", symbol="BTC-USD"),
        asset_class="CRYPTO", base_asset="BTC", quote_asset="USD",
        price_currency="USD", timezone="UTC", trading_calendar="CRYPTO_24_7",
        native_timeframes=("1h", "1d"), quantity_precision=8, price_precision=2)
    payload.update(overrides)
    return InstrumentSpec(**payload)


def test_the_asset_classes_are_a_closed_set():
    assert ASSET_CLASSES == ("CRYPTO", "EQUITY", "ETF", "INDEX", "FX")
    # Naming a class would be a claim the identity layer models it.
    assert "FUTURE" not in ASSET_CLASSES and "OPTION" not in ASSET_CLASSES


def test_an_unknown_asset_class_is_refused():
    with pytest.raises(InstrumentError):
        _spec(asset_class="PERPETUAL_SWAP")


def test_a_specification_hashes_its_own_content():
    assert _spec().instrument_spec_hash == _spec().instrument_spec_hash
    assert _spec().instrument_spec_hash != _spec(price_precision=4).instrument_spec_hash


def test_the_hash_covers_identity_calendar_and_precision():
    baseline = _spec().instrument_spec_hash
    for change in ({"trading_calendar": "NYSE"}, {"quantity_precision": 6},
                   {"timezone": "America/New_York"}, {"asset_class": "EQUITY"},
                   {"native_timeframes": ("1h",)}):
        assert _spec(**change).instrument_spec_hash != baseline, change


def test_the_hash_ignores_the_display_name():
    """A label is for humans; changing it must not restate the instrument."""
    assert _spec(display_name="Bitcoin").instrument_spec_hash == \
        _spec(display_name="BTC").instrument_spec_hash


@pytest.mark.parametrize("bad", [
    {"quantity_precision": -1}, {"price_precision": 99},
    {"base_asset": ""}, {"quote_asset": "  "}, {"native_timeframes": ()},
    {"native_timeframes": ("1 hour",)},
])
def test_an_incoherent_specification_is_refused(bad):
    with pytest.raises(InstrumentError):
        _spec(**bad)


def test_a_specification_bridges_to_the_legacy_identifiers():
    """Committed artefacts say 'BTC-USD'; rewriting them would break hashes."""
    spec = _spec()
    assert spec.legacy_product_id == "BTC-USD"
    assert spec.legacy_asset == "BTC/USD"
    assert spec.canonical_id == "coinbase:BTC-USD"


def test_a_specification_knows_which_timeframes_it_has():
    spec = _spec()
    assert spec.supports("1h") and spec.supports(Timeframe.parse("1d"))
    assert not spec.supports("5m")


# --- timeframe -------------------------------------------------------------


@pytest.mark.parametrize("label,duration", [
    ("1h", timedelta(hours=1)), ("4h", timedelta(hours=4)),
    ("1d", timedelta(days=1)), ("15m", timedelta(minutes=15)),
])
def test_a_timeframe_carries_its_duration(label, duration):
    frame = Timeframe.parse(label)
    assert frame.duration == duration
    assert frame.label == label


def test_the_timeframe_label_matches_the_legacy_spelling_exactly():
    """'1h' is written into committed artefacts; it may not become '1H'."""
    assert Timeframe.parse("1h").label == "1h"
    assert Timeframe.parse("1H").label == "1h"
    assert str(Timeframe(unit="h", count=1)) == "1h"


@pytest.mark.parametrize("bad", ["", "1", "h", "1y", "0h", "-1h", "1hour",
                                 "9999h", None, 3600])
def test_an_unusable_timeframe_is_refused(bad):
    with pytest.raises(InstrumentError):
        Timeframe.parse(bad)


# --- calendar --------------------------------------------------------------


def test_the_crypto_calendar_never_closes():
    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    for moment in ("2026-01-01T00:00:00Z", "2026-08-15T03:30:00Z",
                   "2026-12-25T12:00:00Z"):
        assert CRYPTO_247_CALENDAR.is_session_open(moment)


def test_the_crypto_calendar_states_the_annualisation_factor():
    """8760 is a property of a market that never closes, not of arithmetic."""
    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    assert CRYPTO_247_CALENDAR.bars_per_day("1h") == 24
    assert CRYPTO_247_CALENDAR.annualization_periods("1h") == 8760
    assert CRYPTO_247_CALENDAR.bars_per_day("1d") == 1
    assert CRYPTO_247_CALENDAR.annualization_periods("1d") == 365


def test_the_next_bar_opening_sits_on_a_shared_grid():
    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    # Two callers asking at different instants must agree where the grid falls.
    first = CRYPTO_247_CALENDAR.next_expected_bar_open("2026-08-11T03:17:00Z", "1h")
    second = CRYPTO_247_CALENDAR.next_expected_bar_open("2026-08-11T03:59:59Z", "1h")
    assert first == second == datetime(2026, 8, 11, 4, tzinfo=timezone.utc)
    # exactly on a boundary means the NEXT one, never the current bar
    assert CRYPTO_247_CALENDAR.next_expected_bar_open("2026-08-11T04:00:00Z", "1h") \
        == datetime(2026, 8, 11, 5, tzinfo=timezone.utc)


def test_a_naive_timestamp_is_refused_by_the_calendar():
    from scripts.trading_lab.trading_calendar import (
        CRYPTO_247_CALENDAR, TradingCalendarError)

    with pytest.raises(TradingCalendarError):
        CRYPTO_247_CALENDAR.is_session_open(datetime(2026, 8, 11, 3))


def test_an_unimplemented_calendar_never_falls_back_to_crypto():
    """An equity annualised by 8760 looks twice as good as it is."""
    from scripts.trading_lab.trading_calendar import (
        TradingCalendarError, get_calendar, known_calendars)

    assert known_calendars() == ("CRYPTO_24_7", "US_EQUITY_REGULAR")
    # XNYS is a real MIC and the name the equity calendar uses internally, and
    # it still resolves to nothing here: only the platform's own calendar ids
    # are addressable, so a vendor's exchange code cannot pick a calendar.
    with pytest.raises(TradingCalendarError) as error:
        get_calendar("XNYS")
    assert "must not borrow" in str(error.value)
    for spelling in ("crypto_24_7", "CRYPTO", "", "US_EQUITY", "NASDAQ"):
        with pytest.raises(TradingCalendarError):
            get_calendar(spelling)


def test_a_timeframe_that_does_not_divide_a_day_is_refused():
    from scripts.trading_lab.trading_calendar import (
        CRYPTO_247_CALENDAR, TradingCalendarError)

    with pytest.raises(TradingCalendarError):
        CRYPTO_247_CALENDAR.bars_per_day("7h")


# --- registry --------------------------------------------------------------


def test_the_registry_holds_exactly_the_two_current_products():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    assert INSTRUMENTS_V1.ids() == ("coinbase:BTC-USD", "coinbase:ETH-USD")
    assert INSTRUMENTS_V1.legacy_ids() == ("BTC-USD", "ETH-USD")
    assert len(INSTRUMENTS_V1) == 2


@pytest.mark.parametrize("spelling", [
    "BTC-USD", "btc-usd", "BTC/USD", "BTCUSD", "coinbase:BTC-USD",
    "COINBASE:btc_usd", " BTC-USD ",
])
def test_the_registry_resolves_every_spelling_to_one_specification(spelling):
    from scripts.trading_lab.instrument_registry import BTC_USD, INSTRUMENTS_V1

    assert INSTRUMENTS_V1.resolve(spelling) is BTC_USD


@pytest.mark.parametrize("unknown", [
    "SOL-USD", "nasdaq:AAPL", "AAPL", "", "---", None, 42, "coinbase:",
])
def test_an_unregistered_instrument_fails_closed(unknown):
    from scripts.trading_lab.instrument_registry import (
        INSTRUMENTS_V1, UnknownInstrumentError)

    with pytest.raises(UnknownInstrumentError):
        INSTRUMENTS_V1.resolve(unknown)
    assert INSTRUMENTS_V1.get(unknown) is None
    assert unknown not in INSTRUMENTS_V1


def test_a_duplicate_instrument_is_refused_at_construction():
    from scripts.trading_lab.instrument_registry import (
        BTC_USD, InstrumentRegistry, RegistryError)

    with pytest.raises(RegistryError):
        InstrumentRegistry((BTC_USD, BTC_USD))


def test_two_specifications_sharing_a_legacy_id_are_refused():
    """The event log and every committed artefact are keyed by that string."""
    from scripts.trading_lab.instrument_registry import (
        BTC_USD, InstrumentRegistry, RegistryError)

    twin = InstrumentSpec(
        instrument_id=InstrumentId(venue="kraken", symbol="BTC-USD"),
        asset_class="CRYPTO", base_asset="BTC", quote_asset="USD",
        price_currency="USD", timezone="UTC", trading_calendar="CRYPTO_24_7",
        native_timeframes=("1h",), quantity_precision=8, price_precision=2)
    with pytest.raises(RegistryError):
        InstrumentRegistry((BTC_USD, twin))


def test_the_registry_groups_by_asset_class():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    grouped = INSTRUMENTS_V1.by_asset_class()
    assert list(grouped) == ["CRYPTO"]
    assert len(grouped["CRYPTO"]) == 2


def test_the_registry_loads_nothing_from_the_filesystem():
    """A market that appears because a file appeared is a market nobody read."""
    import inspect

    from scripts.trading_lab import instrument_registry

    source = inspect.getsource(instrument_registry)
    for mechanism in ("importlib", "__import__", "glob", "iterdir", "rglob",
                      "open(", "load_module", "entry_points"):
        assert mechanism not in source, f"registry reaches for {mechanism}"


def test_the_registered_specifications_have_stable_hashes():
    """These identify the instruments in manifests; drifting silently is the
    failure this pins down."""
    from scripts.trading_lab.instrument_registry import BTC_USD, ETH_USD

    assert BTC_USD.instrument_spec_hash == (
        "492c167c1e66a37a377cff8b4e135841c5a13a7c60324ec9b5c8b1976bf5701f")
    assert ETH_USD.instrument_spec_hash == (
        "2a9e1d1c922fbb9af68ad92f8f2638e951a2fef830afb6e514a29ebfab35d7f3")


# --- providers -------------------------------------------------------------


def test_the_coinbase_provider_declares_only_what_it_implements():
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1

    provider = PROVIDERS_V1.resolve("coinbase-public-v1")
    capabilities = provider.capabilities
    assert capabilities.historical_bars is True
    assert capabilities.latest_closed_bar is True
    # HyprL consumes neither, so claiming them would be a lie about this project
    assert capabilities.realtime_ticks is False
    assert capabilities.order_book is False
    assert capabilities.corporate_actions is False
    assert capabilities.market_calendar == "CRYPTO_24_7"


def test_the_public_provider_is_unauthenticated_and_holds_no_account_data():
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1

    capabilities = PROVIDERS_V1.resolve("coinbase-public-v1").capabilities
    assert capabilities.authenticated is False
    assert capabilities.private_account_data is False


def test_private_account_data_without_authentication_is_incoherent():
    from scripts.trading_lab.market_providers import (
        MarketProviderError, ProviderCapabilities)

    with pytest.raises(MarketProviderError):
        ProviderCapabilities(private_account_data=True, authenticated=False)


def test_capabilities_default_to_absent():
    """A capability added later must be off for every provider written before."""
    from scripts.trading_lab.market_providers import ProviderCapabilities

    blank = ProviderCapabilities()
    for name, value in blank.payload().items():
        if isinstance(value, str):
            continue
        assert value is False, name
    # The descriptive fields have an "absent" value too, and it is never one a
    # caller may act on: an unnamed calendar and unspecified freshness both
    # mean "this provider has not said", not "24/7" and not "live".
    assert blank.market_calendar == ""
    assert blank.data_freshness == "UNSPECIFIED"


def test_no_provider_exposes_an_order_or_an_account():
    """Execution is a different interface with a different threat model."""
    from scripts.trading_lab.market_providers import MarketDataProvider

    for forbidden in ("place_order", "cancel_order", "account", "balance",
                      "wallet", "positions", "withdraw", "deposit"):
        assert not hasattr(MarketDataProvider, forbidden), forbidden


def test_a_provider_refuses_an_instrument_it_does_not_serve():
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1
    from scripts.trading_lab.market_providers import MarketProviderError

    provider = PROVIDERS_V1.resolve("coinbase-public-v1")
    assert provider.supports("BTC-USD") is False       # bare form names no venue
    assert provider.supports("coinbase:BTC-USD") is True
    with pytest.raises(MarketProviderError):
        provider.require_supported("coinbase:SOL-USD")


def test_the_abstract_provider_refuses_rather_than_pretending():
    from scripts.trading_lab.market_providers import (
        MarketDataProvider, UnsupportedCapabilityError)

    provider = MarketDataProvider()
    with pytest.raises(UnsupportedCapabilityError):
        provider.get_historical_bars("coinbase:BTC-USD", "1h",
                                     start="2026-01-01T00:00:00Z",
                                     end="2026-01-02T00:00:00Z")
    with pytest.raises(UnsupportedCapabilityError):
        provider.get_latest_closed_bars("coinbase:BTC-USD", "1h")


def test_a_duplicate_provider_is_refused():
    from scripts.trading_lab.instrument_registry import ProviderRegistry, RegistryError
    from scripts.trading_lab.market_providers import CoinbasePublicMarketDataProvider

    one = CoinbasePublicMarketDataProvider(instruments=("coinbase:BTC-USD",))
    with pytest.raises(RegistryError):
        ProviderRegistry((one, CoinbasePublicMarketDataProvider(
            instruments=("coinbase:ETH-USD",))))


def test_the_provider_registry_finds_who_serves_an_instrument():
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1

    found = PROVIDERS_V1.for_instrument("coinbase:BTC-USD")
    assert [provider.provider_id for provider in found] == ["coinbase-public-v1"]
    assert PROVIDERS_V1.for_instrument("coinbase:SOL-USD") == ()


def test_the_coinbase_provider_reuses_the_existing_transport():
    """One parser, one guard. A second copy is a second thing to drift."""
    import inspect

    from scripts.trading_lab import market_providers

    source = inspect.getsource(market_providers)
    # it calls into live_market rather than reimplementing a request
    assert "fetch_closed_candles" in source and "poll_closed_candles" in source
    for duplicated in ("urlopen", "Request(", "json.loads", "Decimal(",
                       "api.exchange.coinbase.com"):
        assert duplicated not in source, f"provider duplicates {duplicated}"


# --- bars bound to an instrument -------------------------------------------


def _rows(count, start="2026-08-01T00:00:00+00:00"):
    from datetime import datetime as dt
    base = dt.fromisoformat(start)
    return [{"bar_open_at": (base + timedelta(hours=index)).isoformat(),
             "open": "20000", "high": "20010", "low": "19990",
             "close": "20005", "volume": "1.0"} for index in range(count)]


def test_a_bar_carries_the_instrument_it_came_from():
    from scripts.trading_lab.market_providers import bind_bars

    bars = bind_bars("coinbase:BTC-USD", "1h", "coinbase-public-v1", _rows(3))
    assert len(bars) == 3
    assert all(bar.instrument_id.canonical == "coinbase:BTC-USD" for bar in bars)
    assert bars[0].belongs_to("coinbase:btcusd")


def test_a_btc_bar_cannot_be_used_as_an_eth_bar():
    """Two candle dicts are structurally identical; only identity catches this."""
    from scripts.trading_lab.market_providers import MarketProviderError, bind_bars

    bar = bind_bars("coinbase:BTC-USD", "1h", "coinbase-public-v1", _rows(1))[0]
    assert not bar.belongs_to("coinbase:ETH-USD")
    with pytest.raises(MarketProviderError) as error:
        bar.require_instrument("coinbase:ETH-USD")
    assert "refusing to mix instruments" in str(error.value)


def test_a_series_mixing_two_markets_is_refused():
    from scripts.trading_lab.market_providers import (
        MarketProviderError, bind_bars, require_single_instrument)

    mixed = (*bind_bars("coinbase:BTC-USD", "1h", "coinbase-public-v1", _rows(2)),
             *bind_bars("coinbase:ETH-USD", "1h", "coinbase-public-v1", _rows(2)))
    with pytest.raises(MarketProviderError) as error:
        require_single_instrument(mixed)
    assert "one instrument" in str(error.value)


def test_a_single_market_series_reports_its_instrument():
    from scripts.trading_lab.market_providers import bind_bars, require_single_instrument

    bars = bind_bars("coinbase:BTC-USD", "1h", "coinbase-public-v1", _rows(4))
    assert require_single_instrument(bars).canonical == "coinbase:BTC-USD"
    assert require_single_instrument(bars, "coinbase:btc/usd").symbol == "BTC-USD"


def test_a_series_checked_against_the_wrong_instrument_is_refused():
    from scripts.trading_lab.market_providers import (
        MarketProviderError, bind_bars, require_single_instrument)

    bars = bind_bars("coinbase:BTC-USD", "1h", "coinbase-public-v1", _rows(2))
    with pytest.raises(MarketProviderError):
        require_single_instrument(bars, "coinbase:ETH-USD")
