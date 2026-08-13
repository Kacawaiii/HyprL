"""The Massive provider, its credential boundary, and corporate actions.

Every test in this file runs offline. Not "mocked so it doesn't hit the
network" -- the provider under test has no transport unless one is handed to
it, so there is nothing to intercept. The credential tests use a sentinel
string that is never a real key, and they fail if that string can be found
anywhere it should not be: a payload, a repr, a log record, an exception, or a
support bundle.

The leak tests are written the way a leak actually happens. Nobody logs a key
on purpose. They log a config dict that contains one, or format an object into
an error message, or serialise a provider into an API response. So each test
takes a whole structure and searches it, rather than asserting that one
particular field was blanked.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from scripts.trading_lab.credentials import (
    CredentialError, EnvCredentialProvider, MASSIVE_API_KEY_ENV,
    MissingCredentialProvider, REDACTED, Secret, assert_absent, redact)
from scripts.trading_lab.equity_market import (
    ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED, ADJUSTMENT_TOTAL_RETURN,
    CashDividend, EquityCorpusSpec, EquityMarketError, StockSplit,
    apply_splits, build_equity_bar, require_single_policy)
from scripts.trading_lab.instrument_registry import (
    AAPL, EQUITY_INSTRUMENTS_V1, PROVIDERS_V1)
from scripts.trading_lab.market_providers import UnsupportedCapabilityError
from scripts.trading_lab.massive_provider import (
    MASSIVE_STOCKS_HISTORICAL_V1, MassiveProviderError,
    MassiveStocksHistoricalProvider, NoNetworkTransport, OfflineTransportError,
    RecordedTransport, parse_reference_ticker)

# Obviously not a key. Distinctive enough that a substring search for it is
# meaningful, and harmless if it ever escapes.
SENTINEL = "sentinel-massive-key-do-not-use-1a2b3c4d5e6f"

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "crypto"


@pytest.fixture(scope="module")
def reference_fixture() -> dict:
    return json.loads((FIXTURES / "massive_reference_tickers.json").read_text())


# 2026-01-15 is a regular session: opens 14:30Z, so 14:30 and 15:00 are the
# first two 30-minute openings. Timestamps are epoch milliseconds, as sent.
BAR_ONE_MS = 1768487400000   # 2026-01-15T14:30:00Z
BAR_TWO_MS = 1768489200000   # 2026-01-15T15:00:00Z

AAPL_BARS_PATH = "/v2/aggs/ticker/AAPL/range/30/minute/2026-01-15/2026-01-16"


def _bars_payload(adjusted=False, rows=None, ticker="AAPL"):
    """One aggregates page in the vendor's shape."""
    rows = rows if rows is not None else [
        {"t": BAR_ONE_MS, "o": 185.10, "h": 186.40, "l": 184.90,
         "c": 186.00, "v": 1250000, "n": 900, "vw": 185.6},
        {"t": BAR_TWO_MS, "o": 186.00, "h": 186.75, "l": 185.55,
         "c": 185.80, "v": 980000, "n": 700, "vw": 186.1},
    ]
    return {"ticker": ticker, "adjusted": adjusted, "status": "OK",
            "queryCount": len(rows), "resultsCount": len(rows),
            "results": rows, "request_id": "fixture"}


def _provider(responses=None, credentials=None, **kwargs):
    transport = RecordedTransport(responses) if responses is not None else None
    return MassiveStocksHistoricalProvider(
        instruments=tuple(spec.instrument_id
                          for spec in EQUITY_INSTRUMENTS_V1.all()),
        transport=transport,
        credentials=credentials or EnvCredentialProvider(MASSIVE_API_KEY_ENV),
        **kwargs)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv(MASSIVE_API_KEY_ENV, SENTINEL)
    return SENTINEL


# --- the provider makes no calls by itself ---------------------------------


def test_a_provider_without_a_transport_cannot_reach_the_network():
    """Offline by construction, not by a mock that happens to be installed."""
    provider = MassiveStocksHistoricalProvider(instruments=(AAPL.instrument_id,))
    assert isinstance(provider.transport, NoNetworkTransport)
    assert provider.payload()["network_enabled"] is False
    with pytest.raises(OfflineTransportError):
        provider.get_historical_bars(AAPL.instrument_id, "30m",
                                     start="2026-01-15T00:00:00Z",
                                     end="2026-01-16T00:00:00Z")


def test_the_registered_provider_is_the_offline_one():
    """Importing the registry must not create anything that can call out."""
    provider = PROVIDERS_V1.resolve(MASSIVE_STOCKS_HISTORICAL_V1)
    assert isinstance(provider.transport, NoNetworkTransport)
    assert provider.payload()["network_enabled"] is False


def test_the_offline_transport_raises_rather_than_returning_nothing():
    """An empty result would read as "this instrument had no bars"."""
    with pytest.raises(OfflineTransportError) as error:
        NoNetworkTransport().request("/v1/stocks/bars", {}, {})
    assert "no transport" in str(error.value)


# --- market data only ------------------------------------------------------


def test_no_account_or_order_path_can_be_requested(keyed):
    provider = _provider({})
    for path in ("/v1/account/balance", "/v1/orders", "/v1/positions",
                 "/v1/account", "/v2/portfolio/holdings", "/v1/trades/execute"):
        with pytest.raises(MassiveProviderError) as error:
            provider._fetch(path, {})
        assert "market data only" in str(error.value)


def test_a_path_outside_the_allowlist_is_refused(keyed):
    provider = _provider({})
    with pytest.raises(MassiveProviderError) as error:
        provider._fetch("/v1/stocks/quotes", {})
    assert "allowlist" in str(error.value)


def test_the_provider_declares_no_private_account_data():
    provider = PROVIDERS_V1.resolve(MASSIVE_STOCKS_HISTORICAL_V1)
    capabilities = provider.capabilities
    assert capabilities.authenticated is True
    assert capabilities.private_account_data is False
    assert capabilities.order_book is False
    assert capabilities.realtime_ticks is False
    assert provider.payload()["brokerage_endpoints"] == []
    for name in ("place_order", "account", "balance", "positions", "withdraw"):
        assert not hasattr(provider, name)


def test_an_end_of_day_source_refuses_to_answer_what_just_closed(keyed):
    """Returning its most recent row would hand a live caller a stale price."""
    provider = _provider({})
    assert provider.capabilities.data_freshness == "END_OF_DAY"
    assert provider.capabilities.latest_closed_bar is False
    with pytest.raises(UnsupportedCapabilityError) as error:
        provider.get_latest_closed_bars(AAPL.instrument_id, "30m")
    assert "END_OF_DAY" in str(error.value)


# --- the credential --------------------------------------------------------


def test_a_secret_refuses_every_way_of_printing_itself():
    secret = Secret(SENTINEL, name="test-key")
    assert SENTINEL not in repr(secret)
    assert SENTINEL not in str(secret)
    assert SENTINEL not in f"{secret}"
    assert SENTINEL not in f"{secret!s}"
    assert SENTINEL not in f"{secret!r}"
    assert SENTINEL not in "{}".format(secret)
    assert SENTINEL not in f"{secret:>40}"
    assert SENTINEL not in json.dumps({"key": secret}, default=str)
    assert REDACTED in str(secret)
    # The one deliberate way out.
    assert secret.reveal() == SENTINEL


def test_a_secret_cannot_be_pickled_or_hashed():
    import pickle

    secret = Secret(SENTINEL)
    with pytest.raises(CredentialError):
        pickle.dumps(secret)
    with pytest.raises(CredentialError):
        hash(secret)
    with pytest.raises(CredentialError):
        {secret: 1}


def test_the_key_never_appears_in_the_provider_payload(keyed):
    provider = _provider({})
    payload = provider.payload()
    assert_absent(SENTINEL, payload, where="provider payload")
    assert payload["credential_configured"] is True
    assert payload["credential_required"] is True
    # And no field shaped like somewhere to put a key.
    rendered = json.dumps(payload).lower()
    for banned in ("api_key", "apikey", "authorization", "bearer", "token"):
        assert banned not in rendered


def test_the_key_never_appears_in_a_repr_or_an_exception(keyed):
    provider = _provider({})
    assert_absent(SENTINEL, repr(provider), where="provider repr")
    assert_absent(SENTINEL, repr(provider.credentials), where="credentials repr")
    with pytest.raises(MassiveProviderError) as error:
        provider._fetch(AAPL_BARS_PATH, {})
    assert_absent(SENTINEL, str(error.value), where="exception message")
    assert_absent(SENTINEL, repr(error.value), where="exception repr")


def test_a_transport_failure_does_not_re_raise_the_vendors_message(keyed):
    """A vendor exception often quotes the request, headers included."""

    class Leaky:
        name = "leaky"

        def request(self, path, params, headers):
            raise RuntimeError(f"HTTP 500 for {path} with {headers}")

    provider = _provider({})
    provider.transport = Leaky()
    with pytest.raises(MassiveProviderError) as error:
        provider.get_historical_bars(AAPL.instrument_id, "30m",
                                     start="2026-01-15T00:00:00Z",
                                     end="2026-01-16T00:00:00Z")
    assert_absent(SENTINEL, str(error.value), where="wrapped exception")
    # The chain is broken too: __cause__ would carry the original text.
    assert error.value.__cause__ is None
    assert_absent(SENTINEL, repr(error.value.__traceback__ and ""),
                  where="traceback")


def test_the_provider_holds_no_attribute_containing_the_key(keyed):
    """The key may pass through the provider; it may not settle in it.

    Caching it on the instance is the natural optimisation and a real leak: a
    stored string outlives the request, survives a key rotation, and lands in
    any debugger, heap dump or object dump that touches the provider. The
    check walks the instance's own state rather than naming one attribute,
    because the attribute a future author adds will not be the one named here.
    """
    transport = RecordedTransport({AAPL_BARS_PATH: _bars_payload()})
    provider = _provider({})
    provider.transport = transport
    provider.get_historical_bars(AAPL.instrument_id, "30m",
                                 start="2026-01-15T00:00:00Z",
                                 end="2026-01-16T00:00:00Z")
    # The request definitely carried it, so this is not a vacuous check.
    assert SENTINEL in transport.calls[0]["headers"]["Authorization"]

    for name, value in vars(provider).items():
        if name == "transport":
            continue                    # the recorder deliberately keeps a log
        assert_absent(SENTINEL, value, where=f"provider attribute {name!r}")
    assert_absent(SENTINEL, vars(provider.credentials),
                  where="credential provider state")


def test_the_key_is_sent_in_the_header_and_nowhere_else(keyed):
    """It must reach the wire -- a provider that never sends it is useless."""
    transport = RecordedTransport({AAPL_BARS_PATH: _bars_payload()})
    provider = _provider({})
    provider.transport = transport
    provider.get_historical_bars(AAPL.instrument_id, "30m",
                                 start="2026-01-15T00:00:00Z",
                                 end="2026-01-16T00:00:00Z")
    call = transport.calls[0]
    assert SENTINEL in call["headers"]["Authorization"]
    assert_absent(SENTINEL, call["params"], where="query parameters")
    assert_absent(SENTINEL, call["path"], where="request path")


def test_the_key_never_appears_in_a_structured_log_record(keyed):
    from scripts.trading_lab.ops.structured_log import redact as log_redact

    provider = _provider({})
    record = {"event": "provider_configured", "provider": provider.payload(),
              "api_key": SENTINEL, "config": {"credential": SENTINEL}}
    scrubbed = log_redact(record)
    assert_absent(SENTINEL, scrubbed, where="structured log record")


def test_the_key_never_appears_in_a_support_bundle(keyed, tmp_path):
    from scripts.trading_lab.ops import support_bundle
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout

    layout = RuntimeLayout(tmp_path / "var" / "trading_lab").ensure()
    bundle = support_bundle.build(layout=layout, root=tmp_path)
    rendered = support_bundle.render(bundle)
    assert SENTINEL not in rendered
    assert MASSIVE_API_KEY_ENV not in rendered
    assert_absent(SENTINEL, bundle, where="support bundle")


def test_the_recursive_scrub_catches_a_key_that_never_went_through_secret():
    """The safety net, for the value someone assigned to a plain dict."""
    payload = {"api_key": SENTINEL, "nested": {"authorization": SENTINEL},
               "rows": [{"secret": SENTINEL}], "harmless": "kept"}
    scrubbed = redact(payload)
    assert_absent(SENTINEL, scrubbed, where="scrubbed payload")
    assert scrubbed["harmless"] == "kept"


def test_a_missing_credential_raises_instead_of_sending_an_empty_one(monkeypatch):
    monkeypatch.delenv(MASSIVE_API_KEY_ENV, raising=False)
    provider = _provider({})
    assert provider.payload()["credential_configured"] is False
    with pytest.raises(CredentialError):
        provider._fetch(AAPL_BARS_PATH, {})

    blank = MissingCredentialProvider()
    assert blank.available() is False
    with pytest.raises(CredentialError):
        blank.get()


def test_a_whitespace_only_credential_is_not_a_credential(monkeypatch):
    monkeypatch.setenv(MASSIVE_API_KEY_ENV, "   ")
    provider = EnvCredentialProvider(MASSIVE_API_KEY_ENV)
    assert provider.available() is False
    with pytest.raises(CredentialError):
        provider.get()


def test_the_credential_is_read_at_call_time_not_cached(monkeypatch):
    """A cached key survives a rotation and keeps sending the old one."""
    provider = EnvCredentialProvider(MASSIVE_API_KEY_ENV)
    monkeypatch.setenv(MASSIVE_API_KEY_ENV, SENTINEL)
    assert provider.get().reveal() == SENTINEL
    monkeypatch.setenv(MASSIVE_API_KEY_ENV, SENTINEL + "-rotated")
    assert provider.get().reveal() == SENTINEL + "-rotated"


def test_no_source_file_contains_anything_shaped_like_a_committed_key():
    """The last line of defence: a key pasted into the repo."""
    root = pathlib.Path(__file__).resolve().parents[2]
    for path in (root / "scripts" / "trading_lab").rglob("*.py"):
        text = path.read_text()
        assert SENTINEL not in text, path
        # The env var may be named; its value may never be assigned inline.
        assert f'{MASSIVE_API_KEY_ENV}"] = "' not in text, path
        assert f"{MASSIVE_API_KEY_ENV}'] = '" not in text, path


# --- reference metadata proves the venue -----------------------------------


def test_every_seed_instrument_matches_its_recorded_vendor_metadata(
        reference_fixture):
    """The registry is asserted against the fixture, never derived at runtime.

    A registry that read this file at import would be a registry a replaced
    file could extend. A test that reads it makes a wrong venue a build
    failure instead.
    """
    assert len(EQUITY_INSTRUMENTS_V1) == 4
    for spec in EQUITY_INSTRUMENTS_V1.all():
        symbol = spec.instrument_id.symbol
        row = reference_fixture[symbol]["results"]
        parsed = parse_reference_ticker(row)
        assert parsed.venue == spec.instrument_id.venue
        assert parsed.asset_class == spec.asset_class
        assert parsed.currency == spec.price_currency
        assert parsed.display_name == spec.display_name


def test_the_etf_is_registered_as_an_etf_and_the_shares_as_equities():
    classes = {spec.instrument_id.symbol: spec.asset_class
               for spec in EQUITY_INSTRUMENTS_V1.all()}
    assert classes == {"AAPL": "EQUITY", "MSFT": "EQUITY",
                       "NVDA": "EQUITY", "QQQ": "ETF"}


def test_an_unknown_exchange_or_security_type_is_refused_not_defaulted():
    """A guessed venue names a different instrument with a plausible id."""
    with pytest.raises(MassiveProviderError) as error:
        parse_reference_ticker({"ticker": "XYZ", "primary_exchange": "XLON",
                                "type": "CS"})
    assert "will not guess" in str(error.value)
    with pytest.raises(MassiveProviderError):
        parse_reference_ticker({"ticker": "XYZ", "primary_exchange": "XNAS",
                                "type": "WARRANT"})
    with pytest.raises(MassiveProviderError):
        parse_reference_ticker({"ticker": "XYZ", "primary_exchange": "",
                                "type": "CS"})
    with pytest.raises(MassiveProviderError):
        parse_reference_ticker({"ticker": "", "primary_exchange": "XNAS",
                                "type": "CS"})


def test_a_non_usd_listing_is_refused_rather_than_converted():
    with pytest.raises(MassiveProviderError) as error:
        parse_reference_ticker({"ticker": "SAP", "primary_exchange": "XNYS",
                                "type": "CS", "currency_name": "EUR"})
    assert "USD only" in str(error.value)


def test_the_provider_is_not_the_venue():
    """massive:AAPL is not a thing. xnas:AAPL served by Massive is."""
    for spec in EQUITY_INSTRUMENTS_V1.all():
        assert spec.instrument_id.venue == "xnas"
        assert "massive" not in spec.instrument_id.canonical
        assert spec.canonical_id.startswith("xnas:")
    provider = PROVIDERS_V1.resolve(MASSIVE_STOCKS_HISTORICAL_V1)
    assert provider.provider_id not in {spec.instrument_id.venue
                                        for spec in EQUITY_INSTRUMENTS_V1.all()}


# --- bars ------------------------------------------------------------------


def test_bars_carry_their_instrument_provider_and_policy(keyed):
    provider = _provider({AAPL_BARS_PATH: _bars_payload()})
    bars = provider.get_historical_bars(AAPL.instrument_id, "30m",
                                        start="2026-01-15T00:00:00Z",
                                        end="2026-01-16T00:00:00Z")
    assert len(bars) == 2
    for bar in bars:
        assert bar.instrument_id == AAPL.instrument_id
        assert bar.provider_id == MASSIVE_STOCKS_HISTORICAL_V1
        assert bar.adjustment_policy == ADJUSTMENT_RAW
        assert bar.timeframe == "30m"
        assert isinstance(bar.close, Decimal)
        bar.require_instrument(AAPL.instrument_id)
    with pytest.raises(EquityMarketError):
        bars[0].require_instrument("xnas:MSFT")


def test_a_bar_is_available_only_at_its_close(keyed):
    provider = _provider({AAPL_BARS_PATH: _bars_payload()})
    bar = provider.get_historical_bars(AAPL.instrument_id, "30m",
                                       start="2026-01-15T00:00:00Z",
                                       end="2026-01-16T00:00:00Z")[0]
    assert bar.available_at == bar.bar_close_at
    assert bar.available_at > bar.bar_open_at


def test_a_response_that_declares_a_different_policy_is_refused(keyed):
    """The silent mismatch: asked for RAW, handed adjusted, labelled RAW."""
    provider = _provider({AAPL_BARS_PATH: _bars_payload(adjusted=True)})
    with pytest.raises(EquityMarketError) as error:
        provider.get_historical_bars(AAPL.instrument_id, "30m",
                                     start="2026-01-15T00:00:00Z",
                                     end="2026-01-16T00:00:00Z",
                                     adjustment_policy=ADJUSTMENT_RAW)
    assert "refusing to relabel" in str(error.value)


def test_an_instrument_the_provider_does_not_serve_is_refused(keyed):
    from scripts.trading_lab.market_providers import MarketProviderError

    provider = _provider({AAPL_BARS_PATH: _bars_payload()})
    with pytest.raises(MarketProviderError):
        provider.get_historical_bars("coinbase:BTC-USD", "30m",
                                     start="2026-01-15T00:00:00Z",
                                     end="2026-01-16T00:00:00Z")


def test_a_malformed_bar_is_refused_rather_than_stored(keyed):
    # high below low: structurally impossible, and it would sail through any
    # check that only looked at the close.
    broken = [{"t": BAR_ONE_MS, "o": 185.10, "h": 180.00, "l": 184.90,
               "c": 182.00, "v": 1}]
    provider = _provider({AAPL_BARS_PATH: _bars_payload(rows=broken)})
    with pytest.raises(EquityMarketError):
        provider.get_historical_bars(AAPL.instrument_id, "30m",
                                     start="2026-01-15T00:00:00Z",
                                     end="2026-01-16T00:00:00Z")


def test_a_non_numeric_vendor_value_is_refused(keyed):
    """A null or a string where a price belongs is a shape mismatch."""
    rows = [{"t": BAR_ONE_MS, "o": None, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}]
    provider = _provider({AAPL_BARS_PATH: _bars_payload(rows=rows)})
    with pytest.raises(MassiveProviderError):
        provider.get_historical_bars(AAPL.instrument_id, "30m",
                                     start="2026-01-15", end="2026-01-16")


def test_a_daily_bar_is_refused_without_a_calendar(keyed):
    """Its close is the session close, which this provider cannot know."""
    provider = _provider({})
    with pytest.raises(UnsupportedCapabilityError) as error:
        provider.get_historical_bars(AAPL.instrument_id, "1d",
                                     start="2026-01-15", end="2026-01-16")
    assert "session-aware" in str(error.value)


# --- adjustment policy identity -------------------------------------------


def test_raw_and_adjusted_bars_have_different_hashes():
    """Otherwise the two series are indistinguishable once stored."""
    common = dict(instrument=AAPL.instrument_id, timeframe="30m",
                  provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
                  bar_open_at="2026-01-15T14:30:00Z",
                  bar_close_at="2026-01-15T15:00:00Z", open="185.10",
                  high="186.40", low="184.90", close="186.00",
                  volume="1250000")
    raw = build_equity_bar(adjustment_policy=ADJUSTMENT_RAW, **common)
    adjusted = build_equity_bar(adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED,
                                **common)
    assert raw.bar_hash != adjusted.bar_hash
    assert raw.close == adjusted.close      # identical numbers, different meaning


def test_a_series_that_mixes_policies_is_refused():
    """Half raw and half adjusted shows a fake return at the split."""
    common = dict(instrument=AAPL.instrument_id, timeframe="30m",
                  provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
                  bar_close_at="2026-01-15T15:00:00Z", open="185.10",
                  high="186.40", low="184.90", close="186.00",
                  volume="1250000")
    mixed = [build_equity_bar(adjustment_policy=ADJUSTMENT_RAW,
                              bar_open_at="2026-01-15T14:30:00Z", **common),
             build_equity_bar(adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED,
                              bar_open_at="2026-01-15T14:30:00Z", **common)]
    with pytest.raises(EquityMarketError) as error:
        require_single_policy(mixed)
    assert "one adjustment policy" in str(error.value)


def test_total_return_is_refused_rather_than_silently_split_adjusted():
    with pytest.raises(EquityMarketError) as error:
        build_equity_bar(instrument=AAPL.instrument_id, timeframe="30m",
                         provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
                         adjustment_policy=ADJUSTMENT_TOTAL_RETURN,
                         bar_open_at="2026-01-15T14:30:00Z",
                         bar_close_at="2026-01-15T15:00:00Z", open="1",
                         high="1", low="1", close="1", volume="1")
    assert "not implemented" in str(error.value)


# --- corporate actions -----------------------------------------------------


def _raw_bar(session_date: str, price: str, volume: str = "1000"):
    return build_equity_bar(
        instrument=AAPL.instrument_id, timeframe="1d",
        provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
        adjustment_policy=ADJUSTMENT_RAW,
        bar_open_at=f"{session_date}T14:30:00Z",
        bar_close_at=f"{session_date}T21:00:00Z",
        open=price, high=price, low=price, close=price, volume=volume,
        session_date=session_date)


def test_a_split_restates_prices_before_it_and_leaves_later_ones_alone():
    split = StockSplit(instrument_id=AAPL.instrument_id,
                       effective_date="2026-06-10", ratio_numerator=4,
                       ratio_denominator=1)
    bars = [_raw_bar("2026-06-09", "400.00"), _raw_bar("2026-06-10", "100.00")]
    adjusted = apply_splits(bars, [split])

    # Compared numerically: Decimal keeps the dividend's scale, so an exact
    # 400.00 / 4 is Decimal("100.00"), and asserting on the string would be
    # testing the formatting rather than the arithmetic.
    assert [bar.close for bar in adjusted] == [Decimal(100), Decimal(100)]
    assert adjusted[0].close == adjusted[0].open == adjusted[0].high
    assert all(bar.adjustment_policy == ADJUSTMENT_SPLIT_ADJUSTED
               for bar in adjusted)
    # Volume moves the other way: four times as many shares.
    assert adjusted[0].volume == Decimal("4000")
    assert adjusted[1].volume == Decimal("1000")


def test_the_effective_date_boundary_is_not_off_by_one():
    """The bar on the effective date is already adjusted; the one before is not."""
    split = StockSplit(instrument_id=AAPL.instrument_id,
                       effective_date="2026-06-10", ratio_numerator=4,
                       ratio_denominator=1)
    assert split.applies_to("2026-06-09") is True
    assert split.applies_to("2026-06-10") is False
    assert split.applies_to("2026-06-11") is False


def test_a_non_integer_ratio_stays_exact():
    """A 3-for-2 is 1.5; a 7-for-3 is not representable and must not be rounded."""
    split = StockSplit(instrument_id=AAPL.instrument_id,
                       effective_date="2026-06-10", ratio_numerator=7,
                       ratio_denominator=3)
    assert split.label == "7-for-3"
    restated = split.adjust_price("210.00")
    assert restated * split.ratio == Decimal("210.00")


def test_adjusting_an_already_adjusted_series_is_refused():
    """Dividing twice produces a smooth, completely wrong history."""
    split = StockSplit(instrument_id=AAPL.instrument_id,
                       effective_date="2026-06-10", ratio_numerator=4,
                       ratio_denominator=1)
    once = apply_splits([_raw_bar("2026-06-09", "400.00")], [split])
    with pytest.raises(EquityMarketError) as error:
        apply_splits(once, [split])
    assert "already" in str(error.value)


def test_a_split_from_another_instrument_cannot_adjust_these_bars():
    from scripts.trading_lab.instruments import InstrumentId

    split = StockSplit(instrument_id=InstrumentId(venue="xnas", symbol="MSFT"),
                       effective_date="2026-06-10", ratio_numerator=2,
                       ratio_denominator=1)
    with pytest.raises(EquityMarketError):
        apply_splits([_raw_bar("2026-06-09", "400.00")], [split])


def test_two_splits_compound_rather_than_replacing_each_other():
    splits = [
        StockSplit(instrument_id=AAPL.instrument_id, effective_date="2026-03-10",
                   ratio_numerator=2, ratio_denominator=1),
        StockSplit(instrument_id=AAPL.instrument_id, effective_date="2026-06-10",
                   ratio_numerator=4, ratio_denominator=1),
    ]
    before_both = apply_splits([_raw_bar("2026-01-05", "800.00")], splits)[0]
    between = apply_splits([_raw_bar("2026-04-05", "400.00")], splits)[0]
    assert before_both.close == Decimal("100")
    assert between.close == Decimal("100")


def test_a_dividend_is_recorded_and_explicitly_not_applied():
    dividend = CashDividend(instrument_id=AAPL.instrument_id,
                            ex_date="2026-02-06", amount="0.25")
    payload = dividend.payload()
    assert payload["applied_to_prices"] is False
    assert payload["amount"] == "0.25"
    with pytest.raises(EquityMarketError):
        CashDividend(instrument_id=AAPL.instrument_id, ex_date="2026-02-06",
                     amount="-1")


def test_splits_come_back_bound_to_their_instrument(keyed):
    provider = _provider({
        "/stocks/v1/splits": {"status": "OK", "results": [
            {"ticker": "AAPL", "execution_date": "2026-06-10",
             "split_from": 1, "split_to": 4}]},
    })
    splits = provider.get_splits(AAPL.instrument_id, start="2026-01-01",
                                end="2026-12-31")
    assert splits[0].instrument_id == AAPL.instrument_id
    assert splits[0].ratio == Decimal(4)
    assert splits[0].label == "4-for-1"
    assert splits[0].source == MASSIVE_STOCKS_HISTORICAL_V1


def test_a_split_read_backwards_would_invert_every_adjustment(keyed):
    """split_from/split_to are shares before and after, in that order.

    Reading them the wrong way round turns a 4-for-1 into a 1-for-4 and
    inverts every adjusted price -- a 16x error dressed as a 4x one.
    """
    provider = _provider({
        "/stocks/v1/splits": {"status": "OK", "results": [
            {"ticker": "AAPL", "execution_date": "2026-06-10",
             "split_from": 2, "split_to": 3}]},
    })
    split = provider.get_splits(AAPL.instrument_id, start="2026-01-01",
                               end="2026-12-31")[0]
    assert split.ratio_numerator == 3 and split.ratio_denominator == 2
    assert split.ratio == Decimal("1.5")


def test_dividends_are_refused_because_no_endpoint_is_documented(keyed):
    """Corpus V1 is split-adjusted and claims no total return."""
    provider = _provider({})
    with pytest.raises(UnsupportedCapabilityError) as error:
        provider.get_dividends(AAPL.instrument_id, start="2026-01-01",
                               end="2026-12-31")
    assert "total return" in str(error.value)


# --- corpus spec -----------------------------------------------------------


def test_the_corpus_spec_is_defined_and_nothing_is_captured():
    from scripts.trading_lab.equity_calendar import US_EQUITY_REGULAR_SPEC

    spec = EquityCorpusSpec(
        instruments=("xnas:AAPL", "xnas:MSFT"), timeframe="30m",
        adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED,
        provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
        calendar_id="US_EQUITY_REGULAR",
        calendar_spec_hash=US_EQUITY_REGULAR_SPEC.spec_hash,
        start="2024-01-01T00:00:00Z", end="2026-01-01T00:00:00Z")
    payload = spec.payload()
    assert payload["captured"] is False
    assert len(spec.corpus_spec_hash) == 64


def test_two_corpora_differing_only_in_policy_are_different_corpora():
    """The whole reason the policy is in the hash."""
    from scripts.trading_lab.equity_calendar import US_EQUITY_REGULAR_SPEC

    common = dict(instruments=("xnas:AAPL",), timeframe="30m",
                  provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
                  calendar_id="US_EQUITY_REGULAR",
                  calendar_spec_hash=US_EQUITY_REGULAR_SPEC.spec_hash,
                  start="2024-01-01T00:00:00Z", end="2026-01-01T00:00:00Z")
    raw = EquityCorpusSpec(adjustment_policy=ADJUSTMENT_RAW, **common)
    adjusted = EquityCorpusSpec(adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED,
                                **common)
    assert raw.corpus_spec_hash != adjusted.corpus_spec_hash


def test_a_corpus_under_a_different_calendar_version_is_a_different_corpus():
    from scripts.trading_lab.equity_calendar import USEquityRegularCalendarSpec

    common = dict(instruments=("xnas:AAPL",), timeframe="30m",
                  adjustment_policy=ADJUSTMENT_RAW,
                  provider_id=MASSIVE_STOCKS_HISTORICAL_V1,
                  calendar_id="US_EQUITY_REGULAR",
                  start="2024-01-01T00:00:00Z", end="2026-01-01T00:00:00Z")
    pinned = EquityCorpusSpec(
        calendar_spec_hash=USEquityRegularCalendarSpec().spec_hash, **common)
    moved = EquityCorpusSpec(
        calendar_spec_hash=USEquityRegularCalendarSpec(
            calendar_provider_version="9.9.9").spec_hash, **common)
    assert pinned.corpus_spec_hash != moved.corpus_spec_hash


# --- the tradable boundary -------------------------------------------------


def test_no_equity_is_tradable():
    """Describable, not tradable. The distinction the registries exist for."""
    from scripts.trading_lab.instrument_registry import (
        CATALOGUE_V1, INSTRUMENTS_V1, is_tradable)

    for spec in EQUITY_INSTRUMENTS_V1.all():
        assert is_tradable(spec.canonical_id) is False
        assert spec.canonical_id in CATALOGUE_V1.ids()
        assert spec.canonical_id not in INSTRUMENTS_V1.ids()
    for product in ("BTC-USD", "ETH-USD"):
        assert is_tradable(product) is True
    assert is_tradable("xnas:TSLA") is False
    assert is_tradable("") is False
    assert is_tradable(None) is False


def test_the_shared_portfolio_universe_is_still_only_crypto():
    """An equity here would mean a batch that never completes and never says so."""
    from scripts.trading_lab.app_api.service import AppService
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    universe = AppService._portfolio_instruments(AppService.__new__(AppService))
    assert universe == list(INSTRUMENTS_V1.ids())
    assert all(item.startswith("coinbase:") for item in universe)


def test_the_legacy_product_bridge_is_untouched_by_the_new_instruments():
    from scripts.trading_lab.app_api.contracts import SUPPORTED_PRODUCTS
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    assert INSTRUMENTS_V1.legacy_ids() == ("BTC-USD", "ETH-USD")
    assert sorted(SUPPORTED_PRODUCTS) == ["BTC-USD", "ETH-USD"]


def test_the_frozen_crypto_instrument_hashes_are_unchanged():
    """Phase 6D adds markets; it does not restate the ones already recorded."""
    from scripts.trading_lab.instrument_registry import BTC_USD, ETH_USD

    assert BTC_USD.instrument_spec_hash == (
        "492c167c1e66a37a377cff8b4e135841c5a13a7c60324ec9b5c8b1976bf5701f")
    assert ETH_USD.instrument_spec_hash == (
        "2a9e1d1c922fbb9af68ad92f8f2638e951a2fef830afb6e514a29ebfab35d7f3")


# --- the vendor's real request and response shapes (Phase 6E) --------------
#
# These pin the adapter written on first contact. Every fixture below is
# derived from the documented endpoint shapes and sanitized: public ticker
# metadata and price rows only, no credential, no request identifiers.


def test_the_bars_path_carries_the_ticker_and_the_window():
    from scripts.trading_lab.massive_provider import bars_path

    assert bars_path("AAPL", multiplier=30, timespan="minute",
                     start="2024-08-01", end="2026-07-31") == (
        "/v2/aggs/ticker/AAPL/range/30/minute/2024-08-01/2026-07-31")


def test_only_the_three_documented_endpoints_are_reachable(keyed):
    """Still an allowlist. A pattern is not a licence to wander."""
    from scripts.trading_lab.massive_provider import (
        ALLOWED_PATHS, bars_path, reference_path)

    provider = _provider({})
    assert len(ALLOWED_PATHS) == 3
    for good in (bars_path("AAPL", multiplier=30, timespan="minute",
                           start="2024-08-01", end="2024-08-02"),
                 reference_path("QQQ"), "/stocks/v1/splits"):
        assert provider._require_allowed(good) == good
    for bad in ("/v2/aggs/ticker/AAPL/../../v1/account",
                "/v2/aggs/ticker/AAPL/range/30/minute/2024-08-01",
                "/v3/reference/tickers/AAPL/financials",
                "/v2/snapshot/locale/us/markets/stocks/tickers",
                "/stocks/v1/splits/../orders",
                # lower case is not the documented ticker form
                "/v3/reference/tickers/aapl"):
        with pytest.raises(MassiveProviderError):
            provider._require_allowed(bad)


def test_a_thirty_minute_timeframe_maps_to_the_documented_window():
    from scripts.trading_lab.massive_provider import (
        MassiveProviderError as Error, require_aggregate_window)

    assert require_aggregate_window("30m") == (30, "minute")
    assert require_aggregate_window("1d") == (1, "day")
    for unmapped in ("5m", "2h", "1w", ""):
        with pytest.raises(Error):
            require_aggregate_window(unmapped)


def test_the_adjusted_flag_means_splits_and_never_dividends():
    """The vendor's flag is a boolean; HyprL's policy is a name.

    ``adjusted=true`` restates for splits only. Mapping TOTAL_RETURN onto it
    would silently claim dividends were applied when they were not.
    """
    from scripts.trading_lab.massive_provider import ADJUSTED_FLAG

    assert ADJUSTED_FLAG[ADJUSTMENT_SPLIT_ADJUSTED] == "true"
    assert ADJUSTED_FLAG[ADJUSTMENT_RAW] == "false"
    assert ADJUSTMENT_TOTAL_RETURN not in ADJUSTED_FLAG


def test_the_millisecond_timestamp_is_read_as_the_window_start():
    """Treating it as the end would shift every bar by one interval."""
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    rows = adapt_aggregate_rows(
        _bars_payload(), instrument_id="xnas:AAPL",
        requested_policy=ADJUSTMENT_RAW)
    assert rows[0]["bar_open_at"] == datetime(2026, 1, 15, 14, 30,
                                              tzinfo=timezone.utc)
    assert rows[1]["bar_open_at"] == datetime(2026, 1, 15, 15, 0,
                                              tzinfo=timezone.utc)


def test_a_timestamp_that_is_not_whole_milliseconds_is_refused():
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    for bad in ("1768487400000", 1768487400000.5, None, True):
        rows = [{"t": bad, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}]
        with pytest.raises(MassiveProviderError):
            adapt_aggregate_rows(_bars_payload(rows=rows),
                                 instrument_id="xnas:AAPL",
                                 requested_policy=ADJUSTMENT_RAW)


def test_a_response_about_another_ticker_is_refused():
    """The vendor echoes the ticker, so a mix-up is detectable rather than fatal."""
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    with pytest.raises(MassiveProviderError) as error:
        adapt_aggregate_rows(_bars_payload(ticker="MSFT"),
                             instrument_id="xnas:AAPL",
                             requested_policy=ADJUSTMENT_RAW)
    assert "under another's name" in str(error.value)


def test_an_error_status_is_refused_before_any_price_is_read():
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    payload = {**_bars_payload(), "status": "ERROR"}
    with pytest.raises(MassiveProviderError) as error:
        adapt_aggregate_rows(payload, instrument_id="xnas:AAPL",
                             requested_policy=ADJUSTMENT_RAW)
    assert "status" in str(error.value)


def test_an_empty_window_is_not_a_schema_mismatch():
    """No results plus a stated count of zero is a legitimate empty answer."""
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    assert adapt_aggregate_rows(
        {"ticker": "AAPL", "status": "OK", "resultsCount": 0, "adjusted": True},
        instrument_id="xnas:AAPL",
        requested_policy=ADJUSTMENT_SPLIT_ADJUSTED) == []


def test_a_shape_with_neither_results_nor_a_count_is_refused():
    from scripts.trading_lab.massive_provider import adapt_aggregate_rows

    with pytest.raises(MassiveProviderError) as error:
        adapt_aggregate_rows({"bars": []}, instrument_id="xnas:AAPL",
                             requested_policy=ADJUSTMENT_RAW)
    assert "does not match this capture's contract" in str(error.value)


def test_the_v3_reference_shape_is_read_as_an_object(reference_fixture):
    """The single-ticker endpoint returns one object, not a list."""
    from scripts.trading_lab.massive_provider import parse_reference_ticker

    parsed = parse_reference_ticker(reference_fixture["AAPL"]["results"])
    assert parsed.symbol == "AAPL"
    assert parsed.venue == "xnas"
    assert parsed.asset_class == "EQUITY"
    assert parsed.currency == "USD"
