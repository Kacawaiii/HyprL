"""The credential-free daily source, and the boundary between V1 and V2.

Every test runs offline against the sanitized fixture. The provider has no
transport unless one is injected, so there is nothing to intercept and no key
to leak -- this source has none, which is the entire reason it exists.

Two things these tests are really protecting.

The first is that V2 can never be mistaken for V1 or quietly substituted for
it. Different provider, different timeframe, different adjustment policy,
different hash. V1 stays frozen and untouched, still waiting on a Massive key.

The second is that "no credential" does not mean "no discipline". The same
fail-closed rules apply: the ticker is checked against the response, the
exchange timezone is checked against the calendar, a column-oriented payload
whose columns disagree is refused rather than zipped, and a hole in the data
is left as a hole so the gap audit reports it.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pytest

pytest.importorskip(
    "pandas_market_calendars",
    reason="the equity calendar needs the optional [equities] extra")

from scripts.trading_lab.equity_corpus import (  # noqa: E402
    CORPUS_SPEC_V1, CORPUS_SPEC_V2)
from scripts.trading_lab.equity_market import (  # noqa: E402
    ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED, EquityMarketError)
from scripts.trading_lab.market_providers import (  # noqa: E402
    UnsupportedCapabilityError)
from scripts.trading_lab.yahoo_chart_provider import (  # noqa: E402
    ALLOWED_PATHS, NoNetworkYahooTransport, YAHOO_CHART_DAILY_V1,
    YahooChartDailyProvider, YahooProviderError, adapt_chart_rows,
    adapt_split_events, chart_path, parse_chart_meta, require_interval)

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "crypto"


@pytest.fixture(scope="module")
def chart() -> dict:
    return json.loads((FIXTURES / "yahoo_chart_daily_aapl.json").read_text())


@pytest.fixture
def provider() -> YahooChartDailyProvider:
    return YahooChartDailyProvider(instruments=CORPUS_SPEC_V2.instruments)


# --- V2 is a different corpus, not a variant of V1 -------------------------


def test_v1_stays_frozen_and_untouched():
    """The whole point of a second corpus is that the first is unchanged."""
    assert CORPUS_SPEC_V1.corpus_spec_hash == (
        "93cfdb1a749bfa1de5c69c5dced2908413cfb8bba7f3f663fd9c747151b1b5ed")
    assert CORPUS_SPEC_V1.timeframe == "30m"
    assert CORPUS_SPEC_V1.adjustment_policy == ADJUSTMENT_SPLIT_ADJUSTED
    assert CORPUS_SPEC_V1.provider_id == "massive-stocks-historical-v1"


def test_v2_cannot_be_mistaken_for_v1():
    """Different provider, timeframe and policy must mean a different hash."""
    assert CORPUS_SPEC_V2.corpus_spec_hash != CORPUS_SPEC_V1.corpus_spec_hash
    assert CORPUS_SPEC_V2.timeframe == "1d"
    assert CORPUS_SPEC_V2.adjustment_policy == ADJUSTMENT_RAW
    assert CORPUS_SPEC_V2.provider_id == YAHOO_CHART_DAILY_V1
    assert CORPUS_SPEC_V2.corpus_id != CORPUS_SPEC_V1.corpus_id


def test_v2_shares_exactly_what_makes_the_two_comparable():
    """Same instruments, same calendar, same range -- so gaps can be read across."""
    assert CORPUS_SPEC_V2.instruments == CORPUS_SPEC_V1.instruments
    assert CORPUS_SPEC_V2.requested_start == CORPUS_SPEC_V1.requested_start
    assert CORPUS_SPEC_V2.requested_end == CORPUS_SPEC_V1.requested_end
    assert (CORPUS_SPEC_V2.calendar_identity()["calendar_spec_hash"]
            == CORPUS_SPEC_V1.calendar_identity()["calendar_spec_hash"]
            == "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314")


def test_a_daily_corpus_expects_one_bar_per_session():
    """Phase 6D's rule: for an equity, a daily bar is the session."""
    assert len(CORPUS_SPEC_V2.sessions()) == 501
    assert len(CORPUS_SPEC_V2.expected_bar_opens()) == 501
    assert len(CORPUS_SPEC_V2.expected_bar_opens()) * 4 == 2004


# --- the source's real limits ---------------------------------------------


def test_intraday_is_refused_because_the_source_refuses_it():
    """HTTP 422 over a multi-year range. Offering 30m would promise a corpus
    that cannot be captured, and V1 remains the spec for that."""
    assert require_interval("1d") == "1d"
    for unavailable in ("30m", "1h", "5m", "1w"):
        with pytest.raises(YahooProviderError) as error:
            require_interval(unavailable)
        assert "422" in str(error.value) or "multi-year" in str(error.value)


def test_the_provider_can_only_claim_raw_prices():
    """The payload carries unadjusted OHLC plus a separate adjusted close."""
    with pytest.raises(EquityMarketError) as error:
        YahooChartDailyProvider(adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED)
    assert "relabel" in str(error.value)


def test_the_provider_declares_no_credential_and_no_account(provider):
    assert provider.capabilities.authenticated is False
    assert provider.capabilities.private_account_data is False
    payload = provider.payload()
    assert payload["credential_required"] is False
    assert payload["brokerage_endpoints"] == []
    # And it is honest about what it is not.
    assert payload["official_contract"] is False
    assert payload["redistribution_permitted"] is False
    rendered = json.dumps(payload).lower()
    for banned in ("api_key", "apikey", "authorization", "bearer", "token"):
        assert banned not in rendered


def test_no_credential_is_ever_sent_because_none_exists(provider):
    headers = provider._headers()
    assert set(headers) == {"Accept"}


# --- offline by construction ----------------------------------------------


def test_a_provider_without_a_transport_cannot_reach_the_network(provider):
    assert isinstance(provider.transport, NoNetworkYahooTransport)
    assert provider.payload()["network_enabled"] is False
    with pytest.raises(YahooProviderError):
        provider.transport.request(chart_path("AAPL"), {}, {})


def test_only_the_one_documented_path_is_reachable(provider):
    assert len(ALLOWED_PATHS) == 1
    assert provider._require_allowed(chart_path("AAPL")) == "/v8/finance/chart/AAPL"
    for bad in ("/v8/finance/chart/AAPL/../../v1/account",
                "/v8/finance/chart/aapl",
                "/v7/finance/quote?symbols=AAPL",
                "/v8/finance/chart/",
                "/v1/account/balance"):
        with pytest.raises(YahooProviderError):
            provider._require_allowed(bad)


def test_bars_go_through_the_calendar_aware_runner_not_the_provider(provider):
    """A daily bar closes at the session close, which the provider cannot know."""
    with pytest.raises(UnsupportedCapabilityError) as error:
        provider.get_historical_bars("xnas:AAPL", "1d", start="2024-08-01",
                                     end="2024-08-02")
    assert "calendar-aware" in str(error.value)


# --- the response shape ---------------------------------------------------


def test_the_identity_block_is_read_before_any_price(chart):
    meta = parse_chart_meta(chart)
    assert meta.symbol == "AAPL"
    assert meta.currency == "USD"
    assert meta.exchange_timezone == "America/New_York"


def test_a_response_about_another_ticker_is_refused(chart):
    other = json.loads(json.dumps(chart))
    other["chart"]["result"][0]["meta"]["symbol"] = "MSFT"
    with pytest.raises(YahooProviderError) as error:
        adapt_chart_rows(other, instrument_id="xnas:AAPL")
    assert "under another's name" in str(error.value)


def test_a_non_usd_or_foreign_exchange_listing_is_refused(chart):
    for field, value in (("currency", "EUR"),
                         ("exchangeTimezoneName", "Europe/Paris")):
        other = json.loads(json.dumps(chart))
        other["chart"]["result"][0]["meta"][field] = value
        with pytest.raises(YahooProviderError):
            parse_chart_meta(other)


def test_rows_land_exactly_on_the_calendar_grid(chart):
    """The fixture's timestamps come from the calendar, so this checks the
    adapter reads them back unchanged rather than checking the fixture."""
    rows = adapt_chart_rows(chart, instrument_id="xnas:AAPL")
    expected = {session.open_at for session in
                CORPUS_SPEC_V2.calendar().sessions_between(
                    "2024-11-25T00:00:00Z", "2024-12-02T23:59:59Z")}
    assert {row["bar_open_at"] for row in rows} == expected
    assert all(row["bar_open_at"].tzinfo is timezone.utc for row in rows)


def test_a_column_oriented_response_whose_columns_disagree_is_refused(chart):
    """Zipping would silently truncate to the shortest and look plausible."""
    broken = json.loads(json.dumps(chart))
    broken["chart"]["result"][0]["indicators"]["quote"][0]["close"].pop()
    with pytest.raises(YahooProviderError) as error:
        adapt_chart_rows(broken, instrument_id="xnas:AAPL")
    assert "corrupt" in str(error.value)


def test_a_hole_in_the_source_is_left_as_a_hole(chart):
    """Dropped, never filled -- so the gap audit reports it as missing."""
    holed = json.loads(json.dumps(chart))
    holed["chart"]["result"][0]["indicators"]["quote"][0]["close"][2] = None
    rows = adapt_chart_rows(holed, instrument_id="xnas:AAPL")
    assert len(rows) == len(chart["chart"]["result"][0]["timestamp"]) - 1


def test_a_missing_quote_series_is_a_schema_mismatch(chart):
    broken = json.loads(json.dumps(chart))
    del broken["chart"]["result"][0]["indicators"]["quote"]
    with pytest.raises(YahooProviderError) as error:
        adapt_chart_rows(broken, instrument_id="xnas:AAPL")
    assert "does not match this capture's contract" in str(error.value)


def test_a_reported_error_is_refused_before_anything_is_read(chart):
    broken = json.loads(json.dumps(chart))
    broken["chart"]["error"] = {"code": "Not Found"}
    with pytest.raises(YahooProviderError) as error:
        adapt_chart_rows(broken, instrument_id="xnas:AAPL")
    assert "reports an error" in str(error.value)


def test_prices_survive_as_exact_text(chart):
    """A rebuild from the stored bytes must land on the identical string."""
    rows = adapt_chart_rows(chart, instrument_id="xnas:AAPL")
    assert rows[0]["open"] == "100.0"
    assert rows[0]["volume"] == "1250000"
    assert all(isinstance(row[name], str)
               for row in rows for name in ("open", "high", "low", "close"))


# --- corporate actions ----------------------------------------------------


def test_split_events_are_read_for_provenance_only(chart):
    splits = adapt_split_events(chart, instrument_id="xnas:AAPL")
    assert len(splits) == 1
    assert splits[0]["ratio_numerator"] == 4
    assert splits[0]["ratio_denominator"] == 1
    assert splits[0]["effective_date"] == "2024-11-27"


def test_a_split_read_backwards_would_invert_every_adjustment(chart):
    """numerator/denominator are new-per-old: a 4-for-1 is 4/1, not 1/4."""
    from decimal import Decimal

    splits = adapt_split_events(chart, instrument_id="xnas:AAPL")
    ratio = Decimal(splits[0]["ratio_numerator"]) / Decimal(
        splits[0]["ratio_denominator"])
    assert ratio == Decimal(4)


def test_a_malformed_split_event_is_refused(chart):
    broken = json.loads(json.dumps(chart))
    event = next(iter(broken["chart"]["result"][0]["events"]["splits"].values()))
    del event["numerator"]
    with pytest.raises(YahooProviderError):
        adapt_split_events(broken, instrument_id="xnas:AAPL")


def test_no_splits_is_not_an_error(chart):
    """Most instruments have none in any given window."""
    quiet = json.loads(json.dumps(chart))
    quiet["chart"]["result"][0]["events"] = {}
    assert adapt_split_events(quiet, instrument_id="xnas:AAPL") == []


# --- non-finite values reaching the canonical boundary (6E-V2-FIX1) -------
#
# The adapter deliberately does NOT reject these. Its job is to record what
# arrived, so a raw payload carrying Infinity stays faithful provenance. The
# rejection belongs to the shared canonical validation, and these tests prove
# the value cannot survive that far.


def _payload_with(chart: dict, field: str, value):
    broken = json.loads(json.dumps(chart))
    series = broken["chart"]["result"][0]["indicators"]["quote"][0][field]
    series[0] = value
    return broken


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize("value,label", [(float("inf"), "inf"),
                                         (float("-inf"), "-inf"),
                                         (float("nan"), "nan")])
def test_a_non_finite_quote_never_reaches_a_canonical_bar(chart, field, value,
                                                          label):
    """Yahoo sends JSON numbers, so a non-finite arrives as a float."""
    from scripts.trading_lab.equity_corpus import (
        CORPUS_SPEC_V2, EquityCorpusError, build_canonical_bar)

    rows = adapt_chart_rows(_payload_with(chart, field, value),
                            instrument_id="xnas:AAPL")
    # The adapter records it faithfully -- that is provenance, not acceptance.
    assert rows, "the adapter dropped the row instead of recording it"

    calendar = CORPUS_SPEC_V2.calendar()
    session = calendar.sessions_between("2024-11-25T00:00:00Z",
                                        "2024-11-25T23:59:59Z")[0]
    opening = calendar.expected_bar_opens(session, "1d")[0]
    row = {name: rows[0][name] for name in
           ("open", "high", "low", "close", "volume")}
    with pytest.raises(EquityCorpusError) as error:
        build_canonical_bar(
            spec=CORPUS_SPEC_V2, instrument_id="xnas:AAPL", session=session,
            bar_open_at=opening, row=row, source_raw_hash="a" * 64,
            source_record_identity="b" * 64)
    assert "finite" in str(error.value)


def test_a_null_still_drops_the_row_and_becomes_a_reported_gap(chart):
    """Unchanged behaviour: a hole stays a hole, so the gap audit sees it.

    Deliberately different from a non-finite value. A null is the source
    saying it has nothing; Infinity is the source saying something impossible.
    """
    rows = adapt_chart_rows(_payload_with(chart, "close", None),
                            instrument_id="xnas:AAPL")
    assert len(rows) == len(chart["chart"]["result"][0]["timestamp"]) - 1


# --- the request window must match the spec's inclusive range -------------


def test_the_query_window_covers_the_session_opening_on_the_final_day():
    """A real capture lost its last session to this off-by-one.

    The source returns bars strictly before `period2`, and a US session opens
    at 13:30 or 14:30 UTC. Midnight of the final day therefore excludes that
    day's session while every other day survives -- producing one permanent,
    identical missing session on every instrument, which looks like a provider
    gap and is not. The expected grid uses T23:59:59Z, so the request bound
    has to as well.
    """
    from datetime import datetime, timezone

    provider = YahooChartDailyProvider()
    params = provider.chart_params(timeframe="1d", start="2024-08-01",
                                   end="2026-07-31")

    final_session_open = int(
        datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc).timestamp())
    assert params["period2"] > final_session_open, (
        "period2 excludes the session opening on the last requested day")
    assert datetime.fromtimestamp(params["period2"], timezone.utc).date() \
        == datetime(2026, 7, 31, tzinfo=timezone.utc).date()

    # The lower bound stays at the start of the first day, which already
    # precedes that day's open.
    first_session_open = int(
        datetime(2024, 8, 1, 13, 30, tzinfo=timezone.utc).timestamp())
    assert params["period1"] <= first_session_open


def test_the_query_window_spans_every_expected_session_in_the_range():
    """Derived from the calendar, so it cannot drift from the expected grid."""
    provider = YahooChartDailyProvider()
    params = provider.chart_params(
        timeframe="1d", start=CORPUS_SPEC_V2.requested_start,
        end=CORPUS_SPEC_V2.requested_end)
    openings = [int(o.timestamp()) for o in CORPUS_SPEC_V2.expected_bar_opens()]
    assert params["period1"] <= min(openings)
    assert params["period2"] > max(openings)
