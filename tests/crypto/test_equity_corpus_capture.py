"""The capture pipeline, exercised offline against fabricated provider responses.

Every test here runs a full capture -- credential gate, reference metadata
check, pagination, canonicalisation, session validation, gap audit, manifest,
verification, rebuild -- with a transport that serves fixtures from a dict.
No socket is opened and no key exists.

The fixtures are built from the *real* calendar rather than from invented
timestamps. A test that made up its own session times would be testing itself:
it would agree with a broken calendar as happily as with a correct one. So the
bar openings come from `expected_bar_opens`, and the tests then break them on
purpose -- an after-hours row, a holiday row, a duplicated page, a wrong
ticker -- to check each is refused.

What is deliberately not tested here: whether the live Massive schema matches
`REQUIRED_BAR_FIELDS`. It cannot be, without a key. The adapter refuses an
unrecognised shape loudly, which is the honest behaviour for a contract that
has not yet met the real endpoint.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pytest

pytest.importorskip(
    "pandas_market_calendars",
    reason="the US equity corpus needs the optional [equities] extra")

from scripts.trading_lab.capture_us_equity_corpus import (  # noqa: E402
    BARS_PATH, CaptureError, CorpusLayout, MissingCredentialStop,
    REFERENCE_PATH, SPLITS_PATH, adapt_bar_rows, run_capture,
    verify_instrument_identity)
from scripts.trading_lab.credentials import (  # noqa: E402
    MASSIVE_API_KEY_ENV, assert_absent, massive_credentials)
from scripts.trading_lab.equity_corpus import (  # noqa: E402
    CORPUS_SPEC_V1, EquityCorpusError, USEquityCorpusV1, audit_gaps,
    corpus_content_hash, instrument_content_hash, iso,
    overlapping_missing_intervals)
from scripts.trading_lab.massive_http_transport import TransportResponse  # noqa: E402
from scripts.trading_lab.massive_provider import (  # noqa: E402
    MassiveStocksHistoricalProvider)
from scripts.trading_lab.verify_us_equity_corpus import (  # noqa: E402
    CorpusVerificationError, rebuild, verify)

SENTINEL = "sentinel-massive-key-do-not-use-1a2b3c4d5e6f"

# A short window rather than the full two years: the pipeline is identical and
# a test suite that captured 26 000 bars four times over would be slow enough
# that nobody would run it.
WINDOW_START = "2024-11-25"
WINDOW_END = "2024-12-02"

REFERENCE = {
    "AAPL": {"symbol": "AAPL", "name": "Apple Inc.", "primary_exchange": "XNAS",
             "type": "CS", "currency_name": "USD", "active": True},
    "MSFT": {"symbol": "MSFT", "name": "Microsoft Corporation",
             "primary_exchange": "XNAS", "type": "CS", "currency_name": "USD",
             "active": True},
    "NVDA": {"symbol": "NVDA", "name": "NVIDIA Corporation",
             "primary_exchange": "XNAS", "type": "CS", "currency_name": "USD",
             "active": True},
    "QQQ": {"symbol": "QQQ", "name": "Invesco QQQ Trust, Series 1",
            "primary_exchange": "XNAS", "type": "ETF", "currency_name": "USD",
            "active": True},
}


@pytest.fixture
def spec() -> USEquityCorpusV1:
    """The real spec over a short window. Every policy is unchanged."""
    return USEquityCorpusV1(requested_start=WINDOW_START,
                            requested_end=WINDOW_END)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv(MASSIVE_API_KEY_ENV, SENTINEL)
    return SENTINEL


def _price_row(opening, *, base="100.00"):
    """A well-formed bar. The price is arbitrary; the shape is not."""
    price = Decimal(base)
    return {
        "bar_open_at": iso(opening),
        "open": str(price),
        "high": str(price + Decimal("1.50")),
        "low": str(price - Decimal("0.75")),
        "close": str(price + Decimal("0.25")),
        "volume": "125000",
    }


class ScriptedTransport:
    """Serves fixtures keyed by (path, symbol, cursor). Records every call.

    Not a mock of `urllib`: it stands in for the whole transport, which means
    these tests exercise the capture runner's own logic and nothing else.
    """

    name = "scripted"

    def __init__(self, pages: dict, reference=None, splits=None):
        self.pages = pages
        self.reference = reference if reference is not None else REFERENCE
        self.splits = splits or {}
        self.calls: list[dict] = []
        self.stats = None

    def fetch(self, path, params, headers) -> TransportResponse:
        self.calls.append({"path": path, "params": dict(params),
                           "headers": dict(headers)})
        symbol = params.get("symbol", "")
        if path == REFERENCE_PATH:
            payload = {"results": [self.reference[symbol]]} if symbol in \
                self.reference else {"results": []}
        elif path == SPLITS_PATH:
            payload = {"splits": self.splits.get(symbol, [])}
        elif path == BARS_PATH:
            cursor = params.get("cursor", "")
            payload = self.pages[(symbol, cursor)]
        else:                                        # pragma: no cover
            raise AssertionError(f"unexpected path {path}")
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return TransportResponse(status=200, raw=raw, payload=payload,
                                 url=f"https://api.massive.com{path}")

    def request(self, path, params, headers):
        return self.fetch(path, params, headers).payload


def _single_page(spec, *, rows_for=None, adjustment=None):
    """One page per instrument, holding every expected bar in the window."""
    openings = spec.expected_bar_opens()
    pages = {}
    for instrument_id in spec.instruments:
        symbol = instrument_id.split(":")[-1]
        rows = (rows_for(instrument_id, openings) if rows_for
                else [_price_row(opening) for opening in openings])
        pages[(symbol, "")] = {
            "adjustment": adjustment or spec.adjustment_policy,
            "bars": rows,
        }
    return pages


def _capture(spec, pages, tmp_path, **kwargs):
    transport = ScriptedTransport(pages, **kwargs)
    # The same credential source the runner would build for itself, so the
    # sentinel actually reaches a header and the leak checks are not vacuous.
    provider = MassiveStocksHistoricalProvider(
        instruments=spec.instruments, transport=transport,
        credentials=massive_credentials(),
        adjustment_policy=spec.adjustment_policy)
    manifest = run_capture(spec=spec, root=tmp_path / "corpus",
                           transport=transport, provider=provider)
    return manifest, transport


# --- the credential gate ---------------------------------------------------


def test_no_credential_stops_before_the_network_and_writes_nothing(
        spec, tmp_path, monkeypatch):
    """A clean stop, not a partial corpus. There is no half-captured state."""
    monkeypatch.delenv(MASSIVE_API_KEY_ENV, raising=False)
    root = tmp_path / "corpus"
    transport = ScriptedTransport(_single_page(spec))

    with pytest.raises(MissingCredentialStop) as error:
        run_capture(spec=spec, root=root, transport=transport)
    assert "credential not configured" in str(error.value)
    assert transport.calls == [], "a request was made without a credential"
    assert not root.exists(), "a directory was created for a capture that never ran"


def test_a_capture_refuses_to_run_into_a_non_empty_directory(
        spec, keyed, tmp_path):
    """Two attempts must never be blended invisibly."""
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    assert manifest["content"]["captured"] is True
    with pytest.raises(CaptureError) as error:
        _capture(spec, _single_page(spec), tmp_path)
    assert "already exists" in str(error.value)


# --- the identity gate -----------------------------------------------------


def test_every_venue_is_verified_against_the_provider_before_any_bar(
        spec, keyed, tmp_path):
    """QQQ is not assumed to be on xnas. It is checked, every capture."""
    _, transport = _capture(spec, _single_page(spec), tmp_path)
    reference_calls = [call for call in transport.calls
                       if call["path"] == REFERENCE_PATH]
    bar_calls = [call for call in transport.calls if call["path"] == BARS_PATH]
    assert {call["params"]["symbol"] for call in reference_calls} == {
        "AAPL", "MSFT", "NVDA", "QQQ"}
    # Every reference check happens before the first bar request.
    first_bar = transport.calls.index(bar_calls[0])
    assert all(transport.calls.index(call) < first_bar
               for call in reference_calls)


def test_a_venue_that_disagrees_with_the_registry_stops_the_capture(
        spec, keyed, tmp_path):
    """xnys:QQQ and xnas:QQQ are different instruments with the same ticker."""
    reference = {**REFERENCE,
                 "QQQ": {**REFERENCE["QQQ"], "primary_exchange": "XNYS"}}
    with pytest.raises(CaptureError) as error:
        _capture(spec, _single_page(spec), tmp_path, reference=reference)
    message = str(error.value)
    # Both canonical ids appear, because the failure is that they are two
    # instruments and the message has to say which two.
    assert "xnys:QQQ" in message and "xnas:QQQ" in message
    assert "different instrument" in message


def test_an_asset_class_that_disagrees_stops_the_capture(spec, keyed, tmp_path):
    reference = {**REFERENCE, "QQQ": {**REFERENCE["QQQ"], "type": "CS"}}
    with pytest.raises(CaptureError) as error:
        _capture(spec, _single_page(spec), tmp_path, reference=reference)
    assert "not interchangeable" in str(error.value)


def test_the_registered_qqq_venue_matches_the_contractual_metadata():
    """The pre-capture proof: the fixture, not a guess about where ETFs list."""
    from scripts.trading_lab.instrument_registry import CATALOGUE_V1
    from scripts.trading_lab.massive_provider import parse_reference_ticker

    fixtures = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "crypto"
         / "massive_reference_tickers.json").read_text())
    parsed = parse_reference_ticker(fixtures["QQQ"]["results"][0])
    registered = CATALOGUE_V1.resolve("xnas:QQQ")
    assert parsed.venue == registered.instrument_id.venue == "xnas"
    assert parsed.asset_class == registered.asset_class == "ETF"


# --- session membership ----------------------------------------------------


def test_an_after_hours_bar_is_refused(spec, keyed, tmp_path):
    """A provider returning more than was asked for is normal. Accepting it is not."""
    from datetime import timedelta

    def rows_for(instrument_id, openings):
        rows = [_price_row(opening) for opening in openings]
        # 30 minutes past the last close of the window: a real timestamp, and
        # not on the grid.
        rows.append(_price_row(openings[-1] + timedelta(minutes=60)))
        return rows

    with pytest.raises(EquityCorpusError) as error:
        _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)
    assert "expected bar grid" in str(error.value)


def test_a_pre_market_bar_is_refused(spec, keyed, tmp_path):
    from datetime import timedelta

    def rows_for(instrument_id, openings):
        return [_price_row(openings[0] - timedelta(minutes=30)),
                *(_price_row(opening) for opening in openings)]

    with pytest.raises(EquityCorpusError):
        _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)


def test_a_holiday_bar_is_refused_and_a_holiday_is_not_a_gap(
        spec, keyed, tmp_path):
    """Thanksgiving 2024 is inside the window and is not a session."""
    from datetime import datetime, timezone

    thanksgiving = datetime(2024, 11, 28, 14, 30, tzinfo=timezone.utc)
    sessions = {session.session_date for session in spec.sessions()}
    assert "2024-11-28" not in sessions

    def rows_for(instrument_id, openings):
        return [*(_price_row(opening) for opening in openings),
                _price_row(thanksgiving)]

    with pytest.raises(EquityCorpusError):
        _capture(spec, _single_page(spec, rows_for=rows_for),
                 tmp_path / "with-holiday")

    # And with the holiday simply absent, nothing is reported missing.
    manifest, _ = _capture(spec, _single_page(spec), tmp_path / "clean")
    for entry in manifest["content"]["instruments"]:
        assert entry["gap_audit"]["missing_expected_bars"] == 0
        assert entry["gap_audit"]["extra_bars"] == 0


def test_the_early_close_yields_fewer_bars_and_that_is_not_a_gap(
        spec, keyed, tmp_path):
    """2024-11-29 closes at 13:00 New York: 7 bars, not 13, and no gap."""
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    layout = CorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    early = [row for row in rows if row["session_date"] == "2024-11-29"]
    regular = [row for row in rows if row["session_date"] == "2024-11-25"]
    assert len(early) == 7
    assert len(regular) == 13
    assert manifest["content"]["instruments"][0][
        "gap_audit"]["missing_expected_bars"] == 0


def test_a_bar_after_an_early_close_is_refused(spec, keyed, tmp_path):
    from datetime import datetime, timezone

    # 18:30Z is half an hour after the 2024-11-29 early close.
    after = datetime(2024, 11, 29, 18, 30, tzinfo=timezone.utc)

    def rows_for(instrument_id, openings):
        return [*(_price_row(opening) for opening in openings),
                _price_row(after)]

    with pytest.raises(EquityCorpusError):
        _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)


def test_dst_openings_are_captured_at_the_right_utc_hour(keyed, tmp_path):
    """Winter opens 14:30Z, summer 13:30Z. Both must land on the grid."""
    winter = USEquityCorpusV1(requested_start="2025-01-13",
                              requested_end="2025-01-17")
    summer = USEquityCorpusV1(requested_start="2025-07-14",
                              requested_end="2025-07-18")
    for spec, expected_open in ((winter, "14:30"), (summer, "13:30")):
        manifest, _ = _capture(spec, _single_page(spec), tmp_path / expected_open)
        layout = CorpusLayout(tmp_path / expected_open / "corpus")
        rows = [json.loads(line) for line in
                layout.canonical_path("xnas:AAPL").read_text().splitlines()]
        assert rows[0]["bar_open_at"][11:16] == expected_open
        assert manifest["content"]["instruments"][0][
            "gap_audit"]["missing_expected_bars"] == 0


# --- pagination ------------------------------------------------------------


def test_pagination_is_followed_to_exhaustion(spec, keyed, tmp_path):
    openings = spec.expected_bar_opens()
    half = len(openings) // 2
    pages = {}
    for instrument_id in spec.instruments:
        symbol = instrument_id.split(":")[-1]
        pages[(symbol, "")] = {
            "adjustment": spec.adjustment_policy,
            "bars": [_price_row(item) for item in openings[:half]],
            "next_cursor": "page-2",
        }
        pages[(symbol, "page-2")] = {
            "adjustment": spec.adjustment_policy,
            "bars": [_price_row(item) for item in openings[half:]],
        }
    manifest, _ = _capture(spec, pages, tmp_path)
    for entry in manifest["content"]["instruments"]:
        assert entry["rows"] == len(openings)
        assert entry["gap_audit"]["missing_expected_bars"] == 0
    # Two bar pages plus one splits request per instrument.
    assert manifest["content"]["raw_requests"] == 4 * 3


def test_a_repeated_pagination_cursor_stops_the_capture(spec, keyed, tmp_path):
    """A provider looping must not produce a corpus that looks whole."""
    openings = spec.expected_bar_opens()
    pages = {}
    for instrument_id in spec.instruments:
        symbol = instrument_id.split(":")[-1]
        page = {"adjustment": spec.adjustment_policy,
                "bars": [_price_row(item) for item in openings[:5]],
                "next_cursor": "stuck"}
        pages[(symbol, "")] = page
        pages[(symbol, "stuck")] = page
    with pytest.raises(CaptureError) as error:
        _capture(spec, pages, tmp_path)
    assert "twice" in str(error.value)
    assert "looks whole" in str(error.value)


def test_a_duplicate_bar_opening_across_pages_is_refused(spec, keyed, tmp_path):
    openings = spec.expected_bar_opens()
    pages = {}
    for instrument_id in spec.instruments:
        symbol = instrument_id.split(":")[-1]
        pages[(symbol, "")] = {
            "adjustment": spec.adjustment_policy,
            "bars": [_price_row(item) for item in openings],
            "next_cursor": "again",
        }
        pages[(symbol, "again")] = {
            "adjustment": spec.adjustment_policy,
            # The same opening at a different price: two answers to one
            # question, and the capture cannot choose.
            "bars": [_price_row(openings[0], base="200.00")],
        }
    with pytest.raises(CaptureError) as error:
        _capture(spec, pages, tmp_path)
    assert "more than once" in str(error.value)


# --- data validation -------------------------------------------------------


def test_out_of_order_rows_are_canonicalised_not_rejected(spec, keyed, tmp_path):
    """Arrival order is the provider's business. Canonical order is ours."""
    def rows_for(instrument_id, openings):
        return [_price_row(opening) for opening in reversed(openings)]

    manifest, _ = _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)
    layout = CorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    assert rows == sorted(rows, key=lambda row: row["bar_open_at"])
    assert manifest["content"]["instruments"][0]["gap_audit"][
        "missing_expected_bars"] == 0


@pytest.mark.parametrize("field,value", [
    ("high", "50.00"),      # below open/close
    ("low", "500.00"),      # above open/close
    ("open", "-1.00"),      # negative price
    ("close", "0"),         # zero price
    ("volume", "-5"),       # negative volume
])
def test_a_bar_that_cannot_be_a_bar_is_refused(spec, keyed, tmp_path, field,
                                               value):
    def rows_for(instrument_id, openings):
        rows = [_price_row(opening) for opening in openings]
        rows[3] = {**rows[3], field: value}
        return rows

    with pytest.raises(EquityCorpusError):
        _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)


def test_a_float_price_is_refused(spec, keyed, tmp_path):
    def rows_for(instrument_id, openings):
        rows = [_price_row(opening) for opening in openings]
        rows[0] = {**rows[0], "close": 100.25}
        return rows

    with pytest.raises(EquityCorpusError) as error:
        _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)
    assert "float" in str(error.value)


def test_a_raw_adjusted_mismatch_stops_the_capture(spec, keyed, tmp_path):
    """Asked for SPLIT_ADJUSTED, handed RAW, labelled SPLIT_ADJUSTED."""
    with pytest.raises(CaptureError) as error:
        _capture(spec, _single_page(spec, adjustment="RAW"), tmp_path)
    assert "refusing to relabel" in str(error.value)


def test_a_response_shape_the_contract_does_not_recognise_is_refused(spec):
    """Better a loud stop than a corpus of plausible nonsense."""
    with pytest.raises(CaptureError) as error:
        adapt_bar_rows({"results": []}, spec=spec, instrument_id="xnas:AAPL")
    assert "does not match this capture's contract" in str(error.value)
    with pytest.raises(CaptureError):
        adapt_bar_rows({"bars": [{"open": "1"}]}, spec=spec,
                       instrument_id="xnas:AAPL")


def test_a_wrong_ticker_in_the_reference_response_is_refused(spec, keyed,
                                                             tmp_path):
    from scripts.trading_lab.massive_provider import MassiveProviderError

    reference = {**REFERENCE, "NVDA": {**REFERENCE["NVDA"], "symbol": "NVDIA"}}
    transport = ScriptedTransport(_single_page(spec), reference=reference)
    provider = MassiveStocksHistoricalProvider(
        instruments=spec.instruments, transport=transport,
        credentials=massive_credentials(),
        adjustment_policy=spec.adjustment_policy)
    with pytest.raises((CaptureError, MassiveProviderError)):
        verify_instrument_identity(provider, "xnas:NVDA")


# --- gaps ------------------------------------------------------------------


def test_a_missing_expected_bar_is_reported_and_never_filled(
        spec, keyed, tmp_path):
    def rows_for(instrument_id, openings):
        rows = [_price_row(opening) for opening in openings]
        del rows[6]
        return rows

    manifest, _ = _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)
    entry = manifest["content"]["instruments"][0]
    assert entry["gap_audit"]["missing_expected_bars"] == 1
    assert entry["gap_audit"]["classification"] == "MISSING_EXPECTED_BAR"
    assert entry["rows"] == len(spec.expected_bar_opens()) - 1

    layout = CorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    missing = entry["gap_audit"]["missing"][0]
    assert all(row["bar_open_at"] != missing for row in rows), \
        "the missing bar was filled in"


def test_overlapping_missing_intervals_are_described_not_diagnosed(
        spec, keyed, tmp_path):
    """Four instruments missing the same interval is a fact, not an outage."""
    def rows_for(instrument_id, openings):
        rows = [_price_row(opening) for opening in openings]
        del rows[2]
        return rows

    manifest, _ = _capture(spec, _single_page(spec, rows_for=rows_for), tmp_path)
    overlap = manifest["content"]["overlapping_missing_intervals"]
    assert len(overlap) == 1
    rendered = json.dumps(manifest).lower()
    for verdict in ("outage", "downtime", "provider_failure"):
        assert verdict not in rendered


def test_the_gap_audit_never_counts_overnight_or_a_weekend(spec, keyed,
                                                           tmp_path):
    """The window spans a weekend and a holiday; neither appears as missing."""
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    for entry in manifest["content"]["instruments"]:
        assert entry["gap_audit"]["missing"] == []
        assert entry["gap_audit"]["expected_bars"] == len(
            spec.expected_bar_opens())
    assert manifest["content"]["sessions"] < 8, "weekend days became sessions"


# --- corporate actions -----------------------------------------------------


def test_split_records_are_captured_with_provenance_and_not_applied(
        spec, keyed, tmp_path):
    splits = {"NVDA": [{"effective_date": "2024-11-26", "ratio_numerator": 10,
                        "ratio_denominator": 1}]}
    manifest, _ = _capture(spec, _single_page(spec), tmp_path, splits=splits)
    entry = next(item for item in manifest["content"]["instruments"]
                 if item["instrument_id"] == "xnas:NVDA")
    assert len(entry["splits"]) == 1
    record = entry["splits"][0]
    assert record["label"] == "10-for-1"
    assert record["applied_by_provider"] is True
    assert record["recomputed_by_hyprl"] is False
    assert len(record["source_raw_hash"]) == 64
    assert len(entry["corporate_actions_hash"]) == 64


# --- the credential never lands anywhere ----------------------------------


def test_the_key_reaches_the_header_and_no_committed_file(spec, keyed,
                                                          tmp_path):
    manifest, transport = _capture(spec, _single_page(spec), tmp_path)
    bar_calls = [call for call in transport.calls if call["path"] == BARS_PATH]
    assert SENTINEL in bar_calls[0]["headers"]["Authorization"]

    assert_absent(SENTINEL, manifest, where="manifest")
    assert manifest["content"]["credential_present"] is True
    for path in sorted((tmp_path / "corpus").rglob("*")):
        if path.is_file():
            assert SENTINEL not in path.read_text(errors="ignore"), path
    rendered = json.dumps(manifest).lower()
    for banned in ("authorization", "bearer", "api_key", "apikey",
                   "hyprl_massive"):
        assert banned not in rendered


def test_a_capture_failure_message_never_quotes_the_credential(
        spec, keyed, tmp_path):
    reference = {**REFERENCE,
                 "AAPL": {**REFERENCE["AAPL"], "primary_exchange": "XNYS"}}
    with pytest.raises(CaptureError) as error:
        _capture(spec, _single_page(spec), tmp_path, reference=reference)
    assert_absent(SENTINEL, str(error.value), where="capture error")
    assert_absent(SENTINEL, repr(error.value), where="capture error repr")


# --- determinism -----------------------------------------------------------


def test_the_same_raw_input_produces_the_same_hashes(spec, keyed, tmp_path):
    first, _ = _capture(spec, _single_page(spec), tmp_path / "a")
    second, _ = _capture(spec, _single_page(spec), tmp_path / "b")
    assert first["content"]["corpus_spec_hash"] == \
        second["content"]["corpus_spec_hash"]
    assert first["content"]["corpus_content_hash"] == \
        second["content"]["corpus_content_hash"]
    for left, right in zip(first["content"]["instruments"],
                           second["content"]["instruments"]):
        assert left["instrument_content_hash"] == right["instrument_content_hash"]
        assert left["canonical_sha256"] == right["canonical_sha256"]


def test_the_content_hash_is_independent_of_the_callers_decimal_context(
        spec, keyed, tmp_path):
    """A hash that moved with an ambient precision would be nobody's hash."""
    import decimal

    baseline, _ = _capture(spec, _single_page(spec), tmp_path / "a")
    with decimal.localcontext() as context:
        context.prec = 6
        shifted, _ = _capture(spec, _single_page(spec), tmp_path / "b")
    assert baseline["content"]["corpus_content_hash"] == \
        shifted["content"]["corpus_content_hash"]


def test_the_content_hash_covers_the_instrument_identity(spec, keyed, tmp_path):
    """Identical prices under two tickers must not hash the same."""
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    hashes = {entry["instrument_id"]: entry["instrument_content_hash"]
              for entry in manifest["content"]["instruments"]}
    # Every instrument received identical fabricated prices in this fixture,
    # so equal hashes would mean the identity is not in the hash.
    assert len(set(hashes.values())) == len(hashes)


def test_the_instrument_identity_alone_changes_the_content_hash(spec):
    """Isolated from provenance, which would otherwise carry the difference.

    The capture-level check above passes even if `instrument_id` is dropped
    from the hashed row, because `source_record_identity` embeds the
    instrument too and keeps the hashes apart. That is an accident, not a
    guarantee: a provider that served two instruments from one response would
    give them the same provenance, and their bars would then hash identically.
    So this builds two bar sets differing in the identity and nothing else.
    """
    from datetime import datetime, timedelta, timezone

    from scripts.trading_lab.equity_corpus import CanonicalBar

    opening = datetime(2024, 11, 25, 14, 30, tzinfo=timezone.utc)
    common = dict(
        provider_id=spec.provider_id, bar_open_at=opening,
        bar_close_at=opening + timedelta(minutes=30),
        session_date="2024-11-25", session_type="REGULAR", timeframe="30m",
        open=Decimal("100.00"), high=Decimal("101.50"), low=Decimal("99.25"),
        close=Decimal("100.25"), volume=Decimal("125000"),
        adjustment_policy=spec.adjustment_policy,
        source_raw_hash="a" * 64, source_record_identity="b" * 64)
    aapl = [CanonicalBar(instrument_id="xnas:AAPL", **common)]
    msft = [CanonicalBar(instrument_id="xnas:MSFT", **common)]
    assert instrument_content_hash(aapl) != instrument_content_hash(msft)
    # And the corpus hash inherits that distinction.
    assert corpus_content_hash({"xnas:AAPL": aapl}) != \
        corpus_content_hash({"xnas:MSFT": msft})


def test_the_spec_hash_does_not_move_with_the_data(spec, keyed, tmp_path):
    def rows_for(instrument_id, openings):
        return [_price_row(opening, base="999.00") for opening in openings]

    plain, _ = _capture(spec, _single_page(spec), tmp_path / "a")
    priced, _ = _capture(spec, _single_page(spec, rows_for=rows_for),
                         tmp_path / "b")
    assert plain["content"]["corpus_spec_hash"] == \
        priced["content"]["corpus_spec_hash"]
    assert plain["content"]["corpus_content_hash"] != \
        priced["content"]["corpus_content_hash"]


# --- offline verification and rebuild -------------------------------------


def test_the_corpus_verifies_offline_from_its_own_files(spec, keyed, tmp_path):
    _capture(spec, _single_page(spec), tmp_path)
    report = verify(tmp_path / "corpus")
    assert report["ok"] is True
    assert report["offline"] is True


def test_verification_never_touches_a_transport(spec, keyed, tmp_path,
                                                monkeypatch):
    """Proven by making any network attempt explode, not by reading the code."""
    _capture(spec, _single_page(spec), tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("verification opened a socket")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.delenv(MASSIVE_API_KEY_ENV, raising=False)
    assert verify(tmp_path / "corpus")["ok"] is True


def test_raw_to_canonical_rebuild_is_byte_identical(spec, keyed, tmp_path):
    _capture(spec, _single_page(spec), tmp_path)
    report = rebuild(tmp_path / "corpus")
    assert report["byte_identical"] is True
    assert verify(tmp_path / "corpus")["ok"] is True


def test_a_corrupted_raw_file_fails_verification(spec, keyed, tmp_path):
    """A silent raw corruption would make every derived hash a lie."""
    _capture(spec, _single_page(spec), tmp_path)
    layout = CorpusLayout(tmp_path / "corpus")
    raw = next(iter(sorted((layout.root / "raw").rglob("*.json"))))
    payload = json.loads(raw.read_text())
    payload["bars"][0]["close"] = "999999.00"
    raw.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(CorpusVerificationError) as error:
        verify(tmp_path / "corpus")
    assert "changed since capture" in str(error.value)


def test_a_corrupted_canonical_file_fails_verification(spec, keyed, tmp_path):
    _capture(spec, _single_page(spec), tmp_path)
    layout = CorpusLayout(tmp_path / "corpus")
    path = layout.canonical_path("xnas:AAPL")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(CorpusVerificationError):
        verify(tmp_path / "corpus")


def test_an_edited_manifest_fails_verification(spec, keyed, tmp_path):
    _capture(spec, _single_page(spec), tmp_path)
    layout = CorpusLayout(tmp_path / "corpus")
    manifest = json.loads(layout.manifest_path.read_text())
    manifest["content"]["instruments"][0]["rows"] += 1
    layout.manifest_path.write_text(json.dumps(manifest, indent=1,
                                               sort_keys=True))
    with pytest.raises(CorpusVerificationError):
        verify(tmp_path / "corpus")


def test_the_manifest_records_the_point_in_time_limitation(spec, keyed,
                                                           tmp_path):
    """The claim this corpus must never make."""
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    content = manifest["content"]
    assert content["point_in_time_exchange_revision_history"] is False
    assert content["historical_market_data_corpus"] is True
    assert content["captured"] is True


def test_the_manifest_names_the_calendar_that_defined_the_grid(spec, keyed,
                                                               tmp_path):
    manifest, _ = _capture(spec, _single_page(spec), tmp_path)
    recorded = manifest["content"]["corpus_spec"]
    assert recorded["calendar_id"] == "US_EQUITY_REGULAR"
    assert recorded["calendar_dependency_version"] == "5.4.0"
    assert len(recorded["calendar_spec_hash"]) == 64
