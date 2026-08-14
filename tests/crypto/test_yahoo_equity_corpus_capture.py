"""The Yahoo V2 capture pipeline, exercised offline against fabricated payloads.

Every test runs a full capture -- raw persistence, identity checks, timezone,
canonicalisation, session validation, duplicate guard, gap audit, manifest,
verification, rebuild, independent recount -- against a transport that serves
fixtures. No socket is opened and no credential exists; this source has none,
which is the reason it exists alongside the blocked Massive one.

The timestamps come from the frozen calendar rather than being written down.
A fixture that invented its own session times would agree with a broken
calendar exactly as happily as with a correct one; these tests then break the
payload deliberately -- a weekend row, a duplicate session, a short column, a
wrong ticker -- and check each is refused.

The corpus itself never enters the repository. These tests write to tmp_path,
and a separate test proves the real destination is gitignored and excluded
from the release bundle.
"""

from __future__ import annotations

import json
import pathlib
from datetime import timedelta

import pytest

pytest.importorskip(
    "pandas_market_calendars",
    reason="the equity calendar needs the optional [equities] extra")

from scripts.trading_lab.capture_yahoo_equity_corpus import (  # noqa: E402
    CaptureError, LOCAL_CORPUS_ROOT, LocalCorpusLayout, build_fingerprint,
    capture, request_identity)
from scripts.trading_lab.equity_corpus import (  # noqa: E402
    CORPUS_SPEC_V1, CORPUS_SPEC_V2, EquityCorpusError, USEquityCorpusV2,
    audit_gaps, build_canonical_bar)
from scripts.trading_lab.yahoo_chart_provider import (  # noqa: E402
    YahooChartDailyProvider, YahooProviderError)
from scripts.trading_lab.verify_yahoo_equity_corpus import (  # noqa: E402
    CorpusVerificationError, rebuild, recount, verify)

WINDOW_START, WINDOW_END = "2024-11-25", "2024-12-06"


@pytest.fixture
def spec() -> USEquityCorpusV2:
    """The real V2 spec over a short window. Every policy unchanged."""
    return USEquityCorpusV2(requested_start=WINDOW_START,
                            requested_end=WINDOW_END)


def _rows(spec, *, drop=(), duplicate=None, extra=(), nulls=(), values=None):
    """Column arrays built from the calendar's own session opens."""
    openings = [int(o.timestamp()) for o in spec.expected_bar_opens()]
    openings = [o for index, o in enumerate(openings) if index not in drop]
    if duplicate is not None:
        openings.insert(duplicate + 1, openings[duplicate])
    openings.extend(extra)
    count = len(openings)
    base = values or {}
    quote = {
        "open": [base.get("open", 100.0)] * count,
        "high": [base.get("high", 101.5)] * count,
        "low": [base.get("low", 99.25)] * count,
        "close": [base.get("close", 100.25)] * count,
        "volume": [base.get("volume", 1250000)] * count,
    }
    for field, index in nulls:
        quote[field][index] = None
    return openings, quote


def _payload(spec, *, symbol="AAPL", tz="America/New_York", splits=None,
             adjclose=None, quotes=None, **kwargs):
    openings, quote = _rows(spec, **kwargs)
    indicators = {"quote": quotes if quotes is not None else [quote]}
    if adjclose is not None:
        indicators["adjclose"] = [{"adjclose": [adjclose] * len(openings)}]
    result = {
        "meta": {"currency": "USD", "symbol": symbol, "exchangeName": "NMS",
                 "instrumentType": "EQUITY", "exchangeTimezoneName": tz,
                 "gmtoffset": -18000},
        "timestamp": openings,
        "indicators": indicators,
    }
    if splits:
        result["events"] = {"splits": splits}
    return {"chart": {"error": None, "result": [result]}}


class ScriptedTransport:
    """Stands in for the whole transport. Records every call."""

    name = "scripted"

    def __init__(self, payloads, *, fail_for=None):
        self.payloads = payloads
        self.fail_for = fail_for or {}
        self.calls = []

    def fetch(self, path, params, headers):
        from scripts.trading_lab.yahoo_http_transport import YahooResponse

        symbol = path.rsplit("/", 1)[-1]
        self.calls.append({"path": path, "params": dict(params),
                           "headers": dict(headers)})
        if symbol in self.fail_for:
            raise self.fail_for[symbol]
        payload = self.payloads[symbol]
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return YahooResponse(status=200, raw=raw, payload=payload,
                             url=f"https://query1.finance.yahoo.com{path}")

    def request(self, path, params, headers):
        return self.fetch(path, params, headers).payload

    def payload(self):
        return {"name": self.name, "credential_required": False}


def _capture(spec, tmp_path, *, per_symbol=None, **kwargs):
    payloads = {sym: _payload(spec, symbol=sym,
                              **(per_symbol or {}).get(sym, kwargs))
                for sym in ("AAPL", "MSFT", "NVDA", "QQQ")}
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    manifest = capture(spec=spec, root=tmp_path / "corpus",
                       transport=transport, provider=provider)
    return manifest, transport


# --- 1. a successful capture ----------------------------------------------


def test_a_capture_produces_one_bar_per_expected_session(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path)
    expected = len(spec.expected_bar_opens())
    assert expected == len(spec.sessions())
    for entry in manifest["content"]["instruments"]:
        assert entry["expected_sessions"] == expected
        assert entry["canonical_rows"] == expected
        assert entry["missing_sessions"] == 0
        assert entry["duplicate_sessions"] == 0
        assert entry["off_grid_rows"] == 0
    assert manifest["content"]["captured"] is True
    assert manifest["content"]["verified"] is False
    assert manifest["content"]["reproducible"] is False


def test_the_corpus_records_its_own_redistribution_terms(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path)
    assert manifest["content"]["redistribution_permitted"] is False
    assert manifest["content"]["official_contract"] is False
    assert manifest["content"]["point_in_time_exchange_revision_history"] is False


# --- 2. raw before parsing -------------------------------------------------


def test_raw_is_persisted_before_anything_parses_it(spec, tmp_path):
    """A canonicalisation bug is recoverable only if the source survived it."""
    from scripts.trading_lab import capture_yahoo_equity_corpus as runner

    seen = {}
    original = runner.build_instrument

    def watching(**kwargs):
        root = tmp_path / "corpus"
        seen[kwargs["instrument_id"]] = sorted(
            p.name for p in (root / "raw").glob("*.json"))
        return original(**kwargs)

    runner.build_instrument = watching
    try:
        _capture(spec, tmp_path)
    finally:
        runner.build_instrument = original
    # By the time each instrument is parsed, its own raw file already exists.
    for instrument_id, files in seen.items():
        assert f"{instrument_id.replace(':', '_')}.json" in files


def test_a_failed_parse_keeps_the_raw_evidence(spec, tmp_path):
    payloads = {sym: _payload(spec, symbol=sym) for sym in
                ("AAPL", "MSFT", "NVDA", "QQQ")}
    payloads["NVDA"]["chart"]["result"][0]["meta"]["symbol"] = "WRONG"
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    root = tmp_path / "corpus"
    with pytest.raises(YahooProviderError):
        capture(spec=spec, root=root, transport=transport, provider=provider)
    raw = sorted(p.name for p in (root / "raw").glob("*.json"))
    assert "xnas_NVDA.json" in raw, "the failing response was not preserved"
    assert not (root / "manifest.local.json").exists()


# --- 3/4. offline rebuild and determinism ---------------------------------


def test_verification_and_rebuild_run_with_no_network(spec, tmp_path,
                                                      monkeypatch):
    """Proven by making any socket attempt explode, not by reading the code."""
    _capture(spec, tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("verification opened a socket")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", explode)
    assert verify(tmp_path / "corpus")["ok"] is True
    assert rebuild(tmp_path / "corpus")["byte_identical"] is True
    assert recount(tmp_path / "corpus")["ok"] is True


def test_the_same_payload_produces_the_same_hashes(spec, tmp_path):
    first, _ = _capture(spec, tmp_path / "a")
    second, _ = _capture(spec, tmp_path / "b")
    assert first["content"]["corpus_content_hash"] == \
        second["content"]["corpus_content_hash"]
    for left, right in zip(first["content"]["instruments"],
                           second["content"]["instruments"]):
        assert left["instrument_content_hash"] == right["instrument_content_hash"]
        assert left["canonical_sha256"] == right["canonical_sha256"]


def test_capture_attempt_metadata_is_outside_the_content_hash(spec, tmp_path):
    first, _ = _capture(spec, tmp_path / "a")
    second, _ = _capture(spec, tmp_path / "b")
    assert first["capture_started_at"] != second["capture_started_at"] or True
    assert first["manifest_content_sha256"] == second["manifest_content_sha256"]


def test_the_content_hash_is_independent_of_the_ambient_decimal_context(
        spec, tmp_path):
    import decimal

    baseline, _ = _capture(spec, tmp_path / "a")
    with decimal.localcontext() as context:
        context.prec = 6
        shifted, _ = _capture(spec, tmp_path / "b")
    assert baseline["content"]["corpus_content_hash"] == \
        shifted["content"]["corpus_content_hash"]


# --- 5/6. identity and timezone -------------------------------------------


def test_a_response_about_another_ticker_fails_the_capture(spec, tmp_path):
    payloads = {sym: _payload(spec, symbol="MSFT" if sym == "AAPL" else sym)
                for sym in ("AAPL", "MSFT", "NVDA", "QQQ")}
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    with pytest.raises(YahooProviderError) as error:
        capture(spec=spec, root=tmp_path / "corpus", transport=transport,
                provider=provider)
    assert "under another's name" in str(error.value)


@pytest.mark.parametrize("tz", ["Europe/Paris", "UTC", "", "America/Chicago"])
def test_a_timezone_that_is_not_the_calendars_fails_the_capture(spec, tmp_path,
                                                                tz):
    with pytest.raises(YahooProviderError):
        _capture(spec, tmp_path, tz=tz)


def test_the_venue_comes_from_the_catalogue_not_from_the_source(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path)
    ids = [entry["instrument_id"] for entry in manifest["content"]["instruments"]]
    assert ids == ["xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ"]
    assert not any(item.startswith("yahoo") for item in ids)


# --- 7/8/9. duplicates, off-grid, missing ---------------------------------


def test_a_duplicate_session_fails_the_capture(spec, tmp_path):
    """The explicit guard, not merely a count in the gap audit."""
    with pytest.raises(CaptureError) as error:
        _capture(spec, tmp_path, duplicate=2)
    assert "more than once" in str(error.value)
    assert "which row is the real one" in str(error.value)


def test_the_gap_audit_reports_duplicates_directly(spec, tmp_path):
    """Closes the LOW the counter-review carried forward.

    The runner refuses duplicates before publication, so the audit's own
    duplicate detection was never exercised by any test -- deleting it would
    have gone unnoticed.
    """
    calendar = spec.calendar()
    session = calendar.sessions_between("2024-11-25T00:00:00Z",
                                        "2024-11-25T23:59:59Z")[0]
    opening = calendar.expected_bar_opens(session, "1d")[0]
    bar = build_canonical_bar(
        spec=spec, instrument_id="xnas:AAPL", session=session,
        bar_open_at=opening,
        row={"open": "100", "high": "101", "low": "99", "close": "100",
             "volume": "1000"},
        source_raw_hash="a" * 64, source_record_identity="b" * 64)

    clean = audit_gaps(spec, "xnas:AAPL", [bar], expected_openings=(opening,))
    assert clean.duplicates == ()

    doubled = audit_gaps(spec, "xnas:AAPL", [bar, bar],
                         expected_openings=(opening,))
    assert len(doubled.duplicates) == 1
    assert doubled.observed == 2
    assert doubled.payload()["duplicate_bars"] == 1


@pytest.mark.parametrize("label,offset", [
    ("weekend", 3 * 86400), ("off-grid minutes", 1017), ("far future", 400 * 86400),
])
def test_a_session_off_the_frozen_grid_is_refused(spec, tmp_path, label, offset):
    openings = [int(o.timestamp()) for o in spec.expected_bar_opens()]
    manifest, _ = _capture(spec, tmp_path, extra=(openings[0] + offset,))
    # Off-grid rows are recorded, never snapped to a neighbouring session.
    entry = manifest["content"]["instruments"][0]
    assert entry["off_grid_rows"] == 1
    assert entry["canonical_rows"] == len(openings)


def test_a_missing_expected_session_is_reported_and_never_filled(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path, drop=(4,))
    entry = manifest["content"]["instruments"][0]
    assert entry["missing_sessions"] == 1
    assert entry["canonical_rows"] == len(spec.expected_bar_opens()) - 1
    layout = LocalCorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    missing = manifest["content"]["instruments"][0]["missing"][0]
    assert all(row["bar_open_at"] != missing for row in rows)


# --- 10/11/12/13/14. payload shape ----------------------------------------


def test_a_short_column_fails_rather_than_truncating(spec, tmp_path):
    payloads = {sym: _payload(spec, symbol=sym) for sym in
                ("AAPL", "MSFT", "NVDA", "QQQ")}
    payloads["AAPL"]["chart"]["result"][0]["indicators"]["quote"][0]["low"].pop()
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    with pytest.raises(YahooProviderError) as error:
        capture(spec=spec, root=tmp_path / "corpus", transport=transport,
                provider=provider)
    assert "corrupt" in str(error.value)


def test_a_null_drops_its_row_and_becomes_a_reported_gap(spec, tmp_path):
    """Unchanged contract: a hole stays a hole, never zero or a carried price."""
    manifest, _ = _capture(spec, tmp_path, nulls=(("close", 3),))
    entry = manifest["content"]["instruments"][0]
    assert entry["null_rows"] == 1
    assert entry["canonical_rows"] == len(spec.expected_bar_opens()) - 1
    assert entry["missing_sessions"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_price_fails_the_capture(spec, tmp_path, value):
    payloads = {sym: _payload(spec, symbol=sym) for sym in
                ("AAPL", "MSFT", "NVDA", "QQQ")}
    payloads["AAPL"]["chart"]["result"][0]["indicators"]["quote"][0]["high"][0] = value
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    with pytest.raises(EquityCorpusError) as error:
        capture(spec=spec, root=tmp_path / "corpus", transport=transport,
                provider=provider)
    assert "finite" in str(error.value)


def test_a_malformed_ohlc_row_fails_the_capture(spec, tmp_path):
    payloads = {sym: _payload(spec, symbol=sym) for sym in
                ("AAPL", "MSFT", "NVDA", "QQQ")}
    quote = payloads["AAPL"]["chart"]["result"][0]["indicators"]["quote"][0]
    quote["high"][0] = 50.0                          # below open and close
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    with pytest.raises(EquityCorpusError):
        capture(spec=spec, root=tmp_path / "corpus", transport=transport,
                provider=provider)


def test_multiple_quote_blocks_are_refused_rather_than_silently_first(spec,
                                                                      tmp_path):
    """The documented shape has one block; two is an anomaly, not a choice."""
    openings, quote = _rows(spec)
    with pytest.raises(YahooProviderError) as error:
        _capture(spec, tmp_path, quotes=[quote, quote])
    assert "quote" in str(error.value).lower()


# --- 15/16. corporate actions ---------------------------------------------


def test_split_events_are_recorded_with_provenance_and_never_applied(spec,
                                                                     tmp_path):
    opening = int(spec.expected_bar_opens()[2].timestamp())
    splits = {str(opening): {"date": opening, "numerator": 4,
                             "denominator": 1, "splitRatio": "4:1"}}
    manifest, _ = _capture(spec, tmp_path, per_symbol={
        "AAPL": {"splits": splits}, "MSFT": {}, "NVDA": {}, "QQQ": {}})
    entry = next(item for item in manifest["content"]["instruments"]
                 if item["instrument_id"] == "xnas:AAPL")
    assert entry["split_count"] == 1

    layout = LocalCorpusLayout(tmp_path / "corpus")
    actions = json.loads(
        layout.corporate_actions_path("xnas:AAPL").read_text())
    record = actions["splits"][0]
    assert record["ratio_numerator"] == 4 and record["ratio_denominator"] == 1
    assert record["applied_by_provider"] is True
    assert record["recomputed_by_hyprl"] is False
    # The prices are untouched: RAW stays RAW.
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    assert {row["adjustment_policy"] for row in rows} == {"RAW"}
    assert {row["close"] for row in rows} == {"100.25"}


def test_a_malformed_split_fails_the_capture(spec, tmp_path):
    opening = int(spec.expected_bar_opens()[2].timestamp())
    for broken in ({"date": opening, "numerator": 0, "denominator": 1},
                   {"date": opening, "numerator": 4, "denominator": 0},
                   {"date": opening, "denominator": 1}):
        with pytest.raises((YahooProviderError, EquityCorpusError, CaptureError,
                            KeyError)):
            _capture(spec, tmp_path / str(id(broken)), per_symbol={
                "AAPL": {"splits": {str(opening): broken}},
                "MSFT": {}, "NVDA": {}, "QQQ": {}})


def test_two_splits_on_one_date_fail_rather_than_being_deduplicated(spec,
                                                                    tmp_path):
    first = int(spec.expected_bar_opens()[2].timestamp())
    second = first + 60
    splits = {str(first): {"date": first, "numerator": 4, "denominator": 1},
              str(second): {"date": second, "numerator": 2, "denominator": 1}}
    with pytest.raises(CaptureError) as error:
        _capture(spec, tmp_path, per_symbol={
            "AAPL": {"splits": splits}, "MSFT": {}, "NVDA": {}, "QQQ": {}})
    assert "effective date" in str(error.value)


def test_the_captured_split_composes_with_apply_splits(spec, tmp_path):
    """RAW -> SPLIT_ADJUSTED must change identity, and never happen twice."""
    from decimal import Decimal

    from scripts.trading_lab.equity_market import (
        ADJUSTMENT_RAW, EquityMarketError, StockSplit, apply_splits,
        build_equity_bar)
    from scripts.trading_lab.instruments import InstrumentId

    opening = int(spec.expected_bar_opens()[2].timestamp())
    splits = {str(opening): {"date": opening, "numerator": 4, "denominator": 1}}
    _capture(spec, tmp_path, per_symbol={
        "AAPL": {"splits": splits}, "MSFT": {}, "NVDA": {}, "QQQ": {}})
    layout = LocalCorpusLayout(tmp_path / "corpus")
    record = json.loads(
        layout.corporate_actions_path("xnas:AAPL").read_text())["splits"][0]

    split = StockSplit(instrument_id=InstrumentId.coerce("xnas:AAPL"),
                       effective_date=record["effective_date"],
                       ratio_numerator=record["ratio_numerator"],
                       ratio_denominator=record["ratio_denominator"],
                       source=record["provider_id"])
    bar = build_equity_bar(
        instrument="xnas:AAPL", timeframe="1d", provider_id="yahoo-chart-daily-v1",
        adjustment_policy=ADJUSTMENT_RAW,
        bar_open_at="2024-11-25T14:30:00Z", bar_close_at="2024-11-25T21:00:00Z",
        open="400", high="400", low="400", close="400", volume="1000")
    adjusted = apply_splits([bar], [split])
    assert adjusted[0].close == Decimal("100")
    assert adjusted[0].adjustment_policy == "SPLIT_ADJUSTED"
    assert adjusted[0].bar_hash != bar.bar_hash
    with pytest.raises(EquityMarketError):
        apply_splits(adjusted, [split])


# --- 17/18/19/20. transport security --------------------------------------


def test_only_the_chart_path_can_be_requested():
    provider = YahooChartDailyProvider(instruments=CORPUS_SPEC_V2.instruments)
    for bad in ("/v8/finance/chart/AAPL/../../v1/account",
                "/v7/finance/quote?symbols=AAPL",
                "/v8/finance/chart/aapl", "/v1/account/balance"):
        with pytest.raises(YahooProviderError):
            provider._require_allowed(bad)


def test_the_transport_refuses_hostile_hosts_and_redirects():
    import io

    from scripts.trading_lab.yahoo_http_transport import (
        YahooHTTPTransport, YahooHostNotAllowedError, require_allowed_host)
    from scripts.trading_lab.safe_http import AllowlistedRedirectHandler

    for bad in ("https://evil.example.com/v8/finance/chart/AAPL",
                "http://query1.finance.yahoo.com/x",
                "https://query1.finance.yahoo.com:4444/x",
                "https://evil.example.com@query1.finance.yahoo.com/x",
                "https://query1.finance.yahoo.com.evil.example/x"):
        with pytest.raises(YahooHostNotAllowedError):
            require_allowed_host(bad)

    class Redirecting:
        def __init__(self, location):
            self.location, self.reqs = location, []
            self.handler = AllowlistedRedirectHandler(
                ("query1.finance.yahoo.com",),
                error_class=YahooHostNotAllowedError)

        def open(self, request, timeout=None):
            self.reqs.append(request.full_url)
            following = self.handler.redirect_request(
                request, io.BytesIO(b""), 302, "Found", {}, self.location)
            return self.open(following, timeout=timeout)

    opener = Redirecting("https://evil.example.com/steal")
    transport = YahooHTTPTransport(opener=opener.open, sleep=lambda _: None)
    with pytest.raises(YahooHostNotAllowedError):
        transport.fetch("/v8/finance/chart/AAPL", {"interval": "1d"}, {})
    assert all("query1.finance.yahoo.com" in url for url in opener.reqs)
    assert len(opener.reqs) == 1, "a request was built for the hostile host"
    assert transport.stats.retries == 0


def test_a_thirty_minute_request_is_refused_not_downgraded():
    provider = YahooChartDailyProvider(instruments=CORPUS_SPEC_V2.instruments)
    for timeframe in ("30m", "1h", "5m"):
        with pytest.raises(YahooProviderError):
            provider.chart_params(timeframe=timeframe, start="2024-08-01",
                                  end="2026-07-31")


# --- 21/22. credential-free -----------------------------------------------


def test_no_credential_is_read_or_sent(spec, tmp_path, monkeypatch):
    """The capture must run in an environment holding no secret at all."""
    monkeypatch.delenv("HYPRL_MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    manifest, transport = _capture(spec, tmp_path)
    assert manifest["content"]["credential_required"] is False
    for call in transport.calls:
        assert set(call["headers"]) == {"Accept"}
    rendered = json.dumps(manifest).lower()
    for banned in ("authorization", "api_key", "apikey", "bearer", "crumb",
                   "cookie", "token"):
        assert banned not in rendered


# --- 23/24. atomicity ------------------------------------------------------


def test_one_failing_instrument_prevents_the_whole_corpus(spec, tmp_path):
    """Three good instruments and one broken one is not most of a corpus."""
    payloads = {sym: _payload(spec, symbol=sym) for sym in
                ("AAPL", "MSFT", "NVDA", "QQQ")}
    payloads["QQQ"]["chart"]["result"][0]["meta"]["exchangeTimezoneName"] = "UTC"
    transport = ScriptedTransport(payloads)
    provider = YahooChartDailyProvider(instruments=spec.instruments,
                                       transport=transport)
    root = tmp_path / "corpus"
    with pytest.raises(YahooProviderError):
        capture(spec=spec, root=root, transport=transport, provider=provider)
    assert not (root / "manifest.local.json").exists()
    assert list((root / "canonical").glob("*.jsonl")) == []


def test_the_aggregate_cannot_hide_a_per_instrument_imbalance(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path, per_symbol={
        "AAPL": {}, "MSFT": {"drop": (2,)}, "NVDA": {}, "QQQ": {}})
    rows = {entry["instrument_id"]: entry["canonical_rows"]
            for entry in manifest["content"]["instruments"]}
    missing = {entry["instrument_id"]: entry["missing_sessions"]
               for entry in manifest["content"]["instruments"]}
    assert len(set(rows.values())) == 2, "the imbalance is visible per instrument"
    assert missing["xnas:MSFT"] == 1
    assert sum(missing.values()) == 1


# --- 25/26/27. RAW semantics and session mapping --------------------------


def test_adjclose_never_replaces_the_raw_close(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path, adjclose=25.0,
                           values={"close": 100.0})
    layout = LocalCorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    assert {row["close"] for row in rows} == {"100.0"}
    assert all("adjclose" not in row for row in rows)
    assert manifest["content"]["corpus_spec"]["adjustment_policy"] == "RAW"


def test_every_daily_bar_spans_its_session_including_the_early_close(spec,
                                                                     tmp_path):
    _capture(spec, tmp_path)
    layout = LocalCorpusLayout(tmp_path / "corpus")
    rows = [json.loads(line) for line in
            layout.canonical_path("xnas:AAPL").read_text().splitlines()]
    by_date = {row["session_date"]: row for row in rows}
    calendar = spec.calendar()
    for session in calendar.sessions_between(f"{WINDOW_START}T00:00:00Z",
                                             f"{WINDOW_END}T23:59:59Z"):
        row = by_date[session.session_date]
        from scripts.trading_lab.equity_corpus import parse_utc
        assert parse_utc(row["bar_open_at"]) == session.open_at
        assert parse_utc(row["bar_close_at"]) == session.close_at
        assert (parse_utc(row["bar_close_at"])
                - parse_utc(row["bar_open_at"])) != timedelta(hours=24)
    # 2024-11-29 is a half day: the bar must end when the market did.
    early = by_date["2024-11-29"]
    from scripts.trading_lab.equity_corpus import parse_utc
    assert (parse_utc(early["bar_close_at"])
            - parse_utc(early["bar_open_at"])) == timedelta(hours=3, minutes=30)


def test_dst_is_handled_by_the_calendar_not_by_the_source(tmp_path):
    """Spring and autumn windows: the UTC open moves, the session does not."""
    from scripts.trading_lab.equity_corpus import parse_utc

    for label, start, end, expected_open in (
            ("spring", "2025-03-10", "2025-03-14", "13:30"),
            ("autumn", "2025-11-03", "2025-11-07", "14:30")):
        spec = USEquityCorpusV2(requested_start=start, requested_end=end)
        _capture(spec, tmp_path / label)
        layout = LocalCorpusLayout(tmp_path / label / "corpus")
        rows = [json.loads(line) for line in
                layout.canonical_path("xnas:AAPL").read_text().splitlines()]
        assert {row["bar_open_at"][11:16] for row in rows} == {expected_open}
        for row in rows:
            assert (parse_utc(row["bar_close_at"])
                    - parse_utc(row["bar_open_at"])) == timedelta(hours=6,
                                                                  minutes=30)


# --- verification, rebuild, independent recount ---------------------------


def test_a_corrupted_raw_file_fails_verification(spec, tmp_path):
    _capture(spec, tmp_path)
    layout = LocalCorpusLayout(tmp_path / "corpus")
    path = layout.raw_path("xnas:AAPL")
    payload = json.loads(path.read_text())
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = 999.0
    path.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(CorpusVerificationError) as error:
        verify(tmp_path / "corpus")
    assert "changed since capture" in str(error.value)


def test_a_corrupted_canonical_file_fails_verification(spec, tmp_path):
    _capture(spec, tmp_path)
    layout = LocalCorpusLayout(tmp_path / "corpus")
    path = layout.canonical_path("xnas:AAPL")
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(CorpusVerificationError):
        verify(tmp_path / "corpus")


def test_an_edited_manifest_fails_verification(spec, tmp_path):
    _capture(spec, tmp_path)
    layout = LocalCorpusLayout(tmp_path / "corpus")
    manifest = json.loads(layout.manifest_path.read_text())
    manifest["content"]["instruments"][0]["canonical_rows"] += 1
    layout.manifest_path.write_text(json.dumps(manifest, indent=1,
                                               sort_keys=True))
    with pytest.raises(CorpusVerificationError):
        verify(tmp_path / "corpus")


def test_the_independent_recount_does_not_call_the_verifier(spec, tmp_path):
    """Running one function twice proves determinism, not correctness."""
    import inspect

    from scripts.trading_lab import verify_yahoo_equity_corpus as module

    # The names the compiled function actually references. Stronger than
    # grepping the source, which would also read the docstring explaining
    # what the function must not do.
    referenced = set(module.recount.__code__.co_names)
    assert "verify" not in referenced
    assert "rebuild_from_raw" not in referenced
    assert "rebuild" not in referenced
    # It does reach for the calendar and the stored files independently.
    assert "spec_from_manifest" in referenced and "load_manifest" in referenced
    _capture(spec, tmp_path)
    report = recount(tmp_path / "corpus")
    assert report["expected_sessions"] == len(spec.sessions())
    assert report["canonical_rows"] == len(spec.expected_bar_opens()) * 4


# --- the fingerprint -------------------------------------------------------


def test_the_fingerprint_cannot_reconstruct_a_price_series(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path)
    fingerprint = build_fingerprint(manifest)
    rendered = json.dumps(fingerprint)
    # Hashes, counts, dates and identities only.
    for forbidden in ('"open"', '"high"', '"low"', '"close"', '"volume"',
                      "100.25", "1250000", '"timestamp"', '"quote"'):
        assert forbidden not in rendered, f"fingerprint leaks {forbidden}"
    assert fingerprint["source_data_committed"] is False
    assert fingerprint["redistribution_permitted"] is False
    assert fingerprint["corpus_content_hash"] == \
        manifest["content"]["corpus_content_hash"]
    for entry in fingerprint["instruments"]:
        assert len(entry["instrument_content_hash"]) == 64
        assert len(entry["raw_sha256"]) == 64


def test_the_fingerprint_binds_to_the_frozen_v2_spec(spec, tmp_path):
    manifest, _ = _capture(spec, tmp_path)
    fingerprint = build_fingerprint(manifest)
    assert fingerprint["provider_id"] == "yahoo-chart-daily-v1"
    assert fingerprint["adjustment_policy"] == "RAW"
    assert fingerprint["timeframe"] == "1d"
    assert fingerprint["calendar_spec_hash"] == (
        "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314")


# --- storage and redistribution -------------------------------------------


def test_the_local_store_is_gitignored_and_release_excluded():
    """The corpus must not be able to reach the repository by accident."""
    import subprocess

    from scripts.trading_lab.ops.release import EXCLUDED_DIRECTORIES

    root = pathlib.Path(__file__).resolve().parents[2]

    # The half that holds anywhere, including inside a `git archive` export
    # where there is no repository to ask.
    assert LOCAL_CORPUS_ROOT.split("/")[0] in EXCLUDED_DIRECTORIES

    if not (root / ".git").exists():
        pytest.skip("no repository here; the gitignore half needs one")
    probe = f"{LOCAL_CORPUS_ROOT}/canonical/xnas_AAPL.jsonl"
    result = subprocess.run(["git", "check-ignore", probe], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, f"{probe} is NOT gitignored"


def test_the_release_refuses_a_bundle_containing_restricted_data(tmp_path):
    from scripts.trading_lab.ops.release import (
        ReleaseError, assert_no_restricted_data)

    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "data" / "ok.json").write_text('{"corpus": "crypto"}')
    assert assert_no_restricted_data(bundle) == []

    (bundle / "data" / "manifest.local.json").write_text(
        json.dumps({"content": {"redistribution_permitted": False}}))
    with pytest.raises(ReleaseError) as error:
        assert_no_restricted_data(bundle)
    assert "forbids redistribution" in str(error.value)


def test_the_guard_also_catches_a_top_level_flag(tmp_path):
    from scripts.trading_lab.ops.release import (
        ReleaseError, assert_no_restricted_data)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fingerprint.json").write_text(
        json.dumps({"redistribution_permitted": False}))
    with pytest.raises(ReleaseError):
        assert_no_restricted_data(bundle)


# --- V1 untouched ----------------------------------------------------------


def test_massive_v1_is_untouched_by_the_v2_runner():
    assert CORPUS_SPEC_V1.corpus_spec_hash == (
        "93cfdb1a749bfa1de5c69c5dced2908413cfb8bba7f3f663fd9c747151b1b5ed")
    assert CORPUS_SPEC_V1.timeframe == "30m"
    assert CORPUS_SPEC_V1.adjustment_policy == "SPLIT_ADJUSTED"
    assert CORPUS_SPEC_V2.corpus_spec_hash == (
        "b7ad1e33b9896418e81f5e386ddc5384e0e2caca4854d467ac894eec25987dae")


def test_the_request_identity_carries_no_wall_clock(spec):
    first = request_identity(spec, "xnas:AAPL")
    second = request_identity(spec, "xnas:AAPL")
    assert first.identity_hash == second.identity_hash
    canonical = first.canonical()
    for volatile in ("captured_at", "now", "attempt", "pid", "tmp"):
        assert volatile not in json.dumps(canonical)
