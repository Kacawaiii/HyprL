"""Phase 6A-R: every identity boundary, tested as a cross-product attack.

Not marked `ml` except where a check must load a fitted artefact.

The shape of every test here is "right type, wrong identity". That is the
failure mode this audit exists for: a BTC candle and an ETH candle are the
same six numbers, a BTC target stream and an ETH target stream are the same
dataclass, and a model fitted on one predicts happily from the other's
features. Nothing raises a type error. The result is plausible and wrong.

The classes being enforced:

* A -- semantic (instrument, venue, provider, timeframe): parsed, canonical,
  aliases welcome.
* B -- opaque (digests, protocol versions): byte-exact, never normalised.
* D -- legacy (`BTC-USD` in artefacts and the event log): reached through an
  adapter.
* E -- runtime (session ids): validated, never canonicalised.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

ALIASES = ("BTC-USD", "btc-usd", "BTC/USD", "BTC_USD", "BTC.USD", "BTCUSD",
           " BTC-USD ", "coinbase:BTC-USD", "COINBASE:btc_usd")


# --- class A: semantic identity resolves, unknown fails closed -------------


@pytest.mark.parametrize("spelling", ALIASES)
def test_every_spelling_resolves_to_one_instrument(spelling):
    from scripts.trading_lab.identity import resolve_instrument

    assert resolve_instrument(spelling).canonical_id == "coinbase:BTC-USD"


@pytest.mark.parametrize("spelling", ALIASES)
def test_the_legacy_adapter_reaches_the_string_artefacts_use(spelling):
    from scripts.trading_lab.identity import resolve_legacy_product

    assert resolve_legacy_product(spelling) == "BTC-USD"


@pytest.mark.parametrize("unknown", [
    "SOL-USD", "AAPL", "nasdaq:AAPL", "", "   ", "---", None, 42, b"BTC-USD",
    "BTC-", "-USD", "BTC–USD", "coinbase:", "A" * 100,
])
def test_an_unknown_identity_never_falls_back_to_a_default(unknown):
    """Not BTC, not the first registry entry, not Coinbase. Nothing."""
    from scripts.trading_lab.identity import IdentityError, resolve_instrument

    with pytest.raises(IdentityError):
        resolve_instrument(unknown)


def test_unknown_is_never_equal_to_anything_including_another_unknown():
    from scripts.trading_lab.identity import same_instrument

    assert not same_instrument("SOL-USD", "SOL-USD")
    assert not same_instrument("SOL-USD", "BTC-USD")
    assert same_instrument("btc-usd", "coinbase:BTC-USD")


def test_a_mismatch_names_both_sides():
    from scripts.trading_lab.identity import (
        InstrumentMismatchError, require_same_instrument)

    with pytest.raises(InstrumentMismatchError) as error:
        require_same_instrument("BTC-USD", "ETH-USD", context="probe")
    message = str(error.value)
    assert "coinbase:BTC-USD" in message and "coinbase:ETH-USD" in message


# --- class B: digests stay byte-exact -------------------------------------


def test_a_digest_is_matched_exactly_and_never_normalised():
    """Lowercasing a digest accepts a value the producer never emitted."""
    from scripts.trading_lab.identity import IdentityError, require_exact_digest

    good = "a" * 64
    assert require_exact_digest(good, field="probe") == good
    for bad in (good.upper(), f" {good} ", f"{good}\n", good[:63], good + "a",
                "g" * 64, "0X" + "a" * 62, None, 1, b"a" * 64):
        with pytest.raises(IdentityError):
            require_exact_digest(bad, field="probe")


def test_two_digests_differing_only_in_case_do_not_match():
    from scripts.trading_lab.identity import IdentityError, digests_match

    with pytest.raises(IdentityError):
        digests_match("a" * 64, "A" * 64, field="probe")


@pytest.mark.parametrize("mutation", ["upper", "pad", "newline"])
def test_a_tampered_artefact_digest_is_refused(mutation):
    """The permissive-digest failure, at the boundary that actually loads one."""
    from scripts.trading_lab.paper_model import PaperModelError, load_paper_model

    directory = REPO_ROOT / "data/models/paper_v1"
    if not (directory / "BTC-USD.json").is_file():
        pytest.skip("no frozen models in this checkout")
    from scripts.trading_lab.paper_model import read_artifact

    artifact = dict(read_artifact(directory / "BTC-USD.json"))
    digest = artifact["artifact_hash"]
    artifact["artifact_hash"] = {
        "upper": digest.upper(), "pad": f" {digest} ", "newline": f"{digest}\n",
    }[mutation]
    with pytest.raises(PaperModelError):
        load_paper_model(artifact, product="BTC-USD")


# --- class B: protocol versions are closed, not normalised ----------------


def test_a_protocol_version_is_matched_exactly():
    from scripts.trading_lab.identity import IdentityError, require_exact_protocol

    expected = "trading-lab.signal-engine.v1"
    assert require_exact_protocol(expected, expected, field="protocol") == expected
    for bad in (expected.upper(), f" {expected} ", "trading-lab.signal-engine.V1",
                "trading-lab.signal-engine.v01", "v1", None, 1):
        with pytest.raises(IdentityError):
            require_exact_protocol(bad, expected, field="protocol")


def test_an_unknown_protocol_fails_closed():
    from scripts.trading_lab.identity import IdentityError, require_exact_protocol

    with pytest.raises(IdentityError):
        require_exact_protocol("trading-lab.signal-engine.v9",
                               ("trading-lab.signal-engine.v1",), field="protocol")


# --- class E: session ids are validated, never canonicalised --------------


def test_a_session_id_is_validated_but_not_folded():
    """Two runs differing only in case are two runs, not one."""
    from scripts.trading_lab.identity import IdentityError, require_session_id

    assert require_session_id("paper-20260811T035750Z") == "paper-20260811T035750Z"
    assert require_session_id("s1") == "s1"
    for bad in ("", "   ", "a/b", "..", "../../etc/passwd", "a\x00b", "a" * 100,
                None, 5, "sess ion"):
        with pytest.raises(IdentityError):
            require_session_id(bad)


def test_a_session_id_never_becomes_a_path():
    from scripts.trading_lab.identity import IdentityError, require_session_id

    for traversal in ("../secrets", "runtime/../../etc", "a\\b", "/abs"):
        with pytest.raises(IdentityError):
            require_session_id(traversal)


# --- cross-product: the paper model ---------------------------------------


@pytest.mark.ml
def test_an_eth_model_cannot_be_loaded_into_a_btc_slot():
    """Five hashes verified and the product ignored: the artefact loaded."""
    from scripts.trading_lab.paper_model import (
        PaperModelError, load_paper_model, read_artifact)

    directory = REPO_ROOT / "data/models/paper_v1"
    if not (directory / "ETH-USD.json").is_file():
        pytest.skip("no frozen models in this checkout")
    eth = read_artifact(directory / "ETH-USD.json")
    with pytest.raises(PaperModelError) as error:
        load_paper_model(eth, product="BTC-USD")
    assert "refusing to use one instrument" in str(error.value)


@pytest.mark.ml
@pytest.mark.parametrize("spelling", ["BTC-USD", "btc-usd", "coinbase:BTC-USD",
                                      "BTCUSD"])
def test_a_matching_model_loads_under_any_spelling(spelling):
    from scripts.trading_lab.paper_model import load_paper_model, read_artifact

    directory = REPO_ROOT / "data/models/paper_v1"
    if not (directory / "BTC-USD.json").is_file():
        pytest.skip("no frozen models in this checkout")
    assert load_paper_model(read_artifact(directory / "BTC-USD.json"),
                            product=spelling) is not None


@pytest.mark.ml
def test_loading_a_model_for_an_unregistered_product_fails_closed():
    from scripts.trading_lab.paper_model import (
        PaperModelError, load_paper_model, read_artifact)

    directory = REPO_ROOT / "data/models/paper_v1"
    if not (directory / "BTC-USD.json").is_file():
        pytest.skip("no frozen models in this checkout")
    with pytest.raises(PaperModelError):
        load_paper_model(read_artifact(directory / "BTC-USD.json"),
                         product="SOL-USD")


# --- cross-product: the economic backtest ---------------------------------


def _targets(count):
    from decimal import Decimal
    from datetime import datetime, timedelta, timezone

    from scripts.trading_lab.risk_engine import (
        RISK_SPEC_V1, PositionTarget, PositionTargetSeries)
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    grid = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    targets = tuple(
        PositionTarget(
            timestamp=(grid + hour * index).isoformat(),
            side="LONG" if index % 2 == 0 else "FLAT",
            target_exposure=Decimal("1") if index % 2 == 0 else Decimal("0"),
            signal_strength=Decimal("1"),
            raw_target_exposure=Decimal("1") if index % 2 == 0 else Decimal("0"),
            risk_scale=Decimal("1"), risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
            source_signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
            source_signal_decision_hash="0" * 64, reason="test")
        for index in range(count))
    return PositionTargetSeries(risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
                                targets=targets)


def _bars(count, base):
    from datetime import datetime, timedelta, timezone

    grid = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    return [{"bar_open_at": (grid + hour * index).isoformat(),
             "open": str(base + index), "high": str(base + index + 5),
             "low": str(base + index - 5), "close": str(base + index),
             "volume": "1"} for index in range(count)]


def _spec(product="BTC-USD"):
    from scripts.trading_lab.economic_backtest import (
        EXECUTION_SPEC_V1, EconomicBacktestSpec)
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    return EconomicBacktestSpec(
        protocol_version="trading-lab.economic-backtest.v1", product=product,
        timeframe="1h", source_benchmark_protocol="probe",
        source_benchmark_spec_hash="0" * 64, source_benchmark_results_hash="0" * 64,
        signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
        risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
        execution_spec_hash=EXECUTION_SPEC_V1.execution_spec_hash,
        market_corpus_spec_hash="0" * 64, market_corpus_content_hash="0" * 64,
        result_schema_version="v1")


def test_btc_targets_against_eth_bars_are_refused():
    """It ran, produced a different final equity, and labelled it BTC-USD."""
    from scripts.trading_lab.economic_backtest import run_economic_backtest
    from scripts.trading_lab.identity import InstrumentMismatchError

    with pytest.raises(InstrumentMismatchError) as error:
        run_economic_backtest(product="BTC-USD", targets=_targets(30),
                              bars=_bars(40, 3000), spec=_spec(),
                              signal_series_hash="0" * 64, bars_product="ETH-USD")
    assert "market bars" in str(error.value)


def test_a_matching_backtest_still_runs_and_is_unaffected():
    from scripts.trading_lab.economic_backtest import run_economic_backtest

    result = run_economic_backtest(
        product="BTC-USD", targets=_targets(30), bars=_bars(40, 60000),
        spec=_spec(), signal_series_hash="0" * 64, bars_product="BTC-USD")
    assert result.metrics.final_equity > 0
    assert len(result.fills) == 30


def test_the_backtest_accepts_aliases_on_either_side():
    from scripts.trading_lab.economic_backtest import run_economic_backtest

    canonical = run_economic_backtest(
        product="BTC-USD", targets=_targets(30), bars=_bars(40, 60000),
        spec=_spec(), signal_series_hash="0" * 64, bars_product="BTC-USD")
    aliased = run_economic_backtest(
        product="BTC-USD", targets=_targets(30), bars=_bars(40, 60000),
        spec=_spec(), signal_series_hash="0" * 64,
        bars_product="coinbase:btc_usd")
    assert aliased.metrics.final_equity == canonical.metrics.final_equity


def test_the_bars_product_has_no_default():
    """A default of None would leave the hole open for every forgetful caller."""
    import inspect

    from scripts.trading_lab.economic_backtest import run_economic_backtest

    parameter = inspect.signature(run_economic_backtest).parameters["bars_product"]
    assert parameter.default is inspect.Parameter.empty


def test_an_unregistered_backtest_product_fails_closed():
    from scripts.trading_lab.economic_backtest import run_economic_backtest
    from scripts.trading_lab.identity import InstrumentMismatchError

    with pytest.raises(InstrumentMismatchError):
        run_economic_backtest(product="SOL-USD", targets=_targets(30),
                              bars=_bars(40, 60000), spec=_spec("SOL-USD"),
                              signal_series_hash="0" * 64, bars_product="SOL-USD")


# --- cross-product: paper ingestion ---------------------------------------


@pytest.mark.ml
def test_a_candle_from_another_market_is_refused_when_the_caller_knows():
    from scripts.trading_lab.identity import InstrumentMismatchError
    from scripts.trading_lab.instruments import Timeframe

    engine, history = _paper_engine()
    row = dict(history[-1])
    with pytest.raises(InstrumentMismatchError):
        engine.ingest_candle("BTC-USD", row, now="2026-08-11T03:00:00Z",
                             row_product="ETH-USD")
    assert Timeframe.parse("1h").label == "1h"      # legacy label untouched


@pytest.mark.ml
def test_a_product_the_session_never_registered_is_refused_clearly():
    from scripts.trading_lab.paper_engine import PaperEngineError

    engine, history = _paper_engine()
    with pytest.raises(PaperEngineError) as error:
        engine.ingest_candle("btc-usd", dict(history[-1]),
                             now="2026-08-11T03:00:00Z")
    assert "not one of this session's products" in str(error.value)


def _paper_engine():
    """A minimal single-product session over synthetic bars."""
    import tempfile
    from datetime import datetime, timedelta, timezone

    from scripts.trading_lab import paper_engine as engine_module
    from scripts.trading_lab import paper_model as model_module
    from scripts.trading_lab.paper_event_store import PaperEventStore

    grid = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    rows = []
    for index in range(400):
        close = 20000 + (index * 7) % 53 - (index % 9) * 2
        rows.append({"bar_open_at": (grid + hour * index).isoformat(),
                     "open": str(close), "high": str(close + 2),
                     "low": str(close - 2), "close": str(close), "volume": "1.0"})
    series = engine_module.series_from_rows(rows, product="BTC-USD")
    artifact = model_module.train_paper_model(series, product="BTC-USD")
    model = model_module.load_paper_model(artifact, product="BTC-USD")
    directory = pathlib.Path(tempfile.mkdtemp())
    store = PaperEventStore(directory / "paper.sqlite")
    spec = engine_module.build_session_spec({"BTC-USD": model},
                                            products=("BTC-USD",))
    engine = engine_module.PaperEngine(store=store, models={"BTC-USD": model},
                                       session_id="s1", session_spec=spec)
    engine.seed_history("BTC-USD", rows[:-1])
    engine.start(now="2026-08-11T02:00:00Z")
    return engine, rows


# --- timeframe: one duration, one identity --------------------------------


@pytest.mark.parametrize("left,right", [("60m", "1h"), ("120m", "2h"),
                                        ("1h", "1h")])
def test_one_duration_has_one_timeframe_identity(left, right):
    from scripts.trading_lab.instruments import Timeframe

    assert Timeframe.parse(left) == Timeframe.parse(right)
    assert Timeframe.parse(left).label == Timeframe.parse(right).label


def test_hours_are_not_folded_into_days():
    """24h equals 1d only on a market that never closes."""
    from scripts.trading_lab.instruments import Timeframe

    assert Timeframe.parse("24h") != Timeframe.parse("1d")
    assert Timeframe.parse("24h").label == "24h"


@pytest.mark.parametrize("unsupported", ["3600", "1y", "1w", "0h", "", "hour",
                                         None, 3600])
def test_an_unsupported_timeframe_never_falls_back_to_one_hour(unsupported):
    from scripts.trading_lab.instruments import InstrumentError, Timeframe

    with pytest.raises(InstrumentError):
        Timeframe.parse(unsupported)


def test_the_legacy_timeframe_labels_are_unchanged():
    from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
    from scripts.trading_lab.instruments import Timeframe

    for label in TIMEFRAME_DURATIONS:
        assert Timeframe.parse(label).label == label


# --- provider identity ----------------------------------------------------


@pytest.mark.parametrize("unknown", ["coinbase", "Coinbase-Public-V1",
                                     "coinbase-public", "COINBASE-PUBLIC-V1",
                                     "some-broker", "", None])
def test_a_provider_lookup_never_guesses(unknown):
    """No substring match, no case folding, no fallback to Coinbase."""
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1, RegistryError

    with pytest.raises(RegistryError):
        PROVIDERS_V1.resolve(unknown)


def test_a_venue_is_not_a_provider():
    """coinbase the venue and coinbase-public-v1 the provider are different."""
    from scripts.trading_lab.instrument_registry import (
        BTC_USD, PROVIDERS_V1, RegistryError)

    from scripts.trading_lab.instrument_registry import CATALOGUE_V1

    assert BTC_USD.instrument_id.venue == "coinbase"
    assert PROVIDERS_V1.ids() == (
        "coinbase-public-v1", "massive-stocks-historical-v1")
    with pytest.raises(RegistryError):
        PROVIDERS_V1.resolve(BTC_USD.instrument_id.venue)
    # The equity case is the one that actually tempts a shortcut: Massive
    # serves AAPL but AAPL is listed on Nasdaq, so the identity is xnas:AAPL
    # and never massive:AAPL. A provider is where bars come from; a venue is
    # where the instrument trades, and conflating them would rename every
    # instrument the day a second vendor was added.
    venues = {spec.instrument_id.venue for spec in CATALOGUE_V1.all()}
    assert venues == {"coinbase", "xnas"}
    assert not venues & set(PROVIDERS_V1.ids())
    for venue in venues:
        with pytest.raises(RegistryError):
            PROVIDERS_V1.resolve(venue)
        assert not CATALOGUE_V1.get(f"{venue}:MASSIVE")


# --- registry uniqueness --------------------------------------------------


def test_no_two_registered_instruments_share_a_canonical_identity():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    canonical = [spec.instrument_id.canonical for spec in INSTRUMENTS_V1.all()]
    legacy = [spec.legacy_product_id for spec in INSTRUMENTS_V1.all()]
    hashes = [spec.instrument_spec_hash for spec in INSTRUMENTS_V1.all()]
    assert len(set(canonical)) == len(canonical)
    assert len(set(legacy)) == len(legacy)
    assert len(set(hashes)) == len(hashes)


def test_normalisation_never_merges_two_legitimate_instruments():
    """Canonicalisation that fused two real markets would be worse than none."""
    from scripts.trading_lab.instruments import normalize_symbol

    assert normalize_symbol("BTC-USD") != normalize_symbol("ETH-USD")
    assert normalize_symbol("BTC-USDT") != normalize_symbol("BTC-USD")
    # a separator-less symbol whose split is ambiguous must not collide with a
    # different, explicitly separated one
    assert normalize_symbol("ABC-DEF") != normalize_symbol("ABCD-EF")


def test_an_ambiguous_separatorless_symbol_does_not_become_a_registered_one():
    from scripts.trading_lab.identity import IdentityError, resolve_instrument

    # "ABCDEF" has no known quote suffix, so it cannot be split and must not
    # be resolved to anything.
    with pytest.raises(IdentityError):
        resolve_instrument("ABCDEF")


# --- cursors bind to identity ---------------------------------------------


def test_a_cursor_from_one_product_is_refused_by_another():
    from scripts.trading_lab.app_api.contracts import AppApiError
    from scripts.trading_lab.app_api.pagination import decode_cursor, encode_cursor

    cursor = encode_cursor(endpoint="market_candles", product="BTC-USD",
                           last_timestamp="2026-01-01T00:00:00+00:00",
                           query={"limit": 200})
    assert decode_cursor(cursor, endpoint="market_candles", product="BTC-USD",
                         query={"limit": 200})
    with pytest.raises(AppApiError):
        decode_cursor(cursor, endpoint="market_candles", product="ETH-USD",
                      query={"limit": 200})
    with pytest.raises(AppApiError):
        decode_cursor(cursor, endpoint="backtest_fills", product="BTC-USD",
                      query={"limit": 200})
    with pytest.raises(AppApiError):
        decode_cursor(cursor, endpoint="market_candles", product="BTC-USD",
                      query={"limit": 100})


def test_a_cursor_is_bound_to_the_canonical_spelling_the_api_accepts():
    """The API resolves before issuing, so a cursor never carries an alias."""
    from scripts.trading_lab.app_api.contracts import AppApiError
    from scripts.trading_lab.app_api.pagination import decode_cursor, encode_cursor

    cursor = encode_cursor(endpoint="market_candles", product="BTC-USD",
                           last_timestamp="2026-01-01T00:00:00+00:00", query={})
    for alias in ("btc-usd", "BTCUSD", "coinbase:BTC-USD"):
        with pytest.raises(AppApiError):
            decode_cursor(cursor, endpoint="market_candles", product=alias,
                          query={})


@pytest.mark.parametrize("malformed", ["", "not-base64!!", "YWJj", "x" * 500])
def test_a_malformed_cursor_fails_closed(malformed):
    from scripts.trading_lab.app_api.contracts import AppApiError
    from scripts.trading_lab.app_api.pagination import decode_cursor

    with pytest.raises(AppApiError):
        decode_cursor(malformed, endpoint="market_candles", product="BTC-USD",
                      query={})


# --- the API product gate -------------------------------------------------


def test_the_api_requires_the_canonical_spelling_and_says_so():
    from scripts.trading_lab.app_api.contracts import NotFoundError
    from scripts.trading_lab.app_api.service import AppService

    service = AppService(REPO_ROOT / "data/crypto")
    assert service._require_product("BTC-USD") == "BTC-USD"
    for alias in ("btc-usd", "BTCUSD", "coinbase:BTC-USD", " BTC-USD"):
        with pytest.raises(NotFoundError):
            service._require_product(alias)


@pytest.mark.parametrize("unknown", ["SOL-USD", "AAPL", "", None, 42,
                                     "../../etc/passwd"])
def test_the_api_product_gate_has_no_fallback(unknown):
    from scripts.trading_lab.app_api.contracts import NotFoundError
    from scripts.trading_lab.app_api.service import AppService

    service = AppService(REPO_ROOT / "data/crypto")
    with pytest.raises(NotFoundError):
        service._require_product(unknown)


# --- the scan: find NEW dangerous boundaries, not every string ------------
#
# The point is not to ban `==` in Python. It is to make a *new* raw-string
# identity comparison impossible to add without either fixing it or writing
# it down here with a reason. Every entry below was reviewed.

ALLOWED_RAW_PRODUCT_COMPARISONS = {
    # The API gate itself. The comparison IS the rule: the caller's spelling
    # must equal the registry-resolved canonical id, or the request is
    # refused. Both sides are canonical by the time they meet.
    "service.py": "the canonical-spelling requirement in _require_product",
    # The closed benchmark contract: a fixed internal pair checked against a
    # fixed internal list. No external input reaches either side.
    "real_benchmark.py": "internal BENCHMARK_PRODUCTS coverage check",
    "real_benchmark_v2.py": "internal BENCHMARK_PRODUCTS coverage check",
    # The guard's fast path: an exact-match shortcut that runs BEFORE the
    # canonical comparison it exists to perform, and changes no answer.
    "protected_holdout.py": "fast path preceding canonical comparison",
    # The Coinbase adapter's own product table. This IS the venue's spelling,
    # and resolving it through the registry would be circular.
    "coinbase_candles.py": "the provider's own product table",
    # Class E slot membership: which products THIS session registered. Both
    # sides are the session's own keys, and the check is fail-closed.
    "paper_engine.py": "session slot membership and replay filtering",
}


def _product_comparison_sites():
    """Modules comparing a product-shaped value with a raw string operator."""
    import ast

    root = REPO_ROOT / "scripts" / "trading_lab"
    found = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                          # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            names = {left.id for left in [node.left] if isinstance(left, ast.Name)}
            names |= {getattr(left, "attr", "") for left in [node.left]}
            if not ({"product", "product_id", "symbol"} & names):
                continue
            if any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn))
                   for op in node.ops):
                found.setdefault(path.name, 0)
                found[path.name] += 1
    return found


def test_every_raw_product_comparison_is_reviewed_and_justified():
    """A new one must be fixed, or added here with a reason. Not ignored."""
    sites = _product_comparison_sites()
    undocumented = sorted(set(sites) - set(ALLOWED_RAW_PRODUCT_COMPARISONS))
    assert not undocumented, (
        f"new raw product comparison(s) in {undocumented}: route the value "
        "through scripts.trading_lab.identity, or document why it is safe in "
        "ALLOWED_RAW_PRODUCT_COMPARISONS")


def test_the_allowlist_has_no_stale_entries():
    """An entry that no longer describes real code is a comment pretending to
    be a control."""
    sites = _product_comparison_sites()
    stale = sorted(set(ALLOWED_RAW_PRODUCT_COMPARISONS) - set(sites))
    assert not stale, f"allowlist mentions modules with no such comparison: {stale}"


def test_no_module_normalises_a_digest():
    """Case folding or stripping a hash is how an integrity check stops
    checking anything."""
    import re

    root = REPO_ROOT / "scripts" / "trading_lab"
    offenders = []
    pattern = re.compile(
        r"(hash|digest|sha256)\w*\s*\.\s*(lower|upper|strip|casefold)\(")
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path.name)
    assert not offenders, f"digest normalisation in {offenders}"


def test_no_module_derives_a_filesystem_path_from_an_external_identity():
    """The static layer resolves and contains; nothing else may build a path
    out of a product, session or provider id."""
    import re

    root = REPO_ROOT / "scripts" / "trading_lab"
    pattern = re.compile(r"(Path|open)\([^)]*\b(product|session_id|provider_id)\b")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    # paper_shadow_cli composes <model_dir>/<product>.json from a CLI argument
    # that the registry has already constrained; it is the only one.
    assert offenders in ([], ["paper_shadow_cli.py"]), offenders


# --- Phase 6D: the new identities ------------------------------------------
#
# The audit is replayed against everything Phase 6D introduced. The rule that
# matters has not changed: semantic identities are canonicalised, opaque ones
# are byte-exact, and an unknown is never equal to anything -- including
# another unknown.


def test_a_calendar_id_is_exact_and_never_canonicalised():
    """A calendar is a closed set, not a name to be parsed leniently."""
    from scripts.trading_lab.trading_calendar import (
        TradingCalendarError, get_calendar, known_calendars)

    assert "US_EQUITY_REGULAR" in known_calendars()
    for alias in ("us_equity_regular", " US_EQUITY_REGULAR ", "US-EQUITY-REGULAR",
                  "USEquityRegular", "XNYS", "NASDAQ", "NYSE", "XNAS"):
        with pytest.raises(TradingCalendarError):
            get_calendar(alias)
    # Exactly one spelling works.
    assert get_calendar("US_EQUITY_REGULAR").calendar_id == "US_EQUITY_REGULAR"


def test_an_adjustment_policy_is_a_closed_set_with_no_default():
    from scripts.trading_lab.equity_market import (
        EquityMarketError, require_adjustment_policy)

    assert require_adjustment_policy("RAW") == "RAW"
    assert require_adjustment_policy("SPLIT_ADJUSTED") == "SPLIT_ADJUSTED"
    for bad in ("raw", " RAW", "ADJUSTED", "", None, "TOTAL_RETURN"):
        with pytest.raises(EquityMarketError):
            require_adjustment_policy(bad)


def test_a_vendor_exchange_code_is_mapped_never_inferred():
    """An unmapped code names no venue, and a wrong venue names another market."""
    from scripts.trading_lab.massive_provider import (
        MassiveProviderError, parse_reference_ticker)

    row = {"ticker": "AAPL", "primary_exchange": "XNAS", "type": "CS",
           "currency_name": "USD", "name": "Apple Inc."}
    assert parse_reference_ticker(row).venue == "xnas"
    # Case is folded on the way in, so "xnas" is the same code as "XNAS" --
    # that is class A behaviour and correct. What must never happen is an
    # unlisted code resolving to some venue anyway.
    assert parse_reference_ticker(
        {**row, "primary_exchange": "xnas"}).venue == "xnas"
    for code in ("Nasdaq Global Select", "XLON", "UNKNOWN", "", "XNA", "XNASS"):
        with pytest.raises(MassiveProviderError):
            parse_reference_ticker({**row, "primary_exchange": code})


def test_the_calendar_spec_hash_is_opaque_and_byte_exact():
    pytest.importorskip("pandas_market_calendars")
    from scripts.trading_lab.equity_calendar import US_EQUITY_REGULAR_SPEC
    from scripts.trading_lab.identity import IdentityError, require_exact_digest

    digest = US_EQUITY_REGULAR_SPEC.spec_hash
    assert require_exact_digest(digest, field="calendar_spec_hash") == digest
    for mangled in (digest.upper(), f" {digest}", f"{digest}\n", digest[:63]):
        with pytest.raises(IdentityError):
            require_exact_digest(mangled, field="calendar_spec_hash")


def test_an_equity_instrument_id_accepts_the_same_aliases_and_no_more():
    from scripts.trading_lab.instrument_registry import CATALOGUE_V1

    for spelling in ("xnas:AAPL", "XNAS:AAPL", " xnas:aapl ", "xnas:aapl"):
        assert CATALOGUE_V1.resolve(spelling).canonical_id == "xnas:AAPL"
    # A bare symbol resolves through the default venue, which is Coinbase --
    # so it must NOT silently become the equity.
    for wrong in ("AAPL", "massive:AAPL", "nasdaq:AAPL", "xnys:AAPL"):
        assert CATALOGUE_V1.get(wrong) is None, wrong


def test_the_holdout_wall_is_untouched_by_the_new_markets():
    """Phase 6D adds markets. It does not look at the reserved window."""
    from scripts.trading_lab.protected_holdout import (
        PROTECTED_WINDOW_V1, embargo_state)

    assert PROTECTED_WINDOW_V1.holdout_hash == (
        "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85")
    assert PROTECTED_WINDOW_V1.observed is False
    assert PROTECTED_WINDOW_V1.single_use is True
    assert sorted(PROTECTED_WINDOW_V1.products) == ["BTC-USD", "ETH-USD"]
    # The guard covers the products it always covered. The equities are not
    # among them because no equity data exists to protect -- and asking about
    # one must not quietly report the crypto window's state as if it applied.
    from datetime import datetime, timezone

    inside = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for product in ("BTC-USD", "ETH-USD"):
        assert embargo_state(product, now=inside)["embargoed"] is True
    for product in ("xnas:AAPL", "xnas:QQQ"):
        assert embargo_state(product, now=inside)["embargoed"] is False
