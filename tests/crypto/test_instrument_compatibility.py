"""Phase 6A: the identity layer is additive. Nothing frozen moved.

A migration that renames how markets are identified is exactly the kind of
change that silently restates a hash. Every committed result, benchmark spec
and recorded event carries 'BTC-USD'; if the new layer had rewritten those to
'coinbase:BTC-USD', every artefact would still verify against itself and none
would match what was published.

So this file pins the boundary from both sides: the frozen values are
literals, and the new layer is required to bridge to them rather than replace
them.

Marked `ml` only where a check reaches through paper_engine or paper_model,
which import the model stack. The rest stays in the core suite.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data/crypto"


# --- frozen specification hashes -------------------------------------------


def test_the_signal_specification_hash_is_unchanged():
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    assert SIGNAL_SPEC_V1.spec_hash == (
        "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939")


def test_the_risk_specification_hash_is_unchanged():
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1

    assert RISK_SPEC_V1.risk_spec_hash == (
        "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad")


def test_the_execution_specification_hash_is_unchanged():
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1

    assert EXECUTION_SPEC_V1.execution_spec_hash == (
        "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb")


def test_the_protected_window_hash_is_unchanged():
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    assert PROTECTED_WINDOW_V1.holdout_hash == (
        "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85")


@pytest.mark.ml
def test_the_paper_specification_hashes_are_unchanged():
    from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1

    assert PAPER_MODEL_SPEC_V1.paper_model_spec_hash == (
        "830f52271af4eba887086d6b00885cc34f5d13562dae24cad45986a86466e033")


# --- committed artefacts ---------------------------------------------------


def _manifest(name: str):
    path = DATA / name / "manifest.json"
    if not path.is_file():
        pytest.skip(f"{name} is not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset", [
    "benchmark_results_v1", "benchmark_results_v2", "economic_backtest_v1",
])
def test_committed_artefacts_still_key_products_by_the_legacy_string(dataset):
    """Rewriting these keys would invalidate every published number."""
    manifest = _manifest(dataset)
    products = sorted(manifest["products"])
    assert products == ["BTC-USD", "ETH-USD"], products
    for product in products:
        assert ":" not in product, "an artefact was rewritten to a canonical id"


# Read from the committed manifests and pinned here as literals. A test that
# recomputes what it checks proves only that the file is self-consistent.
FROZEN_RESULT_HASHES = {
    ("benchmark_results_v1", "benchmark_results_hash"): {
        "BTC-USD": "0343014d81c0cef7ba65ec5a64d326f55bd92cc2d139596fd79fc32eb19902ae",
        "ETH-USD": "db4a0e8198e8c650a87d918b759da6cf40aa8865a95a918ff3fc04188d92062d",
    },
    ("benchmark_results_v2", "benchmark_results_hash"): {
        "BTC-USD": "e810c0e24ea979310cb987df0c6a22ec15d137b913e0d1a13379dead5ac03beb",
        "ETH-USD": "5421de013f27add40d2232da46f7ed371f850c2c66014884a5bf5e873816bd39",
    },
    ("economic_backtest_v1", "economic_results_hash"): {
        "BTC-USD": "4617c6151da9cb6560299149047e6d7954c81e9699b16b044afb48991551bb77",
        "ETH-USD": "47f0e8e324d4b58b28cf15e29ceb2bca25f434279ef6ac915a4bd4f429063bf1",
    },
}


@pytest.mark.parametrize("key", sorted(FROZEN_RESULT_HASHES))
def test_the_committed_result_hashes_are_bit_for_bit_unchanged(key):
    """The one check that would catch 6A silently restating a published result."""
    dataset, field = key
    manifest = _manifest(dataset)
    for product, expected in FROZEN_RESULT_HASHES[key].items():
        assert manifest["products"][product][field] == expected, \
            f"{dataset}/{product}/{field} changed"


def test_the_economic_results_still_carry_the_frozen_spec_hashes():
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    manifest = _manifest("economic_backtest_v1")
    assert manifest["execution_spec_hash"] == EXECUTION_SPEC_V1.execution_spec_hash
    assert manifest["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert manifest["risk_spec_hash"] == RISK_SPEC_V1.risk_spec_hash


def test_the_v2_manifest_still_reserves_the_confirmatory_holdout():
    manifest = _manifest("benchmark_results_v2")
    holdout = manifest["future_confirmatory_holdout"]
    assert holdout["range_start"] == "2026-09-01T00:00:00Z"
    assert holdout["range_end"] == "2026-11-30T23:00:00Z"
    assert manifest["confirmatory_result"] is False


# --- the bridge to the legacy identifiers ----------------------------------


def test_every_registered_instrument_bridges_to_its_legacy_product_id():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    assert INSTRUMENTS_V1.legacy_ids() == ("BTC-USD", "ETH-USD")
    for spec in INSTRUMENTS_V1.all():
        assert spec.legacy_product_id == spec.instrument_id.symbol
        assert ":" not in spec.legacy_product_id


def test_the_registry_agrees_with_the_api_supported_products():
    """Two lists of products would eventually disagree by one entry."""
    from scripts.trading_lab.app_api.contracts import SUPPORTED_PRODUCTS
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    assert sorted(SUPPORTED_PRODUCTS) == sorted(INSTRUMENTS_V1.legacy_ids())


def test_the_registry_agrees_with_the_coinbase_adapter_product_table():
    from scripts.trading_lab.coinbase_candles import PRODUCT_ASSETS
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

    for spec in INSTRUMENTS_V1.all():
        assert spec.legacy_product_id in PRODUCT_ASSETS
        assert PRODUCT_ASSETS[spec.legacy_product_id] == spec.legacy_asset


def test_the_registry_agrees_with_the_market_bar_asset_table():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1
    from scripts.trading_lab.market_bar import ASSETS

    for spec in INSTRUMENTS_V1.all():
        assert spec.legacy_asset in ASSETS


def test_the_registry_agrees_with_the_protected_window_products():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    for product in PROTECTED_WINDOW_V1.products:
        assert INSTRUMENTS_V1.resolve(product) is not None


def test_the_timeframe_type_reproduces_the_legacy_labels_exactly():
    from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
    from scripts.trading_lab.instruments import Timeframe

    for label, duration in TIMEFRAME_DURATIONS.items():
        frame = Timeframe.parse(label)
        assert frame.label == label, "a label used in artefacts was rewritten"
        assert frame.duration == duration


def test_the_calendar_reproduces_the_annualisation_the_engine_already_uses():
    """8760 is what the committed Sharpe figures were computed with."""
    from scripts.trading_lab.trading_calendar import CRYPTO_247_CALENDAR

    assert CRYPTO_247_CALENDAR.annualization_periods("1h") == 8760


# --- runtime compatibility -------------------------------------------------


def test_the_paper_event_log_still_stores_the_legacy_product_string():
    """The recorded log is hash-chained. Rewriting its keys would break it."""
    from scripts.trading_lab.paper_event_store import PaperEventStore
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        store = PaperEventStore(pathlib.Path(directory) / "paper.sqlite")
        store.append(session_id="s1", event_type="CANDLE_INGESTED",
                     event_at="2026-08-01T00:00:00Z", product="BTC-USD",
                     natural_key="2026-08-01T00:00:00Z", payload={"close": "1"})
        events = store.events(session_id="s1", limit=10)
        assert events[0].product == "BTC-USD"
        assert store.verify_chain(session_id="s1")["verified"] is True


def test_a_recorded_log_from_phase_5_still_reads_and_verifies():
    """The live runtime from 5D/5E, opened by 6A code, unchanged."""
    from scripts.trading_lab.paper_event_store import PaperEventStore

    database = REPO_ROOT / "var/trading_lab/paper_v1.sqlite"
    if not database.is_file():
        pytest.skip("no recorded runtime on this machine")
    store = PaperEventStore(database)
    sessions = store.sessions()
    assert sessions, "the recorded runtime has no session"
    for session in sessions:
        assert store.verify_chain(session_id=session)["verified"] is True


def test_no_schema_migration_was_applied_to_the_recorded_log():
    """6A adds a naming layer, not a column. Nothing rewrote the audit trail."""
    import sqlite3

    database = REPO_ROOT / "var/trading_lab/paper_v1.sqlite"
    if not database.is_file():
        pytest.skip("no recorded runtime on this machine")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        columns = [row[1] for row in connection.execute(
            "PRAGMA table_info(paper_events)")]
        products = [row[0] for row in connection.execute(
            "SELECT DISTINCT product FROM paper_events WHERE product IS NOT NULL")]
    finally:
        connection.close()
    assert "product" in columns and "instrument_id" not in columns
    assert all(":" not in product for product in products), products


@pytest.mark.ml
def test_the_shadow_engine_still_speaks_the_legacy_product_ids():
    from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1
    from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1

    assert PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash
    # the engine is addressed by the legacy id, and the registry supplies it
    for legacy in INSTRUMENTS_V1.legacy_ids():
        assert INSTRUMENTS_V1.resolve(legacy).legacy_product_id == legacy


@pytest.mark.ml
def test_the_frozen_model_artefacts_are_still_keyed_by_the_legacy_id():
    directory = REPO_ROOT / "data/models/paper_v1"
    if not directory.is_dir():
        pytest.skip("no frozen models in this checkout")
    names = {path.stem for path in directory.glob("*.json")} - {"manifest"}
    assert names == {"BTC-USD", "ETH-USD"}, names
