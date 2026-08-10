"""Benchmark contract V1: the frozen definition, tested without ever scoring it.

The dangerous move this file guards against is not a wrong number -- no number
exists yet. It is the contract quietly drifting: a feature reordered, a fold
geometry nudged, a corpus swapped underneath. Every test below pins the
declaration; the last few make it physically impossible for building the
contract to have looked at a model's output.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import pathlib
import tempfile
import shutil

import pytest

# Reading the candidates' ModelSpec hashes requires the model classes.
pytestmark = pytest.mark.ml


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = REPO_ROOT / "data" / "crypto"
HOUR = timedelta(hours=1)


@pytest.fixture
def benchmark():
    return importlib.import_module("scripts.trading_lab.real_benchmark")


@pytest.fixture
def spec(benchmark):
    return benchmark.build_benchmark_spec("BTC-USD", corpus_root=CORPUS_ROOT)


# --- the frozen declaration ------------------------------------------------


def test_the_feature_set_is_exactly_five_columns_in_the_declared_order(benchmark):
    """The order is contract, not presentation. It is never sorted."""
    assert benchmark.FEATURE_COLUMNS_V1 == (
        "return_1", "ema_12", "ema_26", "rsi_14", "atr_14")
    assert benchmark.FEATURE_COLUMNS_V1 != tuple(sorted(benchmark.FEATURE_COLUMNS_V1))
    assert [(f.column, f.indicator, f.parameters) for f in benchmark.FEATURE_SET_V1] == [
        ("return_1", "simple_return", ()),
        ("ema_12", "ema", (("period", 12),)),
        ("ema_26", "ema", (("period", 26),)),
        ("rsi_14", "rsi", (("period", 14),)),
        ("atr_14", "atr", (("period", 14),)),
    ]


def test_every_declared_feature_resolves_against_the_real_registry(benchmark):
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")
    for feature in benchmark.FEATURE_SET_V1:
        assert feature.indicator in dataset_module.INDICATOR_REGISTRY, feature.column
    # the primitive that used to be missing is the one being pinned here
    assert "simple_return" in dataset_module.INDICATOR_REGISTRY


def test_the_target_is_the_phase_two_contract_unchanged(benchmark, spec):
    assert benchmark.BENCHMARK_TIMEFRAME == "1h"
    assert benchmark.BENCHMARK_LABEL_HORIZON == 4
    assert benchmark.DATASET_CONFIG_V1.label.horizon == 4
    assert spec.label_name == "forward_return"
    assert spec.label_horizon == 4
    assert spec.timeframe == "1h"


def test_the_walk_forward_geometry_is_frozen_field_by_field(benchmark):
    config = benchmark.WALK_FORWARD_CONFIG_V1
    assert (config.min_train_rows, config.validation_rows, config.test_rows,
            config.step_rows, config.purge_rows) == (720, 168, 168, 168, 4)
    # step == test keeps out-of-sample windows from overlapping
    assert config.step_rows >= config.test_rows


def test_the_two_candidates_are_the_frozen_v1_configurations(benchmark, spec):
    models = importlib.import_module("scripts.trading_lab.models")
    assert benchmark.RIDGE_ALPHA_V1 == Decimal("1.0") == models.DEFAULT_RIDGE_ALPHA
    assert benchmark.XGBoostConfig() == models.XGBoostConfig()
    assert [name for name, _ in spec.candidates] == ["ridge", "xgboost"]
    ridge = models.RidgeRegressionPredictor(
        feature_columns=benchmark.FEATURE_COLUMNS_V1, alpha=Decimal("1.0"))
    assert dict(spec.candidates)["ridge"] == ridge.model_spec_hash


def test_the_selection_and_refit_versions_are_the_phase_three_ones(benchmark, spec):
    selection = importlib.import_module("scripts.trading_lab.model_selection")
    assert spec.selection_rule_version == selection.SELECTION_RULE_VERSION \
        == "validation-rank-ic-mae-rmse-v1"
    assert spec.refit_policy_version == selection.REFIT_POLICY_VERSION \
        == "refit-train-plus-validation-v1"
    assert spec.min_validation_observations == selection.MIN_VALIDATION_OBSERVATIONS == 3


def test_the_robustness_periods_are_calendar_quarters_fixed_in_advance(benchmark, spec):
    assert benchmark.ROBUSTNESS_BOUNDARIES_V1 == (
        "2025-08-01T00:00:00Z", "2025-11-01T00:00:00Z",
        "2026-02-01T00:00:00Z", "2026-05-01T00:00:00Z", "2026-08-01T00:00:00Z")
    assert spec.robustness_boundaries == benchmark.ROBUSTNESS_BOUNDARIES_V1
    parsed = [datetime.fromisoformat(edge.replace("Z", "+00:00"))
              for edge in spec.robustness_boundaries]
    assert parsed == sorted(parsed) and len(set(parsed)) == len(parsed)
    assert all(moment.day == 1 and moment.hour == 0 for moment in parsed)


def test_the_three_sensitivity_scenarios_vary_one_axis_only(benchmark, spec):
    assert [name for name, _ in benchmark.SENSITIVITY_SCENARIOS_V1] == [
        "central", "ridge_low", "ridge_high"]
    assert [Decimal(str(alpha)) for _, alpha in benchmark.SENSITIVITY_SCENARIOS_V1] == [
        Decimal("1.0"), Decimal("0.5"), Decimal("2.0")]
    xgboost_hashes = {entry["candidates"][1][1] for entry in spec.scenarios}
    assert len(xgboost_hashes) == 1              # the boosted config never moves
    ridge_hashes = {entry["candidates"][0][1] for entry in spec.scenarios}
    assert len(ridge_hashes) == 3                # only alpha varies
    for entry in spec.scenarios:
        assert "best" not in entry and "winner" not in entry


# --- binding and identity --------------------------------------------------


def test_the_contract_is_bound_to_the_versioned_corpus(benchmark, spec):
    capture = importlib.import_module("scripts.trading_lab.capture_market_history")
    manifest = capture.load_manifest(CORPUS_ROOT)
    assert spec.corpus_id == manifest["corpus_id"] == "coinbase_history_v1"
    assert spec.corpus_spec_hash == manifest["corpus_spec_hash"]
    assert spec.corpus_content_hash == manifest["corpus_content_hash"]
    assert len(spec.corpus_spec_hash) == len(spec.corpus_content_hash) == 64


def test_the_two_products_are_separate_experiments(benchmark):
    btc = benchmark.build_benchmark_spec("BTC-USD", corpus_root=CORPUS_ROOT)
    eth = benchmark.build_benchmark_spec("ETH-USD", corpus_root=CORPUS_ROOT)
    assert btc.benchmark_spec_hash != eth.benchmark_spec_hash
    # identical protocol, different subject: everything but the product matches
    assert btc.canonical()["walk_forward_config"] == eth.canonical()["walk_forward_config"]
    assert btc.features == eth.features
    assert btc.candidates == eth.candidates
    with pytest.raises(benchmark.RealBenchmarkError, match="not part of benchmark"):
        benchmark.build_benchmark_spec("SOL-USD", corpus_root=CORPUS_ROOT)


def test_the_spec_hash_is_deterministic_and_covers_the_whole_definition(benchmark, spec):
    from dataclasses import replace
    assert benchmark.build_benchmark_spec(
        "BTC-USD", corpus_root=CORPUS_ROOT).benchmark_spec_hash == spec.benchmark_spec_hash
    baseline = spec.benchmark_spec_hash
    assert len(baseline) == 64
    for field, value in (
        ("corpus_content_hash", "0" * 64),
        ("corpus_spec_hash", "0" * 64),
        ("label_horizon", 8),
        ("feature_columns", spec.feature_columns[::-1]),
        ("features", spec.features[1:]),
        ("robustness_boundaries", spec.robustness_boundaries[:-1]),
        ("candidates", spec.candidates[:1]),
        ("scenarios", spec.scenarios[:1]),
        ("dataset_config_hash", "0" * 64),
        ("selection_rule_version", "other-v9"),
        ("walk_forward_config", {**spec.walk_forward_config, "test_rows": 24}),
    ):
        assert replace(spec, **{field: value}).benchmark_spec_hash != baseline, field


def test_the_specification_carries_no_result_of_any_kind(benchmark, spec):
    """Structural, not a substring hunt.

    "mae" legitimately appears inside `validation-rank-ic-mae-rmse-v1`: that is
    the NAME of the rule, not a measurement. So the check is on shape -- no key
    in the payload is a result, and every metric word that does occur occurs
    only inside a declared version string.
    """
    payload = spec.canonical()

    keys = set()
    def walk(node):
        if isinstance(node, dict):
            keys.update(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(payload)
    for forbidden in ("rank_ic", "mae", "rmse", "score", "scores", "prediction",
                      "predictions", "winner", "fitted_model_hash", "results",
                      "metrics", "observations"):
        assert forbidden not in keys, forbidden

    version_strings = {spec.selection_rule_version, spec.metric_version,
                       spec.robustness_protocol_version, spec.protocol_version,
                       spec.refit_policy_version}
    def leaves(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from leaves(value)
        elif isinstance(node, list):
            for value in node:
                yield from leaves(value)
        else:
            yield node
    for leaf in leaves(payload):
        if isinstance(leaf, str) and any(word in leaf.lower()
                                         for word in ("rank_ic", "mae", "rmse")):
            assert leaf in version_strings, leaf

    for attribute in ("results", "metrics", "score", "predictions", "run"):
        assert not hasattr(spec, attribute), attribute
    assert not hasattr(benchmark, "run_benchmark")


def test_a_missing_or_mismatched_corpus_is_refused(benchmark, tmp_path):
    with pytest.raises(benchmark.RealBenchmarkError, match="corpus is unavailable"):
        benchmark.build_benchmark_spec("BTC-USD", corpus_root=tmp_path)


# --- geometry, built but never scored --------------------------------------


def _dataset_for(product, benchmark):
    capture = importlib.import_module("scripts.trading_lab.capture_market_history")
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    series_module = importlib.import_module("scripts.trading_lab.market_series")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    manifest = capture.load_manifest(CORPUS_ROOT)
    entry = next(e for e in manifest["products"] if e["product"] == product)
    rows = capture.load_canonical_rows(
        CORPUS_ROOT / capture.CORPUS_ID / entry["canonical_path"])
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        store = store_module.MarketDataStore(tmp / f"{product}.sqlite3")
        declared = manifest["capture_completed_at"]
        ingested = capture._iso(
            capture._parse_iso(declared, field="d") + timedelta(seconds=1))
        for start in range(0, len(rows), 300):
            store.ingest_coinbase_response(
                capture._coinbase_page(rows[start:start + 300]), product_id=product,
                timeframe="1h", available_at=declared, ingested_at=ingested)
        connection = store._connect()
        try:
            last = capture._parse_iso(rows[-1]["bar_open_at"], field="o")
            snapshot = snapshots._materialize_snapshot(
                connection, provider=capture.CORPUS_PROVIDER, product_id=product,
                timeframe="1h", range_start=rows[0]["bar_open_at"],
                range_end=capture._iso(last + HOUR),
                as_of=capture._iso(capture._parse_iso(ingested, field="i")
                                   + timedelta(seconds=1)))
            series = series_module.load_market_series(
                connection, snapshot_id=snapshot.snapshot_id)
        finally:
            connection.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return series, dataset_module.build_dataset(series, config=benchmark.DATASET_CONFIG_V1)


@pytest.mark.parametrize("product", ["BTC-USD", "ETH-USD"])
def test_the_frozen_geometry_actually_works_on_the_real_corpus(benchmark, product):
    """build_folds only. No fit, no predict, no evaluation, no score."""
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    series, dataset = _dataset_for(product, benchmark)
    assert len(series.points) == 8750
    assert len(series.missing_openings) == 10
    usable = walk_forward.usable_rows(dataset)
    assert len(usable) > 8000

    folds = walk_forward.build_folds(dataset, config=benchmark.WALK_FORWARD_CONFIG_V1)
    assert len(folds) >= 40
    for train, validation, test in folds:
        assert len(validation) >= benchmark.MIN_VALIDATION_OBSERVATIONS
        # every label window closes strictly before the next block opens
        assert max(datetime.fromisoformat(r.bar_open_at) + HOUR * 4 for r in train) \
            < min(datetime.fromisoformat(r.bar_open_at) for r in validation)
        assert max(datetime.fromisoformat(r.bar_open_at) + HOUR * 4 for r in validation) \
            < min(datetime.fromisoformat(r.bar_open_at) for r in test)
    stamps = [row.bar_open_at for _, _, test in folds for row in test]
    assert len(set(stamps)) == len(stamps)      # no out-of-sample overlap


# --- the anti-peek trap ----------------------------------------------------


def test_building_the_contract_is_incapable_of_observing_a_score(benchmark, monkeypatch):
    """Not a promise, a trap: every scoring surface explodes if touched.

    A grep proves the current text has no call. This proves the code path
    cannot make one, which is the property that has to survive future edits.
    """
    models = importlib.import_module("scripts.trading_lab.models")
    selection = importlib.import_module("scripts.trading_lab.model_selection")
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")

    def detonate(*args, **kwargs):
        raise AssertionError("the benchmark contract touched a scoring surface")

    for target, name in (
        (models.RidgeRegressionPredictor, "fit"),
        (models.RidgeRegressionPredictor, "predict"),
        (models.XGBoostRegressionPredictor, "fit"),
        (models.XGBoostRegressionPredictor, "predict"),
        (selection, "evaluate_selection"),
        (selection, "validate_candidates"),
        (walk_forward, "evaluate"),
        (walk_forward, "rank_ic"),
        (walk_forward, "mean_absolute_error"),
        (walk_forward, "root_mean_squared_error"),
    ):
        monkeypatch.setattr(target, name, detonate)

    prerequisites = benchmark.validate_benchmark_prerequisites(CORPUS_ROOT)
    assert prerequisites["features_resolvable"] is True
    for product in benchmark.BENCHMARK_PRODUCTS:
        spec = benchmark.build_benchmark_spec(product, corpus_root=CORPUS_ROOT)
        assert len(spec.benchmark_spec_hash) == 64
        assert spec.candidates                    # model identities still readable


def test_the_contract_module_never_calls_a_scoring_surface(benchmark):
    source = pathlib.Path(benchmark.__file__).read_text()
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    for forbidden in (".fit(", ".predict(", "evaluate_selection(", "evaluate(",
                      "rank_ic(", "mean_absolute_error(", "root_mean_squared_error(",
                      "analyse_robustness("):
        assert forbidden not in code, forbidden
