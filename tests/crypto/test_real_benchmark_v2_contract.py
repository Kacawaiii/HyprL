"""Benchmark contract V2, pre-registered and tested without producing a score.

Two properties matter most here. V1 must be provably untouched -- a frozen
experiment whose identity drifts is no longer evidence of anything. And V2 must
be honest about the corpus it will first run on: that data has already been
observed under V1, so any number it yields is exploratory by construction.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.ml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "data" / "crypto"
HOUR = timedelta(hours=1)
V1_HASHES = {"BTC-USD": "dd7c474b34489ad3ca0c8ce906187821039fd9f0c50bff4a03b2da8cd20332a6",
             "ETH-USD": "3d418bbbe6b0a418b2821fd44cbaa09273f8457f7bc08ee5c9c95d43956ad5f9"}


@pytest.fixture
def v2():
    return importlib.import_module("scripts.trading_lab.real_benchmark_v2")


@pytest.fixture
def v1():
    return importlib.import_module("scripts.trading_lab.real_benchmark")


@pytest.fixture
def spec(v2):
    return v2.build_benchmark_v2_spec("BTC-USD", corpus_root=CORPUS)


# --- V1 is immutable -------------------------------------------------------


def test_the_v1_contract_is_completely_unchanged(v1):
    """A frozen experiment whose identity drifts stops being evidence."""
    for product, expected in V1_HASHES.items():
        assert v1.build_benchmark_spec(
            product, corpus_root=CORPUS).benchmark_spec_hash == expected
    assert v1.FEATURE_COLUMNS_V1 == ("return_1", "ema_12", "ema_26", "rsi_14", "atr_14")
    assert v1.BENCHMARK_LABEL_HORIZON == 4
    config = v1.WALK_FORWARD_CONFIG_V1
    assert (config.min_train_rows, config.validation_rows, config.test_rows,
            config.step_rows, config.purge_rows) == (720, 168, 168, 168, 4)


def test_the_v1_recorded_results_are_untouched():
    import json
    results = CORPUS / "benchmark_results_v1"
    manifest = json.loads((results / "manifest.json").read_bytes().decode("utf-8"))
    for product in ("BTC-USD", "ETH-USD"):
        stored = json.loads((results / f"{product}.json").read_bytes().decode("utf-8"))
        assert stored["benchmark_spec_hash"] == V1_HASHES[product]
        assert stored["benchmark_results_hash"] == \
            manifest["products"][product]["benchmark_results_hash"]
    btc = json.loads((results / "BTC-USD.json").read_bytes().decode("utf-8"))
    eth = json.loads((results / "ETH-USD.json").read_bytes().decode("utf-8"))
    # the numbers V1 produced, exactly as recorded, including the negative quarters
    assert btc["selection"]["global_test_metrics"]["rank_ic"].startswith("-0.00304")
    assert eth["selection"]["global_test_metrics"]["rank_ic"].startswith("0.00417")
    assert len(btc["robustness"]["periods"]) == len(eth["robustness"]["periods"]) == 4


def test_the_frozen_one_bar_return_primitive_keeps_its_identity(v1, v2):
    """V2 adds a parameterised return family; V1's parameterless one is untouched."""
    indicators = importlib.import_module("scripts.trading_lab.market_indicators")
    assert v1.FEATURE_SET_V1[0].indicator == "simple_return"
    assert v1.FEATURE_SET_V1[0].parameters == ()
    assert v2.FEATURE_SET_V2[0] == v1.FEATURE_SET_V1[0]
    assert indicators.simple_return.__name__ == "simple_return"


# --- the V2 declaration ----------------------------------------------------


def test_the_v2_feature_set_is_exactly_six_columns_in_the_declared_order(v2):
    assert v2.FEATURE_COLUMNS_V2 == (
        "return_1", "return_4", "return_12", "ema_spread_12_26", "rsi_14", "atr_pct_14")
    assert v2.FEATURE_COLUMNS_V2 != tuple(sorted(v2.FEATURE_COLUMNS_V2))
    assert [(f.column, f.indicator, f.parameters) for f in v2.FEATURE_SET_V2] == [
        ("return_1", "simple_return", ()),
        ("return_4", "return_over_period", (("period", 4),)),
        ("return_12", "return_over_period", (("period", 12),)),
        ("ema_spread_12_26", "ema_spread", (("fast_period", 12), ("slow_period", 26))),
        ("rsi_14", "rsi", (("period", 14),)),
        ("atr_pct_14", "atr_percent", (("period", 14),)),
    ]


def test_every_v2_indicator_resolves_against_the_real_registry(v2):
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")
    for feature in v2.FEATURE_SET_V2:
        assert feature.indicator in dataset_module.INDICATOR_REGISTRY, feature.column


def test_only_the_feature_representation_changed_between_v1_and_v2(v1, v2, spec):
    """The delta must be one thing, or the experiment measures nothing in particular."""
    v1_spec = v1.build_benchmark_spec("BTC-USD", corpus_root=CORPUS)
    assert spec.walk_forward_config == v1_spec.walk_forward_config
    assert spec.label_name == v1_spec.label_name
    assert spec.label_horizon == v1_spec.label_horizon == 4
    assert spec.timeframe == v1_spec.timeframe == "1h"
    assert spec.selection_rule_version == v1_spec.selection_rule_version
    assert spec.refit_policy_version == v1_spec.refit_policy_version
    assert spec.min_validation_observations == v1_spec.min_validation_observations
    assert spec.robustness_boundaries == v1_spec.robustness_boundaries
    assert [s["scenario_id"] for s in spec.scenarios] == \
           [s["scenario_id"] for s in v1_spec.scenarios]
    assert spec.corpus_content_hash == v1_spec.corpus_content_hash
    # ... and the features genuinely differ
    assert spec.feature_columns != v1_spec.feature_columns


def test_the_models_and_scenarios_are_the_v1_configurations(v2, spec):
    models = importlib.import_module("scripts.trading_lab.models")
    assert v2.RIDGE_ALPHA_V2 == Decimal("1.0") == models.DEFAULT_RIDGE_ALPHA
    assert v2.XGBoostConfig() == models.XGBoostConfig()
    assert [name for name, _ in spec.candidates] == ["ridge", "xgboost"]
    assert [s["scenario_id"] for s in spec.scenarios] == \
        ["central", "ridge_low", "ridge_high"]
    xgboost_hashes = {s["candidates"][1][1] for s in spec.scenarios}
    assert len(xgboost_hashes) == 1
    assert len({s["candidates"][0][1] for s in spec.scenarios}) == 3


# --- exploratory status and the future holdout -----------------------------


def test_the_v1_corpus_is_marked_spent_for_v2(v2, spec):
    """It already answered V1. It cannot also be V2's untouched holdout."""
    assert v2.V1_CORPUS_ROLE_FOR_V2 == "development/exploratory"
    assert spec.corpus_role == "development/exploratory"
    assert spec.confirmatory_holdout is False
    assert spec.canonical()["corpus"]["confirmatory_holdout"] is False


def test_the_future_confirmatory_holdout_is_registered_in_advance(v2, spec):
    holdout = spec.future_holdout
    assert holdout["range_start"] == "2026-09-01T00:00:00Z"
    assert holdout["range_end"] == "2026-11-30T23:00:00Z"
    assert holdout["products"] == ["BTC-USD", "ETH-USD"]
    assert holdout["timeframe"] == "1h"
    assert holdout["role"] == "confirmatory"
    assert holdout["captured"] is False        # nothing was downloaded
    assert holdout["single_use"] is True
    start = datetime.fromisoformat(holdout["range_start"].replace("Z", "+00:00"))
    corpus_end = datetime(2026, 7, 31, 23, tzinfo=timezone.utc)
    assert start > corpus_end                  # entirely outside the spent corpus
    assert not (CORPUS / holdout["holdout_id"]).exists()


def test_the_holdout_is_single_use_by_declaration(v2):
    assert "spent" in v2.CONFIRMATORY_HOLDOUT_V2["note"]
    assert "exploratory" in v2.CONFIRMATORY_HOLDOUT_V2["note"]


# --- identity --------------------------------------------------------------


def test_the_v2_spec_hashes_are_distinct_from_v1_and_from_each_other(v1, v2, spec):
    eth = v2.build_benchmark_v2_spec("ETH-USD", corpus_root=CORPUS)
    assert spec.benchmark_spec_hash != eth.benchmark_spec_hash
    for product in ("BTC-USD", "ETH-USD"):
        assert v2.build_benchmark_v2_spec(
            product, corpus_root=CORPUS).benchmark_spec_hash != V1_HASHES[product]
    assert spec.protocol_version == "trading-lab.real-benchmark.v2"
    assert v2.build_benchmark_v2_spec(
        "BTC-USD", corpus_root=CORPUS).benchmark_spec_hash == spec.benchmark_spec_hash


def test_the_v2_spec_hash_covers_the_whole_definition(v2, spec):
    baseline = spec.benchmark_spec_hash
    for field, value in (
        ("corpus_content_hash", "0" * 64),
        ("feature_columns", spec.feature_columns[::-1]),
        ("features", spec.features[1:]),
        ("label_horizon", 8),
        ("robustness_boundaries", spec.robustness_boundaries[:-1]),
        ("candidates", spec.candidates[:1]),
        ("scenarios", spec.scenarios[:1]),
        ("corpus_role", "confirmatory"),
        ("confirmatory_holdout", True),
        ("future_holdout", {**spec.future_holdout, "range_end": "2026-12-31T23:00:00Z"}),
        ("walk_forward_config", {**spec.walk_forward_config, "test_rows": 24}),
    ):
        assert replace(spec, **{field: value}).benchmark_spec_hash != baseline, field


def test_the_v2_specification_carries_no_result(v2, spec):
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
    for forbidden in ("rank_ic", "mae", "rmse", "score", "prediction", "predictions",
                      "winner", "fitted_model_hash", "results", "metrics"):
        assert forbidden not in keys, forbidden
    assert not hasattr(v2, "run_benchmark")
    assert not hasattr(spec, "results")


# --- geometry, built but never scored --------------------------------------


@pytest.mark.parametrize("product", ["BTC-USD", "ETH-USD"])
def test_the_v2_geometry_is_measured_without_any_model_being_fitted(v2, product,
                                                                    tmp_path, monkeypatch):
    """build_folds is allowed. fit, predict and every metric are booby-trapped."""
    models = importlib.import_module("scripts.trading_lab.models")
    selection = importlib.import_module("scripts.trading_lab.model_selection")
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    runner = importlib.import_module("scripts.trading_lab.run_real_benchmark")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    def detonate(*args, **kwargs):
        raise AssertionError("a V2 score was produced during contract validation")

    for target, name in ((models.RidgeRegressionPredictor, "fit"),
                         (models.RidgeRegressionPredictor, "predict"),
                         (models.XGBoostRegressionPredictor, "fit"),
                         (models.XGBoostRegressionPredictor, "predict"),
                         (selection, "evaluate_selection"),
                         (walk_forward, "evaluate"),
                         (walk_forward, "rank_ic"),
                         (walk_forward, "mean_absolute_error"),
                         (walk_forward, "root_mean_squared_error")):
        monkeypatch.setattr(target, name, detonate)

    spec = v2.build_benchmark_v2_spec(product, corpus_root=CORPUS)
    assert len(spec.benchmark_spec_hash) == 64

    series = runner.load_corpus_series(CORPUS, product=product,
                                       database_path=tmp_path / f"{product}.sqlite3")
    dataset = dataset_module.build_dataset(series, config=v2.DATASET_CONFIG_V2)
    rows = walk_forward.usable_rows(dataset)
    folds = walk_forward.build_folds(dataset, config=v2.WALK_FORWARD_CONFIG_V2)

    assert len(series.points) == 8750
    assert len(series.missing_openings) == 10
    assert len(rows) > 8000
    assert len(folds) >= 40
    for train, validation, test in folds:
        assert len(validation) >= v2.MIN_VALIDATION_OBSERVATIONS
        assert max(datetime.fromisoformat(r.bar_open_at) + HOUR * 4 for r in train) \
            < min(datetime.fromisoformat(r.bar_open_at) for r in validation)
        assert max(datetime.fromisoformat(r.bar_open_at) + HOUR * 4 for r in validation) \
            < min(datetime.fromisoformat(r.bar_open_at) for r in test)
    stamps = [row.bar_open_at for _, _, test in folds for row in test]
    assert len(set(stamps)) == len(stamps)


def test_the_contract_module_never_calls_a_scoring_surface(v2):
    source = pathlib.Path(v2.__file__).read_text()
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    for forbidden in (".fit(", ".predict(", "evaluate_selection(", "rank_ic(",
                      "mean_absolute_error(", "analyse_robustness("):
        assert forbidden not in code, forbidden
