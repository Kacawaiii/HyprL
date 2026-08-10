"""The frozen-benchmark runner, exercised on synthetic corpora only.

Not one test here produces a score from the real BTC-USD or ETH-USD corpus.
The real corpus is touched only for verification and geometry, exactly as the
mission allows before the runner is committed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
import ast
import importlib
import inspect
import json
import pathlib

import pytest

# The runner drives Ridge and XGBoost.
pytestmark = pytest.mark.ml


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REAL_CORPUS = REPO_ROOT / "data" / "crypto"
HOUR = timedelta(hours=1)
SYNTHETIC_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
SYNTHETIC_BARS = 1500


@pytest.fixture
def runner():
    return importlib.import_module("scripts.trading_lab.run_real_benchmark")


@pytest.fixture
def benchmark():
    return importlib.import_module("scripts.trading_lab.real_benchmark")


@pytest.fixture
def capture():
    return importlib.import_module("scripts.trading_lab.capture_market_history")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _synthetic_pages(product):
    """Deterministic, non-monotone candles: enough structure to fit something."""
    seed = 0 if product == "BTC-USD" else 7
    rows = []
    for index in range(SYNTHETIC_BARS):
        close = Decimal(200 + (index * (17 + seed)) % 53 - (index % 9) * 2)
        rows.append([int((SYNTHETIC_START + HOUR * index).timestamp()),
                     str(close - Decimal(2 + index % 4)), str(close + Decimal(1 + index % 5)),
                     str(close), str(close), "1.0"])
    pages = []
    for start in range(0, len(rows), 300):
        chunk = rows[start:start + 300]
        pages.append(json.dumps(list(reversed(chunk)), separators=(",", ":")).encode("utf-8"))
    return pages


@pytest.fixture
def synthetic_corpus(capture, tmp_path):
    """A tiny two-product corpus written through the real capture path."""
    queued = {product: iter(_synthetic_pages(product))
              for product in ("BTC-USD", "ETH-USD")}

    def fetch(url):
        product = "BTC-USD" if "BTC-USD" in url else "ETH-USD"
        return next(queued[product])

    capture.capture_corpus(tmp_path, products=("BTC-USD", "ETH-USD"),
                           range_start=_iso(SYNTHETIC_START),
                           range_end=_iso(SYNTHETIC_START + HOUR * (SYNTHETIC_BARS - 1)),
                           fetch=fetch, spacing_seconds=0)
    return tmp_path


def _synthetic_geometry(runner, benchmark, corpus, product, tmp_path):
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")
    series = runner.load_corpus_series(corpus, product=product,
                                       database_path=tmp_path / f"geo-{product}.sqlite3")
    dataset = dataset_module.build_dataset(series, config=benchmark.DATASET_CONFIG_V1)
    rows = walk_forward.usable_rows(dataset)
    folds = walk_forward.build_folds(dataset, config=benchmark.WALK_FORWARD_CONFIG_V1)
    return {
        "series_points": len(series.points),
        "missing_openings": len(series.missing_openings),
        "usable_rows": len(rows),
        "folds": len(folds),
        "min_effective_validation": min(len(v) for _, v, _ in folds),
        "oos_records": sum(len(t) for _, _, t in folds),
    }


def _run_synthetic(runner, benchmark, corpus, product, tmp_path, monkeypatch):
    geometry = _synthetic_geometry(runner, benchmark, corpus, product, tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_GEOMETRY", geometry)
    spec = benchmark.build_benchmark_spec(product, corpus_root=corpus)
    return runner.run_product_benchmark(corpus, product, spec.benchmark_spec_hash)


# --- the runner defines nothing of its own --------------------------------


def test_the_runner_exposes_no_tuning_control(runner):
    """A knob a caller can turn after seeing a number will eventually be turned."""
    parameters = inspect.signature(runner.run_product_benchmark).parameters
    assert list(parameters) == ["corpus_dir", "product", "expected_benchmark_spec_hash",
                                "database_path"]
    forbidden = ("alpha", "max_depth", "learning_rate", "features", "horizon",
                 "train", "n_estimators", "purge", "window", "period")
    for name in parameters:
        assert not any(word in name.lower() for word in forbidden), name


def test_the_runner_never_redefines_the_frozen_protocol(runner):
    """Every decision must arrive from the committed contract, not from here."""
    source = pathlib.Path(runner.__file__).read_text()
    tree = ast.parse(source)
    constructed = {node.func.id for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for forbidden in ("WalkForwardConfig", "FeatureDefinition", "DatasetConfig",
                      "LabelSpec"):
        assert forbidden not in constructed, forbidden
    # the contract's own objects are imported, not rebuilt
    for expected in ("WALK_FORWARD_CONFIG_V1", "DATASET_CONFIG_V1", "FEATURE_COLUMNS_V1",
                     "ROBUSTNESS_BOUNDARIES_V1", "SENSITIVITY_SCENARIOS_V1"):
        assert expected in source, expected


def test_the_expected_geometry_constant_is_the_frozen_measurement(runner):
    assert runner.EXPECTED_GEOMETRY == {
        "series_points": 8750, "missing_openings": 10, "usable_rows": 8663,
        "folds": 46, "min_effective_validation": 164, "oos_records": 7728,
    }


# --- the three refusals, all before any fit -------------------------------


def test_a_spec_hash_mismatch_stops_the_run_before_fitting(runner, benchmark,
                                                            synthetic_corpus, monkeypatch):
    models = importlib.import_module("scripts.trading_lab.models")

    def detonate(*args, **kwargs):
        raise AssertionError("a model was fitted despite a contract mismatch")

    monkeypatch.setattr(models.RidgeRegressionPredictor, "fit", detonate)
    monkeypatch.setattr(models.XGBoostRegressionPredictor, "fit", detonate)
    with pytest.raises(benchmark.RealBenchmarkError, match="does not match the "
                                                          "pre-registered"):
        runner.run_product_benchmark(synthetic_corpus, "BTC-USD", "0" * 64)


def test_a_geometry_mismatch_stops_the_run_before_fitting(runner, benchmark,
                                                          synthetic_corpus, monkeypatch):
    models = importlib.import_module("scripts.trading_lab.models")

    def detonate(*args, **kwargs):
        raise AssertionError("a model was fitted despite a geometry mismatch")

    monkeypatch.setattr(models.RidgeRegressionPredictor, "fit", detonate)
    monkeypatch.setattr(models.XGBoostRegressionPredictor, "fit", detonate)
    spec = benchmark.build_benchmark_spec("BTC-USD", corpus_root=synthetic_corpus)
    # the frozen constant describes the real corpus, not this synthetic one
    with pytest.raises(runner.RealBenchmarkRunError, match="does not match the frozen"):
        runner.run_product_benchmark(synthetic_corpus, "BTC-USD", spec.benchmark_spec_hash)


def test_a_corrupted_corpus_stops_the_run(runner, benchmark, synthetic_corpus, capture):
    spec = benchmark.build_benchmark_spec("BTC-USD", corpus_root=synthetic_corpus)
    path = synthetic_corpus / capture.CORPUS_ID / "BTC-USD" / "canonical.jsonl"
    data = bytearray(path.read_bytes())
    index = next(i for i, byte in enumerate(data) if chr(byte).isdigit())
    data[index] = ord("7") if chr(data[index]) != "7" else ord("6")
    path.write_bytes(bytes(data))
    with pytest.raises(capture.MarketHistoryCaptureError):
        runner.run_product_benchmark(synthetic_corpus, "BTC-USD", spec.benchmark_spec_hash)


def test_the_runner_never_reaches_for_the_network(runner, benchmark, synthetic_corpus,
                                                  tmp_path, monkeypatch):
    capture = importlib.import_module("scripts.trading_lab.capture_market_history")

    def explode(url):
        raise AssertionError("the benchmark runner attempted a network call")

    monkeypatch.setattr(capture, "_http_get", explode)
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    assert result.product == "BTC-USD"


# --- a real end-to-end run, on synthetic data -----------------------------


def test_the_runner_produces_a_complete_auditable_result(runner, benchmark,
                                                          synthetic_corpus, tmp_path,
                                                          monkeypatch):
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    assert result.geometry["folds"] >= 1
    assert result.oos_records
    assert result.selection["global_test_metrics"]["observations"] == len(result.oos_records)
    assert set(result.selection["selection_counts"]) <= {"ridge", "xgboost"}
    assert sum(result.selection["selection_counts"].values()) == result.geometry["folds"]
    assert len(result.selection["folds"]) == result.geometry["folds"]
    for fold in result.selection["folds"]:
        assert [entry["candidate_id"] for entry in fold["candidates"]] == ["ridge", "xgboost"]
        assert fold["selection_fit_hash"] != fold["final_fit_hash"]
        assert fold["selected_candidate_id"] in {"ridge", "xgboost"}
    # every prediction is kept, so the metrics can be recomputed by anyone
    assert all(set(record) == {"fold_index", "bar_open_at", "prediction",
                               "actual_forward_return"} for record in result.oos_records)
    assert [entry["scenario_id"] for entry in result.scenarios] == [
        "central", "ridge_high", "ridge_low"]
    assert result.robustness["trading_cost_analysis_available"] is False


def test_the_recorded_metrics_are_recomputable_from_the_records(runner, benchmark,
                                                                synthetic_corpus,
                                                                tmp_path, monkeypatch):
    """A result nobody can re-derive is a claim, not evidence."""
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    # With a single fold the global figure and the first fold's figure coincide,
    # so this check would pass even if the runner reported the wrong one.
    assert result.geometry["folds"] >= 3
    predictions = tuple(Decimal(r["prediction"]) for r in result.oos_records)
    actuals = tuple(Decimal(r["actual_forward_return"]) for r in result.oos_records)
    reported = result.selection["global_test_metrics"]
    assert reported != result.selection["folds"][0]["test_metrics"]
    assert reported["rank_ic"] == _text_of(walk_forward.rank_ic(predictions, actuals))
    assert reported["mae"] == _text_of(walk_forward.mean_absolute_error(predictions, actuals))
    assert reported["rmse"] == _text_of(
        walk_forward.root_mean_squared_error(predictions, actuals))


def _text_of(value):
    return None if value is None else str(value)


def test_two_runs_of_the_same_corpus_agree_exactly(runner, benchmark, synthetic_corpus,
                                                   tmp_path, monkeypatch):
    first = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                           tmp_path, monkeypatch)
    second = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    assert first.benchmark_results_hash == second.benchmark_results_hash
    assert first.dataset_hash == second.dataset_hash
    assert first.oos_records == second.oos_records
    assert first.selection["results_hash"] == second.selection["results_hash"]
    assert first.robustness["results_hash"] == second.robustness["results_hash"]


def test_the_two_products_stay_separate_experiments(runner, benchmark, synthetic_corpus,
                                                    tmp_path, monkeypatch):
    """No pooling, ever: two experiments, two identities, two record sets."""
    btc = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                         tmp_path, monkeypatch)
    eth = _run_synthetic(runner, benchmark, synthetic_corpus, "ETH-USD",
                         tmp_path, monkeypatch)
    assert btc.benchmark_spec_hash != eth.benchmark_spec_hash
    assert btc.benchmark_results_hash != eth.benchmark_results_hash
    assert btc.dataset_hash != eth.dataset_hash
    assert btc.oos_records != eth.oos_records
    # each result counts only its own observations
    for result in (btc, eth):
        assert result.selection["global_test_metrics"]["observations"] == \
            len(result.oos_records)
    combined = len(btc.oos_records) + len(eth.oos_records)
    assert btc.selection["global_test_metrics"]["observations"] != combined


def test_the_result_hash_covers_the_predictions_and_the_definition(runner, benchmark,
                                                                    synthetic_corpus,
                                                                    tmp_path, monkeypatch):
    from dataclasses import replace
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    baseline = result.benchmark_results_hash
    assert baseline != result.benchmark_spec_hash        # definition vs outcome
    assert replace(result, oos_records=result.oos_records[:-1]
                   ).benchmark_results_hash != baseline
    assert replace(result, benchmark_spec_hash="0" * 64
                   ).benchmark_results_hash != baseline
    assert replace(result, dataset_hash="0" * 64).benchmark_results_hash != baseline
    assert replace(result, corpus_content_hash="0" * 64
                   ).benchmark_results_hash != baseline


def test_the_robustness_periods_are_the_frozen_quarters(runner, benchmark,
                                                        synthetic_corpus, tmp_path,
                                                        monkeypatch):
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    periods = result.robustness["periods"]
    assert len(periods) == len(benchmark.ROBUSTNESS_BOUNDARIES_V1) - 1 == 4
    edges = [period["start"] for period in periods] + [periods[-1]["end"]]
    assert edges == list(benchmark.ROBUSTNESS_BOUNDARIES_V1)
    for entry in result.scenarios:
        assert "best" not in entry and "winner" not in entry
    assert not hasattr(result, "best_scenario")


def test_the_written_artefact_is_canonical_and_reparses(runner, benchmark,
                                                        synthetic_corpus, tmp_path,
                                                        monkeypatch):
    result = _run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                            tmp_path, monkeypatch)
    path = tmp_path / "out" / "BTC-USD.json"
    digest = runner.write_result(result, path)
    body = path.read_bytes()
    assert len(digest) == 64
    payload = json.loads(body.decode("utf-8"))
    assert payload["benchmark_results_hash"] == result.benchmark_results_hash
    # decimals travel as strings; no ambiguous JSON floats for market values
    assert all(isinstance(record["prediction"], str) for record in payload["oos_records"])
    assert runner.write_result(result, tmp_path / "out" / "again.json") == digest


def test_the_result_is_indifferent_to_the_callers_decimal_context(runner, benchmark,
                                                                   synthetic_corpus,
                                                                   tmp_path, monkeypatch):
    original = getcontext().prec
    seen = set()
    try:
        for precision in (7, 28, 34):
            getcontext().prec = precision
            seen.add(_run_synthetic(runner, benchmark, synthetic_corpus, "BTC-USD",
                                    tmp_path, monkeypatch).benchmark_results_hash)
    finally:
        getcontext().prec = original
    assert len(seen) == 1


# --- the real corpus: verified and measured, never scored -----------------


@pytest.mark.parametrize("product", ["BTC-USD", "ETH-USD"])
def test_the_real_corpus_matches_the_frozen_geometry_without_being_scored(
    runner, benchmark, product, tmp_path, monkeypatch
):
    """Geometry only. Any fit here would be a score observed before the runner ships."""
    models = importlib.import_module("scripts.trading_lab.models")

    def detonate(*args, **kwargs):
        raise AssertionError("the real corpus was fitted during a test")

    monkeypatch.setattr(models.RidgeRegressionPredictor, "fit", detonate)
    monkeypatch.setattr(models.XGBoostRegressionPredictor, "fit", detonate)
    observed = _synthetic_geometry(runner, benchmark, REAL_CORPUS, product, tmp_path)
    assert observed == runner.EXPECTED_GEOMETRY
