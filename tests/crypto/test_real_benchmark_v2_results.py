"""Validate the recorded V2 exploratory result artefacts. Offline, no refitting.

The V2 numbers are re-derived from the stored prediction records, and the V1
artefacts are re-checked here too: a second experiment that quietly disturbed
the first one's recorded evidence would be worse than no second experiment.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import hashlib
import importlib
import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
V1 = REPO_ROOT / "data" / "crypto" / "benchmark_results_v1"
V2 = REPO_ROOT / "data" / "crypto" / "benchmark_results_v2"
PRODUCTS = ("BTC-USD", "ETH-USD")
QUARTERS = ("2025-08-01T00:00:00Z", "2025-11-01T00:00:00Z", "2026-02-01T00:00:00Z",
            "2026-05-01T00:00:00Z", "2026-08-01T00:00:00Z")
V2_SPEC = {"BTC-USD": "ec4d19b55e36ea67119b31289d798adfd37cdb3b9e9ded73cc5fbcdd2c245d79",
           "ETH-USD": "db371b153824d337db1a848fef9a1658a603e84c1911fd3c9de68ab01b0566b0"}
V1_SPEC = {"BTC-USD": "dd7c474b34489ad3ca0c8ce906187821039fd9f0c50bff4a03b2da8cd20332a6",
           "ETH-USD": "3d418bbbe6b0a418b2821fd44cbaa09273f8457f7bc08ee5c9c95d43956ad5f9"}

pytestmark = pytest.mark.skipif(
    not (V2 / "manifest.json").is_file(),
    reason="the V2 exploratory benchmark has not been run in this checkout")


def _load(root, name):
    return json.loads((root / name).read_bytes().decode("utf-8"))


def _moment(text): return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _text(value): return None if value is None else str(value)


@pytest.fixture(params=PRODUCTS)
def result(request):
    return request.param, _load(V2, f"{request.param}.json")


def test_the_v2_manifest_matches_its_result_files():
    manifest = _load(V2, "manifest.json")
    assert manifest["benchmark_contract_commit"] == \
        "4354e6be1520b662b3fdeeffd68e0ce3926d0e48"
    assert manifest["runner_commit"] == "937808f52e161f8a9f2b36d79d4eb434193ebaae"
    assert manifest["benchmark_protocol_version"] == "trading-lab.real-benchmark.v2"
    for product in PRODUCTS:
        entry = manifest["products"][product]
        body = (V2 / entry["result_file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["result_file_sha256"]
        assert len(body) == entry["result_file_bytes"]
        stored = json.loads(body.decode("utf-8"))
        assert stored["benchmark_results_hash"] == entry["benchmark_results_hash"]
        assert stored["benchmark_spec_hash"] == V2_SPEC[product]
    assert manifest["determinism_replay_verified"] is True
    assert manifest["independent_metric_recalculation_verified"] is True


def test_the_v2_result_is_labelled_exploratory_everywhere(result):
    """A number from an already-observed corpus must carry that fact in its payload."""
    _, stored = result
    experiment = stored["experiment"]
    assert experiment["experiment_type"] == "exploratory"
    assert experiment["confirmatory_result"] is False
    assert experiment["corpus_role"] == "development/exploratory"
    manifest = _load(V2, "manifest.json")
    assert manifest["experiment_type"] == "exploratory"
    assert manifest["confirmatory_result"] is False
    assert manifest["corpus"]["already_observed_under_v1"] is True
    assert stored["protocol_version"] == "trading-lab.real-benchmark.v2"


def test_the_v2_global_metrics_are_recomputable_from_the_records(result):
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    _, stored = result
    records = stored["oos_records"]
    predictions = tuple(Decimal(r["prediction"]) for r in records)
    actuals = tuple(Decimal(r["actual_forward_return"]) for r in records)
    reported = stored["selection"]["global_test_metrics"]
    assert reported["observations"] == len(records) == 7728
    assert reported["rank_ic"] == _text(walk_forward.rank_ic(predictions, actuals))
    assert reported["mae"] == _text(
        walk_forward.mean_absolute_error(predictions, actuals))
    assert reported["rmse"] == _text(
        walk_forward.root_mean_squared_error(predictions, actuals))
    assert reported != stored["selection"]["folds"][0]["test_metrics"]


def test_each_v2_quarter_is_recomputable_on_the_frozen_boundaries(result):
    walk_forward = importlib.import_module("scripts.trading_lab.walk_forward")
    _, stored = result
    periods = stored["robustness"]["periods"]
    assert [p["start"] for p in periods] + [periods[-1]["end"]] == list(QUARTERS)
    records = stored["oos_records"]
    for period in periods:
        start, end = _moment(period["start"]), _moment(period["end"])
        inside = [r for r in records if start <= _moment(r["bar_open_at"]) < end]
        predictions = tuple(Decimal(r["prediction"]) for r in inside)
        actuals = tuple(Decimal(r["actual_forward_return"]) for r in inside)
        got = period["metrics"]
        assert got["observations"] == len(inside)
        assert got["rank_ic"] == _text(walk_forward.rank_ic(predictions, actuals))
        assert got["mae"] == _text(walk_forward.mean_absolute_error(predictions, actuals))
        assert got["rmse"] == _text(
            walk_forward.root_mean_squared_error(predictions, actuals))
    assert sum(p["metrics"]["observations"] for p in periods) == len(records)


def test_the_v2_geometry_and_selection_traces_are_complete(result):
    _, stored = result
    assert stored["geometry"] == {
        "series_points": 8750, "missing_openings": 10, "usable_rows": 8663,
        "folds": 46, "min_effective_validation": 164, "oos_records": 7728}
    folds = stored["selection"]["folds"]
    assert len(folds) == 46
    for fold in folds:
        assert [c["candidate_id"] for c in fold["candidates"]] == ["ridge", "xgboost"]
        assert fold["validation_rows"] >= 3
        assert fold["selection_fit_hash"] != fold["final_fit_hash"]
    stamps = [r["bar_open_at"] for r in stored["oos_records"]]
    assert len(set(stamps)) == len(stamps) == 7728
    assert stamps == sorted(stamps)


def test_the_v2_results_hash_covers_the_recorded_payload(result):
    runner = importlib.import_module("scripts.trading_lab.run_real_benchmark")
    _, stored = result
    payload = {k: v for k, v in stored.items() if k != "benchmark_results_hash"}
    assert runner._sha256_canonical(payload) == stored["benchmark_results_hash"]


def test_v1_and_v2_are_distinct_experiments_and_never_pooled():
    for product in PRODUCTS:
        one, two = _load(V1, f"{product}.json"), _load(V2, f"{product}.json")
        assert one["benchmark_spec_hash"] != two["benchmark_spec_hash"]
        assert one["benchmark_results_hash"] != two["benchmark_results_hash"]
        assert one["dataset_hash"] != two["dataset_hash"]
        assert one["oos_records"] != two["oos_records"]
    btc, eth = _load(V2, "BTC-USD.json"), _load(V2, "ETH-USD.json")
    combined = len(btc["oos_records"]) + len(eth["oos_records"])
    for stored in (btc, eth):
        assert stored["selection"]["global_test_metrics"]["observations"] != combined
    assert not (V2 / "combined.json").exists()


def test_the_v1_artefacts_were_not_disturbed_by_v2():
    """The first experiment's recorded evidence must survive the second."""
    manifest = _load(V1, "manifest.json")
    for product in PRODUCTS:
        body = (V1 / f"{product}.json").read_bytes()
        entry = manifest["products"][product]
        assert hashlib.sha256(body).hexdigest() == entry["result_file_sha256"]
        stored = json.loads(body.decode("utf-8"))
        assert stored["benchmark_spec_hash"] == V1_SPEC[product]
        assert "experiment" not in stored        # V1 predates the field
        assert stored["protocol_version"] == "trading-lab.real-benchmark.v1"
    btc = _load(V1, "BTC-USD.json")
    eth = _load(V1, "ETH-USD.json")
    assert btc["selection"]["global_test_metrics"]["rank_ic"].startswith("-0.00304")
    assert eth["selection"]["global_test_metrics"]["rank_ic"].startswith("0.00417")


def test_the_future_confirmatory_holdout_is_still_untouched(result):
    _, stored = result
    holdout = stored["experiment"]["future_holdout"]
    assert holdout["range_start"] == "2026-09-01T00:00:00Z"
    assert holdout["range_end"] == "2026-11-30T23:00:00Z"
    assert holdout["products"] == ["BTC-USD", "ETH-USD"]
    assert holdout["captured"] is False
    assert holdout["single_use"] is True
    manifest = _load(V2, "manifest.json")
    assert manifest["future_confirmatory_holdout"]["captured"] is False
    # nothing was downloaded for it
    corpus = REPO_ROOT / "data" / "crypto"
    assert not (corpus / holdout["holdout_id"]).exists()
    for record in stored["oos_records"]:
        assert record["bar_open_at"] < "2026-08-01"


def test_no_scenario_is_designated_best_in_v2(result):
    _, stored = result
    assert sorted(s["scenario_id"] for s in stored["scenarios"]) == \
        ["central", "ridge_high", "ridge_low"]
    for scenario in stored["scenarios"]:
        assert "best" not in scenario and "winner" not in scenario
        assert scenario["global_test_metrics"]["observations"] == 7728
    assert "best_scenario" not in stored
