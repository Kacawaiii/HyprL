"""Validate the recorded Benchmark V1 result artefacts. Offline, no refitting.

These tests re-derive the reported metrics from the stored prediction records,
so the published numbers stand on evidence anyone can check rather than on
trust in the runner that produced them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib
import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "data" / "crypto" / "benchmark_results_v1"
PRODUCTS = ("BTC-USD", "ETH-USD")
QUARTERS = ("2025-08-01T00:00:00Z", "2025-11-01T00:00:00Z", "2026-02-01T00:00:00Z",
            "2026-05-01T00:00:00Z", "2026-08-01T00:00:00Z")

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(),
    reason="the real benchmark has not been run in this checkout")


def _load(name):
    return json.loads((RESULTS / name).read_bytes().decode("utf-8"))


@pytest.fixture
def manifest():
    return _load("manifest.json")


@pytest.fixture(params=PRODUCTS)
def result(request):
    return request.param, _load(f"{request.param}.json")


def _moment(text): return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _text(value): return None if value is None else str(value)


def test_the_manifest_hashes_match_the_result_files(manifest):
    assert manifest["benchmark_contract_commit"] == \
        "1c92eb95317db2794f468de100ebd01cfad12bfb"
    assert manifest["runner_commit"] == "54244bef023e3379220004f70d23e2fc922e5e1f"
    for product in PRODUCTS:
        entry = manifest["products"][product]
        body = (RESULTS / entry["result_file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["result_file_sha256"]
        assert len(body) == entry["result_file_bytes"]
        stored = json.loads(body.decode("utf-8"))
        assert stored["benchmark_results_hash"] == entry["benchmark_results_hash"]
        assert stored["dataset_hash"] == entry["dataset_hash"]
        assert len(stored["oos_records"]) == entry["oos_records"]
    assert manifest["determinism_replay_verified"] is True
    assert manifest["point_in_time_exchange_revision_history"] is False
    assert manifest["trading_cost_analysis_available"] is False


def test_the_results_are_bound_to_the_frozen_contract_and_corpus(result, manifest):
    product, stored = result
    expected = {"BTC-USD": "dd7c474b34489ad3ca0c8ce906187821039fd9f0c50bff4a03b2da8cd20332a6",
                "ETH-USD": "3d418bbbe6b0a418b2821fd44cbaa09273f8457f7bc08ee5c9c95d43956ad5f9"}
    assert stored["benchmark_spec_hash"] == expected[product]
    assert stored["corpus_spec_hash"] == manifest["corpus"]["corpus_spec_hash"]
    assert stored["corpus_content_hash"] == manifest["corpus"]["corpus_content_hash"]
    assert stored["geometry"] == {
        "series_points": 8750, "missing_openings": 10, "usable_rows": 8663,
        "folds": 46, "min_effective_validation": 164, "oos_records": 7728}


def test_the_global_metrics_are_recomputable_from_the_records(result):
    """The published number has to survive being re-derived by a stranger."""
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
    # not an average of the folds
    assert reported != stored["selection"]["folds"][0]["test_metrics"]


def test_each_quarter_is_recomputable_and_uses_the_frozen_boundaries(result):
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


def test_the_out_of_sample_observations_are_unique_and_ordered(result):
    _, stored = result
    stamps = [r["bar_open_at"] for r in stored["oos_records"]]
    assert len(set(stamps)) == len(stamps)
    assert stamps == sorted(stamps)
    assert sum(f["test_metrics"]["observations"]
               for f in stored["selection"]["folds"]) == len(stamps)


def test_every_fold_selected_on_validation_and_refitted_freshly(result):
    _, stored = result
    folds = stored["selection"]["folds"]
    assert len(folds) == 46
    for fold in folds:
        assert [c["candidate_id"] for c in fold["candidates"]] == ["ridge", "xgboost"]
        assert fold["selected_candidate_id"] in {"ridge", "xgboost"}
        assert fold["selection_reason"] in {"rank_ic", "mae", "rmse", "candidate_id"}
        assert fold["validation_rows"] >= 3
        # the model that faced test is not the one that won validation
        assert fold["selection_fit_hash"] != fold["final_fit_hash"]
    counts = stored["selection"]["selection_counts"]
    assert sum(counts.values()) == 46


def test_the_two_products_were_never_pooled():
    """Two experiments, two record sets, two identities. No combined score."""
    btc, eth = _load("BTC-USD.json"), _load("ETH-USD.json")
    assert btc["benchmark_spec_hash"] != eth["benchmark_spec_hash"]
    assert btc["benchmark_results_hash"] != eth["benchmark_results_hash"]
    assert btc["dataset_hash"] != eth["dataset_hash"]
    assert btc["oos_records"] != eth["oos_records"]
    combined = len(btc["oos_records"]) + len(eth["oos_records"])
    for stored in (btc, eth):
        assert stored["selection"]["global_test_metrics"]["observations"] != combined
    assert not (RESULTS / "combined.json").exists()


def test_no_scenario_is_designated_the_best(result):
    _, stored = result
    ids = [s["scenario_id"] for s in stored["scenarios"]]
    assert sorted(ids) == ["central", "ridge_high", "ridge_low"]
    for scenario in stored["scenarios"]:
        assert "best" not in scenario and "winner" not in scenario
        assert scenario["global_test_metrics"]["observations"] == 7728
    assert "best_scenario" not in stored and "best_candidate" not in stored


def test_the_results_hash_covers_the_recorded_payload(result):
    """Recompute the identity from the file, exactly as the runner defines it."""
    runner = importlib.import_module("scripts.trading_lab.run_real_benchmark")
    _, stored = result
    payload = {k: v for k, v in stored.items() if k != "benchmark_results_hash"}
    assert runner._sha256_canonical(payload) == stored["benchmark_results_hash"]
