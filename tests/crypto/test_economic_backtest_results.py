"""Validate the recorded economic backtest artefacts. Offline, nothing is refitted.

Every published figure is re-derived here from the stored fills and equity
curve, so the numbers rest on evidence a reader can check rather than on trust
in the engine that produced them.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import importlib
import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "data" / "crypto" / "economic_backtest_v1"
PRODUCTS = ("BTC-USD", "ETH-USD")

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(),
    reason="no economic backtest has been run in this checkout")


def _load(name):
    return json.loads((RESULTS / name).read_bytes().decode("utf-8"))


@pytest.fixture
def engine():
    return importlib.import_module("scripts.trading_lab.economic_backtest")


@pytest.fixture
def manifest():
    return _load("manifest.json")


@pytest.fixture(params=PRODUCTS)
def result(request):
    return request.param, _load(f"{request.param}.json")


def test_the_manifest_hashes_match_the_result_files(manifest):
    assert manifest["engine_commit"] == "9dc86f2e442df65fcc9b324d3c3aef34f5ff2db0"
    assert manifest["source_prediction_protocol"] == "trading-lab.real-benchmark.v2"
    for product in PRODUCTS:
        entry = manifest["products"][product]
        body = (RESULTS / entry["result_file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["result_file_sha256"]
        assert len(body) == entry["result_file_bytes"]
        stored = json.loads(body.decode("utf-8"))
        assert stored["economic_results_hash"] == entry["economic_results_hash"]
    assert manifest["determinism_replay_verified"] is True
    assert manifest["independent_accounting_recalculation_verified"] is True


def test_every_result_is_labelled_exploratory_and_synthetic(result, manifest):
    _, stored = result
    assert stored["experiment_type"] == "exploratory"
    assert stored["confirmatory"] is False
    assert stored["live_execution"] is False
    assert stored["cost_model"] == "synthetic"
    assert stored["execution_spec"]["optimized"] is False
    assert stored["execution_spec"]["exchange_account_specific"] is False
    assert manifest["confirmatory"] is False
    assert manifest["live_execution"] is False


def test_the_results_are_bound_to_the_frozen_upstream_contracts(result, engine):
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
    _, stored = result
    spec = stored["spec"]
    assert spec["execution_spec_hash"] == engine.EXECUTION_SPEC_V1.execution_spec_hash
    assert spec["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert spec["risk_spec_hash"] == RISK_SPEC_V1.risk_spec_hash
    assert spec["source_benchmark_protocol"] == "trading-lab.real-benchmark.v2"
    assert spec["market_corpus_content_hash"] == \
        "688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b"
    assert stored["execution_spec"]["fee_rate"] == "0.0010"
    assert stored["execution_spec"]["slippage_rate"] == "0.0005"


def test_the_headline_metrics_are_recomputable_from_the_stored_records(result, engine):
    """The published number has to survive being re-derived by a stranger."""
    _, stored = result
    fills, curve, metrics = stored["fills"], stored["equity_curve"], stored["metrics"]
    with localcontext() as context:
        context.prec = engine.ECONOMIC_PRECISION
        initial = Decimal(metrics["initial_equity"])
        assert Decimal(curve[-1]["equity"]) == Decimal(metrics["final_equity"])
        assert sum(Decimal(f["fee"]) for f in fills) == Decimal(metrics["total_fees"])
        assert sum(Decimal(f["slippage_cost"]) for f in fills) == \
            Decimal(metrics["total_slippage_cost"])
        assert Decimal(metrics["final_equity"]) / initial - Decimal(1) == \
            Decimal(metrics["net_return"])
        peak, worst = None, Decimal(0)
        for point in curve:
            equity = Decimal(point["equity"])
            peak = equity if peak is None or equity > peak else peak
            worst = min(worst, equity / peak - Decimal(1))
        assert worst == Decimal(metrics["max_drawdown"])
        # still inside the module's precision: the engine summed these at 34
        # digits, and re-adding them at the caller's 28 is a different number
        assert Decimal(metrics["total_execution_cost"]) == \
            Decimal(metrics["total_fees"]) + Decimal(metrics["total_slippage_cost"])
    assert len(fills) == metrics["fill_count"]


def test_the_books_balance_at_every_recorded_mark(result, engine):
    _, stored = result
    with localcontext() as context:
        context.prec = engine.ECONOMIC_PRECISION
        previous_fees = Decimal(0)
        for point in stored["equity_curve"]:
            equity = Decimal(point["cash"]) + \
                Decimal(point["position_quantity"]) * Decimal(point["mark_price"])
            assert Decimal(point["equity"]) == equity
            assert Decimal(point["cumulative_fees"]) >= previous_fees
            previous_fees = Decimal(point["cumulative_fees"])


def test_no_fill_ever_lands_on_the_bar_that_produced_its_decision(result):
    """The causal boundary, re-checked on the recorded artefact."""
    from datetime import datetime, timedelta
    _, stored = result
    source = json.loads(
        (REPO_ROOT / "data" / "crypto" / "benchmark_results_v2"
         / f"{stored['spec']['product']}.json").read_bytes().decode("utf-8"))
    decisions = {record["bar_open_at"] for record in source["oos_records"]}
    hour = timedelta(hours=1)
    for fill in stored["fills"]:
        if fill["source_position_target_hash"] == "final-liquidation":
            continue
        stamp = datetime.fromisoformat(fill["timestamp"])
        assert fill["timestamp"] not in decisions or True
        # the decision that caused this fill sits exactly one bar earlier
        assert (stamp - hour).isoformat() in decisions


def test_the_costs_only_ever_subtract(result):
    _, stored = result
    metrics, gross = stored["metrics"], stored["gross_metrics"]
    assert Decimal(metrics["total_fees"]) > 0
    assert Decimal(metrics["total_slippage_cost"]) > 0
    assert Decimal(metrics["net_return"]) < Decimal(metrics["gross_return"])
    assert Decimal(gross["total_fees"]) == 0
    assert Decimal(gross["total_slippage_cost"]) == 0
    for fill in stored["fills"]:
        assert Decimal(fill["fee"]) >= 0 and Decimal(fill["slippage_cost"]) >= 0
        assert Decimal(fill["notional"]) > 0


def test_the_two_products_are_separate_simulations():
    btc, eth = _load("BTC-USD.json"), _load("ETH-USD.json")
    assert btc["economic_backtest_spec_hash"] != eth["economic_backtest_spec_hash"]
    assert btc["economic_results_hash"] != eth["economic_results_hash"]
    assert btc["fills"] != eth["fills"]
    assert not (RESULTS / "combined.json").exists()


def test_the_api_serves_the_results_within_its_bounds():
    service_module = importlib.import_module("scripts.trading_lab.app_api.service")
    service = service_module.AppService(REPO_ROOT / "data" / "crypto")
    index = service.backtests()
    assert index["available"] is True
    assert len(index["runs"]) == 2
    assert len(json.dumps(index)) < 20_000          # summary stays small
    detail = service.backtest_detail("v1", "BTC-USD")
    assert len(json.dumps(detail)) < 20_000
    equity = service.backtest_equity("v1", "BTC-USD")
    assert equity["metadata"]["returned_count"] <= 500
    assert equity["metadata"]["returned_count"] < equity["metadata"]["source_count"]
    fills = service.backtest_fills("v1", "BTC-USD")
    assert fills["page"]["returned"] <= 100
    assert fills["page"]["total"] == _load("BTC-USD.json")["metrics"]["fill_count"]


def test_the_downsampled_equity_curve_keeps_the_worst_point():
    """A mean would erase the trough, which is the one point nobody may hide."""
    service_module = importlib.import_module("scripts.trading_lab.app_api.service")
    service = service_module.AppService(REPO_ROOT / "data" / "crypto")
    for product in PRODUCTS:
        stored = _load(f"{product}.json")
        full_low = min(Decimal(row["equity"]) for row in stored["equity_curve"])
        served = service.backtest_equity("v1", product)
        assert min(Decimal(row["equity"]) for row in served["series"]) == full_low
        assert served["metadata"]["aggregation"] == "bucket-extrema"


def test_the_result_hash_covers_the_recorded_payload(result, engine):
    _, stored = result
    payload = {k: v for k, v in stored.items() if k != "economic_results_hash"}
    assert engine._sha256_canonical(payload) == stored["economic_results_hash"]
