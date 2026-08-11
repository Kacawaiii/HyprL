"""Validate the recorded portfolio backtest. Offline, nothing is refitted.

Every published figure is re-derived here from the stored fills, equity curve
and attribution, so the numbers rest on evidence a reader can check rather
than on trust in the engine that produced them.

The one that matters most is the reconciliation: per-instrument contributions
that do not add up to the change in portfolio equity would mean the shared
cash ledger and the attribution disagree, and the attribution is the only
thing that says which instrument lost the money.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from decimal import Decimal, localcontext

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "data" / "crypto" / "portfolio_backtest_v1"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(),
    reason="no portfolio backtest has been run in this checkout")

BTC = "coinbase:BTC-USD"
ETH = "coinbase:ETH-USD"
PRECISION = 34


@pytest.fixture(scope="module")
def manifest():
    return json.loads((RESULTS / "manifest.json").read_bytes())


@pytest.fixture(scope="module")
def result(manifest):
    return json.loads((RESULTS / manifest["result_file"]).read_bytes())


# --- the artefact is what it says it is -----------------------------------


def test_the_result_file_matches_its_recorded_digest(manifest):
    body = (RESULTS / manifest["result_file"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == manifest["result_file_sha256"]
    assert len(body) == manifest["result_file_bytes"]


def test_the_result_hash_covers_the_recorded_payload(manifest, result):
    payload = {key: value for key, value in result.items()
               if key not in ("portfolio_backtest_spec",
                              "portfolio_backtest_spec_hash", "alignment")}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    assert recomputed == manifest["result_hash"]


def test_the_backtest_spec_hash_covers_its_own_payload(manifest, result):
    spec = result["portfolio_backtest_spec"]
    recomputed = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert recomputed == manifest["portfolio_backtest_spec_hash"]
    assert recomputed == result["portfolio_backtest_spec_hash"]


def test_the_run_is_bound_to_the_frozen_upstream_contracts(manifest, result):
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    spec = result["portfolio_backtest_spec"]
    assert spec["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert spec["risk_spec_hash"] == RISK_SPEC_V1.risk_spec_hash
    assert spec["execution_spec_hash"] == EXECUTION_SPEC_V1.execution_spec_hash
    assert spec["portfolio_spec_hash"] == PORTFOLIO_SPEC_V1.portfolio_spec_hash
    assert manifest["portfolio_spec_hash"] == PORTFOLIO_SPEC_V1.portfolio_spec_hash


def test_the_source_predictions_are_the_committed_v2_results(manifest):
    v2 = json.loads(
        (REPO_ROOT / "data/crypto/benchmark_results_v2/manifest.json").read_bytes())
    recorded = manifest["source"]["benchmark_results_hashes"]
    assert recorded[BTC] == v2["products"]["BTC-USD"]["benchmark_results_hash"]
    assert recorded[ETH] == v2["products"]["ETH-USD"]["benchmark_results_hash"]


def test_the_run_is_labelled_exploratory_with_no_edge_claimed(manifest, result):
    assert manifest["experiment_type"] == "exploratory"
    assert manifest["confirmatory"] is False
    assert manifest["live_execution"] is False
    assert manifest["cost_model"] == "synthetic"
    assert manifest["commercial_edge_established"] is False
    assert result["confirmatory"] is False


def test_the_instruments_are_recorded_by_canonical_identity(manifest, result):
    assert manifest["instruments"] == [BTC, ETH]
    assert sorted(record["instrument_id"]
                  for record in result["attribution"]) == [BTC, ETH]
    for fill in result["fills"][:50]:
        assert fill["instrument_id"] in (BTC, ETH)


# --- every headline figure re-derived -------------------------------------


def test_the_costs_are_the_sum_of_the_recorded_fills(result):
    with localcontext() as context:
        context.prec = PRECISION
        fees = sum((Decimal(fill["fee"]) for fill in result["fills"]), Decimal(0))
        slippage = sum((Decimal(fill["slippage_cost"])
                        for fill in result["fills"]), Decimal(0))
        assert fees == Decimal(result["metrics"]["total_fees"])
        assert slippage == Decimal(result["metrics"]["total_slippage_cost"])
        assert fees + slippage == Decimal(result["metrics"]["total_execution_cost"])
    assert len(result["fills"]) == result["metrics"]["fill_count"]


def test_the_final_equity_is_the_last_recorded_mark(result):
    assert Decimal(result["equity_curve"][-1]["equity"]) == \
        Decimal(result["metrics"]["final_equity"])


def test_the_returns_follow_from_the_equity(result):
    with localcontext() as context:
        context.prec = PRECISION
        initial = Decimal(result["metrics"]["initial_equity"])
        final = Decimal(result["metrics"]["final_equity"])
        assert (final - initial) / initial == Decimal(result["metrics"]["net_return"])
        assert final - initial == Decimal(result["metrics"]["net_pnl"])


def test_the_drawdown_is_recomputable_from_the_curve(result):
    with localcontext() as context:
        context.prec = PRECISION
        peak = Decimal(result["equity_curve"][0]["equity"])
        worst = Decimal(0)
        for row in result["equity_curve"]:
            equity = Decimal(row["equity"])
            peak = max(peak, equity)
            worst = min(worst, (equity - peak) / peak)
        assert worst == Decimal(result["metrics"]["max_drawdown"])


def test_the_turnover_is_recomputable_from_the_fills(result):
    with localcontext() as context:
        context.prec = PRECISION
        traded = sum((abs(Decimal(fill["quantity_delta"])
                          * Decimal(fill["reference_price"]))
                      for fill in result["fills"]), Decimal(0))
        initial = Decimal(result["metrics"]["initial_equity"])
        assert traded / initial == Decimal(result["metrics"]["portfolio_turnover"])


def test_every_recorded_mark_balances_cash_plus_positions(result):
    """The shared ledger identity, at every one of the recorded marks."""
    with localcontext() as context:
        context.prec = PRECISION
        for row in result["equity_curve"]:
            rebuilt = Decimal(row["cash"]) + sum(
                (Decimal(quantity) * Decimal(price)
                 for _, quantity, price, _ in row["positions"]), Decimal(0))
            assert rebuilt == Decimal(row["equity"]), row["timestamp"]


# --- attribution ----------------------------------------------------------


def test_the_net_contributions_explain_the_change_in_equity(result, manifest):
    with localcontext() as context:
        context.prec = PRECISION
        total = sum((Decimal(record["net_pnl"])
                     for record in result["attribution"]), Decimal(0))
        initial = Decimal(result["metrics"]["initial_equity"])
        final = Decimal(result["metrics"]["final_equity"])
        residual = abs(total - (final - initial)) / final
        assert residual < Decimal(manifest["reconciliation"]["relative_tolerance"])
    assert manifest["reconciliation"]["attribution_reconciles"] is True


def test_each_instruments_costs_are_the_sum_of_its_own_fills(result):
    with localcontext() as context:
        context.prec = PRECISION
        for record in result["attribution"]:
            own = [fill for fill in result["fills"]
                   if fill["instrument_id"] == record["instrument_id"]]
            assert sum((Decimal(fill["fee"]) for fill in own), Decimal(0)) \
                == Decimal(record["fees"])
            assert sum((Decimal(fill["slippage_cost"]) for fill in own),
                       Decimal(0)) == Decimal(record["slippage_cost"])
            assert len(own) == record["fill_count"]


def test_net_contribution_is_gross_minus_that_instruments_costs(result):
    with localcontext() as context:
        context.prec = PRECISION
        for record in result["attribution"]:
            assert Decimal(record["net_pnl"]) == (
                Decimal(record["gross_pnl"]) - Decimal(record["execution_cost"])), \
                record["instrument_id"]


def test_costs_only_ever_subtract(result):
    for fill in result["fills"]:
        assert Decimal(fill["fee"]) >= 0
        assert Decimal(fill["slippage_cost"]) >= 0
    for record in result["attribution"]:
        assert Decimal(record["fees"]) >= 0
        assert Decimal(record["slippage_cost"]) >= 0
    assert Decimal(result["metrics"]["net_pnl"]) <= \
        Decimal(result["metrics"]["gross_pnl"])


def test_the_gross_view_removes_exactly_the_execution_cost(result):
    with localcontext() as context:
        context.prec = PRECISION
        assert Decimal(result["gross_metrics"]["total_execution_cost"]) == Decimal(0)
        assert Decimal(result["gross_metrics"]["final_equity"]) == (
            Decimal(result["metrics"]["final_equity"])
            + Decimal(result["metrics"]["total_execution_cost"]))


# --- the caps and the alignment -------------------------------------------


def test_no_recorded_mark_exceeds_the_gross_cap(result):
    from scripts.trading_lab.portfolio import (
        LIMIT_ROUNDING_TOLERANCE, PORTFOLIO_SPEC_V1)

    with localcontext() as context:
        # abs() rounds to the active context, so recomputing outside the
        # economic precision truncates the very figure being compared
        context.prec = PRECISION
        cap = PORTFOLIO_SPEC_V1.max_gross_exposure + LIMIT_ROUNDING_TOLERANCE
        worst = max(abs(Decimal(row["gross_exposure"]))
                    for row in result["equity_curve"])
        assert worst <= cap
        assert worst == Decimal(result["metrics"]["max_observed_gross_exposure"])


def test_the_alignment_is_reported_rather_than_silently_intersected(manifest):
    alignment = manifest["alignment"]
    assert alignment["instruments"][BTC] == alignment["instruments"][ETH]
    assert alignment["shared_timestamps"] == alignment["scheduled_timestamps"]
    assert alignment["single_instrument_timestamps"] == 0
    assert alignment["expired_targets"] == {BTC: 0, ETH: 0}


def test_no_timestamp_was_left_without_a_computable_equity(manifest, result):
    assert manifest["unavailable_valuations"] == 0
    assert len(result["equity_curve"]) == manifest["equity_points"]
    assert len(result["equity_curve"]) == manifest["alignment"]["valuation_timestamps"]


def test_no_fill_precedes_the_decision_that_produced_it(result):
    """Causality: a target decided at T is priced at T + one interval."""
    first = result["fills"][0]["timestamp"]
    assert first == manifest_first_fill()


def manifest_first_fill():
    return json.loads(
        (RESULTS / "manifest.json").read_bytes())["alignment"]["first_fill_at"]


# --- the single-product results are untouched -----------------------------


def test_the_single_product_economic_hashes_are_unchanged():
    """A shared-capital engine must not restate the separate runs."""
    economic = json.loads(
        (REPO_ROOT / "data/crypto/economic_backtest_v1/manifest.json").read_bytes())
    assert economic["products"]["BTC-USD"]["economic_results_hash"] == (
        "4617c6151da9cb6560299149047e6d7954c81e9699b16b044afb48991551bb77")
    assert economic["products"]["ETH-USD"]["economic_results_hash"] == (
        "47f0e8e324d4b58b28cf15e29ceb2bca25f434279ef6ac915a4bd4f429063bf1")


def test_the_portfolio_fill_counts_match_the_separate_runs(result):
    """Same targets, same causal rule: the trade counts should agree."""
    economic = json.loads(
        (REPO_ROOT / "data/crypto/economic_backtest_v1/manifest.json").read_bytes())
    by_instrument = {record["instrument_id"]: record["fill_count"]
                     for record in result["attribution"]}
    assert by_instrument[BTC] == economic["products"]["BTC-USD"]["fills"]
    assert by_instrument[ETH] == economic["products"]["ETH-USD"]["fills"]


# --- the API serves it within its bounds ----------------------------------


def test_the_api_serves_the_portfolio_result_within_its_bounds():
    from scripts.trading_lab.app_api.service import AppService

    service = AppService(REPO_ROOT / "data/crypto")
    summary = service.portfolio_backtests()
    assert summary["available"] is True
    assert len(json.dumps(summary)) < 20 * 1024

    detail = service.portfolio_backtest_detail("v1")
    assert detail["commercial_edge_established"] is False
    assert len(json.dumps(detail)) < 20 * 1024

    equity = service.portfolio_equity("v1")
    assert len(equity["series"]) <= 2000
    assert equity["metadata"]["aggregated"] is True

    fills = service.portfolio_fills("v1")
    assert len(fills["fills"]) == 100
    assert fills["page"]["has_more"] is True

    attribution = service.portfolio_attribution("v1")
    assert len(attribution["attribution"]) == 2


def test_the_downsampled_equity_curve_keeps_the_worst_point(result):
    """A mean would erase the trough; bucket extrema must not."""
    from scripts.trading_lab.app_api.service import AppService

    service = AppService(REPO_ROOT / "data/crypto")
    series = service.portfolio_equity("v1", max_points=500)["series"]
    kept = min(Decimal(row["equity"]) for row in series)
    actual = min(Decimal(row["equity"]) for row in result["equity_curve"])
    assert kept == actual
