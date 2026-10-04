"""Persisted out-of-sample signal runs: the hash gate, determinism, paging and the unavailable path."""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "crypto"
PRODUCTS = ("BTC-USD", "ETH-USD")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "signal_runs_v1" / "manifest.json").is_file()
    or not (DATA_ROOT / "economic_backtest_v1" / "manifest.json").is_file(),
    reason="no signal runs in this checkout")


def _service(root):
    return importlib.import_module("scripts.trading_lab.app_api.service").AppService(root)


def _builder():
    return importlib.import_module("scripts.trading_lab.build_signal_runs")


@pytest.fixture(scope="module")
def served():
    return _service(DATA_ROOT)


@pytest.fixture
def copy_root(tmp_path):
    for name in ("signal_runs_v1", "economic_backtest_v1", "benchmark_results_v2"):
        shutil.copytree(DATA_ROOT / name, tmp_path / name)
    return tmp_path


def _edit(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize("product", PRODUCTS)
def test_the_series_hashes_equal_the_committed_economic_backtest(served, product):
    economic = json.loads((DATA_ROOT / "economic_backtest_v1" / f"{product}.json").read_text())
    signals = served.signals(product=product, limit=5)
    targets = served.risk_targets(product=product, limit=5)
    assert signals["available"] and targets["available"]
    assert signals["signal_series_hash"] == economic["signal_series_hash"]
    assert signals["position_target_series_hash"] == economic["position_target_series_hash"]
    assert signals["out_of_sample"] == targets["out_of_sample"] == "walk_forward"
    assert signals["counts"]["decisions"] == signals["counts"]["targets"] == 7728
    assert all(row["out_of_sample"] == "walk_forward" for row in signals["decisions"])


def test_decisions_carry_their_fold_provenance(served):
    row = served.signals(limit=1)["decisions"][0]
    for field in ("timestamp", "prediction", "direction", "strength", "signal_spec_hash",
                  "decision_hash", "fold_index", "model_spec_hash", "fitted_hash"):
        assert row[field] not in (None, "")
    assert row["direction"] in ("LONG", "FLAT", "SHORT")


@pytest.mark.parametrize("change, fragment", [
    (lambda run: run["decisions"][100].update(direction="SHORT"), "differ"),
    (lambda run: run["decisions"][100].update(prediction="0.5"), "hash"),
    (lambda run: run["targets"][5].update(target_exposure="0.25"), "differ"),
    (lambda run: run["folds"][0].update(fitted_hash="0" * 64), "hash"),
    (lambda run: run.update(signal_series_hash="0" * 64), "hash"),
])
def test_a_tampered_artefact_is_refused(copy_root, change, fragment):
    _edit(copy_root / "signal_runs_v1" / "BTC-USD.json", change)
    service = _service(copy_root)
    for payload, key in ((service.signals(), "decisions"),
                         (service.risk_targets(), "targets")):
        assert payload["available"] is False
        assert payload[key] == []
        assert "refused" in payload["reason"] and fragment in payload["reason"]


def test_a_run_that_no_longer_matches_the_backtest_is_refused(copy_root):
    _edit(copy_root / "economic_backtest_v1" / "BTC-USD.json",
          lambda economic: economic.update(signal_series_hash="1" * 64))
    payload = _service(copy_root).signals()
    assert payload["available"] is False and "refused" in payload["reason"]
    with pytest.raises(_builder().SignalRunError, match="does not match"):
        _builder().build_all(copy_root)


def test_without_the_backtest_the_run_is_not_served(copy_root):
    shutil.rmtree(copy_root / "economic_backtest_v1")
    payload = _service(copy_root).signals()
    assert payload["available"] is False and "unavailable" in payload["reason"]


# --- the unavailable path -----------------------------------------------------


def test_with_no_artefact_the_honest_answer_remains(tmp_path):
    service = _service(tmp_path)
    signals, targets = service.signals(), service.risk_targets()
    assert signals["available"] is False and signals["decisions"] == []
    assert "no persisted signal run" in signals["reason"]
    assert targets["available"] is False and targets["targets"] == []
    assert "no persisted position target run" in targets["reason"]
    assert signals["signal_spec"]["optimized"] is False
    assert targets["risk_spec"]["optimized"] is False


# --- paging -------------------------------------------------------------------


def test_pages_are_bounded_newest_first_and_contiguous(served):
    first = served.signals(limit=3)
    assert first["page"]["returned"] == 3 and first["page"]["has_more"]
    stamps = [row["timestamp"] for row in first["decisions"]]
    assert stamps == sorted(stamps, reverse=True)
    second = served.signals(limit=3, cursor=first["page"]["next_cursor"])
    assert max(row["timestamp"] for row in second["decisions"]) < stamps[-1]
    both = served.signals(limit=6)["decisions"]
    assert [row["decision_hash"] for row in both] == \
        [row["decision_hash"] for row in first["decisions"] + second["decisions"]]


def test_the_oldest_page_has_no_cursor_and_nothing_repeats(served):
    seen, cursor, pages = [], None, 0
    while True:
        page = served.risk_targets(limit=1000, cursor=cursor)
        seen += [row["timestamp"] for row in page["targets"]]
        pages += 1
        cursor = page["page"]["next_cursor"]
        if cursor is None:
            assert page["page"]["has_more"] is False
            break
    assert pages == 8 and len(seen) == len(set(seen)) == 7728
    assert seen == sorted(seen, reverse=True)


def test_limits_and_cursors_are_validated(served):
    for bad in (0, -1, 1001, "x"):
        with pytest.raises(AppApiError):
            served.signals(limit=bad)
    cursor = served.signals(limit=2, product="BTC-USD")["page"]["next_cursor"]
    with pytest.raises(AppApiError):
        served.signals(limit=2, product="ETH-USD", cursor=cursor)
    with pytest.raises(AppApiError):
        served.risk_targets(limit=2, cursor=cursor)
    with pytest.raises(AppApiError):
        served.signals(cursor="not-a-cursor")
    with pytest.raises(Exception):
        served.signals(product="btc-usd")


# --- determinism ----------------------------------------------------------------


def test_a_rebuild_is_identical_to_the_committed_files():
    builder = _builder()
    first = builder.build_all(DATA_ROOT)
    assert first == builder.build_all(DATA_ROOT)
    for name, body in first.items():
        assert (DATA_ROOT / "signal_runs_v1" / name).read_bytes() == body
