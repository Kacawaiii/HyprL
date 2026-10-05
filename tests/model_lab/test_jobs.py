import json
import os
import time

import pytest

from scripts.trading_lab.platform.jobs import JobRunner, JobStore, ResourceLimits, TERMINAL
from scripts.trading_lab.platform.model_lab_demo import wait


def terminal(store, identifier, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = store.status(identifier)
        if status["state"] in TERMINAL:
            return status
        time.sleep(.05)
    raise AssertionError("worker did not reach a terminal state")


def test_worker_pid_persistent_state_progress_logs_and_results(tmp_path):
    with JobRunner(tmp_path) as runner:
        identifier = runner.store.submit("dataset", {"products": ["BTC-USD"], "bars": 80})
        result = wait(runner.store, identifier, 30)
        status = runner.store.status(identifier)
        assert status["worker_pid"] != os.getpid()
        assert status["state"] == "COMPLETE" and status["progress"] == 1
        assert [r["progress"] for r in status["logs"]] == sorted(r["progress"] for r in status["logs"])
        assert [r["code"] for r in status["logs"]] == ["QUEUED", "STARTED", "DATASET_BUILT", "RESULT_READY", "COMPLETE"]
        assert str(tmp_path) not in json.dumps(status)
    reopened = JobStore(tmp_path)
    assert reopened.result(identifier)["result"] == result
    assert reopened.cancel(identifier)["state"] == "COMPLETE"


def test_cancellation_before_start_is_durable(tmp_path):
    store = JobStore(tmp_path)
    identifier = store.submit("dataset", {"bars": 80})
    assert store.cancel(identifier)["state"] == "CANCELLED"
    with JobRunner(tmp_path) as runner:
        assert runner.store.status(identifier)["worker_pid"] is None
        assert runner.store.result(identifier)["result"] is None


def test_running_cancellation_terminates_worker_and_releases_queue(tmp_path):
    with JobRunner(tmp_path) as runner:
        first = runner.store.submit("dataset", {"bars": 600})
        second = runner.store.submit("dataset", {"products": ["BTC-USD"], "bars": 80})
        deadline = time.monotonic() + 20
        while runner.store.status(first)["progress"] == 0 and time.monotonic() < deadline:
            assert runner.store.status(second)["state"] == "QUEUED"
            time.sleep(.02)
        assert runner.store.status(first)["progress"] > 0
        runner.store.cancel(first)
        assert terminal(runner.store, first)["state"] == "CANCELLED"
        assert wait(runner.store, second, 30)["synthetic"] is True


def test_wall_budget_kills_noncompleted_work(tmp_path):
    with JobRunner(tmp_path) as runner:
        identifier = runner.store.submit("dataset", {"bars": 600}, ResourceLimits(wall_seconds=1))
        status = terminal(runner.store, identifier)
        assert status["state"] == "FAILED"
        assert status["error_code"] == "WALL_LIMIT"
        assert runner.store.result(identifier)["result"] is None


def test_small_memory_budget_fails_in_worker_without_affecting_parent(tmp_path):
    with JobRunner(tmp_path) as runner:
        identifier = runner.store.submit("dataset", {"bars": 600}, ResourceLimits(memory_mb=256))
        assert terminal(runner.store, identifier)["state"] == "FAILED"
        assert runner.store.status(identifier)["worker_pid"] != os.getpid()
        assert runner.store.submit("dataset", {"bars": 80})


def test_cpu_budget_terminates_worker(tmp_path):
    with JobRunner(tmp_path) as runner:
        identifier = runner.store.submit("dataset", {"bars": 600}, ResourceLimits(cpu_seconds=1))
        status = terminal(runner.store, identifier)
        assert status["state"] == "FAILED" and status["error_code"] == "WORKER_EXIT"
        assert status["progress"] < 1


def test_single_owner_and_interrupted_worker_recovery(tmp_path):
    store = JobStore(tmp_path)
    identifier = store.submit("dataset", {"bars": 80})
    with store.connect() as db:
        db.execute("UPDATE jobs SET state='RUNNING' WHERE id=?", (identifier,))
    with JobRunner(tmp_path) as runner:
        assert runner.store.status(identifier)["state"] == "FAILED"
        assert runner.store.status(identifier)["error_code"] == "RUNNER_INTERRUPTED"
        with pytest.raises(ValueError, match="already owns"):
            JobRunner(tmp_path)


def test_safe_error_codes_and_immutable_artifacts(tmp_path):
    with JobRunner(tmp_path) as runner:
        secret = "synthetic-private-location"
        identifier = runner.store.submit("dataset", {"unexpected": secret})
        status = terminal(runner.store, identifier)
        assert status["state"] == "FAILED" and status["error_code"] == "WORKLOAD_TypeError"
        assert secret not in json.dumps(status)
        key = runner.store.put_artifact("model", {"synthetic": True})
        assert runner.store.put_artifact("model", {"synthetic": True}) == key
        with pytest.raises(ValueError, match="collision"):
            runner.store.put_artifact("model", {"synthetic": False}, identity=key)


@pytest.mark.parametrize("limits", [{"wall_seconds": 181}, {"cpu_seconds": 0}, {"memory_mb": 2048},
                                    {"output_mb": 64}, {"wall_seconds": True}])
def test_resource_limit_validation(limits):
    with pytest.raises(ValueError):
        ResourceLimits(**limits)


def test_queue_and_dispatch_budgets(tmp_path):
    store = JobStore(tmp_path)
    for _ in range(8):
        store.submit("dataset", {})
    with pytest.raises(ValueError, match="queue budget"):
        store.submit("dataset", {})
    with pytest.raises(ValueError, match="unknown job"):
        store.submit("capture", {})
    with pytest.raises(ValueError, match="payload too large"):
        store.submit("dataset", {"noise": "s" * 65536})
