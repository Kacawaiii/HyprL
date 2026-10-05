from datetime import datetime, timezone
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.trading_lab.app_api.server import make_server
from scripts.trading_lab.ops.control import operate, save, versions
from scripts.trading_lab.ops.telemetry import health, jobs, age
from scripts.trading_lab.platform.jobs import JobStore
from scripts.trading_lab.edgar.closure import _tree_digest


def test_worker_read_retains_bytes_and_reports_lifetime_budgets(config):
    root = config["runtime_root"]
    store = JobStore(root / "lab")
    job = store.submit("dataset", {"products": ["BTC-USD"]})
    store.finish(job, "FAILED", error_code="MEMORY_LIMIT")
    before = _tree_digest(root)
    report = health(ops_root=root)
    assert report["workers"]["budgets"]["jobs_used"] == 1
    assert report["workers"]["budgets"]["jobs_remaining"] == 999
    assert report["errors"] == ["MEMORY_LIMIT"]
    assert _tree_digest(root) == before


def test_http_health_get_head_read_only_and_no_client_paths(config):
    server = make_server("data/crypto", port=0, ops_root=config["runtime_root"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port) + "/api/v1/ops/health"
    try:
        with urlopen(base) as response:
            payload = json.load(response)
        assert payload["read_only"] and payload["running_versions"] == versions()
        assert payload["sources"]["fomc"]["state"] == "NOT_CONFIGURED"
        assert str(config["runtime_root"]) not in json.dumps(payload)
        assert not config["runtime_root"].exists()
        with urlopen(Request(base, method="HEAD")) as response:
            assert response.read() == b""
        for request, expected in ((Request(base, method="POST", data=b"{}"), 405),
                                  (Request(base + "?root=anything"), 400)):
            with pytest.raises(HTTPError) as caught:
                urlopen(request)
            assert caught.value.code == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_service_versions_are_recorded_at_start_and_status_disappears_after_stop(config):
    operate(config, "start", "app")
    root = config["runtime_root"]
    report = health(ops_root=root)
    assert report["services"]["app"]["running_versions"]["git_sha"] == versions()["git_sha"]
    operate(config, "stop", "app")
    report = health(ops_root=root)
    assert report["services"]["app"]["running_versions"] is None
    assert report["last_operations"][-1]["action"] == "stop"


def test_invalid_private_telemetry_does_not_leak_paths_or_exception_strings(config):
    root = config["runtime_root"]
    root.mkdir()
    save(root / "operations.json", [{"at": "2026-01-01T00:00:00+00:00", "action": "start", "service": "app",
        "state": "COMPLETE", "code": "secret/private/location", "token": "synthetic-private-token"}])
    store = JobStore(root / "lab")
    job = store.submit("dataset", {})
    store.finish(job, "FAILED", error_code="secret/private/location")
    encoded = json.dumps(health(ops_root=root))
    assert "secret/private" not in encoded and "synthetic-private-token" not in encoded
    assert "WORKLOAD_ERROR" in encoded


def test_stale_edgar_status_is_not_healthy_and_durable_budget_is_read_only(config):
    from scripts.trading_lab.edgar.store import EdgarStore
    from scripts.trading_lab.edgar import spec
    root = config["runtime_root"]
    at = datetime(2026, 6, 17, tzinfo=timezone.utc)
    store = EdgarStore(root / "edgar", wall_clock=lambda: at)
    store.append("RUN_STARTED", [("RUN_STARTED", "synthetic-epoch", {"epoch": "synthetic-epoch",
        "authorization_sha256": "a" * 64, "authorization": {"max_requests": 3,
        "not_after": "2026-06-18T00:00:00+00:00"}})])
    store.close()
    save(root / "edgar" / "service-status.json", {"pid": 9999999, "state": "running",
        "updated_at": "2026-06-17T00:00:00+00:00", "grants_suspended": True, "storage_incident": {"private": "never-exposed"},
        "pending_incidents": 1})
    before = _tree_digest(root)
    report = health(ops_root=root, now=datetime(2026, 6, 19, tzinfo=timezone.utc))
    assert "EDGAR_STATUS_STALE" in report["errors"] and "EDGAR_STORAGE_INCIDENT" in report["errors"]
    assert report["edgar_service"]["budgets"]["expired"]
    assert report["edgar_service"]["budgets"]["requests_remaining"] == 3
    assert "never-exposed" not in json.dumps(report)
    assert _tree_digest(root) == before


def test_future_or_naive_telemetry_is_unknown():
    at = datetime(2026, 6, 17, tzinfo=timezone.utc)
    assert age("2026-06-18T00:00:00+00:00", at) is None
    assert age("2026-06-17T00:00:00", at) is None


def test_missing_worker_schema_is_an_integrity_error(config):
    import sqlite3
    root = config["runtime_root"]
    (root / "lab").mkdir(parents=True)
    sqlite3.connect(root / "lab" / "jobs.sqlite").close()
    result = health(ops_root=root)
    assert result["workers"]["state"] == "INTEGRITY_ERROR"
    assert result["status"] == "DEGRADED"


def test_source_integrity_failure_is_visible_and_never_leaks_exception_text():
    class BrokenSource:
        def status(self):
            return {"status": "AVAILABLE", "spec_hash": "a" * 64, "suggested_as_of": "2026-06-17T00:00:00+00:00", "horizon": 7}
        def snapshot(self, **kwargs):
            raise ValueError("synthetic-private-source-path")
    result = health(fomc=BrokenSource())
    assert result["sources"]["fomc"]["state"] == "INTEGRITY_ERROR"
    assert result["errors"] == ["FOMC_READ_FAILED"]
    assert "synthetic-private" not in json.dumps(result)


def test_attested_provider_error_is_separate_from_store_availability():
    class UnavailableSource:
        def status(self):
            return {"status": "AVAILABLE", "spec_hash": "a" * 64, "suggested_as_of": "2026-06-17T00:00:00+00:00", "horizon": 7}
        def snapshot(self, **kwargs):
            return {"snapshot": {"read_state": "FOMC_RESOLVED"}, "health": {"primary_statement": {
                "result_state": "SOURCE_UNAVAILABLE", "reason": "SOURCE_UNAVAILABLE", "check_at": "2026-06-16T00:00:00+00:00"}}}
    result = health(fomc=UnavailableSource())
    assert result["sources"]["fomc"]["state"] == "AVAILABLE"
    assert result["errors"] == ["FOMC_SOURCE_UNAVAILABLE"]
    assert result["status"] == "DEGRADED"


def test_mutated_worker_state_and_pid_never_exfiltrate_private_values(config):
    root = config["runtime_root"]
    store = JobStore(root / "lab")
    job = store.submit("dataset", {})
    for field, value in (("state", "synthetic-private-location"), ("worker_pid", "synthetic-private-location")):
        with store.connect() as db:
            db.execute("UPDATE jobs SET state='RUNNING',worker_pid=NULL WHERE id=?", (job,))
            db.execute("UPDATE jobs SET " + field + "=? WHERE id=?", (value, job))
        result = health(ops_root=root)
        assert result["workers"]["state"] == "INTEGRITY_ERROR"
        assert "synthetic-private-location" not in json.dumps(result)
