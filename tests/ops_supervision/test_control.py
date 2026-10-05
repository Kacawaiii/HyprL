import json
import os
from pathlib import Path
import signal
import socket
import time

import pytest

from scripts.trading_lab.ops.control import (OpsRefused, load_config, local_path, inspect, owner, operate,
    read_json, save, start, stop, verify, edgar_preflight, versions)
from scripts.trading_lab.platform.jobs import JobStore


def test_config_rejects_public_file_permissions_and_unknown_commands(config):
    path = config["config_path"]
    path.chmod(0o644)
    with pytest.raises(OpsRefused, match="PRIVATE_OPERATOR_FILE"):
        load_config(path)
    path.chmod(0o600)
    raw = read_json(path)
    raw["command"] = ["arbitrary-process"]
    save(path, raw)
    with pytest.raises(OpsRefused, match="SCHEMA"):
        load_config(path)


def test_mutations_cannot_escape_worktree_or_follow_external_symlink(config, tmp_path):
    with pytest.raises(OpsRefused, match="WORKTREE_VAR"):
        local_path(tmp_path)
    root = config["runtime_root"]
    root.parent.mkdir(exist_ok=True)
    root.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(OpsRefused, match="WORKTREE_VAR"):
            load_config(config["config_path"])
    finally:
        root.unlink()


def test_read_only_sources_must_be_disjoint_from_runtime(config):
    path = config["config_path"]
    data = read_json(path)
    data["edgar_archive"] = str(config["runtime_root"] / "edgar")
    save(path, data)
    with pytest.raises(OpsRefused, match="SEPARATE"):
        load_config(path)


def test_verify_does_not_create_a_runtime_or_telemetry(config):
    result = verify(config)
    assert result["verification"] == "PASS"
    assert result["workers"]["state"] == "NOT_OBSERVED"
    assert not config["runtime_root"].exists()
    assert result["running_versions"] == versions()


def test_one_controller_owner(config):
    with owner(config["runtime_root"]):
        with pytest.raises(OpsRefused, match="OWNER_BUSY"):
            with owner(config["runtime_root"]):
                pass


def test_foreign_and_reused_pid_are_never_signalled(config, monkeypatch):
    from scripts.trading_lab.ops import control
    root = config["runtime_root"]
    root.mkdir()
    save(root / "app.pid.json", {"pid": os.getpid()})
    signals = []
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *args: signals.append(args))
    with pytest.raises(OpsRefused, match="IDENTITY_REFUSED"):
        stop(config, "app")
    assert inspect(root, "app")["state"] == "FOREIGN" and not signals
    # A process with the right arguments but different kernel start ticks is refused.
    fake = os.getpid() + 100000
    argv = ["python", "hyprl-ops-app"]
    save(root / "app.pid.json", {"pid": fake, "boot_id": control.boot_id(), "start_ticks": 1,
        "argv": argv, "service": "app", "started_at": control.stamp()})
    monkeypatch.setattr(control, "process_exists", lambda pid: True)
    monkeypatch.setattr(control, "process_start_ticks", lambda pid: 2)
    monkeypatch.setattr(control, "command_line", lambda pid: [s.encode() for s in argv])
    assert inspect(root, "app")["state"] == "FOREIGN"
    with pytest.raises(OpsRefused):
        stop(config, "app")
    assert not signals


def test_app_start_idempotence_stop_resume_and_occupied_port(config):
    first = operate(config, "start", "app")
    pid = first["app"]["pid"]
    assert operate(config, "start", "app")["app"] == {**inspect(config["runtime_root"], "app"), "started": False}
    assert operate(config, "stop", "app")["app"]["stopped"]
    assert operate(config, "resume", "app")["app"]["pid"] != pid
    operate(config, "stop", "app")
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", config["port"]))
        holder.listen()
        with pytest.raises(OpsRefused, match="START_FAILED"):
            operate(config, "start", "app")
    assert read_json(config["runtime_root"] / "operations.json")[-1]["state"] == "BLOCKED"


def test_workers_complete_queued_work_and_resume_after_clean_stop(config):
    root = config["runtime_root"]
    store = JobStore(root / "lab")
    first = store.submit("dataset", {"products": ["BTC-USD"], "bars": 120})
    operate(config, "start", "workers")
    deadline = time.monotonic() + 30
    while store.status(first)["state"] not in ("COMPLETE", "FAILED") and time.monotonic() < deadline:
        time.sleep(.1)
    status = store.status(first)
    assert status["state"] == "COMPLETE", status
    assert status["worker_pid"] != inspect(root, "workers")["pid"]
    operate(config, "stop", "workers")
    second = store.submit("dataset", {"products": ["BTC-USD"], "bars": 120})
    operate(config, "resume", "workers")
    deadline = time.monotonic() + 30
    while store.status(second)["state"] not in ("COMPLETE", "FAILED") and time.monotonic() < deadline:
        time.sleep(.1)
    assert store.status(second)["state"] == "COMPLETE"


def test_capture_requires_separate_explicit_scope_and_never_starts_by_default(config, monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.trading_lab.ops.control.subprocess.Popen", lambda *a, **k: calls.append(a))
    with pytest.raises(OpsRefused, match="WAITING_AUTHORIZATION"):
        operate(config, "start", "edgar")
    with pytest.raises(OpsRefused, match="GRANT_REQUIRED"):
        edgar_preflight(config, allow_capture=True)
    assert not calls


def test_unknown_pid_start_ticks_and_boot_ids_fail_closed(config, monkeypatch):
    operate(config, "start", "app")
    path = config["runtime_root"] / "app.pid.json"
    original = read_json(path)
    try:
        for field, value in (("start_ticks", None), ("boot_id", "other-boot"), ("argv", ["wrong-argv"])):
            save(path, dict(original, **{field: value}))
            assert inspect(config["runtime_root"], "app")["state"] == "FOREIGN"
            with pytest.raises(OpsRefused):
                stop(config, "app")
    finally:
        save(path, original)


def test_pidfd_unavailable_is_blocked_without_fallback(config, monkeypatch):
    operate(config, "start", "app")
    def denied(_):
        raise OSError()
    monkeypatch.setattr(os, "pidfd_open", denied)
    with pytest.raises(OpsRefused, match="PIDFD_UNAVAILABLE"):
        stop(config, "app")
    assert inspect(config["runtime_root"], "app")["state"] == "RUNNING"


def test_edgar_preflight_uses_the_bound_store_database_name(config, monkeypatch):
    from datetime import datetime, timezone
    from scripts.trading_lab.edgar.store import EdgarStore
    public = {"authorization_sha256": "a" * 64, "requests_consumed": 2, "requests_remaining": 1,
              "not_after": "2099-01-01T00:00:00+00:00"}
    supplied = dict(config, edgar_authorization=config["config_path"])
    monkeypatch.setattr("scripts.trading_lab.edgar.service.check", lambda *a: public)
    with pytest.raises(OpsRefused, match="ORIGINAL_ACCOUNTING_STORE"):
        edgar_preflight(supplied, allow_capture=True)
    store = EdgarStore(config["runtime_root"] / "edgar", wall_clock=lambda: datetime.now(timezone.utc))
    store.close()
    assert edgar_preflight(supplied, allow_capture=True) == public


def test_default_all_never_launches_edgar_even_with_capture_flag(config, monkeypatch):
    called = []
    def started(configuration, service, **kwargs):
        called.append(service)
        return {"state": "RUNNING"}
    monkeypatch.setattr("scripts.trading_lab.ops.control.start", started)
    operate(config, "start", allow_capture=True)
    assert called == ["workers", "app"]


def test_drain_timeout_preserves_the_owner_and_never_escalates(config, monkeypatch):
    operate(config, "start", "app")
    sent = []
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda descriptor, sig: sent.append(sig))
    with pytest.raises(OpsRefused, match="DRAIN_TIMEOUT"):
        stop(config, "app", timeout=.01)
    assert sent == [signal.SIGTERM]
    assert inspect(config["runtime_root"], "app")["state"] == "RUNNING"


def test_unknown_operation_never_creates_state(config):
    with pytest.raises(OpsRefused, match="UNKNOWN_OPERATION"):
        operate(config, "unrecognized-operation")
    assert not config["runtime_root"].exists()
