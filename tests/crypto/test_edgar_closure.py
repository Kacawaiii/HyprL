"""Offline operational safety: synthetic EDGAR stores only; never install a unit or fetch SEC data."""

from datetime import timedelta
import json
import sqlite3
import threading

import pytest

from scripts.trading_lab.edgar import closure, service, snapshot, spec
from scripts.trading_lab.edgar import synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore

A = syn.CIK_A.zfill(10)


def authorization(tmp_path, **changes):
    auth = {"authorizes": spec.PROVIDER_ID, "spec_hash": spec.SPEC_HASH, "ciks": [syn.CIK_A],
            "max_requests": 3, "not_after": "2026-06-18T00:00:00+00:00",
            "user_agent": "Synthetic Lab nobody@example.invalid", "granted_by": "synthetic operator"}
    auth.update(changes)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(auth))
    return path


def capture(tmp_path, *, stopped=False, **bounds):
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[A] = syn.Reply(syn.listing(syn.CIK_A, [syn.filing("0000320193-26-000071")]))
    stop = threading.Event()
    if stopped:
        fetcher.routes[A] = lambda count: (stop.set(), syn.Reply(syn.listing(syn.CIK_A, [])))[1]
    auth = authorization(tmp_path, **bounds)
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=fetcher, stop=stop, log=lambda m: None)
    return clock, auth, summary


def close(tmp_path, auth=None, *, now=None, **kwargs):
    return closure.close(tmp_path / "store", tmp_path / "copy", tmp_path / "report.json",
                         authorization=auth, now=now, **kwargs)


def test_budget_closure_verifies_raws_causal_reads_reopen_replay_health_and_qualification(tmp_path):
    clock, auth, summary = capture(tmp_path)
    before = closure._tree_digest(tmp_path / "store")
    result = close(tmp_path, auth, now=clock.wall())
    assert result["ok"], result
    assert result["integrity"]["verdict"] == "VALID" and result["run"]["verdict"] == "ACCOMPLISHED"
    verified = result["verification"]
    assert verified["raws_verified"] == summary["requests"] == 3
    assert verified["reopen_identical"] == verified["replay_identical"] == 4
    assert verified["health_replay"] and verified["copy_unchanged"]
    assert verified["resolved_reads"] >= 1 and verified["unresolved_reads"] >= 1
    assert set(result["qualification"]["matrix"]) == {f"UV{i}" for i in range(1, 7)}
    assert closure._tree_digest(tmp_path / "store") == before
    persisted = json.loads((tmp_path / "report.json").read_text())
    assert persisted == result and "user_agent" not in json.dumps(persisted)


def test_durable_bounds_suffice_without_the_private_authorization_file(tmp_path):
    clock, auth, _ = capture(tmp_path)
    auth.unlink()
    result = close(tmp_path, now=clock.wall())
    assert result["ok"]


def test_a_run_interrupted_midway_has_valid_integrity_and_is_not_accomplished(tmp_path):
    clock, auth, summary = capture(tmp_path, stopped=True)
    assert summary["reason"] == "stopped" and summary["requests"] == 1
    # Closing long after the expiry must never turn a premature stop into an accomplished run.
    result = close(tmp_path, auth, now=clock.wall() + timedelta(days=2))
    assert result["integrity"]["verdict"] == "VALID"
    assert result["run"]["verdict"] == "NOT_ACCOMPLISHED" and not result["ok"]


def test_expiry_closure_is_separate_from_budget_completion(tmp_path):
    clock, auth, summary = capture(tmp_path, max_requests=50, not_after="2026-06-17T13:30:00+00:00")
    assert summary["reason"] == "authorization expired" and summary["requests"] == 3
    result = close(tmp_path, auth, now=clock.wall())
    assert result["ok"] and result["run"]["expiry_reached"] and not result["run"]["budget_reached"]


def test_legacy_fixture_budget_is_verifiable_without_run_markers(tmp_path):
    clock, auth, _ = capture(tmp_path)
    db = sqlite3.connect(tmp_path / "store" / EdgarStore.DB_NAME)
    db.execute("DELETE FROM rec WHERE kind IN ('RUN_STARTED', 'RUN_ENDED')")
    db.commit()
    db.close()
    result = close(tmp_path, auth, now=clock.wall())
    assert result["ok"] and result["run"]["budget_reached"]


def test_corrupt_raw_is_invalid_even_when_the_budget_was_spent(tmp_path):
    clock, auth, _ = capture(tmp_path)
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    digest = store.rows("RESPONSE")[-1].body["raw_sha"]
    path = store._raw_path(digest)
    store.close()
    path.write_bytes(b"synthetic corruption")
    result = close(tmp_path, auth, now=clock.wall())
    assert result["integrity"]["verdict"] == "INVALID"
    assert result["run"]["verdict"] == "ACCOMPLISHED"
    assert result["verification"]["raw_failures"] and not result["verification"]["health_replay"]


def test_rejected_source_payload_is_valid_evidence_but_unqualified(tmp_path):
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[A] = syn.Reply(b"not a JSON listing")
    auth = authorization(tmp_path, max_requests=1)
    service.run(tmp_path / "store", auth, clock=clock, fetcher=fetcher, log=lambda m: None)
    result = close(tmp_path, auth, now=clock.wall())
    assert result["integrity"]["verdict"] == "VALID"
    assert result["qualification"]["verdict"] == "NOT_QUALIFIED"


def test_health_tampering_is_invalid_including_the_unresolved_tail(tmp_path):
    clock, auth, _ = capture(tmp_path, max_requests=1)
    db = sqlite3.connect(tmp_path / "store" / EdgarStore.DB_NAME)
    rowid, body = db.execute("SELECT rowid, body FROM rec WHERE kind='SOURCE_HEALTH'").fetchone()
    body = dict(json.loads(body), reason="synthetic tampering")
    db.execute("UPDATE rec SET body=? WHERE rowid=?", (json.dumps(body), rowid))
    db.commit()
    db.close()
    result = close(tmp_path, auth, now=clock.wall())
    assert result["integrity"]["verdict"] == "INVALID"
    assert not result["verification"]["health_replay"]


@pytest.mark.parametrize("identity_ok", [True, False])
def test_recorded_reads_keep_their_exact_horizon_and_identity(tmp_path, identity_ok):
    clock, auth, _ = capture(tmp_path)
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    H = store.horizon() - 1
    T = snapshot.parse_iso(store.rows("RESPONSE")[1].body["observed_at"]) + spec.CLOCK_ERROR_BOUND
    snap = snapshot.filings_as_of(store, T, H)
    store.close()
    log = tmp_path / "snapshots.jsonl"
    log.write_text(json.dumps({"T": snap["T"], "H": H, "identity": snap["identity"] if identity_ok else "0" * 64}) + "\n")
    result = close(tmp_path, auth, now=clock.wall(), snapshots_log=log)
    assert result["integrity_ok"] is identity_ok
    assert result["verification"]["recorded_reads"] == 1


def test_owner_and_stop_checks_refuse_to_copy_and_still_write_verdicts(tmp_path):
    clock, auth, _ = capture(tmp_path)
    store = EdgarStore(tmp_path / "store", wall_clock=clock.wall)
    collector = EdgarCollector(store, syn.FakeFetcher(clock), clock)
    before = closure._tree_digest(tmp_path / "store")
    try:
        assert not closure.owner_free(tmp_path / "store")
        result = close(tmp_path, auth, now=clock.wall())
        assert result["integrity"]["verdict"] == "INVALID" and not result["owner_free"]
        assert not (tmp_path / "copy").exists()
        assert closure._tree_digest(tmp_path / "store") == before
    finally:
        collector.close()
        store.close()
    result = close(tmp_path, auth, now=clock.wall(), stopper=lambda: {"was": "active", "stopped": False})
    assert not result["ok"] and not (tmp_path / "copy").exists()


def test_backup_includes_wal_frames_without_changing_the_source(tmp_path):
    clock, auth, _ = capture(tmp_path)
    writer = EdgarStore(tmp_path / "store", wall_clock=clock.wall)
    writer.append("SYNTHETIC", [("SYNTHETIC", None, {"value": 42})])
    # SQLite's read-only WAL reader may update coordination bytes in -shm. Payload files (database,
    # WAL frames and immutable raws) must be unchanged; the standalone copy has no such side file.
    def payloads():
        return {p.relative_to(tmp_path / "store"): p.read_bytes() for p in (tmp_path / "store").rglob("*")
                if p.is_file() and not p.name.endswith("-shm")}
    before = payloads()
    try:
        result = close(tmp_path, auth, now=clock.wall())
        assert result["integrity_ok"] and result["copy_check"]["equal"]
        copied = EdgarStore(tmp_path / "copy", wall_clock=None, read_only=True)
        assert copied.rows("SYNTHETIC")[0].body == {"value": 42}
        copied.close()
        assert payloads() == before
    finally:
        writer.close()


def test_closure_never_writes_outputs_into_the_source(tmp_path):
    clock, auth, _ = capture(tmp_path)
    before = closure._tree_digest(tmp_path / "store")
    with pytest.raises(ValueError):
        closure.close(tmp_path / "store", tmp_path / "copy", tmp_path / "store" / "report.json", authorization=auth)
    assert closure._tree_digest(tmp_path / "store") == before
    result = closure.close(tmp_path / "store", tmp_path / "store" / "copy", tmp_path / "report.json", authorization=auth)
    assert not result["ok"] and not (tmp_path / "store" / "copy").exists()


@pytest.mark.parametrize("microseconds", [0, 123456])
def test_persistent_timer_templates_round_up_and_are_only_written(tmp_path, monkeypatch, microseconds):
    monkeypatch.setattr(closure.subprocess, "run", lambda *a, **k: pytest.fail("templates must not install user units"))
    close_at = syn.START.replace(microsecond=microseconds)
    out = tmp_path / "templates"
    paths = closure.write_closure_units(out, unit="edgar-synthetic", code=tmp_path / "code with spaces",
                                        run=tmp_path / "run", authorization=tmp_path / "authorization.json",
                                        close_at=close_at)
    assert len(paths) == 2
    timer = (out / "edgar-synthetic-closure.timer").read_text()
    line = next(line for line in timer.splitlines() if line.startswith("OnCalendar="))
    fire = snapshot.parse_iso(line[len("OnCalendar="):-len(" UTC")].replace(" ", "T") + "+00:00")
    assert close_at <= fire < close_at + timedelta(seconds=1)
    assert "Persistent=true" in timer and "AccuracySec=1s" in timer
    unit = (out / "edgar-synthetic-closure.service").read_text()
    assert '"scripts.trading_lab.edgar.closure" "close"' in unit
    assert 'WorkingDirectory="' in unit and '"--authorization"' in unit and '"--unit" "edgar-synthetic"' in unit


@pytest.mark.parametrize("operation", ["raw_write", "commit TRANSPORT_INVOKED"])
def test_runner_detects_its_own_stall_records_it_once_and_sends_no_request_during_it(tmp_path, monkeypatch, operation):
    clock = syn.SimClock()
    entered, release, detected = threading.Event(), threading.Event(), threading.Event()
    stores, errors, summaries = [], [], []
    original = service.EdgarStore

    def fault(op):
        if op == operation and not entered.is_set():
            entered.set()
            assert release.wait(5), "test must release the synthetic stall"

    def store_factory(*args, **kwargs):
        store = original(*args, **kwargs)
        store.fault = fault
        stores.append(store)
        return store

    monkeypatch.setattr(service, "EdgarStore", store_factory)
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[A] = syn.Reply(syn.listing(syn.CIK_A, []))
    fetch = fetcher.fetch

    def checked_fetch(*args, **kwargs):
        if entered.is_set():
            assert release.is_set()
            assert len(stores[0].rows("STORAGE_INCIDENT")) == 1  # recorded before the next physical start
        return fetch(*args, **kwargs)

    fetcher.fetch = checked_fetch
    auth = authorization(tmp_path)

    def log(message):
        if "STORAGE_INCIDENT_STARTED" in message:
            detected.set()

    def run():
        try:
            summaries.append(service.run(tmp_path / "store", auth, clock=clock, fetcher=fetcher,
                                         log=log, monitor_interval_s=0.01))
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert entered.wait(5)
        sent = len(fetcher.requests)
        clock.sleep(11)
        assert detected.wait(5)  # the separate monitor detects a blocked runner
        assert len(fetcher.requests) == sent
        assert stores[0].stalled()[1] >= 10
    finally:
        release.set()
        runner.join(5)
    assert not runner.is_alive() and not errors, errors
    assert summaries[0]["requests"] == 3
    store = original(tmp_path / "store", wall_clock=None, read_only=True)
    incidents = store.rows("STORAGE_INCIDENT")
    assert len(incidents) == 1 and incidents[0].body["stalled_s"] >= 11
    assert incidents[0].body["operation"] == ("raw_write" if operation == "raw_write" else "transaction TRANSPORT_INVOKED")
    store.close()


def test_restart_does_not_recreate_the_same_authorization_budget(tmp_path):
    clock, auth, _ = capture(tmp_path)
    fetcher = syn.FakeFetcher(clock)
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=fetcher, log=lambda m: None)
    assert summary["reason"] == "request budget spent" and summary["requests"] == 3
    assert fetcher.requests == []


@pytest.mark.parametrize("termination", ["stop", "expiry"])
def test_a_stalled_invocation_rechecks_stop_and_expiry_before_sending(tmp_path, monkeypatch, termination):
    clock = syn.SimClock()
    entered, release, detected, stop = (threading.Event() for _ in range(4))
    original = service.EdgarStore
    errors, summaries = [], []

    def fault(operation):
        if operation == "commit TRANSPORT_INVOKED":
            entered.set()
            assert release.wait(5)

    def factory(*args, **kwargs):
        store = original(*args, **kwargs)
        store.fault = fault
        return store

    monkeypatch.setattr(service, "EdgarStore", factory)
    auth = authorization(tmp_path, not_after="2026-06-17T13:05:00+00:00")
    fetcher = syn.FakeFetcher(clock)

    def log(message):
        if "STORAGE_INCIDENT_STARTED" in message:
            detected.set()

    def run():
        try:
            summaries.append(service.run(tmp_path / "store", auth, clock=clock, fetcher=fetcher, stop=stop,
                                         log=log, monitor_interval_s=0.01))
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert entered.wait(5)
        clock.sleep(11 if termination == "stop" else 300)
        if termination == "stop":
            stop.set()
        assert detected.wait(5)
        assert fetcher.requests == []
    finally:
        release.set()
        runner.join(5)
    assert not runner.is_alive() and not errors, errors
    assert fetcher.requests == []
    assert summaries[0]["reason"] == ("stopped" if termination == "stop" else "authorization expired")
    result = close(tmp_path, auth, now=clock.wall())
    assert result["integrity_ok"] and result["verification"]["progress"]["storage_incidents"] == 1
    if termination == "stop":
        assert not result["run_ok"]


def test_closure_confirms_the_user_unit_has_stopped_without_installing_anything(monkeypatch):
    calls = []
    states = iter(["active", "inactive"])

    def systemctl(command, **kwargs):
        calls.append(command)
        return closure.subprocess.CompletedProcess(command, 0, next(states) if "is-active" in command else "", "")

    monkeypatch.setattr(closure.subprocess, "run", systemctl)
    result = closure.stop_service("edgar-synthetic")
    assert result == {"was": "active", "left": "inactive", "stopped": True}
    assert calls == [["systemctl", "--user", "is-active", "edgar-synthetic"],
                     ["systemctl", "--user", "stop", "edgar-synthetic"],
                     ["systemctl", "--user", "is-active", "edgar-synthetic"]]


@pytest.mark.parametrize("read_only_directory", [True, False])
def test_published_read_only_copy_needs_no_owner_file_but_writable_stores_require_it(tmp_path, read_only_directory):
    clock, auth, _ = capture(tmp_path)
    source = tmp_path / "store"
    (source / "owner.lock").unlink()
    if read_only_directory:
        source.chmod(0o500)
    try:
        result = close(tmp_path, auth, now=clock.wall())
        assert result["integrity_ok"] is read_only_directory
        assert not (source / "owner.lock").exists()
        if read_only_directory:
            assert "read-only snapshot directory" in result["owner_check"]
    finally:
        source.chmod(0o700)
