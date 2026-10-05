"""Bounded service qualification using synthetic bytes in the observed SEC format; no sockets."""

from datetime import timedelta
from email.utils import format_datetime
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time

import pytest

from scripts.trading_lab.edgar import closure, service, snapshot, spec, synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.edgar.transport import HttpsFetcher
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.store import Rejected

A = syn.CIK_A.zfill(10)
B = syn.CIK_B.zfill(10)


def authorization(root, **changes):
    auth = {"authorizes": spec.PROVIDER_ID, "spec_hash": spec.SPEC_HASH, "ciks": [syn.CIK_A, syn.CIK_B],
            "max_requests": 4, "not_after": snapshot.iso(syn.START + timedelta(days=1)),
            "user_agent": "Synthetic Lab nobody@example.invalid", "granted_by": "synthetic operator"}
    auth.update(changes)
    path = root / "authorization.json"
    path.write_text(json.dumps(auth))
    return path


class Wire:
    """Fake HTTPS connection serving gzip and every header name observed in the fixture trial."""

    def __init__(self, clock, *, status=200, on_read=None):
        self.clock, self.status, self.on_read = clock, status, on_read
        self.requests = []

    def connection(self, host, timeout, context):
        wire = self

        class Connection:
            def request(self, method, path, headers):
                cik = path.removeprefix("/submissions/CIK").removesuffix(".json")
                wire.requests.append((cik, wire.clock.mono()))
                # Both int and null occur in the real extra column; neither is normalized.
                filings = [dict(syn.filing(f"{cik}-26-000001"), isXBRLNumeric=None),
                           syn.filing(f"{cik}-26-000002", form="8-K/A")]
                self.body = gzip.compress(syn.listing(cik, filings))

            def getresponse(self):
                return self

            @property
            def status(self):
                return wire.status

            def getheaders(self):
                return [("access-control-allow-origin", "*"), ("connection", "keep-alive"),
                        ("content-encoding", "gzip"), ("content-length", str(len(self.body))),
                        ("content-type", "application/json"),
                        ("date", format_datetime(wire.clock.wall(), usegmt=True)),
                        ("strict-transport-security", "max-age=31536000; includeSubDomains; preload"),
                        ("vary", "Accept-Encoding"), ("x-amz-apigw-id", "synthetic"),
                        ("x-amzn-requestid", "synthetic"), ("x-amzn-trace-id", "synthetic")]

            def read(self, limit):
                if wire.on_read:
                    wire.on_read()
                chunk, self.body = self.body[:limit], self.body[limit:]
                return chunk

            def close(self):
                pass

        return Connection()

    def fetcher(self, *, fenced=False):
        return HttpsFetcher("Synthetic Lab nobody@example.invalid", connection_factory=self.connection,
                            wall=self.clock.wall, mono=self.clock.mono, fenced=fenced)


def close(root, auth, clock):
    return closure.close(root / "store", root / "copy", root / "closure.json", authorization=auth, now=clock.wall())


def test_observed_wire_format_full_bound_and_closure(tmp_path):
    clock = syn.SimClock()
    wire = Wire(clock)
    auth = authorization(tmp_path)
    service.check(tmp_path / "store", auth, now=clock.wall())
    assert not (tmp_path / "store").exists()
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert summary["requests"] == len(wire.requests) == 4
    status = json.loads((tmp_path / "store" / "service-status.json").read_text())
    assert status["state"] == "ended" and status["reason"] == "request budget spent"
    auth_hash = sha256_canonical(service.load_authorization(auth, syn.START))
    with pytest.raises(service.CaptureRefused, match="budget spent"):
        service.check(tmp_path / "store", auth, now=clock.wall())
    result = close(tmp_path, auth, clock)
    assert result["ok"], result
    qualification = result["qualification"]
    artifact = json.loads(Path("docs/artifacts/edgar_fixture_qualification_v1.json").read_text())
    uv1 = qualification["matrix"]["UV1"]["observation"]
    assert uv1["columns_in_every_listing"] == artifact["matrix"]["UV1"]["observation"]["columns_in_every_listing"]
    assert uv1["columns_not_in_spec"] == ["core_type", "isXBRLNumeric"] and not uv1["type_mismatches"]
    assert sorted(qualification["header_names"]) == artifact["response_header_names"]
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    try:
        assert {r.body["authorization_sha256"] for r in store.rows("TRANSPORT_INVOKED")} == {auth_hash}
        response = store.rows("RESPONSE")[0]
        doc = json.loads(store.read_raw(response.body["raw_sha"]))
        assert doc["cik"] == A and doc["filings"]["recent"]["isXBRLNumeric"] == [None, 1]
        assert response.body["content_encoding_lines"] == ["gzip"]
        assert all(r.body["outcome"] == "LISTING_CLASSIFIED" for r in store.rows("PROCESSING_OUTCOME"))
    finally:
        store.close()


def test_restart_mid_budget_counts_all_epochs_and_reservations(tmp_path):
    clock = syn.SimClock()
    stop = threading.Event()
    wire = Wire(clock, on_read=stop.set)
    auth = authorization(tmp_path, max_requests=3)
    first = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), stop=stop, log=lambda m: None)
    assert first["requests"] == 1 and first["reason"] == "stopped"
    wire.on_read = None
    # Persist a reservation as if the owner crashed before it could send or record an outcome.
    store = EdgarStore(tmp_path / "store", wall_clock=clock.wall)
    auth_hash = sha256_canonical(service.load_authorization(auth, clock.wall()))
    store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", None, {
        "epoch": first["epoch"], "cik": A, "url": "https://data.sec.gov/submissions/CIK0000320193.json",
        "grant_mono": clock.mono(), "authorization_sha256": auth_hash})])
    store.close()
    second = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert second["epoch"] != first["epoch"] and second["requests"] == 3
    assert len(wire.requests) == 2  # the crash reservation consumed the third budget slot
    assert wire.requests[1][1] - wire.requests[0][1] >= spec.POLL_INTERVAL_S
    exhausted = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert exhausted["requests"] == 3 and len(wire.requests) == 2
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    try:
        assert [r.body["outcome"] for r in store.rows("ATTEMPT_OUTCOME")].count("INTERRUPTED") == 1
    finally:
        store.close()
    result = close(tmp_path, auth, clock)
    assert result["integrity_ok"] and not result["run_ok"]


@pytest.mark.parametrize("status", [403, 429])
def test_throttle_termination_is_atomic_and_refuses_restart_even_without_run_ended(tmp_path, monkeypatch, status):
    clock = syn.SimClock()
    wire = Wire(clock, status=status)
    auth = authorization(tmp_path)
    original = EdgarStore.append

    def crash_before_end(store, kind, *args, **kwargs):
        if kind == "RUN_ENDED":
            raise RuntimeError("synthetic owner crash")
        return original(store, kind, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(EdgarStore, "append", crash_before_end)
        with pytest.raises(RuntimeError, match="synthetic owner crash"):
            service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    for action in (service.check, service.run):
        kwargs = {"now": clock.wall()} if action is service.check else {
            "clock": clock, "fetcher": wire.fetcher(), "log": lambda m: None}
        with pytest.raises(service.CaptureRefused, match="terminated"):
            action(tmp_path / "store", auth, **kwargs)
    assert len(wire.requests) == 1
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    try:
        termination = store.rows("AUTHORIZATION_TERMINATED")
        outcome = store.rows("ATTEMPT_OUTCOME")
        assert len(termination) == len(outcome) == 1
        assert termination[0].seq == outcome[0].seq and termination[0].body["status"] == status
    finally:
        store.close()
    result = close(tmp_path, auth, clock)
    assert result["integrity_ok"] and not result["run_ok"]


@pytest.mark.parametrize("operation", ["connect", "headers", "read", "decode", "close", "delivery"])
def test_hung_transport_is_killed_reaped_and_never_returns_late_bytes(monkeypatch, operation):
    monkeypatch.setattr(spec, "DEADLINE_S", 0.2)
    clock = service.RealClock(threading.Event())
    started = multiprocessing.get_context("fork").Event()

    def hang():
        started.set()
        threading.Event().wait(10)

    wire = Wire(clock, on_read=hang if operation == "read" else None)
    from scripts.trading_lab.edgar import transport

    def connection(*args, **kwargs):
        if operation == "connect":
            hang()
        conn = wire.connection(*args, **kwargs)
        if operation == "headers":
            conn.getresponse = hang
        elif operation == "close":
            conn.close = hang
        return conn

    if operation == "decode":
        monkeypatch.setattr(transport, "decode_body", lambda *args: hang())
    elif operation == "delivery":
        from multiprocessing.connection import Connection
        monkeypatch.setattr(Connection, "send", lambda *args: hang())
    fetcher = HttpsFetcher("Synthetic Lab nobody@example.invalid", connection_factory=connection,
                          wall=clock.wall, mono=clock.mono, fenced=True)
    before = {p.pid for p in multiprocessing.active_children()}
    at = time.monotonic()
    result = fetcher.fetch("https://data.sec.gov/submissions/CIK0000320193.json")
    assert started.is_set() and time.monotonic() - at < 2
    assert result.kind == "SOURCE_UNAVAILABLE" and "physical deadline" in result.reason and result.body is None
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_deadline_includes_decode(tmp_path, monkeypatch):
    clock = syn.SimClock()
    wire = Wire(clock)
    from scripts.trading_lab.edgar import transport
    decode = transport.decode_body

    def slow_decode(*args):
        clock.sleep(31)
        return decode(*args)

    monkeypatch.setattr(transport, "decode_body", slow_decode)
    auth = authorization(tmp_path, max_requests=1)
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert summary["requests"] == 1 and summary["records"] == 0
    result = close(tmp_path, auth, clock)
    assert result["ok"] and result["verification"]["health_replay"]
    assert result["qualification"]["attempt_outcomes"] == {"SOURCE_UNAVAILABLE": 1}


def test_sigterm_finishes_inflight_body_processes_it_and_releases_ownership(tmp_path):
    context = multiprocessing.get_context("fork")
    entered, release = context.Event(), context.Event()
    auth = authorization(tmp_path)

    def child():
        clock = syn.SimClock()

        def wait_for_stop():
            entered.set()
            assert release.wait(5)

        wire = Wire(clock, on_read=wait_for_stop)
        run = service.run
        service.run = lambda store, authorization, **kw: run(
            store, authorization, clock=clock, fetcher=wire.fetcher(fenced=True), **kw)
        raise SystemExit(service.main(["--store", str(tmp_path / "store"), "--authorization", str(auth)]))

    worker = context.Process(target=child)
    worker.start()
    try:
        assert entered.wait(5)
        os.kill(worker.pid, signal.SIGTERM)
        release.set()
        worker.join(5)
        assert not worker.is_alive() and worker.exitcode == 0
    finally:
        release.set()
        if worker.is_alive():
            worker.kill()
        worker.join()
        worker.close()
    assert closure.owner_free(tmp_path / "store")
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    try:
        assert len(store.rows("RESPONSE")) == len(store.rows("PROCESSING_OUTCOME")) == 1
        assert store.rows("PROCESSING_OUTCOME")[0].body["outcome"] == "LISTING_CLASSIFIED"
        assert store.rows("RUN_ENDED")[0].body["reason"] == "stopped"
    finally:
        store.close()


def test_status_is_readable_during_a_storage_stall_and_cleared_after_recovery(tmp_path, monkeypatch):
    clock = syn.SimClock()
    entered, release, detected = (threading.Event() for _ in range(3))
    original = service.EdgarStore
    failures = []

    def factory(*args, **kwargs):
        store = original(*args, **kwargs)

        def fault(operation):
            if operation == "commit TRANSPORT_INVOKED" and not entered.is_set():
                entered.set()
                assert release.wait(5)

        store.fault = fault
        return store

    monkeypatch.setattr(service, "EdgarStore", factory)
    auth = authorization(tmp_path, max_requests=2)
    wire = Wire(clock)

    def runner():
        try:
            service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(),
                        monitor_interval_s=0.01, log=lambda line: detected.set() if "STARTED" in line else None)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        clock.sleep(11)
        assert detected.wait(5)
        deadline = time.monotonic() + 2
        while True:
            status = json.loads((tmp_path / "store" / "service-status.json").read_text())
            if status["grants_suspended"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert status["state"] == "running" and status["storage_incident"]["operation"] == "transaction TRANSPORT_INVOKED"
        assert wire.requests == []
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures, failures
    status = json.loads((tmp_path / "store" / "service-status.json").read_text())
    assert status["state"] == "ended" and not status["grants_suspended"]
    assert close(tmp_path, auth, clock)["verification"]["progress"]["storage_incidents"] == 1


def test_owner_initialization_failure_releases_flock(tmp_path, monkeypatch):
    clock = syn.SimClock()
    store = EdgarStore(tmp_path / "store", wall_clock=clock.wall)
    try:
        with monkeypatch.context() as m:
            m.setattr(EdgarCollector, "reconcile", lambda self: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
            with pytest.raises(RuntimeError, match="synthetic failure"):
                EdgarCollector(store, syn.FakeFetcher(clock), clock)
        collector = EdgarCollector(store, syn.FakeFetcher(clock), clock)
        try:
            with pytest.raises(Rejected, match="another collector"):
                EdgarCollector(store, syn.FakeFetcher(clock), clock)
        finally:
            collector.close()
    finally:
        store.close()


def test_stop_during_spacing_wait_keeps_only_the_finished_attempt(tmp_path):
    stop = threading.Event()

    class Clock(syn.SimClock):
        def sleep(self, seconds):
            super().sleep(seconds)
            if wire.requests:
                stop.set()

    clock = Clock()
    wire = Wire(clock)
    auth = authorization(tmp_path)
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), stop=stop, log=lambda m: None)
    assert summary["reason"] == "stopped" and summary["requests"] == len(wire.requests) == 1
    result = close(tmp_path, auth, clock)
    assert result["integrity_ok"] and not result["run_ok"]


def test_a_slow_commit_consumes_the_physical_deadline_without_sending(tmp_path, monkeypatch):
    clock = syn.SimClock()
    wire = Wire(clock)
    original = EdgarStore.append

    def slow_commit(store, kind, *args, **kwargs):
        seq = original(store, kind, *args, **kwargs)
        if kind == "TRANSPORT_INVOKED":
            clock.sleep(spec.DEADLINE_S + 1)
        return seq

    monkeypatch.setattr(EdgarStore, "append", slow_commit)
    auth = authorization(tmp_path, max_requests=1)
    summary = service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert summary["requests"] == 1 and summary["records"] == 0 and not wire.requests
    store = EdgarStore(tmp_path / "store", wall_clock=None, read_only=True)
    try:
        outcome = store.rows("ATTEMPT_OUTCOME")[0].body
        assert outcome["outcome"] == "SOURCE_UNAVAILABLE" and "deadline" in outcome["reason"]
        assert outcome["fetch_seconds"] >= spec.DEADLINE_S
    finally:
        store.close()


def test_canonical_authorization_identity_survives_filename_and_json_order_changes(tmp_path):
    clock = syn.SimClock()
    wire = Wire(clock)
    auth = authorization(tmp_path, max_requests=1)
    service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    moved = tmp_path / "same-grant.json"
    payload = json.loads(auth.read_text())
    payload["ciks"] = [cik.zfill(10) for cik in payload["ciks"]]
    moved.write_text(json.dumps(payload, sort_keys=True, indent=2))
    before = closure._tree_digest(tmp_path / "store")
    with pytest.raises(service.CaptureRefused, match="budget spent"):
        service.check(tmp_path / "store", moved, now=clock.wall())
    assert len(wire.requests) == 1 and closure._tree_digest(tmp_path / "store") == before


def test_a_new_grant_keeps_the_previous_throttle_pause_for_all_ciks(tmp_path):
    clock = syn.SimClock()
    wire = Wire(clock, status=429)
    auth = authorization(tmp_path, max_requests=2)
    service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    # Another grant permits another trial; it cannot bypass the frozen source pause.
    new_grant = authorization(tmp_path, max_requests=1, granted_by="another synthetic operator")
    wire.status = 200
    service.run(tmp_path / "store", new_grant, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    assert len(wire.requests) == 2
    assert wire.requests[1][1] - wire.requests[0][1] >= spec.THROTTLE_PAUSE_S


def test_closure_requires_the_same_grant_identity_even_with_identical_public_bounds(tmp_path):
    clock = syn.SimClock()
    wire = Wire(clock)
    auth = authorization(tmp_path, max_requests=1)
    service.run(tmp_path / "store", auth, clock=clock, fetcher=wire.fetcher(), log=lambda m: None)
    payload = json.loads(auth.read_text())
    payload["granted_by"] = "another synthetic operator"
    auth.write_text(json.dumps(payload))
    result = close(tmp_path, auth, clock)
    assert result["integrity_ok"] and not result["run_ok"]
    assert "supplied authorization identity differs from the durable grant" in result["run"]["reasons"]


def test_the_restart_throttle_pause_holds_even_when_the_cadence_is_shorter(tmp_path, monkeypatch):
    # With the frozen constants the per-CIK cadence (600 s + 30 s) already covers the 600 s pause, which
    # hides the throttle cooldown. Shorten the cadence to prove the pause is enforced on its own.
    monkeypatch.setattr(spec, "POLL_INTERVAL_S", 60)
    test_a_new_grant_keeps_the_previous_throttle_pause_for_all_ciks(tmp_path)
