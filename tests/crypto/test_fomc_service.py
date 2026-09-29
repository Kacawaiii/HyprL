"""The autonomous owner service: tick, task closure at the 60/120/600 s bounds without a blocked owner,
restart without recreated budgets; and the COMMIT/fsync capture blocker, reproduced (xfail strict)."""

from __future__ import annotations

import threading
import time

import pytest

from scripts.trading_lab.fomc import ledger, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.service import CAPTURE_BLOCKERS, FomcService, main
from scripts.trading_lab.fomc.store import FomcStore
from scripts.trading_lab.fomc.transport import FetchResult

from tests.crypto.fomc_support import P1, SID1, Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


@pytest.fixture
def release():
    event = threading.Event()
    yield event
    event.set()  # never leave a parked task behind


def _service(env, tick_s=5.0):
    return FomcService(env.collector, env.clock, tick_s=tick_s, settle_s=5.0)


def _primary_attempts(env):
    return [t for t in env.store.rows("TRANSPORT_INVOKED") if t.body.get("sid") == SID1]


def _park_first_primary_save(env, service, release):
    """The first save of the statement page blocks (not holding the store lock) until released."""
    put_raw, seen = env.store.put_raw, {"n": 0}

    def slow(data):
        if b"article__time" in data and seen["n"] == 0:
            seen["n"] += 1
            service.park(release)
        return put_raw(data)
    env.store.put_raw = slow


def _run_until(service, predicate, limit_s):
    end = service.clock.mono() + limit_s
    while not predicate():
        assert service.clock.mono() < end, "condition not reached"
        service.run_for(service.tick_s)


def test_the_service_captures_autonomously(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    service.run_for(900)
    assert service.errors == [] and service.ticks >= 120  # limiter waits inside fetch tasks also advance the clock
    assert state.anchor(env.store, SID1) is not None
    assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    assert {o["offset"]: o["satisfied"] for o in state.obligations(env.store, SID1)}[300]
    assert all(state.processing_outcome(env.store, r.seq) for r in env.store.rows("RESPONSE"))
    assert env.provider.requests.count(P1) == 2 and not service.workers  # acquisition + O300, no task left


def test_a_hung_save_is_closed_at_120_s_while_the_owner_keeps_ticking(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    _park_first_primary_save(env, service, release)
    _run_until(service, lambda: service._parked, 300)
    first = _primary_attempts(env)[0].seq
    deadline = env.collector._save_deadlines[first]
    ticks = service.ticks
    service.run_for(deadline - env.clock.mono() - service.tick_s)
    assert ledger.outcome_of(env.store, first) is None  # never closed before +120 s
    service.run_for(2 * service.tick_s)
    outcome = ledger.outcome_of(env.store, first).body
    assert outcome["outcome"] == "LOCAL_PERSISTENCE_FAILED" and service._parked  # closed while the task is still blocked
    assert service.ticks - ticks >= 24  # the owner kept ticking through the hang
    polls = env.provider.requests.count(syn.FEED_PATH)
    service.run_for(600)  # the fetch slot is free again: polling and attempt #2 go on around the parked task
    assert env.provider.requests.count(syn.FEED_PATH) > polls
    assert state.anchor(env.store, SID1) is not None and len(_primary_attempts(env)) == 2 and service._parked
    release.set()
    service.run_for(30)
    assert [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == first] == []  # no RESPONSE, no LATE_EVIDENCE
    assert env.provider.requests.count(P1) == 2 and service.errors == []


def test_a_task_stuck_after_transport_invoked_is_interrupted_at_600_s(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    fetch, seen = env.collector.transport.fetch, {"n": 0}

    def stuck(url, surface, *, invoke, may_continue):
        if url == syn.url(P1) and seen["n"] == 0:
            seen["n"] += 1
            attempt = invoke(env.clock.mono())  # durable TRANSPORT_INVOKED, then the task hangs
            service.park(release)
            return FetchResult(kind="SOURCE_UNAVAILABLE", reason="late result", attempt_seq=attempt)
        return fetch(url, surface, invoke=invoke, may_continue=may_continue)
    env.collector.transport.fetch = stuck
    _run_until(service, lambda: service._parked, 300)
    first = _primary_attempts(env)[0]
    started = env.collector._active_attempts[first.seq]
    service.run_for(started + spec.ATTEMPT_ABSOLUTE_DEADLINE_S - env.clock.mono() - service.tick_s)
    assert ledger.outcome_of(env.store, first.seq) is None and service.fetch_in_flight()
    service.run_for(2 * service.tick_s)
    outcome = ledger.outcome_of(env.store, first.seq).body
    assert (outcome["outcome"], outcome["reason"]) == ("INTERRUPTED", "no outcome 600 s after TRANSPORT_INVOKED")
    service.run_for(600)
    assert state.anchor(env.store, SID1) is not None and len(_primary_attempts(env)) == 2  # the budget counts #1
    release.set()
    service.run_for(30)
    assert ledger.outcome_of(env.store, first.seq).body == outcome and service.errors == []  # the late result is fenced


def test_a_hung_processing_run_is_replaced_at_600_s_then_poisoned(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    env.collector.processing_fault = lambda resp: service.park(release) if resp.body.get("sid") == SID1 else None
    _run_until(service, lambda: service._parked, 300)
    record = state.primary_responses(env.store, SID1)[0].seq
    polls = env.provider.requests.count(syn.FEED_PATH)
    service.run_for(spec.RUN_DEADLINE_S + 2 * service.tick_s)
    dead = env.store.rows("RUN_DEAD", key=str(record))
    assert [d.body["reason"] for d in dead] == ["run deadline passed"] and len(service._parked) == 2  # replaced
    service.run_for(spec.RUN_DEADLINE_S + 2 * service.tick_s)
    assert state.processing_outcome(env.store, record).body["outcome"] == "INTERNAL_PROCESSING_ERROR"  # poison guard
    assert env.provider.requests.count(syn.FEED_PATH) >= polls + 15  # the feed kept being polled
    assert "EVENTS_OBSERVED_ZERO" not in env.cycles()  # the unprocessed record keeps every cycle NOT_ZERO
    release.set()
    service.run_for(30)
    assert len(env.store.rows("PROCESSING_OUTCOME", key=str(record))) == 1  # late results are fenced
    assert service.errors == []


def test_a_network_stall_is_cut_by_the_physical_deadline_while_the_owner_ticks(env):
    env.collector.transport._deadline_s = 0.5  # the 60 s bound, scaled to real time for the test
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(body=syn.statement_html(), headers=list(syn.HTML_HEADERS),
                                                    stall_s=3.0)  # the server stalls before its status line
    service = FomcService(env.collector, env.clock, tick_s=1.0, settle_s=0.0)  # production mode: never waits
    ticks_in_stall = 0
    for _ in range(3000):
        service.tick()
        env.clock.sleep(service.tick_s)
        time.sleep(0.002)
        attempts = _primary_attempts(env)
        if attempts and ledger.outcome_of(env.store, attempts[0].seq) is None and service.fetch_in_flight():
            ticks_in_stall += 1
        if attempts and ledger.outcome_of(env.store, attempts[0].seq) is not None:
            break
    outcome = ledger.outcome_of(env.store, _primary_attempts(env)[0].seq).body
    assert outcome["outcome"] == "SOURCE_UNAVAILABLE" and "deadline" in outcome["reason"]
    assert ticks_in_stall >= 10 and service.errors == []  # the owner ticked while its fetch was stalled


def test_a_restart_interrupts_the_old_owners_attempt_at_once_and_keeps_its_budget(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    old = _service(env)
    _park_first_primary_save(env, old, release)
    _run_until(old, lambda: old._parked, 300)
    first = _primary_attempts(env)[0].seq
    env.collector.close()  # the owner process dies: its lock is released, its parked task never resumes
    env.store = FomcStore(env.root, wall_clock=env.clock.wall)
    env.collector = Collector(env.store, env.provider.connector(), env.clock, boot_id="boot-2")
    new = _service(env)
    outcome = ledger.outcome_of(env.store, first).body
    assert (outcome["outcome"], outcome["reason"]) == ("INTERRUPTED", "non-current epoch")  # at once
    new.run_for(600)
    attempts = _primary_attempts(env)
    assert len(attempts) == 2 and attempts[1].body["episode_key"] == attempts[0].body["episode_key"]  # budget kept
    assert state.anchor(env.store, SID1) is not None and new.errors == []
    assert len(env.store.rows("EPISODE_OPEN")) == len({e.key for e in env.store.rows("EPISODE_OPEN")})  # no key recreated


def test_real_capture_is_refused_while_a_blocker_is_open(tmp_path, capsys):
    assert CAPTURE_BLOCKERS
    assert main(["--store", str(tmp_path / "store")]) == 3
    assert "COMMIT_FSYNC_120S" in capsys.readouterr().err and not (tmp_path / "store").exists()


# ------------------------------------------------------------------ the COMMIT/fsync capture blocker ---
class _StallingConnection:
    """Wraps the store's SQLite connection: the next COMMIT runs `stall` first (an fsync that blocks)."""

    def __init__(self, conn):
        self._conn, self.stall = conn, None

    def execute(self, sql, *args):
        if sql == "COMMIT" and self.stall is not None:
            stall, self.stall = self.stall, None
            stall()
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _stall_the_response_commit(env, stall):
    conn = _StallingConnection(env.store._conn)
    env.store._conn = conn
    put_raw = env.store.put_raw

    def arm(data):  # the RESPONSE transaction follows the raw write of the statement page
        digest = put_raw(data)
        if b"article__time" in data:
            conn.stall = stall
        return digest
    env.store.put_raw = arm


@pytest.mark.xfail(strict=True, reason="CAPTURE BLOCKER COMMIT_FSYNC_120S: the admission check precedes SQLite's "
                                       "COMMIT; a COMMIT stalled in fsync past +120 s still creates the RESPONSE")
def test_capture_blocker_a_commit_stalled_past_120_s_never_creates_a_response(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    _stall_the_response_commit(env, lambda: env.clock.sleep(130))  # fsync returns 130 s after the check
    env.drive(200)
    attempt = _primary_attempts(env)[0].seq
    assert ledger.outcome_of(env.store, attempt).body["outcome"] != "RESPONSE"  # required; fails today


@pytest.mark.xfail(strict=True, reason="CAPTURE BLOCKER COMMIT_FSYNC_120S: a COMMIT stalled in fsync holds the store "
                                       "lock, so the owner cannot tick or write LOCAL_PERSISTENCE_FAILED")
def test_capture_blocker_the_owner_keeps_ticking_during_a_stalled_commit(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    _stall_the_response_commit(env, lambda: service.park(release))  # the COMMIT blocks on a real event
    _run_until(service, lambda: service._parked, 300)
    ticked = threading.Event()
    owner = threading.Thread(target=lambda: (service.tick(), ticked.set()), daemon=True)
    owner.start()
    try:
        assert ticked.wait(3.0)  # required: the owner is never blocked; fails today (store lock held)
    finally:
        release.set()
        owner.join(10)
