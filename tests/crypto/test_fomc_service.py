"""The autonomous owner service under spec revision 23: central grant dispatch with concurrent logical
fetches (work_conserving), task closure at the 60/120/600 s bounds without a blocked owner, registries
safe against late workers, storage incidents (LOCAL_SAVE_ADMISSION_BOUND_V1, STORAGE_INCIDENT_V1,
FOMC248-FOMC250), clean stop and restart."""

from __future__ import annotations

import threading
import time

import pytest

from scripts.trading_lab.fomc import identity, ledger, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.clock import parse_iso
from scripts.trading_lab.fomc.collector import CLASS_ORDER, Collector
from scripts.trading_lab.fomc.service import CAPTURE_BLOCKERS, FomcService, main, preflight
from scripts.trading_lab.fomc.store import FomcStore, StoreBusy
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


def _service(env, tick_s=5.0, **kw):
    kw.setdefault("on_alert", lambda alert: None)
    return FomcService(env.collector, env.clock, tick_s=tick_s, settle_s=5.0, owner_wait_s=0.02, **kw)


def _attempts(env, sid=SID1, kind=None):
    return [t for t in env.store.rows("TRANSPORT_INVOKED")
            if t.body.get("sid") == sid and (kind is None or t.body["kind"] == kind)]


def _park_first_primary_save(env, service, release, marker=b"article__time"):
    """The first save of a statement page blocks before it reaches the store: a hung task, not a
    storage operation."""
    put_raw, seen = env.store.put_raw, {"n": 0}

    def slow(data):
        if marker in data and seen["n"] == 0:
            seen["n"] += 1
            service.park(release)
        return put_raw(data)
    env.store.put_raw = slow


def _hold_network(env, service, path, release, response=None):
    """The provider holds its first answer for `path` until released: a slow logical fetch."""
    sid, served = identity.source_item_id(syn.url(path)), {"n": 0}
    response = response or syn.page_response()

    def route(_count):
        served["n"] += 1
        if served["n"] == 1:
            workers = [service.tasks.get(t.seq) for t in env.store.rows("TRANSPORT_INVOKED") if t.body.get("sid") == sid]
            for worker in filter(None, workers):
                service._parked[worker.ident] = release  # its worker is waiting on the network
            release.wait(30)
            for worker in filter(None, workers):
                service._parked.pop(worker.ident, None)  # working again
        return response
    env.provider.routes[path] = route


def _run_until(service, predicate, limit_s):
    end = service.clock.mono() + limit_s
    while not predicate():
        assert service.clock.mono() < end, "condition not reached"
        service.run_for(service.tick_s)


def _log_decisions(env):
    """For every selection decision: the classes with eligible work, the class of the latest durable
    TRANSPORT_INVOKED at that instant, and the chosen class."""
    decisions, seen = [], {}
    eligible, select = env.collector.eligible_work, env.collector.select

    def logged_eligible():
        work = eligible()
        seen["classes"] = {w["class"] for w in work}
        return work

    def logged_select():
        invoked = env.store.view().rows("TRANSPORT_INVOKED")
        chosen = select()
        if chosen is not None:
            decisions.append((frozenset(seen["classes"]), invoked[-1].body["class"] if invoked else None, chosen["class"]))
        return chosen
    env.collector.eligible_work, env.collector.select = logged_eligible, logged_select
    return decisions


def _one_in_flight(env, key):
    """No two attempts of one item (or two feed polls) were ever in flight together."""
    rows = [t for t in env.store.rows("TRANSPORT_INVOKED") if key(t)]
    for earlier, later in zip(rows, rows[1:]):
        outcome = ledger.outcome_of(env.store, earlier.seq)
        assert outcome is not None and outcome.seq < later.seq
    return len(rows)


def _limiter_respected(env):
    starts = sorted(env.collector.limiter.starts)
    assert all(b - a >= spec.SPACING_S for a, b in zip(starts, starts[1:]))
    assert all(sum(1 for s in starts if 0 <= t - s < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for t in starts)
    return len(starts)


# ------------------------------------------------------------------ autonomy and bounds ---------------
def test_the_service_captures_autonomously(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    service.run_for(900)
    assert service.errors == [] and service.ticks == 180 and service.alerts == []
    assert state.anchor(env.store, SID1) is not None
    assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    assert {o["offset"]: o["satisfied"] for o in state.obligations(env.store, SID1)}[300]
    assert all(state.processing_outcome(env.store, r.seq) for r in env.store.rows("RESPONSE"))
    assert env.provider.requests.count(P1) == 2 and not service.workers and not service.tasks


def test_a_hung_save_is_closed_at_120_s_while_the_owner_keeps_polling(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    _park_first_primary_save(env, service, release)
    _run_until(service, lambda: service._parked, 300)
    first = _attempts(env)[0].seq
    deadline = env.collector._save_deadlines[first]
    polls = env.provider.requests.count(syn.FEED_PATH)
    service.run_for(deadline - env.clock.mono() - service.tick_s)
    assert ledger.outcome_of(env.store, first) is None  # never closed before +120 s
    assert env.provider.requests.count(syn.FEED_PATH) > polls  # the feed is polled around the hung task
    assert len(_attempts(env)) == 1  # but the item has one fetch in flight at most
    service.run_for(2 * service.tick_s)
    outcome = ledger.outcome_of(env.store, first).body
    assert outcome["outcome"] == "LOCAL_PERSISTENCE_FAILED" and service._parked  # closed while still blocked
    service.run_for(600)
    assert state.anchor(env.store, SID1) is not None and len(_attempts(env)) == 2 and service._parked
    service.unpark(release)
    service.run_for(30)
    assert [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == first] == []  # no RESPONSE, no LATE_EVIDENCE
    assert service.errors == [] and service.alerts == []  # a hung task is not a storage incident


def test_a_task_stuck_after_transport_invoked_is_interrupted_at_600_s(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    fetch, seen = env.collector.transport.fetch, {"n": 0}

    def stuck(url, surface, *, invoke, may_continue, started=None, continuation=None):
        if url == syn.url(P1) and seen["n"] == 0:
            seen["n"] += 1
            attempt = invoke(None)  # the dispatcher already committed TRANSPORT_INVOKED; the task then hangs
            service.park(release)
            return FetchResult(kind="SOURCE_UNAVAILABLE", reason="late result", attempt_seq=attempt)
        return fetch(url, surface, invoke=invoke, may_continue=may_continue, started=started, continuation=continuation)
    env.collector.transport.fetch = stuck
    _run_until(service, lambda: service._parked, 300)
    first = _attempts(env)[0]
    started = env.collector._active_attempts[first.seq]
    service.run_for(started + spec.ATTEMPT_ABSOLUTE_DEADLINE_S - env.clock.mono() - service.tick_s)
    assert ledger.outcome_of(env.store, first.seq) is None and len(_attempts(env)) == 1
    service.run_for(2 * service.tick_s)
    outcome = ledger.outcome_of(env.store, first.seq).body
    assert (outcome["outcome"], outcome["reason"]) == ("INTERRUPTED", "no outcome 600 s after TRANSPORT_INVOKED")
    service.run_for(600)
    assert state.anchor(env.store, SID1) is not None and len(_attempts(env)) == 2  # the budget counts #1
    service.unpark(release)
    service.run_for(30)
    assert ledger.outcome_of(env.store, first.seq).body == outcome and service.errors == []  # late result fenced


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
    service.unpark(release)
    service.run_for(30)
    assert len(env.store.rows("PROCESSING_OUTCOME", key=str(record))) == 1  # late results are fenced
    assert service.errors == []


def test_an_old_runs_worker_never_erases_or_kills_the_newer_run(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    releases = [threading.Event(), threading.Event()]
    calls = {"n": 0}

    def fault(resp):
        if resp.body.get("sid") == SID1:
            calls["n"] += 1
            service.park(releases[calls["n"] - 1])
    env.collector.processing_fault = fault
    try:
        _run_until(service, lambda: service._parked, 300)
        record = state.primary_responses(env.store, SID1)[0].seq
        first = env.collector._active_runs[record]
        service.run_for(spec.RUN_DEADLINE_S + 2 * service.tick_s)  # run 1 is DEAD, run 2 started and parked
        second = env.collector._active_runs[record]
        assert second != first and len(service._parked) == 2
        service.unpark(releases[0])  # the old worker returns late: fenced
        service.run_for(60)
        assert env.collector._active_runs.get(record) == second  # the newer run's task is still registered
        dead = env.store.rows("RUN_DEAD", key=str(record))
        assert [d.body["run_id"] for d in dead] == [first]  # only the old run is DEAD, once
        assert state.processing_outcome(env.store, record) is None
        service.unpark(releases[1])
        service.run_for(30)
        assert state.processing_outcome(env.store, record).body["outcome"] == "NORMALIZED_REVISION_COMMITTED"  # by run 2
        assert record not in env.collector._active_runs and service.errors == []
    finally:
        for event in releases:
            event.set()


def test_a_network_stall_is_cut_by_the_physical_deadline_while_other_work_goes_on(env):
    env.collector.transport._deadline_s = 0.5  # the 60 s bound, scaled to real time for the test
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(body=syn.statement_html(), headers=list(syn.HTML_HEADERS),
                                                    stall_s=3.0)  # the server stalls before its status line
    service = FomcService(env.collector, env.clock, tick_s=1.0, settle_s=0.0, owner_wait_s=0.2,
                          on_alert=lambda alert: None)  # production mode: the owner never waits on a worker
    ticks_in_stall = polls_in_stall = 0
    for _ in range(3000):
        service.tick()
        env.clock.sleep(service.tick_s)
        time.sleep(0.002)
        attempts = _attempts(env)
        if attempts and ledger.outcome_of(env.store, attempts[0].seq) is None:
            ticks_in_stall += 1
            polls_in_stall = max(polls_in_stall, sum(1 for t in env.store.rows("TRANSPORT_INVOKED")
                                                     if t.body["kind"] == "FEED_POLL" and t.seq > attempts[0].seq))
        if attempts and ledger.outcome_of(env.store, attempts[0].seq) is not None:
            break
    outcome = ledger.outcome_of(env.store, _attempts(env)[0].seq).body
    assert outcome["outcome"] == "SOURCE_UNAVAILABLE" and "deadline" in outcome["reason"]
    assert ticks_in_stall >= 10 and polls_in_stall >= 1 and service.errors == []  # ticks and polls during the stall


# ------------------------------------------------------------------ concurrent fetches ----------------
PATHS = sorted((syn.statement_path(f"202602{d:02d}") for d in range(1, 6)),
               key=lambda p: identity.source_item_id(syn.url(p)))
SLOW, REDIRECT, FAILING, NORMAL_1, NORMAL_2 = PATHS  # the slow item has the smallest sid: it is acquired first
SIDS = {p: identity.source_item_id(syn.url(p)) for p in PATHS}


def test_a_slow_fetch_lets_other_items_and_polls_start_when_fix15_allows(env, release):
    env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate((SLOW, NORMAL_1, NORMAL_2))])
    for p in (NORMAL_1, NORMAL_2):
        env.provider.routes[p] = syn.page_response()
    service = _service(env)
    _hold_network(env, service, SLOW, release)
    _run_until(service, lambda: _attempts(env, SIDS[SLOW]), 300)
    slow = _attempts(env, SIDS[SLOW])[0].seq
    service.run_for(300)
    assert ledger.outcome_of(env.store, slow) is None and slow in service.tasks  # still in flight
    during = [t for t in env.store.rows("TRANSPORT_INVOKED") if t.seq > slow]
    assert {t.body.get("sid") for t in during} >= {SIDS[NORMAL_1], SIDS[NORMAL_2], None}  # other items and polls started
    assert sum(1 for t in during if t.body["kind"] == "FEED_POLL") >= 4
    assert state.anchor(env.store, SIDS[NORMAL_1]) is not None and state.anchor(env.store, SIDS[NORMAL_2]) is not None
    assert len(_attempts(env, SIDS[SLOW])) == 1  # one fetch of the slow item in flight, never a second
    starts = sorted(env.collector.limiter.starts)
    assert min(b - a for a, b in zip(starts, starts[1:])) == spec.SPACING_S  # started as soon as FIX15 allowed
    _limiter_respected(env)
    service.unpark(release)
    service.run_for(60)
    assert state.anchor(env.store, SIDS[SLOW]) is not None and service.errors == []


def test_continuations_rotation_limiter_budgets_and_late_results_under_concurrency(env, release):
    env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate(PATHS)])
    for p in (NORMAL_1, NORMAL_2):
        env.provider.routes[p] = syn.page_response()
    env.provider.routes[REDIRECT] = syn.SyntheticResponse(status=302, headers=[("Location", REDIRECT + "x")])
    env.provider.routes[REDIRECT + "x"] = syn.page_response()
    service = _service(env)  # FAILING has no route: 404 on every attempt
    _hold_network(env, service, SLOW, release)
    decisions = _log_decisions(env)
    service.run_for(900)  # the slow item's first attempt passes its 600 s bound while held
    slow = _attempts(env, SIDS[SLOW])
    assert ledger.outcome_of(env.store, slow[0].seq).body["outcome"] == "INTERRUPTED"
    service.unpark(release)  # its bytes finally arrive
    service.run_for(900)
    assert service.errors == [] and service.alerts == []
    # one fetch per item and one feed poll at a time
    for p in PATHS:
        _one_in_flight(env, lambda t, sid=SIDS[p]: t.body.get("sid") == sid)
    assert _one_in_flight(env, lambda t: t.body["kind"] == "FEED_POLL") >= 25
    # the redirect continuation takes the next grant, before any new attempt, without a TRANSPORT_INVOKED
    requests = env.provider.requests
    assert requests[requests.index(REDIRECT) + 1] == REDIRECT + "x"
    acquired = _attempts(env, SIDS[REDIRECT], "LIVE_ACQUISITION")
    assert len(acquired) == 1  # two physical requests, one TRANSPORT_INVOKED
    anchor = state.anchor(env.store, SIDS[REDIRECT])
    assert anchor.body["attempt"] == acquired[0].seq
    assert anchor.body["redirect_chain"] == [syn.url(REDIRECT), syn.url(REDIRECT + "x")]
    # every decision follows the class rotation from the durable TRANSPORT_INVOKED records
    assert len(decisions) > 30
    for classes, last, chosen in decisions:
        start = (CLASS_ORDER.index(last) + 1) if last in CLASS_ORDER else 0
        expected = next(CLASS_ORDER[(start + i) % 3] for i in range(3) if CLASS_ORDER[(start + i) % 3] in classes)
        assert chosen == expected
    # FIX15 over every physical start, continuations included
    assert _limiter_respected(env) == len(requests)
    # budgets: at most six attempts per key; the failing item keeps its budget and blocks nobody
    keys = {t.key for t in env.store.rows("TRANSPORT_INVOKED") if t.key != ledger.FEED_KEY}
    assert all(len(ledger.attempts_of_episode(env.store, k)) <= spec.ATTEMPTS_PER_EPISODE for k in keys)
    assert 2 <= len(_attempts(env, SIDS[FAILING])) <= spec.ATTEMPTS_PER_EPISODE and state.anchor(env.store, SIDS[FAILING]) is None
    assert all(state.anchor(env.store, SIDS[p]) is not None for p in (NORMAL_1, NORMAL_2))
    # the late result is LATE_EVIDENCE: it never anchors; the item anchored through its second attempt
    late = [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == slow[0].seq]
    assert len(late) == 1 and late[0].body["late_evidence"] is True
    acquisition = _attempts(env, SIDS[SLOW], "LIVE_ACQUISITION")
    anchor = state.anchor(env.store, SIDS[SLOW])
    assert len(acquisition) == 2 and anchor is not None and anchor.body["attempt"] == acquisition[1].seq


# ------------------------------------------------------------------ storage incidents ----------------
def _capture_primary_attempt(env):
    """Record the attempt the dispatcher starts for the statement item."""
    begin, target = env.collector.begin, {}

    def logged(chosen):
        started = begin(chosen)
        if started is not None and started["work"].get("sid") == SID1:
            target.setdefault("attempt", started["attempt"])
        return started
    env.collector.begin = logged
    return target


def test_an_admitted_commit_stalled_past_120_s_is_valid_and_a_storage_incident_fomc248(env, release):
    """The physical delay of the old blocker, under the revision 23 contract: the admission check
    passed, the COMMIT returns 130 s later. The record is valid, its times are untouched, it is only
    available after it was durable; the stall is an incident with no grant and no decision."""
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    target = _capture_primary_attempt(env)

    def fault(operation):  # the RESPONSE transaction of the statement: checked, inserted, then COMMIT stalls
        if operation == "commit RESPONSE" and threading.current_thread().name == f"fomc-fetch-{target.get('attempt')}":
            env.store.fault = None
            service.park(release)
    env.store.fault = fault
    _run_until(service, lambda: service._parked, 300)
    attempt = target["attempt"]
    stalled_at, wall_at_stall = env.clock.mono(), env.clock.wall()
    starts = len(env.collector.limiter.starts)
    service.run_for(spec.STORAGE_STALL_THRESHOLD_S - service.tick_s)
    assert service.incident is None and service.alerts == []  # not before 10 s
    service.run_for(130 - (env.clock.mono() - stalled_at))
    assert service.incident is not None and service.incident["operation"] == "transaction RESPONSE"
    assert [a["alert"] for a in service.alerts] == ["STORAGE_INCIDENT_STARTED"] and service.alerts[0]["grants_suspended"]
    assert env.collector.limiter.suspended and len(env.collector.limiter.starts) == starts  # no grant: the poll waits
    assert service.busy_ticks >= 20  # the owner kept ticking; it decided nothing durable
    with pytest.raises(StoreBusy):
        env.store.rows("ATTEMPT_OUTCOME")  # this thread is bounded too: the store cannot be read or written
    service.unpark(release)  # fsync returns 130 s after the admission check
    wall_at_release = env.clock.wall()
    service.run_for(300)
    assert [a["alert"] for a in service.alerts] == ["STORAGE_INCIDENT_STARTED", "STORAGE_INCIDENT_ENDED"]
    incident = env.store.rows("STORAGE_INCIDENT")
    assert len(incident) == 1 and incident[0].body["operation"] == "transaction RESPONSE"
    assert 130 <= incident[0].body["stalled_s"] <= 130 + 2 * service.tick_s  # ended at the first tick after it
    outcome = ledger.outcome_of(env.store, attempt).body
    assert outcome["outcome"] == "RESPONSE"  # admitted before its bound: valid whatever its durability instant
    record = state.anchor(env.store, SID1)
    assert record is not None and record.body["attempt"] == attempt and record.body["late_evidence"] is False
    assert parse_iso(record.body["observed_at"]) <= wall_at_stall  # observed_at stays the receipt reading
    assert record.body["observed_at"] == record.body["wall_at_receipt"]
    avail = state.avail_of(state.availability(env.store, env.store.horizon()), record.seq)
    assert avail is not None and avail > wall_at_release  # available only after it was durable, never earlier
    assert env.provider.requests.count(P1) == 1 and len(_attempts(env)) == 1  # no repair request, counted once
    assert len(env.collector.limiter.starts) > starts and not env.collector.limiter.suspended  # grants resumed
    assert service.errors == []


def test_a_stalled_store_stops_grants_and_decisions_then_reconciles_by_first_outcome_fomc250(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    hung_save = threading.Event()
    try:
        _park_first_primary_save(env, service, hung_save)  # the item's save hangs before its admission check
        _run_until(service, lambda: service._parked, 300)
        item_attempt = _attempts(env)[0].seq
        bound = env.collector._save_deadlines[item_attempt]

        def fault(operation):  # then the next feed record's COMMIT stalls: the store itself is blocked
            if operation == "commit RESPONSE":
                env.store.fault = None
                service.park(release)
        env.store.fault = fault
        _run_until(service, lambda: len(service._parked) == 2, 300)
        assert env.clock.mono() < bound  # the store stalls before the item's admission bound
        starts, keys = len(env.collector.limiter.starts), len(env.store._mirror.by_kind["EPISODE_OPEN"][0])
        service.run_for(300)  # the bound passes during the incident
        assert env.clock.mono() > bound and service.incident is not None
        assert len(env.collector.limiter.starts) == starts and service.busy_ticks >= 50  # no grant, no decision
        service.unpark(release)
        service.run_for(3 * service.tick_s)
        outcomes = {o.body["attempt"]: o.body["outcome"] for o in env.store.rows("ATTEMPT_OUTCOME")}
        feed_attempt = max(t.seq for t in env.store.rows("TRANSPORT_INVOKED") if t.body["kind"] == "FEED_POLL"
                           and t.seq in outcomes)
        assert outcomes[feed_attempt] == "RESPONSE"  # the stalled record was admitted: it stands
        assert outcomes[item_attempt] == "LOCAL_PERSISTENCE_FAILED"  # decided at the first commit possible
        assert len(env.store.rows("STORAGE_INCIDENT")) == 1
        assert env.provider.requests.count(P1) == 1 and len(_attempts(env)) == 1  # no repair request
        service.unpark(hung_save)  # the item's late bytes are never admitted
        service.run_for(600)
        assert [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == item_attempt] == []
        attempts = _attempts(env)
        assert len(attempts) == 2 and attempts[1].body["episode_key"] == attempts[0].body["episode_key"]  # same budget
        assert state.anchor(env.store, SID1) is not None
        opened = env.store.rows("EPISODE_OPEN")
        assert len(opened) == len({e.key for e in opened}) and len(opened) >= keys  # no key recreated
        assert len(env.collector.limiter.starts) > starts and service.errors == []  # grants resumed in order
        _limiter_respected(env)
    finally:
        service.unpark(hung_save)


def test_a_stalled_raw_write_is_an_incident_but_the_owner_still_decides_at_120_s(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    target = _capture_primary_attempt(env)

    def fault(operation):  # the raw-body write of the statement stalls inside the store (its fsync)
        if operation == "raw_write" and threading.current_thread().name == f"fomc-fetch-{target.get('attempt')}":
            env.store.fault = None
            service.park(release)
    env.store.fault = fault
    _run_until(service, lambda: service._parked, 300)
    attempt = target["attempt"]
    bound = env.collector._save_deadlines[attempt]
    service.run_for(bound - env.clock.mono() + 2 * service.tick_s)
    assert service.incident is not None and service.incident["operation"] == "raw_write"
    starts = len(env.collector.limiter.starts)
    # SQLite itself can write, so the decision is durable at the bound, during the incident
    assert ledger.outcome_of(env.store, attempt).body["outcome"] == "LOCAL_PERSISTENCE_FAILED"
    service.run_for(120)
    assert len(env.collector.limiter.starts) == starts  # no grant while storage is stalled
    service.unpark(release)
    service.run_for(600)
    assert [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == attempt] == []  # checked after the bound
    assert [a["alert"] for a in service.alerts] == ["STORAGE_INCIDENT_STARTED", "STORAGE_INCIDENT_ENDED"]
    assert len(env.store.rows("STORAGE_INCIDENT")) == 1 and state.anchor(env.store, SID1) is not None
    assert len(_attempts(env)) == 2 and service.errors == []


def test_the_monitor_detects_an_incident_when_the_owners_own_commit_stalls(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    service.run_for(120)

    def fault(operation):  # the owner's next TRANSPORT_INVOKED commit stalls
        if operation == "commit TRANSPORT_INVOKED":
            env.store.fault = None
            service.park(release)
    env.store.fault = fault
    owner = threading.Thread(target=lambda: service.run_for(600), daemon=True)
    owner.start()
    for _ in range(500):
        if service._parked:
            break
        time.sleep(0.01)
    assert service._parked  # the owner thread itself is inside the stalled COMMIT
    ticks = service.ticks
    for _ in range(3):  # what the monitor thread does, beside the blocked owner
        env.clock.sleep(5)
        service.watch_storage()
    assert service.ticks == ticks and service.incident is not None  # detected without the owner, without the store
    assert service.incident["operation"] == "transaction TRANSPORT_INVOKED" and env.collector.limiter.suspended
    assert [a["alert"] for a in service.alerts] == ["STORAGE_INCIDENT_STARTED"]
    service.unpark(release)
    owner.join(30)
    assert not owner.is_alive() and service.errors == []
    assert [a["alert"] for a in service.alerts] == ["STORAGE_INCIDENT_STARTED", "STORAGE_INCIDENT_ENDED"]
    assert len(env.store.rows("STORAGE_INCIDENT")) == 1 and not env.collector.limiter.suspended


# ------------------------------------------------------------------ stop and restart -----------------
def test_a_clean_stop_leaves_nothing_in_flight_and_the_next_owner_interrupts_nothing(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    service = _service(env)
    service.run_for(200)
    assert service.stop(wait_s=10) == []  # every attempt has its outcome
    assert all(state.processing_outcome(env.store, r.seq) for r in env.store.rows("RESPONSE"))
    invoked = len(env.store.rows("TRANSPORT_INVOKED"))
    service.tick()
    assert len(env.store.rows("TRANSPORT_INVOKED")) == invoked  # a stopped service grants nothing
    env.store = FomcStore(env.root, wall_clock=env.clock.wall)
    env.collector = Collector(env.store, env.provider.connector(), env.clock, boot_id="boot-2", inline=False)
    new = _service(env)
    assert [o for o in env.store.rows("ATTEMPT_OUTCOME") if o.body["outcome"] == "INTERRUPTED"] == []
    new.run_for(600)
    assert {o["offset"]: o["satisfied"] for o in state.obligations(env.store, SID1)}[300] and new.errors == []


def test_a_restart_interrupts_the_old_owners_attempt_at_once_and_keeps_its_budget(env, release):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    old = _service(env)
    _park_first_primary_save(env, old, release)
    _run_until(old, lambda: old._parked, 300)
    first = _attempts(env)[0].seq
    env.collector.close()  # the owner process dies: its lock is released, its parked task never resumes
    env.store = FomcStore(env.root, wall_clock=env.clock.wall)
    env.collector = Collector(env.store, env.provider.connector(), env.clock, boot_id="boot-2", inline=False)
    new = _service(env)
    outcome = ledger.outcome_of(env.store, first).body
    assert (outcome["outcome"], outcome["reason"]) == ("INTERRUPTED", "non-current epoch")  # at once
    new.run_for(600)
    attempts = _attempts(env)
    assert len(attempts) == 2 and attempts[1].body["episode_key"] == attempts[0].body["episode_key"]  # budget kept
    assert state.anchor(env.store, SID1) is not None and new.errors == []
    assert len(env.store.rows("EPISODE_OPEN")) == len({e.key for e in env.store.rows("EPISODE_OPEN")})  # no key recreated


# ------------------------------------------------------------------ capture entry point --------------
def test_the_capture_preflight_passes_without_network_store_or_ownership(tmp_path, capsys):
    assert CAPTURE_BLOCKERS == ()  # lifted by spec revision 23 and the tests above
    assert preflight(tmp_path / "store") == []
    assert main(["--store", str(tmp_path / "store"), "--check"]) == 0
    assert "preflight ok" in capsys.readouterr().out and not (tmp_path / "store").exists()  # nothing created


def test_the_capture_preflight_refuses_an_incompatible_store(env, capsys):
    env.collector.close()
    env.store._conn.execute("UPDATE meta SET value = ? WHERE name = 'spec_hash'", (spec.SUPERSEDED_SPEC_HASH,))
    env.store.close()
    assert main(["--store", str(env.root), "--check"]) == 3  # a revision-22 store is never opened by this code
    assert "bound to spec" in capsys.readouterr().err
