"""FIX15_GRANT_JOURNAL_V1 (spec revision 25, F83, FOMC259-FOMC260): every consumed grant, initial or
continuation, including grants abandoned before any TRANSPORT_INVOKED, is journaled durably before it is
used; the closure proves FIX15 over the journal and reports NOT_PROVEN on a missing trace."""

from __future__ import annotations

import shutil
import sqlite3
import time

import pytest

from scripts.trading_lab.fomc import ledger, pilot, spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.service import FomcService
from scripts.trading_lab.fomc.store import FomcStore

from tests.crypto.fomc_support import P1, SID1, Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _service(env):
    return FomcService(env.collector, env.clock, tick_s=5.0, settle_s=5.0, owner_wait_s=0.02, on_alert=lambda a: None)


def _grants(env, attempt=None):
    rows = env.store.rows("GRANT")
    if attempt is None:
        return rows
    return sorted((g for g in rows if (g.seq if g.body["attempt"] == "SELF" else g.body["attempt"]) == attempt),
                  key=lambda g: g.body["hop"])


def _primary_attempts(env):
    return [t for t in env.store.rows("TRANSPORT_INVOKED") if t.body.get("sid") == SID1]


def test_a_three_hop_redirect_chain_is_journaled_hop_by_hop_and_proven(env):
    env.feed([statement_item()])
    chain = [P1, P1 + "a", P1 + "b", P1 + "c"]
    for here, there in zip(chain, chain[1:]):
        env.provider.routes[here] = syn.SyntheticResponse(status=302, headers=[("Location", there)])
    env.provider.routes[chain[-1]] = syn.page_response()
    service = _service(env)
    service.run_for(300)
    service.stop(wait_s=10)
    attempt = _primary_attempts(env)[0].seq
    grants = _grants(env, attempt)
    assert [g.body["hop"] for g in grants] == [0, 1, 2, 3]
    assert [g.body["url"] for g in grants] == [syn.url(p) for p in chain]
    assert [g.body["kind"] for g in grants] == ["INITIAL", "CONTINUATION", "CONTINUATION", "CONTINUATION"]
    orders = [g.body["order"] for g in grants]
    assert orders == list(range(orders[0], orders[0] + 4))  # each continuation took the very next grant
    assert all(g.body["group"] == SID1 for g in grants)
    record = env.store.rows("RESPONSE", key=str(attempt))[0].body
    assert record["grants"] == 4 and record["redirect_chain"] == [syn.url(p) for p in chain]
    fix15 = pilot.audit(env.store)["fix15"]
    assert fix15["status"] == "PROVEN" and fix15["continuations"] == 3 and fix15["missing"] == []


def test_a_failure_after_a_redirect_journals_both_grants(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "gone")])  # then 404
    service = _service(env)
    service.run_for(200)
    service.stop(wait_s=10)
    attempt = _primary_attempts(env)[0].seq
    outcome = ledger.outcome_of(env.store, attempt).body
    assert outcome["outcome"] == "SOURCE_UNAVAILABLE" and outcome["grants"] == 2
    assert [g.body["hop"] for g in _grants(env, attempt)] == [0, 1]
    assert pilot.audit(env.store)["fix15"]["status"] == "PROVEN"


def test_grants_abandoned_after_and_before_transport_invoked_stay_in_the_journal(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    invoke = env.collector._invoke

    def late(work, grant, url):  # TRANSPORT_INVOKED commits more than 1 s after its grant
        seq = invoke(work, grant, url)
        env.clock.sleep(2)
        return seq
    env.collector._invoke = late
    result = env.collector.poll_feed()
    env.collector._invoke = invoke
    assert result["status"] == "CANCELLED_AFTER_INVOKE"
    poll = env.store.rows("TRANSPORT_INVOKED")[-1]
    assert ledger.outcome_of(env.store, poll.seq).body["grants"] == 1 and len(_grants(env, poll.seq)) == 1
    ledger.begin_epoch(env.store, "another-owner", "boot-x")  # fenced: the next TRANSPORT_INVOKED cannot commit
    env.clock.sleep(30)
    rejected = env.collector.poll_feed()
    assert rejected["status"] == "REJECTED"
    abandoned = [g for g in env.store.rows("GRANT") if g.body["attempt"] is None]
    assert len(abandoned) == 1 and abandoned[0].body["note"].startswith("abandoned before TRANSPORT_INVOKED")
    fix15 = pilot.audit(env.store)["fix15"]
    assert fix15["abandoned_before_invoke"] == 1 and fix15["status"] == "PROVEN"


def test_a_lost_trace_is_not_proven_and_a_broken_rule_fails(env, tmp_path):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "x")])
    env.provider.routes[P1 + "x"] = syn.page_response()
    service = _service(env)
    service.run_for(300)
    service.stop(wait_s=10)
    assert pilot.audit(env.store)["fix15"]["status"] == "PROVEN"
    edits = {
        "lost-continuation": "DELETE FROM rec WHERE kind = 'GRANT' AND body LIKE '%CONTINUATION%'",
        "lost-initial": "DELETE FROM rec WHERE kind = 'GRANT' AND commit_seq = (SELECT MIN(commit_seq) FROM rec WHERE kind = 'GRANT')",
        "count-differs": "UPDATE rec SET body = json_set(body, '$.grants', 7) WHERE kind = 'RESPONSE' "
                         "AND commit_seq = (SELECT MAX(commit_seq) FROM rec WHERE kind = 'RESPONSE')",
    }
    for name, sql in edits.items():
        copy = tmp_path / name
        shutil.copytree(env.root, copy)
        with sqlite3.connect(copy / "fomc.sqlite3") as conn:
            assert conn.execute(sql).rowcount >= 1
        store = FomcStore(copy, wall_clock=env.clock.wall)
        try:
            fix15 = pilot.audit(store)["fix15"]
            assert fix15["status"] == "NOT_PROVEN" and fix15["missing"], name
        finally:
            store.close()
    copy = tmp_path / "too-close"
    shutil.copytree(env.root, copy)
    with sqlite3.connect(copy / "fomc.sqlite3") as conn:  # move the second grant 1 s after the first
        first = conn.execute("SELECT json_extract(body, '$.mono') FROM rec WHERE kind = 'GRANT' ORDER BY commit_seq LIMIT 1").fetchone()[0]
        conn.execute("UPDATE rec SET body = json_set(body, '$.mono', ?) WHERE kind = 'GRANT' AND commit_seq = "
                     "(SELECT commit_seq FROM rec WHERE kind = 'GRANT' ORDER BY commit_seq LIMIT 1 OFFSET 1)", (first + 1,))
    store = FomcStore(copy, wall_clock=env.clock.wall)
    try:
        fix15 = pilot.audit(store)["fix15"]
        assert fix15["status"] == "FAIL" and fix15["violations"]
    finally:
        store.close()


def test_a_continuation_whose_journal_row_cannot_commit_is_not_granted(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "x")])
    env.provider.routes[P1 + "x"] = syn.page_response()
    service = _service(env)
    failing = {"on": True}

    def fault(operation):
        if operation == "commit GRANT" and failing["on"]:
            raise sqlite3.OperationalError("disk I/O error")
    env.store.fault = fault
    for _ in range(80):
        service.run_for(service.tick_s)
        if service._slots:
            break
    assert service._slots  # the redirect hop waits for its grant
    starts = len(env.collector.limiter.starts)
    service.run_for(60)
    assert len(env.collector.limiter.starts) == starts  # its journal row never committed: nothing granted, nothing sent
    assert env.provider.requests.count(P1 + "x") == 0 and any("disk I/O error" in str(e) for e in service.errors)
    failing["on"] = False
    service.run_for(120)
    service.stop(wait_s=10)
    assert env.provider.requests.count(P1 + "x") == 1
    assert pilot.audit(env.store)["fix15"]["status"] == "PROVEN"


def test_a_restart_starts_a_new_epoch_of_the_journal(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    first = _service(env)
    first.run_for(300)
    first.stop(wait_s=10)
    env.store = FomcStore(env.root, wall_clock=env.clock.wall)
    env.collector = Collector(env.store, env.provider.connector(), env.clock, boot_id="boot-2", inline=False)
    second = _service(env)
    second.run_for(600)
    second.stop(wait_s=10)
    fix15 = pilot.audit(env.store)["fix15"]
    assert fix15["status"] == "PROVEN" and len(fix15["per_epoch"]) == 2
    assert all(e["embargo_ok"] and e["spacing_ok"] and e["window_ok"] for e in fix15["per_epoch"].values())
    for epoch in fix15["per_epoch"]:
        orders = [g.body["order"] for g in env.store.rows("GRANT") if g.body["epoch"] == epoch]
        assert orders == list(range(1, len(orders) + 1))
    assert spec.EMBARGO_S == 60


def test_a_stop_while_a_continuation_waits_consumes_no_grant_and_stays_proven(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "x")])
    env.provider.routes[P1 + "x"] = syn.page_response()
    service = _service(env)
    env.store.fault = lambda operation: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error")) \
        if operation == "commit GRANT" else None
    for _ in range(80):
        service.run_for(service.tick_s)
        if service._slots:
            break
    assert service._slots
    starts = len(env.collector.limiter.starts)
    service.stop(wait_s=10)  # cancelled while its continuation waits
    env.store.fault = None
    assert len(env.collector.limiter.starts) == starts and env.provider.requests.count(P1 + "x") == 0
    # stop() may return while the worker it released is still committing the attempt's outcome (its
    # contract leaves such attempts to the next owner): wait, in real time and bounded, for that commit.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (
        _primary_attempts(env) and ledger.outcome_of(env.store, _primary_attempts(env)[0].seq) is not None
    ):
        time.sleep(0.02)
    for thread in list(service.workers):
        thread.join(max(0.0, deadline - time.monotonic()))
    attempt = _primary_attempts(env)[0].seq
    outcome = ledger.outcome_of(env.store, attempt)
    assert outcome is not None and outcome.body.get("grants", 1) == 1 and len(_grants(env, attempt)) == 1
    assert pilot.audit(env.store)["fix15"]["status"] == "PROVEN"
