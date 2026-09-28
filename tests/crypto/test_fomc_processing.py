"""Checkpoint 2 - processing: the local provider drives feed -> raw -> classification -> acquisition
-> revision -> cycle, with clock faults, redirects, bounded episodes, crashes and operator actions."""

from __future__ import annotations

import pytest

from scripts.trading_lab.fomc import identity, ledger, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector

from tests.crypto.fomc_support import EXACT, P1, SID1, Env, statement_item

@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def test_new_statement_reaches_revision_then_zero(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(200)
    assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    assert "NORMALIZED_REVISION_COMMITTED" in env.outcomes()
    assert len(env.store.rows("REVISION")) == 1
    assert state.anchor(env.store, SID1) is not None
    assert env.provider.requests.count(P1) == 1


def test_clock_unverified_records_are_processed_without_causal_effect(tmp_path):
    env = Env(tmp_path, wall_offset_s=300)  # collector clock 5 min fast (FOMC165)
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response()
        env.drive(150)
        responses = env.store.rows("RESPONSE")
        assert responses and all(r.body["verdict"] == "CLOCK_UNVERIFIED" for r in responses)
        assert env.store.rows("CANDIDATE") and env.store.rows("ACQUISITION")  # classified (FOMC222/232)
        assert env.store.rows("CYCLE_CONCLUSION") == []  # no cycle, no zero
        assert "NORMALIZED_REVISION_COMMITTED" in env.outcomes()  # processed to a terminal outcome (FOMC221)
        assert state.anchor(env.store, SID1) is None  # no causal LIVE effect
        feed_polls = env.provider.requests.count(syn.FEED_PATH)
        assert feed_polls >= 2  # polling continues
        assert [i["key"] for i in state.outstanding(env.store, upto=env.store.horizon())] == [f"SOURCE_ITEM:{SID1}"]
        env.clock.wall_offset_s = 0  # clock corrected
        env.drive(1200, idle=60)
        assert state.anchor(env.store, SID1) is not None
        assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    finally:
        env.provider.close()


def test_unverified_malformed_feed_is_terminal_and_polling_continues(tmp_path):
    env = Env(tmp_path, wall_offset_s=300)
    try:
        env.provider.routes[syn.FEED_PATH] = syn.SyntheticResponse(body=b"<html>maintenance</html>", headers=syn.HTML_HEADERS)
        env.drive(130)
        failed = [o for o in env.store.rows("PROCESSING_OUTCOME") if o.body["outcome"] == "PARSER_FAILED"]
        assert len(failed) >= 2 and all(o.body["channel_level"] for o in failed)  # FOMC242 variant
        assert all(r.body["verdict"] == "CLOCK_UNVERIFIED" for r in env.store.rows("RESPONSE"))
        assert any(i["type"] == "FAILED_FEED" for i in state.outstanding(env.store, upto=env.store.horizon()))
        env.clock.wall_offset_s = 0
        env.feed([])
        env.drive(200)
        assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"  # cleared by a parseable feed
    finally:
        env.provider.close()


def _redirect(to):
    return syn.SyntheticResponse(status=302, headers=[("Location", to)])


def test_redirects_are_bounded_to_four_physical_requests(env):
    chain = ["/r1.htm", "/r2.htm", "/r3.htm"]
    env.feed([statement_item()])
    env.provider.routes[P1] = _redirect(chain[0])
    env.provider.routes[chain[0]] = _redirect(chain[1])
    env.provider.routes[chain[1]] = _redirect(chain[2])
    env.provider.routes[chain[2]] = syn.page_response()
    env.drive(80)
    resp = [r for r in env.store.rows("RESPONSE") if r.body["surface"] == "primary"][0]
    assert resp.body["redirect_chain"] == [syn.url(P1)] + [syn.url(p) for p in chain]  # 4 physical, identity from U0
    assert resp.body["sid"] == SID1
    env.provider.routes[chain[2]] = _redirect("/r4.htm")  # fourth redirect: never requested
    env.collector.manual_retry({"kind": "REOBSERVATION", "sid": SID1, "offset": 300})
    env.drive(900, idle=60)
    assert "/r4.htm" not in env.provider.requests
    fails = [o for o in env.store.rows("ATTEMPT_OUTCOME") if o.body.get("reason") == "redirect transition limit exceeded"]
    assert fails


def test_dead_page_uses_six_attempts_then_suspends(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = _redirect("/loop1.htm")
    for i in range(1, 5):
        env.provider.routes[f"/loop{i}.htm"] = _redirect(f"/loop{i + 1}.htm")
    env.drive(25000, idle=600)
    key = identity.acquisition_episode_key(SID1, env.store.rows("ACQUISITION")[0].seq)
    assert len(ledger.attempts_of_episode(env.store, key)) == spec.ATTEMPTS_PER_EPISODE
    assert ledger.episode_status(env.store, key) == "SUSPENDED"
    item_requests = [p for p in env.provider.requests if p != syn.FEED_PATH]
    assert len(item_requests) == spec.ATTEMPTS_PER_EPISODE * spec.MAX_PHYSICAL_PER_ATTEMPT  # 24, never more
    env.drive(20000, idle=600)  # relisting and restarts add nothing
    env.restart()
    env.drive(2000, idle=600)
    assert len([p for p in env.provider.requests if p != syn.FEED_PATH]) == 24
    assert env.cycles()[-1] == "NOT_ZERO"  # the suspended statement stays outstanding


def test_crash_after_transport_invoked_late_response_and_lost_ack(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.poll_feed()
    env.collector.open_episodes()
    episode = env.store.rows("EPISODE_OPEN")[0]
    work = {"kind": "LIVE_ACQUISITION", "class": "FEED_DISCOVERY", "episode_key": episode.key, "sid": SID1, "mode": "LIVE"}
    old = env.collector
    fetched = old.fetch(work, syn.url(P1), "primary")  # bytes received, crash before the response commits
    old.close()
    env.collector = Collector(env.store, env.provider.connector(), env.clock, boot_id="boot-2")  # reconcile
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body["outcome"] == "INTERRUPTED"
    late = old.commit(fetched, work, syn.url(P1), "primary")  # the old task finally commits
    assert late["late"] is True
    record = env.store.rows("RESPONSE")[-1]
    assert record.body["late_evidence"] and not state.live_eligible(record)
    assert env.store.rows("PROCESSING_RUN", key=str(record.seq)) == []  # the non-owner starts no run
    env.collector.process_pending()  # the owner's trigger after any per-response commit
    assert state.processing_outcome(env.store, record.seq) is not None  # processed as unverified evidence
    assert state.anchor(env.store, SID1) is None  # never anchors or satisfies
    env.drive(400)
    attempts = ledger.attempts_of_episode(env.store, episode.key)
    assert len(attempts) == 2  # the interrupted attempt counted; no new budget was created
    before = list(env.provider.requests)
    env.collector.reconcile()  # a lost ACK is resolved from durable state, never by refetching
    assert env.provider.requests == before
    assert state.anchor(env.store, SID1) is not None


def test_poison_guard_ends_processing_without_stalling_polls(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()

    def boom(resp):
        if resp.body["surface"] == "primary":
            raise RuntimeError("parser crash")

    env.collector.processing_fault = boom
    env.drive(300)
    assert "INTERNAL_PROCESSING_ERROR" in env.outcomes()  # FOMC196: two DEAD runs, no crash loop
    assert env.provider.requests.count(syn.FEED_PATH) >= 3
    assert all(c == "NOT_ZERO" for c in env.cycles())


def test_resolve_cutoff_reopens_and_fomc247(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(title="Federal Reserve issues FOMC minutes")  # conflict
    env.drive(200)
    assert "CLASSIFICATION_CONFLICT" in env.outcomes()
    assert all(c == "NOT_ZERO" for c in env.cycles())
    result = env.collector.resolve(f"SOURCE_ITEM:{SID1}")
    assert result["valid"]
    env.drive(120)
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    env.drive(600, idle=60)  # the O300 recheck opens an episode: a later item record re-opens the item
    assert env.store.rows("EPISODE_OPEN")[-1].body["kind"] == "REOBSERVATION"
    assert env.cycles()[-1] == "NOT_ZERO"


def test_fomc247_resolve_after_manual_anchor_and_no_anchor_control(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(title="Federal Reserve issues FOMC minutes")
    env.collector.poll_feed()  # acquisition Q created, no EPISODE_OPEN yet
    assert env.collector.resolve(f"SOURCE_ITEM:{SID1}")["validity"] == "a LIVE_ACQUISITION is still OPENABLE"
    op = env.collector.manual_retry({"kind": "LIVE_ACQUISITION", "sid": SID1})
    key = identity.manual_episode_key(op)
    ledger.open_episode(env.store, key, {"kind": "MANUAL_RETRY", "sid": SID1, "work": env.store.rows("OPERATOR")[-1].body["work"]})
    env.clock.sleep(60)
    env.collector.run_episode(env.store.rows("EPISODE_OPEN", key=key)[0])
    assert state.anchor(env.store, SID1) is not None
    assert state.openable_acquisitions(env.store, SID1) == []
    assert ledger.episode_status(env.store, key) == "SUCCEEDED"
    assert env.collector.resolve(f"SOURCE_ITEM:{SID1}")["valid"]
