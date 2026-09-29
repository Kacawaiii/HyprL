"""Local deadlines of spec rev 22 with an injectable monotonic clock: the 120 s save deadline after the
network end, the 600 s absolute attempt deadline, the 600 s processing-run deadline, their exact
boundaries, restart, budget conservation, protection against late results and no false zero."""

from __future__ import annotations

import pytest

from scripts.trading_lab.fomc import identity, ledger, spec, state
from scripts.trading_lab.fomc import synthetic as syn

from tests.crypto.fomc_support import P1, SID1, Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _acquisition(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.poll_feed()
    env.collector.open_episodes()
    episode = next(e for e in env.store.rows("EPISODE_OPEN") if e.body["kind"] == "LIVE_ACQUISITION")
    work = {"kind": "LIVE_ACQUISITION", "class": "FEED_DISCOVERY", "episode_key": episode.key, "sid": SID1, "mode": "LIVE"}
    return episode, work


def _network_still_open(env, fetched):
    """Model a network phase that has not ended yet: no save deadline runs until network_ended()."""
    env.collector._save_deadlines.pop(fetched.attempt_seq, None)


def _failing(times):
    calls = {"n": 0}

    def fault():
        calls["n"] += 1
        if calls["n"] <= times:
            raise OSError("local store unavailable")
    return fault


# ------------------------------------------------------------------ 120 s save deadline ---------------
@pytest.mark.parametrize("failures, expected", [(23, "RESPONSE"), (24, "LOCAL_PERSISTENCE_FAILED")])
def test_save_retries_until_120_s_after_the_network_end_without_refetching(env, failures, expected):
    episode, _work = _acquisition(env)
    env.collector.persist_fault = _failing(failures)  # retries every 5 s: the 24th try is at +115 s, the 25th would be at +120 s
    result = env.collector.run_episode(episode)
    env.collector.persist_fault = None
    assert result["status"] == expected
    assert env.provider.requests.count(P1) == 1  # local retries never cause a provider request
    assert len(ledger.attempts_of_episode(env.store, episode.key)) == 1
    responses = state.primary_responses(env.store, SID1)
    assert len(responses) == (1 if expected == "RESPONSE" else 0)
    if expected == "LOCAL_PERSISTENCE_FAILED":
        assert state.anchor(env.store, SID1) is None
        assert ledger.outcome_of(env.store, ledger.attempts_of_episode(env.store, episode.key)[0].seq).body["reason"].startswith("local save")


def _no_record_of(env, attempt):
    """Neither a RESPONSE nor LATE_EVIDENCE exists for `attempt`."""
    return [r for r in env.store.rows("RESPONSE") if r.body["attempt"] == attempt] == []


def _no_false_zero_then_attempt_2(env, episode):
    env.collector.poll_feed()  # the item is still outstanding: the next cycle cannot be a zero
    assert env.cycles()[-1] == "NOT_ZERO"
    env.drive(600, idle=60)
    attempts = ledger.attempts_of_episode(env.store, episode.key)
    assert len(attempts) == 2 and state.anchor(env.store, SID1) is not None  # the budget is kept: next is #2
    assert env.provider.requests.count(P1) == 2


@pytest.mark.parametrize("duration, expected", [(119.9, "RESPONSE"), (120.0, "LOCAL_PERSISTENCE_FAILED"),
                                                (120.1, "LOCAL_PERSISTENCE_FAILED")])
def test_a_slow_local_write_that_succeeds_is_admitted_only_before_120_s(env, monkeypatch, duration, expected):
    episode, _work = _acquisition(env)
    put_raw = env.store.put_raw

    def slow_put_raw(body):  # starts at +0 s, succeeds after `duration`
        env.clock.sleep(duration)
        return put_raw(body)
    monkeypatch.setattr(env.store, "put_raw", slow_put_raw)
    result = env.collector.run_episode(episode)
    monkeypatch.undo()
    assert result["status"] == expected
    assert env.provider.requests.count(P1) == 1
    attempt = ledger.attempts_of_episode(env.store, episode.key)[0].seq
    assert ledger.outcome_of(env.store, attempt).body["outcome"] == expected
    if expected == "RESPONSE":
        assert state.anchor(env.store, SID1) is not None
        return
    assert _no_record_of(env, attempt) and state.anchor(env.store, SID1) is None  # at 120 s, expired
    _no_false_zero_then_attempt_2(env, episode)


def test_reconcile_at_120_s_closes_a_blocked_save_and_its_late_bytes_stay_out(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")  # network ends; the saving task then blocks
    env.clock.sleep(119.9)
    env.collector.reconcile()
    assert ledger.outcome_of(env.store, fetched.attempt_seq) is None
    env.clock.sleep(0.1)  # +120 s, no commit() in between
    env.collector.reconcile()
    outcome = ledger.outcome_of(env.store, fetched.attempt_seq).body
    assert outcome["outcome"] == "LOCAL_PERSISTENCE_FAILED" and outcome["reason"].startswith("local save")
    late = env.collector.commit(fetched, work, syn.url(P1), "primary")  # the blocked task finally resumes
    assert late["status"] == "LOCAL_PERSISTENCE_FAILED"
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body == outcome  # the first outcome stays
    assert _no_record_of(env, fetched.attempt_seq) and state.anchor(env.store, SID1) is None
    assert env.provider.requests.count(P1) == 1
    _no_false_zero_then_attempt_2(env, episode)


# ------------------------------------------------------------------ 600 s attempt deadline ------------
def test_active_attempt_is_interrupted_exactly_at_600_s_and_its_late_response_is_evidence(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")  # the task is alive
    env.clock.sleep(599.5)
    env.collector.network_ended(fetched)  # a slow network phase (limiter waits between hops) ends at +599.5 s
    env.collector.reconcile()
    assert ledger.outcome_of(env.store, fetched.attempt_seq) is None  # never declared dead before its deadline
    env.clock.sleep(0.5)
    env.collector.reconcile()
    outcome = ledger.outcome_of(env.store, fetched.attempt_seq).body
    assert outcome["outcome"] == "INTERRUPTED" and outcome["reason"] == "no outcome 600 s after TRANSPORT_INVOKED"
    late = env.collector.commit(fetched, work, syn.url(P1), "primary")
    assert late["late"] is True and state.anchor(env.store, SID1) is None  # the closure is protected
    assert len(ledger.attempts_of_episode(env.store, episode.key)) == 1  # counted once, never refunded


def test_commit_after_the_attempt_deadline_closes_it_first(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")
    env.clock.sleep(600)
    env.collector.network_ended(fetched)  # the bytes arrive at +600 s, inside their own save deadline
    result = env.collector.commit(fetched, work, syn.url(P1), "primary")  # no reconcile in between
    assert result["late"] is True
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body["outcome"] == "INTERRUPTED"


def test_bytes_held_past_their_save_deadline_are_not_durable_and_not_late_evidence(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")  # network ends now
    env.clock.sleep(600)  # the task stalls before saving: its 120 s save deadline passed long ago
    result = env.collector.commit(fetched, work, syn.url(P1), "primary")
    assert result["status"] == "LOCAL_PERSISTENCE_FAILED"
    assert state.primary_responses(env.store, SID1) == []  # no RESPONSE, no LATE_EVIDENCE, no anchor
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body["outcome"] == "INTERRUPTED"  # the first outcome stays


def test_restart_interrupts_an_active_attempt_at_once_and_keeps_the_budget(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")
    env.clock.sleep(1)
    env.restart()  # new owner, new epoch
    outcome = ledger.outcome_of(env.store, fetched.attempt_seq).body
    assert outcome["outcome"] == "INTERRUPTED" and outcome["reason"] == "non-current epoch"
    env.drive(600, idle=60)
    attempts = ledger.attempts_of_episode(env.store, episode.key)
    assert len(attempts) == 2 and state.anchor(env.store, SID1) is not None  # the next attempt is #2


def test_a_held_feed_poll_blocks_new_polls_until_its_deadline_without_false_zero(env):
    env.feed([])
    env.drive(200)
    cycles_before = len(env.store.rows("CYCLE_CONCLUSION"))
    env.clock.sleep(60)
    poll = {"kind": "FEED_POLL", "class": "FEED_DISCOVERY", "sid": None, "mode": "LIVE"}
    fetched = env.collector.fetch(poll, spec.FEED_URL, "feed")
    _network_still_open(env, fetched)
    polls = env.provider.requests.count(syn.FEED_PATH)
    for _ in range(9):
        env.clock.sleep(60)
        env.collector.step()
    assert env.provider.requests.count(syn.FEED_PATH) == polls  # at most one feed poll in flight
    assert len(env.store.rows("CYCLE_CONCLUSION")) == cycles_before  # no cycle, no zero from a held poll
    env.clock.sleep(60)  # 600 s after TRANSPORT_INVOKED
    env.collector.step()
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body["outcome"] == "INTERRUPTED"
    env.collector.network_ended(fetched)  # its bytes only now arrive
    assert env.collector.commit(fetched, poll, spec.FEED_URL, "feed")["late"] is True
    late = state.feed_responses(env.store)[-1]
    assert late.body["late_evidence"] and not env.store.rows("CYCLE_CONCLUSION", key=str(late.seq))  # no cycle
    env.drive(200)
    assert env.provider.requests.count(syn.FEED_PATH) > polls  # polling resumed


# ------------------------------------------------------------------ 600 s processing-run deadline -----
def _unprocessed_record(env):
    episode, work = _acquisition(env)
    fetched = env.collector.fetch(work, syn.url(P1), "primary")
    digest = env.store.put_raw(fetched.body)
    seq, _late = ledger.commit_response(env.store, fetched.attempt_seq, env.collector._fields(fetched, work, syn.url(P1), "primary", digest))
    env.collector._active_attempts.pop(fetched.attempt_seq, None)
    return seq


def test_a_running_run_is_not_dead_before_600_s_and_is_replaced_at_600_s(env):
    seq = _unprocessed_record(env)
    run1 = env.collector.start_processing(seq)
    env.clock.sleep(599.5)
    env.collector.process_pending()
    assert len(env.store.rows("PROCESSING_RUN", key=str(seq))) == 1 and env.store.rows("RUN_DEAD", key=str(seq)) == []
    env.clock.sleep(0.5)
    env.collector.process_pending()
    dead = env.store.rows("RUN_DEAD", key=str(seq))
    assert [d.body["reason"] for d in dead] == ["run deadline passed"]
    outcome = state.processing_outcome(env.store, seq).body["outcome"]
    assert outcome == "NORMALIZED_REVISION_COMMITTED"  # committed by the second run
    assert env.collector.finish_processing(seq, run1) is None  # the late first run is fenced
    assert state.processing_outcome(env.store, seq).body["outcome"] == outcome


@pytest.mark.parametrize("overrun, committed", [(599.9, True), (600.0, False)])
def test_run_result_is_admitted_only_before_its_deadline(env, overrun, committed):
    seq = _unprocessed_record(env)
    env.collector.processing_fault = lambda resp: env.clock.sleep(overrun)
    run_id = env.collector.start_processing(seq)
    outcome = env.collector.finish_processing(seq, run_id)
    assert (outcome is not None) is committed
    reasons = [d.body["reason"] for d in env.store.rows("RUN_DEAD", key=str(seq))]
    assert reasons == ([] if committed else ["late result after the 600 s run deadline"])


def test_overrunning_feed_runs_are_poisoned_without_false_zero(env):
    env.feed([])
    env.drive(200)
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    env.collector.processing_fault = lambda resp: env.clock.sleep(600) if resp.body["surface"] == "feed" else None
    env.clock.sleep(60)
    env.collector.step()  # feed record F; run 1 overruns and is DEAD
    feed = state.feed_responses(env.store)[-1]
    polls = env.provider.requests.count(syn.FEED_PATH)
    env.collector.step()  # run 2 overruns and is DEAD
    env.collector.processing_fault = None
    env.collector.step()  # two DEAD runs: the owner ends F with INTERNAL_PROCESSING_ERROR
    assert state.processing_outcome(env.store, feed.seq).body["outcome"] == "INTERNAL_PROCESSING_ERROR"
    assert env.store.rows("CYCLE_CONCLUSION", key=str(feed.seq))[0].body["result"] == "NOT_ZERO"
    assert env.provider.requests.count(syn.FEED_PATH) <= polls + 1
    assert "EVENTS_OBSERVED_ZERO" not in [c.body["result"] for c in env.store.rows("CYCLE_CONCLUSION") if c.seq > feed.seq][:1]
