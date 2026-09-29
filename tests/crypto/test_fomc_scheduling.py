"""ELIGIBLE_CLASS_ALTERNATION_V1: durable class rotation, ordering within each class (backfill by
manifest commit_seq, next eligible instant, source_item_id), redirect continuations inside their
logical fetch, FOMC126 (feed versus a persistent recheck backlog) and the same choice after restart."""

from __future__ import annotations

import json

import pytest

from scripts.trading_lab.fomc import identity, ledger, spec, state
from scripts.trading_lab.fomc import synthetic as syn

from tests.crypto.fomc_support import Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _log_decisions(env):
    """Record, for every selection decision, the classes with eligible work and the chosen class."""
    decisions, seen = [], {}
    collector = env.collector
    eligible, select = collector.eligible_work, collector.select

    def logged_eligible():
        work = eligible()
        seen["classes"] = {w["class"] for w in work}
        return work

    def logged_select():
        chosen = select()
        if chosen is not None:
            decisions.append((frozenset(seen["classes"]), chosen["class"]))
        return chosen
    collector.eligible_work, collector.select = logged_eligible, logged_select
    return decisions


def _choice(collector):
    chosen = collector.select()
    return ("FEED_POLL",) if chosen.get("poll") else (chosen["class"], chosen["episode"].key)


def _first_in_class(collector, klass):
    work = [w for w in collector.eligible_work() if w["class"] == klass]
    return min(work, key=lambda w: w["order"])["episode"].key if work else None


def _same_choice_after_restart(env, klass=None):
    """The feed cadence is a liveness timer reset by a restart: wait it out so it is due on both sides.
    Returns the decision and, for `klass`, the first eligible work of that class; both must survive."""
    env.clock.sleep(spec.FEED_CADENCE_S + 1)
    before = (_choice(env.collector), _first_in_class(env.collector, klass))
    env.restart()
    after = (_choice(env.collector), _first_in_class(env.collector, klass))
    assert before == after
    return before[0] if klass is None else before


def _last_class(env):
    invoked = env.store.rows("TRANSPORT_INVOKED")
    return invoked[-1].body["class"] if invoked else None


def _limiter_respected(env):
    grants = sorted(t.body["grant_mono"] for t in env.store.rows("TRANSPORT_INVOKED"))
    assert all(b - a >= spec.SPACING_S for a, b in zip(grants, grants[1:]))
    assert all(sum(1 for g in grants if 0 <= t - g < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for t in grants)


# ------------------------------------------------------------------ FOMC126 ------------------------
PATHS = [syn.statement_path(f"202601{d:02d}") for d in range(1, 25)]  # FOMC126 names 10 items; more makes the backlog last
SIDS = {p: identity.source_item_id(syn.url(p)) for p in PATHS}


def test_fomc126_feed_and_a_persistent_recheck_backlog_alternate_without_starvation(env):
    env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate(PATHS)])
    for p in PATHS:
        env.provider.routes[p] = syn.page_response()
    env.drive(900, idle=30)
    assert all(state.anchor(env.store, SIDS[p]) is not None for p in PATHS)
    failing = PATHS[:5]  # permanently failing from now on: 404 on every recheck
    for p in failing:
        del env.provider.routes[p]
    injected = env.store.horizon()
    env.clock.sleep(2 * 3600)  # the collector is down: at the next verified poll every +300 s and +1 h recheck is PENDING_DUE
    decisions = _log_decisions(env)
    before = env.store.rows("TRANSPORT_INVOKED")
    env.drive(6 * 3600, idle=30)
    invoked = env.store.rows("TRANSPORT_INVOKED")[len(before) - 1:]  # starts with the last one before
    assert len(decisions) == len(invoked) - 1
    both = [i for i, (classes, _c) in enumerate(decisions) if {"FEED_DISCOVERY", "REOBSERVATION"} <= classes]
    assert len(both) >= 3  # the feed is due once a minute while the backlog lasts
    for i in both:  # strict alternation whenever both classes are eligible
        assert decisions[i][1] != invoked[i].body["class"]
    for p in PATHS[len(failing):]:  # healthy items: every due recheck up to +1 h is served
        obligations = {o["offset"]: o for o in state.obligations(env.store, SIDS[p])}
        assert obligations[300]["satisfied"] and obligations[3600]["satisfied"]
    for p in failing:  # bounded and final: 6 attempts per episode, then SUSPENDED
        episodes = [e for e in state.item_episodes(env.store, SIDS[p]) if e.body["kind"] == "REOBSERVATION"]
        assert episodes and all(len(ledger.attempts_of_episode(env.store, e.key)) <= spec.ATTEMPTS_PER_EPISODE for e in episodes)
        first = next(e for e in episodes if e.seq > injected)  # the first recheck that meets the failure
        assert ledger.episode_status(env.store, first.key) == "SUSPENDED"
    polls = [t.body["grant_mono"] for t in invoked[1:] if t.body["kind"] == "FEED_POLL"]
    assert max(b - a for a, b in zip(polls, polls[1:])) <= spec.FEED_CADENCE_S + 3 * spec.SPACING_S  # feed never starved
    _limiter_respected(env)
    requests = [env.provider.requests.count(p) for p in failing]
    assert all(n <= 1 + spec.ATTEMPTS_PER_EPISODE * 4 * len(spec.OFFSETS_S) for n in requests)


def test_fomc126_the_same_choice_is_made_after_restart(env):
    env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate(PATHS)])
    for p in PATHS:
        env.provider.routes[p] = syn.page_response()
    env.drive(900, idle=30)
    for p in PATHS[:5]:
        del env.provider.routes[p]
    env.clock.sleep(2 * 3600)
    checked = set()
    for _ in range(400):
        if env.collector.step() is None:
            env.clock.sleep(30)
        classes = {w["class"] for w in env.collector.eligible_work()}
        if "REOBSERVATION" in classes and _last_class(env) not in checked:
            checked.add(_last_class(env))
            choice = _same_choice_after_restart(env)
            assert choice[0] == ("REOBSERVATION" if _last_class(env) == "FEED_DISCOVERY" else "FEED_POLL")
        if checked == {"FEED_DISCOVERY", "REOBSERVATION"}:
            break
    assert checked == {"FEED_DISCOVERY", "REOBSERVATION"}


# ------------------------------------------------------------------ backfill order ----------------
def _manifest(env, paths):
    doc = {"version": 1, "urls": [syn.url(p) for p in paths]}
    return env.collector.submit_manifest(json.dumps(doc).encode(), "operator")["seq"]


def test_backfill_is_ordered_by_manifest_then_next_eligible_instant_then_sid_and_after_restart(env):
    pool = sorted((syn.statement_path(f"2019{m:02d}{d:02d}") for m in range(1, 13) for d in (1, 8, 15, 22)),
                  key=lambda p: identity.source_item_id(syn.url(p)))
    m1_paths, m2_paths = pool[-2:], pool[:40]  # every sid of the second manifest sorts before the first's
    retried, other = m1_paths  # the smaller sid of manifest 1 fails once, then succeeds
    redirected = m2_paths[5]
    sid = {p: identity.source_item_id(syn.url(p)) for p in m1_paths + m2_paths}
    for p in m1_paths + m2_paths:
        env.provider.routes[p] = syn.page_response()
    del env.provider.routes[retried]
    env.provider.routes[redirected] = syn.SyntheticResponse(status=302, headers=[("Location", redirected + "x")])
    env.provider.routes[redirected + "x"] = syn.page_response()
    env.feed([])
    m1 = _manifest(env, m1_paths)
    m2 = _manifest(env, m2_paths)
    assert m1 < m2
    env.collector.poll_feed()
    assert _same_choice_after_restart(env)[0] == "HISTORICAL_BACKFILL"
    order = []
    restarted_on_retry = False
    for _ in range(900):
        if env.collector.step() is None:
            env.clock.sleep(30)
        order = [t for t in env.store.rows("TRANSPORT_INVOKED") if t.body["class"] == "HISTORICAL_BACKFILL"]
        if len(order) == 1:
            env.provider.routes[retried] = syn.page_response()  # the retry will succeed
        work = env.collector.eligible_work()
        hb = [w for w in work if w["class"] == "HISTORICAL_BACKFILL"]
        if not restarted_on_retry and any(w["episode"].body["sid"] == sid[retried] for w in hb) and len(hb) > 1:
            restarted_on_retry = True  # the retry and later manifest-2 entries are eligible together
            choice, first = _same_choice_after_restart(env, "HISTORICAL_BACKFILL")
            assert first == identity.backfill_episode_key(m1, sid[retried])
            expected = ("HISTORICAL_BACKFILL", first) if _last_class(env) == "FEED_DISCOVERY" else ("FEED_POLL",)
            assert choice == expected
        if len({t.body["sid"] for t in order}) == 42 and len(order) == 43:
            break
    sids = [t.body["sid"] for t in order]
    assert restarted_on_retry
    assert sids[:2] == [sid[retried], sid[other]]  # manifest 1 first, although every manifest-2 sid is smaller
    m2_order = [s for s in sids[2:] if s != sid[retried]]
    assert m2_order == sorted(sid[p] for p in m2_paths)  # then manifest 2 in sid order
    retry_at = sids.index(sid[retried], 1)
    assert 2 < retry_at < len(sids) - 1  # the retry is taken as soon as eligible, ahead of manifest 2's remaining entries
    assert all(len(ledger.attempts_of_episode(env.store, identity.backfill_episode_key(m2, sid[p]))) == 1 for p in m2_paths)
    # the redirect continuation stays inside its logical fetch: next physical request, no new TRANSPORT_INVOKED
    i = env.provider.requests.index(redirected)
    assert env.provider.requests[i + 1] == redirected + "x"
    hops = [t for t in env.store.rows("TRANSPORT_INVOKED") if t.body.get("sid") == sid[redirected]]
    assert len(hops) == 1  # one TRANSPORT_INVOKED for two physical requests: rotation only sees the logical fetch
    record = next(r for r in env.store.rows("RESPONSE") if r.body["attempt"] == hops[0].seq)
    assert record.body["redirect_chain"] == [syn.url(redirected), syn.url(redirected + "x")]
    _limiter_respected(env)
