"""FOMC source health per surface: committed with the outcome that determines it, resolved by
failure_classification_v1.precedence, exposed only under FOMC_RESOLVED within P(T), bound into the
snapshot identity and re-derived by replay at the same (T, H)."""

from __future__ import annotations

from datetime import timedelta
import shutil
import sqlite3

import pytest

from scripts.trading_lab.fomc import ledger, snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.store import FomcStore

from tests.crypto.fomc_support import P1, SID1, Env, statement_item

NO_PROVIDER = "NO_PROVIDER_HEALTH_STATE"
STALE_DATE = "Mon, 01 Jun 2026 00:00:00 GMT"  # 16 days off: this response alone is CLOCK_UNVERIFIED


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _health(env, surface):
    return env.store.rows("SOURCE_HEALTH", key=surface)


def _same_txn_as_outcome(env, row):
    kinds = {r.kind for r in env.store.rows() if r.seq == row.seq}
    return bool(kinds & {"PROCESSING_OUTCOME", "ATTEMPT_OUTCOME"})


def _bad_page(**kw):
    return syn.SyntheticResponse(body=b"<html><body><p>no anchors</p></body></html>", headers=list(syn.HTML_HEADERS), **kw)


@pytest.mark.parametrize("route, expected, verdict", [
    (lambda: syn.page_response(), (None, "NO_FAILURE"), "CLOCK_VERIFIED"),  # a successful check records no failure state
    (lambda: syn.page_response(title="Federal Reserve issues something else"), (None, "NO_FAILURE"), "CLOCK_VERIFIED"),
    (lambda: syn.page_response(date=STALE_DATE), (NO_PROVIDER, "CLOCK_UNVERIFIED"), "CLOCK_UNVERIFIED"),
    (lambda: _bad_page(), ("PARSER_FAILED", "PARSER_FAILED"), "CLOCK_VERIFIED"),
    (lambda: _bad_page(date=STALE_DATE), ("PARSER_FAILED", "PARSER_FAILED"), "CLOCK_UNVERIFIED"),  # precedence
    (None, ("SOURCE_UNAVAILABLE", "SOURCE_UNAVAILABLE"), None),  # 404: no record
])
def test_primary_health_follows_the_precedence_and_never_marks_the_feed(env, route, expected, verdict):
    env.feed([statement_item()])
    if route is not None:
        env.provider.routes[P1] = route()
    env.drive(120)
    first = _health(env, "primary_statement")[0]
    assert (first.body["result_state"], first.body["reason"]) == expected
    assert first.body["provider_id"] == spec.PROVIDER_ID and first.body["sid"] == SID1
    assert _same_txn_as_outcome(env, first)  # durable with, and only with, its outcome
    record = env.store.row_at("RESPONSE", first.body["record"]) if first.body["record"] is not None else None
    assert (record.body["verdict"] if record else None) == verdict  # the clock verdict is kept whatever the state
    feed = _health(env, "discovery_feed")
    assert feed and all(r.body["result_state"] in (None, "EVENTS_OBSERVED_ZERO") for r in feed)  # a primary failure stays on its surface
    assert "EVENTS_OBSERVED_ZERO" not in env.cycles()[:1]  # the new item's cycle is never a zero


def test_feed_zero_local_failures_and_cancellation_map_as_specified(env):
    env.feed([])
    env.drive(200)
    assert _health(env, "discovery_feed")[-1].body["result_state"] == "EVENTS_OBSERVED_ZERO"
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.poll_feed()
    env.collector.open_episodes()
    episode = next(e for e in env.store.rows("EPISODE_OPEN") if e.body["kind"] == "LIVE_ACQUISITION")
    work = {"kind": "LIVE_ACQUISITION", "class": "FEED_DISCOVERY", "episode_key": episode.key, "sid": SID1, "mode": "LIVE"}
    assert _health(env, "discovery_feed")[-1].body["result_state"] is None  # NOT_ZERO: a successful check
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise OSError("local store unavailable")
    env.collector.persist_fault = failing
    lpf = env.collector.run(work, syn.url(P1), "primary")
    env.collector.persist_fault = None
    assert lpf["status"] == "LOCAL_PERSISTENCE_FAILED"
    last = _health(env, "primary_statement")[-1].body
    assert (last["result_state"], last["reason"], last["outcome"]) == (NO_PROVIDER, "LOCAL_PERSISTENCE_FAILURE", "LOCAL_PERSISTENCE_FAILED")
    env.drive(400)  # the retry succeeds
    fetched = env.collector.fetch({"kind": "FEED_POLL", "class": "FEED_DISCOVERY", "sid": None, "mode": "LIVE"}, spec.FEED_URL, "feed")
    env.restart()  # the poll's task dies with the old owner
    assert _health(env, "discovery_feed")[-1].body["reason"] == "INTERRUPTED"
    before = len(env.store.rows("SOURCE_HEALTH"))
    attempt = ledger.transport_invoked(env.store, epoch=env.collector.epoch, grant_mono=env.clock.mono(),
                                       work={"kind": "FEED_POLL", "class": "FEED_DISCOVERY", "sid": None, "mode": "LIVE"})
    ledger.commit_attempt_outcome(env.store, attempt, "CANCELLED_AFTER_INVOKE", {"reason": "cancelled"})
    assert len(env.store.rows("SOURCE_HEALTH")) == before  # cancellation: no source-health result
    assert fetched.attempt_seq != attempt


def test_raw_corruption_before_processing_is_no_provider_state(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.poll_feed()
    env.collector.open_episodes()
    episode = next(e for e in env.store.rows("EPISODE_OPEN") if e.body["kind"] == "LIVE_ACQUISITION")
    work = {"kind": "LIVE_ACQUISITION", "class": "FEED_DISCOVERY", "episode_key": episode.key, "sid": SID1, "mode": "LIVE"}
    fetched = env.collector.fetch(work, syn.url(P1), "primary")
    digest = env.store.put_raw(fetched.body)
    ledger.commit_response(env.store, fetched.attempt_seq, env.collector._fields(fetched, work, syn.url(P1), "primary", digest))
    path = env.root / "raw" / digest[:2] / digest
    path.chmod(0o644)
    path.write_bytes(b"corrupted")
    env.collector.process_pending()
    last = _health(env, "primary_statement")[-1].body
    assert (last["result_state"], last["reason"], last["outcome"]) == (NO_PROVIDER, "RAW_CORRUPTION", "CORRUPTION_FAIL_CLOSED")


def _T(store, H, *, before_seq=None):
    """The latest resolved instant (optionally one whose prefix P(T) ends before transaction `before_seq`)."""
    table = [e for e in state.availability(store, H) if e.resolved]
    limit = state.avail_of(table, before_seq) if before_seq is not None else table[-1].avail
    return max(e.avail for e in table if e.avail < limit)  # a later resolved transaction is the watermark


def test_health_is_exposed_only_when_resolved_bound_to_identity_and_replayed(env, tmp_path):
    env.feed([])
    env.drive(200)
    env.feed([statement_item()])  # its primary page is missing: SOURCE_UNAVAILABLE on primary_statement only
    env.drive(400)
    store, H = env.store, env.store.horizon()
    first_primary = next(t.seq for t in store.rows("TRANSPORT_INVOKED") if t.body["kind"] != "FEED_POLL")
    early = snapshot.events_as_of(store, _T(store, H, before_seq=first_primary), H)
    assert early["read_state"] == "FOMC_RESOLVED"
    assert early["health"]["primary_statement"] == {"result_state": "SOURCE_NOT_CHECKED"}
    assert early["health"]["discovery_feed"]["result_state"] == "EVENTS_OBSERVED_ZERO"
    T = _T(store, H)
    snap = snapshot.events_as_of(store, T, H)
    assert snap["read_state"] == "FOMC_RESOLVED"
    assert snap["health"]["primary_statement"]["result_state"] == "SOURCE_UNAVAILABLE"
    assert snap["health"]["discovery_feed"]["result_state"] is None  # NOT_ZERO: the feed check itself succeeded
    assert snap["discovery"]["state"] == "NOT_ZERO"  # an outage is never a zero
    assert snap["health"]["primary_statement"]["check"] <= snap["P"]  # read within P(T) only
    unresolved = snapshot.events_as_of(store, env.clock.wall() + timedelta(hours=1), H)  # no resolved watermark yet
    retro = snapshot.events_as_of(store, T, H, mode="RETROSPECTIVE_SOURCE")
    assert unresolved["read_state"] == "FOMC_CAUSAL_VISIBILITY_UNRESOLVED" and "health" not in unresolved
    assert retro["read_state"] == "FOMC_NOT_ADMISSIBLE_IN_MODE" and "health" not in retro
    payload = {k: v for k, v in snap.items() if k != "identity"}
    assert "health" in payload and snap["identity"] == spec.sha256_canonical(payload)
    assert snapshot.replay(store, T, H) == snap  # same (T, H): same health, same identity
    # a copy whose recorded health differs: another identity, and replay fails closed
    store._conn.execute("PRAGMA wal_checkpoint(FULL)")
    copy = tmp_path / "copy"
    shutil.copytree(env.root, copy)
    with sqlite3.connect(copy / "fomc.sqlite3") as conn:
        conn.execute("UPDATE rec SET body = replace(body, '\"SOURCE_UNAVAILABLE\"', '\"PARSER_FAILED\"') WHERE kind = 'SOURCE_HEALTH'")
    tampered = FomcStore(copy, wall_clock=env.clock.wall)
    try:
        other = snapshot.events_as_of(tampered, T, H)
        assert other["health"]["primary_statement"]["result_state"] == "PARSER_FAILED"
        assert other["identity"] != snap["identity"]
        with pytest.raises(snapshot.ReplayFailed):
            snapshot.replay(tampered, T, H)
    finally:
        tampered.close()
