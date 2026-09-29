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


# ------------------------------------------------------------------ corruption after the outcome -----
def _raw(env, digest):
    return env.root / "raw" / digest[:2] / digest


def test_corruption_found_after_the_outcome_keeps_it_and_commits_diagnostic_and_health_together(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(240)
    store = env.store
    H1 = store.horizon()
    T1 = _T(store, H1)
    old = snapshot.events_as_of(store, T1, H1)
    assert next(i for i in old["items"] if i["sid"] == SID1)["state"] == "CURRENT_REVISION"
    record = state.primary_responses(store, SID1)[-1]
    kept = state.processing_outcome(store, record.seq).body["outcome"]
    requests, invoked = len(env.provider.requests), len(store.rows("TRANSPORT_INVOKED"))
    _raw(env, record.body["raw_sha"]).write_bytes(b"corrupted")
    assert env.collector.verify_integrity() == [record.seq]
    assert env.collector.verify_integrity() == [record.seq]  # still corrupt; diagnosed once
    diagnostic = store.rows("INTEGRITY_DIAGNOSTIC", key=str(record.seq))
    assert len(diagnostic) == 1 and diagnostic[0].body == {"record": record.seq, "raw_sha": record.body["raw_sha"], "outcome": kept}
    same_txn = [r for r in store.rows() if r.seq == diagnostic[0].seq]
    assert sorted(r.kind for r in same_txn) == ["INTEGRITY_DIAGNOSTIC", "SOURCE_HEALTH"]  # one atomic transaction
    check = next(r for r in same_txn if r.kind == "SOURCE_HEALTH")
    assert check.key == "primary_statement"
    assert (check.body["result_state"], check.body["reason"], check.body["outcome"], check.body["record"]) == \
        (NO_PROVIDER, "RAW_CORRUPTION", kept, record.seq)
    assert state.processing_outcome(store, record.seq).body["outcome"] == kept  # the terminal outcome is kept
    assert len(env.provider.requests) == requests and len(store.rows("TRANSPORT_INVOKED")) == invoked  # no refetch
    env.drive(130)  # later verified feed polls resolve the diagnostic; the +300 s recheck is not due yet
    H2 = store.horizon()
    T2 = _T(store, H2)
    snap = snapshot.events_as_of(store, T2, H2)
    assert snap["read_state"] == "FOMC_RESOLVED" and snap["P"] >= diagnostic[0].seq
    item = next(i for i in snap["items"] if i["sid"] == SID1)
    assert item["step"] == 3 and item["state"] == snapshot.BARRIER and "revision" not in item  # corruption barrier
    assert snap["health"]["primary_statement"]["reason"] == "RAW_CORRUPTION"
    assert snap["health"]["primary_statement"]["record"] == record.seq == item["newest"]["record"]  # consistent
    assert snap["health"]["discovery_feed"]["result_state"] in (None, "EVENTS_OBSERVED_ZERO")  # the feed is untouched
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(store, T1, H1)  # the older (T, H) depended on that raw: it fails closed
    snapshot.verify_health(store, H2)  # health, diagnostic included, re-derives
    with pytest.raises(snapshot.ReplayFailed):
        snapshot.replay(store, T2, H2)  # the corrupt raw fails verified replay closed


# ------------------------------------------------------------------ health replay, field by field ----
@pytest.fixture
def health_store(env, tmp_path):
    """A store with the three kinds of health rows: an attempt outcome (404), a processed record and
    an integrity diagnostic."""
    env.feed([statement_item()])
    env.drive(90)  # the first acquisition attempt meets a 404
    env.provider.routes[P1] = syn.page_response()
    env.drive(400)
    record = state.anchor(env.store, SID1)
    assert record is not None
    _raw(env, record.body["raw_sha"]).write_bytes(b"corrupted")
    env.collector.verify_integrity()
    env.store._conn.execute("PRAGMA wal_checkpoint(FULL)")
    rows = env.store.rows("SOURCE_HEALTH")
    kinds = {"attempt": next(r for r in rows if r.body["outcome"] == "SOURCE_UNAVAILABLE"),
             "record": next(r for r in rows if r.body["record"] == record.seq and r.body["reason"] != "RAW_CORRUPTION"),
             "diagnostic": next(r for r in rows if r.body["reason"] == "RAW_CORRUPTION")}
    return env, kinds, tmp_path


def _copy(env, tmp_path, name, sql, args=()):
    copy = tmp_path / name
    shutil.copytree(env.root, copy)
    with sqlite3.connect(copy / "fomc.sqlite3") as conn:
        assert conn.execute(sql, args).rowcount == 1
    return FomcStore(copy, wall_clock=env.clock.wall)


FIELDS = {"provider_id": '"another_provider"', "surface": '"discovery_feed"', "check_at": '"2026-06-17T18:00:00+00:00"',
          "result_state": '"EVENTS_OBSERVED_ZERO"', "reason": '"CLOCK_UNVERIFIED"', "outcome": '"INTERRUPTED"',  # held by none of the rows
          "attempt": "999", "record": "999", "sid": '"0000"', "diagnostics": '{"reason": "edited"}'}


def test_health_replay_checks_every_durable_field_of_every_row(health_store):
    env, kinds, tmp_path = health_store
    H = env.store.horizon()
    snapshot.verify_health(env.store, H)  # the untouched store re-derives
    undetected = []
    for kind, row in kinds.items():
        edits = [(field, "UPDATE rec SET body = json_set(body, ?, json(?)) WHERE kind = 'SOURCE_HEALTH' AND commit_seq = ?",
                  (f"$.{field}", value, row.seq)) for field, value in FIELDS.items()]
        edits.append(("key", "UPDATE rec SET key = 'discovery_feed' WHERE kind = 'SOURCE_HEALTH' AND commit_seq = ?", (row.seq,)))
        for field, sql, args in edits:
            copy = _copy(env, tmp_path, f"{kind}-{field}", sql, args)
            try:
                edited = next(r for r in copy.rows("SOURCE_HEALTH") if r.seq == row.seq)
                assert (edited.key, edited.body) != (row.key, row.body)  # the edit really changed the row
                snapshot.verify_health(copy, H)
                undetected.append((kind, field, row.body.get(field)))
            except snapshot.ReplayFailed:
                pass
            finally:
                copy.close()
    assert undetected == []  # every isolated edit of every field of every kind of row fails closed


def test_health_replay_binds_check_at_to_durable_provenance_and_fails_on_missing_rows(health_store):
    env, kinds, tmp_path = health_store
    H = env.store.horizon()
    edits = {
        "attempt-wall": ("UPDATE txn SET wall_at_commit = '2026-06-17T18:00:00+00:00' WHERE commit_seq = ?", kinds["attempt"].seq),
        "diagnostic-wall": ("UPDATE txn SET wall_at_commit = '2026-06-17T18:00:00+00:00' WHERE commit_seq = ?", kinds["diagnostic"].seq),
        "record-receipt": ("UPDATE rec SET body = json_set(body, '$.wall_at_receipt', '2026-06-17T18:00:01+00:00') "
                           "WHERE kind = 'RESPONSE' AND commit_seq = ?", kinds["record"].body["record"]),
        "health-deleted": ("DELETE FROM rec WHERE kind = 'SOURCE_HEALTH' AND commit_seq = ?", kinds["record"].seq),
        "diagnostic-health-deleted": ("DELETE FROM rec WHERE kind = 'SOURCE_HEALTH' AND commit_seq = ?", kinds["diagnostic"].seq),
        "diagnostic-deleted": ("DELETE FROM rec WHERE kind = 'INTEGRITY_DIAGNOSTIC' AND commit_seq = ?", kinds["diagnostic"].seq),
        "diagnostic-outcome": ("UPDATE rec SET body = json_set(body, '$.outcome', 'PARSER_FAILED') "
                               "WHERE kind = 'INTEGRITY_DIAGNOSTIC' AND commit_seq = ?", kinds["diagnostic"].seq),
    }
    for name, (sql, seq) in edits.items():
        copy = _copy(env, tmp_path, name, sql, (seq,))
        try:
            with pytest.raises(snapshot.ReplayFailed):
                snapshot.verify_health(copy, H)
        finally:
            copy.close()
    # the diagnosed raw restored to its original bytes: raw is never repaired, so replay refuses it
    record = env.store.row_at("RESPONSE", kinds["diagnostic"].body["record"])
    restored = tmp_path / "restored"
    shutil.copytree(env.root, restored)
    (restored / "raw" / record.body["raw_sha"][:2] / record.body["raw_sha"]).write_bytes(
        env.provider.routes[P1].body)
    copy = FomcStore(restored, wall_clock=env.clock.wall)
    try:
        with pytest.raises(snapshot.ReplayFailed, match="verifies again"):
            snapshot.verify_health(copy, H)
    finally:
        copy.close()
