"""Checkpoint 3 - snapshot and replay: the eight cross-component demonstrations of the offline slice."""

from __future__ import annotations

from datetime import datetime, timedelta
import json

import pytest

from scripts.trading_lab.fomc import identity, ledger, snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.store import FomcStore

from tests.crypto.fomc_support import EXACT, P1, SID1, Env, statement_item

BODY_A = "The Committee decided to maintain the target range."
BODY_B = "The Committee decided to lower the target range."


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def settle(env):
    """Live reads at T = now need a verified response newer than 92 s (otherwise FOMC_CAUSAL_VISIBILITY_UNRESOLVED)."""
    env.drive(130, idle=30)


def item_state(snap, sid=SID1):
    return next(i for i in snap["items"] if i["sid"] == sid)


def served(body):
    return syn.statement_html(body=body)


def test_1_first_item_then_corrected_content(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(240)
    s1 = snapshot.events_as_of(env.store, env.clock.true)
    assert s1["read_state"] == "FOMC_RESOLVED"
    first = item_state(s1)
    assert first["state"] == "CURRENT_REVISION"
    assert env.store.read_raw(first["content_hash"]) == served(BODY_A)
    env.provider.routes[P1] = syn.page_response(body=BODY_B)  # upstream correction
    env.drive(900, idle=60)  # the O300 recheck observes it
    s2 = snapshot.events_as_of(env.store, env.clock.true)
    second = item_state(s2)
    assert second["state"] == "CURRENT_REVISION" and second["content_hash"] != first["content_hash"]
    assert env.store.read_raw(second["content_hash"]) == served(BODY_B)
    old = snapshot.events_as_of(env.store, datetime.fromisoformat(s1["T"]), s1["H"])
    assert old["identity"] == s1["identity"] and item_state(old)["content_hash"] == first["content_hash"]


def test_2_backfill_then_live_same_content_one_revision(env):
    env.feed([])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    manifest = env.collector.submit_manifest(json.dumps({"version": 1, "urls": [syn.url(P1)]}).encode(), "operator: FOMC page")
    assert manifest["valid"]
    env.drive(240)
    backfill = item_state(snapshot.events_as_of(env.store, env.clock.true))
    assert backfill["state"] == "CURRENT_REVISION" and backfill["live_available"] is False
    assert [l["mode"] for l in backfill["links"]] == ["HISTORICAL_BACKFILL"]
    assert env.cycles() and all(c == "EVENTS_OBSERVED_ZERO" for c in env.cycles())  # backfill never blocks LIVE zero
    env.feed([statement_item()])
    env.drive(300)
    live = item_state(snapshot.events_as_of(env.store, env.clock.true))
    assert len(env.store.rows("REVISION")) == 1
    assert live["revision"] == backfill["revision"] and live["live_available"] is True
    modes = {l["mode"]: l["avail"] for l in live["links"]}
    assert set(modes) == {"HISTORICAL_BACKFILL", "LIVE"} and modes["LIVE"] > modes["HISTORICAL_BACKFILL"]
    assert "NORMALIZED_SAME_CONTENT_NO_NEW_REVISION" in env.outcomes()


def test_3_clock_unverified_records_have_no_live_effect_and_no_false_zero(tmp_path):
    env = Env(tmp_path, wall_offset_s=300)
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response()
        env.drive(200)
        snap = snapshot.events_as_of(env.store, env.clock.true)
        assert snap["read_state"] == "FOMC_CAUSAL_VISIBILITY_UNRESOLVED"  # no verified time: nothing exposed
        assert "NORMALIZED_REVISION_COMMITTED" in env.outcomes() and env.store.rows("CYCLE_CONCLUSION") == []
        env.clock.wall_offset_s = 0
        env.drive(200)
        mid = snapshot.events_as_of(env.store, env.clock.true)
        assert mid["read_state"] == "FOMC_RESOLVED"
        assert item_state(mid)["state"] == snapshot.BARRIER  # its only revision link is unverified
        assert "EVENTS_OBSERVED_ZERO" not in env.cycles()
        env.drive(900, idle=60)
        final = snapshot.events_as_of(env.store, env.clock.true)
        assert item_state(final)["state"] == "CURRENT_REVISION" and item_state(final)["live_available"]
    finally:
        env.provider.close()


def test_4_suspended_and_omitted_item_blocks_zero_until_resolve_then_reopens(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(status=404, body=b"gone", headers=[("Content-Type", "text/plain")])
    env.drive(120)
    env.feed([])  # the item rotates out of the feed
    env.drive(40000, idle=900)
    key = identity.acquisition_episode_key(SID1, env.store.rows("ACQUISITION")[0].seq)
    assert ledger.episode_status(env.store, key) == "SUSPENDED"
    assert set(env.cycles()) == {"NOT_ZERO"}  # omitted but outstanding
    resolved = env.collector.resolve(f"SOURCE_ITEM:{SID1}")
    assert resolved["valid"]
    env.drive(150)
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
    env.collector.manual_retry({"kind": "LIVE_ACQUISITION", "sid": SID1})
    env.drive(150)
    assert env.cycles()[-1] == "NOT_ZERO"  # a later item record re-opened it


def test_5_crash_after_transport_invoked_restart_keeps_budget_and_snapshot(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.poll_feed()
    env.collector.open_episodes()
    episode = env.store.rows("EPISODE_OPEN")[0]
    work = {"kind": "LIVE_ACQUISITION", "class": "FEED_DISCOVERY", "episode_key": episode.key, "sid": SID1, "mode": "LIVE"}
    fetched = env.collector.fetch(work, syn.url(P1), "primary")
    env.restart()  # crash after TRANSPORT_INVOKED: INTERRUPTED, counted
    env.drive(600, idle=60)
    keys = {e.key for e in env.store.rows("EPISODE_OPEN") if e.body["kind"] != "MANUAL_RETRY"}
    assert len(keys) <= 5 and all(len(ledger.attempts_of_episode(env.store, k)) <= 6 for k in keys)
    assert len(ledger.attempts_of_episode(env.store, episode.key)) == 2
    snap = snapshot.events_as_of(env.store, env.clock.true)
    env.restart("boot-3")
    assert snapshot.events_as_of(env.store, env.clock.true, snap["H"])["identity"] == snap["identity"]
    assert ledger.outcome_of(env.store, fetched.attempt_seq).body["outcome"] == "INTERRUPTED"


def test_7_aba_unnormalized_source_and_corrupt_raw(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(240)
    env.provider.routes[P1] = syn.page_response(body=BODY_B)
    env.drive(900, idle=60)  # O300
    assert item_state(snapshot.events_as_of(env.store, env.clock.true))["content_hash"] == spec.sha256_bytes(served(BODY_B))
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(3600, idle=300)  # O3600: A again
    settle(env)
    aba = item_state(snapshot.events_as_of(env.store, env.clock.true))
    assert aba["step"] == 4 and aba["content_hash"] == spec.sha256_bytes(served(BODY_A))
    assert len(env.store.rows("REVISION")) == 2  # A-B-A reuses V_A
    env.provider.routes[P1] = syn.SyntheticResponse(body=b"<html><body>template changed</body></html>", headers=syn.HTML_HEADERS)
    env.drive(86400, idle=3600)  # O86400: newer bytes that do not normalize
    settle(env)
    barrier = item_state(snapshot.events_as_of(env.store, env.clock.true))
    assert barrier["state"] == snapshot.BARRIER and barrier["newest"]["outcome"] == "PARSER_FAILED"
    newest = state.primary_responses(env.store, SID1)[-1]
    raw = env.root / "raw" / newest.body["raw_sha"][:2] / newest.body["raw_sha"]
    raw.write_bytes(b"corrupted")
    assert env.collector.verify_integrity() == [newest.seq]
    env.drive(200)
    corrupt = item_state(snapshot.events_as_of(env.store, env.clock.true))
    assert corrupt["step"] == 3 and corrupt["state"] == snapshot.BARRIER  # never back to V_A
    with pytest.raises(snapshot.ReplayFailed):
        snapshot.replay(env.store, env.clock.true, env.store.horizon())


def test_8_same_T_H_after_restart_and_replay(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(400)
    T, H = env.clock.true, env.store.horizon()
    live = snapshot.events_as_of(env.store, T, H)
    env.drive(300)  # more records after H change nothing at (T, H)
    env.restart()
    assert snapshot.events_as_of(env.store, T, H)["identity"] == live["identity"]
    assert snapshot.replay(env.store, T, H)["identity"] == live["identity"]
    reopened = FomcStore(env.root, wall_clock=env.clock.wall)
    assert snapshot.replay(reopened, T, H) == live


def test_unresolved_read_replays_unresolved_fomc244_and_mode_fomc243(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.collector.step()
    env.clock.sleep(100)  # past the first verified response + 92 s, with nothing newer in H
    T, H = env.clock.true, env.store.horizon()
    unresolved = snapshot.events_as_of(env.store, T, H)
    assert unresolved["read_state"] == "FOMC_CAUSAL_VISIBILITY_UNRESOLVED" and "items" not in unresolved
    env.drive(300)
    assert snapshot.events_as_of(env.store, T, H)["identity"] == unresolved["identity"]
    assert snapshot.events_as_of(env.store, T, mode="RETROSPECTIVE_SOURCE")["read_state"] == "FOMC_NOT_ADMISSIBLE_IN_MODE"


def test_discovery_pending_never_shows_an_older_zero_fomc245(env):
    env.feed([])
    env.provider.routes[P1] = syn.page_response()
    env.drive(200)
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"

    def feed_crash(resp):
        if resp.body["surface"] == "feed":
            raise RuntimeError("feed processing crash")

    env.collector.processing_fault = feed_crash
    env.clock.sleep(60)
    assert env.collector.step()["status"] == "RESPONSE"  # feed record F; its run dies
    env.collector.submit_manifest(json.dumps({"version": 1, "urls": [syn.url(P1)]}).encode(), "operator")
    t_start = env.clock.true
    env.collector.step()  # F dies again, a verified backfill response lands, then the poison guard ends F
    env.collector.processing_fault = None
    env.drive(300)
    seen = []
    for second in range(0, 300, 5):
        snap = snapshot.events_as_of(env.store, t_start + timedelta(seconds=second))
        if snap["read_state"] == "FOMC_RESOLVED":
            seen.append(snap["discovery"]["state"])
    assert "DISCOVERY_PENDING" in seen
    after = seen[seen.index("DISCOVERY_PENDING"):]
    assert "EVENTS_OBSERVED_ZERO" not in after[: after.index("NOT_ZERO")]  # the older zero is never current
