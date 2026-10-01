"""Pilot tooling, offline: the manifest is submitted once by the owner, fixtures are exported with
their bytes, headers, digests and processed release fields, snapshot reads are recorded, and the
closure takes a consistent copy (WAL included) and verifies it without network, with separate
integrity and duration verdicts, overlaps checked per source item, FIX15 coverage stated, every
recorded read (resolved or not) re-read and replayed, and recheck status from durable records."""

from __future__ import annotations

from datetime import timedelta
import json

from scripts.trading_lab.fomc import identity, ledger, pilot, spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.service import FomcService, submit_manifest_once

from tests.crypto.fomc_support import P1, SID1, Env, statement_item

PAGES = {
    "summer_edt": (syn.statement_path("20260617"), dict(date_text="June 17, 2026", release="For release at 2:00 p.m. EDT")),
    "winter_est": (syn.statement_path("20260128"), dict(date_text="January 28, 2026", release="For release at 2:00 p.m. EST")),
    "immediate_release": (syn.statement_path("20150318"), dict(date_text="March 18, 2015", release="For immediate release")),
}


def _service(env):
    return FomcService(env.collector, env.clock, tick_s=5.0, settle_s=5.0, owner_wait_s=0.02, on_alert=lambda a: None)


def test_fixtures_snapshots_and_a_verified_closure(tmp_path):
    env = Env(tmp_path / "run")
    try:
        env.feed([])
        for path, kw in PAGES.values():
            env.provider.routes[path] = syn.page_response(**kw)
        manifest = json.dumps({"version": 1, "urls": [syn.url(p) for p, _ in PAGES.values()]}).encode()
        first = submit_manifest_once(env.collector, manifest, "test")
        again = submit_manifest_once(env.collector, manifest, "test")  # a restart re-submits nothing
        assert first["submitted"] and not again["submitted"] and again["seq"] == first["seq"]
        service = _service(env)
        start = env.clock.wall()
        log = tmp_path / "snapshots.jsonl"
        for _ in range(4):
            service.run_for(300)
            assert pilot.take_snapshot(env.root, log, now=env.clock.wall())["resolved"]
        pilot.take_snapshot(env.root, log, now=env.clock.wall() + timedelta(hours=2))  # unresolved reads, recorded too
        index = pilot.export_fixtures(env.root, tmp_path / "fixtures", {n: syn.url(p) for n, (p, _) in PAGES.items()})
        assert all(v["status"] == "ACQUIRED" and v["checks"]["ok"] for v in index.values()), index
        meta = json.loads((tmp_path / "fixtures" / "summer_edt" / "meta.json").read_text())
        assert meta["mode"] == "HISTORICAL_BACKFILL" and meta["date_lines"] and meta["content_type_lines"]
        assert (tmp_path / "fixtures" / "summer_edt" / "body.html").read_bytes() == syn.statement_html(**PAGES["summer_edt"][1])
        end = env.clock.wall()
        report = pilot.close(env.root, tmp_path / "copy", tmp_path / "report.json", snapshots_log=log, notify=False,
                             planned_start=start, planned_end=end,
                             stopper=lambda: {"was": "active", "left": service.stop(wait_s=10)})
        assert report["integrity_ok"] and report["duration_ok"], json.dumps(report, indent=1)[:3000]
        verification = report["verification"]
        assert verification["resolved_reads"] == 4 and verification["unresolved_reads"] == 4
        assert verification["reread_identical"] == verification["replay_identical"] == 8  # unresolved ones too
        assert report["copy_check"]["equal"] and report["copy_check"]["raw_files"] >= 3
        assert verification["audit"]["fix15"]["all_physical_starts"] == "PASS"  # every attempt single-hop by record
        assert verification["audit"]["not_proven"] == [] and verification["progress"]["manifests"] == 1
    finally:
        env.provider.close()


def test_a_premature_stop_keeps_integrity_valid_but_not_the_planned_duration(tmp_path):
    env = Env(tmp_path / "run")
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response()
        service = _service(env)
        start = env.clock.wall()
        service.run_for(600)
        service.stop(wait_s=10)  # the service stops long before its planned end
        log = tmp_path / "snapshots.jsonl"
        pilot.take_snapshot(env.root, log, now=env.clock.wall())
        report = pilot.close(env.root, tmp_path / "copy", tmp_path / "report.json", snapshots_log=log, notify=False,
                             planned_start=start, planned_end=start + timedelta(hours=24),
                             stopper=lambda: {"was": "inactive"})
        assert report["integrity_ok"] and not report["duration_ok"] and not report["ok"]
        reasons = report["pilot_duration"]["reasons"]
        assert any("not running" in r for r in reasons) and any("planned end" in r for r in reasons)
    finally:
        env.provider.close()


def _invoked(store, key, sid, *, kind, epoch, grant, mode="LIVE"):
    body = {"kind": kind, "class": "FEED_DISCOVERY", "sid": sid, "mode": mode, "episode_key": key, "epoch": epoch,
            "grant_mono": grant}
    return store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", key, body)])


def _outcome(store, attempt):
    return store.append("ATTEMPT_OUTCOME", [("ATTEMPT_OUTCOME", str(attempt), {"attempt": attempt, "outcome": "SOURCE_UNAVAILABLE"})])


def test_overlaps_are_checked_per_source_item_across_classes_with_the_feed_apart(tmp_path):
    env = Env(tmp_path / "run")
    try:
        store, epoch = env.store, env.collector.epoch
        a = _invoked(store, "LIVE-KEY", SID1, kind="LIVE_ACQUISITION", epoch=epoch, grant=100.0)
        poll = _invoked(store, ledger.FEED_KEY, None, kind="FEED_POLL", epoch=epoch, grant=110.0)  # the feed: apart
        b = _invoked(store, "BACKFILL-KEY", SID1, kind="HISTORICAL_BACKFILL", epoch=epoch, grant=120.0, mode="HISTORICAL_BACKFILL")
        for attempt in (a, poll, b):
            _outcome(store, attempt)
        audit = pilot.audit(store)
        assert audit["in_flight_overlaps"] == [{"group": f"item:{SID1}", "earlier": a, "later": b, "earlier_outcome_seq": a + 3}]
        assert not audit["checks"]["one_fetch_per_item_across_classes_and_one_feed_poll_in_flight"]
        assert audit["fix15"]["continuation_grants"]["status"] == "NOT_PROVEN"  # no hop record for these attempts
        assert audit["fix15"]["all_physical_starts"] == "NOT_PROVEN" and audit["not_proven"]
    finally:
        env.provider.close()


def test_fix15_over_redirect_hops_is_not_proven_never_passed(tmp_path):
    env = Env(tmp_path / "run")
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "x")])
        env.provider.routes[P1 + "x"] = syn.page_response()
        service = _service(env)
        service.run_for(300)
        service.stop(wait_s=10)
        fix15 = pilot.audit(env.store)["fix15"]
        assert fix15["initial_grants"]["status"] == "PASS"
        assert fix15["continuation_grants"] == {"status": "NOT_PROVEN", "multi_hop_responses": 1, "attempts_without_hop_record": 0}
        assert fix15["all_physical_starts"] == "NOT_PROVEN"
    finally:
        env.provider.close()


def test_recheck_status_comes_from_durable_records(tmp_path):
    env = Env(tmp_path / "run")
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response()
        service = _service(env)
        service.run_for(900)  # anchored; O300 served
        del env.provider.routes[P1]  # the +1 h recheck will fail
        service.run_for(3 * 3600)
        service.stop(wait_s=10)
        status = pilot.recheck_status(env.store.view())
        rows = {o["offset"]: o for o in status["items"][SID1]["obligations"]}
        assert rows[300]["status"] == "SATISFIED"
        assert rows[3600]["status"] == "IN_PROGRESS" and rows[3600]["attempts"] >= 2  # failing, retried, not yet suspended
        assert rows[86400]["status"] == "NOT_YET_DUE" and rows[604800]["status"] == "NOT_YET_DUE"
        assert status["by_offset"]["300"] == {"SATISFIED": 1}
        key = identity.reobservation_episode_key(SID1, status["items"][SID1]["anchor"], 3600)
        assert ledger.episode_status(env.store, key) == "OPEN"
        assert spec.ATTEMPTS_PER_EPISODE > rows[3600]["attempts"]
    finally:
        env.provider.close()
