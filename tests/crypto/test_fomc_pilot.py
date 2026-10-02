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
        assert verification["audit"]["fix15"]["status"] == "PROVEN"  # every grant journaled and compliant
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
        assert report["validation"]["verdict"] == "NOT_VALIDATED" and "duration ACCOMPLISHED" in report["validation"]["missing"]
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
        assert audit["fix15"]["status"] == "NOT_PROVEN" and audit["not_proven"]  # these attempts have no journaled grant
    finally:
        env.provider.close()


def test_fix15_over_redirect_hops_is_proven_from_the_journal(tmp_path):
    env = Env(tmp_path / "run")
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.SyntheticResponse(status=302, headers=[("Location", P1 + "x")])
        env.provider.routes[P1 + "x"] = syn.page_response()
        service = _service(env)
        service.run_for(300)
        service.stop(wait_s=10)
        fix15 = pilot.audit(env.store)["fix15"]
        assert fix15["status"] == "PROVEN" and fix15["continuations"] >= 1 and fix15["missing"] == []
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


class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def _fake_system(calls, *, previous="inactive"):
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return _Proc(previous)
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            return _Proc("fomc-pilot.service loaded active running x\n" if previous == "active" else "")
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return _Proc("4242")
        if cmd[0] == "timedatectl":
            return _Proc("yes")
        return _Proc()
    return run


def _previous(tmp_path, duration):
    run = tmp_path / "rev23"
    (run / "store").mkdir(parents=True, exist_ok=True)
    (run / "closure-report.json").write_text(json.dumps({"integrity": {"verdict": "VALID", "not_proven": ["FIX15"]},
                                                         "pilot_duration": {"verdict": duration}}))
    return run


def test_the_successor_is_launched_only_behind_its_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "owner_free", lambda store: True)
    monkeypatch.setattr(pilot.time, "sleep", lambda s: None)
    monkeypatch.setattr(pilot, "_alert", lambda *a, **k: None)
    soon = pilot._now() + timedelta(minutes=1)
    for duration, previous, launched in (("NOT_ACCOMPLISHED", "inactive", False), ("ACCOMPLISHED", "active", False),
                                         ("ACCOMPLISHED", "inactive", True)):
        calls = []
        monkeypatch.setattr(pilot.subprocess, "run", _fake_system(calls, previous=previous))
        base = tmp_path / f"base-{duration}-{previous}"
        base.mkdir()
        decision = pilot.successor(_previous(base, duration), "fomc-pilot", tmp_path, base, wait_until=soon, minutes=90,
                                   unit_dir=base)
        runs = [c for c in calls if c[0] == "systemd-run"]
        assert decision["launched"] is launched, decision
        written = json.loads((base / "successor-decision.json").read_text())
        assert written["decision"] == ("LAUNCHED" if launched else "NOT_LAUNCHED")
        if not launched:
            assert runs == [] and decision["reasons"]  # nothing started, the reasons are recorded
            continue
        assert len(runs) == 2 and "--unit=fomc-pilot-rev25" in runs[0] and "--unit=fomc-pilot-rev25-supervisor" in runs[1]
        store = runs[0][runs[0].index("--store") + 1]
        assert store.startswith(str(base / "run-rev25-")) and store != str(base / "rev23" / "store")  # a new store
        close_at = pilot.parse_iso(runs[1][runs[1].index("--close-at") + 1])
        assert close_at - pilot.parse_iso(decision["started"]) == timedelta(minutes=90)
        assert "--no-close" in runs[1]  # the persistent timer owns the closure
        timer = (base / "fomc-pilot-rev25-closure.timer").read_text()
        service = (base / "fomc-pilot-rev25-closure.service").read_text()
        fires = next(line for line in timer.splitlines() if line.startswith("OnCalendar="))
        fires = pilot.parse_iso(fires[len("OnCalendar="):-len(" UTC")].replace(" ", "T") + "+00:00")
        assert "Persistent=true" in timer and close_at <= fires < close_at + timedelta(seconds=1)  # never before the end
        assert f"WorkingDirectory={tmp_path}" in service and f"--launch-boot-id {decision['launch_boot_id']}" in service
        assert f"--planned-end {pilot.iso(close_at)}" in service and "--unit fomc-pilot-rev25 " in service
        assert ["systemctl", "--user", "enable", "--now", "fomc-pilot-rev25-closure.timer"] in calls


AUTHORIZATION = {"granted_at": "2026-10-02", "unit": "fomc-pilot-rev25", "waives": ["previous_duration"],
                 "scope": "this trial only", "text": "le pilote rev25 peut demarrer independamment de la duree accomplie par rev23"}


def test_an_explicit_authorization_waives_only_the_previous_duration_and_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "owner_free", lambda store: True)
    monkeypatch.setattr(pilot.time, "sleep", lambda s: None)
    monkeypatch.setattr(pilot, "_alert", lambda *a, **k: None)
    soon = pilot._now() + timedelta(minutes=1)

    def attempt(name, auth, *, integrity="VALID", previous="inactive"):
        calls = []
        monkeypatch.setattr(pilot.subprocess, "run", _fake_system(calls, previous=previous))
        base = tmp_path / name
        base.mkdir(exist_ok=True)
        run = _previous(base, "NOT_ACCOMPLISHED")
        report = json.loads((run / "closure-report.json").read_text())
        report["integrity"]["verdict"] = integrity
        (run / "closure-report.json").write_text(json.dumps(report))
        return pilot.successor(run, "fomc-pilot", tmp_path, base, wait_until=soon, authorization=auth, unit_dir=base), base

    decision, base = attempt("granted", AUTHORIZATION)
    assert decision["launched"] and decision["authorization"] == AUTHORIZATION
    assert decision["waived"] and "NOT_ACCOMPLISHED" in decision["waived"][0]
    assert json.loads((base / "successor-decision.json").read_text())["authorization"] == AUTHORIZATION
    again, _ = attempt("granted", AUTHORIZATION)  # the same authorization, a second trial
    assert not again["launched"] and any("one trial" in r for r in again["reasons"])
    for name, auth, kw in (("integrity", AUTHORIZATION, {"integrity": "INVALID"}),
                           ("running", AUTHORIZATION, {"previous": "active"}),
                           ("wider", dict(AUTHORIZATION, waives=["previous_duration", "previous_integrity"]), {}),
                           ("other-unit", dict(AUTHORIZATION, unit="fomc-pilot-rev26"), {})):
        refused, _ = attempt(name, auth, **kw)
        assert not refused["launched"] and refused["reasons"], name


def test_a_reboot_or_a_capture_gap_keeps_the_duration_not_accomplished():
    start = pilot.parse_iso("2026-10-02T13:30:00+00:00")
    end = start + timedelta(minutes=90)
    progress = {"first_wall": pilot.iso(start), "last_wall": pilot.iso(end), "epochs": 1, "max_gap_s": 75.0}
    ok = pilot.pilot_duration({"was": "active"}, progress, planned_start=start, planned_end=end, closed_at=end,
                              launch_boot_id="a" * 32, closure_boot_id="a" * 32)
    assert ok["verdict"] == "ACCOMPLISHED"
    rebooted = pilot.pilot_duration({"was": "active"}, progress, planned_start=start, planned_end=end, closed_at=end,
                                    launch_boot_id="a" * 32, closure_boot_id="b" * 32)
    assert rebooted["verdict"] == "NOT_ACCOMPLISHED" and any("rebooted" in r for r in rebooted["reasons"])
    gap = pilot.pilot_duration({"was": "active"}, dict(progress, max_gap_s=pilot.CAPTURE_GAP_BOUND_S + 1),
                               planned_start=start, planned_end=end, closed_at=end, launch_boot_id="a" * 32,
                               closure_boot_id="a" * 32)
    assert gap["verdict"] == "NOT_ACCOMPLISHED" and any("suspended" in r for r in gap["reasons"])


def test_a_short_pilot_rehearsal_meets_the_successor_criteria(tmp_path, monkeypatch):
    """The 90-minute pilot's criteria, offline, on statement pages with per-response Cloudflare bytes."""
    from scripts.trading_lab.fomc import canon
    monkeypatch.setattr(canon, "CHALLENGE_SCRIPT_SHA256", syn.CF_SCRIPT_SHA256)
    env = Env(tmp_path / "run")
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.cloudflare_route(page_url=syn.url(P1))
        service = _service(env)
        start = env.clock.wall()
        log = tmp_path / "snapshots.jsonl"
        for _ in range(6):
            service.run_for(900)
            pilot.take_snapshot(env.root, log, now=env.clock.wall())
        end = env.clock.wall()
        report = pilot.close(env.root, tmp_path / "copy", tmp_path / "report.json", snapshots_log=log, notify=False,
                             planned_start=start, planned_end=end,
                             stopper=lambda: {"was": "active", "left": service.stop(wait_s=10)})
        criteria = report["criteria"]
        assert report["integrity_ok"] and report["duration_ok"] and report["criteria_ok"], json.dumps(criteria, indent=1)
        assert report["validation"] == {"verdict": "VALIDATED", "missing": []}
        stable = criteria["identity_stable_over_different_rereads"]
        assert stable["items_reread_with_different_raws"] >= 1 and stable["ok"]
        item = next(iter(stable["per_item"].values()))
        assert item["raws"] >= 3 and item["identities"] == 1 and item["domains"] == {"CANONICAL": item["raws"]}
        assert criteria["fix15_journal_complete_and_compliant"]["status"] == "PROVEN"
        assert criteria["rechecks_300s"]["ok"] and criteria["rechecks_1h"]["ok"]
    finally:
        env.provider.close()
