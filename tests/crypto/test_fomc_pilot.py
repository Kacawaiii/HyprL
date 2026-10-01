"""Pilot tooling, offline: the manifest is submitted once by the owner, fixtures are exported with
their bytes, headers, digests and processed release fields, snapshot reads are recorded, and the
closure takes a consistent copy (WAL included) and verifies it without network."""

from __future__ import annotations

import json

from scripts.trading_lab.fomc import pilot
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.service import FomcService, submit_manifest_once

from tests.crypto.fomc_support import Env

PAGES = {
    "summer_edt": (syn.statement_path("20260617"), dict(date_text="June 17, 2026", release="For release at 2:00 p.m. EDT")),
    "winter_est": (syn.statement_path("20260128"), dict(date_text="January 28, 2026", release="For release at 2:00 p.m. EST")),
    "immediate_release": (syn.statement_path("20150318"), dict(date_text="March 18, 2015", release="For immediate release")),
}


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
        service = FomcService(env.collector, env.clock, tick_s=5.0, settle_s=5.0, owner_wait_s=0.02, on_alert=lambda a: None)
        log = tmp_path / "snapshots.jsonl"
        for _ in range(4):
            service.run_for(300)
            assert pilot.take_snapshot(env.root, log, now=env.clock.wall())["resolved"]
        index = pilot.export_fixtures(env.root, tmp_path / "fixtures", {n: syn.url(p) for n, (p, _) in PAGES.items()})
        assert all(v["status"] == "ACQUIRED" and v["checks"]["ok"] for v in index.values()), index
        meta = json.loads((tmp_path / "fixtures" / "summer_edt" / "meta.json").read_text())
        assert meta["mode"] == "HISTORICAL_BACKFILL" and meta["date_lines"] and meta["content_type_lines"]
        assert (tmp_path / "fixtures" / "summer_edt" / "body.html").read_bytes() == syn.statement_html(**PAGES["summer_edt"][1])
        assert service.stop(wait_s=10) == []
        report = pilot.close(env.root, tmp_path / "copy", tmp_path / "report.json", snapshots_log=log, notify=False)
        assert report["ok"], json.dumps(report, indent=1)[:3000]
        verification = report["verification"]
        assert verification["resolved_reads"] == verification["reread_identical"] == verification["replay_identical"] == 4
        assert report["copy_check"]["equal"] and report["copy_check"]["raw_files"] >= 3
        assert verification["audit"]["ok"] and verification["progress"]["manifests"] == 1
    finally:
        env.provider.close()
