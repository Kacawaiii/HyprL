"""The MVP acceptance check (scripts/trading_lab/mvp_check.py) on synthetic FOMC and EDGAR stores: it passes
on sound stores, fails on a corrupted raw or a wrong recorded identity, blocks (never passes) what it cannot
check, and leaves the store directories untouched. Offline; the server runs on the loopback that accepts
connections."""

from __future__ import annotations

from datetime import timedelta
import json
import shutil
import sqlite3

import pytest

from scripts.trading_lab import mvp_check
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.edgar import snapshot as edgar_snapshot
from scripts.trading_lab.edgar import synthetic as edgar_syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.fomc import canon
from scripts.trading_lab.fomc import synthetic as syn

from tests.crypto import loopback
from tests.crypto.fomc_support import P1, SID1, Env, statement_item


@pytest.fixture(scope="module")
def fomc_store(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setattr(canon, "CHALLENGE_SCRIPT_SHA256", syn.CF_SCRIPT_SHA256)
    env = Env(tmp_path_factory.mktemp("fomc"))
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=syn.url(P1))
    env.drive(4000, idle=60)
    env.drive(130)
    env.collector.close()
    env.store.close()
    env.provider.close()
    with sqlite3.connect(env.root / "fomc.sqlite3") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    yield env.root
    patch.undo()


@pytest.fixture(scope="module")
def edgar_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("edgar") / "store"
    clock = edgar_syn.SimClock()
    fetcher = edgar_syn.FakeFetcher(clock)
    store = EdgarStore(root, wall_clock=clock.wall)
    collector = EdgarCollector(store, fetcher, clock)
    collector.submit_watchlist([edgar_syn.CIK_A])
    fetcher.routes[edgar_syn.CIK_A.zfill(10)] = edgar_syn.Reply(
        edgar_syn.listing(edgar_syn.CIK_A, [edgar_syn.filing("0000320193-26-000071")]))
    for _ in range(3):
        collector.poll(edgar_syn.CIK_A)
        clock.sleep(600)
    collector.close()
    store.close()
    with sqlite3.connect(root / "edgar.sqlite3") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    (root / "owner.lock").unlink(missing_ok=True)
    return root


@pytest.fixture(autouse=True)
def _loopback(monkeypatch):
    monkeypatch.setattr(mvp_check, "loopback_host", loopback.host)


@pytest.fixture
def reads_file(fomc_store, tmp_path):
    """Recorded reads as the pilot writes them: T, H, identity, read state (one RESOLVED, one UNRESOLVED)."""
    views = AppService(tmp_path, fomc_store=fomc_store).fomc
    status = views.status()
    later = (mvp_check.datetime.fromisoformat(status["suggested_as_of"]) + timedelta(days=1)).isoformat()
    lines = []
    for as_of in (status["suggested_as_of"], later):
        header = views.snapshot(as_of=as_of)["snapshot"]
        lines.append({"T": header["T"], "H": header["H"], "identity": header["identity"],
                      "read_state": header["read_state"]})
    path = tmp_path / "snapshots.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _run(tmp_path, capsys, fomc, edgar, *extra):
    out = tmp_path / "report.json"
    code = mvp_check.main(["--fomc-store", str(fomc), "--edgar-store", str(edgar), "--output", str(out), *extra])
    capsys.readouterr()
    return code, json.loads(out.read_text())


def _statuses(report):
    return {key: value["status"] for key, value in report["criteria"].items()}


def test_sound_stores_pass_and_the_unrun_cockpit_blocks(fomc_store, edgar_store, reads_file, tmp_path, capsys):
    before = (mvp_check.fingerprint(fomc_store), mvp_check.fingerprint(edgar_store))
    code, report = _run(tmp_path, capsys, fomc_store, edgar_store, "--fomc-reads", str(reads_file))
    statuses = _statuses(report)
    assert {k: v for k, v in statuses.items() if v != "PASS"} == {
        f"COCKPIT-0{n}": "BLOCKED" for n in range(1, 5)}, report["criteria"]
    assert report["verdict"] == "BLOCKED" and code == 2  # a BLOCKED criterion is never a pass
    assert report["criteria"]["FOMC-08"]["evidence"]["reads"] == 2
    assert report["criteria"]["FOMC-08"]["evidence"]["resolved"] == 1
    assert (mvp_check.fingerprint(fomc_store), mvp_check.fingerprint(edgar_store)) == before


def test_without_a_reads_file_only_that_criterion_blocks(fomc_store, edgar_store, tmp_path, capsys):
    _code, report = _run(tmp_path, capsys, fomc_store, edgar_store)
    assert report["criteria"]["FOMC-08"]["status"] == "BLOCKED"
    assert report["criteria"]["FOMC-04"]["status"] == "PASS" and report["verdict"] == "BLOCKED"


def test_a_wrong_recorded_identity_fails(fomc_store, edgar_store, reads_file, tmp_path, capsys):
    lines = [json.loads(line) for line in reads_file.read_text().splitlines()]
    lines[0]["identity"] = "0" * 64
    reads_file.write_text("".join(json.dumps(line) + "\n" for line in lines))
    code, report = _run(tmp_path, capsys, fomc_store, edgar_store, "--fomc-reads", str(reads_file))
    assert report["criteria"]["FOMC-08"]["status"] == "FAIL" and report["verdict"] == "FAIL" and code == 1
    assert "identity differs" in report["criteria"]["FOMC-08"]["evidence"]["reason"]


def test_a_corrupted_raw_fails_closed_never_passes(fomc_store, edgar_store, tmp_path, capsys):
    corrupt = tmp_path / "corrupt"
    shutil.copytree(fomc_store, corrupt)
    for raw in (corrupt / "raw").rglob("*"):
        if raw.is_file():
            raw.write_bytes(raw.read_bytes()[:-1] + b"!")
    code, report = _run(tmp_path, capsys, corrupt, edgar_store)
    statuses = _statuses(report)
    assert statuses["FOMC-02"] == "FAIL" and report["verdict"] == "FAIL" and code == 1
    assert statuses["EDGAR-02"] == "PASS"  # the other source is judged on its own store


def test_a_missing_store_fails_and_its_dependents_block(edgar_store, fomc_store, tmp_path, capsys):
    _code, report = _run(tmp_path, capsys, tmp_path / "absent", edgar_store)
    statuses = _statuses(report)
    assert statuses["FOMC-00"] == "FAIL" and statuses["FOMC-01"] == "BLOCKED" and statuses["EDGAR-01"] == "PASS"


def test_an_unusable_loopback_blocks_everything_served(fomc_store, edgar_store, tmp_path, capsys, monkeypatch):
    def refused():
        raise mvp_check.Blocked("no loopback address accepts connections on this host")
    monkeypatch.setattr(mvp_check, "loopback_host", refused)
    _code, report = _run(tmp_path, capsys, fomc_store, edgar_store)
    statuses = _statuses(report)
    assert statuses["FOMC-01"] == statuses["EDGAR-02"] == statuses["FOMC-08"] == "BLOCKED"
    assert "PASS" not in {statuses[k] for k in statuses if k.endswith(("-01", "-02", "-03", "-04", "-05", "-06", "-07"))}
    assert report["verdict"] == "BLOCKED"


def test_the_cockpit_criteria_run_the_declared_commands(tmp_path, monkeypatch):
    web = tmp_path / "web"
    web.mkdir()
    report = mvp_check.Report()
    mvp_check.check_cockpit(report, web)  # no node_modules: BLOCKED with its cause
    assert {c["status"] for c in report.criteria.values()} == {"BLOCKED"}
    (web / "node_modules").mkdir()
    python = shutil.which("python3") or "python3"
    monkeypatch.setattr(mvp_check, "WEB_COMMANDS", (("COCKPIT-01", "ok", [python, "-c", "pass"]),
                                                    ("COCKPIT-02", "bad", [python, "-c", "raise SystemExit(3)"])))
    report = mvp_check.Report()
    mvp_check.check_cockpit(report, web)
    assert _statuses({"criteria": report.criteria}) == {"COCKPIT-01": "PASS", "COCKPIT-02": "FAIL"}
