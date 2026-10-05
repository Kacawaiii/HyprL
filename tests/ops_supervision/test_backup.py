import shutil

import pytest

from scripts.trading_lab.ops.control import OpsRefused, operate, read_json, save
from scripts.trading_lab.ops.backup import backup, restore
from scripts.trading_lab.edgar.closure import _tree_digest
from scripts.trading_lab.platform.jobs import JobStore


def test_stopped_backup_restores_jobs_without_private_config_logs_or_archive(config, backup_target):
    root = config["runtime_root"]
    store = JobStore(root / "lab")
    job = store.submit("dataset", {"products": ["BTC-USD"], "bars": 120})
    (root / "app.log").write_text("synthetic private log")
    before = _tree_digest(root / "lab")
    proof = operate(config, "backup", target=backup_target)
    assert proof["verified"] and proof["files"] == 1
    assert _tree_digest(root / "lab") == before
    assert not (backup_target / "app.log").exists()
    with pytest.raises(OpsRefused, match="FRESH_RUNTIME"):
        operate(config, "restore", target=backup_target)
    # Preserve the original; restore into a distinct fresh runtime configuration.
    new_root = root / "restored"
    fresh = dict(config, runtime_root=new_root)
    result = operate(fresh, "restore", target=backup_target)
    assert result["backup_identity"] == proof["backup_identity"]
    assert JobStore(new_root / "lab").status(job)["state"] == "QUEUED"
    assert not result["capture_resumed"]


def test_running_workers_fence_backup(config, backup_target):
    operate(config, "start", "workers")
    with pytest.raises(OpsRefused, match="STOP_SERVICES"):
        operate(config, "backup", target=backup_target)
    assert not backup_target.exists()


def test_backups_reject_symlinks_and_non_runtime_files(config, backup_target, tmp_path):
    root = config["runtime_root"]
    (root / "lab").mkdir(parents=True)
    (root / "lab" / "secret.json").symlink_to(config["config_path"])
    with pytest.raises(OpsRefused, match="SYMLINK"):
        operate(config, "backup", target=backup_target)
    assert not (backup_target / "lab" / "secret.json").exists()


def test_corruption_is_refused_before_restore_writes(config, backup_target):
    root = config["runtime_root"]
    JobStore(root / "lab").submit("dataset", {})
    operate(config, "backup", target=backup_target)
    (backup_target / "lab" / "jobs.sqlite").write_bytes(b"synthetic corruption")
    fresh = dict(config, runtime_root=root / "fresh")
    with pytest.raises(OpsRefused, match="FILE_INTEGRITY"):
        operate(fresh, "restore", target=backup_target)
    assert not (fresh["runtime_root"] / "lab").exists()


def test_unlisted_extra_file_is_refused(config, backup_target):
    root = config["runtime_root"]
    JobStore(root / "lab")
    operate(config, "backup", target=backup_target)
    (backup_target / "extra.json").write_text("{}")
    fresh = dict(config, runtime_root=root / "fresh")
    with pytest.raises(OpsRefused, match="FILE_INTEGRITY"):
        operate(fresh, "restore", target=backup_target)


def test_active_independent_job_runner_fences_backup(config, backup_target):
    from scripts.trading_lab.platform.jobs import JobRunner
    with JobRunner(config["runtime_root"] / "lab"):
        with pytest.raises(OpsRefused, match="WORKER_OWNER_BUSY"):
            operate(config, "backup", target=backup_target)
    assert not backup_target.exists()


def test_edgar_backup_replays_real_shaped_synthetic_raws_and_restore_preserves_budget_boundary(config, backup_target):
    from scripts.trading_lab.edgar import synthetic as syn
    from scripts.trading_lab.edgar.collector import EdgarCollector
    from scripts.trading_lab.edgar.store import EdgarStore
    from scripts.trading_lab.edgar.closure import verify_copy
    root = config["runtime_root"]
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[syn.CIK_A.zfill(10)] = syn.Reply(syn.listing(syn.CIK_A, [syn.filing("0000320193-26-000071")]))
    source = EdgarStore(root / "edgar", wall_clock=clock.wall)
    collector = EdgarCollector(source, fetcher, clock)
    try:
        collector.submit_watchlist([syn.CIK_A])
        for _ in range(3):
            collector.poll(syn.CIK_A)
            clock.sleep(600)
    finally:
        collector.close()
        source.close()
    before = _tree_digest(root / "edgar")
    result = operate(config, "backup", target=backup_target)
    assert result["verified"]
    assert verify_copy(backup_target / "edgar-evidence")["ok"]
    assert _tree_digest(root / "edgar") == before
    fresh = dict(config, runtime_root=root / "fresh")
    operate(fresh, "restore", target=backup_target)
    assert verify_copy(fresh["runtime_root"] / "edgar-evidence")["ok"]
    assert not (fresh["runtime_root"] / "edgar").exists()


def test_corrupt_edgar_raw_never_produces_a_verified_backup(config, backup_target):
    from scripts.trading_lab.edgar import synthetic as syn
    from scripts.trading_lab.edgar.collector import EdgarCollector
    from scripts.trading_lab.edgar.store import EdgarStore
    root = config["runtime_root"]
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[syn.CIK_A.zfill(10)] = syn.Reply(syn.listing(syn.CIK_A, []))
    source = EdgarStore(root / "edgar", wall_clock=clock.wall)
    collector = EdgarCollector(source, fetcher, clock)
    try:
        collector.submit_watchlist([syn.CIK_A])
        collector.poll(syn.CIK_A)
        raw = source.rows("RESPONSE")[0].body["raw_sha"]
    finally:
        collector.close()
        source.close()
    # Deliberate corruption of this test's new synthetic store, never an archive.
    raw_file = next((root / "edgar" / "raw").rglob(raw + "*"))
    raw_file.chmod(0o600)
    raw_file.write_bytes(b"synthetic corrupt body")
    with pytest.raises(OpsRefused, match="VERIFICATION_FAILED"):
        operate(config, "backup", target=backup_target)
    assert not (backup_target / "backup-manifest.json").exists()
