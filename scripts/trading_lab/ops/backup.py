"""Offline stopped-runtime backups. Restore into a fresh runtime, never overwrite evidence."""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3

from scripts.trading_lab.ops.control import OpsRefused, SERVICES, WORKTREE, inspect, local_path, read_json, save
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.store import read_only_connection

MAX_BYTES = 512 * 1024**2
MAX_FILES = 10000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise OpsRefused("BACKUP_SYMLINK_REFUSED")
        if not path.is_file() or path.name == "backup-manifest.json":
            continue
        total += path.stat().st_size
        if len(files) >= MAX_FILES or total > MAX_BYTES:
            raise OpsRefused("BACKUP_RESOURCE_BUDGET_EXCEEDED")
        files.append({"name": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    return files


def stopped(config):
    root = config["runtime_root"]
    if any(inspect(root, name)["state"] in ("RUNNING", "FOREIGN") for name in SERVICES):
        raise OpsRefused("STOP_SERVICES_BEFORE_BACKUP_OR_RESTORE")


def _sqlite(source, destination):
    def validate(db):
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise OpsRefused("BACKUP_SQLITE_INTEGRITY_FAILED")
    with read_only_connection(source, validator=validate) as db:
        target = sqlite3.connect(destination)
        try:
            db.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")
        finally:
            target.close()


def backup(config, target):
    import fcntl
    from scripts.trading_lab.edgar.store import EdgarStore
    root = config["runtime_root"]
    stopped(config)
    target = local_path(target)
    if target.exists() or target.is_relative_to(root) or root.is_relative_to(target):
        raise OpsRefused("BACKUP_REQUIRES_NEW_SEPARATE_TARGET")
    with ExitStack() as stack:
        # Fence a runner started outside this controller while copying lab outputs.
        for name in ("runner.lock", "execution.lock"):
            lock = root / "lab" / name
            if lock.exists():
                handle = stack.enter_context(lock.open("rb"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise OpsRefused("WORKER_OWNER_BUSY_BACKUP_BLOCKED") from None
        target.mkdir(parents=True, mode=0o700)
        lab = root / "lab"
        if lab.exists():
            # Inventory first refuses symlink traversal and oversized backups.
            inventory(lab)
            for source in sorted(lab.rglob("*")):
                relative = source.relative_to(lab)
                if source.is_dir() or source.name.endswith(("-wal", "-shm", ".lock")):
                    continue
                if source.suffix not in (".sqlite", ".json"):
                    raise OpsRefused("UNRECOGNIZED_RUNTIME_FILE")
                destination = target / "lab" / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if source.suffix == ".sqlite":
                    _sqlite(source, destination)
                else:
                    shutil.copyfile(source, destination)
                os.chmod(destination, 0o600)
        if (root / "edgar" / EdgarStore.DB_NAME).exists():
            from scripts.trading_lab.edgar.closure import _owner_guard, consistent_copy, verify_copy
            with _owner_guard(root / "edgar"):
                copied = consistent_copy(root / "edgar", target / "edgar-evidence")
            verified = verify_copy(target / "edgar-evidence")
            if not copied["equal"] or not verified["ok"]:
                raise OpsRefused("EDGAR_BACKUP_VERIFICATION_FAILED")
        files = inventory(target)
        manifest = {"schema": "hyprl-runtime-backup-v1", "files": files,
                    "external_archives": "excluded; remain read-only at original locations",
                    "private_configuration": "excluded", "capture_restore": "evidence only; no budget rewind"}
        save(target / "backup-manifest.json", dict(manifest, identity=sha256_canonical(manifest)))
    return {"state": "COMPLETE", "backup_identity": sha256_canonical(manifest), "files": len(files),
            "bytes": sum(f["bytes"] for f in files), "verified": True}


def restore(config, target):
    root = config["runtime_root"]
    stopped(config)
    if any(p.name not in ("control.lock", "operations.json") for p in root.iterdir()):
        raise OpsRefused("RESTORE_REQUIRES_FRESH_RUNTIME")
    target = local_path(target)
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise OpsRefused("RESTORE_SOURCE_MUST_BE_SEPARATE")
    data = read_json(target / "backup-manifest.json")
    identity = data.pop("identity")
    if data.get("schema") != "hyprl-runtime-backup-v1" or sha256_canonical(data) != identity:
        raise OpsRefused("BACKUP_MANIFEST_INTEGRITY_FAILED")
    files = inventory(target)
    if files != data["files"]:
        raise OpsRefused("BACKUP_FILE_INTEGRITY_FAILED")
    for row in files:
        relative = Path(row["name"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("lab", "edgar-evidence"):
            raise OpsRefused("BACKUP_FILE_SCOPE_REFUSED")
    # Everything is verified before the first restore write.
    for row in files:
        source = target / row["name"]
        destination = root / row["name"]
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
    return {"state": "COMPLETE", "backup_identity": identity, "files": len(files),
            "capture_resumed": False, "next_action": "resume app/workers; EDGAR copy is evidence only"}
