"""Auditable export of a shadow session, and paranoid import of one.

Two hazards, handled explicitly.

**Copying a live SQLite file is not a backup.** With WAL enabled, the ``.sqlite``
file alone is a torn snapshot: committed transactions live in ``-wal`` until a
checkpoint, so a plain file copy can produce a database that opens fine and is
missing the last hour. The export therefore goes through SQLite's own backup
API, which takes a consistent copy of a database that is still being written.

**An archive is untrusted input, even one we wrote.** Extraction is where
``../../.ssh/authorized_keys`` gets written by a tool that assumed its own
output. Every member is validated before anything is written: no absolute
paths, no parent traversal, no symlinks, no device files, no duplicate names,
and a total size cap so a zip bomb cannot fill a disk.

Import never targets the live runtime. Merging an exported session into a
running one would mean reconciling two hash chains, and there is no honest way
to do that -- the result would verify while describing a history that never
happened. Import writes to a separate directory and refuses to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import zipfile
from datetime import datetime, timezone

EXPORT_SCHEMA_VERSION = "trading-lab.runtime-export.v1"
MANIFEST_NAME = "manifest.json"
CHECKSUM_NAME = "SHA256SUMS"
DATABASE_NAME = "paper_v1.sqlite"
PORTFOLIO_DATABASE_NAME = "paper_portfolio_v1.sqlite"

# A local audit export of hourly candles is a few megabytes. The cap is three
# orders of magnitude above that: generous for real data, fatal to a bomb.
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 10_000

# Fixed timestamp for every archive member. Zip stores mtimes, and a real one
# would make two exports of identical content differ byte for byte.
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class ExportError(RuntimeError):
    """Raised when an export cannot be produced or trusted."""


class UnsafeArchiveError(ExportError):
    """Raised when an archive member would write outside its destination."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def consistent_database_copy(source, destination) -> pathlib.Path:
    """Copy a possibly-live SQLite database using the backup API."""
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    if not source.is_file():
        raise ExportError(f"no database at {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    origin = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30)
    try:
        target = sqlite3.connect(destination, timeout=30)
        try:
            origin.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")   # a copy needs no WAL
        finally:
            target.close()
    finally:
        origin.close()
    return destination


def _portfolio_metadata(database: pathlib.Path) -> dict:
    """The shared portfolio log's own chain, verified rather than assumed."""
    from scripts.trading_lab.paper_portfolio_store import PaperPortfolioStore

    store = PaperPortfolioStore(database)
    sessions = store.sessions()
    payload = {"store_type": "shared_portfolio",
               "sessions": list(sessions), "session_count": len(sessions)}
    if not sessions:
        return payload
    latest = sessions[-1]
    chain = store.verify_chain(session_id=latest)
    session = store.session(latest) or {}
    payload.update({
        "latest_session_id": latest,
        "session_spec_hash": session.get("session_spec_hash"),
        "events": chain.get("events"),
        "event_chain_tip": chain.get("head_hash"),
        "event_chain_verified": bool(chain.get("verified")),
        "snapshots": store.snapshot_count(session_id=latest),
    })
    return payload


def _session_metadata(database: pathlib.Path) -> dict:
    from scripts.trading_lab.paper_event_store import PaperEventStore

    store = PaperEventStore(database)
    sessions = store.sessions()
    payload = {"sessions": list(sessions), "session_count": len(sessions)}
    if not sessions:
        return payload
    latest = sessions[-1]
    chain = store.verify_chain(session_id=latest)
    payload.update({
        "latest_session_id": latest,
        "events": chain.get("events"),
        "event_chain_tip": chain.get("head_hash"),
        "event_chain_verified": bool(chain.get("verified")),
    })
    return payload


def _spec_hashes() -> dict:
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
    from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    return {
        "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
        "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
        "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
        "execution_spec_hash": EXECUTION_SPEC_V1.execution_spec_hash,
        "paper_execution_spec_hash":
            PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash,
        "paper_model_spec_hash": PAPER_MODEL_SPEC_V1.paper_model_spec_hash,
        "holdout_hash": PROTECTED_WINDOW_V1.holdout_hash,
        "holdout_observed": PROTECTED_WINDOW_V1.observed,
    }


def _model_hashes(model_dir) -> dict:
    from scripts.trading_lab.paper_model import read_artifact

    model_dir = pathlib.Path(model_dir)
    hashes = {}
    if not model_dir.is_dir():
        return hashes
    for path in sorted(model_dir.glob("*.json")):
        try:
            artifact = read_artifact(path)
        except Exception:                            # pragma: no cover
            continue
        hashes[path.stem] = artifact.get("fitted_hash")
    return hashes


def build_manifest(*, database: pathlib.Path, layout, model_dir,
                   include_logs: bool = False) -> dict:
    """The identity of an export.

    ``created_at`` is deliberately *outside* the hashed content block: two
    exports of the same runtime state should have the same content hash, and
    a wall clock in the identity would make that impossible.
    """
    from scripts.trading_lab.ops.runtime_paths import RUNTIME_SCHEMA_VERSION
    from scripts.trading_lab.paper_event_store import PAPER_EVENT_SCHEMA_VERSION

    content = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "paper_event_schema_version": PAPER_EVENT_SCHEMA_VERSION,
        "database": DATABASE_NAME,
        "database_sha256": _sha256_file(database),
        "database_bytes": database.stat().st_size,
        "session": _session_metadata(database),
        "specs": _spec_hashes(),
        "store_type": "legacy_individual_accounts",
        "models": _model_hashes(model_dir),
        "includes_logs": bool(include_logs),
        "real_money": False,
        "broker_connected": False,
    }
    return {
        "content": content,
        "content_hash": hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _write_archive(destination: pathlib.Path, members) -> pathlib.Path:
    """Write a zip with fixed metadata so identical content gives identical bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members, key=lambda item: item[0]):
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            data = payload.read_bytes() if isinstance(payload, pathlib.Path) else payload
            archive.writestr(info, data)
    return destination


def export_runtime(*, layout, destination, model_dir="data/models/paper_v1",
                   include_logs: bool = False) -> dict:
    """Produce an auditable archive of the current shadow runtime."""
    destination = pathlib.Path(destination)
    if destination.exists():
        raise ExportError(f"{destination.name} already exists; refusing to overwrite")
    if not layout.paper_database.is_file():
        raise ExportError("there is no shadow runtime to export yet")

    layout.ensure()
    staging = layout.tmp / f"export-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    staging.mkdir(parents=True, exist_ok=True)
    copy = staging / DATABASE_NAME
    try:
        consistent_database_copy(layout.paper_database, copy)
        members = [(DATABASE_NAME, copy)]
        portfolio_source = layout.paper_portfolio_database
        portfolio_meta = None
        if portfolio_source.is_file():
            portfolio_copy = staging / PORTFOLIO_DATABASE_NAME
            consistent_database_copy(portfolio_source, portfolio_copy)
            portfolio_meta = _portfolio_metadata(portfolio_copy)
            members.append((PORTFOLIO_DATABASE_NAME, portfolio_copy))
        manifest = build_manifest(database=copy, layout=layout,
                                  model_dir=model_dir, include_logs=include_logs)
        if portfolio_meta is not None:
            manifest["content"]["portfolio_session"] = portfolio_meta
            manifest["content"]["portfolio_database"] = PORTFOLIO_DATABASE_NAME
            manifest["content"]["portfolio_database_sha256"] = _sha256_file(
                portfolio_copy)
            manifest["content_hash"] = hashlib.sha256(
                _canonical(manifest["content"]).encode("utf-8")).hexdigest()
        members += [(MANIFEST_NAME,
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                    .encode("utf-8"))]
        if include_logs:
            for path in sorted(layout.logs.glob("*.jsonl*")):
                members.append((f"logs/{path.name}", path))
        checksums = "".join(
            f"{hashlib.sha256(payload.read_bytes() if isinstance(payload, pathlib.Path) else payload).hexdigest()}  {name}\n"
            for name, payload in sorted(members, key=lambda item: item[0]))
        members.append((CHECKSUM_NAME, checksums.encode("utf-8")))
        _write_archive(destination, members)
    finally:
        for path in sorted(staging.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        staging.rmdir()
    return {
        "archive": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "manifest": manifest,
    }


# --- reading an archive back ----------------------------------------------


def _validate_members(archive: zipfile.ZipFile) -> tuple:
    """Refuse anything that could write outside a destination directory."""
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise UnsafeArchiveError(f"archive has more than {MAX_MEMBERS} members")
    total = 0
    seen = set()
    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        if name in seen:
            raise UnsafeArchiveError(f"duplicate archive member {name!r}")
        seen.add(name)
        if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
            raise UnsafeArchiveError(f"absolute path in archive: {name!r}")
        if "\\" in name:
            raise UnsafeArchiveError(f"backslash in archive member: {name!r}")
        parts = pathlib.PurePosixPath(name).parts
        if ".." in parts:
            raise UnsafeArchiveError(f"parent traversal in archive: {name!r}")
        # Zip stores the unix mode in the top 16 bits; 0o120000 is a symlink.
        mode = info.external_attr >> 16
        if mode and (mode & 0o170000) == 0o120000:
            raise UnsafeArchiveError(f"symlink in archive: {name!r}")
        if mode and (mode & 0o170000) not in (0, 0o100000, 0o040000):
            raise UnsafeArchiveError(f"non-regular file in archive: {name!r}")
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise UnsafeArchiveError("archive expands beyond the size cap")
    return tuple(sorted(seen))


def verify_export(path) -> dict:
    """Validate an archive without extracting it into anything that matters."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise ExportError(f"no archive at {path.name}")
    report = {"archive": path.name, "checks": {}, "ok": False}
    with zipfile.ZipFile(path) as archive:
        names = _validate_members(archive)
        report["checks"]["archive_members_safe"] = True
        report["members"] = list(names)
        if MANIFEST_NAME not in names:
            raise ExportError("archive has no manifest")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        content = manifest.get("content", {})
        recomputed = hashlib.sha256(
            _canonical(content).encode("utf-8")).hexdigest()
        report["checks"]["manifest_hash"] = recomputed == manifest.get("content_hash")
        report["checks"]["schema"] = (
            content.get("export_schema_version") == EXPORT_SCHEMA_VERSION)

        digests = {}
        malformed = []
        if CHECKSUM_NAME in names:
            from scripts.trading_lab.identity import (
                IdentityError, require_exact_digest)

            for line in archive.read(CHECKSUM_NAME).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                if "  " not in line:
                    malformed.append(line[:80])
                    continue
                digest, name = line.split("  ", 1)
                # The digest is validated, not repaired. Stripping it would
                # accept a checksum file the writer never produced, and a
                # checksum that tolerates its own corruption checks nothing.
                try:
                    digests[name.strip()] = require_exact_digest(
                        digest, field=f"{CHECKSUM_NAME} entry")
                except IdentityError:
                    malformed.append(line[:80])
        report["checks"]["checksum_file_wellformed"] = not malformed
        if malformed:
            report["malformed_checksum_lines"] = malformed
        mismatched = [name for name in names
                      if name != CHECKSUM_NAME and name in digests
                      and hashlib.sha256(archive.read(name)).hexdigest()
                      != digests[name]]
        report["checks"]["sha256"] = not mismatched
        if mismatched:
            report["mismatched"] = mismatched
        report["checks"]["database_present"] = content.get("database") in names
        report["checks"]["event_chain_verified"] = bool(
            content.get("session", {}).get("event_chain_verified"))
        report["specs"] = content.get("specs", {})
        report["models"] = content.get("models", {})
        report["session"] = content.get("session", {})
    report["ok"] = all(report["checks"].values())
    return report


def import_export(path, *, destination) -> dict:
    """Extract an archive into an offline directory. Never the live runtime."""
    from scripts.trading_lab.ops.runtime_paths import DEFAULT_RUNTIME_ROOT

    path = pathlib.Path(path)
    destination = pathlib.Path(destination).resolve()
    live = pathlib.Path(DEFAULT_RUNTIME_ROOT).resolve()
    if destination == live or live in destination.parents \
            or destination in live.parents:
        raise ExportError(
            "refusing to import into the live runtime directory; an imported "
            "session is evidence to inspect, not state to resume. Choose a "
            "destination outside var/trading_lab.")
    if destination.exists() and any(destination.iterdir()):
        raise ExportError(f"{destination.name} is not empty; refusing to overwrite")

    report = verify_export(path)
    if not report["ok"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise ExportError(f"archive failed verification: {failed}")

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        names = _validate_members(archive)
        for name in names:
            target = (destination / name).resolve()
            # Re-check after resolution: the member names passed validation,
            # this proves the write actually lands inside the destination.
            try:
                target.relative_to(destination)
            except ValueError:
                raise UnsafeArchiveError(f"{name!r} escapes the destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            target.chmod(0o600)
    report["imported_to"] = destination.name
    report["files"] = len(report["members"])
    return report
