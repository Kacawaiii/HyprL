"""A local release bundle: everything needed to run, nothing else.

The rule that decides what goes in is "does the application read this at
runtime". Not "is it part of the project". A bundle that quietly ships the
git history, the test suite and every intermediate artefact is how a 300 kB
application becomes a 400 MB download that nobody can review.

What that rule includes and why:

* the built frontend -- the app;
* the Python that serves it, source only, no bytecode caches;
* the frozen shadow model artefacts -- the paper engine refuses to start
  without them;
* the research corpus and committed results -- Markets, Research and
  Backtests read these directly. They are the largest thing here by an order
  of magnitude, so they are measured, reported, and can be left out with
  ``--without-research-data`` for someone who only wants the runtime.

What it excludes: ``.git``, tests, ``node_modules``, frontend sources,
scratch directories and the runtime directory itself.

**The manifest identity is content, not time.** ``generated_at`` sits outside
the hashed block, so two builds of the same commit produce the same
``content_hash``. A wall clock inside the identity would make reproducibility
impossible to even state, let alone check.

There is no auto-updater and no `curl | bash`. The version is the commit and
the manifest. An updater that can silently replace the code that decides
trades is a supply chain, and it would need signing before it deserves to
exist.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from datetime import datetime, timezone

RELEASE_SCHEMA_VERSION = "trading-lab.local-release.v1"

DEFAULT_OUTPUT = pathlib.Path("dist/hyprl-local")
MANIFEST_NAME = "manifest.json"
CHECKSUM_NAME = "SHA256SUMS"
README_NAME = "README-RUN.txt"
LAUNCHER_NAME = "hyprl-run.sh"

# Runtime Python, source only.
PYTHON_SOURCES = ("scripts/trading_lab",)
# Read at runtime by the cockpit's research views.
RESEARCH_DATA = ("data/crypto",)
# The shadow engine refuses to start without these.
MODEL_DATA = ("data/models/paper_v1",)

EXCLUDED_DIRECTORIES = {"__pycache__", ".git", "node_modules", ".pytest_cache",
                        ".mypy_cache", ".ruff_cache", "var", "tests"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".sqlite", ".sqlite3"}


class ReleaseError(RuntimeError):
    """Raised when a release cannot be assembled."""


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(path: pathlib.Path) -> bool:
    if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
        return False
    return path.suffix not in EXCLUDED_SUFFIXES


def _copy_tree(source: pathlib.Path, destination: pathlib.Path) -> list:
    copied = []
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        if not _included(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied.append(target)
    return copied


def git_commit(root: pathlib.Path):
    """The commit, read from .git metadata. Worktree-aware."""
    from scripts.trading_lab.ops.git_identity import head_commit

    return head_commit(root)


def _spec_hashes() -> dict:
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    specs = {
        "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
        "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
        "execution_spec_hash": EXECUTION_SPEC_V1.execution_spec_hash,
        "holdout_hash": PROTECTED_WINDOW_V1.holdout_hash,
    }
    try:
        from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
        from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
    except ImportError:                              # a core install can still build
        return specs
    specs["paper_execution_spec_hash"] = \
        PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash
    specs["paper_model_spec_hash"] = PAPER_MODEL_SPEC_V1.paper_model_spec_hash
    return specs


def _model_hashes(root: pathlib.Path) -> dict:
    """Read the frozen artefacts' recorded hashes without loading a model."""
    hashes = {}
    directory = root / "data/models/paper_v1"
    if not directory.is_dir():
        return hashes
    for path in sorted(directory.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):                # pragma: no cover
            continue
        hashes[path.stem] = payload.get("fitted_hash")
    return hashes


def build_manifest(*, root: pathlib.Path, files, frontend=None,
                   includes_research_data: bool = True) -> dict:
    from scripts.trading_lab.app_api.contracts import APP_API_VERSION
    from scripts.trading_lab.ops.runtime_paths import RUNTIME_SCHEMA_VERSION
    from scripts.trading_lab.paper_event_store import PAPER_EVENT_SCHEMA_VERSION
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

    entries = [{"path": name, "sha256": digest, "bytes": size}
               for name, digest, size in files]
    content = {
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "git_commit": git_commit(root),
        "api_protocol": APP_API_VERSION,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "paper_event_schema_version": PAPER_EVENT_SCHEMA_VERSION,
        "frontend_build": frontend or {},
        "specs": _spec_hashes(),
        "models": _model_hashes(root),
        "research_protection": {
            "holdout_id": PROTECTED_WINDOW_V1.holdout_id,
            "start": PROTECTED_WINDOW_V1.start,
            "end": PROTECTED_WINDOW_V1.end,
            "observed": PROTECTED_WINDOW_V1.observed,
            "spent": False,
            "enforced": True,
        },
        "trading_safety": {
            "real_money": False,
            "broker_connected": False,
            "live_trading": False,
            "shadow_mode": True,
        },
        "includes_research_data": bool(includes_research_data),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    return {
        "content": content,
        # Identity is content. A timestamp here would make two builds of the
        # same commit differ, and reproducibility unstatable.
        "content_hash": hashlib.sha256(_canonical(content).encode()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


LAUNCHER = """#!/usr/bin/env bash
# HyprL local release. No real money, no broker, no exchange API key.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m scripts.trading_lab.app_api.server \\
  --data-root "$PWD/data/crypto" \\
  --host 127.0.0.1 --port "${HYPRL_PORT:-8787}" \\
  --dist-root "$PWD/frontend" \\
  --marker hyprl-local-app "$@"
"""

README = """HyprL — local release
=====================

Run it:

    ./hyprl-run.sh

then open http://127.0.0.1:8787/

The application binds 127.0.0.1 and serves the cockpit and its API from that
one origin. It never listens on a public interface by default.

What this is
------------
A local research cockpit and a shadow (paper) trading engine.

  NO real money.        NO broker connection.      NO exchange API key.
  NO order is ever placed, anywhere, by any part of this software.

Shadow trading writes to a local append-only, hash-chained log. It reads
public market data only, and only outside the reserved research window.

Verify what you have
--------------------
    sha256sum -c SHA256SUMS

manifest.json records the commit, the frozen specification hashes and the
model artefact hashes. Two builds of the same commit produce the same
content_hash.

Requirements
------------
Python 3.10 or newer. Nothing else for the cockpit.

The shadow engine additionally needs the optional model stack
(scikit-learn), which is not bundled. Without it every other part of the
application still runs, and `doctor` reports the checks it had to skip.

Commands
--------
    ./hyprl-run.sh                serve the application

From a full checkout of the repository:

    ./scripts/hyprl.sh start|stop|restart|status|doctor|logs
    ./scripts/hyprl.sh paper start|stop|status
    ./scripts/hyprl.sh export <archive.zip>
    ./scripts/hyprl.sh support-bundle <bundle.json>

Runtime state
-------------
Everything written at runtime lives under var/trading_lab/. Nothing outside
that directory is modified.

Updating
--------
There is no auto-updater. Replace this directory with a newer release. An
updater that can silently replace the code deciding trades would need to be
signed before it deserved to exist.
"""


def build_release(*, root=None, output=None, include_research_data: bool = True,
                  clean: bool = True) -> dict:
    """Assemble the bundle. Requires a frontend build to already exist."""
    root = pathlib.Path(root or ".").resolve()
    output = pathlib.Path(output or (root / DEFAULT_OUTPUT))
    dist = root / "apps/web/dist"
    if not (dist / "index.html").is_file():
        raise ReleaseError(
            "no frontend build; run ./scripts/hyprl.sh build first")

    if output.exists():
        if not clean:
            raise ReleaseError(f"{output.name} already exists")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    _copy_tree(dist, output / "frontend")
    for relative in PYTHON_SOURCES:
        _copy_tree(root / relative, output / relative)
    for relative in MODEL_DATA:
        _copy_tree(root / relative, output / relative)
    if include_research_data:
        for relative in RESEARCH_DATA:
            _copy_tree(root / relative, output / relative)

    # A package marker so `python -m scripts.trading_lab...` resolves from the
    # bundle root exactly as it does from a checkout.
    (output / "scripts" / "__init__.py").touch()

    shutil.copy2(root / "scripts/hyprl.sh", output / "scripts/hyprl.sh")
    (output / LAUNCHER_NAME).write_text(LAUNCHER, encoding="utf-8")
    (output / LAUNCHER_NAME).chmod(0o755)
    (output / "scripts/hyprl.sh").chmod(0o755)
    (output / README_NAME).write_text(README, encoding="utf-8")

    files = []
    for item in sorted(output.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(output).as_posix()
        if relative in (MANIFEST_NAME, CHECKSUM_NAME):
            continue
        files.append((relative, _sha256(item), item.stat().st_size))

    frontend = {
        "entry": "frontend/index.html",
        "files": sum(1 for name, _, _ in files if name.startswith("frontend/")),
        "bytes": sum(size for name, _, size in files
                     if name.startswith("frontend/")),
    }
    manifest = build_manifest(root=root, files=files, frontend=frontend,
                              includes_research_data=include_research_data)
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / CHECKSUM_NAME).write_text(
        "".join(f"{digest}  {name}\n" for name, digest, _ in files),
        encoding="utf-8")

    return {
        "output": output.name,
        "files": len(files),
        "bytes": sum(size for _, _, size in files),
        "frontend_bytes": frontend["bytes"],
        "includes_research_data": include_research_data,
        "content_hash": manifest["content_hash"],
        "git_commit": manifest["content"]["git_commit"],
    }


def verify_release(path) -> dict:
    """Re-check a bundle against its own manifest and checksums."""
    path = pathlib.Path(path)
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReleaseError("no manifest in this release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = manifest.get("content", {})
    report = {"checks": {}, "ok": False}
    report["checks"]["manifest_hash"] = (
        hashlib.sha256(_canonical(content).encode()).hexdigest()
        == manifest.get("content_hash"))
    report["checks"]["schema"] = (
        content.get("release_schema_version") == RELEASE_SCHEMA_VERSION)

    mismatched, missing = [], []
    for entry in content.get("files", []):
        target = path / entry["path"]
        if not target.is_file():
            missing.append(entry["path"])
            continue
        if _sha256(target) != entry["sha256"]:
            mismatched.append(entry["path"])
    report["checks"]["files_present"] = not missing
    report["checks"]["sha256"] = not mismatched
    report["checks"]["holdout_unobserved"] = (
        content.get("research_protection", {}).get("observed") is False)
    report["checks"]["no_real_money"] = (
        content.get("trading_safety", {}).get("real_money") is False)
    if missing:
        report["missing"] = missing
    if mismatched:
        report["mismatched"] = mismatched
    report["content_hash"] = manifest.get("content_hash")
    report["git_commit"] = content.get("git_commit")
    report["ok"] = all(report["checks"].values())
    return report
