"""Where the running application is allowed to write, and nowhere else.

Every file HyprL creates at runtime lives under one root. That root is git
ignored, so a session can never turn into a commit, and it is the only place
the operations code is permitted to touch. Research artefacts, model
artefacts and source all stay read-only from the running app's point of view.

The paper event log keeps its Phase 5D location at the root of the runtime
directory rather than moving into ``runtime/``. Moving a live audit log to
gain a tidier tree would mean migrating a hash-chained database for cosmetic
reasons; the subdirectories added here sit beside it instead.

Directories are created 0700 and files 0600. Nothing here is secret today --
there are no keys, no tokens and no broker credentials anywhere in this
project -- but a runtime directory is exactly the kind of place where a
future secret would land, and a world-readable default would then already be
wrong.
"""

from __future__ import annotations

import os
import pathlib

RUNTIME_SCHEMA_VERSION = "trading-lab.runtime-layout.v1"

DEFAULT_RUNTIME_ROOT = pathlib.Path("var/trading_lab")

# Kept at the root for continuity with Phase 5D.
PAPER_DATABASE = "paper_v1.sqlite"
PAPER_SESSION_MARKER = "paper_session.json"

DIRECTORY_MODE = 0o700
FILE_MODE = 0o600

SUBDIRECTORIES = ("runtime", "logs", "exports", "support", "tmp")


class RuntimeLayoutError(RuntimeError):
    """Raised when the runtime directory cannot be used safely."""


class RuntimeLayout:
    """A fixed set of directories under a single resolved root."""

    def __init__(self, root=None):
        self.root = pathlib.Path(root or DEFAULT_RUNTIME_ROOT)

    # --- locations -------------------------------------------------------

    @property
    def runtime(self) -> pathlib.Path:
        return self.root / "runtime"

    @property
    def logs(self) -> pathlib.Path:
        return self.root / "logs"

    @property
    def exports(self) -> pathlib.Path:
        return self.root / "exports"

    @property
    def support(self) -> pathlib.Path:
        return self.root / "support"

    @property
    def tmp(self) -> pathlib.Path:
        return self.root / "tmp"

    @property
    def paper_database(self) -> pathlib.Path:
        return self.root / PAPER_DATABASE

    @property
    def paper_session_marker(self) -> pathlib.Path:
        return self.root / PAPER_SESSION_MARKER

    @property
    def ops_database(self) -> pathlib.Path:
        """Health history and lifecycle state.

        Deliberately not the paper database: that one is an append-only audit
        trail, and operational telemetry with a retention policy has no
        business sharing a file with it.
        """
        return self.runtime / "ops.sqlite"

    @property
    def settings_file(self) -> pathlib.Path:
        return self.runtime / "settings.json"

    @property
    def pid_file(self) -> pathlib.Path:
        return self.runtime / "hyprl-app.pid"

    @property
    def lifecycle_file(self) -> pathlib.Path:
        return self.runtime / "lifecycle.json"

    @property
    def application_log(self) -> pathlib.Path:
        return self.logs / "hyprl.jsonl"

    # --- creation --------------------------------------------------------

    def ensure(self) -> "RuntimeLayout":
        """Create what is missing. Idempotent, and never widens permissions."""
        for path in (self.root, *(self.root / name for name in SUBDIRECTORIES)):
            path.mkdir(parents=True, exist_ok=True)
            _tighten(path, DIRECTORY_MODE)
        return self

    def describe(self) -> dict:
        """A layout report with no absolute path in it.

        Support bundles and the operations API both surface this, and neither
        has any reason to publish where on a disk a user keeps their work.
        """
        return {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "directories": {
                name: (self.root / name).is_dir() for name in SUBDIRECTORIES},
            "paper_database_present": self.paper_database.is_file(),
            "ops_database_present": self.ops_database.is_file(),
            "settings_present": self.settings_file.is_file(),
        }


def _tighten(path: pathlib.Path, mode: int) -> None:
    """Narrow permissions, never widen them.

    A umask can only remove bits, so creating a directory does not guarantee
    it is private. Chmod does, and refusing to widen means an operator who
    deliberately locked something down further keeps their choice.
    """
    try:
        current = path.stat().st_mode & 0o777
    except OSError:                                  # pragma: no cover
        return
    if current & ~mode:
        try:
            os.chmod(path, current & mode)
        except OSError:                              # pragma: no cover
            pass


def write_private(path: pathlib.Path, text: str) -> pathlib.Path:
    """Write a runtime file that is not world-readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _tighten(path.parent, DIRECTORY_MODE)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:                            # pragma: no cover
        raise
    _tighten(path, FILE_MODE)
    return path


def is_world_writable(path: pathlib.Path) -> bool:
    return bool(path.stat().st_mode & 0o002)
