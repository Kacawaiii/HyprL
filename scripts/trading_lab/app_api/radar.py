"""Read-only view over the sanitized cockpit snapshots written by scripts/radar/cockpit_export.py.

The directory is fixed at startup. Only two files are ever served, each bounded in size, and the body is
re-validated as JSON of the expected schema, so a stray file cannot leak through this endpoint."""
import json
from pathlib import Path

from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError, NotFoundError


class Unavailable(AppApiError):
    status = 503

MAX_BYTES = 8 * 1024 * 1024
FILES = {"/api/v1/radar/home": ("radar-home.json", "cockpit-radar-home-v1"),
         "/api/v1/radar/paper": ("paper.json", "cockpit-paper-v1")}


class RadarViews:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None

    def dispatch(self, path, query):
        if self.root is None:
            raise Unavailable("radar snapshots not configured")
        if query:
            raise AppApiError("radar views do not accept query parameters")
        if path not in FILES:
            raise NotFoundError("no such radar endpoint")
        name, schema = FILES[path]
        target = self.root / name
        if target.is_symlink() or not target.is_file():
            raise Unavailable("radar snapshot unavailable")
        if target.stat().st_size > MAX_BYTES:
            raise ConflictError("radar snapshot too large")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ConflictError("radar snapshot unreadable") from None
        if not isinstance(payload, dict) or payload.get("schema") != schema:
            raise ConflictError("radar snapshot has an unexpected schema")
        return payload
