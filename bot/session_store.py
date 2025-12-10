"""Session metadata helpers for Discord routing."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_STORE_PATH = REPO_ROOT / "live" / "logs" / "discord_sessions.json"


def slugify(name: str) -> str:
    """Normalize a session name to a slug (lowercase, alnum + dash)."""
    lowered = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    collapsed = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return collapsed


def get_sessions(path: Path) -> dict[str, dict[str, Any]]:
    """Load session mapping from JSON; returns empty dict on error or missing file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.error("Failed reading %s: %s", path, exc)
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return {}


def save_sessions(path: Path, data: dict[str, dict[str, Any]]) -> None:
    """Persist session mapping atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=path.name,
        suffix=".tmp",
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)


def resolve_session(session_name_or_slug: str, sessions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve a session by slug or case-insensitive name."""
    target = session_name_or_slug.strip().lower()
    if target in sessions:
        return sessions[target]
    for session in sessions.values():
        slug = str(session.get("slug", "")).lower()
        name = str(session.get("name", "")).lower()
        if target in (slug, name):
            return session
    return None


def get_session_channels(session: dict[str, Any]) -> dict[str, int]:
    """Return channel mapping as ints."""
    channels = session.get("channels") or {}
    resolved: dict[str, int] = {}
    for key, value in channels.items():
        try:
            resolved[key] = int(value)
        except (TypeError, ValueError):
            continue
    return resolved


__all__ = [
    "SESSION_STORE_PATH",
    "slugify",
    "get_sessions",
    "save_sessions",
    "resolve_session",
    "get_session_channels",
]
