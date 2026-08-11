"""Structured local logging: one JSON object per line, bounded, redacted.

Two properties matter more than convenience here.

**Redaction is structural, not a habit.** There is no API key, cookie or
Authorization header anywhere in this project today. Relying on that fact
would mean the first component that acquires one also acquires a leak. So the
redactor runs on every record unconditionally: forbidden keys are dropped at
any depth, and values that look like a home directory are rewritten before
anything reaches the disk. A log line cannot opt out.

**Growth is bounded.** A log that rotates only when someone remembers to look
is a disk-full incident with a delay fuse. Size-based rotation with a fixed
number of generations gives a hard ceiling: MAX_LOG_FILE_SIZE * MAX_LOG_FILES.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from datetime import datetime, timezone

LOG_SCHEMA_VERSION = "trading-lab.log-record.v1"

# 10 MiB x 5 generations: a 50 MiB ceiling, chosen so a week of chatty local
# operation still fits and an infinite loop cannot fill a disk.
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024
MAX_LOG_FILES = 5

LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")

# Dropped at any depth, matched case-insensitively as a substring so that
# "x_api_key", "Authorization" and "session-cookie" are all caught.
REDACTED_KEY_PATTERNS = (
    "authorization", "cookie", "api_key", "apikey", "api-key",
    "token", "secret", "password", "passphrase", "credential",
    "private_key", "access_key", "session_key", "bearer", "signature",
)
REDACTION_PLACEHOLDER = "[redacted]"

_HOME = re.compile(r"/(?:home|Users)/[^/\s\"']+")
_WINDOWS_HOME = re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+")


class StructuredLogError(RuntimeError):
    """Raised when a record cannot be written safely."""


def redact_text(value: str) -> str:
    """Replace anything that identifies a person or a machine account."""
    value = _HOME.sub("/<home>", value)
    return _WINDOWS_HOME.sub(r"<home>", value)


def _is_forbidden(key: object) -> bool:
    lowered = str(key).lower()
    return any(pattern in lowered for pattern in REDACTED_KEY_PATTERNS)


def redact(value, *, depth: int = 0):
    """Drop forbidden keys and scrub paths, at any nesting depth."""
    if depth > 12:                                   # defensive: no cycles on disk
        return REDACTION_PLACEHOLDER
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if _is_forbidden(key):
                clean[str(key)] = REDACTION_PLACEHOLDER
                continue
            clean[str(key)] = redact(item, depth=depth + 1)
        return clean
    if isinstance(value, (list, tuple)):
        return [redact(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))


def build_record(*, level: str, component: str, event: str, message: str = "",
                 session_id=None, product=None, error_code=None,
                 context=None, now=None) -> dict:
    """A log record with a fixed shape. Unknown extras live under context."""
    if level not in LEVELS:
        raise StructuredLogError(f"unknown level {level!r}; expected one of {LEVELS}")
    moment = now or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": LOG_SCHEMA_VERSION,
        "timestamp": moment,
        "level": level,
        "component": component,
        "event": event,
    }
    if message:
        record["message"] = redact_text(str(message))
    if session_id is not None:
        record["session_id"] = str(session_id)
    if product is not None:
        record["product"] = str(product)
    if error_code is not None:
        record["error_code"] = str(error_code)
    if context:
        record["context"] = redact(dict(context))
    return record


class StructuredLogger:
    """Append-only JSONL with size-based rotation.

    Not the stdlib logging module: that one is configured globally, which
    means any library in the process can reconfigure handlers and quietly
    disable the redaction this class exists to guarantee.
    """

    def __init__(self, path, *, max_bytes: int = MAX_LOG_FILE_SIZE,
                 max_files: int = MAX_LOG_FILES):
        if max_bytes <= 0 or max_files <= 0:
            raise StructuredLogError("log bounds must be positive; an unbounded "
                                     "log is a disk-full incident with a delay")
        self.path = pathlib.Path(path)
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- writing ---------------------------------------------------------

    def log(self, *, level: str, component: str, event: str, message: str = "",
            session_id=None, product=None, error_code=None, context=None,
            now=None) -> dict:
        record = build_record(level=level, component=component, event=event,
                              message=message, session_id=session_id,
                              product=product, error_code=error_code,
                              context=context, now=now)
        self._append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return record

    def info(self, component, event, **kwargs):
        return self.log(level="INFO", component=component, event=event, **kwargs)

    def warn(self, component, event, **kwargs):
        return self.log(level="WARN", component=component, event=event, **kwargs)

    def error(self, component, event, **kwargs):
        return self.log(level="ERROR", component=component, event=event, **kwargs)

    def _append(self, line: str) -> None:
        payload = (line + "\n").encode("utf-8")
        self._rotate_if_needed(len(payload))
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    # --- rotation --------------------------------------------------------

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self.max_bytes:
            return
        self.rotate()

    def rotate(self) -> None:
        """Shift generations and drop anything past the retention count."""
        if not self.path.exists():
            return
        # Delete the oldest first so the shift below never overwrites a
        # generation that retention still covers.
        oldest = self._generation(self.max_files - 1)
        if oldest.exists():
            oldest.unlink()
        for index in range(self.max_files - 2, 0, -1):
            source = self._generation(index)
            if source.exists():
                source.rename(self._generation(index + 1))
        self.path.rename(self._generation(1))

    def _generation(self, index: int) -> pathlib.Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def generations(self) -> tuple[pathlib.Path, ...]:
        found = [self.path] if self.path.exists() else []
        for index in range(1, self.max_files):
            candidate = self._generation(index)
            if candidate.exists():
                found.append(candidate)
        return tuple(found)

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.generations())

    def max_total_bytes(self) -> int:
        return self.max_bytes * self.max_files

    # --- reading ---------------------------------------------------------

    def tail(self, limit: int = 100) -> tuple[dict, ...]:
        """The most recent records, newest last. Bounded by construction."""
        if limit <= 0:
            return ()
        lines: list[str] = []
        for path in self.generations():                # newest file first
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:                            # pragma: no cover
                continue
            lines = [line for line in text.splitlines() if line.strip()] + lines
            if len(lines) >= limit:
                break
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue                               # a torn line is not fatal
        return tuple(records)
