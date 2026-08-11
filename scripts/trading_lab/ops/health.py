"""Health states and a bounded local history.

One vocabulary, defined once. Before this module the codebase had a paper
engine speaking STOPPED/STARTING/RUNNING/EMBARGOED/DEGRADED/ERROR, an API
speaking ok/no_corpus, and a CLI speaking English sentences. Three vocabularies
for one question is how a dashboard ends up showing a green tick and a red
banner for the same fact.

The mapping that matters:

* an active research embargo is ``EMBARGOED``, never ``ERROR`` -- the guard
  doing its job is the system working, and paging someone for it teaches them
  to ignore the signal;
* a network outage is ``DEGRADED`` -- the machine cannot reach an exchange,
  which says nothing about the integrity of what is already recorded;
* a broken hash chain is ``ERROR`` -- the audit trail is the product.

History lives in its own SQLite file with a row cap. It is telemetry, it is
allowed to be pruned, and it therefore does not belong in the append-only
paper log.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

HEALTH_SCHEMA_VERSION = "trading-lab.health-record.v1"

HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
ERROR = "ERROR"
STOPPED = "STOPPED"
EMBARGOED = "EMBARGOED"

HEALTH_STATES = (HEALTHY, DEGRADED, ERROR, STOPPED, EMBARGOED)

COMPONENTS = (
    "app_api",
    "paper_engine",
    "event_store",
    "market_ingestion",
    "model",
    "holdout_guard",
)

# Bounded on purpose. Health telemetry is the kind of table that grows for
# years without anyone noticing until it is the largest file on the machine.
MAX_HEALTH_RECORDS = 10_000

# The order used when several components must collapse into one headline.
# ERROR outranks everything: a healthy API in front of a corrupt log is not a
# healthy system.
_SEVERITY = {HEALTHY: 0, STOPPED: 1, EMBARGOED: 2, DEGRADED: 3, ERROR: 4}


class HealthError(RuntimeError):
    """Raised when a health record is malformed."""


@dataclass(frozen=True)
class HealthRecord:
    observed_at: str
    component: str
    status: str
    latency_ms: float | None = None
    error_code: str | None = None
    details: dict | None = None

    def __post_init__(self):
        if self.component not in COMPONENTS:
            raise HealthError(
                f"unknown component {self.component!r}; expected one of {COMPONENTS}")
        if self.status not in HEALTH_STATES:
            raise HealthError(
                f"unknown status {self.status!r}; expected one of {HEALTH_STATES}")

    def payload(self) -> dict:
        return {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "observed_at": self.observed_at,
            "component": self.component,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "details": self.details or {},
        }


def worst(statuses) -> str:
    """The headline status for a set of components."""
    found = [status for status in statuses if status in _SEVERITY]
    if not found:
        return STOPPED
    return max(found, key=lambda status: _SEVERITY[status])


class HealthHistory:
    """A capped ring of observations in the operations database."""

    def __init__(self, database_path, *, max_records: int = MAX_HEALTH_RECORDS):
        self._path = pathlib.Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        if self.max_records <= 0:
            raise HealthError("health retention must be positive")
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS health_records (
                        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        observed_at TEXT NOT NULL,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        latency_ms REAL,
                        error_code TEXT,
                        details TEXT NOT NULL
                    )""")
                # The two queries that exist: newest-first overall, and
                # newest-first for one component. Both are covered here.
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS health_by_component "
                    "ON health_records (component, record_id DESC)")
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def record(self, record: HealthRecord) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO health_records (observed_at, component, status, "
                    "latency_ms, error_code, details) VALUES (?, ?, ?, ?, ?, ?)",
                    (record.observed_at, record.component, record.status,
                     record.latency_ms, record.error_code,
                     json.dumps(record.details or {}, sort_keys=True)))
                connection.execute(
                    "DELETE FROM health_records WHERE record_id <= ("
                    "  SELECT MAX(record_id) - ? FROM health_records)",
                    (self.max_records,))
        finally:
            connection.close()

    def observe(self, component: str, status: str, *, latency_ms=None,
                error_code=None, details=None, now=None) -> HealthRecord:
        moment = now or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = HealthRecord(observed_at=moment, component=component,
                              status=status, latency_ms=latency_ms,
                              error_code=error_code, details=details)
        self.record(record)
        return record

    def recent(self, *, component: str | None = None, limit: int = 100) -> tuple:
        if limit <= 0 or limit > MAX_HEALTH_RECORDS:
            raise HealthError(f"limit must sit in 1..{MAX_HEALTH_RECORDS}")
        connection = self._connect()
        try:
            if component is None:
                rows = connection.execute(
                    "SELECT * FROM health_records ORDER BY record_id DESC LIMIT ?",
                    (limit,)).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM health_records WHERE component = ? "
                    "ORDER BY record_id DESC LIMIT ?", (component, limit)).fetchall()
        finally:
            connection.close()
        return tuple({
            "record_id": row["record_id"],
            "observed_at": row["observed_at"],
            "component": row["component"],
            "status": row["status"],
            "latency_ms": row["latency_ms"],
            "error_code": row["error_code"],
            "details": json.loads(row["details"]),
        } for row in rows)

    def latest_per_component(self) -> dict:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT component, status, observed_at, error_code FROM health_records "
                "WHERE record_id IN (SELECT MAX(record_id) FROM health_records "
                "GROUP BY component)").fetchall()
        finally:
            connection.close()
        return {row["component"]: {"status": row["status"],
                                   "observed_at": row["observed_at"],
                                   "error_code": row["error_code"]}
                for row in rows}

    def count(self) -> int:
        connection = self._connect()
        try:
            return connection.execute(
                "SELECT COUNT(*) AS n FROM health_records").fetchone()["n"]
        finally:
            connection.close()
