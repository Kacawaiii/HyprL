"""The shared paper portfolio's own append-only log.

A separate database from Phase 5D's ``paper_v1.sqlite``, deliberately.

That log is hash-chained and records two independent single-product accounts.
Migrating it in place would mean rewriting payloads whose hashes exist
precisely so they cannot be rewritten, and the result would be a chain that
verifies while describing a history that never happened: BTC and ETH never
shared a dollar in those sessions, and no amount of re-keying makes them
have done. So the old store stays exactly as it is -- readable, auditable,
legacy -- and the shared portfolio starts a new one.

The chain, the natural keys and the fail-closed posture are the same as 5D's,
because those were the parts worth keeping. What changed is the unit of
record: an event belongs to a *portfolio batch*, and a batch spans every
instrument in the session.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
from dataclasses import dataclass

PORTFOLIO_EVENT_SCHEMA_VERSION = "trading-lab.paper-portfolio-event.v1"
PORTFOLIO_STORE_SCHEMA_VERSION = "trading-lab.paper-portfolio-store.v1"

GENESIS_HASH = "0" * 64

# Measured from this instrument's own last snapshot, never by testing a running
# total for divisibility. The Phase 5D trigger did the latter and could not
# fire at all when the stride was constant -- 2918 events, zero snapshots.
SNAPSHOT_EVERY_EVENTS = 250

EVENT_TYPES = (
    "PORTFOLIO_SESSION_STARTED",
    "PORTFOLIO_SESSION_STOPPED",
    "PORTFOLIO_BATCH_OPENED",
    "PORTFOLIO_INSTRUMENT_READY",
    "PORTFOLIO_BATCH_READY",
    "PORTFOLIO_BATCH_INCOMPLETE",
    "PORTFOLIO_TARGET_SET_CREATED",
    "PORTFOLIO_TARGET_SCALED",
    "PORTFOLIO_FILL",
    "PORTFOLIO_SNAPSHOT",
    "PORTFOLIO_VALUATION_UNAVAILABLE",
    "PORTFOLIO_RECOVERED",
    "PROTECTED_HOLDOUT_BOUNDARY_REACHED",
    "ERROR",
)


class PaperPortfolioStoreError(RuntimeError):
    """Raised when the log cannot be trusted or extended."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortfolioEvent:
    event_id: int
    session_id: str
    sequence: int
    event_type: str
    event_at: str
    instrument_id: str | None
    natural_key: str | None
    payload: dict
    previous_event_hash: str
    event_hash: str

    def body(self) -> dict:
        return {
            "schema_version": PORTFOLIO_EVENT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "event_at": self.event_at,
            "instrument_id": self.instrument_id,
            "natural_key": self.natural_key,
            "payload": self.payload,
            "previous_event_hash": self.previous_event_hash,
        }

    def recomputed_hash(self) -> str:
        return _sha256(_canonical(self.body()))


class PaperPortfolioStore:
    """SQLite-backed, append-only, hash-chained per session."""

    def __init__(self, database_path):
        path = pathlib.Path(database_path)
        if os.fspath(path) == ":memory:":
            raise PaperPortfolioStoreError(
                "an in-memory store cannot survive a restart, which is the only "
                "reason this log exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        event_at TEXT NOT NULL,
                        instrument_id TEXT,
                        natural_key TEXT,
                        payload TEXT NOT NULL,
                        previous_event_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL,
                        UNIQUE (session_id, sequence),
                        -- The exactly-once constraint. A restart that replays a
                        -- batch it already committed is refused by the database
                        -- rather than by a code path someone must remember.
                        UNIQUE (session_id, event_type, instrument_id, natural_key)
                    )""")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        last_event_id INTEGER NOT NULL,
                        last_event_hash TEXT NOT NULL,
                        state TEXT NOT NULL,
                        state_hash TEXT NOT NULL,
                        UNIQUE (session_id, last_event_id)
                    )""")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_sessions (
                        session_id TEXT PRIMARY KEY,
                        session_spec TEXT NOT NULL,
                        session_spec_hash TEXT NOT NULL,
                        started_at TEXT NOT NULL
                    )""")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS portfolio_events_by_instrument "
                    "ON portfolio_events (session_id, instrument_id, event_id)")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS portfolio_events_by_key "
                    "ON portfolio_events (session_id, natural_key, event_id)")
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    # --- sessions --------------------------------------------------------

    def register_session(self, *, session_id: str, session_spec: dict,
                         session_spec_hash: str, started_at: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO portfolio_sessions (session_id, "
                    "session_spec, session_spec_hash, started_at) VALUES (?, ?, ?, ?)",
                    (session_id, _canonical(session_spec), session_spec_hash,
                     started_at))
        finally:
            connection.close()

    def session(self, session_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM portfolio_sessions WHERE session_id = ?",
                (session_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {"session_id": row["session_id"],
                "session_spec": json.loads(row["session_spec"]),
                "session_spec_hash": row["session_spec_hash"],
                "started_at": row["started_at"]}

    def sessions(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            # GROUP BY, not DISTINCT: MIN() is an aggregate and SQLite
            # rejects it in ORDER BY without one. Ordering by first appearance
            # is what makes sessions()[-1] mean "the most recent session".
            rows = connection.execute(
                "SELECT session_id FROM portfolio_events "
                "GROUP BY session_id ORDER BY MIN(event_id)").fetchall()
            if not rows:
                rows = connection.execute(
                    "SELECT session_id FROM portfolio_sessions "
                    "ORDER BY started_at").fetchall()
        finally:
            connection.close()
        return tuple(row["session_id"] for row in rows)

    # --- writing ---------------------------------------------------------

    def append(self, *, session_id: str, event_type: str, event_at: str,
               instrument_id: str | None = None, natural_key: str | None = None,
               payload: dict | None = None) -> PortfolioEvent:
        """Append one event, or refuse. Never updates anything."""
        if event_type not in EVENT_TYPES:
            raise PaperPortfolioStoreError(f"unknown event type {event_type!r}")
        body_payload = payload or {}
        connection = self._connect()
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT sequence, event_hash FROM portfolio_events "
                    "WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                    (session_id,)).fetchone()
                sequence = (row["sequence"] + 1) if row else 1
                previous = row["event_hash"] if row else GENESIS_HASH
                event = PortfolioEvent(
                    event_id=0, session_id=session_id, sequence=sequence,
                    event_type=event_type, event_at=event_at,
                    instrument_id=instrument_id, natural_key=natural_key,
                    payload=body_payload, previous_event_hash=previous,
                    event_hash="")
                digest = _sha256(_canonical(event.body()))
                try:
                    cursor = connection.execute(
                        "INSERT INTO portfolio_events (session_id, sequence, "
                        "event_type, event_at, instrument_id, natural_key, payload, "
                        "previous_event_hash, event_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, sequence, event_type, event_at, instrument_id,
                         natural_key, _canonical(body_payload), previous, digest))
                except sqlite3.IntegrityError as error:
                    raise PaperPortfolioStoreError(
                        f"{event_type} for {instrument_id} / {natural_key} already "
                        "exists in this session; a restart must not replay it"
                    ) from error
                return PortfolioEvent(
                    event_id=cursor.lastrowid, session_id=session_id,
                    sequence=sequence, event_type=event_type, event_at=event_at,
                    instrument_id=instrument_id, natural_key=natural_key,
                    payload=body_payload, previous_event_hash=previous,
                    event_hash=digest)
        finally:
            connection.close()

    def has_event(self, *, session_id: str, event_type: str,
                  instrument_id: str | None, natural_key: str | None) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM portfolio_events WHERE session_id = ? AND "
                "event_type = ? AND instrument_id IS ? AND natural_key IS ?",
                (session_id, event_type, instrument_id, natural_key)).fetchone()
        finally:
            connection.close()
        return row is not None

    # --- reading ---------------------------------------------------------

    def _row_to_event(self, row) -> PortfolioEvent:
        return PortfolioEvent(
            event_id=row["event_id"], session_id=row["session_id"],
            sequence=row["sequence"], event_type=row["event_type"],
            event_at=row["event_at"], instrument_id=row["instrument_id"],
            natural_key=row["natural_key"], payload=json.loads(row["payload"]),
            previous_event_hash=row["previous_event_hash"],
            event_hash=row["event_hash"])

    def events(self, *, session_id: str | None = None,
               instrument_id: str | None = None, after_event_id: int | None = None,
               event_types=None, limit: int = 100) -> tuple[PortfolioEvent, ...]:
        if limit < 1 or limit > 5000:
            raise PaperPortfolioStoreError("limit must sit in 1..5000")
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if instrument_id:
            clauses.append("instrument_id = ?")
            params.append(instrument_id)
        if after_event_id is not None:
            clauses.append("event_id > ?")
            params.append(int(after_event_id))
        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM portfolio_events {where} ORDER BY event_id LIMIT ?",
                (*params, limit)).fetchall()
        finally:
            connection.close()
        return tuple(self._row_to_event(row) for row in rows)

    def latest_events(self, *, session_id: str | None = None,
                      instrument_id: str | None = None,
                      limit: int = 100) -> tuple[PortfolioEvent, ...]:
        if limit < 1 or limit > 5000:
            raise PaperPortfolioStoreError("limit must sit in 1..5000")
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if instrument_id:
            clauses.append("instrument_id = ?")
            params.append(instrument_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM portfolio_events {where} "
                "ORDER BY event_id DESC LIMIT ?", (*params, limit)).fetchall()
        finally:
            connection.close()
        return tuple(self._row_to_event(row) for row in reversed(rows))

    def count(self, *, session_id: str | None = None) -> int:
        connection = self._connect()
        try:
            if session_id:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM portfolio_events WHERE session_id = ?",
                    (session_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM portfolio_events").fetchone()
        finally:
            connection.close()
        return int(row["n"])

    def verify_chain(self, *, session_id: str) -> dict:
        """Recompute every link. Never repairs, only reports."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM portfolio_events WHERE session_id = ? "
                "ORDER BY sequence", (session_id,)).fetchall()
        finally:
            connection.close()
        previous = GENESIS_HASH
        for index, row in enumerate(rows, start=1):
            event = self._row_to_event(row)
            if event.sequence != index:
                raise PaperPortfolioStoreError(
                    f"sequence gap at {event.sequence}: an event is missing")
            if event.previous_event_hash != previous:
                raise PaperPortfolioStoreError(
                    f"broken chain at sequence {event.sequence}")
            if event.recomputed_hash() != event.event_hash:
                raise PaperPortfolioStoreError(
                    f"event {event.sequence} does not match its own hash")
            previous = event.event_hash
        return {"session_id": session_id, "events": len(rows),
                "head_hash": previous, "verified": True}

    # --- snapshots -------------------------------------------------------

    def write_snapshot(self, *, session_id: str, last_event_id: int,
                       last_event_hash: str, state: dict) -> str:
        state_hash = _sha256(_canonical(state))
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO portfolio_snapshots (session_id, "
                    "last_event_id, last_event_hash, state, state_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, last_event_id, last_event_hash,
                     _canonical(state), state_hash))
        finally:
            connection.close()
        return state_hash

    def latest_snapshot(self, *, session_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM portfolio_snapshots WHERE session_id = ? "
                "ORDER BY last_event_id DESC LIMIT 1", (session_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        state = json.loads(row["state"])
        if _sha256(_canonical(state)) != row["state_hash"]:
            raise PaperPortfolioStoreError(
                "portfolio snapshot does not match its own hash")
        return {"last_event_id": row["last_event_id"],
                "last_event_hash": row["last_event_hash"],
                "state": state, "state_hash": row["state_hash"]}

    def snapshot_count(self, *, session_id: str | None = None) -> int:
        connection = self._connect()
        try:
            if session_id:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM portfolio_snapshots "
                    "WHERE session_id = ?", (session_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM portfolio_snapshots").fetchone()
        finally:
            connection.close()
        return int(row["n"])
