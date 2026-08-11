"""Append-only event log for shadow trading.

A paper session is a long-lived process that will be killed, restarted, and
left running across machine reboots. Its state therefore cannot live in
memory, and it cannot live in a mutable row that the last writer wins: both
lose the ability to answer "what did the system actually believe, and when".

So the store is an append-only log with a per-session hash chain. Each event
commits to its predecessor, which makes silent tampering or partial truncation
detectable rather than merely unlikely. Nothing is ever updated in place.

Idempotency is the other half. After a crash the engine replays whatever is
missing, and a candle that was already processed must not produce a second
prediction or a second fill. Every event that belongs to a specific candle
carries a natural key, and the database refuses the duplicate rather than
trusting the caller to check first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import sqlite3

PAPER_EVENT_SCHEMA_VERSION = "trading-lab.paper-event.v1"
SNAPSHOT_EVERY_EVENTS = 250
GENESIS_HASH = "0" * 64

EVENT_TYPES = (
    "SESSION_STARTED",
    "SESSION_STOPPED",
    "CANDLE_INGESTED",
    "FEATURES_READY",
    "MODEL_LOADED",
    "PREDICTION_CREATED",
    "SIGNAL_CREATED",
    "POSITION_TARGET_CREATED",
    "SIMULATED_FILL",
    "PORTFOLIO_SNAPSHOT",
    "GAP_DETECTED",
    "PROTECTED_HOLDOUT_BOUNDARY_REACHED",
    "ERROR",
)


class PaperEventStoreError(RuntimeError):
    """Raised when the log cannot be trusted or extended."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperEvent:
    event_id: int
    session_id: str
    sequence: int
    event_type: str
    event_at: str
    product: str | None
    natural_key: str | None
    payload: dict
    previous_event_hash: str
    event_hash: str

    def body(self) -> dict:
        return {
            "schema_version": PAPER_EVENT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "event_at": self.event_at,
            "product": self.product,
            "natural_key": self.natural_key,
            "payload": self.payload,
            "previous_event_hash": self.previous_event_hash,
        }

    def recomputed_hash(self) -> str:
        return _sha256(_canonical(self.body()))


class PaperEventStore:
    """SQLite-backed, append-only, hash-chained per session."""

    def __init__(self, database_path):
        path = pathlib.Path(database_path)
        if os.fspath(path) == ":memory:":
            raise PaperEventStoreError(
                "an in-memory store cannot survive a restart, which is the only "
                "reason this log exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS paper_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        event_at TEXT NOT NULL,
                        product TEXT,
                        natural_key TEXT,
                        payload TEXT NOT NULL,
                        previous_event_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL,
                        UNIQUE (session_id, sequence),
                        UNIQUE (session_id, product, event_type, natural_key)
                    )""")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS paper_state_snapshots (
                        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        product TEXT NOT NULL,
                        last_event_id INTEGER NOT NULL,
                        last_event_hash TEXT NOT NULL,
                        state TEXT NOT NULL,
                        state_hash TEXT NOT NULL,
                        UNIQUE (session_id, product, last_event_id)
                    )""")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS paper_events_by_product "
                    "ON paper_events (session_id, product, event_id)")
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

    # --- writing ---------------------------------------------------------

    def append(self, *, session_id: str, event_type: str, event_at: str,
               product: str | None = None, natural_key: str | None = None,
               payload: dict | None = None) -> PaperEvent:
        """Append one event, or refuse. Never updates anything."""
        if event_type not in EVENT_TYPES:
            raise PaperEventStoreError(f"unknown event type {event_type!r}")
        body_payload = payload or {}
        connection = self._connect()
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT sequence, event_hash FROM paper_events "
                    "WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                    (session_id,)).fetchone()
                sequence = (row["sequence"] + 1) if row else 1
                previous = row["event_hash"] if row else GENESIS_HASH
                event = PaperEvent(
                    event_id=0, session_id=session_id, sequence=sequence,
                    event_type=event_type, event_at=event_at, product=product,
                    natural_key=natural_key, payload=body_payload,
                    previous_event_hash=previous, event_hash="")
                digest = _sha256(_canonical(event.body()))
                try:
                    cursor = connection.execute(
                        "INSERT INTO paper_events (session_id, sequence, event_type, "
                        "event_at, product, natural_key, payload, previous_event_hash, "
                        "event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, sequence, event_type, event_at, product,
                         natural_key, _canonical(body_payload), previous, digest))
                except sqlite3.IntegrityError as error:
                    raise PaperEventStoreError(
                        f"{event_type} for {product} / {natural_key} already exists in "
                        "this session; a restart must not replay it") from error
                return PaperEvent(
                    event_id=cursor.lastrowid, session_id=session_id, sequence=sequence,
                    event_type=event_type, event_at=event_at, product=product,
                    natural_key=natural_key, payload=body_payload,
                    previous_event_hash=previous, event_hash=digest)
        finally:
            connection.close()

    def has_event(self, *, session_id: str, event_type: str, product: str | None,
                  natural_key: str | None) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM paper_events WHERE session_id = ? AND event_type = ? "
                "AND product IS ? AND natural_key IS ?",
                (session_id, event_type, product, natural_key)).fetchone()
            return row is not None
        finally:
            connection.close()

    # --- reading ---------------------------------------------------------

    def _row_to_event(self, row) -> PaperEvent:
        return PaperEvent(
            event_id=row["event_id"], session_id=row["session_id"],
            sequence=row["sequence"], event_type=row["event_type"],
            event_at=row["event_at"], product=row["product"],
            natural_key=row["natural_key"], payload=json.loads(row["payload"]),
            previous_event_hash=row["previous_event_hash"],
            event_hash=row["event_hash"])

    def events(self, *, session_id: str | None = None, product: str | None = None,
               after_event_id: int | None = None, event_types=None,
               limit: int = 100) -> tuple[PaperEvent, ...]:
        if limit < 1 or limit > 5000:
            raise PaperEventStoreError("limit must sit in 1..5000")
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?"); params.append(session_id)
        if product:
            clauses.append("product = ?"); params.append(product)
        if after_event_id is not None:
            clauses.append("event_id > ?"); params.append(int(after_event_id))
        if event_types:
            marks = ",".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({marks})"); params.extend(event_types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM paper_events {where} ORDER BY event_id ASC LIMIT ?",
                (*params, limit)).fetchall()
            return tuple(self._row_to_event(row) for row in rows)
        finally:
            connection.close()

    def latest_events(self, *, session_id: str | None = None,
                      product: str | None = None, limit: int = 100
                      ) -> tuple[PaperEvent, ...]:
        if limit < 1 or limit > 5000:
            raise PaperEventStoreError("limit must sit in 1..5000")
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?"); params.append(session_id)
        if product:
            clauses.append("product = ?"); params.append(product)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM paper_events {where} ORDER BY event_id DESC LIMIT ?",
                (*params, limit)).fetchall()
            return tuple(reversed([self._row_to_event(row) for row in rows]))
        finally:
            connection.close()

    def sessions(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT session_id, MAX(event_id) AS last FROM paper_events "
                "GROUP BY session_id ORDER BY last ASC").fetchall()
            return tuple(row["session_id"] for row in rows)
        finally:
            connection.close()

    def count(self, *, session_id: str | None = None) -> int:
        connection = self._connect()
        try:
            if session_id:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM paper_events WHERE session_id = ?",
                    (session_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM paper_events").fetchone()
            return row["n"]
        finally:
            connection.close()

    # --- integrity -------------------------------------------------------

    def verify_chain(self, *, session_id: str) -> dict:
        """Recompute every link. A single altered byte breaks the chain."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM paper_events WHERE session_id = ? ORDER BY sequence ASC",
                (session_id,)).fetchall()
        finally:
            connection.close()
        previous = GENESIS_HASH
        expected_sequence = 1
        for row in rows:
            event = self._row_to_event(row)
            if event.sequence != expected_sequence:
                raise PaperEventStoreError(
                    f"session {session_id} jumps from sequence {expected_sequence - 1} "
                    f"to {event.sequence}; the log is not contiguous")
            if event.previous_event_hash != previous:
                raise PaperEventStoreError(
                    f"event {event.sequence} does not follow its predecessor")
            if event.recomputed_hash() != event.event_hash:
                raise PaperEventStoreError(
                    f"event {event.sequence} does not match its own hash")
            previous = event.event_hash
            expected_sequence += 1
        return {"session_id": session_id, "events": len(rows), "head_hash": previous,
                "verified": True}

    # --- snapshots -------------------------------------------------------

    def write_snapshot(self, *, session_id: str, product: str, last_event_id: int,
                       last_event_hash: str, state: dict) -> str:
        state_hash = _sha256(_canonical(state))
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO paper_state_snapshots (session_id, product, "
                    "last_event_id, last_event_hash, state, state_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, product, last_event_id, last_event_hash,
                     _canonical(state), state_hash))
        finally:
            connection.close()
        return state_hash

    def latest_snapshot(self, *, session_id: str, product: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM paper_state_snapshots WHERE session_id = ? AND product = ? "
                "ORDER BY last_event_id DESC LIMIT 1", (session_id, product)).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        state = json.loads(row["state"])
        if _sha256(_canonical(state)) != row["state_hash"]:
            raise PaperEventStoreError(
                f"snapshot for {product} does not match its own hash")
        return {"last_event_id": row["last_event_id"],
                "last_event_hash": row["last_event_hash"],
                "state": state, "state_hash": row["state_hash"]}
