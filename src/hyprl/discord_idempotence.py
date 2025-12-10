"""SQLite-backed idempotence store for Discord messages."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DiscordMessageRecord:
    message_key: str
    session: str
    discord_message_id: Optional[str]
    status: str  # "SENT", "FAILED_TEMP", "FAILED_PERM"


class DiscordIdempotenceStore:
    """Simple SQLite store keyed by (session, message_key)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path.as_posix())

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_messages (
                    message_key TEXT NOT NULL,
                    session     TEXT NOT NULL,
                    discord_message_id TEXT,
                    status      TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session, message_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS discord_messages_set_updated_at
                AFTER UPDATE ON discord_messages
                BEGIN
                  UPDATE discord_messages
                  SET updated_at = CURRENT_TIMESTAMP
                  WHERE rowid = NEW.rowid;
                END;
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, session: str, message_key: str) -> Optional[DiscordMessageRecord]:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT message_key, session, discord_message_id, status
                FROM discord_messages
                WHERE session = ? AND message_key = ?
                """,
                (session, message_key),
            )
            row = cur.fetchone()
            if not row:
                return None
            return DiscordMessageRecord(
                message_key=row[0],
                session=row[1],
                discord_message_id=row[2],
                status=row[3],
            )
        finally:
            conn.close()

    def upsert(
        self,
        session: str,
        message_key: str,
        status: str,
        discord_message_id: Optional[str] = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO discord_messages (session, message_key, status, discord_message_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session, message_key)
                DO UPDATE SET
                    status = excluded.status,
                    discord_message_id = COALESCE(excluded.discord_message_id, discord_messages.discord_message_id)
                """,
                (session, message_key, status, discord_message_id),
            )
            conn.commit()
        finally:
            conn.close()
