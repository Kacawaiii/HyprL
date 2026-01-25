"""Minimal SQLite entitlements store (accounts ⇄ sessions)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Entitlement:
    entitlement_id: int
    account_id: str
    session_slug: str
    plan_id: str
    valid_from: str
    valid_until: str
    status: str  # "ACTIVE", "GRACE", "EXPIRED"


class EntitlementsStore:
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
                CREATE TABLE IF NOT EXISTS entitlements (
                    entitlement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id     TEXT NOT NULL,
                    session_slug   TEXT NOT NULL,
                    plan_id        TEXT NOT NULL,
                    valid_from     TEXT NOT NULL,
                    valid_until    TEXT NOT NULL,
                    status         TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_entitlement(
        self,
        account_id: str,
        session_slug: str,
        plan_id: str,
        valid_from: str,
        valid_until: str,
        status: str = "ACTIVE",
    ) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO entitlements (account_id, session_slug, plan_id, valid_from, valid_until, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (account_id, session_slug, plan_id, valid_from, valid_until, status),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_active_for_session(self, session_slug: str) -> List[Entitlement]:
        now = datetime.utcnow().isoformat()
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT entitlement_id, account_id, session_slug, plan_id,
                       valid_from, valid_until, status
                FROM entitlements
                WHERE session_slug = ?
                  AND status = 'ACTIVE'
                  AND valid_from <= ?
                  AND valid_until >= ?
                """,
                (session_slug, now, now),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return [
            Entitlement(
                entitlement_id=row[0],
                account_id=row[1],
                session_slug=row[2],
                plan_id=row[3],
                valid_from=row[4],
                valid_until=row[5],
                status=row[6],
            )
            for row in rows
        ]

    def update_status(self, entitlement_id: int, status: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE entitlements SET status = ? WHERE entitlement_id = ?
                """,
                (status, entitlement_id),
            )
            conn.commit()
        finally:
            conn.close()
