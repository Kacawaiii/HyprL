"""Simple JSONL dead-letter queue for Discord poster failures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class DLQEntry:
    ts: str
    session: str
    message_key: str
    reason: str
    payload: Dict[str, Any]


class DiscordDLQ:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(
        self,
        session: str,
        message_key: str,
        reason: str,
        payload: Dict[str, Any],
    ) -> None:
        entry = DLQEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            session=session,
            message_key=message_key,
            reason=reason,
            payload=payload,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
