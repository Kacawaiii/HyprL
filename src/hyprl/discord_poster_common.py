"""Shared Discord posting helpers and constants."""

from __future__ import annotations

import json
import os
from typing import Any, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DISCORD_API_BASE = os.getenv("DISCORD_API_BASE", "https://discord.com/api/v10")
DISCORD_USER_AGENT = os.getenv(
    "DISCORD_USER_AGENT",
    "DiscordBot (HyprL-Ascendant/1.0; +https://example.com)",
)

IDEMP_STATUS_SENT = "SENT"
IDEMP_STATUS_FAILED_TEMP = "FAILED_TEMP"
IDEMP_STATUS_FAILED_PERM = "FAILED_PERM"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def post_discord_message(
    token: str, channel_id: int, payload: dict[str, Any], dry_run: bool
) -> tuple[int | None, str | None, str | None]:
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    if dry_run:
        print(f"[DRY] POST {url} payload={json.dumps(payload)}")
        return 200, None, None

    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": DISCORD_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
            message_id: str | None = None
            try:
                payload_json = json.loads(resp_body) if resp_body else {}
                if isinstance(payload_json, dict) and payload_json.get("id"):
                    message_id = str(payload_json.get("id"))
            except json.JSONDecodeError:
                message_id = None
            return status, message_id, resp_body
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, body_text
    except URLError as exc:
        return None, None, str(exc)
