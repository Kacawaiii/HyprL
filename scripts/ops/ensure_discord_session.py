#!/usr/bin/env python3
"""
Headless Discord session/channel creator.

Creates or reuses a category and three text channels (<slug>-overview/alerts/trades)
under a given guild, then updates live/logs/discord_sessions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Ensure repo root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.session_store import slugify  # noqa: E402

DISCORD_API_BASE = os.getenv("DISCORD_API_BASE", "https://discord.com/api/v10")
SESSIONS_PATH = ROOT / "live" / "logs" / "discord_sessions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure Discord channels exist for a HyprL session.")
    parser.add_argument("--guild-id", required=True, help="Discord guild/server ID.")
    parser.add_argument("--name", required=True, help="Human-friendly session name (used for category label).")
    parser.add_argument("--slug", help="Session slug; defaults to slugified name.")
    parser.add_argument(
        "--category",
        help='Category name (default: "HyprL - <name>").',
    )
    parser.add_argument(
        "--token",
        default=os.getenv("DISCORD_BOT_TOKEN"),
        help="Bot token; defaults to DISCORD_BOT_TOKEN env.",
    )
    return parser.parse_args()


def api_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "User-Agent": "HyprL/discord-session-bootstrap",
    }


def load_sessions() -> dict[str, Any]:
    if not SESSIONS_PATH.exists():
        return {}
    try:
        raw = SESSIONS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_sessions(sessions: dict[str, Any]) -> None:
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_PATH.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_category(client: httpx.Client, guild_id: str, name: str) -> int:
    resp = client.get(f"{DISCORD_API_BASE}/guilds/{guild_id}/channels")
    resp.raise_for_status()
    channels = resp.json()
    for ch in channels:
        if ch.get("type") == 4 and ch.get("name") == name:
            return int(ch["id"])
    resp = client.post(f"{DISCORD_API_BASE}/guilds/{guild_id}/channels", json={"name": name, "type": 4})
    resp.raise_for_status()
    return int(resp.json()["id"])


def upsert_text_channel(client: httpx.Client, guild_id: str, category_id: int, name: str) -> int:
    resp = client.get(f"{DISCORD_API_BASE}/guilds/{guild_id}/channels")
    resp.raise_for_status()
    channels = resp.json()
    for ch in channels:
        if ch.get("type") == 0 and ch.get("name") == name and str(ch.get("parent_id")) == str(category_id):
            return int(ch["id"])
    payload = {"name": name, "type": 0, "parent_id": str(category_id)}
    resp = client.post(f"{DISCORD_API_BASE}/guilds/{guild_id}/channels", json=payload)
    resp.raise_for_status()
    return int(resp.json()["id"])


def main() -> None:
    args = parse_args()
    token = args.token
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN (or --token) required.")
    slug = args.slug or slugify(args.name)
    if not slug:
        raise SystemExit("Invalid slug; provide --slug or a valid --name.")
    category_name = args.category or f"HyprL - {args.name}"

    with httpx.Client(headers=api_headers(token), timeout=15.0) as client:
        category_id = upsert_category(client, args.guild_id, category_name)
        overview_id = upsert_text_channel(client, args.guild_id, category_id, f"{slug}-overview")
        alerts_id = upsert_text_channel(client, args.guild_id, category_id, f"{slug}-alerts")
        trades_id = upsert_text_channel(client, args.guild_id, category_id, f"{slug}-trades")

    sessions = load_sessions()
    sessions[slug] = {
        "name": args.name,
        "slug": slug,
        "category_id": category_id,
        "channels": {
            "overview": overview_id,
            "alerts": alerts_id,
            "trades": trades_id,
        },
    }
    save_sessions(sessions)
    print(f"[OK] Updated {SESSIONS_PATH} with session '{slug}' -> {sessions[slug]}")


if __name__ == "__main__":  # pragma: no cover
    main()
