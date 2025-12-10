#!/usr/bin/env python3
"""Send a Discord message via bot token to a target channel."""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Discord message via bot token.")
    parser.add_argument("--channel-id", required=True, help="Discord channel ID to post into.")
    parser.add_argument(
        "--token",
        default=os.getenv("DISCORD_BOT_TOKEN"),
        help="Discord bot token. Defaults to DISCORD_BOT_TOKEN env var.",
    )
    parser.add_argument(
        "--message",
        required=True,
        help="Message content (max 2000 chars; will be truncated if longer).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10).",
    )
    return parser.parse_args()


def fatal(msg: str) -> NoReturn:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def send_message(token: str, channel_id: str, content: str, *, timeout: float) -> dict:
    if not token:
        fatal("Missing bot token (set --token or DISCORD_BOT_TOKEN).")
    trimmed = content[:2000]
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"content": trimmed}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
    if resp.status_code >= 300:
        fatal(f"Discord API error {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError:
        return {"status_code": resp.status_code, "body": resp.text}


def main() -> None:
    args = parse_args()
    result = send_message(args.token, args.channel_id, args.message, timeout=args.timeout)
    message_id = result.get("id", "unknown")
    print(f"[OK] Sent to channel {args.channel_id}, message_id={message_id}")


if __name__ == "__main__":
    main()
