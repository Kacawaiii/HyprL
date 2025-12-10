#!/usr/bin/env python3
"""Replay Discord DLQ entries after transient failures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Ensure repository root is on sys.path for bot.session_store imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.session_store import (  # noqa: E402
    get_session_channels,
    get_sessions,
    resolve_session,
)
from hyprl.discord_idempotence import DiscordIdempotenceStore  # noqa: E402
from hyprl.discord_poster_common import (  # noqa: E402
    IDEMP_STATUS_FAILED_PERM,
    IDEMP_STATUS_FAILED_TEMP,
    IDEMP_STATUS_SENT,
    TRANSIENT_STATUS_CODES,
    post_discord_message,
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Discord DLQ entries.")
    parser.add_argument("--session", required=True, help="Session slug (e.g., friends-live).")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory containing discord_dlq.jsonl and discord_poster.db (default: live/logs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of DLQ entries to attempt (default: 100).",
    )
    parser.add_argument(
        "--include-permanent",
        action="store_true",
        help="Also attempt entries whose idempotence status is FAILED_PERM (use after fixing root cause).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send to Discord; just log what would be retried.",
    )
    return parser.parse_args(argv)


def load_dlq_entries(path: Path, session: str) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("session") != session:
                continue
            yield entry


def _extract_payload(payload_wrapper: Any) -> Dict[str, Any] | None:
    if not isinstance(payload_wrapper, dict):
        return None
    if "payload" in payload_wrapper and isinstance(payload_wrapper.get("payload"), dict):
        return payload_wrapper["payload"]
    return payload_wrapper if payload_wrapper else None


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("[ERROR] DISCORD_BOT_TOKEN is required", file=sys.stderr)
        return 1

    log_root = Path(args.log_root)
    dlq_path = log_root / "discord_dlq.jsonl"
    idempotence_store = DiscordIdempotenceStore(log_root / "discord_poster.db")

    sessions_file = Path(os.getenv("DISCORD_SESSIONS_FILE") or (log_root / "discord_sessions.json"))
    sessions = get_sessions(sessions_file)
    session = resolve_session(args.session, sessions)
    if not session:
        print(f"[ERROR] Session '{args.session}' not found in {sessions_file}", file=sys.stderr)
        return 1
    channels = get_session_channels(session)
    trades_channel = channels.get("trades")
    if trades_channel is None:
        print(f"[ERROR] Session '{args.session}' has no 'trades' channel configured", file=sys.stderr)
        return 1

    entries = list(load_dlq_entries(dlq_path, session=args.session))
    if not entries:
        print("[REPLAY] No DLQ entries for this session.")
        return 0

    entries = entries[: args.limit]
    attempted = sent_ok = skipped_sent = skipped_perm = skipped_payload = still_failed = 0

    print(
        f"[REPLAY] session={args.session} log_root={log_root} entries_to_attempt={len(entries)} include_perm={args.include_permanent} dry_run={args.dry_run}"
    )

    for entry in entries:
        message_key = entry.get("message_key")
        payload_wrapper = entry.get("payload")
        payload = _extract_payload(payload_wrapper)
        if not message_key or not payload:
            skipped_payload += 1
            continue

        record = idempotence_store.get(args.session, message_key)
        if record and record.status == IDEMP_STATUS_SENT:
            skipped_sent += 1
            continue
        if record and record.status == IDEMP_STATUS_FAILED_PERM and not args.include_permanent:
            skipped_perm += 1
            continue

        attempted += 1
        if args.dry_run:
            print(f"[REPLAY-DRY] would replay message_key={message_key}")
            continue

        status_code, message_id, error_body = post_discord_message(
            token, int(trades_channel), payload, dry_run=False
        )

        if status_code is None:
            still_failed += 1
            idempotence_store.upsert(args.session, message_key, IDEMP_STATUS_FAILED_TEMP)
            print(f"[REPLAY] network error for message_key={message_key}: {error_body}")
            continue
        if 200 <= status_code < 300:
            sent_ok += 1
            idempotence_store.upsert(
                args.session,
                message_key,
                IDEMP_STATUS_SENT,
                discord_message_id=message_id,
            )
            continue
        if status_code in TRANSIENT_STATUS_CODES:
            still_failed += 1
            idempotence_store.upsert(args.session, message_key, IDEMP_STATUS_FAILED_TEMP)
            print(
                f"[REPLAY] transient error status={status_code} message_key={message_key} body={error_body}"
            )
            continue
        still_failed += 1
        idempotence_store.upsert(args.session, message_key, IDEMP_STATUS_FAILED_PERM)
        print(
            f"[REPLAY] permanent error status={status_code} message_key={message_key} body={error_body}"
        )

    print(
        f"[REPLAY] done attempted={attempted} sent_ok={sent_ok} "
        f"skipped_sent={skipped_sent} skipped_perm={skipped_perm} skipped_payload={skipped_payload} "
        f"still_failed={still_failed}"
    )
    if args.dry_run:
        print("[REPLAY] dry run only; no messages sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
