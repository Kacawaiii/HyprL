#!/usr/bin/env python3
"""
Check per-ticker heartbeat JSONs under live/logs and alert if stale/missing.

Example:
    python scripts/ops/check_heartbeat.py \
      --root live/logs \
      --max-age-min 90 \
      --symbols NVDA,MSFT,AMD,META,QQQ \
      --webhook $SLACK_WEBHOOK_URL
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check live runner heartbeats and alert if stale or missing.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument(
        "--max-age-min",
        type=int,
        default=90,
        help="Max allowed age in minutes before a heartbeat is considered stale (default: 90).",
    )
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated list of expected symbols (e.g., NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--webhook",
        help="Webhook URL (Slack/Discord/other JSON endpoint). If omitted, only prints to stdout/stderr.",
    )
    parser.add_argument(
        "--send-ok",
        action="store_true",
        help="Also send a webhook notification when all heartbeats are OK.",
    )
    return parser.parse_args()


def iter_heartbeat_paths(root: Path) -> List[Path]:
    """
    Support both legacy layout live_<sym>/heartbeat.json and dated layout live_<sym>/YYYY-MM-DD/heartbeat.json.
    """
    paths: set[Path] = set()
    paths.update(root.glob("live_*/heartbeat.json"))
    paths.update(root.glob("live_*/*/heartbeat.json"))
    return sorted(paths)


def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    """POST payload as JSON to webhook. Best-effort (logs warning on failure)."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if "text" in payload and "content" not in payload:
        payload["content"] = payload["text"]
    req = urllib.request.Request(webhook_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except urllib.error.URLError as exc:
        print(f"[WARN] Failed to send webhook: {exc}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Unexpected error sending webhook: {exc}", file=sys.stderr)


def parse_iso(ts: str) -> Optional[datetime]:
    """Parse ISO8601 timestamps, including trailing 'Z'."""
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        if ts.endswith("Z"):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None


def load_heartbeat(path: Path) -> Tuple[str, Optional[datetime]]:
    """
    Load heartbeat JSON and return (ticker_upper, timestamp).
    If ticker missing in JSON, infer from directory name.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse heartbeat at {path}: {exc}", file=sys.stderr)
        ticker = path.parent.name.replace("live_", "").upper()
        return ticker, None

    ticker = str(data.get("ticker") or path.parent.name.replace("live_", "")).upper()
    ts_str = data.get("ts_iso") or data.get("timestamp") or data.get("ts") or data.get("as_of")
    if not isinstance(ts_str, str):
        return ticker, None

    ts = parse_iso(ts_str)
    return ticker, ts


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    max_age_min = args.max_age_min
    root = args.root

    heartbeat_paths = iter_heartbeat_paths(root)
    found: Dict[str, Dict[str, Any]] = {}
    for path in heartbeat_paths:
        ticker, ts = load_heartbeat(path)
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_min = (now - ts).total_seconds() / 60.0 if ts else None
        info = {
            "ticker": ticker,
            "path": str(path),
            "timestamp": ts.isoformat() if ts else None,
            "age_min": age_min,
        }
        existing = found.get(ticker)
        replace = False
        if existing is None:
            replace = True
        elif age_min is not None:
            existing_age = existing.get("age_min")
            if existing_age is None or age_min < existing_age:
                replace = True
        found[ticker] = info if replace else existing

    expected_symbols: List[str] = []
    if args.symbols:
        expected_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    stale: List[Dict[str, Any]] = []
    missing: List[str] = []
    parse_issues: List[Dict[str, Any]] = []

    for ticker, info in found.items():
        ts = info["timestamp"]
        age_min = info["age_min"]
        if ts is None or age_min is None:
            parse_issues.append(info)
        elif age_min > max_age_min:
            stale.append(info)

    for sym in expected_symbols:
        if sym not in found:
            missing.append(sym)

    alert = bool(stale or missing or parse_issues)
    if alert:
        text_parts = ["[ALERT][Heartbeat] Issues detected."]
        if stale:
            stale_str = ", ".join(
                f"{x['ticker']}@{x['age_min']:.1f}min" for x in stale if x["age_min"] is not None
            )
            text_parts.append(f"Stale: {stale_str}")
        if missing:
            missing_str = ", ".join(missing)
            text_parts.append(f"Missing: {missing_str}")
        if parse_issues:
            parse_str = ", ".join(f"{x['ticker']} (parse error)" for x in parse_issues)
            text_parts.append(f"Parse issues: {parse_str}")

        text = " | ".join(text_parts)
        print(text, file=sys.stderr)
        payload = {
            "text": text,
            "kind": "heartbeat",
            "severity": "ALERT",
            "max_age_min": max_age_min,
            "time_utc": now.isoformat(),
            "stale": stale,
            "missing": missing,
            "parse_issues": parse_issues,
        }
        if args.webhook:
            send_webhook(args.webhook, payload)
        sys.exit(1)

    summary = ", ".join(
        f"{t}~{info['age_min']:.1f}min"
        for t, info in sorted(found.items())
        if info["age_min"] is not None
    )
    text = f"[OK][Heartbeat] All heartbeats fresh (<= {max_age_min} min). {summary}"
    print(text)
    if args.webhook and args.send_ok:
        payload = {
            "text": text,
            "kind": "heartbeat",
            "severity": "OK",
            "max_age_min": max_age_min,
            "time_utc": now.isoformat(),
            "heartbeats": list(found.values()),
        }
        send_webhook(args.webhook, payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
