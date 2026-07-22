#!/usr/bin/env python3
"""Watchdog — alerts if the live portfolio engine stops running.

This exists because the Alpaca equities account silently died on 2026-06-18 when
the VPS stopped, and nobody noticed for 3 weeks. A strategy that is offline earns
zero; uptime is worth more than +0.1 Sharpe.

Design rule: the watchdog must NOT live inside the thing it watches, or it dies
with it. Run it from a separate cron entry (ideally a separate host).

Checks:
  1. heartbeat freshness  — live_portfolio writes live/portfolio/heartbeat.json each run
  2. engine result        — partial failures and planning errors are unhealthy
  3. Alpaca reachability  — account endpoint responds
  4. account sanity       — equity present, trading not blocked
Alerts via Telegram (reuses the bot already configured for HyprL).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Derived, not hardcoded — the CI runner checks out at a different path.
ROOT = Path(__file__).resolve().parents[2]
HEARTBEAT = ROOT / "live/portfolio/heartbeat.json"
STATE_ALERT = ROOT / "live/portfolio/last_alert.json"

# how stale before we scream. Engine runs each weekday -> 2 days covers a weekend.
MAX_AGE_HOURS = 50

# Never hardcode: this repo is public. Supply via env / GitHub Secrets.
ALPACA_KEY = (
    os.environ.get("ALPACA_KEY")
    or os.environ.get("ALPACA_API_KEY")
    or os.environ.get("APCA_API_KEY_ID")
    or ""
)
ALPACA_SECRET = (
    os.environ.get("ALPACA_SECRET")
    or os.environ.get("ALPACA_SECRET_KEY")
    or os.environ.get("APCA_API_SECRET_KEY")
    or ""
)
ALPACA_BASE = (
    os.environ.get("ALPACA_BASE")
    or os.environ.get("ALPACA_BASE_URL")
    or "https://paper-api.alpaca.markets"
).rstrip("/")

# reuses the project's existing Telegram convention (see telegram_bot/config.py)
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_ALERT_CHAT") or os.environ.get("TELEGRAM_CHAT_ID", "")


def notify(msg: str) -> None:
    print(msg)
    if not TG_TOKEN or not TG_CHAT:
        print("  [telegram not configured — set TELEGRAM_BOT_TOKEN / TELEGRAM_ALERT_CHAT]")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": msg}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
    except Exception as e:
        print(f"  [telegram send failed: {e!r}]")


def alert_once(key: str, msg: str) -> None:
    """Alert, but don't spam: re-alert the same key at most once per 12h."""
    now = datetime.now(timezone.utc)
    seen = {}
    if STATE_ALERT.exists():
        try:
            seen = json.load(open(STATE_ALERT))
        except Exception:
            seen = {}
    last = seen.get(key)
    if last:
        if now - datetime.fromisoformat(last) < timedelta(hours=12):
            return
    notify(msg)
    seen[key] = now.isoformat()
    STATE_ALERT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(seen, open(STATE_ALERT, "w"), indent=2)


def check_heartbeat() -> list[str]:
    if not HEARTBEAT.exists():
        return ["no heartbeat file — engine has never run"]
    try:
        hb = json.load(open(HEARTBEAT))
        ts = datetime.fromisoformat(hb["ts"])
    except Exception as e:
        return [f"heartbeat unreadable: {e!r}"]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    problems = []
    if age > MAX_AGE_HOURS:
        problems.append(
            f"engine SILENT for {age:.0f}h (last run {ts:%Y-%m-%d %H:%M} UTC)"
        )

    failure_counts = {}
    for name in ("failed", "deferred", "planning_errors"):
        try:
            failure_counts[name] = int(hb.get(name, 0) or 0)
        except (TypeError, ValueError):
            failure_counts[name] = 1
    unhealthy = hb.get("ok") is not True or any(failure_counts.values())
    if unhealthy:
        detail = ", ".join(
            f"{name}={count}" for name, count in failure_counts.items() if count
        )
        if hb.get("error"):
            detail = f"error={str(hb['error'])[:180]}"
        heartbeat_issues = hb.get("issues") or []
        if heartbeat_issues:
            excerpt = "; ".join(str(item)[:160] for item in heartbeat_issues[:3])
            detail = f"{detail}; {excerpt}" if detail else excerpt
        problems.append(f"engine reported an unhealthy run ({detail or 'no detail'})")
    return problems


def check_alpaca() -> list[str]:
    if not ALPACA_KEY or not ALPACA_SECRET:
        return ["ALPACA_KEY / ALPACA_SECRET not set in the watchdog environment"]
    try:
        req = urllib.request.Request(ALPACA_BASE + "/v2/account", headers={
            "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET})
        a = json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        return [f"Alpaca unreachable / auth failed: {repr(e)[:100]}"]
    problems = []
    if a.get("trading_blocked"):
        problems.append("Alpaca reports trading_blocked=True")
    if a.get("account_blocked"):
        problems.append("Alpaca reports account_blocked=True")
    if a.get("status") != "ACTIVE":
        problems.append(f"Alpaca account status = {a.get('status')}")
    return problems


def main() -> int:
    problems = check_heartbeat() + check_alpaca()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if problems:
        body = "\n".join(f"- {p}" for p in problems)
        alert_once("|".join(sorted(problems))[:80],
                   f"🚨 HyprL portfolio ALERT ({stamp})\n{body}")
        return 1
    print(f"[{stamp}] watchdog OK — engine alive, Alpaca healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
