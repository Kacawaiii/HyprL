"""Operator CLI: outbound worker/gateway only. No deployment or inbound port."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from .model import Alert, AlertError
from .producers import publish, radar_event, sec_story
from .router import Router
from .store import Store, create_key, private_file


def credential(path, *names):
    if path:
        from io import StringIO

        from dotenv import dotenv_values

        with os.fdopen(private_file(Path(path).expanduser()), "r") as source:
            values = dotenv_values(stream=StringIO(source.read(65536)), interpolate=False)
    else:
        values = os.environ
    return next((values[name] for name in names if values.get(name)), None)


def read_json(path):
    if path == "-":
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    else:
        with os.fdopen(private_file(Path(path).expanduser()), "rb") as source:
            raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise AlertError("INPUT_TOO_LARGE")
    return json.loads(raw)


def configured(args):
    if not args.state or not args.key or not args.policy:
        raise AlertError("CONFIG_REQUIRED")
    policy = read_json(args.policy)
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {"schema_version", "allowed_customers", "telegram_chats", "guild_id", "role_id"}
        or policy["schema_version"] != 1
        or type(policy["guild_id"]) is not int
        or policy["guild_id"] <= 0
        or type(policy["role_id"]) is not int
        or policy["role_id"] <= 0
        or not isinstance(policy["allowed_customers"], list)
        or not policy["allowed_customers"]
        or not isinstance(policy["telegram_chats"], dict)
    ):
        raise AlertError("INVALID_POLICY")
    import re

    if any(
        not isinstance(c, str) or not re.fullmatch(str(policy["guild_id"]) + r"/\d+", c)
        for c in policy["allowed_customers"]
    ):
        raise AlertError("INVALID_POLICY_CUSTOMERS")
    if set(policy["telegram_chats"]) - set(policy["allowed_customers"]) or any(
        not isinstance(chats, list)
        or any(not isinstance(c, str) or not re.fullmatch(r"-?\d{1,20}", c) for c in chats)
        for chats in policy["telegram_chats"].values()
    ):
        raise AlertError("INVALID_POLICY_CHATS")
    # Production state and key must be outside this public checkout.
    checkout = Path(__file__).resolve().parents[2]
    for value in (args.state, args.key, args.policy):
        target = Path(value).expanduser().resolve()
        if target == checkout or checkout in target.parents:
            raise AlertError("PRIVATE_STATE_INSIDE_GIT")
    store = Store(
        Path(args.state).expanduser(),
        Path(args.key).expanduser(),
        allowed_customers=policy["allowed_customers"],
        telegram_chats=policy["telegram_chats"],
    )
    return store, policy


async def worker(store, *, once=False, telegram_token=None):
    router = Router(store, telegram_token=telegram_token)
    last_prune = 0
    while True:
        active = await router.run_once()
        now = time.time()
        if now - last_prune > 3600:
            store.prune(before=now - 30 * 86400)
            last_prune = now
        if once:
            return
        if not active:
            await asyncio.sleep(1)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default=os.environ.get("HYPRL_ALERT_STATE"))
    p.add_argument("--key", default=os.environ.get("HYPRL_ALERT_KEY"))
    p.add_argument("--policy", default=os.environ.get("HYPRL_ALERT_POLICY"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-key")
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true")
    w.add_argument("--telegram-env", default=os.environ.get("HYPRL_ALERT_TELEGRAM_ENV"))
    g = sub.add_parser("gateway")
    g.add_argument("--register", action="store_true", help="Upsert /alertes in this guild only")
    g.add_argument("--discord-env", default=os.environ.get("HYPRL_ALERT_DISCORD_ENV"))
    for name in ("add", "ingest", "status", "dlq", "pause", "remove", "api-token"):
        s = sub.add_parser(name)
        if name == "ingest":
            s.add_argument("--customer")
            s.add_argument("--adapter", choices=("event", "radar", "sec"), default="event")
        else:
            s.add_argument("--customer", required=True)
        if name in ("add", "ingest"):
            s.add_argument(
                "--file",
                default="-",
                help="Private JSON file or stdin; never secret command arguments",
            )
        if name in ("pause", "remove"):
            s.add_argument("--destination", required=True)
        if name == "pause":
            s.add_argument("--resume", action="store_true")
        if name == "api-token":
            s.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init-key":
            if not args.key:
                raise AlertError("CONFIG_REQUIRED")
            checkout = Path(__file__).resolve().parents[2]
            key = Path(args.key).expanduser().resolve()
            if key == checkout or checkout in key.parents:
                raise AlertError("PRIVATE_STATE_INSIDE_GIT")
            create_key(key)
            print('{"key_created":true}')
            return 0
        store, policy = configured(args)
        if args.command == "worker":
            asyncio.run(
                worker(
                    store,
                    once=args.once,
                    telegram_token=credential(args.telegram_env, "TELEGRAM_BOT_TOKEN"),
                )
            )
        elif args.command == "gateway":
            from .discord_gateway import run_gateway

            token = credential(args.discord_env, "DISCORD_BOT_TOKEN", "DISCORD_TOKEN", "BOT_TOKEN")
            if not token:
                raise AlertError("DISCORD_NOT_CONFIGURED")
            logging.disable(logging.CRITICAL)
            run_gateway(
                store,
                token=token,
                guild_id=policy["guild_id"],
                role_id=policy["role_id"],
                register=args.register,
            )
        elif args.command == "add":
            body = read_json(args.file)
            destination = store.add(args.customer, **body)
            print(json.dumps({"destination": destination}))
        elif args.command == "ingest":
            body = read_json(args.file)
            now = time.time()
            if args.adapter == "radar":
                alert = radar_event(body, available_at=body["available_at"])
            elif args.adapter == "sec":
                alert = sec_story(body)
            else:
                alert = Alert.parse(body)
            count = (
                store.enqueue(args.customer, alert, now=now)
                if args.customer
                else publish(store, alert, now=now)
            )
            print(json.dumps({"queued": count}))
        elif args.command in ("status", "dlq"):
            print(
                json.dumps(
                    {
                        "destinations": store.list(args.customer),
                        "jobs": store.jobs(
                            args.customer, state="dead" if args.command == "dlq" else None
                        ),
                    }
                )
            )
        elif args.command == "pause":
            store.pause(args.customer, args.destination, not args.resume)
            print('{"updated":true}')
        elif args.command == "remove":
            store.remove(args.customer, args.destination)
            print('{"removed":true}')
        elif args.command == "api-token":
            with os.fdopen(private_file(Path(args.output).expanduser(), create=True), "w") as out:
                out.write(store.register(args.customer) + "\n")
            print('{"token_written":true}')
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        # No argparse body, traceback, private filename, URL or provider body.
        print("STATUS: BLOCKED — alert router configuration or operation failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
