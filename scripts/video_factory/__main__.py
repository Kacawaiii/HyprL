"""Entry point. Configuration and outputs live outside the public repository."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .content import blocked_verdict, evergreen
from .delivery import alert, atomic_json
from .factory import daily, guard_clock, produce
from .study import PLAN, daily_events, load_candles, plan_hash, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("batch", "daily", "blocked-verdict", "plan", "study", "alert")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    config = json.loads(args.config.read_text()) if args.config else {}
    try:
        if args.command == "alert":
            return 0 if alert(config["notify"], "service") else 1
        if config.get("expected_commit"):
            checkout = Path(__file__).resolve().parents[2]
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != config["expected_commit"]:
                raise ValueError("unexpected checkout revision")
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", "scripts/video_factory"],
                cwd=checkout,
                check=True,
            )
        guard_clock(datetime.now(timezone.utc))
        if args.command == "plan":
            if not args.output:
                parser.error("plan requires --output")
            atomic_json(args.output, {"plan": PLAN, "sha256": plan_hash()})
            result = {"status": "plan_recorded", "sha256": plan_hash()}
        elif args.command == "study":
            if not args.manifest or not args.output:
                parser.error("study requires --manifest and --output")
            bars, step = load_candles(args.manifest)
            events, missing = daily_events(bars, step)
            result = summarize(events, missing)
            result["manifest_sha256"] = (
                __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest()
            )
            atomic_json(args.output, result)
            result = {
                "status": result["status"],
                "days": len(events),
                "excluded_days": len(missing),
            }
        elif args.command == "batch":
            for index in range(5):
                guard_clock(datetime.now(timezone.utc))
                result = produce(
                    config, f"evergreen-v2-{index + 1:02}", evergreen(index), args.publish
                )
                print(json.dumps(result), flush=True)
            return 0
        elif args.command == "blocked-verdict":
            result = produce(config, "verdict-01", blocked_verdict(), args.publish)
        else:
            result = daily(config, publish=args.publish)
        print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        # Class names only: exceptions may embed private webhook URLs or paths.
        delivered = alert(config["notify"], type(exc).__name__) if config.get("notify") else False
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "alert_delivered": delivered}
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
