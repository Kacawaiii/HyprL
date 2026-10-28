"""Private configuration; explicit publish required for every Discord mutation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .delivery import alert
from .factory import SLOTS, batch, run
from .render import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("batch", "run", "verify", "alert"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--slot", choices=SLOTS, default="daily")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    config = json.loads(args.config.read_text())
    try:
        if args.command == "alert":
            return 0 if alert(config["notify"]) else 1
        checkout = Path(__file__).resolve().parents[2]
        expected = config.get("expected_commit")
        if args.publish and not expected:
            raise ValueError("published automation requires a pinned revision")
        if expected:
            actual = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if actual != expected:
                raise ValueError("unexpected carousel checkout revision")
            subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "scripts/carousel_factory",
                    "scripts/video_factory",
                ],
                cwd=checkout,
                check=True,
            )
        if args.command == "verify":
            if not args.bundle:
                parser.error("verify requires --bundle")
            meta = verify(args.bundle)
            result = {"status": "verified", "content_sha256": meta["content_sha256"]}
        elif args.command == "batch":
            result = batch(config, publish=args.publish)
        else:
            result = run(config, args.slot, publish=args.publish)
        print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        # Exception strings can contain a private path or webhook credential.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
