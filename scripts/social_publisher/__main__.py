"""python -m scripts.social_publisher --help (no secrets on argv)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .core import Blocked, Ledger, atomic_json, private_dir, read_regular
from .media import serve
from .ops import check_revision, notify, stage_units
from .runner import daily, default_config, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-revision")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-config")
    run = sub.add_parser("daily")
    run.add_argument("--dry-run", action="store_true")
    sub.add_parser("ledger")
    sub.add_parser("alert")
    sub.add_parser("serve-media")
    stage = sub.add_parser("stage-units")
    stage.add_argument("--output", type=Path, required=True)
    reconcile = sub.add_parser("reconcile-posted")
    reconcile.add_argument(
        "--platform", choices=("instagram", "facebook", "x", "youtube", "tiktok"), required=True
    )
    reconcile.add_argument("--sha256", required=True)
    reconcile.add_argument("--remote-id", required=True)
    reconcile.add_argument("--confirmation", choices=("I_VERIFIED_REMOTE_POST",), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "alert":
            return 0 if notify() else 1
        if args.command == "init-config":
            if not args.config or args.config.exists():
                raise Blocked("new_config_path_required")
            private_dir(args.config.parent)
            atomic_json(args.config, default_config())
            print('{"status":"created_disabled_dry_run"}')
            return 0
        if args.command == "stage-units":
            if (
                not args.config
                or not args.expected_revision
                or not re.fullmatch(r"[a-f0-9]{40}", args.expected_revision)
            ):
                raise Blocked("config_and_full_revision_required")
            print(
                json.dumps(
                    {
                        "status": "staged_only",
                        "units": stage_units(args.output, args.config, args.expected_revision),
                    }
                )
            )
            return 0
        if args.expected_revision:
            check_revision(args.expected_revision)
        if not args.config:
            raise Blocked("config_required")
        config = json.loads(read_regular(args.config, 64_000, private=True))
        validate_config(config)
        if args.command == "daily":
            result = daily(config, dry_run=args.dry_run)
            print(json.dumps(result, sort_keys=True))
            # OnFailure sends a single alert, including for SIGKILL/timeouts.
            # A manual invocation emits no messages; operator can invoke `alert`.
            return 1 if result["status"] == "blocked" else 0
        if args.command == "serve-media":
            if not args.expected_revision:
                raise Blocked("media_service_revision_required")
            serve(Path(config["state_dir"]) / "media")
            return 0
        ledger = Ledger(config["state_dir"])
        try:
            if args.command == "ledger":
                print(json.dumps({"posts": ledger.rows()}, sort_keys=True))
            elif args.command == "reconcile-posted":
                if not re.fullmatch(r"[a-f0-9]{64}", args.sha256) or not any(
                    r["platform"] == args.platform and r["sha256"] == args.sha256
                    for r in ledger.rows()
                ):
                    raise Blocked("reconciliation_reservation_required")
                ledger.finish(args.platform, args.sha256, args.remote_id)
                print('{"status":"operator_reconciled"}')
        finally:
            ledger.db.close()
        return 0
    except Blocked as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}))
        return 1
    except Exception:
        print('{"status":"blocked","reason":"invalid_config_input_or_runtime_failure"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
