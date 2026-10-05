"""Write a new private demo-project config; read the client key from the environment.

No plaintext key is written or printed. Configuration is create-only, never overwritten.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from scripts.trading_lab.b2b.security import Configuration, PERMISSIONS, key_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default="demo")
    parser.add_argument("--key-id", default="local-client")
    parser.add_argument("--request-budget", type=int, default=2000)
    parser.add_argument("--job-budget", type=int, default=8)
    parser.add_argument("--expires-hours", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.expires_hours <= 168:
        parser.error("expiry must be 1..168 hours")
    payload = {"schema": "b2b-config-v1", "projects": {args.project: {
        "request_budget": args.request_budget, "job_budget": args.job_budget,
        "products": ["BTC-USD"], "sources": [], "exports": []}}, "keys": [{
        "key_id": args.key_id, "project_id": args.project, "key_sha256": key_hash(os.environ.get("HYPRL_B2B_KEY", "")),
        "permissions": sorted(PERMISSIONS), "enabled": True,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=args.expires_hours)).isoformat()}]}
    Configuration(payload)
    path = Path(args.output)
    # Caller chooses an existing private directory, outside version control.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
