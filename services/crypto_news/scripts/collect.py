#!/usr/bin/env python
"""Collect allowlisted sources or replay deterministic offline fixtures."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from crypto_news.collection import CollectionService, FetchResult, load_fixture_directory
from crypto_news.evidence import EvidenceStore
from crypto_news.journal import Journal
from crypto_news.sources import DEFAULT_SOURCE_REGISTRY
from crypto_news.transport import HttpTransport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", type=Path, help="offline fixture directory")
    mode.add_argument("--source", help="enabled source_id to fetch over HTTPS")
    parser.add_argument("--journal", type=Path, help="append-only SQLite journal")
    parser.add_argument("--evidence-dir", type=Path, help="raw content-addressed evidence")
    parser.add_argument(
        "--user-agent",
        help="explicit identification required for live source collection",
    )
    return parser


def _live_fetch(args: argparse.Namespace) -> tuple[FetchResult, ...]:
    source = DEFAULT_SOURCE_REGISTRY.get(args.source)
    if not source.enabled or source.endpoint_url is None:
        raise ValueError("source is not enabled for live collection")
    if not args.user_agent:
        raise ValueError("--user-agent is required for live collection")
    transport = HttpTransport(user_agent=args.user_agent)
    first_seen_at = datetime.now(timezone.utc)
    return (
        transport.fetch(
            source_id=source.source_id,
            endpoint_url=source.endpoint_url,
            first_seen_at=first_seen_at,
        ),
    )


def run(args: argparse.Namespace) -> dict[str, int | bool]:
    if (args.journal is None) != (args.evidence_dir is None):
        raise ValueError("--journal and --evidence-dir must be supplied together")
    if args.source is not None and args.journal is None:
        raise ValueError("live collection requires persistent journal and evidence paths")

    with ExitStack() as stack:
        ephemeral = args.journal is None
        if ephemeral:
            temporary_root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            journal_path = temporary_root / "journal.sqlite3"
            evidence_path = temporary_root / "evidence"
        else:
            journal_path = args.journal
            evidence_path = args.evidence_dir
            journal_path.parent.mkdir(parents=True, exist_ok=True)
        fetches = (
            load_fixture_directory(args.fixture)
            if args.fixture is not None
            else _live_fetch(args)
        )
        journal = Journal(journal_path)
        stack.callback(journal.close)
        service = CollectionService(
            journal=journal,
            evidence_store=EvidenceStore(evidence_path),
            registry=DEFAULT_SOURCE_REGISTRY,
        )
        totals = {
            "duplicate_count": 0,
            "ephemeral": ephemeral,
            "fetch_count": len(fetches),
            "network_access": args.source is not None,
            "parsed_count": 0,
            "rejected_count": 0,
            "stored_count": 0,
        }
        for fetch in fetches:
            batch = service.collect(fetch)
            totals["duplicate_count"] += batch.duplicate_count
            totals["parsed_count"] += batch.parsed_count
            totals["rejected_count"] += batch.rejected_count
            totals["stored_count"] += batch.stored_count
        if not journal.verify_chain():
            raise RuntimeError("journal verification failed after collection")
        return totals


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"collection failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
