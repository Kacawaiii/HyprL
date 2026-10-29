"""Read observed outputs offline, project allowed facts, then use the v1 exporter.

No acquisition, model calls, accounts, prices or research execution imports.
Missing provenance is counted, never reconstructed from file modification times.
The private configuration contains paths; public outputs contain only v1 fields.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime, time, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .export import MAX_BYTES, PRIVATE, build, canonical, instant, publish

REPOSITORY = "https://github.com/Kacawaiii/HyprL"
AGENTS = {"opus": "HypR-A1", "gpt": "HypR-E1"}
MODEL = re.compile(r"\b(?:claude|gpt(?:-\w+)*|opus|sonnet|anthropic|openai|gemini)\b", re.I)
SYMBOL = re.compile(r"[A-Z][A-Z0-9.\-]{0,14}\Z")
SESSION = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def research_digest(value):
    """The original research serialization escapes Unicode, unlike member v1."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path, limit=16 * 1024 * 1024):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("INPUT_TOO_LARGE")
    return json.loads(raw)


def readonly_db(path):
    db = sqlite3.connect(Path(path).resolve(strict=True).as_uri() + "?mode=ro", uri=True, timeout=5)
    db.execute("PRAGMA query_only=ON")
    return db


def public_text(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 2000
        and not re.search(r"[<>\x00-\x1f]", value)
        and not PRIVATE.search(value)
        and not MODEL.search(value)
    )


def fact(row, cutoff, excluded, *, kind=None):
    """Keep source timestamps and hashes; updated is never original publication."""
    if kind is None:
        source = row.get("source", "")
        if row.get("primary") is not True:
            excluded["radar_not_primary"] += 1
            return None
        if source.startswith("fed:"):
            kind = "fed"
        elif source == "sec:8k":
            kind = "sec_8k"
        else:
            excluded["radar_unsupported_kind"] += 1
            return None
    published, received = row.get("published_at"), row.get("received_at")
    if not published:
        excluded["radar_publication_unknown"] += 1
        return None
    if not received or not instant(published) <= instant(received) <= cutoff:
        excluded["noncausal_or_future_fact"] += 1
        return None
    url = row.get("url", "")
    u = urlsplit(url)
    expected_host = "www.sec.gov" if kind.startswith("sec_") else "www.federalreserve.gov"
    if kind.startswith("sec_") or row.get("source", "").startswith("fed:"):
        if u.hostname != expected_host:
            excluded["source_host_mismatch"] += 1
            return None
    if (
        u.scheme != "https"
        or not u.hostname
        or u.username
        or u.password
        or u.query
        or u.fragment
        or u.port not in (None, 443)
        or not public_text(row.get("headline"))
        or not HEX.fullmatch(row.get("headline_hash", ""))
    ):
        excluded["unsafe_fact"] += 1
        return None
    symbols = sorted({s for s in row.get("symbols", []) if SYMBOL.fullmatch(s)})
    result = {
        "kind": kind,
        "symbol": symbols[0] if len(symbols) == 1 else None,
        "summary": row["headline"],
        "source": {
            "url": url,
            "sha256": row["headline_hash"],
            "published_at": iso(instant(published)),
            "available_at": iso(instant(received)),
        },
    }
    # URL and prose guards also apply to every whole source projection.
    if PRIVATE.search(json.dumps(result)) or MODEL.search(result["summary"]):
        excluded["unsafe_fact"] += 1
        return None
    return result


def radar(db, now, excluded):
    rows = db.execute("""SELECT body FROM evidence WHERE kind='news' AND
        (json_extract(body,'$.source') LIKE 'fed:%' OR json_extract(body,'$.source')='sec:8k')
        ORDER BY seq DESC LIMIT 10000""").fetchall()
    result, seen = [], set()
    for (body,) in rows:
        value = fact(json.loads(body), now, excluded)
        if value is None:
            continue
        key = (value["source"]["url"], value["source"]["sha256"])
        if key not in seen:
            seen.add(key)
            result.append(value)
    result.sort(key=lambda r: (r["source"]["available_at"], r["source"]["url"]), reverse=True)
    excluded["radar_over_limit"] += max(0, len(result) - 500)
    return result[:500]


def briefs(db, directory, now, excluded):
    """Factual projections of delivered briefs, joined to their recorded report.

    Brief v1 cannot carry the complete Discord prose. Only matching, sourced
    headlines in its supported topics are exported; no regeneration or LLM.
    """
    directory = Path(directory)
    ledger = directory / "deliveries.jsonl"
    if not ledger.exists():
        excluded["brief_delivery_ledger_missing"] += 1
        return []
    if ledger.is_symlink() or ledger.stat().st_size > MAX_BYTES:
        raise ValueError("UNSAFE_BRIEF_LEDGER")
    delivered = {}
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if (
            row.get("status") == "SENT"
            and row.get("messages", 0) > 0
            and row.get("sent") == row["messages"]
            and instant(row["at"]) <= now
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}-(?:semaine|weekend)", row["brief"])
        ):
            delivered[row["brief"]] = row
    results = {}
    for key, receipt in sorted(delivered.items())[-90:]:
        meta = read_json(directory / (key + ".json"))
        cutoff = instant(meta["at"])
        published = instant(receipt["at"])
        if meta["id"] != key or not cutoff <= published <= now:
            excluded["brief_identity_or_clock_unproven"] += 1
            continue
        row = db.execute(
            "SELECT body,at FROM evidence WHERE kind='report' AND key=? ORDER BY seq DESC LIMIT 1",
            (meta["report"]["key"],),
        ).fetchone()
        if (
            row is None
            or instant(row[1]) > cutoff
            or instant(row[1]) != instant(meta["report"]["stored_at"])
        ):
            excluded["brief_report_unavailable"] += 1
            continue
        report = json.loads(row[0])
        if report.get("live") is not True:
            raise ValueError("BRIEF_REPORT_NOT_OBSERVED")
        observed_hash = hashlib.sha256(json.dumps(report.get("markdown", "")).encode()).hexdigest()[
            :16
        ]
        if observed_hash != meta["report"]["hash"]:
            raise ValueError("BRIEF_REPORT_HASH_MISMATCH")
        events = report.get('events', [])
        facts = []
        for item in meta.get("items", []):
            # Topical labels alone do not make an item official. The source URL
            # and attribution remain visible; only these supported topics fit v1.
            headline = item["headline"]
            kind = None
            if item.get("topic") == "fed" and re.search(
                r"\b(?:Fed|Federal Reserve|FOMC)\b", headline, re.I
            ):
                kind = "fed"
            if item.get("topic") == "inflation" and re.search(
                r"\b(?:CPI|Consumer Price Index)\b", headline, re.I
            ):
                kind = "cpi"
            matches = [s for e in events for s in e.get('stories', []) if s.get('headline') == headline]
            if not matches:
                grouped = [e for e in events if e.get('headline') == headline]
                if len(grouped) == 1:
                    # Export the observed source headlines, never attribute the
                    # grouped editorial wording to an invented source.
                    matches = grouped[0].get('stories', [])
            eligible = False
            seen_sources = set()
            for story in matches:
                source_kind = kind
                if story.get('primary') is True:
                    if story.get('source', '').startswith('fed:'):
                        source_kind = 'fed'
                    elif story.get('source') == 'sec:8k':
                        source_kind = 'sec_8k'
                    elif (urlsplit(story.get('url', '')).hostname == 'www.bls.gov'
                          and story.get('macro_kind') in ('cpi', 'nfp')):
                        source_kind = story['macro_kind']
                if source_kind is None:
                    continue
                eligible = True
                identity = (story.get('url'), story.get('headline_hash'))
                if identity in seen_sources:
                    continue
                seen_sources.add(identity)
                value = fact(story, cutoff, excluded, kind=source_kind)
                if value:
                    facts.append(value)
            if not eligible:
                excluded["brief_item_without_v1_source_projection"] += 1
        if facts:
            day = key[:10]
            results[day] = {"date": day, "published_at": iso(published), "facts": facts[:100]}
        else:
            excluded["brief_without_eligible_facts"] += 1
    return [results[k] for k in sorted(results, reverse=True)]


def git(ledger, *args):
    result = subprocess.run(
        ["git", "-C", str(ledger), *args], capture_output=True, timeout=20, check=True
    )
    return result.stdout


def forward(config, now, excluded):
    base = Path(config["forward_dir"])
    value = read_json(base / "scores.json")
    if value.get("provenance") != "LIVE_FORWARD" or value.get("evidence_class") != "LIVE_FORWARD":
        raise ValueError("FORWARD_NOT_OBSERVED")
    if instant(value["scored_at"]) > now:
        raise ValueError("FUTURE_SCORE")
    score_hash = research_digest(value)
    if read_json(base / "score-runs" / (score_hash + ".json")) != value:
        raise ValueError("SCORE_RUN_MISMATCH")
    # p_outperform(horizon) and an immutable whole-decision preimage cannot be
    # relabelled as v1 close_above_open / per-forecast salted commitments.
    excluded["forecast_event_and_commitment_incompatible_v1"] += len(value["rows"])
    failures = []
    for family, group in value.get("failures", {}).items():
        if family not in AGENTS:
            raise ValueError("UNKNOWN_FORWARD_FAMILY")
        for row in group["failed_sessions"]:
            day = row["session"]
            if not SESSION.fullmatch(day):
                raise ValueError("INVALID_SESSION")
            marker = Path(config["reveal_dir"]) / (day + ".json")
            if not marker.exists():
                excluded["failure_not_revealed"] += 1
                continue
            proof = read_json(marker)
            if proof.get("state") != "DELIVERED":
                excluded["failure_not_revealed"] += 1
                continue
            commit = proof["reveal_commit"]
            if not re.fullmatch(r"[a-f0-9]{40}", commit):
                raise ValueError("INVALID_REVEAL_COMMIT")
            ledger = config["ledger"]
            git(
                ledger,
                "merge-base",
                "--is-ancestor",
                commit,
                "refs/remotes/origin/ledger/forward-duel",
            )
            name = "reveals/" + day + "/README.md"
            raw = git(ledger, "show", commit + ":" + name)
            if hashlib.sha256(raw).hexdigest() != proof["files"][name]:
                raise ValueError("REVEAL_MANIFEST_MISMATCH")
            published = instant(git(ledger, "show", "-s", "--format=%cI", commit).decode().strip())
            available = instant(proof["delivered_at"])
            if not published <= available <= now:
                raise ValueError("REVEAL_CLOCK_INVALID")
            # Conservative gate even if a hand-authored marker is supplied.
            close = datetime.combine(
                datetime.fromisoformat(day).date(), time(16, 15), ZoneInfo("America/New_York")
            )
            if published < close:
                raise ValueError("REVEAL_BEFORE_REGULAR_CLOSE")
            original = read_json(base / day / (family + "-failure.json"))
            failure_hash = research_digest(original)
            label = AGENTS[family]
            if ("| " + label + " | NO_DECISION | " + failure_hash + " |") not in raw.decode():
                raise ValueError("FAILURE_NOT_IN_PUBLIC_REVEAL")
            if original["error"] != row["error"]:
                raise ValueError("FAILURE_CODE_MISMATCH")
            code = {
                "FORECAST_COVERAGE_INVALID": "invalid",
                "MODEL_USAGE_LIMIT": "missing",
                "INSUFFICIENT_CALL_WINDOW": "late",
                "DECISION_MISSING": "missing",
            }.get(row["error"], "unscorable")
            failures.append(
                {
                    "agent": label,
                    "variant": "LIVE_FORWARD",
                    "session": day,
                    "code": code,
                    "source": {
                        "url": REPOSITORY + "/blob/" + commit + "/" + name,
                        "sha256": failure_hash,
                        "published_at": iso(published),
                        "available_at": iso(available),
                    },
                }
            )
    return failures, score_hash


def produce(config, *, now=None):
    now = now or datetime.now(timezone.utc)
    excluded = Counter()
    value = {
        "schema_version": 1,
        "provenance": "observed",
        "generated_at": iso(now),
        **{
            k: []
            for k in ("briefs", "radar", "calendar", "forecasts", "failures", "revenue", "claims")
        },
    }
    with closing(readonly_db(config["radar_db"])) as db:
        db.execute("BEGIN")  # One consistent read transaction; never creates a store.
        value["radar"] = radar(db, now, excluded)
        value["briefs"] = briefs(db, config["brief_dir"], now, excluded)
    value["failures"], score_hash = forward(config, now, excluded)
    calendar = read_json(config["calendar_file"])
    if calendar.get("schema") not in ("agent-v2-quality-calendar-v2", "cockpit-calendar-v1"):
        raise ValueError("UNKNOWN_CALENDAR_SHAPE")
    # Current outputs carry risk guard markers / civil dates, not confirmed
    # publication hours and per-event source receipts. V1 cannot represent them.
    excluded["calendar_unattested_time_or_source"] += len(
        calendar.get("events", calendar.get("entries", []))
    )
    revenue = read_json(config["revenue_file"], limit=32 * 1024 * 1024)
    if not isinstance(revenue, dict):
        raise ValueError("UNKNOWN_REVENUE_SHAPE")
    excluded["revenue_missing_filing_intervals_and_receipts"] += len(revenue)
    build(value, now=now)  # Every original exporter guard remains authoritative.
    status = {
        "status": "PARTIAL" if any(excluded.values()) else "COMPLETE",
        "generated_at": iso(now),
        "counts": {
            k: len(value[k])
            for k in ("briefs", "radar", "calendar", "forecasts", "failures", "revenue", "claims")
        },
        "excluded": {k: v for k, v in sorted(excluded.items()) if v},
        "score_sha256": score_hash,
    }
    return value, status


def atomic(path, raw, mode=0o600):
    path = Path(path)
    if path.parent.is_symlink() or not path.parent.is_dir() or path.is_symlink():
        raise ValueError("UNSAFE_OUTPUT")
    if path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 0o022:
        raise ValueError("UNSAFE_OUTPUT_DIRECTORY")
    fd, temporary = tempfile.mkstemp(prefix=".producer-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        root = Path(__file__).resolve().parents[2]
        if args.expected_commit:
            if (
                git(root, "rev-parse", "HEAD").decode().strip() != args.expected_commit
                or git(root, "status", "--porcelain").strip()
            ):
                raise ValueError("PRODUCER_CODE_NOT_PINNED")
        config = read_json(args.config)
        for key in ("handoff", "output_dir", "status_file"):
            output = Path(config[key]).resolve()
            if output == root or root in output.parents:
                raise ValueError("OUTPUT_MUST_BE_OUTSIDE_GIT")
        handoff = Path(config["handoff"])
        with (handoff.parent / "producer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value, status = produce(config)
            snapshot = build(value)
            # Validate and size-check both complete artifacts before replacement.
            if max(len(canonical(value)), len(canonical(snapshot))) > MAX_BYTES:
                raise ValueError("OUTPUT_TOO_LARGE")
            atomic(handoff, canonical(value))
            status["snapshot_sha256"] = publish(snapshot, config["output_dir"])
            atomic(config["status_file"], canonical(status))
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception:
        # No raw source body, path, CLI error, credential or traceback in logs.
        print("MEMBER_PRODUCER_REFUSED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
