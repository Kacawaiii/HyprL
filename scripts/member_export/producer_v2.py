"""Offline member v2 adapter. Reads existing live scores and public Git blobs only."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .export import MAX_BYTES, canonical, instant
from .export_v2 import build, publish
from .producer import AGENTS, atomic, git, iso, read_json, research_digest
from .producer import produce as produce_v1
from .spec import BASELINE_URL, FROZEN_RATE_SHA256, REPOSITORY, baseline_projection

ROOT = Path(__file__).resolve().parents[2]
LEDGER_FILE = "forward-duel-ledger.jsonl"


def public_blob(config, commit, name):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("INVALID_PUBLIC_COMMIT")
    git(
        config["ledger"],
        "merge-base",
        "--is-ancestor",
        commit,
        "refs/remotes/origin/ledger/forward-duel",
    )
    return git(config["ledger"], "show", commit + ":" + name)


def reveal(config, session, family, mode, expected, now, excluded, expected_spec=None):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) or mode not in ("REGULAR_0905", "INTRADAY"):
        raise ValueError("INVALID_FORWARD_SESSION")
    key = session + ("-intraday" if mode == "INTRADAY" else "")
    marker = Path(config["reveal_dir"]) / (key + ".json")
    if not marker.is_file():
        excluded["score_without_public_reveal"] += 1
        return None, None
    proof = read_json(marker)
    agent = AGENTS[family]
    name = "reveals/" + key + "/" + agent + ".json"
    readme = "reveals/" + key + "/README.md"
    status = "REVEALED" if name in proof.get("files", {}) else "NO_DECISION"
    if proof.get("state") != "DELIVERED":
        excluded["score_without_public_reveal"] += 1
        return None, None
    commit = proof["reveal_commit"]
    raw_readme = public_blob(config, commit, readme)
    if hashlib.sha256(raw_readme).hexdigest() != proof["files"].get(readme):
        raise ValueError("REVEAL_MANIFEST_MISMATCH")
    if "| " + agent + " | " + status + " | " + expected + " |" not in raw_readme.decode():
        # UNPUBLISHED and a committed failure are never a forecast reveal.
        excluded["score_without_public_reveal"] += 1
        return None, None
    name = name if status == "REVEALED" else readme
    raw = public_blob(config, commit, name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != proof["files"][name] or (status == "REVEALED" and digest != expected):
        raise ValueError("ORIGINAL_PREIMAGE_MISMATCH")
    receipt = read_json(
        Path(config["forward_dir"]).parent / "commitments/forward" / (key + ".json")
    )
    line = receipt["line"]
    committed = public_blob(config, receipt["ledger_commit"], LEDGER_FILE)
    if (
        receipt.get("state") != "PUBLISHED"
        or receipt.get("route") != "discord"
        or receipt.get("messages", 0) < 1
        or research_digest(line) != receipt["line_sha256"]
        or line.get("kind") != "forward"
        or line.get("key") != key
        or line["hashes"].get(family) != expected
        or (expected_spec is not None and line["protocol_spec_hash"] != expected_spec)
        or sum(json.loads(v) == line for v in committed.splitlines()) != 1
    ):
        raise ValueError("PRIOR_PUBLICATION_UNPROVEN")
    published = instant(
        git(config["ledger"], "show", "-s", "--format=%cI", commit).decode().strip()
    )
    original_commit_at = instant(
        git(config["ledger"], "show", "-s", "--format=%cI", receipt["ledger_commit"])
        .decode()
        .strip()
    )
    opening = datetime.combine(
        datetime.fromisoformat(session).date(), time(9, 30), ZoneInfo("America/New_York")
    )
    closing = datetime.combine(
        datetime.fromisoformat(session).date(), time(16, 15), ZoneInfo("America/New_York")
    )
    if not (
        instant(line["committed_at"]) <= instant(receipt["published_at"]) < opening
        and original_commit_at < opening
        and closing <= published <= instant(proof["delivered_at"]) <= now
    ):
        raise ValueError("PUBLICATION_CLOCK_INVALID")
    result = {
        "id": key + "-" + agent,
        "agent": agent,
        "session": session,
        "mode": mode,
        "status": status,
        "ledger_commit": commit,
        "file": name,
        "sha256": digest,
        "url": REPOSITORY + "/blob/" + commit + "/" + name,
        "published_at": iso(published),
        "available_at": proof["delivered_at"],
        "commitment": {
            "ledger_commit": receipt["ledger_commit"],
            "url": REPOSITORY + "/blob/" + receipt["ledger_commit"] + "/" + LEDGER_FILE,
            "committed_at": line["committed_at"],
            "published_at": receipt["published_at"],
            "sha256": expected,
        },
    }
    return result, json.loads(raw) if status == "REVEALED" else None


def produce(config, *, now=None):
    now = now or datetime.now(timezone.utc)
    if config.get('calendar_config'):
        config = {**config, 'calendar_file': read_json(config['calendar_config'])['calendar']}
    old, status = produce_v1(config, now=now)
    excluded = Counter(status["excluded"])
    excluded.pop("forecast_event_and_commitment_incompatible_v1", None)
    excluded.pop("calendar_unattested_time_or_source", None)
    value = {
        **old,
        "schema_version": 2,
        "forecasts": [],
        "failures": [],
        "reveals": [],
        "baseline": None,
    }
    score = read_json(
        Path(config["forward_dir"]) / "score-runs" / (status["score_sha256"] + ".json")
    )
    rate = read_json(ROOT / "docs/artifacts/replay_duel_test_freeze_v15.json")[
        "frozen_dev_base_rate"
    ]
    if research_digest(rate) != FROZEN_RATE_SHA256:
        raise ValueError("FROZEN_BASE_RATE_MISMATCH")
    source = score['base_rate_source']
    if source != 'FROZEN_DEV_BASE_RATE:' + FROZEN_RATE_SHA256:
        mapping = baseline_projection()
        projected = mapping['source_rate']
        if (mapping['target_rate_sha256'] != FROZEN_RATE_SHA256
                or research_digest(projected) != mapping['source_rate_sha256']
                or source != 'FROZEN_DEV_BASE_RATE:' + mapping['source_rate_sha256']
                or set(projected) != set(rate)
                or {k:v for k,v in projected.items() if k != 'spec_hash'} != {k:v for k,v in rate.items() if k != 'spec_hash'}):
            raise ValueError('FROZEN_BASE_RATE_MISMATCH')
        status['baseline_projection'] = {'source_sha256': mapping['source_rate_sha256'],
                                         'target_sha256': FROZEN_RATE_SHA256,
                                         'identical_observed_outcomes': True}
    pbase = (rate["successes"] + 1) / (rate["n"] + 2)
    if score["primary_endpoint"]["base_rate"] != pbase:
        raise ValueError("SCORE_BASE_RATE_MISMATCH")
    value["baseline"] = {
        "probability": pbase,
        "n": rate["n"],
        "successes": rate["successes"],
        "panel": rate["panel"],
        "smoothing": "laplace_1_1",
        "sha256": FROZEN_RATE_SHA256,
        "url": BASELINE_URL,
    }
    proofs = {}
    trades = {}
    for r in score["trades"]:
        # News trades have a separate holding decision, not a probability at that horizon.
        if r["group"] == "ab":
            k = tuple(r[x] for x in ("family", "mode", "session", "asset", "horizon"))
            if k in trades:
                raise ValueError("DUPLICATE_TRADE_SCORE")
            trades[k] = r["hit"]
    cache = {}
    for r in score["rows"]:
        if r["provenance"] != "LIVE_FORWARD" or r["family"] not in AGENTS:
            raise ValueError("SCORE_NOT_OBSERVED")
        variant = {"ab": "a_v1_as_live", "cd": "c_news_small_mid"}.get(r["group"])
        if variant is None or r["panel"] != variant + ":" + r["horizon"]:
            raise ValueError("SCORE_PANEL_MISMATCH")
        key = (r["session"], r["family"], r["mode"], r["decision_hash"], r["source_spec_hash"])
        if key not in cache:
            cache[key] = reveal(config, *key[:3], key[3], now, excluded, key[4])
        proof, original = cache[key]
        if not proof or proof["status"] != "REVEALED":
            excluded["withheld_forecast_rows"] += 1
            continue
        if (
            research_digest(original) != r["decision_hash"]
            or original["family"] != r["family"]
            or original["provenance"] != "LIVE_FORWARD"
            or original["snapshot_hash"] != r["snapshot_hash"]
            or original["decision_time"] != r["decision_time"]
        ):
            raise ValueError("SCORE_DECISION_MISMATCH")
        rows = (
            original["output"]["analyst"]["views"]
            if r["group"] == "ab"
            else original["output"]["news"]["forecasts"]
        )
        matches = [v for v in rows if v["asset"] == r["asset"] and v["horizon"] == r["horizon"]]
        variant, horizon = r["panel"].split(":")
        if (
            len(matches) != 1
            or matches[0]["p_outperform"] != r["p"]
            or horizon != r["horizon"]
            or r["outcome"] != int(r["rel"] > 0)
            or r["outperform"] != bool(r["outcome"])
            or abs(r["brier"] - (r["p"] - r["outcome"]) ** 2) > 1e-12
        ):
            raise ValueError("ORIGINAL_FORECAST_MISMATCH")
        proofs[proof["id"]] = proof
        trade_key = tuple(r[x] for x in ("family", "mode", "session", "asset", "horizon"))
        value["forecasts"].append(
            {
                "agent": AGENTS[r["family"]],
                "variant": variant,
                "mode": r["mode"],
                "horizon": horizon,
                "session": r["session"],
                "symbol": r["asset"],
                "event": "outperform_spy",
                "probability": r["p"],
                "outcome": r["outcome"],
                "decision_time": r["decision_time"],
                "entry_at": r["entry_at"],
                "scored_at": score["scored_at"],
                "decision_sha256": r["decision_hash"],
                "snapshot_sha256": r["snapshot_hash"],
                "score_sha256": status["score_sha256"],
                "source_spec_sha256": r["source_spec_hash"],
                "source_code_sha256": r["source_code_hash"],
                "reveal_id": proof["id"],
                "trade_hit": trades.get(trade_key),
            }
        )
    reverse_agents = {v: k for k, v in AGENTS.items()}
    for r in old["failures"]:
        family = reverse_agents[r["agent"]]
        original = read_json(
            Path(config["forward_dir"]) / r["session"] / (family + "-failure.json")
        )
        proof, _ = reveal(
            config, r["session"], family, "REGULAR_0905", research_digest(original), now, excluded
        )
        if proof:
            proofs[proof["id"]] = proof
            value["failures"].append(
                {
                    "agent": r["agent"],
                    "mode": "REGULAR_0905",
                    "session": r["session"],
                    "status": "MODEL_FAILURE",
                    "code": original["error"],
                    "reveal_id": proof["id"],
                }
            )
    value["reveals"] = sorted(proofs.values(), key=lambda r: r["id"])
    calendar = read_json(config["calendar_file"])
    if calendar.get("schema") == "agent-v2-quality-calendar-v2":
        received = calendar["received_at"]
        if instant(received) > now:
            raise ValueError("FUTURE_CALENDAR")
        seen = set()
        for r in calendar["events"]:
            kind = {"FOMC": "fed", "EARNINGS": "earnings"}.get(r["kind"])
            if kind is None or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T(?:00:00:00|23:59:59)Z", r["at"]
            ):
                excluded["unsupported_calendar_marker"] += 1
                continue
            key = (kind, r["symbol"], r["at"][:10])
            if key in seen:
                continue
            seen.add(key)
            value["calendar"].append(
                {
                    "kind": kind,
                    "symbol": r["symbol"],
                    "date": r["at"][:10],
                    "precision": "date",
                    "scheduled_at": None,
                    "source": None,
                    "evidence": {
                        "kind": "observed_day_marker",
                        "sha256": research_digest(r),
                        "dataset_sha256": research_digest(calendar),
                        "received_at": iso(instant(received)),
                    },
                }
            )
    else:
        excluded["unsupported_calendar_shape_v2"] += 1
    snapshot = build(value, now=now)
    status.update(
        counts={
            k: len(value[k])
            for k in (
                "briefs",
                "radar",
                "calendar",
                "forecasts",
                "failures",
                "reveals",
                "revenue",
                "claims",
            )
        },
        excluded={k: v for k, v in sorted(excluded.items()) if v},
        series_with_points=sum(bool(r["points"]) for r in snapshot["series"]),
    )
    status["status"] = "PARTIAL" if status["excluded"] else "COMPLETE"
    return value, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if (
            git(ROOT, "rev-parse", "HEAD").decode().strip() != args.expected_commit
            or git(ROOT, "status", "--porcelain").strip()
        ):
            raise ValueError("PRODUCER_CODE_NOT_PINNED")
        config = read_json(args.config)
        for key in ("handoff", "output_dir", "status_file"):
            path = Path(config[key]).resolve()
            if path == ROOT or ROOT in path.parents:
                raise ValueError("OUTPUT_MUST_BE_OUTSIDE_GIT")
        # Refuse an unsafe status destination before replacing either artifact.
        destinations = (
            Path(config["handoff"]),
            Path(config["output_dir"]) / "member-v2.json",
            Path(config["status_file"]),
        )
        for path in destinations:
            parent = path.parent
            if (
                path.is_symlink()
                or parent.is_symlink()
                or not parent.is_dir()
                or parent.stat().st_uid != os.getuid()
                or parent.stat().st_mode & 0o022
            ):
                raise ValueError("UNSAFE_MEMBER_OUTPUT")
        handoff = Path(config["handoff"])
        with (handoff.parent / "producer-v2.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value, status = produce(config)
            snapshot = build(value)
            if max(len(canonical(value)), len(canonical(snapshot))) > MAX_BYTES:
                raise ValueError("OUTPUT_TOO_LARGE")
            atomic(handoff, canonical(value))
            status["snapshot_sha256"] = publish(snapshot, config["output_dir"])
            atomic(config["status_file"], canonical(status))
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception:
        print("MEMBER_V2_PRODUCER_REFUSED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
