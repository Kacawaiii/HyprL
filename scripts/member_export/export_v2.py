"""Closed member v2 projection of published benchmark-relative forecast scores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft7Validator, FormatChecker

from .export import MAX_BYTES, PRIVATE, canonical, instant, walk
from .export import build as build_v1
from .producer import atomic
from .spec import AGENTS, BASELINE_URL, FROZEN_RATE_SHA256, MODES, PANELS, REPOSITORY, schema


def validate(value, name):
    Draft7Validator(schema(name), format_checker=FormatChecker()).validate(value)
    if PRIVATE.search(json.dumps(value, ensure_ascii=False)):
        raise ValueError("PRIVATE_MEMBER_CONTENT")
    return value


def series(value):
    grouped = defaultdict(list)
    for r in value["forecasts"]:
        grouped[(r["agent"], r["mode"], r["variant"], r["horizon"])].append(r)
    result = []
    base = value["baseline"]
    for agent in AGENTS:
        for mode in MODES:
            failures = sum(f["agent"] == agent and f["mode"] == mode for f in value["failures"])
            for variant, horizon in PANELS:
                rows = sorted(
                    grouped[(agent, mode, variant, horizon)],
                    key=lambda r: (r["session"], r["decision_time"], r["symbol"]),
                )
                error = reference = 0.0
                hits, points = [], []
                for r in rows:
                    p, y = r["probability"], r["outcome"]
                    error += (p - y) ** 2
                    if base is not None:
                        reference += (base["probability"] - y) ** 2
                    hits.append(int((p >= 0.5) == bool(y)))
                    points.append(
                        {
                            "at": r["scored_at"],
                            "session": r["session"],
                            "symbol": r["symbol"],
                            "n": len(hits),
                            "hits": sum(hits),
                            "brier": error / len(hits),
                            "brier_base_rate": reference / len(hits) if base else None,
                            "brier_skill": 1 - error / reference if reference else None,
                            "rolling_hit_rate": sum(hits[-20:]) / len(hits[-20:]),
                        }
                    )
                last = points[-1] if points else {}
                result.append(
                    {
                        "agent": agent,
                        "mode": mode,
                        "variant": variant,
                        "horizon": horizon,
                        "n": len(rows),
                        "hits": sum(hits),
                        **{
                            k: last.get(k)
                            for k in ("brier", "brier_base_rate", "brier_skill", "rolling_hit_rate")
                        },
                        "failures": failures,
                        "model_failures": failures,
                        "trade_n": sum(r["trade_hit"] is not None for r in rows),
                        "trade_hits": sum(r["trade_hit"] is True for r in rows),
                        "points": points,
                    }
                )
    return result


def build(value, *, now=None, allow_synthetic=False):
    validate(value, "handoff")
    now = now or datetime.now(timezone.utc)
    cutoff = instant(value["generated_at"])
    if value["provenance"] != "observed" and not allow_synthetic:
        raise ValueError("SYNTHETIC_MEMBER_DATA")
    if cutoff > now or (now - cutoff).total_seconds() > 86400:
        raise ValueError("STALE_HANDOFF")
    # Retain the original source, brief, quarterly-comparability and vocabulary guards.
    common = {k: value[k] for k in ("provenance", "generated_at", "briefs", "radar", "claims")}
    revenues = json.loads(json.dumps(value["revenue"]))
    for r in revenues:
        for p in r["periods"]:
            accession = p.pop("accession")
            if accession.replace("-", "") not in p["source"]["url"]:
                raise ValueError("FILING_ACCESSION_MISMATCH")
    checked = build_v1(
        {
            **common,
            "schema_version": 1,
            "revenue": revenues,
            "calendar": [],
            "forecasts": [],
            "failures": [],
        },
        now=now,
        allow_synthetic=allow_synthetic,
    )
    for row in walk(value):
        for key in ("published_at", "available_at", "scored_at", "received_at"):
            if key in row and instant(row[key]) > cutoff:
                raise ValueError("UNAVAILABLE_MEMBER_EVIDENCE")
    base = value["baseline"]
    if base and (
        base["sha256"] != FROZEN_RATE_SHA256
        or base["url"] != BASELINE_URL
        or base["n"] != 1214
        or base["successes"] != 590
        or base["probability"] != (base["successes"] + 1) / (base["n"] + 2)
    ):
        raise ValueError("FROZEN_BASE_RATE_MISMATCH")
    reveals = {}
    for r in value["reveals"]:
        suffix = "-intraday" if r["mode"] == "INTRADAY" else ""
        key = r["session"] + suffix
        filename = r["agent"] + ".json" if r["status"] == "REVEALED" else "README.md"
        c = r["commitment"]
        if (
            r["id"] != key + "-" + r["agent"]
            or r["id"] in reveals
            or r["file"] != "reveals/" + key + "/" + filename
            or r["url"] != REPOSITORY + "/blob/" + r["ledger_commit"] + "/" + r["file"]
            or c["url"] != REPOSITORY + "/blob/" + c["ledger_commit"] + "/forward-duel-ledger.jsonl"
            or not instant(c["committed_at"])
            <= instant(c["published_at"])
            < instant(r["published_at"])
            <= instant(r["available_at"])
            <= cutoff
            or (r["status"] == "REVEALED" and r["sha256"] != c["sha256"])
        ):
            raise ValueError("INVALID_SESSION_REVEAL")
        reveals[r["id"]] = r
    seen = set()
    for r in value["forecasts"]:
        key = tuple(r[k] for k in ("agent", "mode", "variant", "horizon", "session", "symbol"))
        proof = reveals.get(r["reveal_id"])
        if (
            key in seen
            or (r["variant"], r["horizon"]) not in PANELS
            or not proof
            or proof["status"] != "REVEALED"
            or proof["sha256"] != r["decision_sha256"]
            or any(proof[k] != r[k] for k in ("agent", "mode", "session"))
            or not instant(r["decision_time"]) < instant(proof["published_at"])
            or instant(r["decision_time"]) >= instant(r["scored_at"])
            or (
                r["entry_at"]
                and not instant(r["decision_time"])
                < instant(r["entry_at"])
                <= instant(r["scored_at"])
            )
        ):
            raise ValueError("UNPROVEN_FORECAST")
        seen.add(key)
    seen = set()
    for r in value["failures"]:
        key = (r["agent"], r["mode"], r["session"])
        proof = reveals.get(r["reveal_id"])
        if (
            key in seen
            or not proof
            or proof["status"] != "NO_DECISION"
            or any(proof[k] != r[k] for k in ("agent", "mode", "session"))
        ):
            raise ValueError("UNPROVEN_FAILURE")
        seen.add(key)
    seen = set()
    for r in value["calendar"]:
        key = (r["kind"], r["symbol"], r["date"])
        if key in seen or (r["scheduled_at"] and r["scheduled_at"][:10] != r["date"]):
            raise ValueError("INVALID_CALENDAR_PRECISION")
        if r["source"] and instant(r["source"]["available_at"]) < instant(
            r["source"]["published_at"]
        ):
            raise ValueError("NONCAUSAL_CALENDAR_SOURCE")
        seen.add(key)
    screener = []
    by_symbol = {r["symbol"]: r for r in value["revenue"]}
    for r in checked["screener"]:
        screener.append({**r, "periods": by_symbol[r["symbol"]]["periods"]})
    return validate({**value, "series": series(value), "screener": screener}, "snapshot")


def publish(value, directory):
    raw = canonical(value)
    if len(raw) > MAX_BYTES:
        raise ValueError("MEMBER_SNAPSHOT_TOO_LARGE")
    atomic(Path(directory) / "member-v2.json", raw, 0o440)
    return hashlib.sha256(raw).hexdigest()
