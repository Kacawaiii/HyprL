"""Validate an editorial handoff, compute forecast metrics and atomically publish.

No network, model, account, trading-store or holdout access. Input is a dedicated
producer handoff, never a raw store. Source files are never copied to the site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from jsonschema import Draft7Validator, FormatChecker

SCHEMAS = Path(__file__).resolve().parents[2] / "apps/hyprl-site/docs/member"
MAX_BYTES = 2 * 1024 * 1024
PRIVATE = re.compile(
    r"(?:/home/|/srv/|[A-Z]:\\|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|\b(?:sk-|Bearer\s)|\b(?:buy|sell|acheter|vendre|achetez|vendez|stop.loss|position.sizing|take.profit|paper.pnl)\b)",
    re.I,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(value, name):
    schema = json.loads((SCHEMAS / (name + "-v1.schema.json")).read_text())
    Draft7Validator(schema, format_checker=FormatChecker()).validate(value)
    if PRIVATE.search(json.dumps(value, ensure_ascii=False)):
        raise ValueError("private/advisory vocabulary refused")
    return value


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def build(value, *, now=None, allow_synthetic=False):
    value = validate(value, "handoff")
    now = now or datetime.now(timezone.utc)
    if value["provenance"] != "observed" and not allow_synthetic:
        raise ValueError("synthetic handoff refused")
    cutoff = instant(value["generated_at"])
    if cutoff > now or (now - cutoff).total_seconds() > 86400:
        raise ValueError("handoff is future or stale")
    for row in walk(value):
        if "url" in row:
            u = urlsplit(row["url"])
            if u.username or u.password or not u.hostname or u.port not in (None, 443):
                raise ValueError("invalid public source")
        for key in ("available_at", "published_at", "scored_at", "verified_at"):
            if key in row and instant(row[key]) > cutoff:
                raise ValueError("unavailable evidence")
        if (
            "available_at" in row
            and "published_at" in row
            and instant(row["available_at"]) < instant(row["published_at"])
        ):
            raise ValueError("noncausal source")
    for brief in value["briefs"]:
        if any(
            instant(fact["source"]["available_at"]) > instant(brief["published_at"])
            for fact in brief["facts"]
        ):
            raise ValueError("noncausal brief")
    seen = set()
    grouped = defaultdict(list)
    for row in value["forecasts"]:
        c = row["commitment"]
        identity = (c["agent"], c["variant"], c["session"], c["symbol"])
        if identity in seen or hashlib.sha256(canonical(c)).hexdigest() != row["sha256"]:
            raise ValueError("duplicate or invalid commitment")
        seen.add(identity)
        if not (
            instant(c["committed_at"])
            < instant(c["reveal_at"])
            <= instant(row["scored_at"])
            <= cutoff
        ):
            raise ValueError("unrevealed forecast")
        if (
            instant(row["scored_at"]) < instant(row["source"]["available_at"])
            or c["committed_at"][:10] != c["session"]
        ):
            raise ValueError("noncausal scoring")
        grouped[(c["agent"], c["variant"])].append(row)
    series = []
    for (agent, variant), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (r["scored_at"], r["sha256"]))
        error = baseline = 0.0
        hits = []
        distribution = [1.0]
        points = []
        for row in rows:
            c = row["commitment"]
            p, b, y = float(c["probability"]), float(c["base_rate"]), row["outcome"]
            error += (p - y) ** 2
            baseline += (b - y) ** 2
            hits.append(int((p >= 0.5) == bool(y)))
            q = b if p >= 0.5 else 1 - b
            distribution = (
                [distribution[0] * (1 - q)]
                + [
                    distribution[i] * (1 - q) + distribution[i - 1] * q
                    for i in range(1, len(distribution))
                ]
                + [distribution[-1] * q]
            )
            points.append(
                {
                    "at": row["scored_at"],
                    "n": len(hits),
                    "brier_skill": 1 - error / baseline if baseline else None,
                    "rolling_hit_rate": sum(hits[-20:]) / len(hits[-20:]),
                    "luck_tail": min(1.0, max(0.0, math.fsum(distribution[sum(hits) :]))),
                }
            )
        series.append({"agent": agent, "variant": variant, "points": points})
    screener = []
    seen_symbols = set()
    for row in value["revenue"]:
        periods = row["periods"]
        if (
            row["symbol"] in seen_symbols
            or len(periods) != 3
            or len({p["currency"] for p in periods}) != 1
            or [p["end"] for p in periods] != sorted({p["end"] for p in periods})
        ):
            raise ValueError("three comparable ordered revenue periods required")
        for i, period in enumerate(periods):
            start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
            if not 70 <= (end - start).days <= 110 or (
                i and start != date.fromisoformat(periods[i - 1]["end"]) + timedelta(days=1)
            ):
                raise ValueError("incomparable quarterly intervals")
            if urlsplit(period["source"]["url"]).hostname != "www.sec.gov" or not urlsplit(
                period["source"]["url"]
            ).path.startswith("/Archives/edgar/data/"):
                raise ValueError("SEC archive provenance required")
        seen_symbols.add(row["symbol"])
        amounts = [Decimal(p["revenue"]) for p in periods]
        if min(amounts[:2]) <= 0:
            continue  # Undefined growth is unavailable, never manufactured.
        g0 = (amounts[1] / amounts[0] - 1) * 100
        g1 = (amounts[2] / amounts[1] - 1) * 100
        screener.append(
            {
                **row,
                "growth_previous": float(g0),
                "growth_latest": float(g1),
                "acceleration_pp": float(g1 - g0),
            }
        )
    return validate(
        {**value, "series": series, "screener": sorted(screener, key=lambda r: r["symbol"])},
        "snapshot",
    )


def publish(value, directory):
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("dedicated existing output directory required")
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o022:
        raise ValueError("unsafe output directory")
    raw = canonical(value)
    if len(raw) > MAX_BYTES:
        raise ValueError("snapshot too large")
    fd, temporary = tempfile.mkstemp(prefix=".member-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
            os.fchmod(file.fileno(), 0o440)
        os.replace(temporary, directory / "member-v1.json")
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    # No synthetic switch in the production CLI.
    if args.handoff.is_symlink() or args.handoff.stat().st_size > MAX_BYTES:
        raise ValueError("unsafe handoff")
    value = build(json.loads(args.handoff.read_text()))
    digest = publish(value, args.output_dir)
    print(
        json.dumps(
            {"sha256": digest, "forecasts": len(value["forecasts"]), "series": len(value["series"])}
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("MEMBER_EXPORT_REFUSED", file=sys.stderr)
        raise SystemExit(1) from None
