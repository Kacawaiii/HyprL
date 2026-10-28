"""Factory contract, private files, durable at-most-once reservations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
PLATFORMS = ("instagram", "facebook", "x", "youtube", "tiktok")
DISCLAIMER = "Information, pas un conseil en investissement"
FACTORY_REVISION = "f11d90c2191991b1b64b8c6c75e2b714bed733ee"
PRODUCT_SCRIPT_SHA256 = "5023a8ca5ffd4bacb9f52e53b34bf26aeb34960faac09fc857caa1727ad1880e"


class Blocked(Exception):
    """Only fixed, non-sensitive messages may cross the CLI boundary."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise Blocked("private_directory_permissions")
    return path


def read_regular(path, limit, *, private=False):
    """No symlinks, devices, hardlinks or oversized reads; secrets require 0600."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or info.st_size > limit
                or (private and stat.S_IMODE(info.st_mode) != 0o600)
            ):
                raise Blocked("invalid_input_permissions_or_size")
            raw = handle.read(limit + 1)
            if len(raw) > limit:
                raise Blocked("input_too_large")
            return raw
    except OSError:
        raise Blocked("input_unavailable") from None


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise Blocked("timezone_required")
    return result


def session(now):
    local = now.astimezone(NY)
    # Never catch up yesterday; never compete with the trading-job blackout.
    if time(11, 30) <= now.astimezone(UTC).time().replace(tzinfo=None) < time(17, 15):
        return None
    if local.time().replace(tzinfo=None) < time(17, 20):
        return None
    import pandas_market_calendars as mcal

    if mcal.get_calendar("NYSE").schedule(start_date=local.date(), end_date=local.date()).empty:
        return None
    return local.date().isoformat()


@dataclass(frozen=True)
class Bundle:
    video: bytes
    metadata: dict
    metadata_sha256: str
    day: str

    @property
    def sha256(self):
        return self.metadata["sha256"]

    @property
    def caption(self):
        m = self.metadata
        return f"{m['title']}\n\n{m['description']}\n\n{' '.join(m['hashtags'])}"


def load_bundle(root, factory_state, day, *, probe=None):
    """Accept only a completed, digest-bound daily factory-v2 delivery."""
    bundle = Path(root) / ("daily-" + day)
    if bundle.is_symlink() or Path(root).is_symlink():
        raise Blocked("symlink_bundle")
    raw = read_regular(bundle / "metadata.json", 100_000)
    m = json.loads(raw)
    video = read_regular(bundle / "video.mp4", 8_999_999)
    receipt = json.loads(read_regular(bundle / "delivery.json", 10_000))
    daily = json.loads(read_regular(Path(factory_state) / (day + ".json"), 10_000))
    if (
        m.get("schema") != "hyprl-video-v2"
        or not video
        or m.get("sha256") != digest(video)
        or m.get("bytes") != len(video)
        or m.get("disclaimer") != DISCLAIMER
        or m.get("kind") not in ("evergreen", "market", "weekly", "duel", "verdict")
        or any(
            r.get("status") != "delivered" or r.get("sha256") != m["sha256"]
            for r in (receipt, daily)
        )
        or daily.get("day") != day
        or daily.get("id") != "daily-" + day
    ):
        raise Blocked("factory_checks_or_receipts_invalid")
    provenance = m.get("provenance")
    if not isinstance(provenance, dict):
        raise Blocked("invalid_factory_provenance_shape")
    if m["kind"] in ("market", "weekly", "duel"):
        expected = (
            "observed_member_v2_brief_calendar" if m["kind"] == "market" else "observed_member_v2"
        )
        if provenance.get("kind") != expected or not re.fullmatch(
            r"[a-f0-9]{64}", provenance.get("snapshot_sha256", "")
        ):
            raise Blocked("non_observed_daily_metadata")
        instant(provenance["generated_at"])
        if m["kind"] == "market":
            instant(provenance["brief_published_at"])
    elif m["kind"] == "evergreen":
        if (
            provenance.get("kind") != "operator_approved_product_script"
            or provenance.get("revision") != "v2"
            or provenance.get("sha256") != PRODUCT_SCRIPT_SHA256
        ):
            raise Blocked("unapproved_product_script")
    elif (
        provenance.get("kind") != "research_blockage" or provenance.get("study_status") != "BLOCKED"
    ):
        raise Blocked("unapproved_research_verdict")
    if (
        type(m.get("width")) is not int
        or type(m.get("height")) is not int
        or (m["width"], m["height"]) != (720, 1280)
        or type(m.get("duration_seconds")) not in (int, float)
        or not math.isfinite(m["duration_seconds"])
        or not 25 <= m["duration_seconds"] <= 45.1
        or type(m.get("synthetic_voice")) is not bool
        or not isinstance(m.get("hashtags"), list)
        or any(not isinstance(h, str) or not re.fullmatch(r"#[\w]+", h) for h in m["hashtags"])
        or not isinstance(m.get("captions"), list)
        or not 3 <= len(m["captions"]) <= 8
        or any(not isinstance(c, str) or not c.strip() for c in m["captions"])
        or any(not isinstance(m.get(k), str) or not m[k].strip() for k in ("title", "description"))
        or DISCLAIMER not in m["description"]
    ):
        raise Blocked("invalid_factory_metadata")
    # Verify the actual media again; metadata is not a substitute for a probe.
    if probe is None:

        def probe(path):
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_name,width,height,codec_type",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return json.loads(result.stdout)

    measured = probe(bundle / "video.mp4")
    streams = measured["streams"]
    duration = float(measured["format"]["duration"])
    if (
        len(streams) != 1
        or streams[0].get("codec_name") != "h264"
        or streams[0].get("codec_type") != "video"
        or (streams[0].get("width"), streams[0].get("height")) != (720, 1280)
        or not math.isfinite(duration)
        or not 25 <= duration <= 45.1
        or abs(duration - m["duration_seconds"]) > 0.1
    ):
        raise Blocked("video_probe_mismatch")
    # A change during the probe cannot authorize different bytes.
    if digest(read_regular(bundle / "video.mp4", 8_999_999)) != m["sha256"]:
        raise Blocked("factory_video_changed")
    return Bundle(video, m, digest(raw), day)


class Ledger:
    """Reserve before network I/O. Interrupted/uncertain attempts never auto-replay."""

    def __init__(self, directory):
        directory = private_dir(directory)
        path = directory / "ledger.sqlite3"
        if path.exists() or path.is_symlink():
            read_regular(path, 100_000_000, private=True)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                platform TEXT NOT NULL, sha TEXT NOT NULL, day TEXT NOT NULL,
                metadata_sha TEXT NOT NULL, account_sha TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL, remote_id TEXT,
                PRIMARY KEY(platform, sha), UNIQUE(platform, day));
            CREATE TABLE IF NOT EXISTS budgets (grant_sha TEXT PRIMARY KEY, used INTEGER NOT NULL);
        """)
        self.db.commit()

    def reserve(self, platform, bundle, account, *, monthly_cap=25):
        with self.db:
            old = self.db.execute(
                "SELECT status FROM posts WHERE platform=? AND sha=?", (platform, bundle.sha256)
            ).fetchone()
            if old:
                if old[0] == "posted":
                    return False
                raise Blocked("previous_attempt_needs_reconciliation")
            if self.db.execute(
                "SELECT 1 FROM posts WHERE platform=? AND day=?", (platform, bundle.day)
            ).fetchone():
                raise Blocked("daily_post_limit")
            count = self.db.execute(
                "SELECT count(*) FROM posts WHERE platform=? AND day LIKE ?",
                (platform, bundle.day[:7] + "%"),
            ).fetchone()[0]
            if count >= monthly_cap:
                raise Blocked("monthly_post_limit")
            self.db.execute(
                "INSERT INTO posts VALUES (?,?,?,?,?,'pending','reserved',NULL)",
                (
                    platform,
                    bundle.sha256,
                    bundle.day,
                    bundle.metadata_sha256,
                    digest(account.encode()),
                ),
            )
        return True

    def status(self, platform, sha):
        row = self.db.execute(
            "SELECT status FROM posts WHERE platform=? AND sha=?", (platform, sha)
        ).fetchone()
        return row[0] if row else None

    def checkpoint(self, platform, sha, stage, remote_id=None):
        if remote_id is not None and not re.fullmatch(r"[A-Za-z0-9_.~:-]{1,256}", str(remote_id)):
            raise Blocked("invalid_remote_identity")
        with self.db:
            self.db.execute(
                "UPDATE posts SET stage=?, remote_id=COALESCE(?,remote_id) WHERE platform=? AND sha=?",
                (stage, remote_id, platform, sha),
            )

    def finish(self, platform, sha, remote_id):
        self.checkpoint(platform, sha, "confirmed", remote_id)
        with self.db:
            self.db.execute(
                "UPDATE posts SET status='posted' WHERE platform=? AND sha=?", (platform, sha)
            )

    def consume(self, grant_sha, maximum):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO budgets VALUES (?,0)", (grant_sha,))
            cursor = self.db.execute(
                "UPDATE budgets SET used=used+1 WHERE grant_sha=? AND used < ?",
                (grant_sha, maximum),
            )
            if cursor.rowcount != 1:
                raise Blocked("authorization_request_budget_exhausted")

    def rows(self):
        return [
            dict(zip(("platform", "sha256", "day", "status", "stage", "remote_id"), r))
            for r in self.db.execute(
                "SELECT platform,sha,day,status,stage,remote_id FROM posts ORDER BY day,platform"
            )
        ]
