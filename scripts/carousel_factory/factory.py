"""Bounded carousel selection, durable idempotency and thirty-post retention."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.video_factory.delivery import atomic_json
from scripts.video_factory.factory import trading_day
from scripts.video_factory.market import digest

from .content import evergreen
from .data import (
    agenda_example,
    calendar_rows,
    documents,
    excerpt,
    load,
    reveal,
    weekly_calendar,
)
from .delivery import remove_owned, upload
from .editorial import edit
from .facts import select, to_post
from .public_study import verdict as public_verdict
from .render import render, verify

PARIS = ZoneInfo("Europe/Paris")
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
SAFE_ID = re.compile(r"carousel-v[12]-[a-z0-9][a-z0-9-]{0,100}")
SLOTS = {
    "daily": (18, 30),
    "calendar": (8, 0),
    "chance": (18, 45),
    "documents": (18, 50),
    "evergreen": (19, 15),
    "prepare": (11, 15),
}


def retain(config, limit=30):
    root, state = Path(config["output_dir"]), Path(config["state_dir"])
    owned = sorted(
        (
            p
            for p in root.iterdir()
            if SAFE_ID.fullmatch(p.name)
            and p.is_dir()
            and not p.is_symlink()
            and (p / "metadata.json").is_file()
        ),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )
    while len(owned) > limit:
        old = owned[0]
        receipt = state / f"{old.name}.json"
        if not receipt.exists():
            raise RuntimeError("retention blocked by undelivered carousel")
        remove_owned(receipt, config["webhook_env"])
        shutil.rmtree(old)
        owned.pop(0)


def produce(config, identifier, content, publish=False):
    if not SAFE_ID.fullmatch(identifier):
        raise ValueError("unsafe carousel identifier")
    root, state = Path(config["output_dir"]), Path(config["state_dir"])
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (root / "factory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = state / f"{identifier}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved.get("status") != "delivered":
                raise RuntimeError("uncertain delivery requires reconciliation")
            if saved.get("content_sha256") != digest(content):
                raise ValueError("delivered carousel differs from selection")
            return {"id": identifier, **saved}
        bundle = root / identifier
        if bundle.is_symlink():
            raise ValueError("symlink carousel refused")
        if bundle.exists():
            meta = verify(bundle)
            logo = (config.get("brand") or {}).get("logo")
            logo_digest = hashlib.sha256(Path(logo).read_bytes()).hexdigest() if logo else None
            if meta.get("style", {}).get("logo_sha256") != logo_digest:
                raise ValueError("existing carousel uses a different logo")
            if meta["content_sha256"] != digest(content):
                raise ValueError("existing carousel differs from selection")
        else:
            render(content, bundle, config["font_dir"], config.get("brand"))
        if not publish:
            return {"id": identifier, "status": "rendered", "kind": content["kind"]}
        result = upload(bundle, config["webhook_env"])
        saved = {
            **result,
            "content_sha256": digest(content),
            "kind": content["kind"],
            "delivered_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(receipt, saved)
        retain(config)
        return {"id": identifier, **saved}


def inputs(config, now):
    try:
        x = load(config["snapshot"], now)
    except FileNotFoundError:
        return None, []
    if config.get("calendar_config"):
        calendar = Path(json.loads(Path(config["calendar_config"]).read_text())["calendar"])
        config = {
            **config,
            "calendar_file": str(calendar),
            "calendar_manifest": str(
                calendar.parent.parent / "calendar-refresh/calendar-source-manifest.json"
            ),
        }
    return x, calendar_rows(x, config) if x else []


def product(index, x, rows, day, translations=None):
    proof = reveal(x) if x and index == 1 else None
    quote = excerpt(x, translations) if x and index == 3 else None
    if (index == 1 and not proof) or (index == 3 and not quote):
        return None
    return evergreen(
        index, proof=proof, excerpt=quote, agenda=agenda_example(rows, day) if index == 3 else None
    )


def batch(config, now=None, publish=False):
    now = now or datetime.now(UTC)
    day = now.astimezone(NY).date().isoformat()
    x, rows = inputs(config, now)
    selected, skipped = [], []
    for i in range(4):
        item = product(i, x, rows, day, config.get("excerpt_translations"))
        if item:
            selected.append((f"product-{i + 1:02}", item))
        else:
            skipped.append({"kind": f"product-{i + 1:02}", "reason": "no real example"})
    if x:
        # Preview the next session's week after Friday; absolute dates stay visible.
        calendar_day = date.fromisoformat(day)
        if calendar_day.weekday() > 4:
            calendar_day += timedelta(days=7 - calendar_day.weekday())
        cal = weekly_calendar(x, rows, calendar_day.isoformat())
        if cal:
            selected.append(("calendar-example-" + calendar_day.isoformat(), cal))
    for i, item in enumerate(documents(config.get("documents"), now)):
        if i == 0:
            selected.append(("document-example-" + digest(item)[:16], item))
    study = public_verdict() if config.get("public_study", True) else None
    if study:
        selected.append(("chance-example-" + digest(study)[:16], study))
    for kind in ("calendar", "document", "chance"):
        if not any(p["kind"] == kind for _, p in selected):
            skipped.append({"kind": kind, "reason": "no eligible observed evidence"})
    results = []
    for name, item in selected:
        # Content-addressed batch names permit a later real excerpt without overwrite.
        logo = (config.get("brand") or {}).get("logo")
        suffix = "-" + hashlib.sha256(Path(logo).read_bytes()).hexdigest()[:8] if logo else ""
        identifier = "carousel-v2-" + name + "-" + digest(item)[:12] + suffix
        results.append(produce(config, identifier, item, publish))
    return {"status": "completed", "posts": results, "skipped": skipped}


def run(config, slot, now=None, publish=False):
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("timezone required")
    local = now.astimezone(UTC if slot == "prepare" else PARIS)
    expected = datetime.combine(local.date(), time(*SLOTS[slot]), local.tzinfo)
    if not 0 <= (local - expected).total_seconds() < 600:
        return {"status": "skipped", "reason": "outside scheduled slot"}
    if (
        local.weekday() > 4
        or (slot == "calendar" and local.weekday() != 0)
        or (slot == "chance" and local.weekday() != 2)
        or (slot == "evergreen" and local.weekday() != 4)
    ):
        return {"status": "skipped", "reason": "outside scheduled weekday"}
    day = local.date().isoformat()
    if not trading_day(day):
        return {"status": "skipped", "reason": "NYSE closed"}
    # One slot owns the whole selection/upload. A second invocation reuses the
    # receipt before looking at refreshed upstream evidence.
    state = Path(config["state_dir"])
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state / "schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = state / f"slot-{slot}-{day}.json"
        if record.exists() and json.loads(record.read_text()).get("status") == "delivered":
            return {"status": "already_delivered", "slot": slot, "day": day}
        x, rows = inputs(config, now)
        item = None
        if slot in ("daily", "prepare"):
            facts = select(x, rows, day, config.get("documents"))
            if not facts:
                return {"status": "skipped", "reason": "no eligible facts today"}
            copy = edit(config, facts, now)
            if not copy:
                return {"status": "skipped", "reason": "editorial unavailable or rejected"}
            if slot == "prepare":
                return {"status": "prepared", "day": day}
            item = to_post(facts, copy)
        elif slot == "calendar" and x:
            item = weekly_calendar(x, rows, day)
        elif slot == "chance":
            item = public_verdict() if config.get("public_study", True) else None
            if item and (state / f"chance-{digest(item)}.json").exists():
                return {"status": "skipped", "reason": "study already delivered"}
        elif slot == "documents":
            candidates = documents(config.get("documents"), now)
            for candidate in candidates:
                identity = digest(candidate)
                if not (state / f"document-{identity}.json").exists():
                    item = candidate
                    break
            if item is None:
                return {"status": "skipped", "reason": "no new eligible official document"}
        elif slot == "evergreen":
            iso = date.fromisoformat(day).isocalendar()
            week_record = state / f"evergreen-{iso.year}-W{iso.week:02}.json"
            if week_record.exists():
                return {"status": "skipped", "reason": "weekly evergreen already reserved"}
            item = product(iso.week % 4, x, rows, day, config.get("excerpt_translations"))
        if item is None:
            return {"status": "skipped", "reason": "no eligible evidence"}
        if publish and slot == "evergreen":
            # Reserve before any upload. An uncertain delivery consumes the week's slot.
            atomic_json(week_record, {"status": "pending", "day": day})
        result = produce(config, f"carousel-v2-{slot}-{day}", item, publish)
        if publish:
            atomic_json(record, result)
            if slot == "evergreen":
                atomic_json(week_record, result)
            if item["kind"] == "chance":
                atomic_json(state / f"chance-{digest(item)}.json", result)
            if item["kind"] == "document":
                atomic_json(state / f"document-{digest(item)}.json", result)
        return result
