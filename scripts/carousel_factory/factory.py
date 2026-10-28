"""Bounded carousel selection, durable idempotency and thirty-post retention."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
from datetime import date, datetime, time, timezone
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
    market,
    reveal,
    verdict,
    weekly_calendar,
)
from .delivery import remove_owned, upload
from .render import render, verify

PARIS = ZoneInfo("Europe/Paris")
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
SAFE_ID = re.compile(r"carousel-v1-[a-z0-9][a-z0-9-]{0,100}")
SLOTS = {"daily": (18, 30), "calendar": (8, 0), "chance": (18, 45), "documents": (18, 50)}


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
            if meta["content_sha256"] != digest(content):
                raise ValueError("existing carousel differs from selection")
        else:
            render(content, bundle, config["font_dir"])
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
    return x, calendar_rows(x, config) if x else []


def product(index, x, rows, day):
    proof = reveal(x) if x and index == 1 else None
    quote = excerpt(x) if x and index == 3 else None
    if (index == 1 and not proof) or (index == 3 and not quote):
        return evergreen(0)
    return evergreen(
        index, proof=proof, excerpt=quote, agenda=agenda_example(rows, day) if index == 3 else None
    )


def batch(config, now=None, publish=False):
    now = now or datetime.now(UTC)
    day = now.astimezone(NY).date().isoformat()
    x, rows = inputs(config, now)
    selected = [(f"product-{i + 1:02}", product(i, x, rows, day)) for i in range(4)]
    skipped = []
    if x:
        # Initial examples retain their actual brief date, never today's label.
        for b in sorted(x.get("briefs", []), key=lambda r: r["date"], reverse=True):
            example = market(x, rows, b["date"])
            if example:
                example["caption_hook"] = f"Exemple daté du {b['date']}. " + example["caption_hook"]
                selected.append(("market-example-" + b["date"], example))
                break
        cal = weekly_calendar(x, rows, day)
        if cal:
            selected.append(("calendar-example-" + day, cal))
    for i, item in enumerate(documents(config.get("documents"), now)):
        if i == 0:
            selected.append(("document-example-" + digest(item)[:16], item))
    study = verdict(config.get("study_result"), now)
    if study:
        selected.append(("chance-example-" + digest(study)[:16], study))
    for kind in ("market", "calendar", "document", "chance"):
        if not any(p["kind"] == kind for _, p in selected):
            skipped.append({"kind": kind, "reason": "no eligible observed evidence"})
    results = []
    for name, item in selected:
        # Content-addressed batch names permit a later real excerpt without overwrite.
        identifier = "carousel-v1-" + name + "-" + digest(item)[:12]
        results.append(produce(config, identifier, item, publish))
    return {"status": "completed", "posts": results, "skipped": skipped}


def run(config, slot, now=None, publish=False):
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("timezone required")
    local = now.astimezone(PARIS)
    expected = datetime.combine(local.date(), time(*SLOTS[slot]), PARIS)
    if not 0 <= (local - expected).total_seconds() < 600:
        return {"status": "skipped", "reason": "outside scheduled slot"}
    if (
        local.weekday() > 4
        or (slot == "calendar" and local.weekday() != 0)
        or (slot == "chance" and local.weekday() != 2)
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
        if slot == "daily" and x:
            item = market(x, rows, day)
        elif slot == "calendar" and x:
            item = weekly_calendar(x, rows, day)
        elif slot == "chance":
            item = verdict(config.get("study_result"), now)
        elif slot == "documents":
            candidates = documents(config.get("documents"), now)
            for candidate in candidates:
                identity = digest(candidate)
                if not (state / f"document-{identity}.json").exists():
                    item = candidate
                    break
            if item is None:
                return {"status": "skipped", "reason": "no new eligible official document"}
        if item is None:
            index = (date.fromisoformat(day).toordinal() + list(SLOTS).index(slot)) % 4
            item = product(index, x, rows, day)
        result = produce(config, f"carousel-v1-{slot}-{day}", item, publish)
        if publish:
            atomic_json(record, result)
            if item["kind"] == "document":
                atomic_json(state / f"document-{digest(item)}.json", result)
        return result
