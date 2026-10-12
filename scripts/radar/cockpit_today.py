"""The three small blocks of the cockpit home that no other snapshot carries.

* `calendar`: the next scheduled events (CPI, Fed, earnings), read from a private operator calendar file and from the
  dated entries of the Claude book watchlist. Nothing is fetched; an entry without a known time says so.
* `ai_today`: one line per decision the AI paper accounts would execute from the latest real run, selected by the
  executor's own rule (`execution_tiers.selections`), with the broker outcome when the paper report has one.
* `system`: one light (green/orange/red) and the list of problems behind it, from the health the refresh already reads.

Every block is plain data for the read-only API; values are copied or classified, never forecast.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CALENDAR_SCHEMA = "cockpit-calendar-v1"
KINDS = {"cpi", "fed", "earnings", "election", "other"}
LEVELS = {"OFFICIEL", "FIL", "NON VERIFIE"}
MAX_CALENDAR = 12
NEW_YORK = ZoneInfo("America/New_York")
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Watchlist dates are written by hand, e.g. "2026-10-13 07:30 ET"; only a full day (and optional ET time) is used.
WATCH_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}):(\d{2})\s*ET)?$")
EARNINGS = re.compile(r"result|earnings|résultat|resultat", re.I)
EXECUTED = {"FILLED", "CLOSED", "PARTIAL_EXIT", "EXIT_PENDING", "INTERNAL_SETTLEMENT"}
FAILED_UNIT_RESULTS = {"failed", "exit-code", "signal", "core-dump", "timeout", "watchdog", "start-limit-hit"}


def _instant(value):
    if not isinstance(value, str):
        return None
    try:
        out = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo else None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value, limit):
    return value.strip()[:limit] if isinstance(value, str) and value.strip() else None


def _entry(*, at=None, day=None, title, kind, verification, assets, origin):
    return {"at": _iso(at) if at else None, "date": at.astimezone(timezone.utc).date().isoformat() if at else day,
            "time_known": at is not None, "title": title, "kind": kind, "verification": verification,
            "assets": assets[:6], "origin": origin}


def calendar_entries(calendar=None, watchlist=None, *, now: datetime):
    """Upcoming entries, soonest first. An entry already past (or a day already over) is dropped."""
    out = []
    for raw in (calendar or {}).get("entries", []) if (calendar or {}).get("schema") == CALENDAR_SCHEMA else []:
        title = _text(raw.get("title"), 120)
        if not title:
            continue
        kind = raw.get("kind") if raw.get("kind") in KINDS else "other"
        level = raw.get("verification") if raw.get("verification") in LEVELS else "NON VERIFIE"
        assets = [a for a in (raw.get("assets") or []) if isinstance(a, str) and len(a) <= 12]
        at = _instant(raw.get("at"))
        if at:
            out.append(_entry(at=at, title=title, kind=kind, verification=level, assets=assets, origin="calendar"))
        elif isinstance(raw.get("date"), str) and DAY.match(raw["date"]):
            out.append(_entry(day=raw["date"], title=title, kind=kind, verification=level, assets=assets, origin="calendar"))
    for symbol, raw in (watchlist or {}).items():
        if not isinstance(raw, dict) or not isinstance(raw.get("date"), str):
            continue
        match = WATCH_DATE.match(raw["date"].strip())
        catalyst = _text(raw.get("catalyst"), 100)
        if not match or not catalyst or not isinstance(symbol, str):
            continue
        day, hour, minute = match.groups()
        at = None
        if hour is not None:
            at = datetime.fromisoformat(f"{day}T{hour}:{minute}:00").replace(tzinfo=NEW_YORK)
        kind = "earnings" if EARNINGS.search(catalyst) else "other"
        entry = dict(at=at, title=f"{symbol[:12]} : {catalyst}", kind=kind, verification="NON VERIFIE",
                     assets=[symbol[:12]], origin="watchlist")
        out.append(_entry(**entry) if at else _entry(day=day, **{k: v for k, v in entry.items() if k != "at"}))
    today = now.astimezone(timezone.utc).date().isoformat()
    upcoming = [e for e in out if (e["at"] and _instant(e["at"]) >= now) or (not e["at"] and e["date"] >= today)]
    # The operator calendar wins over a watchlist line for the same asset on the same day.
    seen, unique = set(), []
    for e in sorted(upcoming, key=lambda e: (e["origin"] != "calendar", e["date"], e["at"] or "")):
        keys = {(e["date"], a.upper()) for a in e["assets"]}
        if keys & seen:
            continue
        seen |= keys
        unique.append(e)
    return sorted(unique, key=lambda e: (e["date"], e["at"] or e["date"] + "T99"))[:MAX_CALENDAR]


def _account_for(asset) -> str:
    s = str(asset or "")
    return "ia_crypto" if "/" in s or s.endswith("-USD") else "ia_actions"


def _norm(symbol) -> str:
    return str(symbol or "").upper().replace("/", "").replace("-", "")


def ai_today(runs, report=None):
    """Decisions the executor would take from the latest real run, one line per account, asset and horizon.

    Status: `executed` when the broker filled the entry, `expired` when it ended unfilled, `sent` while the broker has it,
    `blocked` when the account or its half-size tier is halted, `skipped` when the executor already acted on this run
    (or a later day) without sending it (its eligibility guards), `planned` when no order trace exists yet.
    """
    from scripts.trading_lab.trader_agent.execution_tiers import selections
    done = [r for r in runs or [] if r.get("status") in ("COMPLETE", "DEGRADED") and not r.get("synthetic")]
    if not done:
        return {"run_id": None, "at": None, "session": None, "lines": []}
    latest = done[-1]
    day = latest["at"][:10]
    # One run per slot of the latest day: the pm slot decides independently of the am slot.
    by_slot = {}
    for r in done:
        if r["at"][:10] == day:
            by_slot[r.get("slot", "am")] = r
    report = report or {}
    accounts = {a.get("account"): a for a in report.get("accounts", []) if isinstance(a, dict)}
    outcomes = [o for o in report.get("execution_outcomes", []) or [] if o.get("purpose") == "entry"]
    lines = []
    for slot, run in sorted(by_slot.items()):
        for account in ("ia_actions", "ia_crypto"):
            try:
                chosen = selections(run.get("views") or [], account)
            except Exception:  # a binding error in the executor is its own alert; the home shows no line for it
                continue
            state = accounts.get(account, {})
            guard = (state.get("downgrade_guard") or {}).get("state")
            acted = (any(o.get("account") == account and str(o.get("at") or "") >= run["at"] for o in outcomes)
                     or str(report.get("at") or "")[:10] > day)
            for (asset, horizon), (view, multiplier, tier) in sorted(chosen.items()):
                if _account_for(asset) != account:
                    continue
                match = [o for o in outcomes if o.get("account") == account and _norm(o.get("symbol")) == _norm(asset)
                         and o.get("horizon") == horizon and str(o.get("at") or "") >= run["at"]]
                status = "planned"
                if match:
                    last = match[-1]
                    status = ("executed" if last.get("state") in EXECUTED else "sent" if last.get("state") == "PENDING"
                              else "expired")
                elif state.get("halted") or (tier and "downgrade" in tier and guard == "HALTED"):
                    status = "blocked"
                elif acted:
                    status = "skipped"
                lines.append({"account": account, "asset": asset, "horizon": horizon, "slot": slot,
                              "direction": view.get("view"), "size": "full" if multiplier == 1 else "half",
                              "analyst": view.get("analyst"), "verdict": view.get("verdict"), "status": status})
    order = {"executed": 0, "sent": 1, "planned": 2, "blocked": 3, "skipped": 4, "expired": 5}
    lines.sort(key=lambda l: (order[l["status"]], l["account"], l["asset"], l["horizon"]))
    return {"run_id": latest["run_id"], "at": latest["at"], "session": day, "lines": lines}


def system_light(system=None, *, radar_cutoff=None, now: datetime):
    """Problems a beginner should know about, each with a level; the light is the worst level, or `unknown`."""
    system = system or {}
    problems = []
    for unit in system.get("units") or []:
        if unit.get("state") == "failed" or (unit.get("result") or "") in FAILED_UNIT_RESULTS:
            problems.append({"level": "red", "text": f"Tâche en échec : {_text(unit.get('unit'), 80)}"})
    trader = system.get("trader") or {}
    health = (trader.get("health") or {}).get("state")
    if health and health != "HEALTHY":
        problems.append({"level": "orange" if health in ("DEGRADED", "EMBARGOED") else "red",
                         "text": f"Santé du trader IA : {health}"})
    if trader.get("paused"):
        problems.append({"level": "orange", "text": "Le trader IA est en pause."})
    recent = sorted({f.get("code") for f in trader.get("failures") or []
                     if _instant(f.get("at")) and now - _instant(f.get("at")) <= timedelta(hours=24) and f.get("code")})
    if recent:
        problems.append({"level": "orange", "text": "Incidents du trader sur 24 h : " + ", ".join(recent[:4])})
    cutoff = _instant(radar_cutoff)
    if cutoff is None or now - cutoff > timedelta(hours=14):
        problems.append({"level": "orange", "text": "Les actualités n’ont pas été mises à jour depuis plus de 14 h."})
    level = "red" if any(p["level"] == "red" for p in problems) else "orange" if problems else "green"
    if level == "green" and not health and not system.get("units"):
        # Without trader health or unit states (the one-shot CLI export reads neither) nothing was checked: no green.
        level = "unknown"
    return {"level": level, "problems": problems, "checked_at": _iso(now)}
