"""Isolated daily job: published observations or the approved evergreen rotation."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .content import evergreen
from .delivery import atomic_json, upload
from .render import render

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
AGENTS = ("HypR-A1", "HypR-E1")
SAFE_ID = re.compile(r"(?:daily-\d{4}-\d{2}-\d{2}|evergreen-0[1-5]|verdict-01)")
PUBLIC_REVEAL = re.compile(r"https://github\.com/Kacawaiii/HyprL/blob/[0-9a-f]{40}/reveals/[0-9-]+/(?:HypR-[AE]1\.json|README\.md)")


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(UTC)


def guard_clock(now):
    if now.tzinfo is None:
        raise ValueError("timezone required")
    utc = now.astimezone(UTC).time().replace(tzinfo=None)
    if time(11, 30) <= utc < time(17, 15):
        raise RuntimeError("protected trading-job window")


def trading_day(day):
    import pandas_market_calendars as mcal
    return not mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty


def duel(snapshot_path, day, now):
    """Allowlist fields; keep horizons and sessions separate; no model names."""
    raw = Path(snapshot_path).read_bytes()
    snapshot = json.loads(raw)
    generated = instant(snapshot["generated_at"])
    if snapshot.get("provenance") != "observed" or not 0 <= (now-generated).total_seconds() <= 86400:
        return None
    proofs = {r["id"]: r for r in snapshot.get("reveals", [])
              if r["session"] <= day and r["mode"] == "REGULAR_0905" and r["agent"] in AGENTS
              and r["status"] in ("REVEALED", "NO_DECISION") and PUBLIC_REVEAL.fullmatch(r["url"])
              and instant(r["available_at"]) <= generated}
    current = {p["agent"]: p for p in proofs.values() if p["session"] == day}
    if set(current) != set(AGENTS):
        return None
    selected = {agent: [] for agent in AGENTS}
    seen = set()
    for row in snapshot.get("forecasts", []):
        if (row["session"] > day or row["mode"] != "REGULAR_0905"
                or row["variant"] != "a_v1_as_live" or row["horizon"] != "1d" or row["agent"] not in AGENTS):
            continue
        # This factory consumes equity forecast scores only, never BTC/ETH prices.
        if not re.fullmatch(r"[A-Z]{1,5}", row["symbol"]) or row["symbol"] in ("BTC", "ETH"):
            raise ValueError("non-equity score refused")
        proof = proofs.get(row["reveal_id"])
        c = proof.get("commitment", {}) if proof else {}
        if (not proof or proof["status"] != "REVEALED" or proof["agent"] != row["agent"]
                or proof["session"] != row["session"] or row["decision_sha256"] != proof["sha256"]
                or not proof["url"].endswith("/reveals/"+row["session"]+"/"+row["agent"]+".json")
                or c.get("sha256") != proof["sha256"]
                or not instant(c["committed_at"]) <= instant(c["published_at"]) < instant(proof["published_at"]) <= instant(proof["available_at"])
                or not instant(row["decision_time"]) < instant(row["scored_at"]) <= generated
                or not instant(row["decision_time"]) < instant(proof["published_at"])):
            raise ValueError("unproven forecast refused")
        p, y = row["probability"], row["outcome"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 or type(y) is not int or y not in (0, 1):
            raise ValueError("invalid probability/outcome")
        key = row["agent"], row["session"], row["symbol"]
        if key in seen:
            raise ValueError("duplicate forecast")
        seen.add(key)
        selected[row["agent"]].append(row)
    # A one-day forecast can mature on a later trading session. Never present an
    # older forecast as a result of today's decision, or combine different dates.
    comparable = set(r["session"] for r in selected[AGENTS[0]]) & set(r["session"] for r in selected[AGENTS[1]])
    score_session = max(comparable) if comparable else None
    selected = {a: [r for r in rows if r["session"] == score_session] for a, rows in selected.items()}
    status = "; ".join(a + (" : décisions révélées" if p["status"] == "REVEALED" else " : aucune décision publiée") for a, p in current.items())
    chart, captions = [], [f"Duel du {day}. {status}."]
    for agent, rows in selected.items():
        if not rows:
            captions.append(f"{agent} : aucun score comparable disponible pour le panel actions à un jour. On attend les échéances.")
            continue
        hits = sum(int((r["probability"] >= .5) == bool(r["outcome"])) for r in rows)
        brier = sum((r["probability"] - r["outcome"])**2 for r in rows) / len(rows)
        chart.append({"agent": agent, "hits": hits, "n": len(rows)})
        captions.append(f"Prévisions du {score_session}, horizon un jour. {agent} : {hits} correctes sur {len(rows)} évaluées. Brier : {brier:.3f}, plus bas est meilleur.")
    luck = []
    for s in snapshot.get("series", []):
        if (s.get("agent") in AGENTS and s.get("variant") == "a_v1_as_live"
                and s.get("mode") == "REGULAR_0905" and s.get("horizon") == "1d" and s.get("points")):
            point = s["points"][-1]
            tail = point.get("luck_tail")
            if (point.get("session") == day and tail is not None and isinstance(tail, (int, float))
                    and not isinstance(tail, bool) and math.isfinite(tail) and 0 <= tail <= 1
                    and instant(point["at"]) <= generated):
                luck.append(f"{s['agent']} : {100*tail:.1f} %")
    captions.append("Indicateur de hasard publié, cumulatif : " + "; ".join(luck) + ". Ce n'est pas une probabilité de talent."
                    if luck else "L'indicateur de hasard n'est pas disponible pour ce panel. Ces quelques prévisions ne prouvent aucun avantage.")
    captions.append("On mesure une prévision face au SPY, pas un gain en euros. Les horizons encore ouverts attendent leur résultat.")
    used_proofs = list(current.values()) + [proofs[r["reveal_id"]] for rows in selected.values() for r in rows]
    return {"kind": "duel", "title": f"Le duel du {day}", "captions": captions, "chart": chart, "score_session": score_session,
            "notes": "Panel actions, prévisions face au SPY à un jour ; aucun rendement de portefeuille. "
                     "Dénominateurs affichés séparément, observations corrélées possibles. "
                     "Échantillon complet publié disponible pour cette séance au moment de la génération.",
            "sources": sorted({p["url"] for p in used_proofs}),
            "provenance": {"kind": "observed_member_v2", "snapshot_sha256": hashlib.sha256(raw).hexdigest(), "generated_at": snapshot["generated_at"]}}


def retain(root, limit=14):
    root = Path(root)
    owned = sorted((p for p in root.iterdir() if SAFE_ID.fullmatch(p.name) and p.is_dir() and not p.is_symlink()
                    and (p / "metadata.json").is_file()), key=lambda p: (p.stat().st_mtime_ns, p.name))
    while len(owned) > limit:
        old = owned[0]
        receipt = old / "delivery.json"
        if not receipt.exists() or json.loads(receipt.read_text()).get("status") != "delivered":
            raise RuntimeError("retention blocked by undelivered video")
        shutil.rmtree(old)
        owned.pop(0)


def produce(config, identifier, content, publish=True):
    if not SAFE_ID.fullmatch(identifier):
        raise ValueError("unsafe video id")
    root = Path(config["output_dir"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / "factory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = datetime.now(UTC)
        guard_clock(now)
        # Reserve the bounded render + upload budget before the blackout starts.
        guard_clock(now + timedelta(minutes=10))
        bundle = root / identifier
        if bundle.is_symlink():
            raise ValueError("symlink bundle refused")
        if not bundle.exists():
            render(content, bundle, config.get("ffmpeg", "ffmpeg"))
        if publish:
            result = upload(bundle, config["webhook_env"])
            retain(root)
            return {"id": identifier, **result}
        return {"id": identifier, "status": "rendered"}


def daily(config, now=None, publish=True):
    now = now or datetime.now(UTC)
    guard_clock(now)
    local = now.astimezone(NY)
    if local.time().replace(tzinfo=None) < time(17, 5) or not trading_day(local.date().isoformat()):
        return {"status": "skipped", "reason": "outside post-close trading day"}
    day = local.date().isoformat()
    state_dir = Path(config["state_dir"])
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state_dir / "daily.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = state_dir / f"{day}.json"
        if state.exists() and json.loads(state.read_text()).get("status") == "delivered":
            return {"status": "already_delivered", "day": day}
        content = None
        if local.weekday() == 3 and config.get("verdict_content"):
            content = json.loads(Path(config["verdict_content"]).read_text())
            if content.get("kind") != "verdict":
                raise ValueError("invalid verdict content")
        if content is None and local.weekday() != 1:
            try:
                content = duel(config["snapshot"], day, now)
            except FileNotFoundError:
                content = None
        if content is None:
            content = evergreen(date.fromisoformat(day).toordinal() % 5)
        result = produce(config, "daily-"+day, content, publish=publish)
        if publish:
            atomic_json(state, {**result, "day": day, "kind": content["kind"]})
        return result
