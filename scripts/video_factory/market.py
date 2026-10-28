"""Offline factual market cards from the existing observed member export."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
DIGEST = re.compile(r"[a-f0-9]{64}")


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(UTC)


def public_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
        or parsed.hostname in ("localhost", "example.invalid")
        or "." not in parsed.hostname
        or re.fullmatch(r"[\d.:]+", parsed.hostname)
    ):
        raise ValueError("invalid public source URL")
    return value


def source(row, cutoff):
    """No source is fetched. Stored clocks and digests must precede the cutoff."""
    public_url(row["url"])
    if not DIGEST.fullmatch(row["sha256"]):
        raise ValueError("invalid source digest")
    if not instant(row["published_at"]) <= instant(row["available_at"]) <= cutoff:
        raise ValueError("noncausal market source")
    return row


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def marker_sources(calendar_file, calendar_manifest, cutoff):
    """Recover source URLs for the producer's digest-only date markers.

    The parsed calendar, individual event and captured-source hash must all bind.
    Captured bodies are never opened, copied or fetched. A collection timestamp
    does not become a fabricated publication timestamp.
    """
    if not calendar_file or not calendar_manifest:
        return {}
    calendar = json.loads(Path(calendar_file).read_text())
    manifest = json.loads(Path(calendar_manifest).read_text())
    if calendar.get("schema") != "agent-v2-quality-calendar-v2":
        raise ValueError("unsupported calendar evidence")
    received = instant(calendar["received_at"])
    if received > cutoff:
        return {}
    dataset = digest(calendar)
    sources = []
    for row in manifest:
        if (
            row["method"] != "GET"
            or row["status"] != 200
            or row["body_sha256"] not in calendar["source_hashes"]
            or not DIGEST.fullmatch(row["authorization_file_sha256"])
            or not DIGEST.fullmatch(row["body_sha256"])
        ):
            raise ValueError("unbound calendar source")
        if not instant(row["started_at"]) <= instant(row["received_at"]) <= received:
            raise ValueError("noncausal calendar source")
        public_url(row["url"])
        sources.append(row)
    result = {}
    for event in calendar["events"]:
        kind = {"EARNINGS": "earnings", "FOMC": "fed"}.get(event["kind"])
        if kind is None:
            continue
        day = event["at"][:10]
        for row in sources:
            url = urlsplit(row["url"])
            match = (
                kind == "earnings"
                and url.hostname == "api.nasdaq.com"
                and url.path == "/api/calendar/earnings"
                and parse_qs(url.query) == {"date": [day]}
            )
            match = match or (
                kind == "fed"
                and url.hostname == "www.federalreserve.gov"
                and url.path == "/monetarypolicy/fomccalendars.htm"
                and not url.query
            )
            if match:
                result[(dataset, digest(event), kind, event["symbol"], day)] = {
                    "url": row["url"],
                    "sha256": row["body_sha256"],
                    "published_at": None,
                    "available_at": row["received_at"],
                    "calendar_received_at": calendar["received_at"],
                    "dataset_sha256": dataset,
                    "event_sha256": digest(event),
                    "authorization_file_sha256": row["authorization_file_sha256"],
                }
    return result


def market_point(snapshot_path, day, now, *, calendar_file=None, calendar_manifest=None):
    raw = Path(snapshot_path).read_bytes()
    snapshot = json.loads(raw)
    generated = instant(snapshot["generated_at"])
    if (
        snapshot.get("provenance") != "observed"
        or not 0 <= (now - generated).total_seconds() <= 86400
    ):
        return None
    briefs = [b for b in snapshot.get("briefs", []) if b["date"] == day]
    if not briefs:
        return None
    if len(briefs) != 1:
        raise ValueError("ambiguous daily brief")
    brief = briefs[0]
    published = instant(brief["published_at"])
    if published.astimezone(NY).date().isoformat() != day or published > generated:
        raise ValueError("invalid brief date")
    cards, evidence, labels = [], [], []
    # Preserve the producer's summaries verbatim: no inferred market move,
    # investment recommendation, translation or new causal explanation.
    for fact in brief["facts"]:
        s = source(fact["source"], published)
        text = fact["summary"]
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 320:
            raise ValueError("invalid brief fact")
        # A morning brief can report the previous day's events. Date each source
        # explicitly instead of suggesting that every event happened today.
        stamp = instant(s["published_at"]).astimezone(NY).strftime("%d/%m à %H h %M")
        cards.append(f"Dans le brief du jour : {text}")
        labels.append(f"{urlsplit(s['url']).hostname} · {stamp} New York")
        evidence.append(s)
        if len(cards) == 2:
            break
    if not cards:
        return None
    tomorrow = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    upcoming = []
    recovered = marker_sources(calendar_file, calendar_manifest, generated)
    for entry in sorted(snapshot.get("calendar", []), key=lambda e: e["kind"] == "earnings"):
        if entry["date"] != tomorrow:
            continue
        if entry.get("source"):
            s = source(entry["source"], generated)
        else:
            marker = entry.get("evidence") or {}
            if marker.get("kind") != "observed_day_marker":
                continue
            s = recovered.get(
                (
                    marker.get("dataset_sha256"),
                    marker.get("sha256"),
                    entry["kind"],
                    entry["symbol"],
                    entry["date"],
                )
            )
            if not s:
                continue
            if (
                instant(marker["received_at"]) != instant(s["calendar_received_at"])
                or entry["precision"] != "date"
                or entry["scheduled_at"] is not None
            ):
                raise ValueError("calendar marker identity mismatch")
        event = {
            "fed": "rendez-vous de la Fed",
            "cpi": "publication de l'inflation américaine",
            "nfp": "publication de l'emploi américain",
        }.get(entry["kind"])
        if entry["kind"] == "earnings" and re.fullmatch(
            r"[A-Z][A-Z0-9.\-]{0,14}", entry["symbol"] or ""
        ):
            event = f"résultats de {entry['symbol']}"
        if event is None:
            continue
        if entry["precision"] not in ("date", "time"):
            raise ValueError("unsupported calendar precision")
        if (entry["precision"] == "date") != (entry.get("scheduled_at") is None):
            raise ValueError("calendar precision mismatch")
        timing = "horaire non confirmé"
        if entry["precision"] == "time" and entry.get("scheduled_at"):
            at = instant(entry["scheduled_at"]).astimezone(NY)
            if at.date().isoformat() != tomorrow:
                raise ValueError("calendar date mismatch")
            timing = at.strftime("%H h %M à New York")
        upcoming.append((f"Demain, {tomorrow} : {event}, {timing}.", s))
        if len(upcoming) == 2:
            break
    if not upcoming:
        return None
    for text, s in upcoming:
        cards.append(text)
        labels.append(f"Calendrier · {urlsplit(s['url']).hostname}")
        evidence.append(s)
    cards.append(
        "Le brief et le calendrier HyprL : les faits, leurs sources, et les prochains rendez-vous."
    )
    labels.append("hyprlcore.com")
    # At most one mention of AI across the complete visible edition. If the
    # underlying brief is about AI, prefer evergreen to rewriting its facts.
    if len(re.findall(r"\b(?:IA|AI|intelligence artificielle)\b", " ".join(cards), re.I)) > 1:
        return None
    return {
        "kind": "market",
        "title": "Le point marché du jour",
        "day": day,
        "captions": cards,
        "source_labels": labels,
        "sources": sorted({s["url"] for s in evidence}),
        "source_evidence": evidence,
        "notes": f"Brief publié le {day}. Les dates et heures des sources sont indiquées. "
        f"Agenda du {tomorrow}, susceptible de modification.",
        "provenance": {
            "kind": "observed_member_v2_brief_calendar",
            "generated_at": snapshot["generated_at"],
            "brief_published_at": brief["published_at"],
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
