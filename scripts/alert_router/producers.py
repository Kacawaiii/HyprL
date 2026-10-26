"""Adapters over already captured local records; never import a live collector.

Raw headlines, summaries, analyses and trading decisions are intentionally
excluded. Unknown publication times stay blocked rather than guessed.
"""

import hashlib

from .model import AGENTS, Alert, AlertError, canonical


def fact_event(kind, facts, *, source, occurred_at, available_at, symbols=()):
    event = {
        "schema_version": 1,
        "kind": kind,
        "facts": facts,
        "source": source,
        "occurred_at": occurred_at,
        "available_at": available_at,
        "symbols": sorted(set(symbols)),
        "agents": list(AGENTS),
    }
    event["event_id"] = hashlib.sha256(canonical(event)).hexdigest()
    return Alert.parse(event)


def sec_story(story):
    """scripts.radar.official.parse_sec output (CIK stays a padded string)."""
    if story.get("source") != "sec:8k" or not story.get("primary") or not story.get("published_at"):
        raise AlertError("SOURCE_NOT_VERIFIED")
    f = story["filing"]
    return fact_event(
        "sec_filing",
        {"form": f["form"], "cik": f["cik"], "accession": f["accession"]},
        source={
            "name": "SEC",
            "url": story["url"],
            "sha256": hashlib.sha256(canonical(story)).hexdigest(),
        },
        occurred_at=story["published_at"],
        available_at=story["received_at"],
        symbols=story["symbols"],
    )


def radar_event(event, *, available_at):
    """scripts.radar.analysis.cluster output; only corroborated high impact items."""
    if (
        event.get("importance", 0) < 70
        or event.get("evidence", {}).get("status")
        not in ("primary_statement", "corroborated_reporting")
        or not event.get("published_at")
    ):
        raise AlertError("SOURCE_NOT_VERIFIED")
    stories = [s for s in event["stories"] if not s["retail"]]
    if not stories:
        raise AlertError("SOURCE_NOT_VERIFIED")
    lead = sorted(stories, key=lambda s: (not s["primary"], s["id"]))[0]
    themes = event["entities"]["themes"]
    category = next(
        (x for x in ("macro", "earnings", "regulation") if x in themes),
        "merger" if "m&a" in themes else "other",
    )
    symbols = [s.replace("/", "-") for s in event["entities"]["symbols"]]
    return fact_event(
        "radar_headline",
        {"category": category, "impact": "high"},
        source={
            "name": "radar",
            "url": lead["url"],
            "sha256": hashlib.sha256(canonical(event)).hexdigest(),
        },
        occurred_at=event["published_at"],
        available_at=available_at,
        symbols=symbols,
    )


def publish(store, alert, *, now):
    """Fan out only to the operator's allowed accounts and their subscriptions."""
    return sum(store.enqueue(customer, alert, now=now) for customer in sorted(store.allowed))
