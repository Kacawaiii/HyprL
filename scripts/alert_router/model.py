"""Closed factual vocabulary: no free-text recommendations or model commentary."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

DISCLAIMER = "information, pas un conseil en investissement"
AGENTS = ["HypR-A1", "HypR-E1"]
KINDS = {
    "sec_filing",
    "fed_release",
    "cpi_release",
    "nfp_release",
    "earnings_date",
    "earnings_result",
    "unusual_move",
    "radar_headline",
    "duel_hashes",
    "duel_results",
    "weekly_luck",
    "system_test",
}


class AlertError(ValueError):
    """Only fixed error codes may leave the service boundary."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def instant(value):
    try:
        if not isinstance(value, str) or len(value) > 35:
            raise ValueError
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
            raise ValueError
        return result
    except (ValueError, TypeError, AttributeError):
        raise AlertError("INVALID_TIMESTAMP") from None


def match(value, pattern):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def source_url(value):
    try:
        u = urlsplit(value)
        return (
            isinstance(value, str)
            and len(value) <= 400
            and value.isascii()
            and not re.search(r"[\s<>\\]", value)
            and u.scheme == "https"
            and u.hostname
            and not u.username
            and not u.password
            and not u.fragment
            and u.port in (None, 443)
        )
    except (ValueError, TypeError, AttributeError):
        return False


def number(value):
    return match(value, r"-?\d{1,15}(?:\.\d{1,8})?")


def hash_value(value):
    return match(value, r"[a-f0-9]{64}")


def day(value):
    return match(value, r"\d{4}-\d{2}-\d{2}") and bool(instant(value + "T00:00:00Z"))


def period(value):
    return match(value, r"FY\d{4}(?:Q[1-4])?")


SCHEMAS = {
    "sec_filing": {
        "form": lambda v: v in ("8-K", "10-Q", "10-K", "S-1", "4"),
        "cik": lambda v: match(v, r"\d{10}"),
        "accession": lambda v: match(v, r"\d{10}-\d{2}-\d{6}"),
    },
    "fed_release": {"release": lambda v: v in ("statement", "minutes", "decision", "speech")},
    "cpi_release": {
        "period": lambda v: match(v, r"\d{4}-(?:0[1-9]|1[0-2])"),
        "yoy_pct": number,
        "mom_pct": number,
    },
    "nfp_release": {
        "period": lambda v: match(v, r"\d{4}-(?:0[1-9]|1[0-2])"),
        "jobs_change": lambda v: match(v, r"-?\d{1,9}"),
        "unemployment_pct": number,
    },
    "earnings_date": {"scheduled_at": instant, "period": period},
    "earnings_result": {
        "period": period,
        "eps": number,
        "revenue": number,
        "currency": lambda v: v in ("USD", "EUR", "GBP", "JPY", "CAD"),
    },
    "unusual_move": {
        "price_change_pct": number,
        "volume_multiple": number,
        "window_minutes": lambda v: type(v) is int and 1 <= v <= 10080,
    },
    "radar_headline": {
        "category": lambda v: v in ("macro", "earnings", "regulation", "merger", "other"),
        "impact": lambda v: v == "high",
    },
    "duel_hashes": {
        "session": day,
        "open_at": instant,
        "a1_sha256": hash_value,
        "e1_sha256": hash_value,
    },
    "duel_results": {
        "session": day,
        "close_at": instant,
        "a1_sha256": hash_value,
        "e1_sha256": hash_value,
        "a1_return_pct": number,
        "e1_return_pct": number,
    },
    "weekly_luck": {
        "week_ending": day,
        "a1_percentile": number,
        "e1_percentile": number,
        "observations": lambda v: type(v) is int and v >= 1,
    },
    "system_test": {"synthetic": lambda v: v is True},
}


@dataclass(frozen=True)
class Alert:
    # Store canonical immutable bytes; callers cannot mutate a validated alert.
    raw: bytes

    @classmethod
    def parse(cls, event):
        required = {
            "schema_version",
            "event_id",
            "kind",
            "occurred_at",
            "available_at",
            "source",
            "symbols",
            "facts",
            "agents",
        }
        if not isinstance(event, dict) or set(event) != required:
            raise AlertError("INVALID_EVENT_FIELDS")
        if type(event["schema_version"]) is not int or event["schema_version"] != 1:
            raise AlertError("UNSUPPORTED_SCHEMA")
        if (
            not hash_value(event["event_id"])
            or not isinstance(event["kind"], str)
            or event["kind"] not in KINDS
        ):
            raise AlertError("INVALID_EVENT_ID_OR_KIND")
        at, available = instant(event["occurred_at"]), instant(event["available_at"])
        if at > available:
            raise AlertError("INVALID_AVAILABILITY")
        source = event["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"name", "url", "sha256"}
            or source["name"]
            not in (
                "SEC",
                "Federal Reserve",
                "BLS",
                "issuer",
                "market-data",
                "radar",
                "HyprL",
                "synthetic",
            )
            or not source_url(source["url"])
            or not hash_value(source["sha256"])
        ):
            raise AlertError("INVALID_SOURCE")
        symbols = event["symbols"]
        if (
            not isinstance(symbols, list)
            or len(symbols) > 50
            or any(not match(s, r"[A-Z][A-Z0-9.\-]{0,14}") for s in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise AlertError("INVALID_SYMBOLS")
        if event["agents"] != AGENTS:
            raise AlertError("INVALID_AGENTS")
        kind, facts = event["kind"], event["facts"]
        schema = SCHEMAS[kind]
        if not isinstance(facts, dict) or set(facts) != set(schema):
            raise AlertError("INVALID_FACT_FIELDS")
        try:
            if not all(check(facts[key]) for key, check in schema.items()):
                raise AlertError("INVALID_FACT")
        except (TypeError, ValueError):
            raise AlertError("INVALID_FACT") from None
        if (
            kind in ("sec_filing", "earnings_date", "earnings_result", "unusual_move")
            and not symbols
        ):
            raise AlertError("SYMBOL_REQUIRED")
        if kind == "duel_hashes" and available >= instant(facts["open_at"]):
            raise AlertError("HASHES_NOT_BEFORE_OPEN")
        if kind == "duel_results" and at < instant(facts["close_at"]):
            raise AlertError("RESULTS_NOT_AFTER_CLOSE")
        if kind == "weekly_luck" and any(
            not 0 <= float(facts[k]) <= 100 for k in ("a1_percentile", "e1_percentile")
        ):
            raise AlertError("INVALID_PERCENTILE")
        return cls(canonical(event))

    def event(self):
        return json.loads(self.raw)

    def payload(self):
        return {**self.event(), "disclaimer": DISCLAIMER}

    def text(self):
        e = self.event()
        titles = {
            "sec_filing": "Dépôt SEC",
            "fed_release": "Publication Fed",
            "cpi_release": "Publication CPI",
            "nfp_release": "Publication NFP",
            "earnings_date": "Date de résultats",
            "earnings_result": "Résultats publiés",
            "unusual_move": "Mouvement prix/volume inhabituel",
            "radar_headline": "Radar : actualité à fort impact",
            "duel_hashes": "Duel : empreintes avant ouverture",
            "duel_results": "Duel : résultats après clôture",
            "weekly_luck": "Indicateur de chance hebdomadaire",
            "system_test": "TEST STAFF — données synthétiques",
        }
        facts = "; ".join(f"{k}={v}" for k, v in e["facts"].items())
        return (
            f"{titles[e['kind']]} {' / '.join(e['symbols'])}\n{facts}\n"
            f"HypR-A1 / HypR-E1\nSource : {e['source']['name']} {e['source']['url']}\n"
            f"Publication : {e['occurred_at']} | Disponible : {e['available_at']}\n{DISCLAIMER}"
        )


def synthetic_alert():
    return {
        "schema_version": 1,
        "event_id": hashlib.sha256(b"hyprl-staff-test-v1").hexdigest(),
        "kind": "system_test",
        "occurred_at": "2026-10-26T12:00:00Z",
        "available_at": "2026-10-26T12:00:00Z",
        "agents": list(AGENTS),
        "symbols": [],
        "source": {
            "name": "synthetic",
            "url": "https://example.org/hyprl/staff-test",
            "sha256": hashlib.sha256(b"synthetic").hexdigest(),
        },
        "facts": {"synthetic": True},
    }
