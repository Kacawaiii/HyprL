"""Read-only adapters. No HTTP, raw capture, price store or holdout access."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from scripts.video_factory.factory import AGENTS, PUBLIC_REVEAL
from scripts.video_factory.market import DIGEST, instant, marker_sources, public_url, source
from scripts.video_factory.study import plan_hash

from .content import post, slide

PARIS = ZoneInfo("Europe/Paris")
NY = ZoneInfo("America/New_York")
DAYS = ("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi")
MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
SESSIONS = {"time-pre-market": "avant l'ouverture", "time-after-hours": "après la clôture"}
UNKNOWN_HOURS = "Horaires non indiqués : à confirmer."


def human_date(day):
    d = date.fromisoformat(day)
    return f"{DAYS[d.weekday()] if d.weekday() < 5 else ('Samedi', 'Dimanche')[d.weekday() - 5]} {d.day} {MONTHS[d.month - 1]}"


def calendar_label(s):
    if urlsplit(s["url"]).hostname == "api.nasdaq.com":
        return f"Source : calendrier Nasdaq, relevé le {instant(s['available_at']).astimezone(PARIS):%d/%m}"
    return label(s)


def nasdaq_details(config, recovered):
    """Read existing local feed bodies only after matching their recovered source digest.

    The producer's date markers omit names/session hints. This optional enrichment
    never expands the selected universe and never fetches a source.
    """
    if not config.get("calendar_manifest"):
        return {}
    root = Path(config["calendar_manifest"]).parent / "calendar-sources"
    details = {}
    for s in {s["url"]: s for s in recovered.values()}.values():
        url = urlsplit(s["url"])
        if url.hostname != "api.nasdaq.com" or url.path != "/api/calendar/earnings":
            continue
        day = parse_qs(url.query)["date"][0]
        date.fromisoformat(day)
        path = root / f"nasdaq-{day}.body"
        if not path.exists():
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != s["sha256"]:
            raise ValueError("calendar detail source digest mismatch")
        rows = (json.loads(raw).get("data") or {}).get("rows") or []
        for position, row in enumerate(rows):
            key = (s["sha256"], row["symbol"], day)
            if key in details:
                raise ValueError("duplicate calendar detail symbol")
            details[key] = {
                "company_name": row.get("name"),
                "session": SESSIONS.get(row.get("time")),
                "feed_order": position,
            }
    return details


def label(s):
    publisher = urlsplit(s["url"]).hostname
    if s.get("published_at"):
        return f"{publisher}\nPublié le {instant(s['published_at']).astimezone(PARIS):%d/%m/%Y à %H:%M} Paris"
    return f"{publisher} · publication non horodatée\nCollecté le {instant(s['available_at']).astimezone(PARIS):%d/%m/%Y à %H:%M} Paris"


def load(snapshot_path, now):
    raw = Path(snapshot_path).read_bytes()
    x = json.loads(raw)
    if (
        x.get("provenance") != "observed"
        or not 0 <= (now - instant(x["generated_at"])).total_seconds() <= 86400
    ):
        return None
    x["_provenance"] = {
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_at": x["generated_at"],
        "kind": "observed_member_v2",
    }
    return x


def calendar_rows(x, config):
    cutoff = instant(x["generated_at"])
    try:
        recovered = marker_sources(
            config.get("calendar_file"), config.get("calendar_manifest"), cutoff
        )
    except FileNotFoundError:
        recovered = {}
    details = nasdaq_details(config, recovered)
    rows, seen = [], set()
    for entry in x.get("calendar", []):
        key = (entry["kind"], entry.get("symbol"), entry["date"])
        if key in seen:
            raise ValueError("duplicate calendar event")
        seen.add(key)
        if entry.get("source"):
            s = source(entry["source"], cutoff)
        else:
            marker = entry.get("evidence") or {}
            if marker.get("kind") != "observed_day_marker":
                continue
            s = recovered.get((marker.get("dataset_sha256"), marker.get("sha256"), *key))
            if not s:
                continue
            if (
                instant(marker["received_at"]) != instant(s["calendar_received_at"])
                or entry["precision"] != "date"
                or entry["scheduled_at"] is not None
            ):
                raise ValueError("calendar evidence mismatch")
        if entry["precision"] not in ("date", "time") or (entry["precision"] == "date") != (
            entry.get("scheduled_at") is None
        ):
            raise ValueError("calendar precision mismatch")
        timing = "horaire non confirmé"
        if entry["precision"] == "time":
            at = instant(entry["scheduled_at"]).astimezone(NY)
            if at.date().isoformat() != entry["date"]:
                raise ValueError("calendar date mismatch")
            timing = f"{at:%H:%M} New York"
        if entry["kind"] == "earnings":
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", entry.get("symbol") or ""):
                raise ValueError("invalid earnings symbol")
            title = f"Résultats de {entry['symbol']}"
        else:
            title = {
                "fed": "Rendez-vous de la Fed",
                "cpi": "Inflation américaine",
                "nfp": "Emploi américain",
            }.get(entry["kind"])
        if title:
            extra = details.get((s["sha256"], entry.get("symbol"), entry["date"]), {})
            rows.append({**entry, **extra, "source": s, "title": title, "timing": timing})
    return rows


def excerpt(x, translations=None):
    for b in sorted(x.get("briefs", []), key=lambda r: r["date"], reverse=True):
        cutoff = instant(b["published_at"])
        if (
            cutoff > instant(x["generated_at"])
            or cutoff.astimezone(NY).date().isoformat() != b["date"]
        ):
            raise ValueError("brief date mismatch")
        for fact in b["facts"]:
            # Preserve a verbatim excerpt; prefer another fact to rewriting the source.
            text = fact["summary"]
            if (
                "—" in text
                or len(text) > 280
                or len(re.findall(r"\b(?:IA|AI|intelligence artificielle)\b", text, re.I)) > 1
            ):
                continue
            translated = (translations or {}).get(hashlib.sha256(text.encode()).hexdigest())
            if translated:
                from .editorial import check_text

                check_text(translated, text)
                text = translated
            s = source(fact["source"], cutoff)
            return {
                "title": f"Brief du {date.fromisoformat(b['date']):%d/%m/%Y}",
                "body": text,
                "translated": bool(translated),
                "source": label(s),
                "urls": [s["url"]],
                "evidence": [s],
                "provenance": x["_provenance"],
            }
    return None


def reveal(x):
    candidates = []
    cutoff = instant(x["generated_at"])
    for r in x.get("reveals", []):
        if r["status"] != "REVEALED" or r["mode"] != "REGULAR_0905" or r["agent"] not in AGENTS:
            continue
        if not PUBLIC_REVEAL.fullmatch(r["url"]) or not r["url"].endswith(
            f"/reveals/{r['session']}/{r['agent']}.json"
        ):
            raise ValueError("invalid public reveal")
        c = r["commitment"]
        public_url(c["url"])
        if not re.fullmatch(
            r"https://github\.com/Kacawaiii/HyprL/blob/[a-f0-9]{40}/forward-duel-ledger\.jsonl",
            c["url"],
        ):
            raise ValueError("invalid public commitment")
        if not DIGEST.fullmatch(r["sha256"]) or r["sha256"] != c["sha256"]:
            raise ValueError("reveal commitment mismatch")
        committed, published, disclosed, available = map(
            instant, (c["committed_at"], c["published_at"], r["published_at"], r["available_at"])
        )
        session = date.fromisoformat(r["session"])
        # Validate against the actual exchange schedule (DST and early closes).
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar("NYSE").schedule(start_date=session, end_date=session)
        if schedule.empty:
            raise ValueError("reveal on non-trading session")
        opening, closing = schedule.iloc[0]["market_open"], schedule.iloc[0]["market_close"]
        if not committed <= published < opening < closing < disclosed <= available <= cutoff:
            raise ValueError("noncausal public reveal")
        candidates.append(r)
    if not candidates:
        return None
    r = max(candidates, key=lambda r: (r["session"], r["agent"]))
    c = r["commitment"]
    return {
        "commit_text": f"{r['agent']} · séance du {r['session']}\nPublié le {instant(c['published_at']).astimezone(PARIS):%d/%m à %H:%M} Paris, avant l'ouverture.\nSHA-256 : {c['sha256'][:16]}… (empreinte complète en légende).",
        "reveal_text": f"Fichier de la séance du {r['session']}, révélé le {instant(r['published_at']).astimezone(PARIS):%d/%m à %H:%M} Paris. L'export associe la même empreinte au fichier et au registre.",
        "commit_source": f"Registre public HyprL · GitHub\nEmpreinte publiée le {instant(c['published_at']).astimezone(PARIS):%d/%m/%Y à %H:%M} Paris",
        "source": f"Registre public HyprL · GitHub\nRévélation publiée le {instant(r['published_at']).astimezone(PARIS):%d/%m/%Y à %H:%M} Paris",
        "urls": [c["url"], r["url"]],
        "evidence": [r],
        "provenance": x["_provenance"],
    }


def agenda_example(rows, day):
    choices = [r for r in rows if r["date"] >= day]
    if not choices:
        return None
    r = sorted(choices, key=lambda r: (r["date"], r["kind"] == "earnings", r.get("symbol") or ""))[
        0
    ]
    return {
        "body": f"{human_date(r['date'])} : {r['title']}, {r.get('session') or r['timing']}.",
        "source": calendar_label(r["source"]),
        "urls": [r["source"]["url"]],
        "evidence": [r["source"]],
    }


def market(x, rows, day):
    briefs = [b for b in x.get("briefs", []) if b["date"] == day]
    if not briefs:
        return None
    if len(briefs) != 1:
        raise ValueError("ambiguous daily brief")
    b = briefs[0]
    published = instant(b["published_at"])
    if published > instant(x["generated_at"]) or published.astimezone(NY).date().isoformat() != day:
        raise ValueError("brief date mismatch")
    items, evidence = [], []
    for f in b["facts"][:3]:
        s = source(f["source"], published)
        items.append(
            slide(
                "Dans le brief",
                f["summary"]
                + "\nÀ retenir : distinguer le commentaire cité d'une décision officielle.",
                source=label(s),
            )
        )
        evidence.append(s)
    # A sparse brief can be completed with genuine events for that session.
    for r in sorted(rows, key=lambda r: (r["kind"] == "earnings", r.get("symbol") or "")):
        if len(items) >= 3:
            break
        if r["date"] != day:
            continue
        items.append(
            slide(
                r["title"],
                f"{day} · {r['timing']}.\nÀ retenir : vérifier cette publication dans le calendrier avant de préparer la séance.",
                source=label(r["source"]),
            )
        )
        evidence.append(r["source"])
    if len(items) != 3:
        return None
    tomorrow = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    coming = sorted(
        (r for r in rows if r["date"] == tomorrow), key=lambda r: r["kind"] == "earnings"
    )[:1]
    if not coming:
        return None
    agenda = (
        "\n".join(f"{r['title']} : {r['timing']}" for r in coming)
        if coming
        else "Agenda sourcé indisponible."
    )
    evidence.extend(r["source"] for r in coming)
    return post(
        "market",
        "Value-Stack",
        "3 choses qui comptent avant l'ouverture",
        [
            slide(
                "3 choses qui comptent avant l'ouverture",
                f"Brief du {day}. Faits et rendez-vous de cette séance.",
                metric="3",
            ),
            *items,
            slide(
                f"Demain · {tomorrow}",
                agenda,
                source="\n".join(dict.fromkeys(label(r["source"]) for r in coming)) or None,
                cta="Le brief complet chaque matin, gratuit : lien en bio",
            ),
        ],
        f"Votre séance en trois repères sourcés. Édition du {day} ; le soir, ce carrousel récapitule le brief du matin.",
        sources=[s["url"] for s in evidence],
        evidence=evidence,
        provenance=x["_provenance"],
    )


def weekly_calendar(x, rows, day):
    start = date.fromisoformat(day) - timedelta(days=date.fromisoformat(day).weekday())
    end = start + timedelta(days=4)
    events = [
        r
        for r in rows
        if r["kind"] == "earnings" and start.isoformat() <= r["date"] <= end.isoformat()
    ]
    if not events:
        return None
    events = [
        r
        for _, r in sorted(
            enumerate(events),
            key=lambda item: (item[1]["date"], item[1].get("feed_order", item[0])),
        )
    ]

    def company(r):
        name = (r.get("company_name") or r.get("name") or "").strip()
        return f"{name} ({r['symbol']})" if name else r["symbol"]

    def timing(r):
        return (
            r.get("session")
            or SESSIONS.get(r.get("time"))
            or (r["timing"] if r["precision"] == "time" else "")
        )

    unknown = any(not timing(r) for r in events)
    note = UNKNOWN_HOURS if unknown else "Dates et horaires à confirmer."
    period = f"Du {start.day} au {end.day} {MONTHS[end.month - 1]} {end.year}"
    if start.month != end.month:
        period = f"Du {start.day} {MONTHS[start.month - 1]} au {end.day} {MONTHS[end.month - 1]} {end.year}"
    slides = [
        slide(
            f"{len(events)} résultats US cette semaine",
            "\n".join(company(r) for r in events[:3]),
            eyebrow=period,
            source=calendar_label(events[0]["source"]) + ("\n" + note if note else ""),
        )
    ]
    for offset in range(5):
        d = (start + timedelta(days=offset)).isoformat()
        group = [r for r in events if r["date"] == d]
        if not group:
            continue
        entries = [{"name": company(r), "timing": timing(r)} for r in group[:6]]
        remaining = len(group) - len(entries)
        text = "\n".join(e["name"] + (" · " + e["timing"] if e["timing"] else "") for e in entries)
        if remaining:
            text += f"\net {remaining} {'autre' if remaining == 1 else 'autres'}"
        slides.append(
            slide(
                human_date(d),
                text,
                source="\n".join(dict.fromkeys(calendar_label(r["source"]) for r in group)),
                entries=entries,
                remaining=remaining,
            )
        )
    slides.append(
        slide(
            "Gardez les prochaines dates en vue.",
            "Consultez les mises à jour et les heures confirmées avant chaque publication.",
            cta="Le calendrier : lien en bio",
            flip=True,
        )
    )
    value = post(
        "calendar",
        "Value-Stack",
        slides[0]["title"],
        slides,
        f"{period}. Sélection partielle. {note}",
        sources=[r["source"]["url"] for r in events],
        evidence=[
            {
                "date": r["date"],
                "symbol": r["symbol"],
                "company_name": r.get("company_name") or r.get("name"),
                "timing": timing(r),
                "source": r["source"],
            }
            for r in events
        ],
        provenance=x["_provenance"],
        cover_action="Le détail, jour par jour",
    )
    listing, previous_day = [], None
    for r in events:
        if r["date"] != previous_day:
            listing.extend(["", human_date(r["date"])])
            previous_day = r["date"]
        listing.append(company(r) + (" · " + timing(r) if timing(r) else ""))
    value["calendar_listing"] = "\n".join(listing).strip()
    return value


def documents(path, now):
    """Optional reviewed local projection; a headline alone is never four facts."""
    if not path or not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    x = json.loads(raw)
    if x.get("schema") != "carousel-reviewed-documents-v1" or x.get("provenance") != "observed":
        raise ValueError("unsupported document projection")
    result = []
    for doc in x["documents"]:
        if (
            doc["form"] not in ("8-K", "10-Q", "4")
            or doc["symbol"] not in x["well_known_us_symbols"]
        ):
            continue
        s = source(doc["source"], now)
        url = urlsplit(s["url"])
        if url.hostname != "www.sec.gov" or not url.path.startswith("/Archives/edgar/data/"):
            raise ValueError("official filing citation required")
        if not 0 <= (now - instant(s["available_at"])).total_seconds() <= 7 * 86400:
            continue
        if len(doc["facts"]) != 4 or any(
            not f.get("citation") or not f.get("text") for f in doc["facts"]
        ):
            raise ValueError("four cited document facts required")
        title = f"{doc['symbol']} : 4 points du {doc['form']} officiel"
        cards = [slide(title, "Ce que dit le document officiel.")]
        cards.extend(
            slide(f"Point {i + 1}", f["text"], source=label(s) + "\n" + f["citation"])
            for i, f in enumerate(doc["facts"])
        )
        cards.append(
            slide(
                "Revenir au document",
                "Les citations renvoient au dépôt officiel.",
                cta="Suivez le radar sur hyprlcore.com",
            )
        )
        result.append(
            post(
                "document",
                "Value-Stack",
                title,
                cards,
                f"{doc['symbol']} a publié un {doc['form']}. Quatre faits cités pour lire le dépôt.",
                sources=[s["url"]],
                evidence=[doc],
                provenance={"projection_sha256": hashlib.sha256(raw).hexdigest()},
            )
        )
    return result


def verdict(path, now):
    if not path or not Path(path).exists():
        return None
    raw = Path(path).read_bytes()
    r = json.loads(raw)
    if r.get("status") != "DONE":
        return None
    if (
        r.get("schema") != "asia-low-study-result-v1"
        or r["plan_sha256"] != plan_hash()
        or not DIGEST.fullmatch(r.get("manifest_sha256", ""))
    ):
        raise ValueError("unbound study result")
    chart = []
    for year in r["years"]:
        values = []
        for key in ("friday", "other_days", "random_day"):
            v = year[key]
            if v["n"] == 0:
                values.append(None)
            elif (
                type(v["n"]) is not int
                or type(v["k"]) is not int
                or not 0 <= v["k"] <= v["n"]
                or not math.isclose(v["rate"], v["k"] / v["n"])
            ):
                raise ValueError("invalid yearly rate")
            else:
                values.append(v["rate"])
        chart.append({"year": year["year"], "values": values})
    if not chart or not any(all(v is not None for v in y["values"]) for y in chart):
        return None
    proven = r.get("friday_advantage_established")
    decision = (
        "Effet vendredi distinguable du hasard"
        if proven is True
        else "Le hasard reste une explication possible"
    )
    stamp = "Étude HyprL · résultat calculé\nPublication non horodatée"
    ci = r.get("week_block_difference_ci95")
    detail = (
        f"Écart vendredi / autres jours : intervalle 95 % [{ci[0] * 100:.1f} ; {ci[1] * 100:.1f}] points."
        if ci
        else "Semaines complètes insuffisantes pour conclure."
    )
    totals = []
    for key in ("friday", "other_days", "random_day"):
        n = sum(y[key]["n"] for y in r["years"])
        k = sum(y[key]["k"] for y in r["years"])
        if not n:
            return None
        totals.append((k, n, k / n))
    bars = [
        {"label": name, "value": value[2]}
        for name, value in zip(("Vendredi", "Autres jours", "Jour tiré au hasard"), totals)
    ]
    cards = [
        slide(
            "Bitcoin : 60 % des vendredis ?",
            "Le plus bas d'Asie serait cassé plus tard. On a vérifié.",
            metric="60 %",
        ),
        slide(
            "Un seuil à franchir dans la journée.",
            "Le plus bas est fixé pendant la séance asiatique. Un simple contact ne compte pas.",
        ),
        slide(
            "Ce qu'on a mesuré",
            f"{totals[0][0]} vendredis sur {totals[0][1]}. " + "De 2018 à août 2026.",
            metric=f"{totals[0][2] * 100:.1f} %".replace(".", ","),
            source=stamp,
        ),
        slide("Comparons aux autres jours.", chart=bars, source=stamp),
        slide(
            decision,
            "Cette fréquence historique ne démontre pas des gains futurs.",
            source=stamp,
            flip=True,
        ),
        slide(
            "Une stat se juge avec sa comparaison.",
            "La règle, les données et les limites accompagnent le verdict.",
            cta="La méthode : lien en bio",
        ),
    ]
    value = post(
        "chance",
        "Problem-Proof",
        cards[0]["title"],
        cards,
        "Une affirmation sur le BTC confrontée à des jours comparables. Étude historique exploratoire. "
        + detail
        + " "
        + "Règle : 02 h à 11 h Paris, franchissement strict après 11 h. Les jours incomplets sont exclus.",
        evidence=[r],
        provenance={"result_sha256": hashlib.sha256(raw).hexdigest()},
    )
    value["hashtags"] = ["#bitcoin", "#bourse", "#investir"]
    return value
