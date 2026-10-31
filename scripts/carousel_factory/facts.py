"""Daily fact selection from existing verified local projections; never fetch sources."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scripts.video_factory.market import digest, instant, source

from .data import NY, label

# Editorial scope, not a market-cap claim or an assertion of exhaustive coverage.
COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "WMT": "Walmart",
    "V": "Visa",
    "MA": "Mastercard",
    "UNH": "UnitedHealth",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "COST": "Costco",
    "ABBV": "AbbVie",
    "BAC": "Bank of America",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "INTC": "Intel",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "DIS": "Disney",
    "PFE": "Pfizer",
    "QCOM": "Qualcomm",
    "UBER": "Uber",
    "PLTR": "Palantir",
    "LLY": "Eli Lilly",
    "BRK.B": "Berkshire Hathaway",
}


def official(kind, citation, day):
    """Reject unrelated news and URL lookalikes even when upstream says observed."""
    url = urlsplit(citation["url"])
    if kind == "earnings":
        return (
            url.hostname == "api.nasdaq.com"
            and url.path == "/api/calendar/earnings"
            and parse_qs(url.query) == {"date": [day]}
        )
    if kind == "fed":
        return url.hostname == "www.federalreserve.gov" and url.path.startswith(
            ("/monetarypolicy/", "/newsevents/")
        )
    if kind in ("cpi", "nfp"):
        return url.hostname == "www.bls.gov" and url.path.startswith(
            ("/news.release/", "/schedule/")
        )
    if kind == "sec":
        return url.hostname == "www.sec.gov" and url.path.startswith("/Archives/edgar/data/")
    return False


def select(x, rows, day, documents_path=None):
    """Only today's events/releases, with per-fact evidence and deterministic IDs."""
    if not x:
        return None
    date.fromisoformat(day)
    cutoff = instant(x["generated_at"])
    items = []

    def add(kind, text, citation, symbol=None, timing=None, identity=None):
        if not official(kind, citation, day):
            return
        if symbol and symbol not in COMPANIES:
            return
        # A missing source publication clock stays missing; collection is not publication.
        if citation.get("published_at"):
            source(citation, cutoff)
        elif not (citation.get("event_sha256") and citation.get("dataset_sha256")):
            return
        fact = {
            "kind": kind,
            "date": day,
            "symbol": symbol,
            "company": COMPANIES.get(symbol),
            "text": text,
            "timing": timing,
            "source": citation,
        }
        fact["id"] = digest(
            {
                "kind": kind,
                "symbol": symbol,
                "source": citation["sha256"],
                "text": text,
                "identity": identity,
            }
        )[:20]
        if not any(i["id"] == fact["id"] for i in items):
            items.append(fact)

    for r in sorted(rows, key=lambda r: (r["kind"] == "earnings", r.get("symbol") or "")):
        if r["date"] != day:
            continue
        symbol = r.get("symbol")
        name = COMPANIES.get(symbol, symbol)
        title = f"{name} ({symbol}) : résultats" if r["kind"] == "earnings" else r["title"]
        add(r["kind"], f"{title} le {day}, {r['timing']}.", r["source"], symbol, r["timing"])
    for brief in x.get("briefs", []):
        if brief["date"] != day:
            continue
        published = instant(brief["published_at"])
        if published > cutoff or published.astimezone(NY).date().isoformat() != day:
            raise ValueError("brief date mismatch")
        for f in brief["facts"]:
            citation = source(f["source"], published)
            # A reprinted old article is not today's fact.
            if instant(citation["published_at"]).astimezone(NY).date().isoformat() != day:
                continue
            kind = "sec" if f["kind"] in ("sec", "8-K", "10-Q", "4") else f["kind"]
            if kind == "sec" and not f.get("symbol"):
                continue
            add(kind, f["summary"], citation, f.get("symbol"))
    if documents_path and Path(documents_path).exists():
        projection = json.loads(Path(documents_path).read_text())
        if (
            projection.get("schema") != "carousel-reviewed-documents-v1"
            or projection.get("provenance") != "observed"
        ):
            raise ValueError("unsupported document projection")
        for doc in projection["documents"]:
            citation = source(doc["source"], cutoff)
            if (
                doc["symbol"] not in COMPANIES
                or doc["form"] not in ("8-K", "10-Q", "4")
                or instant(citation["published_at"]).astimezone(NY).date().isoformat() != day
            ):
                continue
            for fact in doc["facts"]:
                if not fact.get("citation"):
                    continue
                add(
                    "sec",
                    f"{doc['symbol']} ({doc['form']}) : {fact['text']} [{fact['citation']}]",
                    citation,
                    doc["symbol"],
                    identity=fact["citation"],
                )
    for row in x.get("radar", []):
        citation = source(row["source"], cutoff)
        if instant(citation["published_at"]).astimezone(NY).date().isoformat() != day:
            continue
        kind = (
            "sec"
            if row["kind"] in ("sec_8k", "sec_10q", "sec_form4", "sec", "8-K", "10-Q", "4")
            else row["kind"]
        )
        if kind == "sec" and not row.get("symbol"):
            continue
        add(kind, row["summary"], citation, row.get("symbol"))
    chosen = items[:3]
    if not chosen:
        return None
    return {
        "schema": "carousel-daily-facts-v2",
        "date": day,
        "count": len(chosen),
        "facts": chosen,
        "provenance": x["_provenance"],
    }


def to_post(facts, copy):
    """All sources, dates, count and CTA are supplied by code after copy validation."""
    from .content import post, slide

    cards = [
        slide(
            copy["hook"],
            f"Édition du {facts['date']}. Les publications et rendez-vous du jour.",
            metric=str(facts["count"]),
        )
    ]
    for item, fact in zip(copy["slides"], facts["facts"]):
        cards.append(slide(item["title"], item["body"], source=label(fact["source"])))
    cards[-1]["flip"] = True
    cards.append(
        slide(
            "Les faits gardent leurs sources.",
            "Retrouvez les documents d'origine et le calendrier dans le brief HyprL.",
            cta="Le brief : lien en bio",
        )
    )
    return post(
        "market",
        "Value-Stack",
        copy["hook"],
        cards,
        copy["hook"] + f" Édition du {facts['date']}.",
        sources=[f["source"]["url"] for f in facts["facts"]],
        evidence=facts["facts"],
        provenance={"facts_sha256": digest(facts), **facts["provenance"]},
    )
