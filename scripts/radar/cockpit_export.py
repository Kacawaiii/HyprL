"""Read-only, sanitized JSON snapshots for the cockpit Radar home and Paper pages.

Inputs are files the other jobs already wrote: the radar report, the trader evidence store (read only), the
paper reports and the Claude book journal. Nothing here calls a broker or a data source, and nothing is
written back to those inputs. Output is two JSON files, `radar-home.json` and `paper.json`, that the read-only
application API serves from a private directory (never from Git).

What is kept: headline, link and publisher of a story (no article text), point-in-time stamps, hashes,
scores, model views and their reviews, outcomes, positions with stops/targets, equity points.
What is refused: tokens, keys, e-mails, private paths, full account numbers (suffix only).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HOME_SCHEMA = "cockpit-radar-home-v1"
PAPER_SCHEMA = "cockpit-paper-v1"
TOP_EVENTS = 40
MAX_ASSETS = 8
MAX_TIMELINE = 12
MAX_JOURNAL = 60
MAX_EQUITY_POINTS = 500
ANALYSTS = ("analyst_claude", "analyst_gpt", "reviewer_claude", "reviewer_gpt", "consensus")
SECRET_PARAMS = re.compile(r"key|token|secret|sig|auth|pass", re.I)
# A secret is refused wherever it hides, even in a free-text field.
SECRET_TEXT = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\b(?:PK|AK|SK)[A-Z0-9]{16,}\b"),
    re.compile(r"\bBearer\s+\S+", re.I),
    re.compile(r"(?:api[_-]?key|secret|password|token)\s*[=:]\s*\S+", re.I),
    re.compile(r"/home/[a-z_][\w-]*", re.I),
]
FORBIDDEN_KEYS = {"account_id", "account_number", "api_key", "secret", "token", "password", "email", "summary_raw",
                  "body", "article", "content", "order_id", "id_order"}


class ExportError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def clean_url(url):
    """Keep the link, drop any query parameter that looks like a credential."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not SECRET_PARAMS.search(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))[:500]


def text(value, limit=400):
    return value[:limit] if isinstance(value, str) else None


def number(value, digits=6):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, digits) if out == out and abs(out) != float("inf") else None


def norm_symbol(symbol) -> str:
    """BTC/USD, BTC-USD and BTCUSD name one asset; the trader uses the dashed form."""
    s = str(symbol or "").upper().replace("/", "").replace("-", "")
    return s


def assert_clean(value, path="$"):
    """Raise ExportError on a forbidden key or a secret-shaped string anywhere in the snapshot."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise ExportError(f"forbidden field {key} at {path}")
            assert_clean(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_clean(item, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in SECRET_TEXT:
            if pattern.search(value):
                raise ExportError(f"secret-shaped text at {path}")


# ---------------------------------------------------------------- trader store

def _scan(store, kind):
    after = 0
    while True:
        page = store.records(kind, after=after, limit=1000)
        if not page:
            return
        yield from page
        after = page[-1]["sequence"]


def load_trader(store, scorecard=None, include_synthetic=False):
    """Plain dict from an open read-only ResearchStore (or anything with `records`)."""
    runs = []
    for record in _scan(store, "replay-summary"):
        payload = record["payload"]
        if payload.get("schema") != "trader-run-v1" or (payload.get("synthetic") and not include_synthetic):
            continue
        decision = payload.get("decision") or {}
        runs.append({"run_id": payload["run_id"], "at": payload["at"], "status": payload.get("status"),
                     "recorded_at": record.get("recorded_at"), "synthetic": bool(payload.get("synthetic")),
                     "preregistration_hash": decision.get("preregistration_hash"),
                     "context_hash": decision.get("context_hash"), "models": decision.get("models") or {},
                     "views": decision.get("views") or []})
    predictions = {}
    for record in _scan(store, "prediction"):
        p = record["payload"]
        if p.get("synthetic") and not include_synthetic:
            continue
        view = (p.get("signal") or {}).get("view") or {}
        predictions[p["prediction_id"]] = {"model_id": p["model_id"], "asset": p["product"],
                                           "horizon": (p.get("signal") or {}).get("label_definition", {}).get("horizon"),
                                           "run_id": (p.get("signal") or {}).get("run_id"),
                                           "decision_at": p.get("decision_at"), "view": view.get("view")}
    labels = []
    for record in _scan(store, "label"):
        p = record["payload"]
        prediction = predictions.get(p["prediction_id"])
        if prediction is None:
            continue
        value = p.get("value") or {}
        labels.append({**prediction, "realized_at": p.get("realized_at"), "available_at": p.get("available_at"),
                       "raw_return": number(value.get("raw_return")),
                       "spy_relative_return": number(value.get("spy_relative_return")),
                       "net_unit_pnl": number(value.get("net_unit_pnl")),
                       "cost_roundtrip": number(value.get("cost_roundtrip"))})
    return {"runs": runs, "labels": labels, "scorecard": scorecard}


def load_scorecard(store, synthetic=False):
    """The trader scorecard, or None when its (ML-extra) module cannot be imported here."""
    try:
        from scripts.trading_lab.trader_agent.scoring import scorecard
    except ImportError:
        return None
    return scorecard(store, synthetic=synthetic)


# ------------------------------------------------------------------ home build

def _quality(scorecard):
    """Predictive quality per analyst, straight from the scorecard. Never merged with any other score."""
    if not scorecard:
        return {}
    out = {}
    for name, entry in (scorecard.get("scores") or {}).items():
        out[name] = {"issued": entry.get("issued"), "realized": entry.get("realized"),
                     "hit_rate": number(entry.get("hit_rate")), "brier": number(entry.get("brier")),
                     "climatology_brier": number(entry.get("climatology_brier")), "ic": number(entry.get("ic")),
                     "mean_unit_pnl_after_costs": number(entry.get("mean_unit_pnl_after_costs")),
                     "days": entry.get("days")}
    return out


def _asset_class(symbol) -> str:
    s = str(symbol or "")
    return "crypto" if "/" in s or s.endswith("-USD") or s.endswith("USD") and len(s) > 5 else "equity_etf"


def _quality_for(quality, classes):
    """`analyst/class/horizon/target` scorecard keys narrowed to the raw target of the classes an event touches."""
    out = {}
    for key, entry in quality.items():
        parts = key.split("/")
        if len(parts) == 4 and parts[1] in classes and parts[3] == "raw" and parts[0] in ANALYSTS:
            out[f"{parts[0]}/{parts[2]}"] = entry
    return out or None


def _views_by_asset(runs):
    """asset -> run index list of {analyst,horizon,view,p,verdict}, chronological."""
    table = defaultdict(list)
    for run in sorted(runs, key=lambda r: r["at"]):
        by_asset = defaultdict(list)
        for v in run["views"]:
            raw = v.get("raw_view") or {}
            review = v.get("review") or {}
            by_asset[norm_symbol(v.get("asset"))].append({
                "analyst": v.get("analyst"), "horizon": v.get("horizon"), "view": v.get("view"),
                "p_outperform": number(v.get("p_outperform"), 4), "verdict": v.get("verdict"),
                "reason": text(raw.get("confidence_reason"), 280), "falsifier": text(raw.get("falsifier"), 200),
                "review_note": text(review.get("note"), 200),
                "catalysts": [{"url": clean_url(c.get("url")), "published_at": c.get("published_at")}
                              for c in (raw.get("catalysts") or [])[:3] if clean_url(c.get("url"))]})
        for asset, views in by_asset.items():
            table[asset].append({"run_id": run["run_id"], "at": run["at"], "status": run["status"], "views": views})
    return table


def _anticipation(asset, table, runs):
    history = table.get(norm_symbol(asset), [])
    if not history:
        return {"state": "NOT_COVERED", "latest": None, "timeline": []}
    latest = history[-1]
    timeline = []
    for entry in history[-MAX_TIMELINE:]:
        point = {"run_id": entry["run_id"], "at": entry["at"], "by_analyst": {}}
        for v in entry["views"]:
            if v["horizon"] == "1d" or v["analyst"] not in point["by_analyst"]:
                point["by_analyst"][v["analyst"]] = {"view": v["view"], "p_outperform": v["p_outperform"],
                                                     "verdict": v["verdict"], "horizon": v["horizon"]}
        timeline.append(point)
    return {"state": "COVERED", "latest": {"run_id": latest["run_id"], "at": latest["at"], "views": latest["views"]},
            "timeline": timeline}


def _outcomes(asset, labels):
    key = norm_symbol(asset)
    rows = [l for l in labels if norm_symbol(l["asset"]) == key]
    rows.sort(key=lambda l: l.get("realized_at") or "")
    return [{"model_id": l["model_id"].split(":")[-1], "run_id": l["run_id"], "horizon": l["horizon"],
             "view": l["view"], "realized_at": l["realized_at"], "available_at": l["available_at"],
             "raw_return": l["raw_return"], "spy_relative_return": l["spy_relative_return"],
             "net_unit_pnl": l["net_unit_pnl"], "cost_roundtrip": l["cost_roundtrip"]} for l in rows[-6:]]


def _event(rank, event, table, runs, labels, quality, paper_after_cost):
    scenario = event.get("scenario") or None
    exposures = {norm_symbol(x.get("symbol")): x for x in (scenario or {}).get("exposures", [])}
    symbols, seen = [], set()
    for h in event.get("transmission_hypotheses") or []:
        key = norm_symbol(h.get("symbol"))
        if key and key not in seen:
            seen.add(key)
            symbols.append((h.get("symbol"), h.get("role"), h.get("mechanism")))
    for key, x in exposures.items():
        if key not in seen:
            seen.add(key)
            symbols.append((x.get("symbol"), x.get("role"), x.get("mechanism")))
    priced = {norm_symbol(m.get("symbol")): m for m in (event.get("priced_in") or {}).get("measurements", [])}
    assets = []
    for symbol, role, mechanism in symbols[:MAX_ASSETS]:
        key = norm_symbol(symbol)
        measured = priced.get(key)
        assets.append({
            "symbol": text(symbol, 24), "role": text(role, 40), "mechanism": text(mechanism, 300),
            "direction": text(exposures.get(key, {}).get("direction"), 20),
            "priced_in": None if not measured else {
                "return_pct": number(measured.get("return_pct")), "move_atr": number(measured.get("move_atr")),
                "baseline_at": measured.get("baseline_at"), "price_at": measured.get("price_at"),
                "note": text(measured.get("interpretation"), 200)},
            "anticipation": _anticipation(symbol, table, runs),
            "outcomes": _outcomes(symbol, labels)})
    stories = []
    for s in (event.get("stories") or [])[:5]:
        url = clean_url(s.get("url") or s.get("source_url"))
        if url:
            stories.append({"publisher": text(s.get("publisher") or s.get("source"), 80), "url": url,
                            "headline": text(s.get("headline"), 300), "published_at": s.get("published_at"),
                            "received_at": s.get("received_at"), "primary": bool(s.get("primary"))})
    evidence = event.get("evidence") or {}
    covered = [a for a in assets if a["anticipation"]["state"] == "COVERED"]
    convictions = {}
    for a in covered:
        for v in a["anticipation"]["latest"]["views"]:
            if v["analyst"] in ANALYSTS and v["p_outperform"] is not None:
                convictions.setdefault(v["analyst"], []).append(abs(v["p_outperform"] - 0.5) * 2)
    outcomes = [o for a in assets for o in a["outcomes"] if o["net_unit_pnl"] is not None]
    return {
        "id": event["id"], "rank": rank, "headline": text(event.get("headline"), 300),
        "link": stories[0]["url"] if stories else None,
        "source": stories[0]["publisher"] if stories else None,
        "published_at": event.get("published_at"), "available_at": event.get("first_received_at"),
        "novelty": event.get("novelty"),
        "themes": [text(t, 40) for t in (event.get("entities") or {}).get("themes", [])][:6],
        "countries": [text(t, 40) for t in (event.get("entities") or {}).get("countries", [])][:6],
        "badges": {
            "event_importance": {"value": number(event.get("importance"), 1),
                                 "components": event.get("importance_components")},
            "evidence_strength": {"value": number(evidence.get("score"), 1), "status": evidence.get("status"),
                                  "publishers": len(evidence.get("publishers") or []),
                                  "primary": bool(evidence.get("primary"))},
            "model_conviction": {"label": (scenario or {}).get("conviction") or event.get("conviction"),
                                 "by_analyst": {k: round(sum(v) / len(v), 3) for k, v in convictions.items()} or None,
                                 "basis": "abs(p_outperform - 0.5) * 2, latest run, covered assets"},
            "predictive_quality": {
                "by_analyst": _quality_for(quality, {_asset_class(a["symbol"]) for a in covered}) if covered else None,
                "basis": "trader scorecard, raw target, class of the covered assets; sample sizes in each entry"},
            "after_cost_performance": {
                "event_labels": None if not outcomes else {
                    "n": len(outcomes), "mean_net_unit_pnl": round(sum(o["net_unit_pnl"] for o in outcomes) / len(outcomes), 6)},
                "paper_accounts": paper_after_cost,
                "basis": "labelled net_unit_pnl for this event's assets; account return since paper start"}},
        "what_changed": {
            "expectations": text(event.get("expectations"), 300),
            "changed_expectations": text((scenario or {}).get("changed_expectations"), 400),
            "summary": text((scenario or {}).get("summary"), 400),
            "impact": text((scenario or {}).get("impact"), 400),
            "horizon": text((scenario or {}).get("horizon"), 200),
            "priced_in": text((scenario or {}).get("priced_in"), 300),
            "invalidation": text((scenario or {}).get("invalidation"), 300),
            "source": "llm_scenario" if scenario else "rules_only"},
        "assets": assets, "stories": stories,
        "provenance": {"priced_in_status": (event.get("priced_in") or {}).get("status"),
                       "independence": text(evidence.get("independence"), 200),
                       "retail_hype": number((event.get("retail_hype") or {}).get("score"), 1)}}


def _paper_after_cost(paper):
    out = []
    for account in (paper or {}).get("accounts", []):
        out.append({"account": account["account"], "suffix": account["suffix"],
                    "return_since_start": account.get("return_since_start")})
    return out


def build_home(radar, trader, paper=None, *, generated_at=None, top=TOP_EVENTS):
    runs = trader.get("runs", []) if trader else []
    labels = trader.get("labels", []) if trader else []
    quality = _quality((trader or {}).get("scorecard"))
    table = _views_by_asset(runs)
    ranked = sorted(radar.get("events", []), key=lambda e: (-(e.get("importance") or 0),
                    -((e.get("evidence") or {}).get("score") or 0), str(e.get("first_received_at")), e["id"]))
    after_cost = _paper_after_cost(paper)
    events = [_event(i + 1, e, table, runs, labels, quality, after_cost) for i, e in enumerate(ranked[:top])]
    regime = {}
    for name, item in (radar.get("regime") or {}).items():
        if isinstance(item, dict):
            regime[name] = {"symbol": item.get("symbol"), "last": number(item.get("last")), "at": item.get("at"),
                            "returns_pct": {k: number(v) for k, v in (item.get("returns_pct") or {}).items()},
                            "status": item.get("status")}
    sources = defaultdict(int)
    for s in radar.get("sources", []):
        sources[s.get("status")] += 1
    snapshot = {
        "schema": HOME_SCHEMA, "generated_at": generated_at or utc_now(),
        "radar": {"date": radar.get("date"), "slot": radar.get("slot"), "cutoff": radar.get("cutoff"),
                  "status": radar.get("status"), "report_hash": digest(radar), "total_events": len(radar.get("events", [])),
                  "shown_events": len(events), "sources_by_status": dict(sources),
                  "limitations": [text(x, 200) for x in radar.get("limitations", [])][:8]},
        "trader": {"runs": len(runs), "labels": len(labels), "latest_run": runs[-1]["run_id"] if runs else None,
                   "latest_run_at": runs[-1]["at"] if runs else None,
                   "models": runs[-1]["models"] if runs else {},
                   "preregistration_hash": runs[-1].get("preregistration_hash") if runs else None,
                   "predictive_quality": quality,
                   "hypothesis_state": ((trader or {}).get("scorecard") or {}).get("hypothesis_state")},
        "regime": regime, "events": events}
    assert_clean(snapshot)
    return snapshot


# ----------------------------------------------------------------- paper build

def _engine_index(journal):
    """symbol -> last buy intent: stop, target, engine and the reasons the Claude book wrote down."""
    out = {}
    for row in journal:
        if row.get("action") == "buy_intent" and row.get("symbol"):
            j = row.get("journal") or {}
            out[norm_symbol(row["symbol"])] = {
                "at": row.get("at"), "stop": number(row.get("stop")), "target": number(row.get("target")),
                "limit": number(row.get("limit")), "risk_usd": number(row.get("risk_usd")),
                "engine": text(j.get("engine"), 40),
                "event": text(j.get("event"), 300), "mechanism": text(j.get("mechanism"), 300),
                "priced_in": text(j.get("priced_in"), 300), "scenario": text(j.get("scenario"), 300),
                "invalidation": text(j.get("invalidation"), 300)}
    return out


def _journal_rows(journal):
    out = []
    for row in journal[-MAX_JOURNAL:]:
        action = row.get("action")
        if action not in ("buy_intent", "close", "note", "flatten", "submitted", "post_mortem"):
            continue
        j = row.get("journal") or {}
        out.append({"at": row.get("at"), "action": action, "symbol": text(row.get("symbol"), 20),
                    "qty": number(row.get("qty")), "limit": number(row.get("limit")), "stop": number(row.get("stop")),
                    "target": number(row.get("target")), "engine": text(j.get("engine"), 40),
                    "reason": text(j.get("event") or row.get("reason") or row.get("start") or row.get("lesson"), 300),
                    "mechanism": text(j.get("mechanism"), 300), "invalidation": text(j.get("invalidation"), 300)})
    return out


def build_claude_book(status, journal, sleeve=None, history=None):
    index = _engine_index(journal)
    sleeve_symbols = {norm_symbol(k) for k in (sleeve or {})}
    positions = []
    for p in status.get("positions", []):
        key = norm_symbol(p.get("symbol"))
        intent = index.get(key, {})
        tag = "sleeve" if key in sleeve_symbols else (intent.get("engine") or "discretionary")
        entry = number(p.get("avg_entry_price"), 6)
        positions.append({
            "symbol": text(p.get("symbol"), 20), "asset_class": p.get("asset_class"), "tag": tag,
            "qty": number(p.get("qty"), 8), "entry": entry, "last": number(p.get("current_price"), 6),
            "unrealized_pl": number(p.get("unrealized_pl"), 2), "unrealized_pct": number(p.get("unrealized_plpc"), 4),
            "stop": intent.get("stop"), "target": intent.get("target"),
            "protection": "policy" if intent.get("stop") is not None else "none_defined",
            "opened_at": intent.get("at"),
            "reason": {k: intent.get(k) for k in ("event", "mechanism", "priced_in", "scenario", "invalidation")}
            if intent else None})
    orders = [{"symbol": text(o.get("symbol"), 20), "side": o.get("side"), "qty": number(o.get("qty"), 8),
               "type": o.get("type"), "limit": number(o.get("limit")), "status": o.get("status"),
               "submitted": o.get("submitted"),
               "legs": [{"type": l.get("type"), "limit": number(l.get("limit")), "stop": number(l.get("stop"))}
                        for l in o.get("legs") or []]} for o in status.get("open_orders", [])]
    return {"account": "claude_book", "suffix": "2EQN", "label": "Claude book",
            "equity": number(status.get("equity"), 2), "start_equity": number(status.get("start_equity"), 2),
            "cash": number(status.get("cash"), 2), "peak": number(status.get("peak"), 2),
            "return_since_start": number((number(status.get("equity")) or 0) / number(status.get("start_equity")) - 1, 6)
            if number(status.get("start_equity")) else None,
            "positions": positions, "open_orders": orders,
            "tags": sorted({p["tag"] for p in positions}), "journal": _journal_rows(journal)}


def build_ai_accounts(report):
    out = []
    for a in report.get("accounts", []):
        out.append({"account": a["account"], "suffix": str(a.get("account_suffix", ""))[-4:],
                    "label": {"ia_actions": "AI stocks", "ia_crypto": "AI crypto"}.get(a["account"], a["account"]),
                    "equity": number(a.get("equity"), 2), "pnl": number(a.get("pnl"), 2),
                    "day_pnl": number(a.get("day_pnl"), 2), "peak": number(a.get("peak"), 2),
                    "open_lots": a.get("open_lots"), "halted": bool(a.get("halted")),
                    "return_since_start": number(a.get("return_since_paper_start")),
                    "planned_orders": len(a.get("planned_orders") or []), "positions": [], "journal": []})
    return out


def append_history(path: Path, point: dict, keep=2000):
    """Private equity history, one JSON line per export; this is where the P&L curve comes from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows or rows[-1].get("at") != point["at"]:
        rows.append(point)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows[-keep:]) + "\n")
    return rows[-keep:]


def build_paper(report, book_status, journal, *, sleeve=None, history=None, radar=None, generated_at=None):
    at = generated_at or utc_now()
    accounts = build_ai_accounts(report)
    book = build_claude_book(book_status, journal, sleeve)
    accounts.append(book)
    bench = report.get("benchmarks") or {}
    btc = ((radar or {}).get("regime") or {}).get("BTC") or {}
    benchmarks = {"SPY": {"price": number((bench.get("SPY") or {}).get("price"), 2),
                          "return_since_paper_start": number((bench.get("SPY") or {}).get("return_since_paper_start")),
                          "base_at": (bench.get("SPY") or {}).get("base_at")},
                  "BTC": {"price": number(btc.get("last"), 2), "at": btc.get("at")}}
    curve = [{"at": h["at"], "equity": h.get("equity", {}), "spy": h.get("spy"), "btc": h.get("btc")}
             for h in (history or [])][-MAX_EQUITY_POINTS:]
    snapshot = {"schema": PAPER_SCHEMA, "generated_at": at, "report_at": report.get("at"),
                "accounts": accounts, "benchmarks": benchmarks, "curve": curve,
                "limitations": [text(x, 200) for x in report.get("limitations", [])][:6],
                "notes": ["paper accounts only; the Claude book is not part of any AI trader variant or baseline"]}
    assert_clean(snapshot)
    return snapshot


def history_point(paper_snapshot, at=None):
    return {"at": at or paper_snapshot["generated_at"],
            "equity": {a["account"]: a["equity"] for a in paper_snapshot["accounts"] if a.get("equity") is not None},
            "spy": paper_snapshot["benchmarks"]["SPY"]["price"], "btc": paper_snapshot["benchmarks"]["BTC"]["price"]}


# ----------------------------------------------------------------------- main

def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_snapshot(directory: Path, name: str, snapshot: dict):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    temporary = directory / (name + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    return target


def main(argv=None):  # pragma: no cover - thin CLI over the functions above
    import argparse
    parser = argparse.ArgumentParser(description="Write sanitized cockpit snapshots (read-only over its inputs)")
    parser.add_argument("--radar", required=True, help="radar report JSON")
    parser.add_argument("--trader-root", help="trader runtime directory (evidence store is opened read only)")
    parser.add_argument("--paper-report", help="alpaca paper report JSON")
    parser.add_argument("--book-status", help="JSON written by `claude-book status`")
    parser.add_argument("--book-journal", help="claude book journal.jsonl")
    parser.add_argument("--book-sleeve", help="claude book sleeve.json")
    parser.add_argument("--history", help="private equity history jsonl (appended)")
    parser.add_argument("--out", required=True, help="private output directory, outside Git")
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[2]
    if repo in out.parents or out == repo:
        raise SystemExit("refusing to write snapshots inside the repository")
    radar = read_json(args.radar)
    trader = None
    if args.trader_root:
        from scripts.trading_lab.research.store import ResearchStore
        store = ResearchStore(Path(args.trader_root) / "evidence", read_only=True)
        try:
            trader = load_trader(store, load_scorecard(store, args.include_synthetic), args.include_synthetic)
        finally:
            store.close()
    paper = None
    if args.paper_report and args.book_status and args.book_journal:
        history = []
        if args.history:
            history = read_jsonl(args.history) if Path(args.history).is_file() else []
        paper = build_paper(read_json(args.paper_report), read_json(args.book_status), read_jsonl(args.book_journal),
                            sleeve=read_json(args.book_sleeve) if args.book_sleeve else None, history=history,
                            radar=radar)
        if args.history:
            history = append_history(Path(args.history), history_point(paper))
            paper["curve"] = [{"at": h["at"], "equity": h["equity"], "spy": h.get("spy"), "btc": h.get("btc")}
                              for h in history][-MAX_EQUITY_POINTS:]
        write_snapshot(out, "paper.json", paper)
    write_snapshot(out, "radar-home.json", build_home(radar, trader, paper))
    print(json.dumps({"written": ["radar-home.json"] + (["paper.json"] if paper else [])}))


if __name__ == "__main__":  # pragma: no cover
    main()
