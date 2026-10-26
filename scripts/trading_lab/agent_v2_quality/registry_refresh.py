"""Rebuild the admitted registry from granted sources without changing source shapes."""

from collections import Counter, defaultdict
from datetime import timedelta
from hashlib import sha256
import json
import re
from pathlib import Path
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parents[3]
from scripts.trading_lab.agent_v2.core import ranking
from scripts.trading_lab.agent_v2.inputs import previous_session, sessions
from scripts.trading_lab.agent_v2_quality.contracts import Blocked, digest, instant, iso, utc
from scripts.trading_lab.agent_v2_quality.producer import (
    atomic_json,
    calendar,
    read_local,
    registry,
    radar_headlines,
    radar_projection,
)
from scripts.trading_lab.agent_v2_quality.registry import admissible_registry, select_scope
from scripts.trading_lab.agent_v2_quality.spec import SPEC_HASH, PREREG_HASH, implementation_hash


def load(p):
    return read_local(p)


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def build(
    output, evidence_dir, radar_source, original, seed_registry, journal, client, *, clock=utc
):
    ROOT = Path(output)
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=False)

    def save(n, v):
        atomic_json(ROOT / n, v)

    source_dir = ROOT / "sources"
    source_dir.mkdir(mode=0o700)

    # Each completed exact request is reused with its original receipt. Failed reservations
    # remain in the durable journal and are not retried by this builder.
    def get_json(url, name):
        body = source_dir / (name + ".body")
        rp = source_dir / (name + ".receipt.json")
        if rp.exists():
            r = load(rp)
            if r.get("status") != 200 or not body.exists() or sha(body) != r["sha256"]:
                raise Blocked("CACHED_SOURCE_INVALID")
            return load(body), r
        if any(r["body"]["url"] == url for r in journal.records("http_reservation")):
            raise Blocked("PREVIOUS_REQUEST_NOT_RETRIED")
        raw, r = client.get(url)
        body.write_bytes(raw)
        save("sources/" + name + ".receipt.json", r)
        return json.loads(raw), r

    try:
        now = clock()
        hunter = load(original["hunter"])
        oldreg = load(seed_registry)
        members = sorted(r["symbol"] for r in oldreg["instruments"] if r["momentum_member"])
        previous = previous_session(now)
        end = sessions(previous, previous).iloc[0].market_close.to_pydatetime() + timedelta(
            seconds=1
        )
        params = {
            "symbols": ",".join(members),
            "timeframe": "1Day",
            "start": iso(now - timedelta(days=520)),
            "end": iso(end),
            "feed": "iex",
            "adjustment": "raw",
            "limit": 10000,
            "sort": "asc",
        }
        bars = defaultdict(list)
        receipts = []
        seen = set()
        for index in range(20):
            data, receipt = get_json(
                "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params),
                "momentum-bars-" + str(index).zfill(2),
            )
            receipts.append(receipt)
            for symbol, rows in data["bars"].items():
                if symbol not in members:
                    raise Blocked("RANKING_SYMBOL_INVALID")
                bars[symbol].extend({k: b[k] for k in ("t", "c", "h", "l", "v")} for b in rows)
            token = data.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise Blocked("RANKING_PAGINATION_LOOP")
            seen.add(token)
            params["page_token"] = token
        else:
            raise Blocked("RANKING_PAGINATION_INCOMPLETE")
        eligible = {}
        missing = []
        for symbol in members:
            rows = bars.get(symbol, [])
            if len(rows) < 200 or rows[-1]["t"][:10] != previous:
                missing.append(
                    {
                        "symbol": symbol,
                        "reason": "PREVIOUS_SESSION_HISTORY_UNAVAILABLE",
                        "bars": len(rows),
                    }
                )
                continue
            # Prove order, complete prior sessions, positive fields and no future bars.
            dates = [r["t"][:10] for r in rows]
            if dates != sorted(set(dates)) or any(
                not (0 < r["l"] <= r["c"] <= r["h"]) or r["v"] <= 0 for r in rows[-300:]
            ):
                missing.append(
                    {"symbol": symbol, "reason": "INVALID_OBSERVED_RANKING_BARS", "bars": len(rows)}
                )
                continue
            eligible[symbol] = {"asset": "equity", "momentum_member": True, "bars": rows[-300:]}
        ranks = list(ranking(eligible).values())
        save(
            "momentum-ranking.json",
            {
                "previous_session": previous,
                "received_at": max(r["received_at"] for r in receipts),
                "seed_registry_sha256": sha(seed_registry),
                "membership_sha256": digest(members),
                "membership_count": len(members),
                "ranked_count": len(ranks),
                "unavailable": missing,
                "ranking": ranks,
                "receipts": receipts,
                "feed": "iex",
                "adjustment": "raw",
            },
        )
        scope, selection = select_scope(ranks, [r["symbol"] for r in hunter["candidates"]])
        metadata = {r["symbol"]: r for r in oldreg["instruments"]}
        earnings = defaultdict(list)
        for e in load(Path(evidence_dir) / "earnings-evidence.json"):
            earnings[e["symbol"]].append(e)
            # Use the newly acquired granted-source cap when present; never estimate.
            cap = e.get("wire_row", {}).get("marketCap")
            if (
                isinstance(cap, str)
                and re.fullmatch(r"\$[0-9,]+(?:\.[0-9]+)?", cap)
                and e["symbol"] in metadata
            ):
                metadata[e["symbol"]]["market_cap"] = float(cap.replace("$", "").replace(",", ""))
                metadata[e["symbol"]]["provenance"]["market_cap"] = {
                    "source_hash": e["source_hash"],
                    "received_at": e["received_at"],
                    "value": metadata[e["symbol"]]["market_cap"],
                }
        submissions = {}
        sec_receipts = {}
        # Exclude undated names before acquiring SIC; no speculative bulk submissions.
        for symbol in scope:
            if not earnings[symbol] or not metadata.get(symbol, {}).get("cik"):
                continue
            cik = metadata[symbol]["cik"]
            url = "https://data.sec.gov/submissions/CIK" + cik + ".json"
            payload, receipt = get_json(url, "submissions-" + cik)
            submissions[symbol] = {"cik": payload.get("cik"), "sic": payload.get("sic")}
            sec_receipts[symbol] = receipt
        extras = sorted((set(scope) - set(members)) | {"SPY"})
        extra_params = {k: v for k, v in params.items() if k != "page_token"}
        extra_params["symbols"] = ",".join(extras)
        extra_seen = set()
        for page in range(20):
            extra, receipt = get_json(
                "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(extra_params),
                "extra-bars-" + str(page).zfill(2),
            )
            for symbol, rs in extra["bars"].items():
                if symbol not in extras:
                    raise Blocked("EXTRA_HISTORY_SYMBOL_INVALID")
                bars[symbol].extend({k: b[k] for k in ("t", "c", "h", "l", "v")} for b in rs)
            token = extra.get("next_page_token")
            if not token:
                break
            if token in extra_seen:
                raise Blocked("EXTRA_HISTORY_PAGINATION_LOOP")
            extra_seen.add(token)
            extra_params["page_token"] = token
        else:
            raise Blocked("EXTRA_HISTORY_PAGINATION_INCOMPLETE")
        bad_history = {}
        for symbol in scope + ["SPY"]:
            history = bars.get(symbol, [])[-300:]
            if (
                len(history) < 200
                or history[-1]["t"][:10] != previous
                or any(not (0 < r["l"] <= r["c"] <= r["h"]) or r["v"] <= 0 for r in history)
            ):
                bad_history[symbol] = "BARS_UNUSABLE"
        if "SPY" in bad_history:
            raise Blocked("REGIME_HISTORY_UNUSABLE")
        rows, review = admissible_registry(
            ranks, [r["symbol"] for r in hunter["candidates"]], metadata, earnings, submissions
        )
        kept = []
        for row in rows:
            if row["symbol"] in bad_history:
                review["excluded"].append({"symbol": row["symbol"], "reasons": ["BARS_UNUSABLE"]})
            else:
                kept.append(row)
        rows = kept
        review["admitted_equities"] = len(rows)
        review["excluded_by_primary_reason"] = dict(
            Counter(r["reasons"][0] for r in review["excluded"])
        )
        review["excluded_by_any_reason"] = dict(
            Counter(reason for r in review["excluded"] for reason in r["reasons"])
        )
        review["sec_sic_checked"] = len(submissions)
        review["sec_sic_missing_in_responses"] = sum(not s.get("sic") for s in submissions.values())
        review["sec_sic_not_requested_after_earlier_exclusion"] = len(scope) - len(submissions)
        for row in rows:
            row["provenance"] = {
                **row["provenance"],
                "cluster": {
                    "method": "SEC submissions SIC -> fixed revision-three range table",
                    "sic": row["sic"],
                    "receipt": sec_receipts[row["symbol"]],
                    "mapping_sha256": sha(
                        REPO / "scripts/trading_lab/agent_v2_quality/registry.py"
                    ),
                },
            }
        spy = metadata["SPY"]
        spy["market_cap"] = None
        reg = {
            **oldreg,
            "received_at": iso(clock()),
            "instruments": sorted(rows + [spy], key=lambda r: r["symbol"]),
            "ready": True,
            "selection": review,
            "spec_hash": SPEC_HASH,
            "preregistration_hash": PREREG_HASH,
            "sources": {
                **oldreg["sources"],
                "momentum_ranking_sha256": sha(ROOT / "momentum-ranking.json"),
                "sic_submissions_receipts": sec_receipts,
            },
        }
        admitted = {r["symbol"] for r in reg["instruments"]}
        cal = load(Path(evidence_dir) / "calendar.json")
        cal["events"] = [
            e for e in cal["events"] if e["kind"] != "EARNINGS" or e["symbol"] in admitted
        ]
        cal["covered_symbols"] = sorted(admitted)
        cal["missing_earnings_dates"] = []
        cal["ready"] = True
        cal["selection_source_sha256"] = sha(Path(evidence_dir) / "calendar.json")
        cal["coverage_notes"].append(
            "Scope projection only; source receipt time and source coverage are unchanged."
        )
        projected_hunter = {k: hunter[k] for k in ["asof", "generated", "candidates"]}
        projected_hunter["candidates"] = [
            r for r in hunter["candidates"] if r["symbol"] in admitted
        ]
        projected_hunter["provenance"] = {
            "source_file_sha256": sha(original["hunter"]),
            "projection": "admitted registry symbols only; native fields/times unchanged",
        }
        reg["sources"]["hunter"] = {
            "sha256": sha(original["hunter"]),
            "generated": hunter["generated"],
            "projected_file": "hunter.json",
        }
        radar_source = Path(radar_source)
        # Release ranking histories before parsing the larger native radar output.
        del bars, eligible, data, extra
        upstream = read_local(radar_source, maximum=64_000_000)
        radar = radar_projection(upstream, sorted(admitted), clock())
        radar.update(
            {
                "upstream": {"sha256": sha(radar_source), "canonical_sha256": digest(upstream)},
                "projection": "causal related stock stories only; no timestamp changed",
            }
        )
        save("radar-causal-stocks.json", radar)
        registry(reg, clock())
        calendar(cal, sorted(admitted), clock())
        radar_headlines(radar, sorted(admitted), clock())
        save("registry.json", reg)
        save("calendar.json", cal)
        save("hunter.json", projected_hunter)
        newconfig = {
            **original,
            "registry": str(ROOT / "registry.json"),
            "calendar": str(ROOT / "calendar.json"),
            "hunter": str(ROOT / "hunter.json"),
            "radar": str(ROOT / "radar-causal-stocks.json"),
        }
        save("operator-config.json", newconfig)
        review.update(
            {
                "registry_total": len(reg["instruments"]),
                "sec_requests": len({r["url"] for r in sec_receipts.values()}),
                "calendar_events": len(cal["events"]),
                "hunter_admitted": len(projected_hunter["candidates"]),
                "radar_counts": radar["excluded_counts"],
                "registry_file_sha256": sha(ROOT / "registry.json"),
                "calendar_file_sha256": sha(ROOT / "calendar.json"),
                "ranking_unavailable": missing,
                "implementation_hash": implementation_hash(),
                "earnings_evidence_file_sha256": sha(Path(evidence_dir) / "earnings-evidence.json"),
            }
        )
        save("input-review.json", review)
        return newconfig, review
    except Exception:
        # The caller owns the runtime; durable reservations survive every failure.
        raise
