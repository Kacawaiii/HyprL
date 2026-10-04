"""Offline scale measurements; temporary synthetic stores only, never official HTTP.

Run: python -m scripts.trading_lab.sources.scale --polls 24 --fomc-hours 72
Reports time, decoded/store-view rows and compact JSON bytes, including cold
identity construction and warm pages. The FOMC run uses soak's synthetic service.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import sys

from scripts.trading_lab.app_api.sources import EdgarViews, FomcViews
from scripts.trading_lab.edgar import snapshot as edgar_snapshot, synthetic as edgar_syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.fomc import canon, snapshot as fomc_snapshot, soak, synthetic as fomc_syn
from scripts.trading_lab.fomc.store import FomcStore


def build_edgar(root: Path, *, ciks=10, listing_rows=1000, polls=24):
    clock = edgar_syn.SimClock()
    fetcher = edgar_syn.FakeFetcher(clock)
    store = EdgarStore(root, wall_clock=clock.wall)
    collector = EdgarCollector(store, fetcher, clock)
    watch = [str(100000 + i).zfill(10) for i in range(ciks)]
    try:
        collector.submit_watchlist(watch)
        for cik in watch:
            filings = [edgar_syn.filing(f"{cik}-26-{i:06d}") for i in range(listing_rows)]
            fetcher.routes[cik] = edgar_syn.Reply(edgar_syn.listing(cik, filings))
        for _ in range(polls):
            collector.poll_all()
            clock.sleep(600)
        # Two later attestations: the first makes every original poll available;
        # the second resolves the read boundary beyond T (including with one CIK).
        attesting = collector.poll(watch[0])
        T = edgar_snapshot.parse_iso(store.row_at("RESPONSE", attesting["record"]).body["observed_at"]) + timedelta(seconds=93)
        clock.sleep(200)
        collector.poll(watch[0])
        return T, store.horizon()
    finally:
        collector.close()
        store.close()


def measure(store, operation):
    store.reads.update(queries=0, rows=0, view_rows=0)
    start = time.perf_counter()
    payload = operation()
    return payload, {"milliseconds": round((time.perf_counter() - start) * 1000, 3),
                     **store.reads, "payload_bytes": len(json.dumps(payload, separators=(",", ":")).encode())}


def measure_source(root, views_type, store_type, derive, T, H):
    reader = store_type(root, wall_clock=None, read_only=True)
    try:
        full, direct = measure(reader, lambda: derive(reader, T, H))
    finally:
        reader.close()
    del reader
    views = views_type(root)
    metrics = {"direct_snapshot": direct, "identity": full["identity"], "horizon": H}
    try:
        # Measure the private DB copy separately; the cold snapshot has an empty mirror.
        started = time.perf_counter()
        store = views._open()
        metrics["api_open_milliseconds"] = round((time.perf_counter() - started) * 1000, 3)
        page, metrics["api_snapshot_cold"] = measure(store, lambda: views.snapshot(as_of=T.isoformat(), horizon=H))
        assert page["snapshot"]["identity"] == full["identity"]
        _, metrics["api_snapshot_warm"] = measure(store, lambda: views.snapshot(as_of=T.isoformat(), horizon=H))
        timeline, metrics["api_timeline_first"] = measure(store, lambda: views.timeline(as_of=T.isoformat(), horizon=H))
        if timeline["pagination"]["next_cursor"]:
            _, metrics["api_timeline_next"] = measure(store, lambda: views.timeline(
                as_of=T.isoformat(), horizon=H, cursor=timeline["pagination"]["next_cursor"]))
        if full.get("filings"):
            name = full["filings"][0]["accession_number"]
            detail = lambda **kw: views.filing(name, as_of=T.isoformat(), horizon=H, **kw)
            metrics["filings"] = len(full["filings"])
        else:
            name = full["items"][0]["sid"]
            detail = lambda **kw: views.item(name, as_of=T.isoformat(), horizon=H, **kw)
            metrics["items"] = len(full["items"])
        first, metrics["api_detail_first"] = measure(store, lambda: detail(limit=5))
        metrics["detail_totals"] = first["pagination"]["totals"]
        _, metrics["api_detail_warm"] = measure(store, lambda: detail(limit=5, cursor=first["pagination"]["next_cursor"]))
        if page["pagination"]["next_cursor"]:
            _, metrics["api_snapshot_next"] = measure(store, lambda: views.snapshot(
                as_of=T.isoformat(), horizon=H, cursor=page["pagination"]["next_cursor"]))
        _, metrics["api_status"] = measure(store, views.status)
        replay, metrics["api_replay"] = measure(store, lambda: views.replay(as_of=T.isoformat(), horizon=H))
        assert replay["identical"]
        return metrics
    finally:
        views.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polls", type=int, default=24)
    parser.add_argument("--fomc-hours", type=float, default=72)
    parser.add_argument("--source", choices=("both", "edgar", "fomc"), default="both")
    args = parser.parse_args(argv)
    if args.polls < 2 or args.fomc_hours < 26:
        parser.error("use at least 2 EDGAR polls and 26 FOMC hours (the full soak fault schedule)")
    with tempfile.TemporaryDirectory(prefix="source-scale-", dir=Path.cwd()) as temp:
        root = Path(temp)
        if args.source in {"both", "edgar"}:
            T, H = build_edgar(root / "edgar", polls=args.polls)
            edgar = measure_source(root / "edgar", EdgarViews, EdgarStore, edgar_snapshot.filings_as_of, T, H)
            print(json.dumps({"edgar": edgar}, sort_keys=True), flush=True)
        if args.source == "edgar":
            return 0
        old_script = canon.CHALLENGE_SCRIPT_SHA256
        try:
            summary = soak.run(root / "fomc", hours=args.fomc_hours,
                               progress=lambda message: print(message, file=sys.stderr, flush=True))
            canon.CHALLENGE_SCRIPT_SHA256 = fomc_syn.CF_SCRIPT_SHA256
            status = FomcViews(root / "fomc").status()
            T = fomc_snapshot.clockmod.parse_iso(status["suggested_as_of"])
            fomc = measure_source(root / "fomc", FomcViews, FomcStore, fomc_snapshot.events_as_of,
                                  T, status["horizon"])
            print(json.dumps({"fomc": fomc, "soak_hours": summary["hours"],
                              "soak_responses": summary["responses"]}, sort_keys=True), flush=True)
        finally:
            canon.CHALLENGE_SCRIPT_SHA256 = old_script
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
