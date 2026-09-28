"""Offline end-to-end run of the FOMC V1 slice against the local synthetic provider.

    python -m scripts.trading_lab.fomc.demo

feed -> durable raw -> classification -> primary acquisition -> revision + observation link -> cycle
-> events_as_of(T, H) -> reopen the store -> offline replay at the same (T, H). No network is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from datetime import timedelta

from scripts.trading_lab.fomc import snapshot, spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.store import FomcStore


def _drive(collector, clock, seconds, idle=30):
    end = clock.true + timedelta(seconds=seconds)
    while clock.true < end:
        if collector.step() is None:
            clock.sleep(idle)


def _show(label, snap):
    print(f"\n== {label}: T={snap['T']} H={snap['H']} read_state={snap['read_state']}")
    print(f"   discovery={snap.get('discovery')}  identity={snap['identity'][:16]}")
    for item in snap.get("items", []):
        extra = f" hash={item['content_hash'][:12]} live={item['live_available']} links={[l['mode'] for l in item['links']]}" \
            if item["state"] == "CURRENT_REVISION" else ""
        print(f"   item {item['sid'][:12]} step={item['step']} {item['state']}{extra}")


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    spec.verify_spec_binding()
    root = Path(tempfile.mkdtemp(prefix="fomc-slice-")) / "store"
    clock = syn.SimClock(syn.START)
    provider = syn.LocalProvider(clock)
    try:
        store = FomcStore(root, wall_clock=clock.wall)
        collector = Collector(store, provider.connector(), clock)
        p1, p2 = syn.statement_path("20260617"), syn.statement_path("20260729")
        provider.routes[syn.FEED_PATH] = syn.feed_response([{"title": spec.FEED_TITLE_EXACT, "link": syn.url(p1), "guid": "g1"}])
        provider.routes[p1] = syn.page_response(body="The Committee decided to maintain the target range.")
        provider.routes[p2] = syn.page_response(date_text="July 29, 2026", body="The Committee decided to lower the target range.")
        print(f"spec {spec.SPEC_HASH[:12]} rev {spec.SPEC_REVISION}; store {root}")

        _drive(collector, clock, 240)
        _show("1. first statement", first := snapshot.events_as_of(store, clock.true))
        provider.routes[p1] = syn.page_response(body="The Committee decided to maintain the target range, as corrected.")
        _drive(collector, clock, 900, idle=60)
        _show("1. after the O300 recheck saw corrected bytes", snapshot.events_as_of(store, clock.true))

        collector.submit_manifest(json.dumps({"version": 1, "urls": [syn.url(p2)]}).encode(), "operator: July statement")
        _drive(collector, clock, 240)
        _show("2. July statement via backfill only", snapshot.events_as_of(store, clock.true))
        provider.routes[syn.FEED_PATH] = syn.feed_response([
            {"title": spec.FEED_TITLE_EXACT, "link": syn.url(p1), "guid": "g1"},
            {"title": spec.FEED_TITLE_EXACT, "link": syn.url(p2), "guid": "g2"}])
        _drive(collector, clock, 300)
        final = snapshot.events_as_of(store, clock.true)
        _show("2. same July bytes now observed LIVE", final)

        cycles = [c.body["result"] for c in store.rows("CYCLE_CONCLUSION")]
        print(f"\ncycles: {len(cycles)} ({cycles.count('EVENTS_OBSERVED_ZERO')} zero), revisions: {len(store.rows('REVISION'))}, "
              f"provider requests: {len(provider.requests)}")
        T, H = clock.true, final["H"]
        collector.close()
        store.close()
        reopened = FomcStore(root, wall_clock=clock.wall)
        replayed = snapshot.replay(reopened, T, H)
        same = replayed == final and snapshot.events_as_of(reopened, T, H) == final
        print(f"8. reopen + offline replay at the same (T, H): identical={same}; first snapshot identity {first['identity'][:16]}")
        return 0 if same else 1
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
