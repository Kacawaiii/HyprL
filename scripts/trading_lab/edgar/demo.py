"""Offline demo of the EDGAR slice: a synthetic watched CIK goes through a new 8-K, a corrected listing,
an amendment, a disappearance inside the window and a reappearance; reads at chosen instants, then the
store is reopened read-only and replayed at the same (T, H). No request leaves the machine.

    python -m scripts.trading_lab.edgar.demo
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile

from scripts.trading_lab.edgar import snapshot
from scripts.trading_lab.edgar import synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore

OLD = syn.filing("0000320193-26-000040", filed="2026-05-01")
K8 = "0000320193-26-000071"
STEPS = (
    ("a new 8-K", [syn.filing(K8), OLD]),
    ("its items corrected", [syn.filing(K8, items="2.02,7.01,9.01"), OLD]),
    ("an 8-K/A filed", [syn.filing(K8, items="2.02,7.01,9.01"), syn.filing("0000320193-26-000073", form="8-K/A", filed="2026-06-17"), OLD]),
    ("the 8-K missing from the listing", [syn.filing("0000320193-26-000073", form="8-K/A", filed="2026-06-17"), OLD]),
    ("the 8-K listed again", [syn.filing(K8, items="2.02,7.01,9.01"), syn.filing("0000320193-26-000073", form="8-K/A", filed="2026-06-17"), OLD]),
)


def _show(title: str, snap: dict) -> None:
    print(f"\n== {title}: T={snap['T']} H={snap['H']} {snap['read_state']} identity={snap['identity'][:16]}")
    for f in snap.get("filings", []):
        print(f"   {f['accession_number']} {f['fields']['form']:<5} {f['state']:<19} items={f['fields']['items']:<14} "
              f"revisions={f['revisions_seen']} available_at={f['first_available_at']} "
              f"acceptance(provenance)={f['provenance']['acceptance_datetime_text']}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="edgar-demo-")) / "store"
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    store = EdgarStore(root, wall_clock=clock.wall)
    collector = EdgarCollector(store, fetcher, clock)
    collector.submit_watchlist([syn.CIK_A])
    reads = []
    for title, filings in STEPS:
        fetcher.routes[syn.CIK_A.zfill(10)] = syn.Reply(syn.listing(syn.CIK_A, filings))
        for _ in range(2):  # the step's listing, then the poll that attests it
            collector.poll(syn.CIK_A)
            clock.sleep(600)
        attesting = store.rows("RESPONSE")[-1].body["observed_at"]
        reads.append((title, snapshot.parse_iso(attesting) + timedelta(seconds=93)))
    collector.poll(syn.CIK_A)  # the last read needs a later attested transaction to resolve
    clock.sleep(10)
    collector.poll(syn.CIK_A)
    snaps = []
    for title, T in reads:
        snap = snapshot.filings_as_of(store, T)
        snaps.append(snap)
        _show(title, snap)
    collector.close()
    store.close()
    reopened = EdgarStore(root, wall_clock=None, read_only=True)
    identical = all(snapshot.replay(reopened, snapshot.parse_iso(s["T"]), s["H"]) == s for s in snaps)
    print(f"\nrequests: {len(fetcher.requests)} (synthetic), records: {len(reopened.rows('RESPONSE'))}, "
          f"revisions: {len(reopened.rows('FILING_REVISION'))}, absences: {len(reopened.rows('FILING_ABSENCE'))}")
    print(f"reopened read-only and replayed at every (T, H): identical={identical}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
