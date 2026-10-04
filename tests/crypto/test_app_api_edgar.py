"""The EDGAR journey of the read-only application API: status, a causal read at (as_of, horizon) with its
identity, a filing's revisions, observations and absences with provenance, the verified replay, all over a
store opened read-only (never written)."""

from __future__ import annotations

from datetime import timedelta
import shutil
import sqlite3

import pytest

from scripts.trading_lab.app_api import server as server_module
from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError, NotFoundError
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.edgar import snapshot
from scripts.trading_lab.edgar import synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore

K8, OLD_ACC = "0000320193-26-000071", "0000320193-26-000040"
OLD = syn.filing(OLD_ACC, filed="2026-05-01")


def _fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


@pytest.fixture(scope="module")
def closed_store(tmp_path_factory):
    """A watched CIK: a new 8-K, its correction, then its disappearance inside the window."""
    root = tmp_path_factory.mktemp("edgar") / "store"
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    store = EdgarStore(root, wall_clock=clock.wall)
    collector = EdgarCollector(store, fetcher, clock)
    collector.submit_watchlist([syn.CIK_A])
    reads = {}
    for name, filings in (("new", [syn.filing(K8), OLD]), ("corrected", [syn.filing(K8, items="2.02,7.01"), OLD]),
                          ("gone", [OLD])):
        fetcher.routes[syn.CIK_A.zfill(10)] = syn.Reply(syn.listing(syn.CIK_A, filings))
        for _ in range(2):
            collector.poll(syn.CIK_A)
            clock.sleep(600)
        reads[name] = snapshot.parse_iso(store.rows("RESPONSE")[-1].body["observed_at"]) + timedelta(seconds=93)
    collector.poll(syn.CIK_A)
    clock.sleep(10)
    collector.poll(syn.CIK_A)
    collector.close()
    store.close()
    with sqlite3.connect(root / "edgar.sqlite3") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    (root / "owner.lock").unlink()
    return root, {k: snapshot.iso(v) for k, v in reads.items()}


@pytest.fixture
def views(closed_store, tmp_path):
    return AppService(tmp_path, edgar_store=closed_store[0]).edgar


def test_status_reads_filing_detail_and_replay(views, closed_store):
    root, reads = closed_store
    before = _fingerprint(root)
    status = views.status()
    assert status["status"] == "AVAILABLE" and status["watchlist"] == [syn.CIK_A.zfill(10)] and status["suggested_as_of"]
    assert status["counts"]["revisions"] == 3 and status["counts"]["absences"] == 1
    new = views.snapshot(as_of=reads["new"])
    first = next(f for f in new["filings"] if f["accession_number"] == K8)
    assert new["snapshot"]["read_state"] == "EDGAR_RESOLVED" and first["state"] == "PRESENT" and first["items"] == "2.02,9.01"
    assert first["acceptance_datetime_text"] == "2026-06-16T16:31:05.000Z" and first["first_available_at"] > reads["new"][:10]
    gone = views.snapshot(as_of=reads["gone"])
    last = next(f for f in gone["filings"] if f["accession_number"] == K8)
    assert last["state"] == "ABSENT_FROM_LISTING" and last["items"] == "2.02,7.01" and last["revisions_seen"] == 2
    assert gone["health"][syn.CIK_A.zfill(10)]["result_state"] is None
    detail = views.filing(K8, as_of=reads["gone"])
    assert len(detail["revisions"]) == 2 and len(detail["absences"]) == 1 and len(detail["observations"]) >= 4
    assert detail["filing"]["provenance"]["filing_index_url"].endswith(f"{K8}-index.htm")
    replay = views.replay(as_of=reads["gone"])
    assert replay["identical"] and replay["replay_identity"] == gone["snapshot"]["identity"]
    direct = snapshot.filings_as_of(EdgarStore(root, wall_clock=None, read_only=True), snapshot.parse_iso(reads["gone"]))
    assert direct["identity"] == gone["snapshot"]["identity"]
    assert _fingerprint(root) == before


def test_requests_outside_the_contract_and_refused_stores(views, closed_store, tmp_path):
    root, reads = closed_store
    for bad in ({}, {"as_of": "2026-06-17T14:00:00"}):
        with pytest.raises(AppApiError):
            views.snapshot(**bad)
    with pytest.raises(AppApiError):
        views.filing("../etc/passwd", as_of=reads["new"])
    with pytest.raises(NotFoundError):
        views.filing("0000000000-00-000000", as_of=reads["new"])
    assert AppService(tmp_path).edgar.status()["status"] == "NOT_CONFIGURED"
    old = tmp_path / "old"
    shutil.copytree(root, old)
    with sqlite3.connect(old / "edgar.sqlite3") as conn:
        conn.execute("UPDATE meta SET value = 'edgar-store-v0' WHERE name = 'schema_version'")
    refused = AppService(tmp_path, edgar_store=old).edgar
    assert refused.status()["status"] == "REJECTED"
    with pytest.raises(ConflictError):
        refused.snapshot(as_of=reads["new"])


def test_the_router_serves_the_edgar_journey(closed_store, tmp_path):
    root, reads = closed_store
    handler = type("H", (), {"service": AppService(tmp_path, edgar_store=root)})()

    def get(path, **query):
        return server_module.AppApiHandler._dispatch(handler, path, {k: [v] for k, v in query.items()})
    assert get("/api/v1/sources/edgar")["status"] == "AVAILABLE"
    assert get("/api/v1/sources/edgar/snapshot", as_of=reads["new"])["snapshot"]["read_state"] == "EDGAR_RESOLVED"
    assert get(f"/api/v1/sources/edgar/filings/{K8}", as_of=reads["new"])["filing"]["accession_number"] == K8
    assert get("/api/v1/sources/edgar/replay", as_of=reads["new"])["identical"] is True


def test_filing_detail_pages_a_history_above_the_source_bound(views, closed_store, monkeypatch):
    from scripts.trading_lab.app_api import sources
    monkeypatch.setattr(sources, "MAX_SOURCE_ITEMS", 1)
    page = views.filing(K8, as_of=closed_store[1]["gone"])
    assert len(page["observations"]) == 1
    assert page["pagination"]["next_cursor"]
    assert page["pagination"]["totals"]["observations"] > 1
