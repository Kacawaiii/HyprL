"""Offline cross-source timeline: independent causal reads, attested ordering and read-only stores."""

from copy import deepcopy
from datetime import timedelta
import hashlib
import shutil
import sqlite3

import pytest

from scripts.trading_lab.app_api import server
from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.app_api.sources import TimelineViews
from scripts.trading_lab.edgar import snapshot as edgar_snapshot, synthetic as edgar_syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.fomc import snapshot as fomc_snapshot, state, synthetic as fomc_syn
from scripts.trading_lab.fomc.clock import iso, parse_iso
from scripts.trading_lab.fomc.store import FomcStore
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.crypto.fomc_support import Env, P1, SID1, statement_item

ACC = "0000320193-26-000071"
OLD_ACC = "0000320193-26-000040"


def fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns,
                   hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
                  for p in root.rglob("*"))


def fold(root, filename):
    with sqlite3.connect(root / filename) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    env = Env(tmp_path_factory.mktemp("timeline-fomc"))
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = fomc_syn.page_response(body="Content A")
        env.drive(240)
        env.provider.routes[P1] = fomc_syn.page_response(body="Content B")
        env.drive(900, idle=60)
        env.provider.routes[P1] = fomc_syn.page_response(body="Content A")
        env.drive(3600, idle=300)
        env.drive(130)
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()
    fold(env.root, "fomc.sqlite3")

    root = tmp_path_factory.mktemp("timeline-edgar") / "store"
    clock = edgar_syn.SimClock(fomc_syn.START + timedelta(seconds=600))
    fetcher = edgar_syn.FakeFetcher(clock)
    store = EdgarStore(root, wall_clock=clock.wall)
    collector = EdgarCollector(store, fetcher, clock)
    reads = {}
    try:
        collector.submit_watchlist([edgar_syn.CIK_A])
        old = edgar_syn.filing(OLD_ACC, filed="2026-05-01")
        # Acceptance text predates FOMC's availability, while this filing is available later.
        for name, filings in (("new", [edgar_syn.filing(ACC), old]),
                              ("corrected", [edgar_syn.filing(ACC, items="2.02,7.01"), old]),
                              ("gone", [old])):
            fetcher.routes[edgar_syn.CIK_A.zfill(10)] = edgar_syn.Reply(edgar_syn.listing(edgar_syn.CIK_A, filings))
            for _ in range(2):
                collector.poll(edgar_syn.CIK_A)
                clock.sleep(600)
            reads[name] = iso(parse_iso(store.rows("RESPONSE")[-1].body["observed_at"]) + timedelta(seconds=93))
        collector.poll(edgar_syn.CIK_A)
        clock.sleep(10)
        collector.poll(edgar_syn.CIK_A)
    finally:
        collector.close()
        store.close()
    fold(root, "edgar.sqlite3")
    return env.root, root, reads


@pytest.fixture
def service(stores, tmp_path):
    fomc, edgar, _reads = stores
    return AppService(tmp_path, fomc_store=fomc, edgar_store=edgar)


def test_availability_order_provenance_and_current_revisions(service, stores):
    fomc_root, edgar_root, reads = stores
    before = fingerprint(fomc_root), fingerprint(edgar_root)
    first = service.timeline.snapshot(as_of=reads["new"])
    assert all(s["read_state"].endswith("_RESOLVED") for s in first["sources"].values())
    assert [(r["source"], r["id"]) for r in first["rows"]] == [("fomc", SID1), ("edgar", OLD_ACC), ("edgar", ACC)]
    fomc = first["rows"][0]
    edgar = next(r for r in first["rows"] if r["id"] == ACC)
    assert parse_iso(edgar["provenance"]["acceptance_datetime_text"]) < parse_iso(fomc["available_at"]) < parse_iso(edgar["available_at"])
    assert fomc["provenance"] == {"declared_release_at": "2026-06-17T18:00:00+00:00",
                                   "declared_release_text": "For release at 2:00 p.m. EDT"}
    fstore = FomcStore(fomc_root, wall_clock=None, read_only=True)
    estore = EdgarStore(edgar_root, wall_clock=None, read_only=True)
    try:
        direct_f = fomc_snapshot.events_as_of(fstore, parse_iso(reads["new"]))
        direct_e = edgar_snapshot.filings_as_of(estore, parse_iso(reads["new"]))
        assert first["sources"]["fomc"]["identity"] == direct_f["identity"]
        assert first["sources"]["edgar"]["identity"] == direct_e["identity"]
        item = next(i for i in direct_f["items"] if i["sid"] == SID1)
        created = fstore.rows("REVISION", key=item["revision"])[0]
        assert created.seq != fstore.rows("REVISION")[0].seq  # selected B, not the item's first revision A
        assert fomc["available_at"] == iso(state.avail_of(state.availability(fstore, direct_f["H"]), created.seq))
        assert fomc["revision"] == item["revision"] and fomc["content_identity"] == item["content_hash"]
        filing = next(f for f in direct_e["filings"] if f["accession_number"] == ACC)
        assert edgar["available_at"] == filing["first_available_at"] and edgar["revision"] == filing["revision"]
    finally:
        fstore.close()
        estore.close()
    gone = service.timeline.snapshot(as_of=reads["gone"])
    absent = next(r for r in gone["rows"] if r["id"] == ACC)
    assert absent["state"] == "ABSENT_FROM_LISTING" and absent["form"] == "8-K"
    assert absent["revision"] != edgar["revision"] and absent["available_at"] == edgar["available_at"]
    assert (fingerprint(fomc_root), fingerprint(edgar_root)) == before


def test_aba_returns_the_revisions_first_availability(service, stores):
    root, _edgar, _reads = stores
    T = service.fomc.status()["suggested_as_of"]
    row = next(r for r in service.timeline.snapshot(as_of=T)["rows"] if r["source"] == "fomc")
    store = FomcStore(root, wall_clock=None, read_only=True)
    try:
        revisions = store.rows("REVISION")
        assert len(revisions) == 2 and row["revision"] == revisions[0].key
        table = state.availability(store, store.horizon())
        assert row["available_at"] == iso(state.revision_available_at(store.view(), revisions[0].key, table))
        assert parse_iso(row["available_at"]) < state.avail_of(table, store.rows("LINK")[-1].seq)
    finally:
        store.close()


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_each_unresolved_source_keeps_the_other_read(service, stores, source):
    read = service.timeline.snapshot(as_of=stores[2]["new"], **{f"{source}_horizon": "0"})
    other = "edgar" if source == "fomc" else "fomc"
    assert read["sources"][source]["read_state"] == f"{source.upper()}_CAUSAL_VISIBILITY_UNRESOLVED"
    assert read["sources"][source]["identity"] and read["sources"][source]["H"] == 0
    assert read["sources"][other]["read_state"] == f"{other.upper()}_RESOLVED"
    assert read["rows"] and all(r["source"] == other for r in read["rows"])


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_unconfigured_and_refused_sources_are_explicit(stores, tmp_path, source):
    fomc, edgar, reads = stores
    configured = {"fomc_store": fomc, "edgar_store": edgar}
    configured.pop(f"{source}_store")
    missing = AppService(tmp_path, **configured).timeline.snapshot(as_of=reads["new"])
    assert missing["sources"][source]["read_state"] == "NOT_CONFIGURED"
    assert missing["sources"][source]["identity"] is None and missing["rows"]
    bad = tmp_path / "incompatible"
    shutil.copytree(fomc if source == "fomc" else edgar, bad)
    with sqlite3.connect(bad / f"{source}.sqlite3") as conn:
        conn.execute("UPDATE meta SET value = 'incompatible' WHERE name = 'schema_version'")
    configured[f"{source}_store"] = bad
    before = fingerprint(bad)
    refused = AppService(tmp_path, **configured).timeline.snapshot(as_of=reads["new"])
    assert refused["sources"][source]["read_state"] == "REJECTED"
    assert refused["sources"][source]["reason"] and refused["rows"] == missing["rows"]
    assert refused["identity"] != missing["identity"] and fingerprint(bad) == before


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_corrupt_raw_refuses_only_its_source(stores, tmp_path, source):
    fomc, edgar, reads = stores
    bad = tmp_path / "corrupt"
    shutil.copytree(fomc if source == "fomc" else edgar, bad)
    # Every synthetic raw is a read dependency at this instant; never use or mutate a real store.
    for raw in (bad / "raw").rglob("*"):
        if raw.is_file():
            raw.write_bytes(b"corrupt synthetic raw")
    configured = {"fomc_store": fomc, "edgar_store": edgar, f"{source}_store": bad}
    before = fingerprint(bad)
    read = AppService(tmp_path, **configured).timeline.snapshot(as_of=reads["new"])
    assert read["sources"][source]["read_state"] == "REFUSED" and read["sources"][source]["identity"] is None
    assert read["rows"] and all(r["source"] != source for r in read["rows"])
    assert fingerprint(bad) == before


def test_identity_binds_source_snapshots_read_states_and_rows(service, stores, monkeypatch):
    T = stores[2]["new"]
    first = service.timeline.snapshot(as_of=T)
    assert service.timeline.snapshot(as_of=T) == first
    explicit = service.timeline.snapshot(as_of=T, fomc_horizon=first["sources"]["fomc"]["H"],
                                         edgar_horizon=first["sources"]["edgar"]["H"])
    assert explicit == first
    binding = {"T": first["T"], "sources": {name: {k: s[k] for k in ("identity", "read_state")}
                                          for name, s in first["sources"].items()}, "rows": first["rows"]}
    assert first["identity"] == sha256_canonical(binding)
    # H changes the source snapshot identity even when its rows at this T stay identical.
    changed_source = service.timeline.snapshot(as_of=T, edgar_horizon=first["sources"]["edgar"]["H"] - 1)
    assert changed_source["rows"] == first["rows"]
    assert changed_source["sources"]["edgar"]["identity"] != first["sources"]["edgar"]["identity"]
    assert changed_source["identity"] != first["identity"]
    project = TimelineViews._fomc_rows

    def changed_row(view, snap):
        rows = deepcopy(project(view, snap))
        rows[0]["title"] = "Changed synthetic display title"
        return rows

    monkeypatch.setattr(TimelineViews, "_fomc_rows", staticmethod(changed_row))
    changed = service.timeline.snapshot(as_of=T)
    assert changed["sources"] == first["sources"] and changed["identity"] != first["identity"]


def test_ties_order_by_source_then_id_without_provenance(service, stores, monkeypatch):
    project_f, project_e = TimelineViews._fomc_rows, TimelineViews._edgar_rows
    instant = "2026-06-17T18:00:00+00:00"

    def tied(project):
        def rows(view, snap):
            result = project(view, snap)
            for row in result:
                row["available_at"] = instant
            return list(reversed(result))
        return staticmethod(rows)

    monkeypatch.setattr(TimelineViews, "_fomc_rows", tied(project_f))
    monkeypatch.setattr(TimelineViews, "_edgar_rows", tied(project_e))
    read = service.timeline.snapshot(as_of=stores[2]["new"])
    assert [(r["source"], r["id"]) for r in read["rows"]] == [("edgar", OLD_ACC), ("edgar", ACC), ("fomc", SID1)]


def test_contract_bounds_and_router_dispatch(service, stores, monkeypatch):
    handler = type("Handler", (), {"service": service})()
    T = stores[2]["new"]
    query = {"as_of": [T], "fomc_horizon": ["0"]}
    routed = server.AppApiHandler._dispatch(handler, "/api/v1/events/timeline", query)
    assert routed == service.timeline.snapshot(as_of=T, fomc_horizon="0")
    for params in ({}, {"as_of": "2026-06-17T18:00:00"}, {"as_of": "yesterday"},
                   {"as_of": T, "fomc_horizon": "1x"}, {"as_of": T, "edgar_horizon": "-1"},
                   {"as_of": T, "fomc_horizon": service.fomc.status()["horizon"] + 1},
                   {"as_of": T, "edgar_horizon": service.edgar.status()["horizon"] + 1}):
        with pytest.raises(AppApiError):
            service.timeline.snapshot(**params)
    monkeypatch.setattr("scripts.trading_lab.app_api.sources.MAX_SOURCE_ITEMS", 2)
    with pytest.raises(AppApiError, match="3 timeline rows exceed"):
        service.timeline.snapshot(as_of=T)


def test_future_and_fully_unconfigured_reads_do_not_claim_confirmed_emptiness(service, stores, tmp_path):
    later = iso(parse_iso(stores[2]["gone"]) + timedelta(days=1))
    future = service.timeline.snapshot(as_of=later)
    assert future["rows"] == []
    assert all("UNRESOLVED" in source["read_state"] for source in future["sources"].values())
    absent = AppService(tmp_path).timeline
    missing = absent.snapshot(as_of=later)
    assert missing["rows"] == [] and all(s["read_state"] == "NOT_CONFIGURED" for s in missing["sources"].values())
    assert future["identity"] != missing["identity"]
    for horizon in ("bad", "-1"):
        with pytest.raises(AppApiError):
            absent.snapshot(as_of=later, edgar_horizon=horizon)
