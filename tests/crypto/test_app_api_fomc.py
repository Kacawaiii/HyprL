"""The FOMC journey of the read-only application API: status, a point-in-time snapshot with its
identity, an item's revisions and provenance, source health and the verified offline replay, all over a
store opened read-only (it is never written, also through the HTTP server)."""

from __future__ import annotations

from datetime import timedelta
import json
import shutil
import sqlite3

import pytest

from scripts.trading_lab.app_api import server as server_module
from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError, NotFoundError
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.fomc import canon, snapshot
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.clock import iso

from tests.crypto.fomc_support import P1, SID1, Env, statement_item


def _fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


@pytest.fixture(scope="module")
def closed_store(tmp_path_factory):
    """A small FOMC store with per-response Cloudflare bytes, closed and folded like a closure copy."""
    patch = pytest.MonkeyPatch()
    patch.setattr(canon, "CHALLENGE_SCRIPT_SHA256", syn.CF_SCRIPT_SHA256)
    env = Env(tmp_path_factory.mktemp("fomc"))
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=syn.url(P1))
    env.drive(4000, idle=60)
    env.drive(130)
    end = env.clock.true
    env.collector.close()
    env.store.close()
    env.provider.close()
    with sqlite3.connect(env.root / "fomc.sqlite3") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    yield env.root, end
    patch.undo()


@pytest.fixture
def views(closed_store, tmp_path):
    root, _end = closed_store
    return AppService(tmp_path, fomc_store=root).fomc


def test_status_snapshot_item_and_replay_over_a_read_only_store(views, closed_store):
    root, end = closed_store
    before = _fingerprint(root)
    status = views.status()
    assert status["status"] == "AVAILABLE" and status["read_only"] and status["spec_revision"] == 25
    assert status["horizon"] > 0 and status["counts"]["responses"] > 0 and status["suggested_as_of"]
    as_of = status["suggested_as_of"]
    snap = views.snapshot(as_of=as_of)
    header = snap["snapshot"]
    assert header["read_state"] == "FOMC_RESOLVED" and header["H"] == status["horizon"]
    item = next(i for i in snap["items"] if i["sid"] == SID1)
    assert item["state"] == "CURRENT_REVISION" and item["content_domain"] == "CANONICAL" and item["observations"] >= 3
    assert set(snap["health"]) == {"discovery_feed", "primary_statement"} and snap["discovery"]["state"]
    from scripts.trading_lab.fomc.store import FomcStore  # the API ships the FOMC derivation verbatim
    direct = snapshot.events_as_of(FomcStore(root, wall_clock=None, read_only=True),
                                   snapshot.clockmod.parse_iso(as_of), header["H"])
    assert direct["identity"] == header["identity"]
    detail = views.item(SID1, as_of=as_of)
    assert detail["item"]["revision"] == item["revision"] and len(detail["revisions"]) == 1  # Cloudflare churn merged
    observations = detail["observations"]
    assert len(observations) >= 3 and len({o["raw_sha256"] for o in observations}) == len(observations)  # distinct raws
    assert all(o["revision"] == item["revision"] and o["verdict"] == "CLOCK_VERIFIED" for o in observations)
    assert all(o["request_url"] == syn.url(P1) and o["byte_length"] > 0 for o in observations)
    replay = views.replay(as_of=as_of)
    assert replay["identical"] and replay["replay_identity"] == header["identity"] and replay["error"] is None
    assert _fingerprint(root) == before  # nothing was written


def test_a_read_at_an_earlier_horizon_or_after_now_lb_shows_what_was_known(views, closed_store):
    status = views.status()
    early = views.snapshot(as_of=status["first_durable_activity"], horizon=3)
    assert early["snapshot"]["H"] == 3 and early["snapshot"]["P"] == 0 and early["items"] == []  # nothing known yet
    with pytest.raises(NotFoundError):
        views.item(SID1, as_of=status["first_durable_activity"], horizon=3)
    _root, end = closed_store
    later = views.snapshot(as_of=iso(end + timedelta(days=1)))
    assert later["snapshot"]["read_state"] == "FOMC_CAUSAL_VISIBILITY_UNRESOLVED"  # never a guess past NOW_LB


def test_requests_outside_the_contract_are_refused(views, tmp_path):
    status = views.status()
    for bad in ({}, {"as_of": "2026-10-02T14:00:00"}, {"as_of": "yesterday"}):
        with pytest.raises(AppApiError):
            views.snapshot(**bad)
    with pytest.raises(AppApiError):
        views.snapshot(as_of=status["suggested_as_of"], horizon=status["horizon"] + 1)
    with pytest.raises(AppApiError):
        views.item("../../etc/passwd", as_of=status["suggested_as_of"])
    with pytest.raises(NotFoundError):
        views.item("0" * 64, as_of=status["suggested_as_of"])
    unconfigured = AppService(tmp_path).fomc
    assert unconfigured.status()["status"] == "NOT_CONFIGURED"
    with pytest.raises(NotFoundError):
        unconfigured.snapshot(as_of=status["suggested_as_of"])


def test_an_incompatible_or_corrupt_store_is_refused_not_repaired(closed_store, tmp_path):
    root, _end = closed_store
    old = tmp_path / "old"
    shutil.copytree(root, old)
    with sqlite3.connect(old / "fomc.sqlite3") as conn:
        conn.execute("UPDATE meta SET value = 'fomc-store-v4' WHERE name = 'schema_version'")
    views = AppService(tmp_path, fomc_store=old).fomc
    assert views.status()["status"] == "REJECTED"
    with pytest.raises(ConflictError):
        views.snapshot(as_of="2026-06-17T20:00:00+00:00")
    corrupt = tmp_path / "corrupt"
    shutil.copytree(root, corrupt)
    good = AppService(tmp_path, fomc_store=corrupt).fomc
    as_of = good.status()["suggested_as_of"]
    digest = good.item(SID1, as_of=as_of)["observations"][-1]["raw_sha256"]
    raw = corrupt / "raw" / digest[:2] / digest
    raw.write_bytes(raw.read_bytes()[:-1] + b"!")  # the newest statement raw no longer matches its digest
    with pytest.raises(ConflictError):
        good.snapshot(as_of=as_of)
    replay = good.replay  # the replay names the failure instead of returning an identity
    with pytest.raises(ConflictError):
        replay(as_of=as_of)


def test_the_router_serves_the_journey(closed_store, tmp_path):
    """The route table and the dynamic item route, dispatched like the HTTP handler does (no socket)."""
    root, _end = closed_store
    handler = type("H", (), {"service": AppService(tmp_path, fomc_store=root)})()

    def get(path, **query):
        return server_module.AppApiHandler._dispatch(handler, path, {k: [v] for k, v in query.items()})
    status = get("/api/v1/sources/fomc")
    as_of = status["suggested_as_of"]
    assert get("/api/v1/sources/fomc/snapshot", as_of=as_of)["snapshot"]["read_state"] == "FOMC_RESOLVED"
    detail = get(f"/api/v1/sources/fomc/items/{SID1}", as_of=as_of)
    assert detail["item"]["sid"] == SID1 and detail["observations"]
    assert get("/api/v1/sources/fomc/replay", as_of=as_of)["identical"] is True
    with pytest.raises(AppApiError):
        get("/api/v1/sources/fomc/snapshot")
    with pytest.raises(AppApiError):
        get("/api/v1/sources/fomc/items")


def test_item_detail_pages_a_history_above_the_source_bound(views, monkeypatch):
    from scripts.trading_lab.app_api import sources
    monkeypatch.setattr(sources, "MAX_SOURCE_ITEMS", 1)
    page = views.item(SID1, as_of=views.status()["suggested_as_of"])
    assert len(page["observations"]) == 1
    assert page["pagination"]["next_cursor"]
    assert page["pagination"]["totals"]["observations"] > 1
