"""Synthetic scale, frozen identity, cost and raw-integrity regressions."""

from datetime import timedelta
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError
from scripts.trading_lab.app_api.sources import EdgarViews, FomcViews
from scripts.trading_lab.app_api.server import AppApiHandler
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.edgar import snapshot
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.sources.scale import build_edgar, measure
from tests.crypto.test_app_api_fomc import closed_store as fomc_closed_store
from tests.crypto.fomc_support import SID1


@pytest.fixture(scope="module")
def large_edgar(tmp_path_factory):
    root = tmp_path_factory.mktemp("edgar-scale")
    T, H = build_edgar(root, polls=4)
    return root, T, H


def concatenate(operation, fields, *, limit=137):
    cursor = None
    rows = {name: [] for name in fields}
    identities = set()
    while True:
        page = operation(limit=limit, cursor=cursor)
        identities.add(page["snapshot"]["identity"])
        for name in fields:
            assert len(page[name]) <= limit
            rows[name].extend(page[name])
        cursor = page["pagination"]["next_cursor"]
        if cursor is None:
            assert {name: len(values) for name, values in rows.items()} == {
                name: page["pagination"]["totals"][name] for name in fields}
            break
    assert len(identities) == 1
    return rows


def test_ten_thousand_filings_and_timeline_page_without_rederiving(large_edgar):
    root, T, H = large_edgar
    views = EdgarViews(root)
    try:
        store = views._open()
        first, cost = measure(store, lambda: views.snapshot(as_of=T.isoformat(), horizon=H, limit=137))
        assert first["pagination"]["totals"]["filings"] == 10000
        full = views._cached[1]
        # Cold read decodes every stored row/txn once, with linear view work.
        assert cost["rows"] == len(store.rows()) + len(store.txns())
        assert cost["view_rows"] < 4 * cost["rows"]
        def page(**kw):
            result, warm = measure(store, lambda: views.snapshot(as_of=T.isoformat(), horizon=H, **kw))
            assert warm["rows"] == warm["view_rows"] == 0
            assert warm["queries"] <= 1
            assert warm["payload_bytes"] < 150000
            return result
        filings = concatenate(page, ["filings"])["filings"]
        assert [f["accession_number"] for f in filings] == [f["accession_number"] for f in full["filings"]]
        assert first["snapshot"]["identity"] == snapshot.filings_as_of(store, T, H)["identity"]
        status, status_cost = measure(store, views.status)
        assert status["counts"]["observations"] == 42000
        assert status_cost["rows"] == 0 and status_cost["view_rows"] <= 3

        initial = views.timeline(as_of=T.isoformat(), horizon=H, limit=137)
        expected = views._timeline[:]
        assert len(expected) > 40000
        def timeline(**kw):
            result, warm = measure(store, lambda: views.timeline(as_of=T.isoformat(), horizon=H, **kw))
            assert warm["rows"] == 0 and warm["view_rows"] <= kw["limit"]
            return result
        assert concatenate(timeline, ["rows"])["rows"] == expected
        assert initial["snapshot"] == first["snapshot"]
    finally:
        views.close()


def test_detail_pages_concatenate_and_warm_cost_is_bounded(large_edgar):
    root, T, H = large_edgar
    views = EdgarViews(root)
    accession = "0000100000-26-000000"
    try:
        first = views.filing(accession, as_of=T.isoformat(), horizon=H, limit=1)
        full = views._detail[1]
        def detail(**kw):
            result, cost = measure(views._reader, lambda: views.filing(accession, as_of=T.isoformat(), horizon=H, **kw))
            assert cost["rows"] == 0
            assert cost["view_rows"] <= kw["limit"]
            return result
        joined = concatenate(detail, ["observations", "revisions", "absences"], limit=1)
        assert joined == {name: full[name] for name in joined}
        assert len(joined["observations"]) == 4
        assert first["pagination"]["next_cursor"]
    finally:
        views.close()


def test_detail_lookup_at_end_of_large_snapshot_does_not_walk_other_filings(large_edgar):
    root, T, H = large_edgar
    views = EdgarViews(root)
    class CountedFilings(list):
        visited = 0

        def __iter__(self):
            for filing in super().__iter__():
                self.visited += 1
                yield filing
    try:
        views.snapshot(as_of=T.isoformat(), horizon=H)
        full = views._cached[1]
        accession = full["filings"][-1]["accession_number"]
        counted = CountedFilings(full["filings"])
        full["filings"] = counted
        first = views.filing(accession, as_of=T.isoformat(), horizon=H, limit=1)
        second = views.filing(accession, as_of=T.isoformat(), horizon=H, limit=1,
                              cursor=first["pagination"]["next_cursor"])
        assert second["filing"]["accession_number"] == accession
        assert counted.visited == 0
        assert first["observations"][0] != second["observations"][0]
    finally:
        views.close()


def test_large_replay_reads_each_processing_transaction_without_history_rescans(large_edgar):
    root, T, H = large_edgar
    store = EdgarStore(root, wall_clock=None, read_only=True)
    try:
        full = snapshot.filings_as_of(store, T, H)
        recorded_rows = len(store.rows()) + len(store.txns())
        replayed, cost = measure(store, lambda: snapshot.replay(store, T, H))
        assert replayed == full
        assert cost["rows"] == 0
        assert cost["view_rows"] < 8 * recorded_rows
    finally:
        store.close()


def test_versioned_latest_events_equal_full_scans_at_historical_horizons(tmp_path):
    from scripts.trading_lab.edgar.collector import latest_events
    from scripts.trading_lab.edgar import synthetic as syn
    from tests.crypto.test_edgar_slice import Env, ACC1, OLD, A
    env = Env(tmp_path)
    try:
        horizons = []
        for filings in ((syn.filing(ACC1), OLD), (OLD,), (syn.filing(ACC1, items="synthetic correction"), OLD)):
            env.serve(*filings)
            env.poll()
            horizons.append(env.store.horizon())
        for H in horizons:
            view = env.store.view(H)
            expected = {}
            for kind in ("FILING_OBSERVATION", "FILING_ABSENCE"):
                for row in view.select(kind, "cik", A):
                    acc = row.body["accession_number"]
                    old = expected.get(acc)
                    if old is None or row.seq > old.seq or (row.seq == old.seq and kind == "FILING_ABSENCE"):
                        expected[acc] = row
            assert latest_events(view, A) == expected
            for kind in ("FILING_OBSERVATION", "FILING_ABSENCE", "RESPONSE"):
                assert view.count(kind) == len(view.rows(kind))
                for seq in range(1, H + 2):
                    assert view.rows_at(kind, seq) == [r for r in view.rows(kind) if r.seq == seq]
            txns = view.txns()
            assert view.activity() == (txns[0][2], txns[-1][2])
            seen = [snapshot.causal.observed_at(r) for r in view.rows("RESPONSE") if snapshot.causal.verified(r)]
            assert snapshot.now_lb(view) == max(seen) - snapshot.spec.CLOCK_ERROR_BOUND
    finally:
        env.collector.close()
        env.store.close()


def test_cursors_refuse_other_reads_entities_and_endpoints(large_edgar):
    root, T, H = large_edgar
    views = EdgarViews(root)
    try:
        first = views.snapshot(as_of=T.isoformat(), horizon=H, limit=1)
        cursor = first["pagination"]["next_cursor"]
        for args in ({"as_of": (T - timedelta(seconds=1)).isoformat(), "horizon": H},
                     {"as_of": T.isoformat(), "horizon": H - 1}):
            with pytest.raises(AppApiError, match="different query"):
                views.snapshot(**args, cursor=cursor)
        with pytest.raises(AppApiError, match="different endpoint"):
            views.timeline(as_of=T.isoformat(), horizon=H, cursor=cursor)
        detail = views.filing("0000100000-26-000000", as_of=T.isoformat(), horizon=H, limit=1)
        with pytest.raises(AppApiError, match="different product"):
            views.filing("0000100000-26-000001", as_of=T.isoformat(), horizon=H,
                         cursor=detail["pagination"]["next_cursor"])
        for bad in (0, 1001, "bad", -1):
            with pytest.raises(AppApiError):
                views.snapshot(as_of=T.isoformat(), horizon=H, limit=bad)
        with pytest.raises(AppApiError):
            views.snapshot(as_of=T.isoformat(), horizon=H, cursor="bad")
    finally:
        views.close()


def test_fomc_history_and_nested_links_are_bounded(fomc_closed_store):
    root, _ = fomc_closed_store
    views = FomcViews(root)
    try:
        T = views.status()["suggested_as_of"]
        first = views.item(SID1, as_of=T, limit=1)
        full = views._detail[1]
        joined = concatenate(lambda **kw: views.item(SID1, as_of=T, horizon=first["snapshot"]["H"], **kw),
                             ["observations", "revisions"], limit=1)
        assert joined["observations"] == full["observations"]
        assert joined["revisions"] == full["revisions"]
        cursor, links = None, []
        while True:
            page, cost = measure(views._reader, lambda: views.item(SID1, as_of=T, limit=1, cursor=cursor))
            assert cost["rows"] == cost["view_rows"] == 0
            assert len(page["item"].get("links", [])) <= 1
            links.extend(page["item"].get("links", []))
            cursor = page["pagination"]["next_cursor"]
            if cursor is None:
                break
        assert links == full["item"]["links"]
        views.timeline(as_of=T, horizon=first["snapshot"]["H"], limit=17)
        expected_timeline = views._timeline[:]
        def timeline(**kw):
            result, cost = measure(views._reader, lambda: views.timeline(
                as_of=T, horizon=first["snapshot"]["H"], **kw))
            assert cost["rows"] == 0 and cost["view_rows"] <= kw["limit"]
            return result
        assert concatenate(timeline, ["rows"], limit=17)["rows"] == expected_timeline
    finally:
        views.close()


def test_cache_hits_reverify_raw_integrity(tmp_path):
    root = tmp_path / "store"
    T, H = build_edgar(root, ciks=1, listing_rows=3, polls=3)
    views = EdgarViews(root)
    try:
        first = views.snapshot(as_of=T.isoformat(), horizon=H, limit=1)
        digest = views._cached[2][0]
        (root / "raw" / digest[:2] / digest).write_bytes(b"synthetic corruption")
        with pytest.raises(ConflictError, match="integrity"):
            views.snapshot(as_of=T.isoformat(), horizon=H, cursor=first["pagination"]["next_cursor"])
    finally:
        views.close()


def test_timeline_verifies_older_revision_raws_not_used_by_current_snapshot(tmp_path):
    from scripts.trading_lab.edgar import synthetic as syn
    from tests.crypto.test_edgar_slice import Env, ACC1
    env = Env(tmp_path)
    views = EdgarViews(env.root)
    try:
        middle = None
        for items in ("first", "middle", "first"):
            env.serve(syn.filing(ACC1, items=items))
            result = env.poll()
            if items == "middle":
                middle = env.store.row_at("RESPONSE", result["record"]).body["raw_sha"]
        snap = env.settle()
        views.snapshot(as_of=snap["T"], horizon=snap["H"])
        # Populate the timeline, then prove its cached page still verifies history.
        views.timeline(as_of=snap["T"], horizon=snap["H"])
        (env.root / "raw" / middle[:2] / middle).write_bytes(b"synthetic corruption")
        views.snapshot(as_of=snap["T"], horizon=snap["H"])
        with pytest.raises(ConflictError, match="integrity"):
            views.timeline(as_of=snap["T"], horizon=snap["H"])
    finally:
        views.close()
        env.collector.close()
        env.store.close()


def test_more_than_one_thousand_observations_page_with_constant_warm_cost(tmp_path):
    root = tmp_path / "long"
    T, H = build_edgar(root, ciks=1, listing_rows=1, polls=1005)
    views = EdgarViews(root)
    try:
        first = views.filing("0000100000-26-000000", as_of=T.isoformat(), horizon=H)
        assert first["pagination"]["totals"]["observations"] == 1005
        assert len(first["observations"]) == 200
        def detail(**kw):
            result, cost = measure(views._reader, lambda: views.filing(
                "0000100000-26-000000", as_of=T.isoformat(), horizon=H, **kw))
            assert cost["rows"] == 0 and cost["view_rows"] <= kw["limit"]
            return result
        joined = concatenate(detail, ["observations", "revisions", "absences"], limit=200)
        assert len(joined["observations"]) == 1005
        assert len({row["record"] for row in joined["observations"]}) == 1005
    finally:
        views.close()


def test_append_preserves_cursor_at_old_horizon_and_schema_changes_refuse(tmp_path):
    import sqlite3
    root = tmp_path / "append"
    T, H = build_edgar(root, ciks=1, listing_rows=3, polls=3)
    views = EdgarViews(root)
    try:
        first = views.snapshot(as_of=T.isoformat(), horizon=H, limit=1)
        cursor = first["pagination"]["next_cursor"]
        before = views.snapshot(as_of=T.isoformat(), horizon=H, limit=1, cursor=cursor)
        writer = EdgarStore(root, wall_clock=lambda: T)
        try:
            writer.append("SYNTHETIC", [("EPOCH", "synthetic-append", {})])
        finally:
            writer.close()
        after, cost = measure(views._reader, lambda: views.snapshot(
            as_of=T.isoformat(), horizon=H, limit=1, cursor=cursor))
        assert after == before
        assert cost["rows"] == 2 and cost["view_rows"] == 0  # only the appended row and txn
        with pytest.raises(AppApiError, match="different query"):
            views.snapshot(as_of=T.isoformat(), cursor=cursor)
        with sqlite3.connect(root / EdgarStore.DB_NAME) as conn:
            conn.execute("UPDATE meta SET value = 'synthetic-other-spec' WHERE name = 'spec_hash'")
        with pytest.raises(ConflictError, match="schema/spec"):
            views.snapshot(as_of=T.isoformat(), horizon=H)
    finally:
        views.close()


def test_router_and_concurrent_pages_share_one_frozen_read(large_edgar, tmp_path):
    root, T, H = large_edgar
    service = AppService(tmp_path, edgar_store=root)
    handler = type("H", (), {"service": service})()
    query = {"as_of": [T.isoformat()], "horizon": [str(H)], "limit": ["3"]}
    try:
        first = AppApiHandler._dispatch(handler, "/api/v1/sources/edgar/snapshot", query)
        query["cursor"] = [first["pagination"]["next_cursor"]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            pages = list(pool.map(lambda _: AppApiHandler._dispatch(handler, "/api/v1/sources/edgar/snapshot", query), range(3)))
        assert pages[0] == pages[1] == pages[2]
        assert pages[0]["filings"][0] != first["filings"][0]
        query.pop("cursor")
        assert len(AppApiHandler._dispatch(handler, "/api/v1/sources/edgar/timeline", query)["rows"]) == 3
        assert len(AppApiHandler._dispatch(handler, "/api/v1/sources/edgar/filings/0000100000-26-000000", query)["observations"]) == 3
        assert str(root) not in json.dumps(pages)
    finally:
        service.edgar.close()
