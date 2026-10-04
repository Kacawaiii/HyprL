"""Attested PIT features against real-shaped synthetic stores; no official requests."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.trading_lab.edgar import snapshot as es, synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.event_features import EventFeatures, SourceJoin
from scripts.trading_lab.event_features.features import values
from scripts.trading_lab.event_features.join import instant
from scripts.trading_lab.event_features.mapping import MAPPING, product_mapping
from scripts.trading_lab.event_features.matrix import build_matrix, fingerprint
from scripts.trading_lab.fomc import synthetic as fsyn
from tests.crypto.fomc_support import Env, P1, statement_item

A, B = "0000320193", "0000789019"
ACC, AMEND, MS = "0000320193-26-000071", "0000320193-26-000072", "0000789019-26-000051"
MICRO = timedelta(microseconds=1)


class EdgarEnv:
    def __init__(self, root, ciks=(A, B)):
        self.root = root / "edgar"
        self.clock = syn.SimClock(datetime(2026, 10, 4, tzinfo=timezone.utc))
        self.fetch = syn.FakeFetcher(self.clock)
        self.store = EdgarStore(self.root, wall_clock=self.clock.wall)
        self.collector = EdgarCollector(self.store, self.fetch, self.clock)
        self.collector.submit_watchlist(list(ciks))

    def serve(self, cik, *filings):
        doc = json.loads(syn.listing(cik, list(filings)))
        # Real fixture qualification: submissions.cik is a zero-padded string.
        doc["cik"] = cik
        doc["name"] = "Synthetic Apple" if cik == A else "Synthetic Microsoft"
        self.fetch.routes[cik] = syn.Reply(json.dumps(doc).encode())

    def poll(self, cik=A, gap=600):
        self.collector.poll(cik)
        self.clock.sleep(gap)

    def settle(self):
        self.poll()
        self.poll()
        return instant(self.store.rows("RESPONSE")[-2].body["observed_at"]) + timedelta(seconds=93)

    def close(self):
        self.collector.close()
        self.store.close()


@pytest.fixture
def edgar(tmp_path):
    env = EdgarEnv(tmp_path)
    env.serve(A, syn.filing(ACC, filed="2001-01-01", acceptance="2001-01-01T00:00:00.000Z"),
              syn.filing(AMEND, form="8-K/A", items="5.02,7.01,8.01"))
    env.serve(B, syn.filing(MS, items="8.01"))
    env.poll(A)
    env.poll(B)
    env.T = env.settle()
    yield env
    env.close()


@pytest.fixture
def fomc(tmp_path):
    env = Env(tmp_path)
    env.feed([statement_item()])
    env.provider.routes[P1] = fsyn.page_response(body="Synthetic initial statement.")
    env.drive(360)
    yield env
    env.collector.close()
    env.store.close()
    env.provider.close()


def test_mapping_matches_catalogue_and_evidence():
    for m in MAPPING["products"]:
        assert product_mapping(m["product"]) == m
        assert m["evidence"] and m["fomc"] == "MACRO"
    assert product_mapping("xnas:AAPL")["cik"] == A
    assert product_mapping("NVDA")["cik"] is None
    assert product_mapping("QQQ")["cik"] == "0001067839"


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_batch_equals_public_per_t_snapshot_at_all_boundaries(source, fomc, edgar, monkeypatch):
    root = fomc.root if source == "fomc" else edgar.root
    reader = SourceJoin(source, root)
    try:
        assert reader.error is None and reader.store.read_only
        times = sorted({t + d for t in reader.values for d in (-MICRO, timedelta(), MICRO)})
        times += [times[0] - timedelta(days=1), times[-1] + timedelta(days=1)]
        expected = [reader.read_one(T) for T in times]
        # Once the reader is built, the batch path must not recompute availability.
        module = "scripts.trading_lab.fomc.state" if source == "fomc" else "scripts.trading_lab.edgar.snapshot"
        monkeypatch.setattr(module + ".availability", lambda *_: pytest.fail("availability rescanned in batch"))
        assert reader.read_many(times) == expected
        assert reader.read_many(list(reversed(times))) == list(reversed(expected))
        before, after = reader.read_many([reader.values[0] - MICRO, reader.values[-1] + MICRO])
        assert before["state"] == "NOT_OBSERVED" and before["events"] is None
        assert after["state"] == "UNRESOLVED" and after["events"] is None
        assert reader.read_many([reader.values[-1]])[0]["state"] == "UNRESOLVED"
    finally:
        reader.close()


def test_attested_boundary_counts_and_no_historical_backfill(edgar):
    with SourceJoin("edgar", edgar.root) as reader:
        snapshot = reader.read_one(edgar.T)
        assert snapshot["state"] == "RESOLVED"
        first = instant(next(e for e in snapshot["events"] if e["cik"] == A)["available_at"])
    with EventFeatures(edgar_store=edgar.root) as features:
        earlier, exact, historical = features.rows([("AAPL", first - MICRO), ("AAPL", first),
                                                   ("AAPL", "2001-01-01T00:00:00Z")])
        assert earlier["sources"]["edgar"]["features"]["count_7d"] == 0  # resolved but before first filing
        f = exact["sources"]["edgar"]["features"]
        assert f == {"count_7d": 2, "count_30d": 2, "hours_since_last": 0.0,
                     "item_2_02": True, "item_5_02": True, "item_7_01": True, "item_8_01": True}
        assert historical["sources"]["edgar"]["state"] == "NOT_OBSERVED"
        assert all(v is None for v in historical["sources"]["edgar"]["features"].values())


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_window_edges(source):
    T = instant("2026-08-01T00:00:00Z")
    def e(at, n):
        return {"event_id": str(n), "available_at": at, "items": "2.02,5.02,7.01,8.01"}
    events = [e(T - timedelta(days=30), 1), e(T - timedelta(days=30) + MICRO, 2),
              e(T - timedelta(days=7), 3), e(T - timedelta(days=7) + MICRO, 4),
              e(T, 5), e(T + MICRO, 6)]
    result = values(events, T, source)
    assert result["count_7d"] == 2 and result["count_30d"] == 4 and result["hours_since_last"] == 0
    if source == "fomc":
        assert result["statement_within_24h"]
        assert values([e(T - timedelta(hours=24), 7)], T, source)["statement_within_24h"]
        assert not values([e(T - timedelta(hours=24) - MICRO, 7)], T, source)["statement_within_24h"]
    else:
        assert all(result[k] for k in ("item_2_02", "item_5_02", "item_7_01", "item_8_01"))
        assert not values([e(T - timedelta(days=7), 7)], T, source)["item_2_02"]
        assert not values([{**e(T, 7), "items": "12.02,15.02,17.01,18.01"}], T, source)["item_2_02"]


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_resolved_empty_counts_are_zero_last_is_null(source):
    result = values([], instant("2026-06-01T00:00:00Z"), source)
    assert result["count_7d"] == result["count_30d"] == 0 and result["hours_since_last"] is None
    assert all(v is False for k, v in result.items() if k.startswith("item_") or k == "statement_within_24h")


def test_missing_optional_edgar_items_do_not_become_false():
    T = instant("2026-08-01T00:00:00Z")
    unknown = {"event_id": "a", "available_at": T, "items": None}
    known = {"event_id": "b", "available_at": T, "items": "2.02"}
    result = values([unknown, known], T, "edgar")
    assert result["count_7d"] == 2 and result["item_2_02"] is True and result["item_5_02"] is None


def test_late_edgar_revision_uses_old_metadata_at_old_T(edgar):
    oldT, oldH = edgar.T, edgar.store.horizon()
    with EventFeatures(edgar_store=edgar.root) as features:
        old = features.rows([("AAPL", oldT)])[0]
    edgar.serve(A, syn.filing(ACC, items="9.01"), syn.filing(AMEND, form="8-K/A", items="9.01"))
    edgar.poll()
    newT = edgar.settle()
    with EventFeatures(edgar_store=edgar.root) as features:
        early, late = features.rows([("AAPL", oldT), ("AAPL", newT)])
    assert early["sources"]["edgar"]["features"] == old["sources"]["edgar"]["features"]
    assert late["sources"]["edgar"]["features"]["item_2_02"] is False
    assert late["sources"]["edgar"]["features"]["count_7d"] == 2  # revisions do not add filings
    with SourceJoin("edgar", edgar.root, horizon=oldH) as pinned:
        assert pinned.read_one(oldT)["snapshot"]["identity"] == old["sources"]["edgar"]["snapshot_identity"]


def test_fomc_revision_earlier_reads_and_event_availability_stay_unchanged(fomc):
    T = fomc.clock.true
    with SourceJoin("fomc", fomc.root) as reader:
        oldH = reader.H
        old = reader.read_one(T)
        assert len(old["events"]) == 1
    fomc.provider.routes[P1] = fsyn.page_response(body="Synthetic corrected statement.")
    fomc.drive(1000, idle=60)
    with SourceJoin("fomc", fomc.root) as reader:
        early, late = reader.read_many([T, fomc.clock.true])
    assert early["events"] == old["events"]
    assert late["events"][0]["revision"] != old["events"][0]["revision"]
    assert late["events"][0]["available_at"] == old["events"][0]["available_at"]
    with SourceJoin("fomc", fomc.root, horizon=oldH) as reader:
        assert reader.read_one(T) == old


def test_fomc_manifest_never_uses_declared_release_as_availability(tmp_path):
    env = Env(tmp_path)
    try:
        env.feed([])
        env.provider.routes[P1] = fsyn.page_response()
        env.collector.submit_manifest(json.dumps({"version": 1, "urls": [fsyn.url(P1)]}).encode(), "synthetic manifest")
        env.drive(360)
        with SourceJoin("fomc", env.root) as reader:
            old = reader.read_one("2026-06-17T18:00:00Z")
            snap = reader.read_one(env.clock.true)
            assert old["state"] == "NOT_OBSERVED"  # declaration predates the observation start
            assert snap["state"] == "RESOLVED" and len(snap["events"]) == 1
            event = snap["events"][0]
            item = snap["snapshot"]["items"][0]
            assert item["links"][0]["observation_mode"] == "HISTORICAL_BACKFILL"
            assert instant(event["available_at"]) > instant(item["normalized"]["declared_release_at"])
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()


@pytest.mark.parametrize("product,state", [("BTC-USD", "NOT_APPLICABLE"), ("ETH-USD", "NOT_APPLICABLE"),
                                          ("QQQ", "NOT_APPLICABLE"), ("NVDA", "UNKNOWN_MAPPING")])
def test_nonapplicable_unknown_mapping_and_holdout(edgar, product, state):
    with EventFeatures(edgar_store=edgar.root) as features:
        row = features.rows([(product, edgar.T)])[0]
        assert row["inside_protected_holdout"]
        result = row["sources"]["edgar"]
        assert result["state"] == state and result["source_state"] == "RESOLVED"
        assert all(v is None for v in result["features"].values())


def test_holdout_bounds_and_not_configured():
    times = ["2026-08-31T23:59:59.999999Z", "2026-09-01T00:00:00Z", "2026-11-30T23:59:59.999999Z", "2026-12-01T00:00:00Z"]
    with EventFeatures() as features:
        rows = features.rows([("AAPL", t) for t in times])
    assert [r["inside_protected_holdout"] for r in rows] == [False, True, True, False]
    assert all(s["state"] == "NOT_CONFIGURED" for r in rows for s in r["sources"].values())
    assert all(v is None for r in rows for s in r["sources"].values() for v in s["features"].values())


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_feature_values_missing_outside_coverage(source, fomc, edgar):
    env = fomc if source == "fomc" else edgar
    with EventFeatures(**{source + "_store": env.root}) as features:
        reader = features.sources[source]
        rows = features.rows([("AAPL", reader.values[0] - MICRO), ("AAPL", reader.values[-1] + MICRO)])
        assert [r["sources"][source]["state"] for r in rows] == ["NOT_OBSERVED", "UNRESOLVED"]
        assert all(v is None for r in rows for v in r["sources"][source]["features"].values())


def test_unwatched_issuer_is_not_a_zero(tmp_path):
    env = EdgarEnv(tmp_path, ciks=(A,))
    try:
        env.serve(A, syn.filing(ACC))
        env.poll()
        T = env.settle()
        with EventFeatures(edgar_store=env.root) as features:
            row = features.rows([("MSFT", T)])[0]["sources"]["edgar"]
        assert row["source_state"] == "RESOLVED" and row["state"] == "NOT_OBSERVED"
        assert all(v is None for v in row["features"].values())
    finally:
        env.close()


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_corruption_after_a_successful_batch_fails_closed(source, fomc, edgar):
    env = fomc if source == "fomc" else edgar
    T = fomc.clock.true if source == "fomc" else edgar.T
    with SourceJoin(source, env.root) as reader:
        assert reader.read_many([T])[0]["state"] == "RESOLVED"
        snap = reader.read_one(T)["snapshot"]
        digest = (snap["items"][0]["normalized"]["first_raw_sha256"] if source == "fomc" else
                  snap["filings"][0]["provenance"]["first_raw_sha256"])
        (env.root / "raw" / digest[:2] / digest).write_bytes(b"synthetic corruption")
        batch, single = reader.read_many([T])[0], reader.read_one(T)
        assert batch["state"] == single["state"] == "INTEGRITY_ERROR"
        assert batch["snapshot"] is None and batch["events"] is None and batch["reason"]
    with EventFeatures(**{source + "_store": env.root}) as features:
        row = features.rows([("AAPL", T)])[0]["sources"][source]
        assert row["state"] == "INTEGRITY_ERROR" and all(v is None for v in row["features"].values())


def test_stores_unchanged_deterministic_outputs_and_matrix(fomc, edgar):
    before = [fingerprint(e.root) for e in (fomc, edgar)]
    decisions = [(p, t) for p in ("AAPL", "MSFT", "QQQ", "NVDA", "BTC-USD", "ETH-USD")
                 for t in ("2026-06-01T00:00:00Z", edgar.T, fomc.clock.true, "2026-12-01T00:00:00Z")]
    with EventFeatures(fomc.root, edgar.root) as features:
        first = features.rows(decisions)
        matrix = build_matrix(features.sources)
        assert features.rows(decisions) == first
        assert build_matrix(features.sources) == matrix
        assert all(r["price_coverage_overlap"] is None for r in matrix["rows"] if r["source"] == "edgar")
        # Returned objects cannot mutate a later read or another row.
        first[0]["sources"]["fomc"]["features"]["count_7d"] = 999
        assert features.rows(decisions)[0] != first[0]
    assert [fingerprint(e.root) for e in (fomc, edgar)] == before


def test_invalid_time_and_failed_open_are_explicit(tmp_path):
    with SourceJoin("edgar", tmp_path / "missing") as reader:
        assert reader.read_one("2026-01-01T00:00:00Z")["state"] == "INTEGRITY_ERROR"
    with EventFeatures() as features:
        with pytest.raises(ValueError, match="offset"):
            features.rows([("AAPL", datetime(2026, 1, 1))])
    with pytest.raises(ValueError):
        SourceJoin("guessed")
