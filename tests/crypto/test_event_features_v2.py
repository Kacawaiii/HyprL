"""EVENT_FEATURES_V2: inventory is onboarding, new observations are news, revisions are separate."""

from datetime import timedelta
import json
from pathlib import Path

import pytest

from scripts.trading_lab.edgar import synthetic as syn
from scripts.trading_lab.event_features import EventFeatures, features as v1_features
from scripts.trading_lab.event_features import v2
from scripts.trading_lab.event_features.join import SourceJoin, instant
from scripts.trading_lab.fomc import synthetic as fsyn
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.crypto.fomc_support import Env, P1, statement_item
from tests.crypto.test_event_features import ACC, AMEND, MICRO, MS, A, EdgarEnv

NEW = "0000320193-26-000090"
P2 = fsyn.statement_path("20260729")
REPO = Path(__file__).resolve().parents[2]
KINDS = lambda obs: [(o["kind"]) for o in obs]  # noqa: E731


@pytest.fixture
def edgar(tmp_path):
    env = EdgarEnv(tmp_path)
    env.serve(A, syn.filing(ACC), syn.filing(AMEND, form="8-K/A", items="5.02,7.01,8.01"))
    env.serve("0000789019", syn.filing(MS, items="8.01"))
    env.poll(A)
    env.poll("0000789019")
    env.T0 = env.settle()
    yield env
    env.close()


def with_new_accession(env, *, items="2.02"):
    env.serve(A, syn.filing(ACC), syn.filing(AMEND, form="8-K/A", items="5.02,7.01,8.01"),
              syn.filing(NEW, items=items, filed="2026-10-05"))
    env.poll()
    return env.settle()


def aapl(env, *times, version=v2):
    with version.EventFeaturesV2(edgar_store=env.root) as features:
        return features.rows([("AAPL", t) for t in times])


def edgar_features(row):
    return row["sources"]["edgar"]["features"]


def test_inventory_burst_is_not_counted_as_new(edgar):
    row = aapl(edgar, edgar.T0)[0]
    f = edgar_features(row)
    assert row["sources"]["edgar"]["state"] == "RESOLVED"
    assert f["new_accessions_attested_7d"] == f["new_accessions_attested_30d"] == 0
    assert f["accession_revisions_attested_7d"] == 0 and f["hours_since_last_new_accession"] is None
    assert not any(f[k] for k in f if k.startswith("new_accession_item_"))
    # the inventory is metadata, with the attested moment of the first valid listing
    inventory = row["sources"]["edgar"]["inventory"]
    assert inventory["inventory_observations_attested"] == 2
    assert inventory["first_valid_read_available_at"] == inventory["inventory_last_available_at"]
    assert not any("inventory" in key for key in f)
    # the same instant counted two events in V1: that is the burst V2 stops calling news
    with EventFeatures(edgar_store=edgar.root) as old:
        assert old.rows([("AAPL", edgar.T0)])[0]["sources"]["edgar"]["features"]["count_7d"] == 2
    with SourceJoin("edgar", edgar.root) as reader:
        assert KINDS(v2.classify(reader)) == [v2.INITIAL_INVENTORY] * 3


def test_a_later_new_accession_is_counted_and_boundaries(edgar):
    T1 = with_new_accession(edgar)
    with SourceJoin("edgar", edgar.root) as reader:
        observations = v2.classify(reader)
    new = [o for o in observations if o["kind"] == v2.NEWLY_OBSERVED]
    assert len(new) == 1 and new[0]["cik"] == A and new[0]["items"] == "2.02"
    assert [o["kind"] for o in observations if o["cik"] == A].count(v2.INITIAL_INVENTORY) == 2
    at = new[0]["available_at"]
    before, exact = aapl(edgar, at - MICRO, at)
    # attested availability is the condition of use: closed on the right
    assert edgar_features(before)["new_accessions_attested_7d"] == 0
    assert edgar_features(exact)["new_accessions_attested_7d"] == 1
    assert edgar_features(exact)["hours_since_last_new_accession"] == 0
    assert edgar_features(exact)["new_accession_item_2_02_7d"] is True
    # the earlier read keeps its value after later evidence arrives
    assert edgar_features(aapl(edgar, edgar.T0)[0])["new_accessions_attested_7d"] == 0
    assert T1 > at


@pytest.mark.parametrize("source", ["fomc", "edgar"])
def test_window_edges_open_left_closed_right(source):
    T = instant("2027-05-01T00:00:00Z")
    day7, day30 = timedelta(days=7), timedelta(days=30)

    def obs(kind, at, n):
        return {"kind": kind, "available_at": at, "event_id": str(n), "items": "2.02,5.02"}
    new = [obs(v2.NEWLY_OBSERVED, t, i) for i, t in enumerate(
        [T - day30, T - day30 + MICRO, T - day7, T - day7 + MICRO, T, T + MICRO])]
    f, _ = v2.values(new, T, source)
    n = "statements" if source == "fomc" else "accessions"
    assert f[f"new_{n}_attested_7d"] == 2 and f[f"new_{n}_attested_30d"] == 4  # T-7d and T-30d excluded
    assert f["hours_since_last_new_" + n[:-1]] == 0
    revisions = [obs(v2.REVISION, t, i) for i, t in enumerate([T - day7, T - day7 + MICRO, T, T + MICRO])]
    f, _ = v2.values(revisions + [obs(v2.INITIAL_INVENTORY, T, 9)], T, source)
    assert f[f"{n[:-1]}_revisions_attested_7d"] == 2 and f[f"new_{n}_attested_30d"] == 0
    assert f["hours_since_last_new_" + n[:-1]] is None
    if source == "fomc":
        edge = lambda at: v2.values([obs(v2.NEWLY_OBSERVED, at, 1)], T, "fomc")[0]["new_statement_attested_within_24h"]  # noqa: E731
        assert edge(T - timedelta(hours=24)) and not edge(T - timedelta(hours=24) - MICRO)
    else:
        # items of the new accessions in the 7-day window only; unknown items are null, never false
        f, _ = v2.values([obs(v2.NEWLY_OBSERVED, T, 1), {**obs(v2.NEWLY_OBSERVED, T, 2), "items": None}], T, "edgar")
        assert f["new_accession_item_2_02_7d"] is True and f["new_accession_item_8_01_7d"] is None
        f, _ = v2.values([obs(v2.REVISION, T, 1), obs(v2.INITIAL_INVENTORY, T, 2)], T, "edgar")
        assert f["new_accession_item_2_02_7d"] is False  # neither revisions nor inventory carry items


def test_filing_date_never_orders_or_counts(edgar):
    # an accession dated long before the store existed, first seen later, is newly observed today
    edgar.serve(A, syn.filing(ACC), syn.filing(AMEND, form="8-K/A", items="5.02,7.01,8.01"),
                syn.filing(NEW, filed="2001-01-01", acceptance="2001-01-01T00:00:00.000Z"))
    edgar.poll()
    T = edgar.settle()
    assert edgar_features(aapl(edgar, T)[0])["new_accessions_attested_7d"] == 1


def test_revision_is_separated_from_new_accessions(edgar):
    edgar.serve(A, syn.filing(ACC, items="9.01"), syn.filing(AMEND, form="8-K/A", items="5.02,7.01,8.01"))
    edgar.poll()
    T = edgar.settle()
    with SourceJoin("edgar", edgar.root) as reader:
        observations = v2.classify(reader)
    revisions = [o for o in observations if o["kind"] == v2.REVISION]
    assert len(revisions) == 1 and revisions[0]["items"] == "9.01"
    at = revisions[0]["available_at"]
    f = edgar_features(aapl(edgar, T)[0])
    assert f["accession_revisions_attested_7d"] == 1 and f["accession_revisions_attested_30d"] == 1
    assert f["new_accessions_attested_7d"] == 0 and f["hours_since_last_new_accession"] is None
    assert edgar_features(aapl(edgar, at - MICRO)[0])["accession_revisions_attested_7d"] == 0
    # an unchanged re-listing is not an event
    edgar.poll()
    again = edgar.settle()
    with SourceJoin("edgar", edgar.root) as reader:
        assert len(v2.classify(reader)) == len(observations)
    assert edgar_features(aapl(edgar, again)[0])["accession_revisions_attested_7d"] == 1


def test_first_valid_listing_per_issuer_not_first_in_store(edgar, tmp_path):
    # Microsoft's first valid listing came after Apple's: its filing is inventory, not new
    with SourceJoin("edgar", edgar.root) as reader:
        by_cik = {o["cik"]: o["kind"] for o in v2.classify(reader)}
    assert by_cik == {A: v2.INITIAL_INVENTORY, "0000789019": v2.INITIAL_INVENTORY}


def test_unwatched_or_inapplicable_states_stay_null(edgar):
    T = edgar.T0
    with v2.EventFeaturesV2(edgar_store=edgar.root) as features:
        rows = features.rows([("BTC-USD", T), ("QQQ", T), ("NVDA", T), ("AAPL", instant("2026-01-01T00:00:00Z")),
                              ("AAPL", T + timedelta(days=400))])
    states = [r["sources"]["edgar"]["state"] for r in rows]
    assert states == ["NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN_MAPPING", "NOT_OBSERVED", "UNRESOLVED"]
    assert all(v is None for r in rows for v in r["sources"]["edgar"]["features"].values())
    assert all(r["sources"]["edgar"]["inventory"] is None for r in rows)
    assert all(set(r["sources"]["fomc"]["features"]) == set(v2.FOMC_FEATURES) for r in rows)
    assert all(r["sources"]["fomc"]["state"] == "NOT_CONFIGURED" for r in rows)


def test_not_configured_and_protection_flags():
    times = ["2026-08-31T23:59:59.999999Z", "2026-09-01T00:00:00Z", "2026-11-30T23:59:59.999999Z",
             "2026-12-01T00:00:00Z", "2026-12-31T00:00:00Z"]
    with v2.EventFeaturesV2() as features:
        rows = features.rows([("BTC-USD", t) for t in times] + [("AAPL", "2026-12-01T00:00:00Z"),
                                                                ("AAPL", "2027-03-01T00:00:00Z"),
                                                                ("AAPL", "2027-03-31T00:00:00Z")])
    assert [r["protection"]["inside"] != [] for r in rows] == [False, True, True, False, False, True, False, False]
    # the 30-day event window (T-30d, T] also touches the interval for 30 days after it
    assert [bool(r["protection"]["event_window_touches"]) for r in rows[:5]] == [False, True, True, True, False]
    assert [bool(r["protection"]["event_window_touches"]) for r in rows[5:]] == [True, True, False]
    assert all(v is None for r in rows for s in r["sources"].values() for v in s["features"].values())


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


def fomc_rows(env, *times):
    with v2.EventFeaturesV2(fomc_store=env.root) as features:
        return features.rows([("BTC-USD", t) for t in times])


def test_fomc_statement_at_first_feed_is_inventory_later_statement_is_new(fomc):
    T0 = fomc.clock.true
    first = fomc_rows(fomc, T0)[0]["sources"]["fomc"]
    assert first["state"] == "RESOLVED" and first["features"]["new_statements_attested_7d"] == 0
    assert first["inventory"]["inventory_observations_attested"] == 1
    fomc.feed([statement_item(), statement_item(P2, guid="g2")])
    fomc.provider.routes[P2] = fsyn.page_response(body="Synthetic later statement.", date_text="July 29, 2026")
    fomc.drive(600, idle=60)
    with SourceJoin("fomc", fomc.root) as reader:
        observations = v2.classify(reader)
    assert [o["kind"] for o in observations] == [v2.INITIAL_INVENTORY, v2.NEWLY_OBSERVED]
    at = observations[1]["available_at"]
    before, exact = fomc_rows(fomc, at - MICRO, at)
    assert before["sources"]["fomc"]["features"]["new_statements_attested_7d"] == 0
    f = exact["sources"]["fomc"]["features"]
    assert f["new_statements_attested_7d"] == f["new_statements_attested_30d"] == 1
    assert f["new_statement_attested_within_24h"] is True and f["hours_since_last_new_statement"] == 0


def test_fomc_revision_is_separate_and_manifest_is_inventory(fomc, tmp_path):
    fomc.provider.routes[P1] = fsyn.page_response(body="Synthetic corrected statement.")
    fomc.drive(1000, idle=60)
    with SourceJoin("fomc", fomc.root) as reader:
        observations = v2.classify(reader)
    assert [o["kind"] for o in observations] == [v2.INITIAL_INVENTORY, v2.REVISION]
    f = fomc_rows(fomc, fomc.clock.true)[0]["sources"]["fomc"]["features"]
    assert f["statement_revisions_attested_7d"] == 1 and f["new_statements_attested_7d"] == 0
    env = Env(tmp_path / "backfill")
    try:
        env.feed([])
        env.provider.routes[P1] = fsyn.page_response()
        env.collector.submit_manifest(json.dumps({"version": 1, "urls": [fsyn.url(P1)]}).encode(), "synthetic manifest")
        env.drive(360)
        with SourceJoin("fomc", env.root) as reader:
            assert KINDS(v2.classify(reader)) == [v2.INITIAL_INVENTORY]
        f = fomc_rows(env, env.clock.true)[0]["sources"]["fomc"]["features"]
        assert f["new_statements_attested_7d"] == 0
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()


def test_v1_outputs_and_identities_unchanged(edgar):
    artifact = json.loads((REPO / "docs/artifacts/event_features_matrix_v1.json").read_text())
    identity = artifact.pop("identity")
    assert identity.startswith("a5382f91") and sha256_canonical(artifact) == identity
    assert v1_features.POLICY == "ATTESTED_EVENT_FEATURES_V1" and v2.POLICY == "ATTESTED_EVENT_FEATURES_V2"
    with EventFeatures(edgar_store=edgar.root) as old:
        row = old.rows([("AAPL", edgar.T0)])[0]
    assert set(row["sources"]["edgar"]["features"]) == {
        "count_7d", "count_30d", "hours_since_last", "item_2_02", "item_5_02", "item_7_01", "item_8_01"}
    assert row["policy"] == "ATTESTED_EVENT_FEATURES_V1" and "protection" not in row
    assert set(edgar_features(aapl(edgar, edgar.T0)[0])) == set(v2.EDGAR_FEATURES)


def test_v2_is_deterministic_and_leaves_the_store_untouched(edgar):
    from scripts.trading_lab.event_features.matrix import fingerprint
    before = fingerprint(edgar.root)
    times = [edgar.T0, edgar.T0 + timedelta(days=1)]
    assert aapl(edgar, *times) == aapl(edgar, *times)
    assert fingerprint(edgar.root) == before
