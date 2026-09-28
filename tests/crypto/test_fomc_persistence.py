"""Checkpoint 1 - persistence: spec binding, identities, clock evidence, store and atomic ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from scripts.trading_lab.fomc import clock, identity, ledger, spec
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt, Rejected

T0 = datetime(2026, 9, 22, 18, 0, 5, tzinfo=timezone.utc)
STATEMENT = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm"


def _store(tmp_path):
    return FomcStore(tmp_path / "fomc", wall_clock=lambda: T0)


def test_spec_binding_is_revision_22():
    assert spec.verify_spec_binding() == spec.SPEC_HASH


@pytest.mark.parametrize("scheme", ["https", "HTTPS", "Https", "hTtPs", "HtTpS"])
def test_scheme_case_converges_fomc31(scheme):
    admitted = identity.admit_url(STATEMENT.replace("https", scheme, 1))
    assert admitted.canonical == STATEMENT
    assert identity.source_item_id(admitted.canonical) == identity.source_item_id(STATEMENT)


def test_explicit_443_and_absent_port_are_one_identity_fomc33():
    explicit = identity.admit_url(STATEMENT.replace(".gov/", ".gov:443/", 1))
    assert explicit.canonical == STATEMENT
    assert identity.admit_url(STATEMENT.replace(".gov/", ".gov:0443/", 1)).canonical == STATEMENT


@pytest.mark.parametrize("url", [
    STATEMENT.replace("https", "http", 1),  # FOMC32
    STATEMENT.replace(".gov/", ".gov:444/", 1),  # FOMC34
    STATEMENT.replace(".gov/", ".gov:/", 1),
    STATEMENT.replace(".gov/", ".gov:443abc/", 1),
    STATEMENT.replace(".gov/", ".gov:65536/", 1),
    STATEMENT.replace("a.htm", "a%.htm"),  # FOMC35
    "https://www.federalreserve.gov/path%2",  # FOMC36
    "https://www.federalreserve.gov/path%GG",  # FOMC37
    STATEMENT + "?x=1",
    "https://user@www.federalreserve.gov/x.htm",
    "https://federalreserve.gov/x.htm",
    " " + STATEMENT,
])
def test_inadmissible_urls_are_rejected_without_repair(url):
    with pytest.raises(identity.UrlRejected):
        identity.admit_url(url)


def test_valid_escape_passes_syntax_and_fragment_is_stripped_fomc38():
    assert identity.admit_url("https://www.federalreserve.gov/path%2Fsegment").path == "/path%2Fsegment"
    assert identity.admit_url(STATEMENT + "#frag").canonical == STATEMENT


def test_statement_family_path_grammar():
    assert identity.is_statement_family_path("/newsevents/pressreleases/monetary20260617a.htm")
    assert identity.is_statement_family_path("/newsevents/pressreleases/monetary20260617b1.htm")
    assert not identity.is_statement_family_path("/newsevents/pressreleases/monetary20260230a.htm")
    assert not identity.is_statement_family_path("/newsevents/pressreleases/other20260915a.htm")


def _date(dt):
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def test_clock_check_cases():
    assert clock.is_clock_verified(T0 - timedelta(seconds=3), [_date(T0)], [])  # FOMC163
    assert not clock.is_clock_verified(T0 + timedelta(seconds=280), [_date(T0)], [])  # FOMC166
    cached = T0 - timedelta(seconds=1200)
    assert clock.server_time([_date(cached)], ["1200"]) == T0  # FOMC167
    assert clock.is_clock_verified(T0, [_date(cached)], ["1200"])
    assert clock.server_time([_date(T0), _date(T0)], []) is None  # FOMC189
    assert clock.server_time([_date(T0)], []) == T0  # FOMC190
    assert clock.server_time([_date(T0)], ["90000"]) is None  # FOMC191
    assert clock.server_time([], []) is None  # FOMC193
    assert clock.server_time([_date(T0)], ["1", "1"]) is None
    assert clock.parse_http_date("Tue, 22 Sep 2026 18:00:60 GMT") is None  # leap second
    assert clock.parse_http_date("Mon, 22 Sep 2026 18:00:05 GMT") is None  # weekday mismatch
    assert clock.parse_http_date("Tuesday, 22-Sep-26 18:00:05 GMT") is None  # RFC 850


def test_store_is_wal_full_append_only_and_survives_reopen(tmp_path):
    store = _store(tmp_path)
    seqs = [store.append("X", [("NOTE", str(i), {"i": i})]) for i in range(3)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    store.close()
    reopened = _store(tmp_path)
    assert [row.body["i"] for row in reopened.rows("NOTE")] == [0, 1, 2]
    conn = sqlite3.connect(tmp_path / "fomc" / "fomc.sqlite3")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_raw_is_content_addressed_and_corruption_fails_closed(tmp_path):
    store = _store(tmp_path)
    digest = store.put_raw(b"<rss/>")
    assert store.read_raw(digest) == b"<rss/>"
    path = tmp_path / "fomc" / "raw" / digest[:2] / digest
    path.write_bytes(b"<rss>tampered</rss>")
    with pytest.raises(RawCorrupt):
        store.read_raw(digest)


def _episode(store, key="E1", sid="S1"):
    assert ledger.open_episode(store, key, {"kind": "LIVE_ACQUISITION", "sid": sid}) is not None
    return {"kind": "LIVE_ACQUISITION", "episode_key": key, "sid": sid}


def test_transport_invoked_budget_is_six_and_needs_an_outcome_between_attempts(tmp_path):
    store = _store(tmp_path)
    ledger.begin_epoch(store, "e1", "boot")
    work = _episode(store)
    for n in range(6):
        attempt = ledger.transport_invoked(store, epoch="e1", work=work, grant_mono=float(n))
        with pytest.raises(Rejected):  # one attempt of the item in flight
            ledger.transport_invoked(store, epoch="e1", work=work, grant_mono=float(n))
        ledger.commit_attempt_outcome(store, attempt, "SOURCE_UNAVAILABLE")
    with pytest.raises(Rejected):
        ledger.transport_invoked(store, epoch="e1", work=work, grant_mono=9.0)
    assert len(ledger.attempts_of_episode(store, "E1")) == 6
    assert ledger.open_episode(store, "E1", {"kind": "LIVE_ACQUISITION"}) is None  # one episode per key


def test_old_epoch_is_fenced(tmp_path):
    store = _store(tmp_path)
    ledger.begin_epoch(store, "old", "boot1")
    work = _episode(store)
    ledger.begin_epoch(store, "new", "boot2")
    with pytest.raises(Rejected):
        ledger.transport_invoked(store, epoch="old", work=work, grant_mono=0.0)


def test_concurrent_writers_never_exceed_six_attempts(tmp_path):
    first = _store(tmp_path)
    ledger.begin_epoch(first, "e", "boot")
    work = _episode(first)
    handles = [first] + [_store(tmp_path) for _ in range(3)]

    def worker(store):
        committed = 0
        for _ in range(10):
            try:
                attempt = ledger.transport_invoked(store, epoch="e", work=work, grant_mono=0.0)
            except Rejected:
                continue
            committed += 1
            ledger.commit_attempt_outcome(store, attempt, "SOURCE_UNAVAILABLE")
        return committed

    with ThreadPoolExecutor(4) as pool:
        total = sum(pool.map(worker, handles))
    assert total == 6 == len(ledger.attempts_of_episode(first, "E1"))


def test_one_outcome_per_attempt_and_late_evidence(tmp_path):
    store = _store(tmp_path)
    ledger.begin_epoch(store, "e", "boot")
    work = _episode(store)
    attempt = ledger.transport_invoked(store, epoch="e", work=work, grant_mono=0.0)
    assert ledger.commit_attempt_outcome(store, attempt, "INTERRUPTED") is not None
    seq, late = ledger.commit_response(store, attempt, {"surface": "primary"})
    assert late and store.rows("RESPONSE")[0].body["late_evidence"] is True
    second = ledger.transport_invoked(store, epoch="e", work=work, grant_mono=1.0)
    seq, late = ledger.commit_response(store, second, {"surface": "primary"})
    assert not late
    assert ledger.commit_attempt_outcome(store, second, "INTERRUPTED") is None  # outcome is final


def test_crash_after_transport_invoked_counts_the_attempt(tmp_path):
    store = _store(tmp_path)
    ledger.begin_epoch(store, "e1", "boot1")
    work = _episode(store)
    attempt = ledger.transport_invoked(store, epoch="e1", work=work, grant_mono=0.0)
    store.close()  # crash before transport: no outcome recorded
    owner = _store(tmp_path)
    ledger.begin_epoch(owner, "e2", "boot2")
    assert [r.seq for r in ledger.attempts_without_outcome(owner)] == [attempt]
    assert ledger.commit_attempt_outcome(owner, attempt, "INTERRUPTED") is not None
    assert len(ledger.attempts_of_episode(owner, "E1")) == 1  # counted, never refunded
