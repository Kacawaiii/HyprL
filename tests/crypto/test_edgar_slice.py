"""SEC EDGAR V1 offline slice (docs/artifacts/edgar_capture_spec_v1.json, cases EDGAR01-EDGAR16): accession
identity, revisions and corrections, amendments, absences inside the documented window, strict listing
shape, acceptanceDateTime as provenance only, causal reads, raw integrity, replay, the store opening rule,
request outcomes and the fetcher's declared User-Agent. Synthetic provider only: no request to the SEC."""

from __future__ import annotations

from datetime import timedelta
import gzip
import json
import shutil
import sqlite3

import pytest

from scripts.trading_lab.edgar import spec, snapshot
from scripts.trading_lab.edgar import synthetic as syn
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.listing import filing_identity, source_item_id
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.edgar.transport import HttpsFetcher
from scripts.trading_lab.sources.store import Rejected, StoreRejected

A = syn.CIK_A.zfill(10)
ACC1, ACC2, ACC_OLD, ACC_AMEND = "0000320193-26-000071", "0000320193-26-000072", "0000320193-26-000040", "0000320193-26-000073"
OLD = syn.filing(ACC_OLD, filed="2026-05-01")


class Env:
    def __init__(self, tmp_path):
        self.root = tmp_path / "edgar"
        self.clock = syn.SimClock()
        self.fetch = syn.FakeFetcher(self.clock)
        self.store = EdgarStore(self.root, wall_clock=self.clock.wall)
        self.collector = EdgarCollector(self.store, self.fetch, self.clock)
        self.collector.submit_watchlist([syn.CIK_A])

    def serve(self, *filings, **kw):
        self.fetch.routes[A] = syn.Reply(syn.listing(syn.CIK_A, list(filings)), **kw)

    def poll(self, gap: float = 600):
        result = self.collector.poll(syn.CIK_A)
        self.clock.sleep(gap)
        return result

    def read(self, offset: float = 0):
        return snapshot.filings_as_of(self.store, self.clock.true + timedelta(seconds=offset))

    def settle(self):
        """Two more verified polls, then a read 93 s after the first of them: everything recorded before it is
        available (V3: its response attests it), and the read is resolved (the next transaction is
        attested by the second response). A read at "now" is never resolved: the newest records always
        wait for later evidence."""
        self.poll()
        self.poll(gap=10)
        attesting = self.store.rows("RESPONSE")[-2].body["observed_at"]
        return snapshot.filings_as_of(self.store, snapshot.parse_iso(attesting) + timedelta(seconds=93))


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.collector.close()


def _filing(snap, accession):
    return next(f for f in snap["filings"] if f["accession_number"] == accession)


def test_edgar01_a_new_filing_is_available_only_after_later_server_evidence(env):
    env.serve(syn.filing(ACC1))
    first = env.poll(gap=200)
    assert first["outcome"] == "LISTING_CLASSIFIED" and first["detail"]["new_revisions"] == 1
    unresolved = env.read()
    assert unresolved["read_state"] == "EDGAR_CAUSAL_VISIBILITY_UNRESOLVED" and "filings" not in unresolved
    snap = env.settle()
    item = _filing(snap, ACC1)
    assert snap["read_state"] == "EDGAR_RESOLVED" and item["state"] == "PRESENT" and item["revisions_seen"] == 1
    assert item["source_item_id"] == source_item_id(ACC1) and item["fields"]["form"] == "8-K"
    later = env.store.rows("RESPONSE")[-2].body["observed_at"]  # the response that attests it
    assert item["first_available_at"] == snapshot.iso(snapshot.parse_iso(later) + spec.CLOCK_ERROR_BOUND)  # V3, not acceptance
    assert item["provenance"]["filing_index_url"].endswith(f"/320193/{ACC1.replace('-', '')}/{ACC1}-index.htm")


def test_edgar02_the_same_listing_again_adds_observations_not_revisions(env):
    env.serve(syn.filing(ACC1))
    for _ in range(3):
        env.poll()
    snap = env.settle()
    assert len(env.store.rows("FILING_REVISION")) == 1 and _filing(snap, ACC1)["observations"] == 3
    assert len(env.store.rows("FILING_OBSERVATION")) == 5  # the two attesting polls observed it too, after T


def test_edgar03_corrected_metadata_is_a_new_revision_and_old_reads_keep_the_old_one(env):
    env.serve(syn.filing(ACC1, items="2.02,9.01"))
    env.poll()
    before = env.settle()
    env.serve(syn.filing(ACC1, items="2.02,7.01,9.01"))  # a post-acceptance correction (VF10)
    env.poll()
    after = env.settle()
    old, new = _filing(before, ACC1), _filing(after, ACC1)
    assert old["fields"]["items"] == "2.02,9.01" and new["fields"]["items"] == "2.02,7.01,9.01"
    assert new["revisions_seen"] == 2 and new["revision"] != old["revision"] and len(env.store.rows("FILING_REVISION")) == 2
    again = snapshot.filings_as_of(env.store, snapshot.parse_iso(before["T"]), before["H"])
    assert again == before  # the earlier read is unchanged


def test_edgar04_an_amendment_is_its_own_filing_with_no_inferred_link(env):
    env.serve(syn.filing(ACC1), syn.filing(ACC_AMEND, form="8-K/A", filed="2026-06-17"))
    env.poll()
    snap = env.settle()
    amendment, original = _filing(snap, ACC_AMEND), _filing(snap, ACC1)
    assert amendment["amends"] is None and amendment["amendment_link"] == "NOT_PROVIDED_BY_SOURCE"
    assert original["amendment_link"] is None and amendment["source_item_id"] != original["source_item_id"]


def test_edgar05_06_07_absence_inside_the_window_only_and_reappearance(env):
    env.serve(syn.filing(ACC1), syn.filing(ACC2, filed="2026-04-02"), OLD)
    env.poll()
    present = env.settle()
    env.serve(OLD)  # ACC1 (06-16 > 05-01) inside the window; ACC2 (04-02 <= 05-01) outside it
    env.poll()
    gone = env.settle()
    assert _filing(gone, ACC1)["state"] == "ABSENT_FROM_LISTING"
    assert _filing(gone, ACC2)["state"] == "PRESENT"  # EDGAR06: nothing inferred outside the window
    assert {r.body["accession_number"] for r in env.store.rows("FILING_ABSENCE")} == {ACC1}
    assert _filing(present, ACC1)["state"] == "PRESENT"  # the earlier read is unchanged
    assert len(env.store.rows("FILING_OBSERVATION")) >= 3  # nothing deleted
    env.serve(syn.filing(ACC1), OLD)
    env.poll()
    back = env.settle()
    assert _filing(back, ACC1)["state"] == "PRESENT" and _filing(back, ACC1)["revisions_seen"] == 1  # EDGAR07


@pytest.mark.parametrize("body, content_type, reason", [
    (b"{not json", "application/json", "not a UTF-8 JSON document"),
    (json.dumps({"cik": "320193", "filings": {"recent": {"accessionNumber": [ACC1], "form": ["8-K"],
                 "filingDate": ["2026-06-16"], "acceptanceDateTime": ["x"]}}}).encode(), "application/json", "no primaryDocument column"),
    (json.dumps({"cik": "320193", "filings": {"recent": {"accessionNumber": [ACC1, ACC2], "form": ["8-K"],
                 "filingDate": ["2026-06-16"], "acceptanceDateTime": ["x"], "primaryDocument": ["d"]}}}).encode(),
     "application/json", "unequal lengths"),
    (syn.listing("789019", [syn.filing(ACC1)]), "application/json", "not 0000320193"),
    (syn.listing(syn.CIK_A, [syn.filing("320193-26-71")]), "application/json", "invalid accession"),
    (syn.listing(syn.CIK_A, [syn.filing(ACC1), syn.filing(ACC1)]), "application/json", "listed twice"),
    (syn.listing(syn.CIK_A, [syn.filing(ACC1)]), "text/html", "Content-Type"),
])
def test_edgar08_a_listing_of_another_shape_derives_nothing(env, body, content_type, reason):
    env.serve(syn.filing(ACC1), OLD)
    env.poll()
    env.fetch.routes[A] = syn.Reply(body, content_type=content_type)
    result = env.poll()
    assert result["outcome"] == "PARSER_FAILED" and reason in result["detail"]["reason"]
    record = result["record"]
    assert not [r for k in ("FILING_REVISION", "FILING_OBSERVATION", "FILING_ABSENCE") for r in env.store.rows(k) if r.seq > record]
    env.serve(syn.filing(ACC1), OLD)
    snap = env.settle()
    assert _filing(snap, ACC1)["state"] == "PRESENT"  # no absence from a refused listing
    failed = [h for h in env.store.rows("SOURCE_HEALTH") if h.body["record"] == record]
    assert failed[0].body["result_state"] == "PARSER_FAILED"


@pytest.mark.parametrize("acceptance", ["1999-01-01T00:00:00.000Z", "2031-12-31T23:59:59.000Z", "16:31 ET", ""])
def test_edgar09_acceptance_time_is_provenance_text_only(tmp_path, acceptance):
    plain, odd = Env(tmp_path / "plain"), Env(tmp_path / "odd")
    try:
        for env, value in ((plain, "2026-06-16T16:31:05.000Z"), (odd, acceptance)):
            env.serve(syn.filing(ACC1, acceptance=value))
            env.poll()
        first, second = plain.settle(), odd.settle()
        a, b = _filing(first, ACC1), _filing(second, ACC1)
        assert b["provenance"]["acceptance_datetime_text"] == acceptance == b["fields"]["acceptance_datetime_text"]
        assert a["first_available_at"] == b["first_available_at"] and first["P"] == second["P"]  # E5
    finally:
        plain.collector.close()
        odd.collector.close()


def test_edgar10_other_forms_are_counted_not_normalized(env):
    env.serve(syn.filing(ACC1), syn.filing(ACC2, form="10-K"), syn.filing(ACC_OLD, form="4"))
    result = env.poll()
    assert result["detail"]["out_of_scope_forms"] == {"10-K": 1, "4": 1} and result["detail"]["in_scope"] == 1
    assert {r.body["accession_number"] for r in env.store.rows("FILING_REVISION")} == {ACC1}


def test_edgar11_an_unverified_clock_never_makes_earlier_records_available(env):
    env.serve(syn.filing(ACC1))
    processed = env.poll()["record"] + 1  # the processing transaction of the first listing

    def entry():
        return next(e for e in snapshot.availability(env.store, env.store.horizon()) if e.seq == processed)
    env.serve(syn.filing(ACC1), date=None)  # no Date header (UV5)
    env.poll()
    env.serve(syn.filing(ACC1), date="Tue, 16 Jun 2026 09:00:00 GMT")  # skewed by hours
    env.poll()
    assert [r.body["verdict"] for r in env.store.rows("RESPONSE")] == ["CLOCK_VERIFIED", "CLOCK_UNVERIFIED", "CLOCK_UNVERIFIED"]
    assert entry().resolved is False  # unverified responses attest nothing
    env.serve(syn.filing(ACC1))
    env.poll()
    attesting = env.store.rows("RESPONSE")[-1].body["observed_at"]
    assert entry().resolved and entry().avail == snapshot.parse_iso(attesting) + spec.CLOCK_ERROR_BOUND
    snap = env.settle()
    assert _filing(snap, ACC1)["state"] == "PRESENT"


def test_edgar12_a_corrupt_raw_fails_the_read_and_the_replay_closed(env):
    env.serve(syn.filing(ACC1))
    env.poll()
    snap = env.settle()
    digest = _filing(snap, ACC1)["provenance"]["first_raw_sha256"]
    path = env.root / "raw" / digest[:2] / digest
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.filings_as_of(env.store, snapshot.parse_iso(snap["T"]), snap["H"])
    with pytest.raises(snapshot.ReplayFailed):
        snapshot.replay(env.store, snapshot.parse_iso(snap["T"]), snap["H"])


def _fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_edgar13_the_store_opening_rule_and_read_only_opening(env, tmp_path):
    env.serve(syn.filing(ACC1))
    env.poll()
    env.collector.close()
    env.store.close()
    with sqlite3.connect(env.root / "edgar.sqlite3") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    before = _fingerprint(env.root)
    reader = EdgarStore(env.root, wall_clock=None, read_only=True)
    assert len(reader.rows("FILING_REVISION")) == 1
    with pytest.raises(PermissionError):
        reader.append("X", [])
    reader.close()
    assert _fingerprint(env.root) == before
    for name, value in (("schema_version", "edgar-store-v0"), ("spec_hash", "0" * 64)):
        copy = tmp_path / name
        shutil.copytree(env.root, copy)
        with sqlite3.connect(copy / "edgar.sqlite3") as conn:
            conn.execute("UPDATE meta SET value = ? WHERE name = ?", (value, name))
        frozen = _fingerprint(copy)
        with pytest.raises(StoreRejected):
            EdgarStore(copy, wall_clock=None)
        assert _fingerprint(copy) == frozen
    env.store = EdgarStore(env.root, wall_clock=env.clock.wall)  # the fixture closes a live collector
    env.collector = EdgarCollector(env.store, env.fetch, env.clock, boot_id="boot-2")


def test_edgar14_request_outcomes_and_the_throttle_pause(env):
    for status, kind in ((404, "SOURCE_NOT_FOUND"), (301, "SOURCE_UNAVAILABLE"), (500, "SOURCE_UNAVAILABLE")):
        env.fetch.routes[A] = syn.Reply(b"", status=status)
        assert env.poll()["status"] == kind
    env.fetch.routes[A] = None
    assert env.poll()["status"] == "SOURCE_UNAVAILABLE"
    env.fetch.routes[A] = syn.Reply(b"", status=429)
    assert env.poll(gap=60)["status"] == "SOURCE_THROTTLED"
    sent = len(env.fetch.requests)
    assert env.poll(gap=60)["status"] == "PAUSED" and len(env.fetch.requests) == sent  # no request during the pause
    env.clock.sleep(spec.THROTTLE_PAUSE_S)
    env.serve(syn.filing(ACC1))
    assert env.poll()["status"] == "RESPONSE"
    states = [h.body["result_state"] for h in env.store.rows("SOURCE_HEALTH")]
    assert states == ["SOURCE_NOT_FOUND", "SOURCE_UNAVAILABLE", "SOURCE_UNAVAILABLE", "SOURCE_UNAVAILABLE",
                      "SOURCE_THROTTLED", None]


def test_edgar15_reopen_and_replay_give_the_same_identity_and_tampering_fails(env, tmp_path):
    env.serve(syn.filing(ACC1), OLD)
    env.poll()
    env.serve(syn.filing(ACC1, items="2.02"), OLD)
    env.poll()
    env.serve(OLD)
    env.poll()
    snap = env.settle()
    reopened = EdgarStore(env.root, wall_clock=None, read_only=True)
    T = snapshot.parse_iso(snap["T"])
    assert snapshot.filings_as_of(reopened, T, snap["H"]) == snap
    assert snapshot.replay(reopened, T, snap["H"]) == snap
    for name, sql in (("observation", "UPDATE rec SET body = json_set(body, '$.position', 9) WHERE kind = 'FILING_OBSERVATION'"),
                      ("health", "UPDATE rec SET body = json_set(body, '$.reason', 'X') WHERE kind = 'SOURCE_HEALTH'"),
                      ("verdict", "UPDATE rec SET body = json_set(body, '$.verdict', 'CLOCK_UNVERIFIED') WHERE kind = 'RESPONSE'")):
        copy = tmp_path / name
        shutil.copytree(env.root, copy, ignore=shutil.ignore_patterns("owner.lock"))
        with sqlite3.connect(copy / "edgar.sqlite3") as conn:
            assert conn.execute(sql).rowcount >= 1
        with pytest.raises(snapshot.ReplayFailed):
            snapshot.replay(EdgarStore(copy, wall_clock=None, read_only=True), T, snap["H"])


def test_watchlist_ownership_and_spec_binding(env):
    assert spec.verify_spec_binding() == spec.SPEC_HASH
    assert env.collector.submit_watchlist([syn.CIK_A]) == env.store.rows("MANIFEST")[0].seq  # given once
    with pytest.raises(Rejected):
        env.collector.submit_watchlist([syn.CIK_B])
    with pytest.raises(Rejected):
        env.collector.poll(syn.CIK_B)  # never a CIK outside the watchlist
    with pytest.raises(Rejected):
        EdgarCollector(env.store, env.fetch, env.clock)  # a single owner
    with pytest.raises(ValueError):
        env.collector.submit_watchlist(["12a"])
    assert filing_identity({"a": 1}) != filing_identity({"a": 2})


class _Response:
    def __init__(self, status, headers, body, on_read=None):
        self.status, self._headers, self._body, self._on_read = status, headers, body, on_read

    def getheaders(self):
        return self._headers

    def read(self, limit):
        if self._on_read:
            self._on_read()
        chunk, self._body = self._body[:limit], self._body[limit:]
        return chunk


class _Connection:
    calls: list = []

    def __init__(self, host, timeout, context):
        self.host, self.timeout = host, timeout
        _Connection.calls.append(self)

    def request(self, method, path, headers):
        self.sent = (method, path, headers)

    def getresponse(self):
        return _Connection.reply

    def close(self):
        pass


def test_edgar16_the_fetcher_declares_its_user_agent_and_never_follows_a_redirect():
    for bad in ("", "python-requests/2.31", "Example Lab", "ops@example.org"):
        with pytest.raises(ValueError):
            HttpsFetcher(bad, connection_factory=_Connection)
    fetcher = HttpsFetcher("Example Lab ops@example.org", connection_factory=_Connection)
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    _Connection.calls.clear()
    _Connection.reply = _Response(200, [("Content-Type", "application/json"), ("Content-Encoding", "gzip")],
                                  gzip.compress(b'{"cik": "320193"}'))
    result = fetcher.fetch(url)
    method, path, headers = _Connection.calls[-1].sent
    assert (method, path, headers["User-Agent"]) == ("GET", "/submissions/CIK0000320193.json", "Example Lab ops@example.org")
    assert _Connection.calls[-1].host == "data.sec.gov" and 0 < _Connection.calls[-1].timeout <= spec.DEADLINE_S
    assert result.kind == "RESPONSE" and result.body == b'{"cik": "320193"}'
    _Connection.reply = _Response(301, [("Location", "https://evil.example/")], b"")
    redirected = fetcher.fetch(url)
    assert redirected.kind == "SOURCE_UNAVAILABLE" and "never followed" in redirected.reason and len(_Connection.calls) == 2
    for outside in ("http://data.sec.gov/submissions/CIK0000320193.json", "https://www.sec.gov/cgi-bin/browse-edgar",
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"):
        assert fetcher.fetch(outside).kind == "SOURCE_UNAVAILABLE"
    assert len(_Connection.calls) == 2  # refused before any connection
    _Connection.reply = _Response(200, [("Content-Type", "application/json")], b"x" * (spec.BODY_CAP + 1))
    assert "above" in fetcher.fetch(url).reason


def test_one_deadline_bounds_the_whole_fetch():
    now = [0.0]
    fetcher = HttpsFetcher("Example Lab ops@example.org", connection_factory=_Connection, mono=lambda: now[0])
    _Connection.reply = _Response(200, [("Content-Type", "application/json")], b"x" * 300_000,
                                  on_read=lambda: now.__setitem__(0, now[0] + 8))  # a slow body: 8 s per chunk
    result = fetcher.fetch("https://data.sec.gov/submissions/CIK0000320193.json")
    assert result.kind == "SOURCE_UNAVAILABLE" and "deadline" in result.reason and now[0] < spec.DEADLINE_S + 8


def test_a_restart_interrupts_open_attempts_and_processes_saved_records(tmp_path):
    env = Env(tmp_path)
    env.serve(syn.filing(ACC1))
    env.poll()
    # a crash between the attempt record and its outcome, and one between a record and its processing
    env.store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", None, {"epoch": env.collector.epoch, "cik": A,
                                                                          "url": "u", "grant_mono": 0.0})])
    digest = env.store.put_raw(syn.listing(syn.CIK_A, [syn.filing(ACC1), syn.filing(ACC2, filed="2026-06-17")]))
    attempt = env.store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", None, {"epoch": env.collector.epoch, "cik": A,
                                                                                    "url": "u", "grant_mono": 1.0})])
    wall = env.clock.wall()
    record = env.store.append("RESPONSE", [
        ("RESPONSE", str(attempt), {"attempt": attempt, "cik": A, "url": "u", "status": 200,
                                    "content_type_lines": ["application/json"], "content_encoding_lines": [],
                                    "date_lines": [], "age_lines": [], "wall_at_receipt": snapshot.iso(wall),
                                    "verdict": "CLOCK_UNVERIFIED", "observed_at": None, "raw_sha": digest,
                                    "byte_length": 1, "mode": "LIVE", "late_evidence": False}),
        ("ATTEMPT_OUTCOME", str(attempt), {"attempt": attempt, "outcome": "RESPONSE", "status": 200, "reason": None})])
    env.collector.close()
    env.collector = EdgarCollector(env.store, env.fetch, env.clock, boot_id="boot-2")
    assert env.collector.reconciled == {"interrupted": 1, "processed": 1}
    outcomes = [o.body["outcome"] for o in env.store.rows("ATTEMPT_OUTCOME")]
    assert outcomes.count("INTERRUPTED") == 1
    assert env.store.rows("PROCESSING_OUTCOME", key=str(record))[0].body["detail"]["new_revisions"] == 1  # ACC2
    env.serve(syn.filing(ACC1), syn.filing(ACC2, filed="2026-06-17"))
    env.poll()
    snap = env.settle()
    assert {f["accession_number"] for f in snap["filings"]} == {ACC1, ACC2}
    assert snapshot.replay(env.store, snapshot.parse_iso(snap["T"]), snap["H"]) == snap  # health of INTERRUPTED included
    env.collector.close()


# ------------------------------------------------------------------ the capture runner's gate -----
from scripts.trading_lab.edgar import service  # noqa: E402


def _authorization(tmp_path, **changes):
    auth = {"authorizes": spec.PROVIDER_ID, "spec_hash": spec.SPEC_HASH, "ciks": [syn.CIK_A], "max_requests": 3,
            "not_after": "2026-06-18T00:00:00+00:00", "user_agent": "Example Lab ops@example.org",
            "granted_by": "operator", "granted_at": "2026-06-17"}
    auth.update(changes)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({k: v for k, v in auth.items() if v is not None}))
    return path


@pytest.mark.parametrize("changes", [
    {"authorizes": "federal_reserve_fomc_statements_v1"}, {"spec_hash": "0" * 64}, {"ciks": []},
    {"ciks": [str(i) for i in range(11)]}, {"ciks": ["12a"]}, {"max_requests": 0}, {"max_requests": 51},
    {"max_requests": True}, {"not_after": "2026-06-17T12:00:00+00:00"}, {"not_after": "tomorrow"},
    {"user_agent": "python-requests/2.31"}, {"granted_by": None},
])
def test_the_runner_refuses_without_a_valid_authorization_and_sends_nothing(tmp_path, changes):
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    with pytest.raises(service.CaptureRefused):
        service.run(tmp_path / "store", _authorization(tmp_path, **changes), fetcher=fetcher, clock=clock, log=lambda m: None)
    assert fetcher.requests == [] and not (tmp_path / "store").exists()


def test_the_runner_rejects_an_authorization_expiry_without_an_offset(tmp_path):
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    with pytest.raises(service.CaptureRefused, match="offset"):
        service.run(tmp_path / "store", _authorization(tmp_path, not_after="2026-06-18T00:00:00"),
                    fetcher=fetcher, clock=clock, log=lambda m: None)
    assert fetcher.requests == [] and not (tmp_path / "store").exists()


def test_check_writes_nothing_and_the_run_stops_at_its_budget(tmp_path):
    path = _authorization(tmp_path)
    store_dir = tmp_path / "store"
    report = service.check(store_dir, path, now=syn.START)
    assert report["max_requests"] == 3 and report["ciks"] == [A] and not store_dir.exists()
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[A] = syn.Reply(syn.listing(syn.CIK_A, [syn.filing(ACC1)]))
    summary = service.run(store_dir, path, fetcher=fetcher, clock=clock, log=lambda m: None)
    assert summary["reason"] == "request budget spent" and summary["requests"] == 3 == len(fetcher.requests)
    assert summary["revisions"] == 1
    expiring = _authorization(tmp_path, max_requests=50, not_after="2026-06-17T13:30:00+00:00")
    clock2, fetcher2 = syn.SimClock(), syn.FakeFetcher(syn.SimClock())
    fetcher2.clock = clock2
    fetcher2.routes[A] = syn.Reply(syn.listing(syn.CIK_A, [syn.filing(ACC1)]))
    ended = service.run(tmp_path / "store2", expiring, fetcher=fetcher2, clock=clock2, log=lambda m: None)
    assert ended["reason"] == "authorization expired" and ended["requests"] == 3  # 13:01, 13:11, 13:21; not 13:31


def test_the_deadline_counts_from_the_grant_and_headers_are_kept(tmp_path):
    now = [100.0]
    fetcher = HttpsFetcher("Example Lab ops@example.org", connection_factory=_Connection, mono=lambda: now[0])
    _Connection.calls.clear()
    late = fetcher.fetch("https://data.sec.gov/submissions/CIK0000320193.json", started=now[0] - spec.DEADLINE_S - 1)
    assert late.kind == "SOURCE_UNAVAILABLE" and "deadline" in late.reason and _Connection.calls == []  # nothing sent
    env = Env(tmp_path)
    env.serve(syn.filing(ACC1), extra=[("Server", "synthetic"), ("Cache-Control", "no-cache")])
    env.poll()
    resp = env.store.rows("RESPONSE")[-1].body
    assert ["Server", "synthetic"] in resp["header_lines"] and resp["fetch_seconds"] >= 0.4
    env.collector.close()


def test_a_throttled_answer_stops_the_trial(tmp_path):
    clock = syn.SimClock()
    fetcher = syn.FakeFetcher(clock)
    fetcher.routes[A] = syn.Reply(b"", status=429)
    summary = service.run(tmp_path / "store", _authorization(tmp_path, max_requests=8), fetcher=fetcher, clock=clock,
                          log=lambda m: None)
    assert summary["reason"].startswith("throttled") and summary["requests"] == 1 == len(fetcher.requests)


def test_the_qualification_matrix_separates_observation_from_unknown(env):
    from scripts.trading_lab.edgar.qualify import qualify
    env.serve(syn.filing(ACC1), syn.filing(ACC_AMEND, form="8-K/A", filed="2026-06-17"), syn.filing(ACC2, form="10-K"))
    env.poll()
    env.poll()
    result = qualify(EdgarStore(env.root, wall_clock=None, read_only=True))
    matrix = result["matrix"]
    assert matrix["UV1"]["verdict"] == "OBSERVED_COMPATIBLE" and matrix["UV1"]["observation"]["columns_not_in_spec"] == []
    assert matrix["UV2"]["verdict"] == "FORMAT_OBSERVED_SEMANTICS_UNKNOWN"
    assert matrix["UV2"]["observation"]["shapes"] == {"YYYY-MM-DDTHH:MM:SS.000Z": 6}
    assert matrix["UV3"]["verdict"] == matrix["UV4"]["verdict"] == "UNKNOWN_PERSISTS"  # never provoked
    assert matrix["UV5"]["verdict"] == "OBSERVED_PRESENT_AND_WITHIN_TOLERANCE"
    assert matrix["UV6"]["verdict"] == "NO_LINK_COLUMN_OBSERVED_UNIVERSALITY_UNKNOWN"
    assert matrix["UV6"]["observation"]["amendment_rows"] == 2 and result["listings"] == 2
