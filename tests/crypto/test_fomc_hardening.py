"""Regressions for read-time raw integrity, raw immutability and the 60 s physical-attempt deadline."""

from __future__ import annotations

from datetime import datetime
import json
import time

import pytest

from scripts.trading_lab.fomc import snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.limiter import Limiter
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt
from scripts.trading_lab.fomc.transport import Transport

from tests.crypto.fomc_support import P1, SID1, Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _raw_path(env, digest):
    return env.root / "raw" / digest[:2] / digest


def _item(snap):
    return next(i for i in snap["items"] if i["sid"] == SID1)


def _write_count(env):
    return env.store.horizon(), len(env.store.rows())


# ------------------------------------------------------------------ 1. read-time integrity -----------
def test_corrupt_revision_raw_fails_the_read_without_verify_integrity(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(240)
    old = snapshot.events_as_of(env.store, env.clock.true)
    digest = _item(old)["content_hash"]
    _raw_path(env, digest).write_bytes(b"tampered")
    before = _write_count(env)
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(env.store, env.clock.true)  # no verify_integrity() was called
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(env.store, datetime.fromisoformat(old["T"]), old["H"])  # the old (T, H) too
    with pytest.raises(snapshot.ReplayFailed):
        snapshot.replay(env.store, env.clock.true, env.store.horizon())
    assert _write_count(env) == before  # reads and replay write nothing


def test_missing_raw_fails_the_read(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(240)
    snap = snapshot.events_as_of(env.store, env.clock.true)
    _raw_path(env, _item(snap)["content_hash"]).unlink()
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(env.store, env.clock.true, snap["H"])


def test_corrupt_feed_raw_fails_a_read_exposing_its_cycle(env):
    env.feed([])
    env.drive(300)
    snap = snapshot.events_as_of(env.store, env.clock.true)
    assert snap["discovery"]["state"] == "EVENTS_OBSERVED_ZERO"
    feed = next(r for r in env.store.rows("RESPONSE") if r.seq == snap["discovery"]["cycle_id"])
    _raw_path(env, feed.body["raw_sha"]).write_bytes(b"<rss>tampered</rss>")
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(env.store, env.clock.true, snap["H"])


# ------------------------------------------------------------------ 2. raw immutability --------------
def test_put_raw_never_overwrites_corrupt_bytes(tmp_path):
    store = FomcStore(tmp_path / "s", wall_clock=lambda: syn.START)
    digest = store.put_raw(b"statement bytes")
    path = tmp_path / "s" / "raw" / digest[:2] / digest
    path.write_bytes(b"corrupted")
    with pytest.raises(RawCorrupt):
        store.put_raw(b"statement bytes")  # same bytes again: no repair
    assert path.read_bytes() == b"corrupted"


def test_corrupt_slot_surfaces_in_durable_processing_without_repair(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response()
    env.drive(240)
    live = state.primary_responses(env.store, SID1)[0]
    path = _raw_path(env, live.body["raw_sha"])
    path.write_bytes(b"corrupted")
    requests_before = len(env.provider.requests)
    env.collector.submit_manifest(json.dumps({"version": 1, "urls": [syn.url(P1)]}).encode(), "operator")
    env.drive(120)
    assert path.read_bytes() == b"corrupted"  # the same bytes fetched again did not repair the slot
    backfill = state.primary_responses(env.store, SID1)[-1]
    assert backfill.body["mode"] == "HISTORICAL_BACKFILL" and backfill.body["raw_sha"] == live.body["raw_sha"]
    assert state.processing_outcome(env.store, backfill.seq).body["outcome"] == "CORRUPTION_FAIL_CLOSED"
    assert state.processing_outcome(env.store, live.seq).body["outcome"] == "NORMALIZED_REVISION_COMMITTED"  # never rewritten
    assert env.store.rows("INTEGRITY_DIAGNOSTIC", key=str(live.seq))
    extra = [p for p in env.provider.requests[requests_before:] if p != syn.FEED_PATH]
    assert extra == [P1]  # only the scheduled backfill fetch; no network repair
    env.drive(130)
    assert _item(snapshot.events_as_of(env.store, env.clock.true))["step"] == 3  # explicit barrier


# ------------------------------------------------------------------ 3. physical-attempt deadline -----
class FakeIO:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeFile:
    def __init__(self, io, segments):
        self.io, self.segments, self.buf = io, list(segments), b""

    def _pull(self):
        if not self.segments:
            return False
        delay, data = self.segments.pop(0)
        self.io.advance(delay)
        self.buf += data
        return True

    def readline(self, limit=-1):
        while b"\n" not in self.buf and self._pull():
            pass
        cut = self.buf.find(b"\n") + 1 if b"\n" in self.buf else len(self.buf)
        line, self.buf = self.buf[:cut], self.buf[cut:]
        return line

    def read(self, n=-1):
        if not self.buf:
            self._pull()
        n = len(self.buf) if n is None or n < 0 else n
        data, self.buf = self.buf[:n], self.buf[n:]
        return data

    def close(self):
        pass


class FakeSocket:
    def __init__(self, io, segments):
        self.io, self.segments, self.sent = io, segments, b""

    def settimeout(self, value):
        assert value <= spec.ATTEMPT_DEADLINE_S  # no stage timeout outlives the remaining deadline

    def sendall(self, data):
        self.sent += data

    def makefile(self, mode):
        return FakeFile(self.io, self.segments)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class FakeConnector:
    """Scripted DNS/connect/TLS delays and per-hop response segments on a fake monotonic clock."""

    def __init__(self, io, hops, *, dns_s=0.0, connect_s=0.0, tls=False, tls_s=0.0):
        self.io, self.hops, self.dns_s, self.connect_s, self.tls, self.tls_s = io, list(hops), dns_s, connect_s, tls, tls_s
        self.connects = 0

    def resolve(self):
        self.io.advance(self.dns_s)
        return ["addr"]

    def connect(self, address, timeout):
        self.connects += 1
        self.io.advance(self.connect_s)
        return FakeSocket(self.io, self.hops.pop(0))

    def wrap(self, sock, timeout):
        self.io.advance(self.tls_s)
        return sock


def _headers(status=200, extra=b"", length=0):
    return b"HTTP/1.1 %d X\r\nContent-Type: text/html\r\nContent-Length: %d\r\n%s\r\n" % (status, length, extra)


def _transport(connector, io):
    clock = syn.SimClock(syn.START)
    return Transport(connector, Limiter(clock.mono, clock.sleep), wall=clock.wall, mono=clock.mono, io_clock=io)


def _fetch(transport):
    return transport.fetch(syn.url(P1), "primary", invoke=lambda grant: 1, may_continue=lambda attempt: True)


def test_late_dns_answer_never_starts_a_connection():
    io = FakeIO()
    connector = FakeConnector(io, [[(0, _headers())]], dns_s=61)
    result = _fetch(_transport(connector, io))
    assert result.kind == "SOURCE_UNAVAILABLE" and "DNS" in result.reason
    assert connector.connects == 0


def test_late_tls_handshake_sends_nothing():
    io = FakeIO()
    connector = FakeConnector(io, [[(0, _headers())]], connect_s=20, tls=True, tls_s=41)
    result = _fetch(_transport(connector, io))
    assert result.kind == "SOURCE_UNAVAILABLE" and "TLS" in result.reason


@pytest.mark.parametrize("segments", [
    [(61, _headers())],  # status line stalled past the deadline
    [(29, b"HTTP/1.1 200 X\r\n"), (29, b"Content-Type: text/html\r\n"), (29, b"Content-Length: 0\r\n\r\n")],  # drip
])
def test_blocked_or_dripped_headers_are_never_admitted(segments):
    io = FakeIO()
    result = _fetch(_transport(FakeConnector(io, [segments]), io))
    assert result.kind == "SOURCE_UNAVAILABLE" and "headers" in result.reason and result.body is None


def test_body_served_in_small_pieces_is_not_admitted_after_60_s():
    io = FakeIO()
    body = [(29, b"x" * 10) for _ in range(10)]
    result = _fetch(_transport(FakeConnector(io, [[(0, _headers(length=100))] + body]), io))
    assert result.kind == "SOURCE_UNAVAILABLE" and "body" in result.reason and result.body is None
    assert io.now < 120  # stopped at the first read past the deadline


def test_each_redirect_has_its_own_grant_and_deadline():
    io = FakeIO()
    hop1 = [(50, _headers(302, b"Location: /newsevents/pressreleases/monetary20260617b.htm\r\n"))]
    hop2 = [(55, _headers(length=2)), (0, b"ok")]
    result = _fetch(_transport(FakeConnector(io, [hop1, hop2]), io))
    assert result.kind == "RESPONSE_200" and len(result.hops) == 2  # 105 s in total, each hop < 60 s
    io2 = FakeIO()
    late_hop1 = [(61, _headers(302, b"Location: /newsevents/pressreleases/monetary20260617b.htm\r\n"))]
    connector = FakeConnector(io2, [late_hop1, hop2])
    result = _fetch(_transport(connector, io2))
    assert result.kind == "SOURCE_UNAVAILABLE" and connector.connects == 1  # the late redirect is never followed


def test_watchdog_cuts_a_real_blocked_socket(tmp_path):
    clock = syn.SimClock(syn.START)
    provider = syn.LocalProvider(clock)
    try:
        provider.routes[P1] = syn.SyntheticResponse(body=b"late", headers=syn.HTML_HEADERS, stall_s=3.0)
        transport = Transport(provider.connector(), Limiter(clock.mono, clock.sleep), wall=clock.wall, mono=clock.mono,
                              deadline_s=0.3)
        started = time.monotonic()
        result = _fetch(transport)
        assert result.kind == "SOURCE_UNAVAILABLE" and result.body is None
        assert time.monotonic() - started < 2.0  # aborted at the deadline, not after the server's stall
    finally:
        provider.close()
