"""MANUAL_REDIRECT_LOOP_V1 with per-hop validation, limiter grant and the 60 s physical-attempt
deadline (http_policy.physical_attempt_deadline, request_accounting). One logical fetch is at most
4 physical requests. No automatic redirect following, no proxy, no conditional or range requests,
final exact 200 only.

The deadline starts at each hop's FIX15 grant, on the transport's monotonic I/O clock, and covers
DNS, TCP, TLS, request write, status/headers, body and content decoding. It is enforced three ways:
every stage gets at most the remaining time as its timeout; a watchdog aborts the socket at the
deadline; and every stage result is checked against the deadline before it is used, so a late DNS
answer never starts a connection and a late body is never admitted. A redirect's next hop gets its
own grant and its own fresh deadline; nothing extends a previous hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import http.client
import socket
import ssl
import threading
import time
from typing import Callable
import urllib.parse
import zlib

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.fomc.identity import UrlRejected, admit_url
from scripts.trading_lab.fomc.limiter import Limiter


class HttpsConnector:
    """Production connector: one resolver lookup, sequential addresses, direct TLS on 443, no proxy."""

    tls = True

    def __init__(self, cafile: str | None = None):
        self.cafile = cafile  # None: the system trust store (production); a file: tests with a local CA

    def resolve(self):
        return socket.getaddrinfo(spec.ALLOWED_HOST, 443, type=socket.SOCK_STREAM)

    def connect(self, address, timeout: float):
        family, kind, proto, _canon, sockaddr = address
        sock = socket.socket(family, kind, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
        except BaseException:
            sock.close()
            raise
        return sock

    def context(self) -> ssl.SSLContext:
        """Certificate chain and hostname verification required, TLS >= 1.2 (Python defaults)."""
        return ssl.create_default_context(cafile=self.cafile)

    def wrap(self, sock, timeout: float):
        sock.settimeout(timeout)
        return self.context().wrap_socket(sock, server_hostname=spec.ALLOWED_HOST)  # SNI and name check


@dataclass
class Hop:
    url: str
    grant_mono: float
    status: int | None = None


@dataclass
class FetchResult:
    kind: str  # RESPONSE_200 | SOURCE_UNAVAILABLE | PARSER_FAILED | NOT_STARTED | ABANDONED
    reason: str = ""
    attempt_seq: int | None = None
    hops: list[Hop] = field(default_factory=list)
    final_url: str | None = None
    status: int | None = None
    content_type_lines: list[str] = field(default_factory=list)
    content_encoding: str | None = None
    date_lines: list[str] = field(default_factory=list)
    age_lines: list[str] = field(default_factory=list)
    body: bytes | None = None
    wall_at_receipt: datetime | None = None
    network_end_mono: float | None = None  # set by the collector: start of the 120 s save deadline


class _Fail(Exception):
    def __init__(self, kind: str, reason: str):
        super().__init__(reason)
        self.kind, self.reason = kind, reason


class _HeldConnection(http.client.HTTPConnection):
    """The transport owns the socket's lifetime so its per-stage timeouts stay settable to the end."""

    def close(self):
        pass


class Deadline:
    """PHYSICAL_ATTEMPT_TOTAL_DEADLINE_V1 for one hop: begins at its grant, never reset or extended."""

    def __init__(self, clock: Callable[[], float], seconds: float):
        self.clock = clock
        self.end = clock() + seconds

    def remaining(self) -> float:
        return self.end - self.clock()

    def check(self, stage: str) -> None:
        if self.remaining() <= 0:
            raise _Fail("SOURCE_UNAVAILABLE", f"physical attempt deadline expired during {stage}")

    def timeout(self, stage_limit: float, stage: str) -> float:
        self.check(stage)
        return min(stage_limit, self.remaining())


class Transport:
    def __init__(self, connector, limiter: Limiter, *, wall: Callable[[], datetime], mono: Callable[[], float],
                 io_clock: Callable[[], float] = time.monotonic, deadline_s: float = spec.ATTEMPT_DEADLINE_S):
        self.connector, self.limiter, self._wall, self._mono = connector, limiter, wall, mono
        self._io, self._deadline_s = io_clock, deadline_s
        self._count_lock = threading.Lock()
        self.physical_requests = 0

    def deadline(self) -> "Deadline":
        """The physical-attempt deadline of a start granted now."""
        return Deadline(self._io, self._deadline_s)

    def fetch(self, url: str, surface: str, *, invoke: Callable[[float], int],
              may_continue: Callable[[int], bool], started: tuple | None = None,
              continuation: Callable[[int], tuple | None] | None = None) -> FetchResult:
        """`invoke(grant)` commits TRANSPORT_INVOKED (may raise store.Rejected: nothing is sent);
        `may_continue(attempt)` is false once the attempt already has an outcome.

        Step-driven use waits for its own grants. A dispatcher passes `started` = (grant, deadline) for
        a start it already granted and `continuation(attempt)`, which blocks until the dispatcher grants
        the validated redirect hop and returns (grant, deadline), or None when no grant will come."""
        admit_url(url)  # validation precedes the grant
        if started is None:
            grant = self.limiter.grant()
            deadline = Deadline(self._io, self._deadline_s)  # starts at the grant
        else:
            grant, deadline = started
        attempt = invoke(grant)
        result = FetchResult(kind="NOT_STARTED", attempt_seq=attempt)
        if self._mono() - grant > spec.GRANT_TO_TRANSPORT_S:
            result.kind, result.reason = "ABANDONED", "TRANSPORT_INVOKED later than 1 s after the grant"
            return result
        cap = spec.FEED_BODY_CAP if surface == "feed" else spec.STATEMENT_BODY_CAP
        current = url
        try:
            while True:
                hop = Hop(url=current, grant_mono=grant)
                result.hops.append(hop)
                with self._count_lock:
                    self.physical_requests += 1
                assert len(result.hops) <= spec.MAX_PHYSICAL_PER_ATTEMPT
                target = self._hop(result, hop, current, deadline, cap)
                if target is None:
                    return result
                if not may_continue(attempt):
                    raise _Fail("SOURCE_UNAVAILABLE", "attempt already has an outcome; no continuation grant")
                if continuation is None:
                    grant = self.limiter.grant()
                    deadline = Deadline(self._io, self._deadline_s)  # the next hop's own deadline
                else:
                    granted = continuation(attempt)  # the validated hop waits for the dispatcher's grant
                    if granted is None:
                        raise _Fail("SOURCE_UNAVAILABLE", "attempt already has an outcome; no continuation grant")
                    grant, deadline = granted
                current = target
        except _Fail as fail:
            result.kind, result.reason = fail.kind, fail.reason
        result.final_url = current
        return result

    # ---- one physical request under its deadline ---------------------------------------------------
    def _resolve(self, deadline: Deadline):
        box: dict = {}

        def run():
            try:
                box["value"] = self.connector.resolve()
            except BaseException as exc:  # noqa: BLE001 - reported below
                box["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(max(deadline.remaining(), 0.0))
        if worker.is_alive():
            raise _Fail("SOURCE_UNAVAILABLE", "physical attempt deadline expired during DNS")  # late answer discarded
        deadline.check("DNS")  # an answer that arrived after the deadline never starts a connection
        if "error" in box:
            raise _Fail("SOURCE_UNAVAILABLE", f"DNS resolution failed: {type(box['error']).__name__}")
        return box["value"]

    def _connect(self, deadline: Deadline):
        addresses = self._resolve(deadline)
        last = None
        for address in addresses:  # sequentially, in resolver order: no racing connections
            timeout = deadline.timeout(spec.CONNECT_TIMEOUT_S, "connect")
            try:
                sock = self.connector.connect(address, timeout)
            except OSError as exc:
                last = exc
                continue
            try:
                deadline.check("connect")
                if getattr(self.connector, "tls", False):
                    sock = self.connector.wrap(sock, deadline.timeout(spec.READ_TIMEOUT_S, "TLS"))
                    deadline.check("TLS")
            except OSError as exc:  # handshake, certificate or hostname verification failure, TLS stall
                sock.close()
                deadline.check("TLS")
                raise _Fail("SOURCE_UNAVAILABLE", f"TLS: {type(exc).__name__}: {getattr(exc, 'verify_message', None) or exc}") from exc
            except BaseException:
                sock.close()
                raise
            return sock
        deadline.check("connect")
        raise _Fail("SOURCE_UNAVAILABLE", f"TCP connection failed: {type(last).__name__ if last else 'no address'}")

    def _hop(self, result: FetchResult, hop: Hop, current: str, deadline: Deadline, cap: int) -> str | None:
        """Run one physical request. Returns the next target of an allowed redirect, or None when the
        final 200 was admitted; raises _Fail otherwise."""
        sock = self._connect(deadline)
        watchdog = threading.Timer(max(deadline.remaining(), 0.0), _abort, args=(sock,))
        watchdog.daemon = True
        watchdog.start()
        try:
            conn = _HeldConnection(spec.ALLOWED_HOST)
            conn.sock = sock
            try:
                sock.settimeout(deadline.timeout(spec.READ_TIMEOUT_S, "request write"))
                conn.request("GET", urllib.parse.urlsplit(current).path, headers={
                    "Host": spec.ALLOWED_HOST, "User-Agent": spec.USER_AGENT, "Accept-Encoding": "identity"})
                deadline.check("request write")
                sock.settimeout(deadline.timeout(spec.READ_TIMEOUT_S, "headers"))
                response = conn.getresponse()
                deadline.check("headers")
                hop.status = response.status
                if response.status in spec.REDIRECT_STATUSES:
                    return self._redirect_target(response, current, len(result.hops) - 1)
                if response.status != 200:
                    raise _Fail("SOURCE_UNAVAILABLE", f"final status {response.status}")
                self._admit_200(result, response, sock, current, cap, deadline)
                return None
            except (OSError, http.client.HTTPException) as exc:
                deadline.check("transfer")  # an abort by the watchdog reports the deadline
                raise _Fail("SOURCE_UNAVAILABLE", f"transport: {type(exc).__name__}") from exc
        finally:
            watchdog.cancel()
            _abort(sock)  # redirect and error bodies are never drained

    @staticmethod
    def _redirect_target(response, current: str, followed: int) -> str:
        locations = response.msg.get_all("Location") or []
        if len(locations) != 1 or not locations[0].strip(" \t"):
            raise _Fail("SOURCE_UNAVAILABLE", "redirect Location missing or malformed")
        target = urllib.parse.urljoin(current, locations[0].strip(" \t"))
        try:
            admit_url(target)
        except UrlRejected as exc:
            raise _Fail("SOURCE_UNAVAILABLE", f"redirect target rejected: {exc}") from exc
        if followed == spec.MAX_REDIRECTS:
            raise _Fail("SOURCE_UNAVAILABLE", "redirect transition limit exceeded")
        return target

    def _admit_200(self, result: FetchResult, response, sock, url: str, cap: int, deadline: Deadline) -> None:
        coding = (response.msg.get("Content-Encoding") or "identity").strip().lower()
        if coding not in spec.ADMITTED_CONTENT_CODINGS:
            raise _Fail("SOURCE_UNAVAILABLE", f"content coding {coding!r} not admitted")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if coding == "gzip" else None
        out = bytearray()
        while True:
            sock.settimeout(deadline.timeout(spec.READ_TIMEOUT_S, "body"))
            chunk = response.read(65536)
            deadline.check("body")
            if not chunk:
                break
            piece = chunk if decoder is None else decoder.decompress(chunk, cap + 1 - len(out))
            deadline.check("content decoding")
            if len(out) + len(piece) > cap or (decoder is not None and decoder.unconsumed_tail):
                raise _Fail("PARSER_FAILED", "decoded body exceeds the frozen size bound")
            out.extend(piece)
        if getattr(response, "length", None):  # Content-Length shortfall: http.client does not raise
            raise _Fail("SOURCE_UNAVAILABLE", "body truncated before Content-Length")
        if decoder is not None and not decoder.eof:
            raise _Fail("SOURCE_UNAVAILABLE", "content-coding stream truncated")
        deadline.check("admission")  # success only if the terminal event is before the deadline
        result.kind, result.final_url, result.status = "RESPONSE_200", url, 200
        result.content_type_lines = list(response.msg.get_all("Content-Type") or [])
        result.content_encoding = response.msg.get("Content-Encoding")
        result.date_lines = list(response.msg.get_all("Date") or [])
        result.age_lines = list(response.msg.get_all("Age") or [])
        result.body = bytes(out)
        result.wall_at_receipt = self._wall()  # physical_attempt_deadline.ends_at (a)


def _abort(sock) -> None:
    """Watchdog: at the deadline the attempt is aborted; any blocked read or write fails."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
