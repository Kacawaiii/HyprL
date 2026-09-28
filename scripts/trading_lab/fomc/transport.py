"""MANUAL_REDIRECT_LOOP_V1 with per-hop validation, limiter grant and 60 s deadline (http_policy,
request_accounting). One logical fetch is at most 4 physical requests. No automatic redirect
following, no proxy, no conditional or range requests, final exact 200 only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import http.client
import socket
import ssl
import time
from typing import Callable
import urllib.parse
import zlib

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.fomc.identity import UrlRejected, admit_url
from scripts.trading_lab.fomc.limiter import Limiter


class HttpsConnector:
    """Production connector: direct TLS to the allowlisted host on 443 (no ambient proxy)."""

    def open(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPSConnection(spec.ALLOWED_HOST, 443, timeout=timeout, context=ssl.create_default_context())


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


class _Fail(Exception):
    def __init__(self, kind: str, reason: str):
        super().__init__(reason)
        self.kind, self.reason = kind, reason


class Transport:
    def __init__(self, connector, limiter: Limiter, *, wall: Callable[[], datetime], mono: Callable[[], float]):
        self.connector, self.limiter, self._wall, self._mono = connector, limiter, wall, mono
        self.physical_requests = 0

    def fetch(self, url: str, surface: str, *, invoke: Callable[[float], int],
              may_continue: Callable[[int], bool]) -> FetchResult:
        """`invoke(grant)` commits TRANSPORT_INVOKED (may raise store.Rejected: nothing is sent);
        `may_continue(attempt)` is false once the attempt already has an outcome."""
        admit_url(url)  # validation precedes the grant
        grant = self.limiter.grant()
        attempt = invoke(grant)
        result = FetchResult(kind="NOT_STARTED", attempt_seq=attempt)
        if self._mono() - grant > spec.GRANT_TO_TRANSPORT_S:
            result.kind, result.reason = "ABANDONED", "TRANSPORT_INVOKED later than 1 s after the grant"
            return result
        cap = spec.FEED_BODY_CAP if surface == "feed" else spec.STATEMENT_BODY_CAP
        current, followed = url, 0
        try:
            while True:
                hop = Hop(url=current, grant_mono=grant)
                result.hops.append(hop)
                self.physical_requests += 1
                assert len(result.hops) <= spec.MAX_PHYSICAL_PER_ATTEMPT
                response, conn = self._request(current)
                try:
                    hop.status = response.status
                    if response.status in spec.REDIRECT_STATUSES:
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
                    elif response.status != 200:
                        raise _Fail("SOURCE_UNAVAILABLE", f"final status {response.status}")
                    else:
                        return self._admit_200(result, response, current, cap)
                finally:
                    conn.close()  # redirect and error bodies are never drained
                followed += 1
                if not may_continue(attempt):
                    raise _Fail("SOURCE_UNAVAILABLE", "attempt already has an outcome; no continuation grant")
                grant = self.limiter.grant()
                current = target
        except _Fail as fail:
            result.kind, result.reason = fail.kind, fail.reason
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            result.kind, result.reason = "SOURCE_UNAVAILABLE", f"transport: {type(exc).__name__}"
        result.final_url = current
        return result

    def _request(self, url: str):
        parts = urllib.parse.urlsplit(url)
        conn = self.connector.open(spec.CONNECT_TIMEOUT_S)
        self._deadline = time.monotonic() + spec.ATTEMPT_DEADLINE_S
        conn.request("GET", parts.path, headers={
            "Host": spec.ALLOWED_HOST, "User-Agent": spec.USER_AGENT, "Accept-Encoding": "identity"})
        if conn.sock is not None:
            conn.sock.settimeout(spec.READ_TIMEOUT_S)
        return conn.getresponse(), conn

    def _admit_200(self, result: FetchResult, response, url: str, cap: int) -> FetchResult:
        coding = (response.msg.get("Content-Encoding") or "identity").strip().lower()
        if coding not in spec.ADMITTED_CONTENT_CODINGS:
            raise _Fail("SOURCE_UNAVAILABLE", f"content coding {coding!r} not admitted")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if coding == "gzip" else None
        out = bytearray()
        while True:
            if time.monotonic() > self._deadline:
                raise _Fail("SOURCE_UNAVAILABLE", "physical attempt deadline expired")
            chunk = response.read(65536)
            if not chunk:
                break
            pieces = [chunk] if decoder is None else [decoder.decompress(chunk, cap + 1 - len(out))]
            for piece in pieces:
                if len(out) + len(piece) > cap:
                    raise _Fail("PARSER_FAILED", "decoded body exceeds the frozen size bound")
                out.extend(piece)
            if decoder is not None and decoder.unconsumed_tail:
                raise _Fail("PARSER_FAILED", "decoded body exceeds the frozen size bound")
        if getattr(response, "length", None):  # Content-Length shortfall: http.client does not raise
            raise _Fail("SOURCE_UNAVAILABLE", "body truncated before Content-Length")
        if decoder is not None and not decoder.eof:
            raise _Fail("SOURCE_UNAVAILABLE", "content-coding stream truncated")
        result.kind, result.final_url, result.status = "RESPONSE_200", url, 200
        result.content_type_lines = list(response.msg.get_all("Content-Type") or [])
        result.content_encoding = response.msg.get("Content-Encoding")
        result.date_lines = list(response.msg.get_all("Date") or [])
        result.age_lines = list(response.msg.get_all("Age") or [])
        result.body = bytes(out)
        result.wall_at_receipt = self._wall()  # physical_attempt_deadline.ends_at (a)
        return result
