"""The fetch of one submissions listing (spec: request_accounting). The production fetcher speaks HTTPS to
data.sec.gov only, with the operator's declared User-Agent, never follows a redirect and maps every
failure to an outcome; it is not exercised against the SEC by this slice (no capture is authorized)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import http.client
import io
import re
import ssl
import time
from typing import Callable
from urllib.parse import urlsplit

from scripts.trading_lab.edgar import spec

_PATH = re.compile(r"^/submissions/CIK[0-9]{10}\.json$")


@dataclass
class FetchResult:
    kind: str  # RESPONSE, SOURCE_UNAVAILABLE, SOURCE_THROTTLED, SOURCE_NOT_FOUND
    status: int | None = None
    headers: list[tuple[str, str]] | None = None
    body: bytes | None = None
    wall_at_receipt: datetime | None = None
    reason: str | None = None

    def header_lines(self, name: str) -> list[str]:
        return [v for k, v in (self.headers or []) if k.lower() == name.lower()]


def classify_status(status: int) -> tuple[str, str | None]:
    if status == 200:
        return "RESPONSE", None
    if status == 404:
        return "SOURCE_NOT_FOUND", "404: no submissions for this CIK"
    if status in (403, 429):
        return "SOURCE_THROTTLED", f"{status}: treated as a fair-access limit (UV4)"
    if 300 <= status < 400:
        return "SOURCE_UNAVAILABLE", f"{status}: redirects are never followed"
    return "SOURCE_UNAVAILABLE", f"unexpected status {status}"


def decode_body(raw: bytes, coding_lines: list[str]) -> tuple[bytes | None, str | None]:
    coding = coding_lines[0].strip().lower() if coding_lines else "identity"
    if len(coding_lines) > 1 or coding not in spec.CONTENT_CODINGS:
        return None, f"content coding {coding_lines!r} is not admitted"
    if coding == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
                body = handle.read(spec.BODY_CAP + 1)
        except (OSError, EOFError) as exc:
            return None, f"gzip body does not decode: {exc}"
        raw = body
    if len(raw) > spec.BODY_CAP:
        return None, f"body above {spec.BODY_CAP} bytes"
    return raw, None


class HttpsFetcher:
    def __init__(self, user_agent: str, *, connection_factory: Callable = http.client.HTTPSConnection,
                 wall: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 mono: Callable[[], float] = time.monotonic):
        if not isinstance(user_agent, str) or not spec.USER_AGENT.match(user_agent.strip()):
            raise ValueError("the SEC asks automated clients to declare a User-Agent naming an organization and a "
                             "contact e-mail (e.g. 'Example Lab ops@example.org'); refusing to start without one")
        self.user_agent = user_agent.strip()
        self._factory, self._wall, self._mono = connection_factory, wall, mono
        self._context = ssl.create_default_context()  # system trust store, hostname checking, CERT_REQUIRED

    @staticmethod
    def _bound(conn, seconds: float) -> None:
        sock = getattr(conn, "sock", None)
        if sock is not None:
            sock.settimeout(seconds)

    def fetch(self, url: str, *, started: float | None = None) -> FetchResult:
        """One listing under one deadline of DEADLINE_S counted from `started` (the grant, on this fetcher's
        monotonic clock) to the decoded body; without `started`, from now."""
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != spec.SUBMISSIONS_HOST or parts.port not in (None, 443) \
                or parts.query or parts.fragment or not _PATH.match(parts.path):
            return FetchResult("SOURCE_UNAVAILABLE", reason=f"refused URL outside the submissions surface: {url}")
        deadline = (self._mono() if started is None else started) + spec.DEADLINE_S  # grant to decoded body

        def remaining() -> float:
            left = deadline - self._mono()
            if left <= 0:
                raise TimeoutError(f"the {spec.DEADLINE_S} s deadline of the attempt passed")
            return left

        try:
            conn = self._factory(spec.SUBMISSIONS_HOST, timeout=remaining(), context=self._context)
            try:
                conn.request("GET", parts.path, headers={"User-Agent": self.user_agent, "Accept": spec.JSON_MEDIA,
                                                         "Accept-Encoding": "gzip"})
                self._bound(conn, remaining())
                response = conn.getresponse()
                wall = self._wall()
                headers = list(response.getheaders())
                kind, reason = classify_status(response.status)
                if kind != "RESPONSE":
                    return FetchResult(kind, response.status, headers, None, wall, reason)
                chunks, size = [], 0
                while size <= spec.BODY_CAP:
                    self._bound(conn, remaining())
                    chunk = response.read(min(65536, spec.BODY_CAP + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                remaining()
                raw = b"".join(chunks)
            finally:
                conn.close()
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:  # TimeoutError is an OSError
            return FetchResult("SOURCE_UNAVAILABLE", reason=f"{type(exc).__name__}: {exc}")
        result = FetchResult("RESPONSE", 200, headers, None, wall)
        body, problem = decode_body(raw, result.header_lines("Content-Encoding"))
        if problem:
            return FetchResult("SOURCE_UNAVAILABLE", 200, headers, None, wall, problem)
        if self._mono() > deadline:  # decoding is inside the deadline too
            return FetchResult("SOURCE_UNAVAILABLE", 200, headers, None, wall,
                               f"TimeoutError: the {spec.DEADLINE_S} s deadline of the attempt passed while decoding")
        result.body = body
        return result
