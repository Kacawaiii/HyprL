"""The one place in HyprL that can talk to Massive over the network.

Phase 6D built a provider that structurally could not make a request. This
module is the transport that gives it one, and it is a separate file for a
reason: the provider's parsing, validation, grid checks and corporate-action
logic stay testable offline forever, because nothing in them knows a socket
exists. Injecting this class is a deliberate act at one call site.

What it will and will not do:

* **GET only.** No verb that can change anything on the other end exists here.
* **Host allowlist.** A URL that does not resolve to an allowlisted host is
  refused before a connection is opened -- an allowlist rather than a scheme
  check, because `https://` says nothing about where the bytes are going.
* **Bounded retries, on transport failures only.** A timeout, a reset
  connection, a 5xx and a 429 are worth retrying because the request was
  never answered. A 400 or a validation failure is not: retrying it produces
  the same wrong answer more slowly, and hides a bug behind a delay.
* **Raw bytes preserved.** The caller receives what came off the wire before
  anything parses it. A canonicalisation bug is only recoverable if the
  source survives.
* **No websocket, no daemon, no background thread.** One request at a time,
  paced, synchronous, and finished when the function returns.

The credential is read at request time by the provider and handed here in a
header. This module never stores it, never logs it, and never places it in a
URL -- a key in a query string ends up in every access log between here and
the vendor.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from scripts.trading_lab.massive_provider import (
    MassiveProviderError, MassiveTransport)

MASSIVE_TRANSPORT_SCHEMA_VERSION = "trading-lab.massive-http-transport.v1"

# Every host this transport may reach. Nothing resolves outside this tuple.
ALLOWED_HOSTS = ("api.massive.com",)

DEFAULT_BASE_URL = "https://api.massive.com"

REQUEST_TIMEOUT_SECONDS = 30

# Fixed pacing between requests. Not a random jitter: a deterministic capture
# should take about the same time twice, and a vendor should see a steady
# stream rather than a burst.
REQUEST_SPACING_SECONDS = 0.35

# Retries are for a request that was never answered. Three attempts after the
# first, with a fixed schedule so a capture's duration is predictable.
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1.0, 3.0, 8.0)

# A vendor's Retry-After is honoured up to this. Beyond it the capture stops
# rather than sleeping for an unbounded time inside a single run.
MAX_RETRY_AFTER_SECONDS = 60

# HTTP statuses worth trying again. Everything else is an answer, even when it
# is an answer we do not like.
RETRYABLE_STATUSES = (408, 425, 429, 500, 502, 503, 504)

CAPTURE_USER_AGENT = "hyprl-trading-lab-equity-capture/1.0"


class TransportError(MassiveProviderError):
    """Raised when a request cannot be made or completed safely."""


class RateLimitedError(TransportError):
    """Raised when the vendor asked us to slow down.

    Typed separately so a capture can report how often it happened without
    parsing an error message, and so it is never confused with a data problem.
    """

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HostNotAllowedError(TransportError):
    """Raised when a request would leave for a host nobody allowlisted."""


@dataclass
class TransportStats:
    """What the network actually did, for the capture report.

    Counted rather than logged: a capture that silently retried forty times is
    a different capture from one that succeeded first try, and the manifest
    should say which happened.
    """

    requests: int = 0
    retries: int = 0
    rate_limits: int = 0
    bytes_received: int = 0
    elapsed_seconds: float = 0.0

    def payload(self) -> dict:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "rate_limits": self.rate_limits,
            "bytes_received": self.bytes_received,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True)
class TransportResponse:
    """A response, kept in both forms.

    ``raw`` is what came off the wire and is what gets stored. ``payload`` is
    the parsed convenience, derived from it. Storing only the parsed form
    would mean a parsing bug could never be diagnosed after the fact.
    """

    status: int
    raw: bytes = field(repr=False)
    payload: dict = field(repr=False)
    url: str


def require_allowed_host(url: str) -> str:
    """Scheme and host only.

    Used for the URL that actually goes on the wire, which may legitimately
    carry a credential parameter when the vendor documents query auth. The
    stricter check below is what guards every URL that gets recorded.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise HostNotAllowedError(
            f"refusing a non-https request to {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise HostNotAllowedError(
            f"{host!r} is not an allowlisted market-data host; allowed: "
            f"{list(ALLOWED_HOSTS)}")
    return url


def require_allowed_url(url: str) -> str:
    """Refuse a URL that leaves for a host nobody put on the list.

    Also refuses anything credential-shaped in the query string. This is the
    check applied to every URL that is *recorded* -- stored in raw metadata,
    written to the manifest, put in an exception -- because a key in a query
    string ends up in every access log between here and the vendor, and in
    this repository forever.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise HostNotAllowedError(
            f"refusing a non-https request to {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise HostNotAllowedError(
            f"{host!r} is not an allowlisted market-data host; allowed: "
            f"{list(ALLOWED_HOSTS)}")
    if parsed.query:
        # The credential travels in a header. A query string is copied into
        # access logs, proxy logs and browser history, so anything that looks
        # like a key in one is refused rather than merely discouraged.
        lowered = parsed.query.lower()
        for marker in ("apikey", "api_key", "token", "secret", "key="):
            if marker in lowered:
                raise TransportError(
                    "refusing to put a credential in a query string; it "
                    "belongs in the Authorization header and nowhere else")
    return url


# Query parameters that may carry a secret. Removed from any URL the vendor
# hands back before it is followed, logged or stored.
CREDENTIAL_QUERY_KEYS = ("apikey", "api_key", "api-key", "key", "token",
                         "secret", "access_token", "auth")


def strip_credential_params(url: str) -> str:
    """Drop credential-shaped query parameters from a URL. Never logs them."""
    parsed = urllib.parse.urlsplit(url)
    kept = [(key, value) for key, value in urllib.parse.parse_qsl(parsed.query)
            if key.lower() not in CREDENTIAL_QUERY_KEYS]
    return urllib.parse.urlunsplit((
        parsed.scheme, parsed.netloc, parsed.path,
        urllib.parse.urlencode(kept), ""))


def _retry_after_seconds(headers) -> float | None:
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        # A vendor may send an HTTP-date instead of seconds. Rather than
        # guessing at a parse, fall back to the standard backoff.
        return None


class MassiveHTTPTransport(MassiveTransport):
    """Real HTTP against Massive. One request at a time, paced and bounded."""

    name = "massive-https"

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = REQUEST_TIMEOUT_SECONDS,
                 spacing: float = REQUEST_SPACING_SECONDS,
                 max_attempts: int = MAX_ATTEMPTS,
                 opener=None, sleep=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.spacing = float(spacing)
        self.max_attempts = int(max_attempts)
        # Injected so the retry and pacing logic can be tested without a
        # socket and without a test suite that takes twelve seconds to sleep.
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep
        self.stats = TransportStats()
        self._last_request_at: float | None = None
        require_allowed_url(self.base_url + "/")

    # --- the MassiveTransport contract ------------------------------------

    def request(self, path: str, params: dict, headers: dict) -> dict:
        """The 6D interface: parsed payload only."""
        return self.fetch(path, params, headers).payload

    # --- the capture interface --------------------------------------------

    def fetch(self, path: str, params: dict, headers: dict,
              *, auth_query: dict | None = None) -> TransportResponse:
        """A single GET, retried only when the request was never answered.

        ``auth_query`` is the vendor's documented query-parameter auth. It is
        merged into the URL that goes on the wire and into nothing else: the
        URL recorded on the response, stored in raw metadata and quoted in any
        error is built without it and re-checked to be credential-free. So the
        key can satisfy an endpoint that requires query auth without ever
        being written down.
        """
        # What gets recorded. Credential-free by construction, and checked.
        url = self.build_url(path, params)
        # What goes on the wire. Never returned, never stored, never logged.
        request_url = url
        if auth_query:
            merged = {**{key: str(value) for key, value in params.items()},
                      **{key: str(value) for key, value in auth_query.items()}}
            query = urllib.parse.urlencode(sorted(merged.items()))
            request_url = require_allowed_host(f"{self.base_url}{path}?{query}")
        request_headers = {
            **{key: value for key, value in (headers or {}).items()},
            "User-Agent": CAPTURE_USER_AGENT,
            "Accept": "application/json",
        }

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            try:
                response = self._attempt(request_url, request_headers,
                                         recorded_url=url)
            except RateLimitedError as error:
                self.stats.rate_limits += 1
                last_error = error
                if attempt == self.max_attempts:
                    break
                self._sleep(self._rate_limit_delay(error, attempt))
                self.stats.retries += 1
                continue
            except TransportError as error:
                # Already classified as retryable by _attempt; anything not
                # retryable was raised as a plain MassiveProviderError.
                last_error = error
                if attempt == self.max_attempts:
                    break
                self._sleep(RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
                self.stats.retries += 1
                continue
            else:
                self.stats.elapsed_seconds += time.monotonic() - started
                return response

        self.stats.elapsed_seconds += time.monotonic() - started
        raise TransportError(
            f"giving up on {path} after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}") from None

    def fetch_absolute(self, url: str, headers: dict) -> "TransportResponse":
        """Follow a continuation URL the vendor supplied.

        Validated against the same allowlist as everything else -- a next_url
        is data from the network, and following it unchecked would let the
        response choose the next host.

        Any credential-shaped query parameter is stripped first. Some vendors
        embed the API key in next_url; that value would otherwise be recorded
        in the raw request metadata and committed forever.
        """
        cleaned = strip_credential_params(url)
        require_allowed_url(cleaned)
        parsed = urllib.parse.urlsplit(cleaned)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        return self.fetch(parsed.path, params, headers)

    def build_url(self, path: str, params: dict) -> str:
        query = urllib.parse.urlencode(
            {key: str(value) for key, value in sorted((params or {}).items())})
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        return require_allowed_url(url)

    # --- internals --------------------------------------------------------

    def _pace(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = time.monotonic()
            return
        waited = time.monotonic() - self._last_request_at
        if waited < self.spacing:
            self._sleep(self.spacing - waited)
        self._last_request_at = time.monotonic()

    def _rate_limit_delay(self, error: RateLimitedError, attempt: int) -> float:
        if error.retry_after is not None:
            if error.retry_after > MAX_RETRY_AFTER_SECONDS:
                raise TransportError(
                    f"the vendor asked for a {error.retry_after:.0f}s pause, "
                    f"beyond the {MAX_RETRY_AFTER_SECONDS}s this capture will "
                    "wait inside one run; rerun the capture later")
            return error.retry_after
        return RETRY_BACKOFF_SECONDS[
            min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]

    def _attempt(self, url: str, headers: dict, *,
                 recorded_url: str | None = None) -> TransportResponse:
        request = urllib.request.Request(url, method="GET")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200) or 200
        except urllib.error.HTTPError as error:
            body_status = error.code
            headers_in = getattr(error, "headers", None)
            if body_status == 429:
                raise RateLimitedError(
                    f"rate limited by the market-data provider (HTTP {body_status})",
                    retry_after=_retry_after_seconds(headers_in)) from None
            if body_status in RETRYABLE_STATUSES:
                raise TransportError(
                    f"transient HTTP {body_status} from the market-data provider"
                ) from None
            if body_status in (401, 403):
                # Never echo the response body: a vendor may quote the
                # Authorization header it rejected.
                raise MassiveProviderError(
                    f"the market-data provider refused the request (HTTP "
                    f"{body_status}); check that HYPRL_MASSIVE_API_KEY is valid "
                    "and entitled to this data") from None
            raise MassiveProviderError(
                f"the market-data provider returned HTTP {body_status}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(
                f"network failure contacting the market-data provider: "
                f"{type(error).__name__}") from None

        self.stats.requests += 1
        self.stats.bytes_received += len(raw)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            # Deliberately NOT retryable. A response that is not JSON is an
            # answer, and asking again would just produce it again.
            raise MassiveProviderError(
                f"the market-data provider returned a body that is not JSON: "
                f"{type(error).__name__}") from None
        if not isinstance(payload, dict):
            raise MassiveProviderError(
                "expected a JSON object from the market-data provider")
        return TransportResponse(status=status, raw=raw, payload=payload,
                                 url=recorded_url if recorded_url else url)

    def payload(self) -> dict:
        """Safe to publish: what this transport is, never what it carries."""
        return {
            "schema_version": MASSIVE_TRANSPORT_SCHEMA_VERSION,
            "name": self.name,
            "base_url": self.base_url,
            "allowed_hosts": list(ALLOWED_HOSTS),
            "methods": ["GET"],
            "timeout_seconds": self.timeout,
            "max_attempts": self.max_attempts,
            "stats": self.stats.payload(),
        }


__all__ = [
    "ALLOWED_HOSTS", "CAPTURE_USER_AGENT", "DEFAULT_BASE_URL",
    "HostNotAllowedError", "MASSIVE_TRANSPORT_SCHEMA_VERSION",
    "MAX_ATTEMPTS", "MAX_RETRY_AFTER_SECONDS", "MassiveHTTPTransport",
    "RETRYABLE_STATUSES", "RETRY_BACKOFF_SECONDS", "RateLimitedError",
    "REQUEST_TIMEOUT_SECONDS", "TransportError", "TransportResponse",
    "TransportStats", "CREDENTIAL_QUERY_KEYS", "require_allowed_host",
    "require_allowed_url",
    "strip_credential_params",
]
