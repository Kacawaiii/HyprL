"""The one place this project reaches the Yahoo chart endpoint.

Deliberately small. The provider's parsing, validation, grid checks and
corporate-action logic stay testable offline forever because none of them
knows a socket exists; injecting this class is a separate, deliberate act.

Credential-free by construction. There is no key to read, so there is no
header to build, no cookie, no crumb and no session. A capture can run in an
environment holding no secret at all -- which is the entire reason this
source exists alongside the blocked Massive one.

The redirect and allowlist policy is the shared primitive, not a second copy.
Yahoo carries no credential, so a hostile redirect cannot leak one, but it
could still make bytes fetched from an unvalidated host be recorded as having
come from Yahoo. Provenance is the thing being protected here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from scripts.trading_lab.safe_http import (
    AllowlistedRedirectHandler, build_hardened_opener, require_https_host,
    strip_credential_params)
from scripts.trading_lab.yahoo_chart_provider import (
    YahooProviderError, YahooTransport)

YAHOO_TRANSPORT_SCHEMA_VERSION = "trading-lab.yahoo-http-transport.v1"

# Every host this transport may reach. Nothing resolves outside this tuple.
ALLOWED_HOSTS = ("query1.finance.yahoo.com",)

DEFAULT_BASE_URL = "https://query1.finance.yahoo.com"

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_SPACING_SECONDS = 0.5
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1.0, 3.0, 8.0)
MAX_RETRY_AFTER_SECONDS = 60

RETRYABLE_STATUSES = (408, 425, 429, 500, 502, 503, 504)

CAPTURE_USER_AGENT = "hyprl-trading-lab-equity-research/1.0"


class YahooTransportError(YahooProviderError):
    """Raised when a request cannot be made or completed safely."""


class YahooHostNotAllowedError(YahooTransportError):
    """Raised when a request would leave for a host nobody allowlisted."""


class YahooRateLimitedError(YahooTransportError):
    """Raised when the source asked us to slow down.

    Typed so a capture can count it without parsing an error message, and so
    it is never mistaken for a data problem.
    """

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def require_allowed_host(url: str) -> str:
    """Scheme, host, port and authority, against the Yahoo allowlist."""
    return require_https_host(url, ALLOWED_HOSTS,
                              error_class=YahooHostNotAllowedError)


@dataclass
class YahooTransportStats:
    requests: int = 0
    retries: int = 0
    rate_limits: int = 0
    redirects: int = 0
    bytes_received: int = 0
    elapsed_seconds: float = 0.0

    def payload(self) -> dict:
        return {
            "requests": self.requests, "retries": self.retries,
            "rate_limits": self.rate_limits, "redirects": self.redirects,
            "bytes_received": self.bytes_received,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True)
class YahooResponse:
    """A response, kept in both forms.

    ``raw`` is what came off the wire and is what gets stored. ``payload`` is
    derived from it. Storing only the parsed form would mean a parsing bug
    could never be diagnosed after the fact.
    """

    status: int
    raw: bytes = field(repr=False)
    payload: dict = field(repr=False)
    url: str


def _retry_after_seconds(headers) -> float | None:
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


class YahooHTTPTransport(YahooTransport):
    """Real HTTPS against the Yahoo chart endpoint. Paced, bounded, GET only."""

    name = "yahoo-https"

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = REQUEST_TIMEOUT_SECONDS,
                 spacing: float = REQUEST_SPACING_SECONDS,
                 max_attempts: int = MAX_ATTEMPTS,
                 opener=None, sleep=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.spacing = float(spacing)
        self.max_attempts = int(max_attempts)
        # The production default is the hardened opener, never bare urlopen:
        # bare urlopen follows a redirect to any host.
        self._opener = opener or build_hardened_opener(
            ALLOWED_HOSTS, error_class=YahooHostNotAllowedError).open
        self._sleep = sleep or time.sleep
        self.stats = YahooTransportStats()
        self._last_request_at: float | None = None
        require_allowed_host(self.base_url + "/")

    # --- the YahooTransport contract --------------------------------------

    def request(self, path: str, params: dict, headers: dict) -> dict:
        return self.fetch(path, params, headers).payload

    def fetch(self, path: str, params: dict, headers: dict) -> YahooResponse:
        """A single GET, retried only when the request was never answered."""
        url = self.build_url(path, params)
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
                response = self._attempt(url, request_headers)
            except YahooHostNotAllowedError:
                # A destination outside the allowlist is not transient.
                # Retrying asks the same forbidden question three more times
                # and then reports a timeout-shaped error, burying the refusal.
                raise
            except YahooRateLimitedError as error:
                self.stats.rate_limits += 1
                last_error = error
                if attempt == self.max_attempts:
                    break
                self._sleep(self._rate_limit_delay(error, attempt))
                self.stats.retries += 1
                continue
            except YahooTransportError as error:
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
        raise YahooTransportError(
            f"giving up on {path} after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}") from None

    def build_url(self, path: str, params: dict) -> str:
        query = urllib.parse.urlencode(
            {key: str(value) for key, value in sorted((params or {}).items())})
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        return require_allowed_host(url)

    # --- internals --------------------------------------------------------

    def _pace(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = time.monotonic()
            return
        waited = time.monotonic() - self._last_request_at
        if waited < self.spacing:
            self._sleep(self.spacing - waited)
        self._last_request_at = time.monotonic()

    def _rate_limit_delay(self, error, attempt: int) -> float:
        if error.retry_after is not None:
            if error.retry_after > MAX_RETRY_AFTER_SECONDS:
                raise YahooTransportError(
                    f"the source asked for a {error.retry_after:.0f}s pause, "
                    f"beyond the {MAX_RETRY_AFTER_SECONDS}s this capture will "
                    "wait inside one run; rerun the capture later")
            return error.retry_after
        return RETRY_BACKOFF_SECONDS[
            min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]

    def _attempt(self, url: str, headers: dict) -> YahooResponse:
        request = urllib.request.Request(url, method="GET")
        for key, value in headers.items():
            request.add_header(key, value)
        final_url = url
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200) or 200
                resolved = getattr(response, "geturl", None)
                if callable(resolved):
                    final_url = resolved() or url
        except YahooHostNotAllowedError:
            raise
        except urllib.error.HTTPError as error:
            code = error.code
            if code == 429:
                raise YahooRateLimitedError(
                    f"rate limited by the source (HTTP {code})",
                    retry_after=_retry_after_seconds(
                        getattr(error, "headers", None))) from None
            if code in RETRYABLE_STATUSES:
                raise YahooTransportError(
                    f"transient HTTP {code} from the source") from None
            if code == 422:
                raise YahooProviderError(
                    f"the source refused the requested range or interval "
                    f"(HTTP {code}); this endpoint serves intraday only for a "
                    "short lookback, which is why V2 is a daily corpus"
                ) from None
            raise YahooProviderError(
                f"the source returned HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise YahooTransportError(
                f"network failure contacting the source: "
                f"{type(error).__name__}") from None

        # The handler runs per hop; this runs once on the destination that
        # actually answered. A future handler change, or an injected opener,
        # cannot quietly widen the boundary.
        require_allowed_host(final_url)
        if final_url != url:
            self.stats.redirects += 1

        self.stats.requests += 1
        self.stats.bytes_received += len(raw)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            # Deliberately NOT retryable. A body that is not JSON is an
            # answer, and asking again would just produce it again.
            raise YahooProviderError(
                f"the source returned a body that is not JSON: "
                f"{type(error).__name__}") from None
        if not isinstance(payload, dict):
            raise YahooProviderError("expected a JSON object from the source")
        return YahooResponse(status=status, raw=raw, payload=payload,
                             url=strip_credential_params(final_url))

    def payload(self) -> dict:
        return {
            "schema_version": YAHOO_TRANSPORT_SCHEMA_VERSION,
            "name": self.name,
            "base_url": self.base_url,
            "allowed_hosts": list(ALLOWED_HOSTS),
            "methods": ["GET"],
            "credential_required": False,
            "timeout_seconds": self.timeout,
            "max_attempts": self.max_attempts,
            "stats": self.stats.payload(),
        }


__all__ = [
    "ALLOWED_HOSTS", "CAPTURE_USER_AGENT", "DEFAULT_BASE_URL",
    "MAX_ATTEMPTS", "MAX_RETRY_AFTER_SECONDS", "REQUEST_TIMEOUT_SECONDS",
    "RETRYABLE_STATUSES", "RETRY_BACKOFF_SECONDS",
    "YAHOO_TRANSPORT_SCHEMA_VERSION", "YahooHTTPTransport",
    "YahooHostNotAllowedError", "YahooRateLimitedError", "YahooResponse",
    "YahooTransportError", "YahooTransportStats", "require_allowed_host",
]
