"""The one safe way this project reaches a market-data host.

Extracted from the Massive transport when a second provider needed the same
guarantee. A security control with two implementations has one that drifts,
and the copy that drifts is the one nobody re-reads -- so there is a single
allowlist check and a single redirect policy here, parameterised by host.

What it enforces, and why each clause exists rather than being obvious:

* **https only.** A scheme check is not a destination check, but its absence
  is disqualifying on its own.
* **Exact host.** `parsed.hostname` alone would not be enough, which is the
  point of the next two clauses.
* **No userinfo.** `https://evil.example.com@api.vendor.com/` has hostname
  `api.vendor.com`. The reverse spelling is what a reader skims past.
* **Port 443 only.** `api.vendor.com:4444` also has hostname `api.vendor.com`.
* **Redirects validated before they are followed.** `urllib` follows 3xx
  automatically, and its `redirect_request` copies every header except
  Content-Length and Content-Type onto the new request -- `Authorization`
  among them. A vendor redirect to another host would otherwise hand that
  host the credential, bypassing every other control in one step, because
  they all guard the value everywhere *except* the moment it leaves the
  process. Checking inside `redirect_request` means the off-allowlist host is
  never contacted at all.

Even a provider with no credential wants this: bytes fetched from an
unvalidated host must never be recorded as having come from the vendor.
"""

from __future__ import annotations

import urllib.parse
import urllib.request

SAFE_HTTP_SCHEMA_VERSION = "trading-lab.safe-http.v1"

# The only port a market-data host is reached on. Named rather than implied,
# because a port silently survives a hostname comparison.
ALLOWED_PORT = 443

# Query parameters that may carry a secret. Removed from any URL that is
# recorded, logged or followed.
CREDENTIAL_QUERY_KEYS = ("apikey", "api_key", "api-key", "key", "token",
                         "secret", "access_token", "auth")


class SafeHTTPError(RuntimeError):
    """Raised when a URL cannot be reached or recorded safely."""


class HostNotAllowedError(SafeHTTPError):
    """Raised when a request would leave for a host nobody allowlisted."""


def require_https_host(url: str, allowed_hosts, *, error_class=None) -> str:
    """Scheme, host, port and authority shape. Raises, or returns the URL.

    ``error_class`` lets each provider keep its own exception hierarchy while
    sharing one implementation of the check. The alternative -- a single
    shared type -- would force every caller's ``except`` clauses to know about
    this module, which is how a shared helper starts leaking upward.
    """
    fail = error_class or HostNotAllowedError
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise fail(f"refusing a non-https request to {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise fail(
            "refusing a URL carrying userinfo; the authority it appears to "
            "name is not the host that would be contacted")
    host = (parsed.hostname or "").lower()
    if host not in tuple(allowed_hosts):
        raise fail(
            f"{host!r} is not an allowlisted market-data host; allowed: "
            f"{list(allowed_hosts)}")
    try:
        port = parsed.port
    except ValueError as error:
        raise fail(f"malformed port in {host!r}") from error
    if port not in (None, ALLOWED_PORT):
        raise fail(
            f"refusing port {port} on {host!r}; only {ALLOWED_PORT} is allowed")
    return url


def strip_credential_params(url: str) -> str:
    """Drop credential-shaped query parameters. Never logs what it dropped."""
    parsed = urllib.parse.urlsplit(url)
    kept = [(key, value) for key, value in urllib.parse.parse_qsl(parsed.query)
            if key.lower() not in CREDENTIAL_QUERY_KEYS]
    return urllib.parse.urlunsplit((
        parsed.scheme, parsed.netloc, parsed.path,
        urllib.parse.urlencode(kept), ""))


class AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only while it stays inside the allowlist."""

    def __init__(self, allowed_hosts, *, error_class=None):
        self.allowed_hosts = tuple(allowed_hosts)
        self.error_class = error_class or HostNotAllowedError

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            require_https_host(newurl, self.allowed_hosts,
                               error_class=self.error_class)
        except Exception as error:
            # Raised rather than returned as None: None makes urllib surface
            # the original 3xx as an opaque HTTPError, which would read like a
            # vendor problem instead of a refused destination.
            raise self.error_class(
                f"refusing to follow a redirect to a host outside the "
                f"market-data allowlist: {error}") from None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_hardened_opener(allowed_hosts, *,
                          error_class=None) -> urllib.request.OpenerDirector:
    """The production opener. Never bare urlopen.

    Bare ``urlopen`` uses the default handler chain, whose redirect handler
    follows a 3xx anywhere. This chain is assembled explicitly so the only
    redirect policy in play is the one above.
    """
    return urllib.request.build_opener(
        AllowlistedRedirectHandler(allowed_hosts, error_class=error_class))


__all__ = [
    "ALLOWED_PORT", "AllowlistedRedirectHandler", "CREDENTIAL_QUERY_KEYS",
    "HostNotAllowedError", "SAFE_HTTP_SCHEMA_VERSION", "SafeHTTPError",
    "build_hardened_opener", "require_https_host", "strip_credential_params",
]
