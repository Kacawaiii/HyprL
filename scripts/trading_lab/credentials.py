"""Secrets that can be used but not read.

A market-data API key has exactly one legitimate destination: an outbound
request header. Every other path it can take is a leak, and the leaks are not
exotic -- they are the ordinary conveniences of debugging. A repr in a
traceback. A config dict serialised into a support bundle. A provider payload
returned by the API and rendered into a page. An exception message that helpfully
quotes the request that failed. None of those are written by someone trying to
expose a key; they are written by someone trying to be helpful, months apart,
and each one is individually reasonable.

So the key is never held as a string on any object that can be printed,
serialised, logged or returned. ``Secret`` wraps it, refuses to render it, and
hands out the real value only through ``reveal()`` -- a call that is easy to
grep for and impossible to make by accident.

**What this is not.** This is not brokerage authentication. There is no account
endpoint, no balance, no order, no position, no withdrawal. A market-data key
that leaks is a billing problem; a brokerage key that leaks is a financial one,
and this project deliberately never holds the second kind.
"""

from __future__ import annotations

import os

CREDENTIAL_SCHEMA_VERSION = "trading-lab.credential.v1"

# The one variable. Named for the provider, not shared across providers: a
# single HYPRL_API_KEY would eventually be sent to whichever host asked.
MASSIVE_API_KEY_ENV = "HYPRL_MASSIVE_API_KEY"

# What a payload says instead of the key. A fixed string, so a test asserting
# "the key is absent" cannot be satisfied by an empty value that merely looks
# absent because it was never set.
REDACTED = "***REDACTED***"


class CredentialError(RuntimeError):
    """Raised when a credential is missing, malformed, or asked to render."""


class Secret:
    """A string that will not print itself.

    ``__repr__`` and ``__str__`` are the two functions that turn an object
    into a log line, and both are overridden. ``__format__`` too, because
    f-strings go through it and bypass ``__str__`` when a spec is given.
    """

    __slots__ = ("_value", "name")

    def __init__(self, value: str, *, name: str = "credential"):
        if not isinstance(value, str) or not value.strip():
            raise CredentialError(f"{name} must be a non-empty string")
        self._value = value
        self.name = name

    def reveal(self) -> str:
        """The actual value. The only way out, and deliberately conspicuous."""
        return self._value

    # Every rendering path, closed.
    def __repr__(self) -> str:
        return f"<Secret {self.name} {REDACTED}>"

    def __str__(self) -> str:
        return REDACTED

    def __format__(self, spec: str) -> str:
        return REDACTED

    def __reduce__(self):
        # Pickling would write the value to disk in cleartext.
        raise CredentialError(f"{self.name} must not be serialised")

    def __eq__(self, other: object) -> bool:
        # Constant-time-ish, and never against a bare string: comparing a
        # Secret to a literal is how a key ends up written in a source file.
        if not isinstance(other, Secret):
            return NotImplemented
        import hmac
        return hmac.compare_digest(self._value, other._value)

    def __hash__(self) -> int:
        # Hashing the value would let it be recovered from a rainbow table of
        # candidate keys, and Secrets have no reason to be dict keys.
        raise CredentialError(f"{self.name} must not be hashed")

    def __bool__(self) -> bool:
        return bool(self._value)


class CredentialProvider:
    """Where a secret comes from. Read at call time, never cached in a field."""

    env_var: str = ""

    def available(self) -> bool:
        raise NotImplementedError

    def get(self) -> Secret:
        raise NotImplementedError

    def payload(self) -> dict:
        """Safe to serialise: says whether a key exists, never what it is."""
        return {
            "schema_version": CREDENTIAL_SCHEMA_VERSION,
            "env_var": self.env_var,
            "configured": self.available(),
            "value": REDACTED,
        }


class EnvCredentialProvider(CredentialProvider):
    """Reads one environment variable, on demand.

    On demand rather than at construction so that a process which never makes
    a request never holds the key in memory, and so that a test can set and
    unset the variable without rebuilding the provider.
    """

    def __init__(self, env_var: str):
        self.env_var = env_var

    def available(self) -> bool:
        return bool(os.environ.get(self.env_var, "").strip())

    def get(self) -> Secret:
        raw = os.environ.get(self.env_var, "")
        if not raw.strip():
            raise CredentialError(
                f"{self.env_var} is not set. Market-data requests need it; "
                "no default and no fallback key exists.")
        return Secret(raw.strip(), name=self.env_var)


class MissingCredentialProvider(CredentialProvider):
    """A provider that has no credential and says so plainly.

    The default for offline and mock-transport use. It is not an empty string
    pretending to be a key: asking for the value raises, so an accidental real
    request fails loudly instead of going out unauthenticated.
    """

    def __init__(self, env_var: str = MASSIVE_API_KEY_ENV):
        self.env_var = env_var

    def available(self) -> bool:
        return False

    def get(self) -> Secret:
        raise CredentialError(
            f"no credential configured for {self.env_var}; this context is "
            "offline by construction")


def massive_credentials() -> EnvCredentialProvider:
    return EnvCredentialProvider(MASSIVE_API_KEY_ENV)


# --- leak detection --------------------------------------------------------

# The names a secret hides behind. Anything matching these in a payload is
# redacted regardless of where it came from, because the recursive scrub below
# is a safety net for values that never went through Secret at all.
SENSITIVE_KEY_HINTS = (
    "api_key", "apikey", "secret", "token", "password", "passwd",
    "credential", "authorization", "auth_header", "private_key", "bearer",
)


def _is_sensitive(key: object) -> bool:
    return isinstance(key, str) and any(
        hint in key.lower() for hint in SENSITIVE_KEY_HINTS)


def redact(payload: object) -> object:
    """Recursively replace secret-shaped values before anything is written out.

    Applied at every serialisation boundary -- API responses, support bundles,
    log records -- rather than at each call site, because the call site that
    forgets is exactly the one that leaks.
    """
    if isinstance(payload, Secret):
        return REDACTED
    if isinstance(payload, dict):
        return {key: (REDACTED if _is_sensitive(key) else redact(value))
                for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        rendered = [redact(item) for item in payload]
        return type(payload)(rendered) if isinstance(payload, tuple) else rendered
    return payload


def assert_absent(needle: str, haystack: object, *, where: str = "payload") -> None:
    """Fail loudly if a known secret appears anywhere in a structure.

    Used with a sentinel value in tests. The check is on the rendered text,
    not on the keys, so a key that leaked into a *value* -- inside a URL, an
    exception message, a header dump -- is caught too.
    """
    if not needle:
        raise CredentialError("assert_absent needs a non-empty sentinel")
    import json

    try:
        rendered = json.dumps(haystack, default=str)
    except (TypeError, ValueError):                  # pragma: no cover
        rendered = repr(haystack)
    if needle in rendered:
        raise CredentialError(
            f"credential leaked into {where}: the sentinel value is present. "
            "This is the failure this check exists to catch.")


__all__ = [
    "CREDENTIAL_SCHEMA_VERSION", "CredentialError", "CredentialProvider",
    "EnvCredentialProvider", "MASSIVE_API_KEY_ENV", "MissingCredentialProvider",
    "REDACTED", "SENSITIVE_KEY_HINTS", "Secret", "assert_absent",
    "massive_credentials", "redact",
]
