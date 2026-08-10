"""Opaque, deterministic cursors for bounded time-series pagination.

Offset pagination (`offset=700000`) is the obvious approach and the wrong one:
it forces the server to walk everything it skips, and it silently changes
meaning when the underlying series grows. A cursor that carries the last
timestamp it handed out does neither.

The cursor is opaque but not secret -- this API is local and read-only, so
there is nothing to hide. What matters is that it is *bound* to the query that
produced it: a cursor from BTC-USD must not silently work on ETH-USD, because
the resulting page would look perfectly valid while describing another
instrument entirely. Any mismatch, and any malformed cursor, fails closed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json

from scripts.trading_lab.app_api.contracts import (
    APP_API_VERSION,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AppApiError,
)

CURSOR_VERSION = "trading-lab.app-api.cursor.v1"


def _query_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]


def encode_cursor(*, endpoint: str, product: str, last_timestamp: str,
                  query: dict) -> str:
    body = {
        "v": CURSOR_VERSION,
        "api": APP_API_VERSION,
        "endpoint": endpoint,
        "product": product,
        "after": last_timestamp,
        "q": _query_digest(query),
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, *, endpoint: str, product: str, query: dict) -> str:
    """Return the timestamp to resume after, or refuse.

    A cursor issued for another endpoint, another product or a different query
    shape is rejected rather than reinterpreted.
    """
    if not isinstance(cursor, str) or not cursor:
        raise AppApiError("cursor must be a non-empty string")
    padding = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(cursor + padding)
        body = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as error:
        raise AppApiError("cursor is malformed") from error
    if not isinstance(body, dict):
        raise AppApiError("cursor is malformed")
    if body.get("v") != CURSOR_VERSION or body.get("api") != APP_API_VERSION:
        raise AppApiError("cursor was issued by a different API version")
    if body.get("endpoint") != endpoint:
        raise AppApiError("cursor was issued for a different endpoint")
    if body.get("product") != product:
        raise AppApiError("cursor was issued for a different product")
    if body.get("q") != _query_digest(query):
        raise AppApiError("cursor was issued for a different query")
    after = body.get("after")
    if not isinstance(after, str) or not after:
        raise AppApiError("cursor is malformed")
    return after


def require_limit(value: object, *, default: int = DEFAULT_PAGE_SIZE,
                  maximum: int = MAX_PAGE_SIZE) -> int:
    """Validate a page size. Refuses rather than clamping."""
    if value is None or value == "":
        return default
    try:
        limit = int(value)
    except (TypeError, ValueError) as error:
        raise AppApiError(f"limit must be an integer, got {value!r}") from error
    if limit < 1:
        raise AppApiError(f"limit must be at least 1, got {limit}")
    if limit > maximum:
        raise AppApiError(
            f"limit {limit} exceeds the maximum of {maximum}; "
            "narrow the window or page with the cursor instead")
    return limit
