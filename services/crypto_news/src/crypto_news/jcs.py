"""Restricted RFC 8785 JSON Canonicalization Scheme implementation.

V0 deliberately excludes binary floating-point values. Exact market decimals are
represented as strings and ordinal values as IEEE-754-safe integers. The accepted
JSON domain is therefore a strict subset of RFC 8785 and canonicalizes identically
to JCS for every accepted value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _string(value: str) -> bytes:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("JCS strings must not contain lone Unicode surrogates") from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _utf16_sort_key(value: str) -> bytes:
    try:
        return value.encode("utf-16-be", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("JCS object keys must be valid Unicode") from exc


def _encode(value: Any) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise ValueError("JCS integers must remain in the IEEE-754 safe range")
        return str(value).encode("ascii")
    if isinstance(value, float):
        raise TypeError("float values are forbidden; use an exact decimal string")
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JCS object keys must be strings")
        chunks = []
        for key in sorted(value, key=_utf16_sort_key):
            chunks.append(_string(key) + b":" + _encode(value[key]))
        return b"{" + b",".join(chunks) + b"}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return b"[" + b",".join(_encode(item) for item in value) + b"]"
    raise TypeError(f"unsupported JCS value: {type(value).__name__}")


def canonicalize(value: Any) -> bytes:
    """Return RFC 8785-compatible UTF-8 bytes for the restricted V0 JSON domain."""

    return _encode(value)
