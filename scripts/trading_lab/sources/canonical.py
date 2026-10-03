"""canonical_serialization.rule: the repo's sha256_canonical byte form, shared by every source spec."""

from __future__ import annotations

import hashlib
import json


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
