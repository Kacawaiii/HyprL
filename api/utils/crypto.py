"""Argon2id helpers for token hashing."""

from __future__ import annotations

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError

_HASHER = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=2, hash_len=32, salt_len=16, type=Type.ID)


def hash_token(plain: str) -> str:
    """Return Argon2id hash for the provided token string."""
    if not isinstance(plain, str) or not plain:
        raise ValueError("Token must be a non-empty string")
    return _HASHER.hash(plain)


def verify_token(plain: str, hashed: str) -> bool:
    """Verify plain token against stored hash."""
    try:
        return _HASHER.verify(hashed, plain)
    except VerifyMismatchError:
        return False
