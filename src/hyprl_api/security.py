"""Token helpers for the HyprL API."""

from __future__ import annotations

import hashlib
import secrets

from hyprl_api.config import get_settings

TOKEN_PREFIX_PREFIX = "tok"


def generate_token_prefix() -> str:
    suffix = secrets.token_hex(6)
    return f"{TOKEN_PREFIX_PREFIX}_{suffix}"


def generate_token_secret() -> str:
    return secrets.token_urlsafe(24)


def format_token(prefix: str, secret: str) -> str:
    return f"{prefix}.{secret}"


def hash_token_secret(secret: str) -> str:
    settings = get_settings()
    payload = f"{settings.token_hash_secret}:{secret}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()