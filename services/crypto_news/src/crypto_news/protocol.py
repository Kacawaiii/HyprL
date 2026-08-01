"""Load and validate the immutable V0 research protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from crypto_news.jcs import canonicalize


PROTOCOL_V0_SHA256 = "64013d519c4de2d37de6116e2fce4592c535ef396e38f25e8c3f55e097ff8b9b"


class ProtocolValidationError(RuntimeError):
    """Raised when a protocol payload differs from frozen V0."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_copy(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    _payload: Mapping[str, Any]
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return _deep_copy(self._payload)


def _resource_payload() -> dict[str, Any]:
    resource = files("crypto_news").joinpath("protocol_v0.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolValidationError("V0 protocol must be a JSON object")
    return payload


def validate_protocol_payload(payload: Mapping[str, Any]) -> FrozenProtocol:
    digest = hashlib.sha256(canonicalize(payload)).hexdigest()
    if digest != PROTOCOL_V0_SHA256:
        raise ProtocolValidationError(
            f"V0 protocol hash mismatch: expected {PROTOCOL_V0_SHA256}, got {digest}"
        )
    return FrozenProtocol(_deep_freeze(dict(payload)), digest)


def load_frozen_protocol() -> FrozenProtocol:
    return validate_protocol_payload(_resource_payload())
