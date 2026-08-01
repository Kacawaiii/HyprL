"""Ed25519 manifest interface that never loads or stores a private key."""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from crypto_news.jcs import canonicalize


class SigningValidationError(ValueError):
    """Raised when a signature envelope is malformed or invalid."""


class Ed25519DigestSigner(Protocol):
    """Boundary for a dedicated signer that receives only a 32-byte digest."""

    key_id: str

    @property
    def public_key_bytes(self) -> bytes: ...

    def sign_digest(self, digest: bytes) -> bytes: ...


def manifest_digest(manifest: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonicalize(manifest)).digest()


def sign_manifest(
    manifest: Mapping[str, Any], signer: Ed25519DigestSigner
) -> dict[str, str]:
    key_id = signer.key_id
    if not isinstance(key_id, str) or not key_id.strip():
        raise SigningValidationError("key_id must be a non-empty string")
    public_key = signer.public_key_bytes
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise SigningValidationError("Ed25519 public key must contain 32 bytes")
    digest = manifest_digest(manifest)
    signature = signer.sign_digest(digest)
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise SigningValidationError("Ed25519 signature must contain 64 bytes")
    return {
        "algorithm": "Ed25519",
        "key_id": key_id,
        "manifest_sha256": digest.hex(),
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_manifest_signature(
    manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    expected_key_id: str,
    expected_public_key_bytes: bytes,
) -> bool:
    if not isinstance(expected_key_id, str) or not expected_key_id.strip():
        raise SigningValidationError("expected_key_id must be a non-empty string")
    if (
        not isinstance(expected_public_key_bytes, bytes)
        or len(expected_public_key_bytes) != 32
    ):
        raise SigningValidationError("expected Ed25519 public key must contain 32 bytes")
    if signature.get("algorithm") != "Ed25519":
        raise SigningValidationError("unsupported manifest signature algorithm")
    if signature.get("key_id") != expected_key_id:
        raise SigningValidationError("manifest signature key_id is not trusted")
    digest = manifest_digest(manifest)
    if signature.get("manifest_sha256") != digest.hex():
        raise SigningValidationError("manifest digest does not match signature envelope")
    try:
        public_key = base64.b64decode(
            str(signature["public_key_base64"]), validate=True
        )
        signature_bytes = base64.b64decode(
            str(signature["signature_base64"]), validate=True
        )
    except (KeyError, ValueError) as exc:
        raise SigningValidationError("invalid base64 signature envelope") from exc
    if len(public_key) != 32 or len(signature_bytes) != 64:
        raise SigningValidationError("invalid Ed25519 key or signature length")
    if not hmac.compare_digest(public_key, expected_public_key_bytes):
        raise SigningValidationError("manifest public key is not trusted")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature_bytes, digest)
    except (InvalidSignature, ValueError) as exc:
        raise SigningValidationError("invalid Ed25519 manifest signature") from exc
    return True
