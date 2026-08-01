"""Content-addressed raw evidence and deterministic corroboration policy."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from crypto_news.models import FactStatus
from crypto_news.sources import SourceRegistry, SourceTier


_HASH = re.compile(r"^[0-9a-f]{64}$")


class EvidenceIntegrityError(RuntimeError):
    """Raised when content-addressed evidence fails integrity checks."""


@dataclass(frozen=True, slots=True)
class RawArtifact:
    sha256: str
    size_bytes: int
    path: Path


class EvidenceStore:
    """Write-once store addressed by the SHA-256 of exact response bytes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if not _HASH.fullmatch(digest):
            raise EvidenceIntegrityError("artifact digest must be lowercase SHA-256")
        return self.root / digest[:2] / f"{digest}.bin"

    def store(self, body: bytes) -> RawArtifact:
        if not isinstance(body, bytes) or not body:
            raise EvidenceIntegrityError("raw evidence must be non-empty bytes")
        digest = hashlib.sha256(body).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = self.read(digest)
            if existing != body:
                raise EvidenceIntegrityError("artifact digest collision or corruption")
            return RawArtifact(digest, len(body), target)

        fd, temporary_name = tempfile.mkstemp(prefix="artifact-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if self.read(digest) != body:
                    raise EvidenceIntegrityError("artifact digest collision or corruption")
        finally:
            temporary.unlink(missing_ok=True)
        return RawArtifact(digest, len(body), target)

    def read(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            body = path.read_bytes()
        except FileNotFoundError as exc:
            raise EvidenceIntegrityError("raw artifact is missing") from exc
        if hashlib.sha256(body).hexdigest() != digest:
            raise EvidenceIntegrityError("raw artifact hash mismatch")
        return body


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    receipt_id: str
    source_id: str
    claim_key: str
    first_seen_at: datetime

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.source_id or not self.claim_key.strip():
            raise ValueError("receipt_id, source_id, and claim_key are required")
        object.__setattr__(self, "claim_key", self.claim_key.strip())
        if self.first_seen_at.tzinfo is None:
            raise ValueError("first_seen_at must be timezone-aware")
        object.__setattr__(self, "first_seen_at", self.first_seen_at.astimezone(timezone.utc))


@dataclass(frozen=True, slots=True)
class EvidenceChain:
    claim_key: str
    fact_status: FactStatus
    source_receipt_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    primary_source_receipt_id: str | None
    promotable: bool
    rejection_reason: str | None


def build_evidence_chain(
    candidates: tuple[EvidenceCandidate, ...], registry: SourceRegistry
) -> EvidenceChain:
    """Apply the frozen Phase 2 policy without inferring event semantics."""

    if not candidates:
        raise ValueError("at least one evidence candidate is required")
    if len({candidate.claim_key for candidate in candidates}) != 1:
        raise ValueError("evidence candidates must share the same claim_key")
    ordered = sorted(candidates, key=lambda item: (item.first_seen_at, item.receipt_id))
    deduplicated: list[EvidenceCandidate] = []
    seen_receipts: set[str] = set()
    for candidate in ordered:
        if candidate.receipt_id not in seen_receipts:
            registry.get(candidate.source_id)
            deduplicated.append(candidate)
            seen_receipts.add(candidate.receipt_id)

    primary = next(
        (
            candidate
            for candidate in deduplicated
            if registry.get(candidate.source_id).primary_authority
        ),
        None,
    )
    receipt_ids = tuple(candidate.receipt_id for candidate in deduplicated)
    if primary is not None:
        return EvidenceChain(
            claim_key=deduplicated[0].claim_key,
            fact_status=FactStatus.PRIMARY_CONFIRMED,
            source_receipt_ids=receipt_ids,
            source_ids=tuple(candidate.source_id for candidate in deduplicated),
            primary_source_receipt_id=primary.receipt_id,
            promotable=True,
            rejection_reason=None,
        )

    independent_groups = {
        registry.get(candidate.source_id).independence_group
        for candidate in deduplicated
        if registry.get(candidate.source_id).tier <= SourceTier.TIER_2
    }
    if len(independent_groups) >= 2:
        return EvidenceChain(
            claim_key=deduplicated[0].claim_key,
            fact_status=FactStatus.CORROBORATED,
            source_receipt_ids=receipt_ids,
            source_ids=tuple(candidate.source_id for candidate in deduplicated),
            primary_source_receipt_id=None,
            promotable=True,
            rejection_reason=None,
        )
    return EvidenceChain(
        claim_key=deduplicated[0].claim_key,
        fact_status=FactStatus.UNVERIFIED,
        source_receipt_ids=receipt_ids,
        source_ids=tuple(candidate.source_id for candidate in deduplicated),
        primary_source_receipt_id=None,
        promotable=False,
        rejection_reason="insufficient independent corroboration",
    )
