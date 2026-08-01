"""Append-only SQLite journal, JCS hash chain, and signed JSONL export."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from crypto_news.evidence import EvidenceChain
from crypto_news.jcs import canonicalize
from crypto_news.models import (
    Analysis,
    EventVersion,
    FactStatus,
    HumanDecision,
    MarketSnapshot,
    Outcome,
    OutcomeAnchor,
    Retraction,
    SourceReceipt,
)
from crypto_news.signing import (
    Ed25519DigestSigner,
    sign_manifest,
    verify_manifest_signature,
)
from crypto_news.sources import (
    DEFAULT_SOURCE_REGISTRY,
    SourceDefinition,
    SourceRegistry,
    SourceRegistryError,
)


GENESIS_RECORD_HASH = "0" * 64
Record = (
    SourceReceipt
    | EventVersion
    | MarketSnapshot
    | Analysis
    | HumanDecision
    | Outcome
    | Retraction
)


class JournalValidationError(RuntimeError):
    """Raised when append-only or causal journal invariants are violated."""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    sequence: int
    record_id: str
    record_type: str
    recorded_at: datetime
    canonical_payload: bytes
    previous_record_hash: str
    record_hash: str


@dataclass(frozen=True, slots=True)
class ExportBundle:
    jsonl_path: Path
    manifest_path: Path
    signature_path: Path
    manifest: dict[str, Any]
    signature: dict[str, str]


RECORD_IDENTIFIERS = {
    SourceReceipt: "receipt_id",
    EventVersion: "version_id",
    MarketSnapshot: "snapshot_id",
    Analysis: "analysis_id",
    HumanDecision: "decision_id",
    Outcome: "outcome_id",
    Retraction: "retraction_id",
}


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise JournalValidationError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise JournalValidationError(f"{name} must use UTC")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise JournalValidationError(f"stored {name} is not an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JournalValidationError(f"stored {name} is invalid") from exc
    return _utc(parsed, name)


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry_envelope(record_type: str, record_id: str, recorded_at: str, payload: Any) -> dict[str, Any]:
    return {
        "payload": payload,
        "record_id": record_id,
        "record_type": record_type,
        "recorded_at": recorded_at,
    }


def verify_jsonl_export(
    path: Path,
    *,
    expected_first_sequence: int | None = 1,
    expected_previous_record_hash: str | None = GENESIS_RECORD_HASH,
) -> bool:
    previous = expected_previous_record_hash
    expected_sequence = expected_first_sequence
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JournalValidationError("invalid JSONL record") from exc
            if not isinstance(line, dict):
                raise JournalValidationError("JSONL record must be an object")
            if expected_sequence is None:
                expected_sequence = line.get("sequence")
            if previous is None:
                previous = line.get("previous_record_hash")
            if line.get("sequence") != expected_sequence:
                raise JournalValidationError("non-contiguous JSONL sequence")
            if line.get("previous_record_hash") != previous:
                raise JournalValidationError("previous_record_hash breaks the chain")
            envelope = _entry_envelope(
                line.get("record_type"),
                line.get("record_id"),
                line.get("recorded_at"),
                line.get("payload"),
            )
            digest = hashlib.sha256(
                bytes.fromhex(previous) + canonicalize(envelope)
            ).hexdigest()
            if line.get("record_hash") != digest:
                raise JournalValidationError("record_hash does not match canonical payload")
            previous = digest
            expected_sequence += 1
    return True


def _read_canonical_json_object(path: Path, name: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JournalValidationError(f"invalid {name} JSON") from exc
    if not isinstance(payload, dict):
        raise JournalValidationError(f"{name} must be a JSON object")
    if raw != canonicalize(payload) + b"\n":
        raise JournalValidationError(f"{name} is not canonical JCS")
    return payload


def verify_signed_export_bundle(
    jsonl_path: Path,
    manifest_path: Path,
    signature_path: Path,
    *,
    expected_key_id: str,
    expected_public_key_bytes: bytes,
) -> bool:
    """Verify a daily export against an externally trusted Ed25519 key."""

    jsonl_path = Path(jsonl_path)
    manifest = _read_canonical_json_object(Path(manifest_path), "manifest")
    signature = _read_canonical_json_object(Path(signature_path), "signature")
    verify_manifest_signature(
        manifest,
        signature,
        expected_key_id=expected_key_id,
        expected_public_key_bytes=expected_public_key_bytes,
    )
    raw = jsonl_path.read_bytes()
    if manifest.get("jsonl_filename") != jsonl_path.name:
        raise JournalValidationError("manifest JSONL filename does not match")
    if manifest.get("jsonl_sha256") != hashlib.sha256(raw).hexdigest():
        raise JournalValidationError("manifest JSONL digest does not match")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as exc:
        raise JournalValidationError("invalid JSONL record") from exc
    if manifest.get("record_count") != len(rows):
        raise JournalValidationError("manifest record_count does not match")
    start = _parse_timestamp(manifest.get("period_start_utc"), "period_start_utc")
    end = _parse_timestamp(manifest.get("period_end_utc"), "period_end_utc")
    if (
        start.hour != 0
        or start.minute != 0
        or start.second != 0
        or start.microsecond != 0
        or end != start + timedelta(days=1)
    ):
        raise JournalValidationError("manifest period must be one complete UTC day")
    if rows:
        first = rows[0]
        last = rows[-1]
        expected_fields = {
            "first_record_hash": first.get("record_hash"),
            "first_sequence": first.get("sequence"),
            "last_record_hash": last.get("record_hash"),
            "last_sequence": last.get("sequence"),
            "previous_record_hash": first.get("previous_record_hash"),
        }
        if any(manifest.get(key) != value for key, value in expected_fields.items()):
            raise JournalValidationError("manifest chain anchors do not match JSONL")
        verify_jsonl_export(
            jsonl_path,
            expected_first_sequence=first.get("sequence"),
            expected_previous_record_hash=first.get("previous_record_hash"),
        )
        for row in rows:
            recorded_at = _parse_timestamp(row.get("recorded_at"), "recorded_at")
            if not start <= recorded_at < end:
                raise JournalValidationError("JSONL record is outside manifest period")
    else:
        for key in (
            "first_record_hash",
            "first_sequence",
            "last_record_hash",
            "last_sequence",
        ):
            if manifest.get(key) is not None:
                raise JournalValidationError("empty manifest has non-empty chain anchors")
    return True


class Journal:
    """Single-writer append API with database-enforced immutability."""

    def __init__(
        self,
        path: Path,
        *,
        source_registry: SourceRegistry = DEFAULT_SOURCE_REGISTRY,
    ) -> None:
        self.path = Path(path)
        self.source_registry = source_registry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS journal_records (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                record_type TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                canonical_payload BLOB NOT NULL,
                previous_record_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS journal_links (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                source_record_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                target_record_id TEXT NOT NULL,
                UNIQUE(source_record_id, relationship, target_record_id),
                FOREIGN KEY(source_record_id) REFERENCES journal_records(record_id),
                FOREIGN KEY(target_record_id) REFERENCES journal_records(record_id)
            );

            CREATE TABLE IF NOT EXISTS signing_keys (
                key_id TEXT PRIMARY KEY,
                public_key_base64 TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                replaces_key_id TEXT,
                rotation_record_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY(replaces_key_id) REFERENCES signing_keys(key_id),
                FOREIGN KEY(rotation_record_id) REFERENCES journal_records(record_id)
            );

            CREATE TRIGGER IF NOT EXISTS journal_records_no_update
            BEFORE UPDATE ON journal_records
            BEGIN SELECT RAISE(ABORT, 'journal_records is insert-only'); END;
            CREATE TRIGGER IF NOT EXISTS journal_records_no_delete
            BEFORE DELETE ON journal_records
            BEGIN SELECT RAISE(ABORT, 'journal_records is insert-only'); END;
            CREATE TRIGGER IF NOT EXISTS journal_links_no_update
            BEFORE UPDATE ON journal_links
            BEGIN SELECT RAISE(ABORT, 'journal_links is insert-only'); END;
            CREATE TRIGGER IF NOT EXISTS journal_links_no_delete
            BEFORE DELETE ON journal_links
            BEGIN SELECT RAISE(ABORT, 'journal_links is insert-only'); END;
            CREATE TRIGGER IF NOT EXISTS signing_keys_no_update
            BEFORE UPDATE ON signing_keys
            BEGIN SELECT RAISE(ABORT, 'signing_keys is insert-only'); END;
            CREATE TRIGGER IF NOT EXISTS signing_keys_no_delete
            BEFORE DELETE ON signing_keys
            BEGIN SELECT RAISE(ABORT, 'signing_keys is insert-only'); END;
            """
        )
        self._connection.commit()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def count_records(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM journal_records"
        ).fetchone()
        return int(row["count"])

    def has_record(self, record_id: str) -> bool:
        """Return whether a stable record identity already exists."""

        if not isinstance(record_id, str) or not record_id:
            raise JournalValidationError("record_id must be a non-empty string")
        row = self._connection.execute(
            "SELECT 1 FROM journal_records WHERE record_id = ? LIMIT 1",
            (record_id,),
        ).fetchone()
        return row is not None

    def _stored_record(self, record_id: str, expected_type: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT record_type, canonical_payload FROM journal_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None or row["record_type"] != expected_type:
            raise JournalValidationError(
                f"referenced {expected_type} record does not exist: {record_id}"
            )
        envelope = json.loads(bytes(row["canonical_payload"]))
        return envelope["payload"]

    def _registered_source(self, payload: Mapping[str, Any]) -> SourceDefinition:
        try:
            source = self.source_registry.get_by_name(str(payload["source_name"]))
        except (KeyError, SourceRegistryError) as exc:
            raise JournalValidationError(
                "SourceReceipt must identify a registered source"
            ) from exc
        if int(source.tier) != payload["source_tier"]:
            raise JournalValidationError(
                "SourceReceipt tier does not match the registered source"
            )
        receipt_host = urlsplit(str(payload["canonical_url"])).hostname
        if receipt_host not in source.canonical_hosts:
            raise JournalValidationError(
                "SourceReceipt URL host does not match the registered source"
            )
        return source

    def _reject_duplicate_source_content(self, content_sha256: str) -> None:
        rows = self._connection.execute(
            """
            SELECT canonical_payload FROM journal_records
            WHERE record_type = ?
            """,
            (SourceReceipt.record_type,),
        ).fetchall()
        for row in rows:
            envelope = json.loads(bytes(row["canonical_payload"]))
            if envelope["payload"]["content_sha256"] == content_sha256:
                raise JournalValidationError(
                    "SourceReceipt content_sha256 already exists in the journal"
                )

    def _validate_evidence_chain(
        self,
        record: EventVersion,
        chain: EvidenceChain | None,
        source_payloads: list[dict[str, Any]],
    ) -> None:
        if chain is None:
            raise JournalValidationError(
                "CORROBORATED requires a matching evidence chain"
            )
        if chain.claim_key != record.event_id:
            raise JournalValidationError("evidence chain claim_key differs from event_id")
        if chain.fact_status is not record.fact_status or not chain.promotable:
            raise JournalValidationError("evidence chain status is not promotable")
        if chain.source_receipt_ids != record.source_receipt_ids:
            raise JournalValidationError("evidence chain receipt IDs differ from event")
        if chain.primary_source_receipt_id != record.primary_source_receipt_id:
            raise JournalValidationError("evidence chain primary source differs from event")
        if len(chain.source_ids) != len(source_payloads):
            raise JournalValidationError("evidence chain source IDs are incomplete")
        for source_id, payload in zip(chain.source_ids, source_payloads, strict=True):
            try:
                source = self.source_registry.get(source_id)
            except SourceRegistryError as exc:
                raise JournalValidationError(
                    "evidence chain references an unknown source"
                ) from exc
            if (
                source.name != payload["source_name"]
                or int(source.tier) != payload["source_tier"]
            ):
                raise JournalValidationError(
                    "evidence chain source identity differs from SourceReceipt"
                )

    def _links_for(
        self, record: Record, evidence_chain: EvidenceChain | None = None
    ) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        if isinstance(record, EventVersion):
            source_payloads: list[dict[str, Any]] = []
            for receipt_id in record.source_receipt_ids:
                source_payloads.append(
                    self._stored_record(receipt_id, SourceReceipt.record_type)
                )
                links.append(("source_receipt", receipt_id))
            source_definitions = [
                self._registered_source(payload) for payload in source_payloads
            ]
            content_hashes = {
                payload["content_sha256"] for payload in source_payloads
            }
            if len(content_hashes) != len(source_payloads):
                raise JournalValidationError(
                    "event source receipts must have distinct content_sha256 values"
                )
            if record.fact_status is FactStatus.PRIMARY_CONFIRMED:
                primary = self._stored_record(
                    str(record.primary_source_receipt_id), SourceReceipt.record_type
                )
                primary_source = self._registered_source(primary)
                if not primary_source.primary_authority:
                    raise JournalValidationError(
                        "PRIMARY_CONFIRMED requires a registered Tier 0 primary authority"
                    )
                if evidence_chain is not None:
                    self._validate_evidence_chain(
                        record, evidence_chain, source_payloads
                    )
            if record.fact_status is FactStatus.CORROBORATED:
                independent_sources = {
                    source.independence_group
                    for source in source_definitions
                    if source.tier <= 2
                }
                if len(independent_sources) < 2:
                    raise JournalValidationError(
                        "CORROBORATED requires two distinct Tier 0-2 sources"
                    )
                self._validate_evidence_chain(record, evidence_chain, source_payloads)
            if record.previous_version_id is not None:
                previous = self._stored_record(
                    record.previous_version_id, EventVersion.record_type
                )
                if previous["event_id"] != record.event_id:
                    raise JournalValidationError("previous event version belongs to another event")
                if previous["version_number"] + 1 != record.version_number:
                    raise JournalValidationError("event version numbers must be contiguous")
                if previous["first_seen_at"] != _timestamp(record.first_seen_at):
                    raise JournalValidationError("event first_seen_at is immutable across versions")
                links.append(("previous_version", record.previous_version_id))
        elif isinstance(record, MarketSnapshot):
            event = self._stored_record(
                record.event_version_id, EventVersion.record_type
            )
            first_seen = _parse_timestamp(event["first_seen_at"], "first_seen_at")
            if record.observed_at <= first_seen:
                raise JournalValidationError(
                    "market price must be strictly after first_seen_at"
                )
            if record.asset.value not in event["affected_assets"]:
                raise JournalValidationError("snapshot asset is not affected by the event")
            links.append(("event_version", record.event_version_id))
        elif isinstance(record, Analysis):
            event = self._stored_record(
                record.event_version_id, EventVersion.record_type
            )
            if event["first_seen_at"] != _timestamp(record.first_seen_at):
                raise JournalValidationError("analysis first_seen_at differs from event")
            if event["fact_status"] != record.fact_status.value:
                raise JournalValidationError("analysis fact_status differs from event")
            if event["event_type"] != record.event_type.value:
                raise JournalValidationError("analysis event_type differs from event")
            if event["affected_assets"] != [asset.value for asset in record.affected_assets]:
                raise JournalValidationError("analysis affected_assets differ from event")
            primary_id = event.get("primary_source_receipt_id")
            if not primary_id:
                raise JournalValidationError("analysis requires a primary source_receipt")
            primary = self._stored_record(primary_id, SourceReceipt.record_type)
            if primary["canonical_url"] != record.primary_source_url:
                raise JournalValidationError("analysis primary source URL differs from receipt")
            if primary["published_at"] != _timestamp(record.published_at):
                raise JournalValidationError("analysis published_at differs from primary receipt")
            corroborating_urls = {
                self._stored_record(receipt_id, SourceReceipt.record_type)[
                    "canonical_url"
                ]
                for receipt_id in event["source_receipt_ids"]
                if receipt_id != primary_id
            }
            if not set(record.corroborating_source_urls).issubset(
                corroborating_urls
            ):
                raise JournalValidationError(
                    "analysis corroborating source is not linked to the event"
                )
            links.append(("event_version", record.event_version_id))
            for snapshot_id in record.market_snapshot_ids:
                snapshot = self._stored_record(
                    snapshot_id, MarketSnapshot.record_type
                )
                if snapshot["event_version_id"] != record.event_version_id:
                    raise JournalValidationError("analysis snapshot belongs to another event")
                captured = _parse_timestamp(snapshot["captured_at"], "captured_at")
                if captured > record.frozen_at:
                    raise JournalValidationError(
                        "analysis frozen_at precedes a referenced market snapshot"
                    )
                links.append(("market_snapshot", snapshot_id))
        elif isinstance(record, HumanDecision):
            analysis = self._stored_record(record.analysis_id, Analysis.record_type)
            frozen = _parse_timestamp(analysis["frozen_at"], "frozen_at")
            if record.human_decided_at < frozen:
                raise JournalValidationError("human_decided_at precedes frozen_at")
            links.append(("analysis", record.analysis_id))
        elif isinstance(record, Outcome):
            analysis = self._stored_record(record.analysis_id, Analysis.record_type)
            if record.asset.value not in analysis["affected_assets"]:
                raise JournalValidationError("outcome asset is not affected by the analysis")
            if record.anchor is OutcomeAnchor.AI:
                frozen = _parse_timestamp(analysis["frozen_at"], "frozen_at")
                if record.anchor_at != frozen:
                    raise JournalValidationError("AI outcome anchor must equal frozen_at")
            else:
                decision = self._stored_record(
                    str(record.human_decision_id), HumanDecision.record_type
                )
                if decision["analysis_id"] != record.analysis_id:
                    raise JournalValidationError("human decision belongs to another analysis")
                decided = _parse_timestamp(
                    decision["human_decided_at"], "human_decided_at"
                )
                if record.anchor_at != decided:
                    raise JournalValidationError(
                        "human outcome anchor must equal human_decided_at"
                    )
                links.append(("human_decision", str(record.human_decision_id)))
            links.append(("analysis", record.analysis_id))
        elif isinstance(record, Retraction):
            event = self._stored_record(
                record.event_version_id, EventVersion.record_type
            )
            self._stored_record(record.source_receipt_id, SourceReceipt.record_type)
            if record.source_receipt_id not in event["source_receipt_ids"]:
                raise JournalValidationError("retraction source is not linked to the event")
            links.extend(
                [
                    ("event_version", record.event_version_id),
                    ("source_receipt", record.source_receipt_id),
                ]
            )
        return links

    @staticmethod
    def _effective_at(record: Record) -> datetime:
        if isinstance(record, SourceReceipt):
            return record.retrieved_at
        if isinstance(record, EventVersion):
            return record.created_at
        if isinstance(record, MarketSnapshot):
            return record.captured_at
        if isinstance(record, Analysis):
            return record.frozen_at
        if isinstance(record, HumanDecision):
            return record.human_decided_at
        if isinstance(record, Outcome):
            return record.observed_at
        return record.recorded_at

    def _insert(
        self,
        *,
        record_type: str,
        record_id: str,
        payload: Mapping[str, Any],
        recorded_at: datetime,
        links: list[tuple[str, str]],
    ) -> JournalEntry:
        row = self._connection.execute(
            """
            SELECT sequence, record_hash, recorded_at
            FROM journal_records ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        if row is not None:
            previous_recorded_at = _parse_timestamp(
                row["recorded_at"], "recorded_at"
            )
            if recorded_at < previous_recorded_at:
                raise JournalValidationError(
                    "journal recorded_at values must be non-decreasing"
                )
        previous = GENESIS_RECORD_HASH if row is None else str(row["record_hash"])
        timestamp = _timestamp(recorded_at)
        envelope = _entry_envelope(record_type, record_id, timestamp, payload)
        canonical_payload = canonicalize(envelope)
        record_hash = hashlib.sha256(
            bytes.fromhex(previous) + canonical_payload
        ).hexdigest()
        cursor = self._connection.execute(
            """
            INSERT INTO journal_records(
                record_id, record_type, recorded_at, canonical_payload,
                previous_record_hash, record_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                record_type,
                timestamp,
                canonical_payload,
                previous,
                record_hash,
            ),
        )
        for relationship, target_id in links:
            self._connection.execute(
                """
                INSERT INTO journal_links(source_record_id, relationship, target_record_id)
                VALUES (?, ?, ?)
                """,
                (record_id, relationship, target_id),
            )
        return JournalEntry(
            sequence=int(cursor.lastrowid),
            record_id=record_id,
            record_type=record_type,
            recorded_at=recorded_at,
            canonical_payload=canonical_payload,
            previous_record_hash=previous,
            record_hash=record_hash,
        )

    def append(
        self,
        record: Record,
        *,
        recorded_at: datetime | None = None,
        evidence_chain: EvidenceChain | None = None,
    ) -> JournalEntry:
        if type(record) not in RECORD_IDENTIFIERS:
            raise JournalValidationError("unsupported journal record type")
        moment = _utc(recorded_at or datetime.now(timezone.utc), "recorded_at")
        if moment < self._effective_at(record):
            raise JournalValidationError("recorded_at precedes the record's effective timestamp")
        record_id = str(getattr(record, RECORD_IDENTIFIERS[type(record)]))
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            payload = record.to_payload()
            if isinstance(record, SourceReceipt):
                self._registered_source(payload)
                self._reject_duplicate_source_content(record.content_sha256)
            links = self._links_for(record, evidence_chain)
            entry = self._insert(
                record_type=record.record_type,
                record_id=record_id,
                payload=payload,
                recorded_at=moment,
                links=links,
            )
            self._connection.commit()
            return entry
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, JournalValidationError):
                raise
            if isinstance(exc, sqlite3.IntegrityError):
                raise JournalValidationError(f"journal append rejected: {exc}") from exc
            raise

    def register_signing_key(
        self,
        *,
        key_id: str,
        public_key_bytes: bytes,
        valid_from: datetime,
        replaces_key_id: str | None = None,
        recorded_at: datetime | None = None,
    ) -> JournalEntry:
        if not isinstance(key_id, str) or not key_id.strip():
            raise JournalValidationError("key_id must be a non-empty string")
        if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
            raise JournalValidationError("Ed25519 public key must contain 32 bytes")
        activation = _utc(valid_from, "valid_from")
        registered = _utc(
            recorded_at or datetime.now(timezone.utc), "recorded_at"
        )
        rotation_id = f"signing-key-rotation:{key_id}"
        encoded_key = base64.b64encode(public_key_bytes).decode("ascii")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            links: list[tuple[str, str]] = []
            existing_keys = self._connection.execute(
                "SELECT key_id, valid_from FROM signing_keys"
            ).fetchall()
            current_key = max(
                existing_keys,
                key=lambda row: _parse_timestamp(row["valid_from"], "valid_from"),
                default=None,
            )
            if (
                current_key is not None
                and replaces_key_id != current_key["key_id"]
            ):
                raise JournalValidationError(
                    "a signing key rotation must replace the current signing key"
                )
            if replaces_key_id is not None:
                previous = self._connection.execute(
                    """
                    SELECT rotation_record_id, valid_from
                    FROM signing_keys WHERE key_id = ?
                    """,
                    (replaces_key_id,),
                ).fetchone()
                if previous is None:
                    raise JournalValidationError(
                        f"replaced signing key is not registered: {replaces_key_id}"
                    )
                previous_valid_from = _parse_timestamp(
                    previous["valid_from"], "valid_from"
                )
                if activation <= previous_valid_from:
                    raise JournalValidationError(
                        "rotated key valid_from must be after replaced key"
                    )
                links.append(("replaces_signing_key", previous["rotation_record_id"]))
            payload = {
                "key_id": key_id,
                "public_key_base64": encoded_key,
                "replaces_key_id": replaces_key_id,
                "valid_from": _timestamp(activation),
            }
            entry = self._insert(
                record_type="signing_key_rotation",
                record_id=rotation_id,
                payload=payload,
                recorded_at=registered,
                links=links,
            )
            self._connection.execute(
                """
                INSERT INTO signing_keys(
                    key_id, public_key_base64, valid_from,
                    replaces_key_id, rotation_record_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    encoded_key,
                    _timestamp(activation),
                    replaces_key_id,
                    rotation_id,
                ),
            )
            self._connection.commit()
            return entry
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, JournalValidationError):
                raise
            if isinstance(exc, sqlite3.IntegrityError):
                raise JournalValidationError(
                    f"signing key registration rejected: {exc}"
                ) from exc
            raise

    def verify_chain(self) -> bool:
        previous = GENESIS_RECORD_HASH
        rows = self._connection.execute(
            "SELECT * FROM journal_records ORDER BY sequence"
        ).fetchall()
        for row in rows:
            canonical_payload = bytes(row["canonical_payload"])
            try:
                envelope = json.loads(canonical_payload)
            except json.JSONDecodeError as exc:
                raise JournalValidationError("stored canonical payload is invalid JSON") from exc
            if canonicalize(envelope) != canonical_payload:
                raise JournalValidationError("stored payload is not canonical JCS")
            if row["previous_record_hash"] != previous:
                raise JournalValidationError("stored previous_record_hash breaks the chain")
            expected = hashlib.sha256(
                bytes.fromhex(previous) + canonical_payload
            ).hexdigest()
            if row["record_hash"] != expected:
                raise JournalValidationError("stored record_hash is invalid")
            previous = expected
        return True

    def _rows_for_period(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> list[sqlite3.Row]:
        rows = self._connection.execute(
            "SELECT * FROM journal_records ORDER BY sequence"
        ).fetchall()
        if start is None or end is None:
            return rows
        return [
            row
            for row in rows
            if start
            <= _parse_timestamp(row["recorded_at"], "recorded_at")
            < end
        ]

    def _jsonl_bytes(self, rows: list[sqlite3.Row] | None = None) -> bytes:
        lines = []
        selected_rows = self._rows_for_period() if rows is None else rows
        for row in selected_rows:
            envelope = json.loads(bytes(row["canonical_payload"]))
            exported = {
                **envelope,
                "previous_record_hash": row["previous_record_hash"],
                "record_hash": row["record_hash"],
                "sequence": row["sequence"],
            }
            lines.append(canonicalize(exported) + b"\n")
        return b"".join(lines)

    def export_jsonl(self, destination: Path) -> Path:
        self.verify_chain()
        path = Path(destination)
        _write_once(path, self._jsonl_bytes())
        verify_jsonl_export(path)
        return path

    def _registered_signer(
        self, signer: Ed25519DigestSigner, *, valid_at: datetime
    ) -> None:
        rows = self._connection.execute(
            "SELECT key_id, public_key_base64, valid_from FROM signing_keys"
        ).fetchall()
        eligible = [
            row
            for row in rows
            if _parse_timestamp(row["valid_from"], "valid_from") <= valid_at
        ]
        row = max(
            eligible,
            key=lambda item: _parse_timestamp(item["valid_from"], "valid_from"),
            default=None,
        )
        expected = base64.b64encode(signer.public_key_bytes).decode("ascii")
        if (
            row is None
            or row["key_id"] != signer.key_id
            or row["public_key_base64"] != expected
        ):
            raise JournalValidationError(
                "manifest signer must be the registered active signing key"
            )

    def export_signed_jsonl(
        self,
        destination: Path,
        *,
        period_start: datetime,
        period_end: datetime,
        signer: Ed25519DigestSigner,
    ) -> ExportBundle:
        start = _utc(period_start, "period_start")
        end = _utc(period_end, "period_end")
        if (
            start.hour != 0
            or start.minute != 0
            or start.second != 0
            or start.microsecond != 0
            or end != start + timedelta(days=1)
        ):
            raise JournalValidationError(
                "manifest period must be one complete UTC day"
            )
        self._registered_signer(signer, valid_at=end)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = path.with_name(f"{path.name}.manifest.json")
        signature_path = path.with_name(f"{path.name}.signature.json")
        for candidate in (path, manifest_path, signature_path):
            if candidate.exists():
                raise FileExistsError(candidate)
        self.verify_chain()
        all_rows = self._rows_for_period()
        rows_for_period = self._rows_for_period(start, end)
        raw = self._jsonl_bytes(rows_for_period)
        rows = [json.loads(line) for line in raw.splitlines()]
        earlier_rows = [
            row
            for row in all_rows
            if _parse_timestamp(row["recorded_at"], "recorded_at") < start
        ]
        previous_record_hash = (
            rows[0]["previous_record_hash"]
            if rows
            else (
                str(earlier_rows[-1]["record_hash"])
                if earlier_rows
                else GENESIS_RECORD_HASH
            )
        )
        manifest: dict[str, Any] = {
            "first_record_hash": rows[0]["record_hash"] if rows else None,
            "first_sequence": rows[0]["sequence"] if rows else None,
            "jsonl_filename": path.name,
            "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
            "last_record_hash": rows[-1]["record_hash"] if rows else None,
            "last_sequence": rows[-1]["sequence"] if rows else None,
            "manifest_version": 1,
            "period_end_utc": _timestamp(end),
            "period_start_utc": _timestamp(start),
            "previous_record_hash": previous_record_hash,
            "record_count": len(rows),
        }
        signature = sign_manifest(manifest, signer)
        with tempfile.TemporaryDirectory(
            prefix=f".{path.name}.bundle-", dir=path.parent
        ) as staging_directory:
            staging_root = Path(staging_directory)
            staging_jsonl = staging_root / path.name
            staging_manifest = staging_root / manifest_path.name
            staging_signature = staging_root / signature_path.name
            _write_once(staging_jsonl, raw)
            _write_once(staging_manifest, canonicalize(manifest) + b"\n")
            _write_once(staging_signature, canonicalize(signature) + b"\n")
            verify_signed_export_bundle(
                staging_jsonl,
                staging_manifest,
                staging_signature,
                expected_key_id=signer.key_id,
                expected_public_key_bytes=signer.public_key_bytes,
            )

            published: list[tuple[Path, int, int]] = []
            try:
                for staged, final in (
                    (staging_jsonl, path),
                    (staging_manifest, manifest_path),
                    (staging_signature, signature_path),
                ):
                    os.link(staged, final)
                    metadata = final.stat()
                    published.append((final, metadata.st_dev, metadata.st_ino))
            except Exception:
                for final, device, inode in reversed(published):
                    try:
                        metadata = final.stat()
                        if (metadata.st_dev, metadata.st_ino) == (device, inode):
                            final.unlink()
                    except FileNotFoundError:
                        pass
                raise
        try:
            verify_signed_export_bundle(
                path,
                manifest_path,
                signature_path,
                expected_key_id=signer.key_id,
                expected_public_key_bytes=signer.public_key_bytes,
            )
        except Exception:
            for final, device, inode in reversed(published):
                try:
                    metadata = final.stat()
                    if (metadata.st_dev, metadata.st_ino) == (device, inode):
                        final.unlink()
                except FileNotFoundError:
                    pass
            raise
        return ExportBundle(
            jsonl_path=path,
            manifest_path=manifest_path,
            signature_path=signature_path,
            manifest=manifest,
            signature=signature,
        )
