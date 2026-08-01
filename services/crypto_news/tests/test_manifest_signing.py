import base64
import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from datetime import timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import crypto_news.journal as journal_module
from crypto_news.jcs import canonicalize
from crypto_news.journal import (
    Journal,
    JournalValidationError,
    verify_jsonl_export,
    verify_signed_export_bundle,
)
from crypto_news.signing import (
    SigningValidationError,
    sign_manifest,
    verify_manifest_signature,
)
from tests.test_models import BASE, source_receipt


DAY_START = BASE.replace(hour=0, minute=0, second=0, microsecond=0)


class EphemeralSigner:
    """Test-only signer; production code receives only this digest interface."""

    def __init__(self, key_id: str) -> None:
        self.key_id = key_id
        self._private_key = Ed25519PrivateKey.generate()
        self.received_digests: list[bytes] = []

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign_digest(self, digest: bytes) -> bytes:
        self.received_digests.append(digest)
        return self._private_key.sign(digest)


class FailingSigner:
    def __init__(self, delegate: EphemeralSigner) -> None:
        self.key_id = delegate.key_id
        self.public_key_bytes = delegate.public_key_bytes

    def sign_digest(self, _digest: bytes) -> bytes:
        raise RuntimeError("injected signing failure")


def test_signed_export_uses_registered_public_key_and_digest_only(tmp_path) -> None:
    signer = EphemeralSigner("daily-key-1")
    database = tmp_path / "journal.sqlite3"
    destination = tmp_path / "events.jsonl"

    with Journal(database) as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        bundle = journal.export_signed_jsonl(
            destination,
            period_start=DAY_START,
            period_end=DAY_START + timedelta(days=1),
            signer=signer,
        )

        assert len(signer.received_digests) == 1
        assert len(signer.received_digests[0]) == 32
        assert bundle.manifest["record_count"] == 2
        assert verify_manifest_signature(
            bundle.manifest,
            bundle.signature,
            expected_key_id=signer.key_id,
            expected_public_key_bytes=signer.public_key_bytes,
        ) is True
        assert verify_signed_export_bundle(
            bundle.jsonl_path,
            bundle.manifest_path,
            bundle.signature_path,
            expected_key_id=signer.key_id,
            expected_public_key_bytes=signer.public_key_bytes,
        ) is True
        assert verify_jsonl_export(destination) is True
        assert "private" not in json.dumps(bundle.signature).lower()
        assert bundle.signature["key_id"] == "daily-key-1"
        assert base64.b64decode(bundle.signature["public_key_base64"]) == signer.public_key_bytes

        with pytest.raises(FileExistsError):
            journal.export_signed_jsonl(
                destination,
                period_start=DAY_START,
                period_end=DAY_START + timedelta(days=1),
                signer=signer,
            )


def test_signing_failure_leaves_no_partial_bundle_and_retry_succeeds(tmp_path) -> None:
    signer = EphemeralSigner("daily-key-1")
    destination = tmp_path / "events.jsonl"
    manifest = tmp_path / "events.jsonl.manifest.json"
    signature = tmp_path / "events.jsonl.signature.json"

    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))

        with pytest.raises(RuntimeError, match="injected signing failure"):
            journal.export_signed_jsonl(
                destination,
                period_start=DAY_START,
                period_end=DAY_START + timedelta(days=1),
                signer=FailingSigner(signer),
            )

        assert not destination.exists()
        assert not manifest.exists()
        assert not signature.exists()

        bundle = journal.export_signed_jsonl(
            destination,
            period_start=DAY_START,
            period_end=DAY_START + timedelta(days=1),
            signer=signer,
        )
        assert verify_signed_export_bundle(
            bundle.jsonl_path,
            bundle.manifest_path,
            bundle.signature_path,
            expected_key_id=signer.key_id,
            expected_public_key_bytes=signer.public_key_bytes,
        ) is True


def test_final_verification_failure_cleans_bundle_and_allows_retry(
    tmp_path, monkeypatch
) -> None:
    signer = EphemeralSigner("daily-key-1")
    destination = tmp_path / "events.jsonl"
    manifest = tmp_path / "events.jsonl.manifest.json"
    signature = tmp_path / "events.jsonl.signature.json"
    real_verify = journal_module.verify_signed_export_bundle
    calls = 0

    def fail_second_verification(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected final verification failure")
        return real_verify(*args, **kwargs)

    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        monkeypatch.setattr(
            journal_module,
            "verify_signed_export_bundle",
            fail_second_verification,
        )
        with pytest.raises(RuntimeError, match="final verification failure"):
            journal.export_signed_jsonl(
                destination,
                period_start=DAY_START,
                period_end=DAY_START + timedelta(days=1),
                signer=signer,
            )
        assert not destination.exists()
        assert not manifest.exists()
        assert not signature.exists()

        monkeypatch.setattr(
            journal_module,
            "verify_signed_export_bundle",
            real_verify,
        )
        journal.export_signed_jsonl(
            destination,
            period_start=DAY_START,
            period_end=DAY_START + timedelta(days=1),
            signer=signer,
        )


def test_unregistered_signer_is_rejected(tmp_path) -> None:
    signer = EphemeralSigner("unknown-key")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        with pytest.raises(JournalValidationError, match="registered"):
            journal.export_signed_jsonl(
                tmp_path / "events.jsonl",
                period_start=DAY_START,
                period_end=DAY_START + timedelta(days=1),
                signer=signer,
            )


def test_key_rotation_is_linked_and_insert_only(tmp_path) -> None:
    first = EphemeralSigner("daily-key-1")
    second = EphemeralSigner("daily-key-2")
    path = tmp_path / "journal.sqlite3"

    with Journal(path) as journal:
        journal.register_signing_key(
            key_id=first.key_id,
            public_key_bytes=first.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.register_signing_key(
            key_id=second.key_id,
            public_key_bytes=second.public_key_bytes,
            valid_from=BASE + timedelta(days=1),
            replaces_key_id=first.key_id,
            recorded_at=BASE + timedelta(seconds=1),
        )
        assert journal.count_records() == 2
        assert journal.verify_chain() is True

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
            connection.execute(
                "UPDATE signing_keys SET key_id = 'changed' WHERE key_id = 'daily-key-1'"
            )


def test_future_key_activation_does_not_advance_the_journal_clock(tmp_path) -> None:
    first = EphemeralSigner("daily-key-1")
    second = EphemeralSigner("daily-key-2")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=first.key_id,
            public_key_bytes=first.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.register_signing_key(
            key_id=second.key_id,
            public_key_bytes=second.public_key_bytes,
            valid_from=BASE + timedelta(days=1),
            replaces_key_id=first.key_id,
            recorded_at=BASE + timedelta(seconds=1),
        )

        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        assert journal.count_records() == 3
        assert journal.verify_chain() is True


def test_export_tampering_breaks_hash_chain(tmp_path) -> None:
    path = tmp_path / "journal.sqlite3"
    export = tmp_path / "events.jsonl"
    with Journal(path) as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        journal.export_jsonl(export)

    payload = json.loads(export.read_text(encoding="utf-8"))
    payload["payload"]["source_name"] = "tampered"
    export.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(JournalValidationError, match="record_hash"):
        verify_jsonl_export(export)


def test_standalone_jsonl_verifier_rejects_a_truncated_genesis_prefix(tmp_path) -> None:
    complete = tmp_path / "complete.jsonl"
    truncated = tmp_path / "truncated.jsonl"
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        second = replace(
            source_receipt(),
            receipt_id="receipt-2",
            content_sha256="b" * 64,
        )
        journal.append(second, recorded_at=BASE + timedelta(seconds=4))
        journal.export_jsonl(complete)

    truncated.write_bytes(complete.read_bytes().splitlines(keepends=True)[1])
    with pytest.raises(JournalValidationError, match="sequence|previous_record_hash"):
        verify_jsonl_export(truncated)


def test_manifest_period_must_be_one_complete_utc_day(tmp_path) -> None:
    signer = EphemeralSigner("daily-key-1")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        with pytest.raises(JournalValidationError, match="complete UTC day"):
            journal.export_signed_jsonl(
                tmp_path / "events.jsonl",
                period_start=BASE,
                period_end=BASE + timedelta(days=1),
                signer=signer,
            )


def test_key_rotation_validity_must_move_forward(tmp_path) -> None:
    first = EphemeralSigner("daily-key-1")
    second = EphemeralSigner("daily-key-2")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=first.key_id,
            public_key_bytes=first.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        with pytest.raises(JournalValidationError, match="after replaced key"):
            journal.register_signing_key(
                key_id=second.key_id,
                public_key_bytes=second.public_key_bytes,
                valid_from=BASE - timedelta(seconds=1),
                replaces_key_id=first.key_id,
                recorded_at=BASE + timedelta(seconds=1),
            )


def test_manifest_verification_rejects_attacker_controlled_trust_root() -> None:
    trusted = EphemeralSigner("trusted-key")
    attacker = EphemeralSigner("attacker-key")
    manifest = {"manifest_version": 1, "record_count": 0}
    signature = sign_manifest(manifest, attacker)

    with pytest.raises(SigningValidationError, match="key_id"):
        verify_manifest_signature(
            manifest,
            signature,
            expected_key_id=trusted.key_id,
            expected_public_key_bytes=trusted.public_key_bytes,
        )


def test_signed_export_contains_only_records_from_requested_utc_day(tmp_path) -> None:
    signer = EphemeralSigner("daily-key-1")
    day_two_receipt = replace(
        source_receipt(),
        receipt_id="receipt-day-two",
        published_at=BASE + timedelta(days=1),
        first_seen_at=BASE + timedelta(days=1, seconds=2),
        retrieved_at=BASE + timedelta(days=1, seconds=3),
        content_sha256="b" * 64,
    )

    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.append(
            day_two_receipt,
            recorded_at=BASE + timedelta(days=1, seconds=3),
        )
        bundle = journal.export_signed_jsonl(
            tmp_path / "day-one.jsonl",
            period_start=DAY_START,
            period_end=DAY_START + timedelta(days=1),
            signer=signer,
        )

    rows = [json.loads(line) for line in bundle.jsonl_path.read_text().splitlines()]
    assert bundle.manifest["record_count"] == 1
    assert all(
        DAY_START
        <= datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00"))
        < DAY_START + timedelta(days=1)
        for row in rows
    )


def test_rotated_key_prevents_old_signer_for_later_period(tmp_path) -> None:
    first = EphemeralSigner("daily-key-1")
    second = EphemeralSigner("daily-key-2")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=first.key_id,
            public_key_bytes=first.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.register_signing_key(
            key_id=second.key_id,
            public_key_bytes=second.public_key_bytes,
            valid_from=BASE + timedelta(days=1),
            replaces_key_id=first.key_id,
            recorded_at=BASE + timedelta(seconds=1),
        )

        with pytest.raises(JournalValidationError, match="active signing key"):
            journal.export_signed_jsonl(
                tmp_path / "day-two.jsonl",
                period_start=DAY_START + timedelta(days=1),
                period_end=DAY_START + timedelta(days=2),
                signer=first,
            )


def test_signing_key_rotations_form_one_linear_chain(tmp_path) -> None:
    first = EphemeralSigner("daily-key-1")
    second = EphemeralSigner("daily-key-2")
    third = EphemeralSigner("daily-key-3")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=first.key_id,
            public_key_bytes=first.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        with pytest.raises(JournalValidationError, match="replace the current"):
            journal.register_signing_key(
                key_id=second.key_id,
                public_key_bytes=second.public_key_bytes,
                valid_from=BASE + timedelta(days=1),
                recorded_at=BASE + timedelta(seconds=1),
            )

        journal.register_signing_key(
            key_id=second.key_id,
            public_key_bytes=second.public_key_bytes,
            valid_from=BASE + timedelta(days=1),
            replaces_key_id=first.key_id,
            recorded_at=BASE + timedelta(seconds=1),
        )
        with pytest.raises(JournalValidationError, match="replace the current"):
            journal.register_signing_key(
                key_id=third.key_id,
                public_key_bytes=third.public_key_bytes,
                valid_from=BASE + timedelta(days=2),
                replaces_key_id=first.key_id,
                recorded_at=BASE + timedelta(seconds=2),
            )


def test_bundle_verifier_rejects_non_midnight_daily_period(tmp_path) -> None:
    signer = EphemeralSigner("daily-key-1")
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.register_signing_key(
            key_id=signer.key_id,
            public_key_bytes=signer.public_key_bytes,
            valid_from=BASE,
            recorded_at=BASE,
        )
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        bundle = journal.export_signed_jsonl(
            tmp_path / "events.jsonl",
            period_start=DAY_START,
            period_end=DAY_START + timedelta(days=1),
            signer=signer,
        )

    shifted_manifest = {
        **bundle.manifest,
        "period_start_utc": BASE.isoformat().replace("+00:00", "Z"),
        "period_end_utc": (BASE + timedelta(days=1))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    shifted_signature = sign_manifest(shifted_manifest, signer)
    bundle.manifest_path.write_bytes(canonicalize(shifted_manifest) + b"\n")
    bundle.signature_path.write_bytes(canonicalize(shifted_signature) + b"\n")

    with pytest.raises(JournalValidationError, match="complete UTC day"):
        verify_signed_export_bundle(
            bundle.jsonl_path,
            bundle.manifest_path,
            bundle.signature_path,
            expected_key_id=signer.key_id,
            expected_public_key_bytes=signer.public_key_bytes,
        )
