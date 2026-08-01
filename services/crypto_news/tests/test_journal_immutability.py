import hashlib
import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from crypto_news.evidence import EvidenceCandidate, build_evidence_chain
from crypto_news.journal import Journal, JournalValidationError
from crypto_news.models import (
    Asset,
    EventType,
    FactStatus,
    HumanDecision,
    HumanDecisionAction,
    Outcome,
    OutcomeAnchor,
    OutcomeHorizon,
    Retraction,
)
from tests.test_models import BASE, analysis, event_version, market_snapshot, source_receipt
from crypto_news.sources import (
    DEFAULT_SOURCE_REGISTRY,
    SourceDefinition,
    SourceRegistry,
)


def append_analysis_graph(journal: Journal) -> None:
    journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
    journal.append(event_version(), recorded_at=BASE + timedelta(seconds=4))
    journal.append(market_snapshot(), recorded_at=BASE + timedelta(seconds=6))
    journal.append(analysis(), recorded_at=BASE + timedelta(seconds=7))


def test_chain_uses_raw_previous_digest_plus_jcs_payload(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        first = journal.append(
            source_receipt(), recorded_at=BASE + timedelta(seconds=3)
        )
        second = journal.append(
            event_version(), recorded_at=BASE + timedelta(seconds=4)
        )

        assert first.previous_record_hash == "0" * 64
        assert first.record_hash == hashlib.sha256(
            bytes.fromhex(first.previous_record_hash) + first.canonical_payload
        ).hexdigest()
        assert second.previous_record_hash == first.record_hash
        assert journal.verify_chain() is True


def test_sqlite_records_links_and_keys_are_insert_only(tmp_path) -> None:
    path = tmp_path / "journal.sqlite3"
    with Journal(path) as journal:
        append_analysis_graph(journal)

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
            connection.execute(
                "UPDATE journal_records SET record_type = 'changed' WHERE sequence = 1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
            connection.execute("DELETE FROM journal_links")


def test_journal_requires_existing_and_causally_valid_links(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        with pytest.raises(JournalValidationError, match="source_receipt"):
            journal.append(event_version(), recorded_at=BASE + timedelta(seconds=4))

        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        journal.append(event_version(), recorded_at=BASE + timedelta(seconds=4))

        simultaneous = replace(
            market_snapshot(),
            snapshot_id="snapshot-simultaneous",
            observed_at=BASE + timedelta(seconds=2),
            captured_at=BASE + timedelta(seconds=2),
        )
        with pytest.raises(JournalValidationError, match="strictly after first_seen_at"):
            journal.append(simultaneous, recorded_at=BASE + timedelta(seconds=2))

        journal.append(market_snapshot(), recorded_at=BASE + timedelta(seconds=6))
        with pytest.raises(JournalValidationError, match="frozen_at"):
            journal.append(
                replace(analysis(), frozen_at=BASE + timedelta(seconds=5)),
                recorded_at=BASE + timedelta(seconds=7),
            )


def test_all_v0_record_types_can_be_linked_without_mutation(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        append_analysis_graph(journal)
        decision = HumanDecision(
            decision_id="decision-1",
            analysis_id="analysis-1",
            action=HumanDecisionAction.WAIT,
            human_decided_at=BASE + timedelta(minutes=1),
            actor_ref="reviewer-pseudonym",
            rationale="Awaiting corroboration.",
        )
        journal.append(decision, recorded_at=BASE + timedelta(minutes=1))
        ai_outcome = Outcome(
            outcome_id="outcome-ai-1",
            analysis_id="analysis-1",
            human_decision_id=None,
            anchor=OutcomeAnchor.AI,
            anchor_at=BASE + timedelta(seconds=7),
            entry_price_at=BASE + timedelta(seconds=67),
            horizon=OutcomeHorizon.FIVE_MINUTES,
            horizon_at=BASE + timedelta(minutes=5, seconds=7),
            observed_at=BASE + timedelta(minutes=5, seconds=8),
            asset=Asset.BTC,
            venue="coinbase",
            entry_price_usd="65000.00",
            exit_price_usd="65100.00",
            spread_bps="1.25",
            fees_bps="2.00",
            depth_usd="250000.00",
        )
        journal.append(ai_outcome, recorded_at=BASE + timedelta(minutes=5, seconds=8))
        human_outcome = replace(
            ai_outcome,
            outcome_id="outcome-human-1",
            human_decision_id="decision-1",
            anchor=OutcomeAnchor.HUMAN,
            anchor_at=BASE + timedelta(minutes=1),
            entry_price_at=BASE + timedelta(minutes=1, seconds=30),
            horizon_at=BASE + timedelta(minutes=6),
            observed_at=BASE + timedelta(minutes=6, seconds=1),
        )
        journal.append(
            human_outcome, recorded_at=BASE + timedelta(minutes=6, seconds=1)
        )
        retraction = Retraction(
            retraction_id="retraction-1",
            event_version_id="event-version-1",
            source_receipt_id="receipt-1",
            retracted_at=BASE + timedelta(hours=1),
            recorded_at=BASE + timedelta(hours=1, seconds=1),
            reason="Primary source retracted the statement.",
        )
        journal.append(
            retraction, recorded_at=BASE + timedelta(hours=1, seconds=1)
        )

        assert journal.count_records() == 8
        assert journal.verify_chain() is True


def test_event_version_preserves_first_seen_timestamp(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        journal.append(event_version(), recorded_at=BASE + timedelta(seconds=4))
        second = event_version(
            version_id="event-version-2",
            version_number=2,
            previous_version_id="event-version-1",
            first_seen_at=BASE + timedelta(seconds=3),
            created_at=BASE + timedelta(minutes=2),
        )
        with pytest.raises(JournalValidationError, match="first_seen_at is immutable"):
            journal.append(second, recorded_at=BASE + timedelta(minutes=2))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fact_status": FactStatus.CORROBORATED}, "fact_status"),
        ({"event_type": EventType.PROTOCOL}, "event_type"),
        ({"affected_assets": (Asset.BTC,)}, "affected_assets"),
        ({"published_at": BASE - timedelta(seconds=1)}, "published_at"),
        (
            {"corroborating_source_urls": ("https://example.com/unrecorded",)},
            "corroborating source",
        ),
    ],
)
def test_analysis_must_match_its_journaled_event_and_sources(
    tmp_path, changes, message
) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        journal.append(event_version(), recorded_at=BASE + timedelta(seconds=4))
        journal.append(market_snapshot(), recorded_at=BASE + timedelta(seconds=6))

        with pytest.raises(JournalValidationError, match=message):
            journal.append(
                replace(analysis(), **changes),
                recorded_at=BASE + timedelta(seconds=7),
            )


def test_journal_rejects_backdated_append_after_a_later_record(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=10))

        second = replace(
            source_receipt(),
            receipt_id="receipt-2",
            content_sha256="b" * 64,
        )
        with pytest.raises(JournalValidationError, match="non-decreasing"):
            journal.append(second, recorded_at=BASE + timedelta(seconds=9))


def test_journal_rejects_a_fabricated_tier_zero_source_receipt(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        forged = replace(
            source_receipt(),
            source_name="Attacker-controlled official source",
        )
        with pytest.raises(JournalValidationError, match="registered source"):
            journal.append(forged, recorded_at=BASE + timedelta(seconds=3))

        impersonated = replace(
            source_receipt(),
            canonical_url="https://attacker.example/fake-sec-release",
        )
        with pytest.raises(JournalValidationError, match="URL host"):
            journal.append(impersonated, recorded_at=BASE + timedelta(seconds=3))

        secondary_impersonation = replace(
            source_receipt(),
            source_name="Reuters",
            source_tier=1,
            canonical_url="https://attacker.example/pretend-reuters",
        )
        with pytest.raises(JournalValidationError, match="URL host"):
            journal.append(
                secondary_impersonation,
                recorded_at=BASE + timedelta(seconds=3),
            )


def test_journal_rejects_duplicate_source_content_under_an_alias(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(source_receipt(), recorded_at=BASE + timedelta(seconds=3))
        alias = replace(
            source_receipt(),
            receipt_id="receipt-alias",
            source_name="Reuters",
            source_tier=1,
            canonical_url="https://www.reuters.com/world/example",
        )
        with pytest.raises(JournalValidationError, match="content_sha256"):
            journal.append(alias, recorded_at=BASE + timedelta(seconds=3))


def test_corroborated_event_requires_two_distinct_tier_zero_to_two_sources(
    tmp_path,
) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        lead = replace(
            source_receipt(),
            source_name="Reuters",
            source_tier=1,
            canonical_url="https://www.reuters.com/world/example",
        )
        journal.append(lead, recorded_at=BASE + timedelta(seconds=3))
        unconfirmed = replace(
            event_version(),
            fact_status=FactStatus.CORROBORATED,
            primary_source_receipt_id=None,
        )
        with pytest.raises(JournalValidationError, match="two distinct"):
            journal.append(unconfirmed, recorded_at=BASE + timedelta(seconds=4))


def test_corroborated_event_requires_a_matching_claim_chain(tmp_path) -> None:
    first = replace(
        source_receipt(),
        source_name="Reuters",
        source_tier=1,
        canonical_url="https://www.reuters.com/world/example",
    )
    second = replace(
        source_receipt(),
        receipt_id="receipt-2",
        canonical_url="https://www.coindesk.com/markets/example-2",
        source_name="CoinDesk",
        source_tier=2,
        content_sha256="b" * 64,
    )
    event = replace(
        event_version(),
        source_receipt_ids=(first.receipt_id, second.receipt_id),
        primary_source_receipt_id=None,
        fact_status=FactStatus.CORROBORATED,
    )
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(first, recorded_at=BASE + timedelta(seconds=3))
        journal.append(second, recorded_at=BASE + timedelta(seconds=3))
        with pytest.raises(JournalValidationError, match="evidence chain"):
            journal.append(event, recorded_at=BASE + timedelta(seconds=4))

        wrong_claim = build_evidence_chain(
            (
                EvidenceCandidate(first.receipt_id, "reuters", "other-event", first.first_seen_at),
                EvidenceCandidate(second.receipt_id, "coindesk", "other-event", second.first_seen_at),
            ),
            DEFAULT_SOURCE_REGISTRY,
        )
        with pytest.raises(JournalValidationError, match="claim_key"):
            journal.append(
                event,
                recorded_at=BASE + timedelta(seconds=4),
                evidence_chain=wrong_claim,
            )

        chain = build_evidence_chain(
            (
                EvidenceCandidate(first.receipt_id, "reuters", event.event_id, first.first_seen_at),
                EvidenceCandidate(second.receipt_id, "coindesk", event.event_id, second.first_seen_at),
            ),
            DEFAULT_SOURCE_REGISTRY,
        )
        journal.append(
            event,
            recorded_at=BASE + timedelta(seconds=4),
            evidence_chain=chain,
        )


def test_primary_event_rejects_a_supplied_chain_for_another_claim(tmp_path) -> None:
    receipt = source_receipt()
    wrong_claim = build_evidence_chain(
        (
            EvidenceCandidate(
                receipt.receipt_id,
                "sec_press_releases",
                "other-event",
                receipt.first_seen_at,
            ),
        ),
        DEFAULT_SOURCE_REGISTRY,
    )
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.append(receipt, recorded_at=BASE + timedelta(seconds=3))
        with pytest.raises(JournalValidationError, match="claim_key"):
            journal.append(
                event_version(),
                recorded_at=BASE + timedelta(seconds=4),
                evidence_chain=wrong_claim,
            )


def test_corroboration_uses_registry_independence_groups_not_source_names(
    tmp_path,
) -> None:
    registry = SourceRegistry(
        (
            SourceDefinition(
                source_id="wire_alias_a",
                name="Wire Alias A",
                tier=1,
                independence_group="same_wire_owner",
                canonical_hosts=("wire-a.example",),
            ),
            SourceDefinition(
                source_id="wire_alias_b",
                name="Wire Alias B",
                tier=2,
                independence_group="same_wire_owner",
                canonical_hosts=("wire-b.example",),
            ),
        )
    )
    first = replace(
        source_receipt(),
        source_name="Wire Alias A",
        source_tier=1,
        canonical_url="https://wire-a.example/item",
    )
    second = replace(
        source_receipt(),
        receipt_id="receipt-2",
        source_name="Wire Alias B",
        source_tier=2,
        canonical_url="https://wire-b.example/item",
        content_sha256="b" * 64,
    )
    event = replace(
        event_version(),
        source_receipt_ids=(first.receipt_id, second.receipt_id),
        primary_source_receipt_id=None,
        fact_status=FactStatus.CORROBORATED,
    )

    with Journal(tmp_path / "journal.sqlite3", source_registry=registry) as journal:
        journal.append(first, recorded_at=BASE + timedelta(seconds=3))
        journal.append(second, recorded_at=BASE + timedelta(seconds=3))
        with pytest.raises(JournalValidationError, match="two distinct"):
            journal.append(event, recorded_at=BASE + timedelta(seconds=4))


def test_unexpected_append_failure_rolls_back_the_transaction(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        invalid = source_receipt()
        object.__setattr__(invalid, "author", 1.5)
        with pytest.raises(TypeError, match="float values are forbidden"):
            journal.append(invalid, recorded_at=BASE + timedelta(seconds=3))

        valid = replace(
            source_receipt(),
            receipt_id="receipt-after-failure",
            content_sha256="b" * 64,
        )
        journal.append(valid, recorded_at=BASE + timedelta(seconds=4))
        assert journal.count_records() == 1
        assert journal.verify_chain() is True
