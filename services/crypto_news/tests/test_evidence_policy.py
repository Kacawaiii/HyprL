from datetime import datetime, timedelta, timezone

from crypto_news.evidence import EvidenceCandidate, build_evidence_chain
from crypto_news.models import FactStatus
from crypto_news.sources import DEFAULT_SOURCE_REGISTRY


UTC = timezone.utc
BASE = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def candidate(
    receipt_id: str,
    source_id: str,
    offset: int = 0,
    claim_key: str = "btc-etf-order",
) -> EvidenceCandidate:
    return EvidenceCandidate(
        receipt_id=receipt_id,
        source_id=source_id,
        claim_key=claim_key,
        first_seen_at=BASE + timedelta(seconds=offset),
    )


def test_tier_zero_source_builds_a_primary_confirmed_chain() -> None:
    chain = build_evidence_chain(
        (candidate("sec-1", "sec_press_releases"),),
        DEFAULT_SOURCE_REGISTRY,
    )
    assert chain.fact_status is FactStatus.PRIMARY_CONFIRMED
    assert chain.primary_source_receipt_id == "sec-1"
    assert chain.source_receipt_ids == ("sec-1",)
    assert chain.source_ids == ("sec_press_releases",)
    assert chain.claim_key == "btc-etf-order"
    assert chain.promotable is True
    assert chain.rejection_reason is None


def test_two_independent_secondary_sources_are_corroborated() -> None:
    chain = build_evidence_chain(
        (
            candidate("media-1", "reuters", 1),
            candidate("media-2", "coindesk", 2),
        ),
        DEFAULT_SOURCE_REGISTRY,
    )
    assert chain.fact_status is FactStatus.CORROBORATED
    assert chain.primary_source_receipt_id is None
    assert chain.promotable is True


def test_single_secondary_or_tier_three_rumor_is_rejected() -> None:
    for source_id in ("reuters", "social_unverified"):
        chain = build_evidence_chain(
            (candidate(f"{source_id}-1", source_id),),
            DEFAULT_SOURCE_REGISTRY,
        )
        assert chain.fact_status is FactStatus.UNVERIFIED
        assert chain.promotable is False
        assert chain.rejection_reason == "insufficient independent corroboration"


def test_duplicate_sources_do_not_count_as_independent_corroboration() -> None:
    chain = build_evidence_chain(
        (
            candidate("media-1", "reuters", 1),
            candidate("media-2", "reuters", 2),
        ),
        DEFAULT_SOURCE_REGISTRY,
    )
    assert chain.fact_status is FactStatus.UNVERIFIED
    assert chain.promotable is False


def test_unrelated_claims_cannot_be_combined_as_corroboration() -> None:
    import pytest

    with pytest.raises(ValueError, match="same claim_key"):
        build_evidence_chain(
            (
                candidate("media-1", "reuters", claim_key="claim-a"),
                candidate("media-2", "coindesk", claim_key="claim-b"),
            ),
            DEFAULT_SOURCE_REGISTRY,
        )
