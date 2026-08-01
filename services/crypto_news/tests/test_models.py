from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from crypto_news.models import (
    Analysis,
    Asset,
    Direction,
    EventType,
    EventVersion,
    FactStatus,
    HumanDecision,
    HumanDecisionAction,
    MarketConfirmation,
    MarketSnapshot,
    ModelValidationError,
    Novelty,
    Outcome,
    OutcomeAnchor,
    OutcomeHorizon,
    Playbook,
    ResearchDecision,
    Retraction,
    SourceReceipt,
    Surprise,
    TopRiskStatus,
)
from crypto_news.protocol import PROTOCOL_V0_SHA256


UTC = timezone.utc
BASE = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def source_receipt(**changes: object) -> SourceReceipt:
    values: dict[str, object] = {
        "receipt_id": "receipt-1",
        "canonical_url": "https://www.sec.gov/news/example",
        "source_name": "U.S. Securities and Exchange Commission",
        "source_tier": 0,
        "published_at": BASE,
        "first_seen_at": BASE + timedelta(seconds=2),
        "retrieved_at": BASE + timedelta(seconds=3),
        "content_sha256": "a" * 64,
        "author": None,
    }
    values.update(changes)
    return SourceReceipt(**values)


def event_version(**changes: object) -> EventVersion:
    values: dict[str, object] = {
        "version_id": "event-version-1",
        "event_id": "event-1",
        "version_number": 1,
        "previous_version_id": None,
        "source_receipt_ids": ("receipt-1",),
        "primary_source_receipt_id": "receipt-1",
        "fact_status": FactStatus.PRIMARY_CONFIRMED,
        "event_type": EventType.REGULATION_LITIGATION,
        "affected_assets": (Asset.BTC, Asset.ETH),
        "first_seen_at": BASE + timedelta(seconds=2),
        "created_at": BASE + timedelta(seconds=4),
        "summary": "A verified regulatory event.",
    }
    values.update(changes)
    return EventVersion(**values)


def market_snapshot(**changes: object) -> MarketSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "snapshot-1",
        "event_version_id": "event-version-1",
        "asset": Asset.BTC,
        "venue": "coinbase",
        "observed_at": BASE + timedelta(seconds=5),
        "captured_at": BASE + timedelta(seconds=6),
        "last_price_usd": "65000.25",
        "bid_usd": "65000.20",
        "ask_usd": "65000.30",
        "depth_usd": "250000.00",
    }
    values.update(changes)
    return MarketSnapshot(**values)


def analysis(**changes: object) -> Analysis:
    values: dict[str, object] = {
        "analysis_id": "analysis-1",
        "event_version_id": "event-version-1",
        "market_snapshot_ids": ("snapshot-1",),
        "protocol_sha256": PROTOCOL_V0_SHA256,
        "fact_status": FactStatus.PRIMARY_CONFIRMED,
        "primary_source_url": "https://www.sec.gov/news/example",
        "corroborating_source_urls": (),
        "published_at": BASE,
        "first_seen_at": BASE + timedelta(seconds=2),
        "frozen_at": BASE + timedelta(seconds=7),
        "event_type": EventType.REGULATION_LITIGATION,
        "affected_assets": (Asset.BTC, Asset.ETH),
        "event_quality_score": 84,
        "direction": Direction.BEARISH,
        "horizon": "24h",
        "novelty": Novelty.HIGH,
        "surprise": Surprise.UNKNOWN,
        "priced_in_score": 35,
        "priced_in_data_coverage": 80,
        "market_confirmation": MarketConfirmation.PARTIAL,
        "top_risk_score": None,
        "top_risk_status": TopRiskStatus.UNKNOWN,
        "playbook": Playbook.SELL_THE_NEWS,
        "decision": ResearchDecision.WAIT,
        "trade_opportunity_score": 58,
        "thesis": "The event may already be partially reflected in price.",
        "invalidation": "Price closes above the frozen invalidation level.",
        "entry_zone": None,
        "stop_level": None,
        "target_zones": (),
        "max_holding_time": "24h",
        "risk_budget_bps": 25,
        "expires_at": BASE + timedelta(hours=24),
        "missing_evidence": ("multi-venue funding",),
    }
    values.update(changes)
    return Analysis(**values)


def test_source_receipt_is_immutable_and_causally_timestamped() -> None:
    receipt = source_receipt()
    with pytest.raises(FrozenInstanceError):
        receipt.source_name = "changed"  # type: ignore[misc]

    with pytest.raises(ModelValidationError, match="source_tier"):
        source_receipt(source_tier=0.0)

    with pytest.raises(ModelValidationError, match="published_at"):
        source_receipt(first_seen_at=BASE - timedelta(seconds=1))
    with pytest.raises(ModelValidationError, match="UTC"):
        source_receipt(published_at=BASE.replace(tzinfo=None))


def test_event_versions_are_linked_and_limited_to_btc_eth() -> None:
    assert event_version().affected_assets == (Asset.BTC, Asset.ETH)

    with pytest.raises(ModelValidationError, match="previous_version_id"):
        event_version(version_number=2)
    with pytest.raises(ModelValidationError, match="positive integer"):
        event_version(version_number=1.5)
    with pytest.raises(ModelValidationError, match="affected_assets"):
        event_version(affected_assets=("SOL",))
    with pytest.raises(ModelValidationError, match="primary_source_receipt_id"):
        event_version(primary_source_receipt_id="receipt-2")


def test_market_snapshot_uses_exact_decimal_strings_and_closed_timestamps() -> None:
    snapshot = market_snapshot()
    assert snapshot.last_price_usd == "65000.25"

    with pytest.raises(ModelValidationError, match="float"):
        market_snapshot(last_price_usd=65000.25)
    with pytest.raises(ModelValidationError, match="bid_usd"):
        market_snapshot(bid_usd="65001", ask_usd="65000")
    with pytest.raises(ModelValidationError, match="captured_at"):
        market_snapshot(captured_at=BASE + timedelta(seconds=4))


def test_analysis_enforces_ordinal_fields_and_clock_name() -> None:
    item = analysis()
    assert item.priced_in_data_coverage == 80
    assert "frozen_at" in item.to_payload()

    with pytest.raises(TypeError):
        Analysis(**(item.to_payload() | {"t_freeze": BASE}))
    with pytest.raises(ModelValidationError, match="event_quality_score"):
        analysis(event_quality_score=True)
    with pytest.raises(ModelValidationError, match="priced_in_data_coverage"):
        analysis(priced_in_data_coverage=101)
    with pytest.raises(ModelValidationError, match="top_risk_score"):
        analysis(top_risk_score=70, top_risk_status=TopRiskStatus.UNKNOWN)
    with pytest.raises(ModelValidationError, match="invalidation"):
        analysis(decision=ResearchDecision.LONG, invalidation="")
    with pytest.raises(ModelValidationError, match="expires_at"):
        analysis(expires_at=BASE + timedelta(seconds=7))


def test_human_decision_and_retraction_are_research_records() -> None:
    decision = HumanDecision(
        decision_id="decision-1",
        analysis_id="analysis-1",
        action=HumanDecisionAction.WAIT,
        human_decided_at=BASE + timedelta(minutes=1),
        actor_ref="reviewer-pseudonym",
        rationale="Evidence remains incomplete.",
    )
    retraction = Retraction(
        retraction_id="retraction-1",
        event_version_id="event-version-1",
        source_receipt_id="receipt-1",
        retracted_at=BASE + timedelta(hours=1),
        recorded_at=BASE + timedelta(hours=1, seconds=2),
        reason="The primary source withdrew the statement.",
    )
    assert decision.action is HumanDecisionAction.WAIT
    assert retraction.retracted_at < retraction.recorded_at


def test_outcome_enforces_canonical_ai_and_human_delays() -> None:
    outcome = Outcome(
        outcome_id="outcome-1",
        analysis_id="analysis-1",
        human_decision_id=None,
        anchor=OutcomeAnchor.AI,
        anchor_at=BASE,
        entry_price_at=BASE + timedelta(seconds=60),
        horizon=OutcomeHorizon.FIVE_MINUTES,
        horizon_at=BASE + timedelta(minutes=5),
        observed_at=BASE + timedelta(minutes=5, seconds=2),
        asset=Asset.BTC,
        venue="coinbase",
        entry_price_usd="65000.00",
        exit_price_usd="65100.00",
        spread_bps="1.25",
        fees_bps="2.00",
        depth_usd="250000.00",
    )
    assert outcome.entry_price_at == outcome.anchor_at + timedelta(seconds=60)

    with pytest.raises(ModelValidationError, match="60 seconds"):
        replace(outcome, entry_price_at=BASE + timedelta(seconds=59))
    with pytest.raises(ModelValidationError, match="human_decision_id"):
        replace(
            outcome,
            anchor=OutcomeAnchor.HUMAN,
            entry_price_at=BASE + timedelta(seconds=30),
        )
