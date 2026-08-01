"""Immutable V0 records and deterministic causal validations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, ClassVar
from urllib.parse import urlsplit

from crypto_news.protocol import PROTOCOL_V0_SHA256


class ModelValidationError(ValueError):
    """Raised when a V0 record violates its frozen contract."""


class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"


class FactStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    CORROBORATED = "CORROBORATED"
    PRIMARY_CONFIRMED = "PRIMARY_CONFIRMED"
    RETRACTED = "RETRACTED"


class EventType(str, Enum):
    MACRO_LIQUIDITY = "macro_liquidity"
    REGULATION_LITIGATION = "regulation"
    EXCHANGE_CUSTODY = "exchange_custody"
    PROTOCOL = "protocol"
    STABLECOIN = "stablecoin"
    INSTITUTIONAL_FLOWS = "institutional_flows"
    DERIVATIVES_LEVERAGE = "derivatives_leverage"
    ON_CHAIN = "on_chain"
    SUPPLY = "supply"
    SOCIAL_POLITICAL = "social_political"


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "UNKNOWN"


class Novelty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "UNKNOWN"


class Surprise(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "UNKNOWN"


class MarketConfirmation(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    CONFIRMED = "confirmed"
    CONTRADICTORY = "contradictory"
    UNKNOWN = "UNKNOWN"


class TopRiskStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    NORMAL = "NORMAL"
    VIGILANCE = "VIGILANCE"
    POSSIBLE_DISTRIBUTION = "POSSIBLE_DISTRIBUTION"
    HIGH_RISK = "HIGH_RISK"


class Playbook(str, Enum):
    OFFICIAL_UNDERREACTION = "official_underreaction"
    SELL_THE_NEWS = "sell_the_news"
    BAD_NEWS_ABSORBED = "bad_news_absorbed"
    SYSTEMIC_RISK = "systemic_risk"


class ResearchDecision(str, Enum):
    LONG = "LONG"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


class HumanDecisionAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    WAIT = "WAIT"


class OutcomeAnchor(str, Enum):
    AI = "AI"
    HUMAN = "HUMAN"


class OutcomeHorizon(str, Enum):
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    TWENTY_FOUR_HOURS = "24h"
    SEVENTY_TWO_HOURS = "72h"


HORIZON_DELTAS = {
    OutcomeHorizon.FIVE_MINUTES: timedelta(minutes=5),
    OutcomeHorizon.FIFTEEN_MINUTES: timedelta(minutes=15),
    OutcomeHorizon.ONE_HOUR: timedelta(hours=1),
    OutcomeHorizon.FOUR_HOURS: timedelta(hours=4),
    OutcomeHorizon.TWENTY_FOUR_HOURS: timedelta(hours=24),
    OutcomeHorizon.SEVENTY_TWO_HOURS: timedelta(hours=72),
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} is not a valid {enum_type.__name__}") from exc


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ModelValidationError(f"{name} must be a non-empty string")
    return value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ModelValidationError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ModelValidationError(f"{name} must use UTC")
    return value.astimezone(timezone.utc)


def _ordinal(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ModelValidationError(f"{name} must be an integer from 0 to 100")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ModelValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _https_url(value: Any, name: str) -> str:
    text = _text(value, name)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ModelValidationError(f"{name} must be an HTTPS URL without credentials")
    return text


def _decimal_text(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> str:
    if isinstance(value, float):
        raise ModelValidationError(f"{name} must be an exact decimal string, not a float")
    if not isinstance(value, str):
        raise ModelValidationError(f"{name} must be an exact decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ModelValidationError(f"{name} must be a finite decimal string") from exc
    if not number.is_finite():
        raise ModelValidationError(f"{name} must be a finite decimal string")
    if positive and number <= 0:
        raise ModelValidationError(f"{name} must be positive")
    if non_negative and number < 0:
        raise ModelValidationError(f"{name} must be non-negative")
    return value


def _tuple_text(values: Any, name: str, *, non_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ModelValidationError(f"{name} must be a sequence")
    result = tuple(_text(item, name) for item in values)
    if non_empty and not result:
        raise ModelValidationError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ModelValidationError(f"{name} must not contain duplicates")
    return result


def _assets(values: Any) -> tuple[Asset, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ModelValidationError("affected_assets must contain BTC and/or ETH")
    result = tuple(_enum(item, Asset, "affected_assets") for item in values)
    if len(set(result)) != len(result):
        raise ModelValidationError("affected_assets must not contain duplicates")
    return result  # type: ignore[return-value]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


class RecordMixin:
    record_type: ClassVar[str]

    def to_payload(self) -> dict[str, Any]:
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class SourceReceipt(RecordMixin):
    record_type: ClassVar[str] = "source_receipt"

    receipt_id: str
    canonical_url: str
    source_name: str
    source_tier: int
    published_at: datetime
    first_seen_at: datetime
    retrieved_at: datetime
    content_sha256: str
    author: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "canonical_url", _https_url(self.canonical_url, "canonical_url"))
        object.__setattr__(self, "source_name", _text(self.source_name, "source_name"))
        if (
            isinstance(self.source_tier, bool)
            or not isinstance(self.source_tier, int)
            or self.source_tier not in (0, 1, 2, 3)
        ):
            raise ModelValidationError("source_tier must be an integer from 0 to 3")
        published = _utc(self.published_at, "published_at")
        first_seen = _utc(self.first_seen_at, "first_seen_at")
        retrieved = _utc(self.retrieved_at, "retrieved_at")
        if not published <= first_seen <= retrieved:
            raise ModelValidationError(
                "published_at must be <= first_seen_at <= retrieved_at"
            )
        object.__setattr__(self, "published_at", published)
        object.__setattr__(self, "first_seen_at", first_seen)
        object.__setattr__(self, "retrieved_at", retrieved)
        object.__setattr__(self, "content_sha256", _hash(self.content_sha256, "content_sha256"))
        if self.author is not None:
            object.__setattr__(self, "author", _text(self.author, "author"))


@dataclass(frozen=True, slots=True)
class EventVersion(RecordMixin):
    record_type: ClassVar[str] = "event_version"

    version_id: str
    event_id: str
    version_number: int
    previous_version_id: str | None
    source_receipt_ids: tuple[str, ...]
    primary_source_receipt_id: str | None
    fact_status: FactStatus
    event_type: EventType
    affected_assets: tuple[Asset, ...]
    first_seen_at: datetime
    created_at: datetime
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "version_id", _text(self.version_id, "version_id"))
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        if (
            isinstance(self.version_number, bool)
            or not isinstance(self.version_number, int)
            or self.version_number < 1
        ):
            raise ModelValidationError("version_number must be a positive integer")
        if self.version_number == 1 and self.previous_version_id is not None:
            raise ModelValidationError("previous_version_id must be absent for version 1")
        if self.version_number > 1 and not self.previous_version_id:
            raise ModelValidationError("previous_version_id is required after version 1")
        sources = _tuple_text(self.source_receipt_ids, "source_receipt_ids", non_empty=True)
        object.__setattr__(self, "source_receipt_ids", sources)
        status = _enum(self.fact_status, FactStatus, "fact_status")
        object.__setattr__(self, "fact_status", status)
        if self.primary_source_receipt_id is not None:
            primary = _text(self.primary_source_receipt_id, "primary_source_receipt_id")
            if primary not in sources:
                raise ModelValidationError(
                    "primary_source_receipt_id must be present in source_receipt_ids"
                )
            object.__setattr__(self, "primary_source_receipt_id", primary)
        elif status is FactStatus.PRIMARY_CONFIRMED:
            raise ModelValidationError(
                "primary_source_receipt_id is required for PRIMARY_CONFIRMED"
            )
        object.__setattr__(self, "event_type", _enum(self.event_type, EventType, "event_type"))
        object.__setattr__(self, "affected_assets", _assets(self.affected_assets))
        first_seen = _utc(self.first_seen_at, "first_seen_at")
        created = _utc(self.created_at, "created_at")
        if created < first_seen:
            raise ModelValidationError("created_at must be >= first_seen_at")
        object.__setattr__(self, "first_seen_at", first_seen)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "summary", _text(self.summary, "summary"))


@dataclass(frozen=True, slots=True)
class MarketSnapshot(RecordMixin):
    record_type: ClassVar[str] = "market_snapshot"

    snapshot_id: str
    event_version_id: str
    asset: Asset
    venue: str
    observed_at: datetime
    captured_at: datetime
    last_price_usd: str
    bid_usd: str | None = None
    ask_usd: str | None = None
    depth_usd: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "event_version_id", _text(self.event_version_id, "event_version_id"))
        object.__setattr__(self, "asset", _enum(self.asset, Asset, "asset"))
        object.__setattr__(self, "venue", _text(self.venue, "venue"))
        observed = _utc(self.observed_at, "observed_at")
        captured = _utc(self.captured_at, "captured_at")
        if captured < observed:
            raise ModelValidationError("captured_at must be >= observed_at")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "last_price_usd", _decimal_text(self.last_price_usd, "last_price_usd", positive=True))
        for name in ("bid_usd", "ask_usd"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _decimal_text(value, name, positive=True))
        if self.bid_usd is not None and self.ask_usd is not None:
            if Decimal(self.bid_usd) > Decimal(self.ask_usd):
                raise ModelValidationError("bid_usd must be <= ask_usd")
        if self.depth_usd is not None:
            object.__setattr__(self, "depth_usd", _decimal_text(self.depth_usd, "depth_usd", non_negative=True))


@dataclass(frozen=True, slots=True)
class Analysis(RecordMixin):
    record_type: ClassVar[str] = "analysis"

    analysis_id: str
    event_version_id: str
    market_snapshot_ids: tuple[str, ...]
    protocol_sha256: str
    fact_status: FactStatus
    primary_source_url: str
    corroborating_source_urls: tuple[str, ...]
    published_at: datetime
    first_seen_at: datetime
    frozen_at: datetime
    event_type: EventType
    affected_assets: tuple[Asset, ...]
    event_quality_score: int
    direction: Direction
    horizon: str
    novelty: Novelty
    surprise: Surprise
    priced_in_score: int | None
    priced_in_data_coverage: int
    market_confirmation: MarketConfirmation
    top_risk_score: int | None
    top_risk_status: TopRiskStatus
    playbook: Playbook
    decision: ResearchDecision
    trade_opportunity_score: int
    thesis: str
    invalidation: str
    entry_zone: tuple[str, str] | None
    stop_level: str | None
    target_zones: tuple[str, ...]
    max_holding_time: str
    risk_budget_bps: int
    expires_at: datetime
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _text(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "event_version_id", _text(self.event_version_id, "event_version_id"))
        object.__setattr__(self, "market_snapshot_ids", _tuple_text(self.market_snapshot_ids, "market_snapshot_ids", non_empty=True))
        digest = _hash(self.protocol_sha256, "protocol_sha256")
        if digest != PROTOCOL_V0_SHA256:
            raise ModelValidationError("protocol_sha256 must identify frozen V0")
        object.__setattr__(self, "protocol_sha256", digest)
        object.__setattr__(self, "fact_status", _enum(self.fact_status, FactStatus, "fact_status"))
        object.__setattr__(self, "primary_source_url", _https_url(self.primary_source_url, "primary_source_url"))
        corroborations = tuple(
            _https_url(value, "corroborating_source_urls")
            for value in _tuple_text(self.corroborating_source_urls, "corroborating_source_urls")
        )
        object.__setattr__(self, "corroborating_source_urls", corroborations)
        published = _utc(self.published_at, "published_at")
        first_seen = _utc(self.first_seen_at, "first_seen_at")
        frozen = _utc(self.frozen_at, "frozen_at")
        expires = _utc(self.expires_at, "expires_at")
        if not published <= first_seen < frozen < expires:
            raise ModelValidationError(
                "published_at <= first_seen_at < frozen_at < expires_at is required"
            )
        object.__setattr__(self, "published_at", published)
        object.__setattr__(self, "first_seen_at", first_seen)
        object.__setattr__(self, "frozen_at", frozen)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "event_type", _enum(self.event_type, EventType, "event_type"))
        object.__setattr__(self, "affected_assets", _assets(self.affected_assets))
        object.__setattr__(self, "event_quality_score", _ordinal(self.event_quality_score, "event_quality_score"))
        object.__setattr__(self, "direction", _enum(self.direction, Direction, "direction"))
        object.__setattr__(self, "horizon", _text(self.horizon, "horizon"))
        object.__setattr__(self, "novelty", _enum(self.novelty, Novelty, "novelty"))
        object.__setattr__(self, "surprise", _enum(self.surprise, Surprise, "surprise"))
        if self.priced_in_score is not None:
            object.__setattr__(self, "priced_in_score", _ordinal(self.priced_in_score, "priced_in_score"))
        object.__setattr__(self, "priced_in_data_coverage", _ordinal(self.priced_in_data_coverage, "priced_in_data_coverage"))
        object.__setattr__(self, "market_confirmation", _enum(self.market_confirmation, MarketConfirmation, "market_confirmation"))
        top_status = _enum(self.top_risk_status, TopRiskStatus, "top_risk_status")
        object.__setattr__(self, "top_risk_status", top_status)
        if top_status is TopRiskStatus.UNKNOWN:
            if self.top_risk_score is not None:
                raise ModelValidationError("top_risk_score must be null when status is UNKNOWN")
        elif self.top_risk_score is None:
            raise ModelValidationError("top_risk_score is required for a known status")
        else:
            score = _ordinal(self.top_risk_score, "top_risk_score")
            ranges = {
                TopRiskStatus.NORMAL: range(0, 40),
                TopRiskStatus.VIGILANCE: range(40, 60),
                TopRiskStatus.POSSIBLE_DISTRIBUTION: range(60, 75),
                TopRiskStatus.HIGH_RISK: range(75, 101),
            }
            if score not in ranges[top_status]:
                raise ModelValidationError("top_risk_score does not match top_risk_status")
            object.__setattr__(self, "top_risk_score", score)
        playbook = _enum(self.playbook, Playbook, "playbook")
        object.__setattr__(self, "playbook", playbook)
        decision = _enum(self.decision, ResearchDecision, "decision")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "trade_opportunity_score", _ordinal(self.trade_opportunity_score, "trade_opportunity_score"))
        object.__setattr__(self, "thesis", _text(self.thesis, "thesis"))
        invalidation = _text(self.invalidation, "invalidation", allow_empty=True)
        if decision in {ResearchDecision.LONG, ResearchDecision.REDUCE, ResearchDecision.EXIT} and not invalidation.strip():
            raise ModelValidationError("invalidation is required for a directional decision")
        object.__setattr__(self, "invalidation", invalidation)
        if self.entry_zone is not None:
            if not isinstance(self.entry_zone, (tuple, list)) or len(self.entry_zone) != 2:
                raise ModelValidationError("entry_zone must contain exactly two decimal strings")
            low = _decimal_text(self.entry_zone[0], "entry_zone", positive=True)
            high = _decimal_text(self.entry_zone[1], "entry_zone", positive=True)
            if Decimal(low) > Decimal(high):
                raise ModelValidationError("entry_zone lower bound must be <= upper bound")
            object.__setattr__(self, "entry_zone", (low, high))
        if self.stop_level is not None:
            object.__setattr__(self, "stop_level", _decimal_text(self.stop_level, "stop_level", positive=True))
        targets = _tuple_text(self.target_zones, "target_zones")
        object.__setattr__(self, "target_zones", tuple(_decimal_text(item, "target_zones", positive=True) for item in targets))
        holding = _text(self.max_holding_time, "max_holding_time")
        expected_holding = {
            Playbook.OFFICIAL_UNDERREACTION: "4h",
            Playbook.SELL_THE_NEWS: "24h",
            Playbook.BAD_NEWS_ABSORBED: "24h",
            Playbook.SYSTEMIC_RISK: "24h",
        }
        if holding != expected_holding[playbook]:
            raise ModelValidationError("max_holding_time differs from frozen V0")
        object.__setattr__(self, "max_holding_time", holding)
        if isinstance(self.risk_budget_bps, bool) or not isinstance(self.risk_budget_bps, int):
            raise ModelValidationError("risk_budget_bps must be an integer")
        if not 0 <= self.risk_budget_bps <= 25:
            raise ModelValidationError("risk_budget_bps must be between 0 and 25")
        object.__setattr__(self, "missing_evidence", _tuple_text(self.missing_evidence, "missing_evidence"))


@dataclass(frozen=True, slots=True)
class HumanDecision(RecordMixin):
    record_type: ClassVar[str] = "human_decision"

    decision_id: str
    analysis_id: str
    action: HumanDecisionAction
    human_decided_at: datetime
    actor_ref: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _text(self.decision_id, "decision_id"))
        object.__setattr__(self, "analysis_id", _text(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "action", _enum(self.action, HumanDecisionAction, "action"))
        object.__setattr__(self, "human_decided_at", _utc(self.human_decided_at, "human_decided_at"))
        object.__setattr__(self, "actor_ref", _text(self.actor_ref, "actor_ref"))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))


@dataclass(frozen=True, slots=True)
class Outcome(RecordMixin):
    record_type: ClassVar[str] = "outcome"

    outcome_id: str
    analysis_id: str
    human_decision_id: str | None
    anchor: OutcomeAnchor
    anchor_at: datetime
    entry_price_at: datetime
    horizon: OutcomeHorizon
    horizon_at: datetime
    observed_at: datetime
    asset: Asset
    venue: str
    entry_price_usd: str
    exit_price_usd: str
    spread_bps: str
    fees_bps: str
    depth_usd: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome_id", _text(self.outcome_id, "outcome_id"))
        object.__setattr__(self, "analysis_id", _text(self.analysis_id, "analysis_id"))
        anchor = _enum(self.anchor, OutcomeAnchor, "anchor")
        object.__setattr__(self, "anchor", anchor)
        anchor_at = _utc(self.anchor_at, "anchor_at")
        entry_at = _utc(self.entry_price_at, "entry_price_at")
        horizon = _enum(self.horizon, OutcomeHorizon, "horizon")
        object.__setattr__(self, "horizon", horizon)
        horizon_at = _utc(self.horizon_at, "horizon_at")
        observed_at = _utc(self.observed_at, "observed_at")
        delay = timedelta(seconds=60 if anchor is OutcomeAnchor.AI else 30)
        if entry_at != anchor_at + delay:
            raise ModelValidationError(
                f"entry_price_at must be anchor_at + {int(delay.total_seconds())} seconds"
            )
        if horizon_at != anchor_at + HORIZON_DELTAS[horizon]:
            raise ModelValidationError("horizon_at must be anchor_at plus the frozen horizon")
        if observed_at < horizon_at:
            raise ModelValidationError("observed_at must be >= horizon_at")
        if anchor is OutcomeAnchor.HUMAN and not self.human_decision_id:
            raise ModelValidationError("human_decision_id is required for a human outcome")
        if anchor is OutcomeAnchor.AI and self.human_decision_id is not None:
            raise ModelValidationError("human_decision_id must be absent for an AI outcome")
        if self.human_decision_id is not None:
            object.__setattr__(self, "human_decision_id", _text(self.human_decision_id, "human_decision_id"))
        object.__setattr__(self, "anchor_at", anchor_at)
        object.__setattr__(self, "entry_price_at", entry_at)
        object.__setattr__(self, "horizon_at", horizon_at)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "asset", _enum(self.asset, Asset, "asset"))
        object.__setattr__(self, "venue", _text(self.venue, "venue"))
        for name in ("entry_price_usd", "exit_price_usd"):
            object.__setattr__(self, name, _decimal_text(getattr(self, name), name, positive=True))
        for name in ("spread_bps", "fees_bps", "depth_usd"):
            object.__setattr__(self, name, _decimal_text(getattr(self, name), name, non_negative=True))


@dataclass(frozen=True, slots=True)
class Retraction(RecordMixin):
    record_type: ClassVar[str] = "retraction"

    retraction_id: str
    event_version_id: str
    source_receipt_id: str
    retracted_at: datetime
    recorded_at: datetime
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "retraction_id", _text(self.retraction_id, "retraction_id"))
        object.__setattr__(self, "event_version_id", _text(self.event_version_id, "event_version_id"))
        object.__setattr__(self, "source_receipt_id", _text(self.source_receipt_id, "source_receipt_id"))
        retracted = _utc(self.retracted_at, "retracted_at")
        recorded = _utc(self.recorded_at, "recorded_at")
        if recorded < retracted:
            raise ModelValidationError("recorded_at must be >= retracted_at")
        object.__setattr__(self, "retracted_at", retracted)
        object.__setattr__(self, "recorded_at", recorded)
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
