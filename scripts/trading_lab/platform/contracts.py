"""Immutable v1 interchange records. Hashes reuse the source canonical byte form."""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import ClassVar, Mapping
import re

from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical

OUTPUTS = ("return", "target_price", "class", "probabilities", "quantiles", "scenarios")
MODEL_CAPABILITIES = frozenset({"train", "predict", "serialize", "infer"})
STATES = frozenset({"RESOLVED", "UNRESOLVED", "NOT_OBSERVED", "NOT_CONFIGURED", "INTEGRITY_ERROR",
                    "NOT_APPLICABLE", "UNKNOWN_MAPPING", "PARTIAL", "PROTECTED"})


def timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an offset-qualified string")
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamp must carry an explicit UTC offset")
    return moment.astimezone(timezone.utc).isoformat()


def digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 digest")


def positive(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError("contract values must be finite JSON values")


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


class Contract:
    schema: ClassVar[str]

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))
        if hasattr(self, "synthetic") and type(self.synthetic) is not bool:
            raise ValueError("synthetic must be boolean")
        for field in fields(self):
            if field.name.endswith("_id") or field.name in ("version", "implementation_version"):
                if not isinstance(getattr(self, field.name), str) or not getattr(self, field.name):
                    raise ValueError("contract identifiers and versions must be nonempty strings")
        self.validate()

    def validate(self):
        pass

    def to_dict(self) -> dict:
        return {"schema": self.schema, **{f.name: _plain(getattr(self, f.name)) for f in fields(self)}}

    @classmethod
    def from_dict(cls, payload: Mapping):
        data = dict(payload)
        if data.pop("schema", None) != cls.schema:
            raise ValueError("contract schema mismatch")
        return cls(**data)

    def canonical_json(self) -> str:
        return canonical_bytes(self.to_dict()).decode("utf-8")

    @property
    def identity(self) -> str:
        return sha256_canonical(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return self.identity


@dataclass(frozen=True, kw_only=True)
class ProviderContract(Contract):
    schema: ClassVar[str] = "provider-contract-v1"
    provider_id: str
    version: str
    capabilities: tuple[str, ...]
    identities: Mapping
    formats: Mapping
    clocks: Mapping
    limits: Mapping
    corrections: Mapping
    historical_availability: Mapping
    health: Mapping
    evidence: tuple[Mapping, ...]
    activation: str
    shape_verification: str

    def validate(self):
        if not self.provider_id or not self.version or not self.capabilities or not self.evidence:
            raise ValueError("provider needs identity, capabilities and evidence")
        for name in ("identities", "formats", "clocks", "limits", "corrections", "historical_availability", "health"):
            if not isinstance(getattr(self, name), Mapping) or not getattr(self, name):
                raise ValueError("provider descriptor sections must be nonempty objects")
        if self.activation not in ("READ_ONLY_ARCHIVE", "WAITING_AUTHORIZATION"):
            raise ValueError("unsupported activation state")


@dataclass(frozen=True, kw_only=True)
class InformationSnapshot(Contract):
    schema: ClassVar[str] = "information-snapshot-v1"
    as_of: str
    products: tuple[str, ...]
    companies: Mapping
    sources: Mapping
    prices: Mapping
    events: tuple[Mapping, ...]
    features: Mapping
    policies: Mapping
    coverage: Mapping
    quality: Mapping
    synthetic: bool = False

    def validate(self):
        object.__setattr__(self, "as_of", timestamp(self.as_of))
        if not self.products or len(set(self.products)) != len(self.products):
            raise ValueError("products must be nonempty and unique")
        if set(self.prices) != set(self.products) or set(self.features) != set(self.products):
            raise ValueError("price and feature states required for every product")
        if self.policies.get("visibility") != "DURABLE_OBSERVED":
            raise ValueError("snapshot v1 requires explicit DURABLE_OBSERVED visibility")
        for product in self.products:
            price = self.prices[product]
            if price["state"] not in STATES:
                raise ValueError("unknown price state")
            if price.get("price") and timestamp(price["price"]["available_at"]) > self.as_of:
                raise ValueError("snapshot cannot expose a future price")
        for source in self.sources.values():
            if source["state"] not in STATES:
                raise ValueError("unknown source state")
            H = source["H"]
            if H is not None and (type(H) is not int or H < 0):
                raise ValueError("each source has its own nonnegative horizon or null")
        for event in self.events:
            if timestamp(event["available_at"]) > self.as_of:
                raise ValueError("snapshot cannot expose a future event")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be boolean")


@dataclass(frozen=True, kw_only=True)
class ModelContract(Contract):
    schema: ClassVar[str] = "model-contract-v1"
    model_id: str
    version: str
    inputs: Mapping
    outputs: Mapping
    horizons_seconds: tuple[int, ...]
    capabilities: tuple[str, ...]
    limits: Mapping
    implementation_version: str
    synthetic: bool = False

    def validate(self):
        if not self.model_id or not self.version or not self.inputs or not self.implementation_version:
            raise ValueError("model identity, version and inputs required")
        if set(self.outputs) != set(OUTPUTS) or not any(v is not None for v in self.outputs.values()):
            raise ValueError("declare all output kinds; unsupported kinds are null")
        if any(value is not None and not isinstance(value, (str, Mapping)) for value in self.outputs.values()):
            raise ValueError("model output declarations must describe type or method")
        if not set(self.capabilities) <= MODEL_CAPABILITIES or not self.capabilities:
            raise ValueError("unknown or absent model capability")
        if not self.horizons_seconds:
            raise ValueError("model must declare horizons")
        for horizon in self.horizons_seconds:
            positive(horizon)


@dataclass(frozen=True, kw_only=True)
class DatasetManifest(Contract):
    schema: ClassVar[str] = "dataset-manifest-v1"
    dataset_id: str
    version: str
    products: tuple[str, ...]
    decision_start: str
    decision_end: str
    target: str
    horizon_seconds: int
    snapshot_hashes: tuple[str, ...]
    features_hash: str
    exclusions: tuple[Mapping, ...]
    splits: Mapping
    policies: Mapping
    counts: Mapping
    synthetic: bool

    def validate(self):
        for field in ("decision_start", "decision_end"):
            object.__setattr__(self, field, timestamp(getattr(self, field)))
        if self.decision_start >= self.decision_end:
            raise ValueError("dataset interval must be nonempty")
        positive(self.horizon_seconds)
        digest(self.features_hash)
        for value in self.snapshot_hashes:
            digest(value)
        if not self.products or not self.dataset_id or not self.target:
            raise ValueError("dataset identity, products and target required")


@dataclass(frozen=True, kw_only=True)
class ExperimentManifest(Contract):
    schema: ClassVar[str] = "experiment-manifest-v1"
    experiment_id: str
    version: str
    dataset_hash: str
    model_contract_hash: str
    hypothesis: Mapping
    parameters: Mapping
    splits: Mapping
    baselines: tuple[str, ...]
    decision_criteria: Mapping
    costs: Mapping
    budgets: Mapping
    status: str
    artifacts: Mapping
    synthetic: bool

    def validate(self):
        digest(self.dataset_hash)
        digest(self.model_contract_hash)
        if self.status not in ("PREPARED", "RUNNING", "COMPLETE", "FAILED", "CANCELLED", "BLOCKED"):
            raise ValueError("unknown experiment status")
        if not self.experiment_id or not self.baselines or not self.decision_criteria:
            raise ValueError("experiment identity, baselines and predeclared criteria required")


@dataclass(frozen=True, kw_only=True)
class PredictionRecord(Contract):
    schema: ClassVar[str] = "prediction-record-v1"
    prediction_id: str
    model_id: str
    model_contract_hash: str
    artifact_hash: str
    product: str
    decision_at: str
    horizon_seconds: int
    snapshot_hash: str
    features_hash: str
    event_ids: tuple[str, ...]
    outputs: Mapping
    signal: Mapping | None = None
    risk: Mapping | None = None
    proposed_position: Mapping | None = None
    execution: Mapping | None = None
    uncertainty: Mapping | None = None
    errors: tuple[str, ...] = ()
    costs: Mapping | None = None
    synthetic: bool = False

    def validate(self):
        object.__setattr__(self, "decision_at", timestamp(self.decision_at))
        positive(self.horizon_seconds)
        for value in (self.model_contract_hash, self.artifact_hash, self.snapshot_hash, self.features_hash):
            digest(value)
        if set(self.outputs) != set(OUTPUTS):
            raise ValueError("all output kinds required; absent outputs are null")
        if not self.prediction_id or not self.model_id or not self.product:
            raise ValueError("prediction identity, model and product required")


@dataclass(frozen=True, kw_only=True)
class LabelRecord(Contract):
    schema: ClassVar[str] = "label-record-v1"
    label_id: str
    prediction_id: str
    prediction_hash: str
    product: str
    horizon_seconds: int
    realized_at: str
    available_at: str
    recorded_at: str
    target: str
    value: object
    provenance: Mapping
    version: str

    def validate(self):
        digest(self.prediction_hash)
        positive(self.horizon_seconds)
        for field in ("realized_at", "available_at", "recorded_at"):
            object.__setattr__(self, field, timestamp(getattr(self, field)))
        if self.realized_at > self.available_at or self.available_at > self.recorded_at:
            raise ValueError("label cannot be available before realization or recorded before availability")
        if not self.label_id or not self.target or not self.provenance:
            raise ValueError("label identity, target and provenance required")


def enrich_prediction(prediction: PredictionRecord, labels: tuple[LabelRecord, ...], *, as_of: str) -> dict:
    """Read-time enrichment; the immutable prediction and its digest never change."""
    from datetime import timedelta
    T = timestamp(as_of)
    visible = []
    for label in labels:
        if (label.prediction_id, label.prediction_hash, label.product, label.horizon_seconds) != (
                prediction.prediction_id, prediction.identity, prediction.product, prediction.horizon_seconds):
            raise ValueError("label does not bind this prediction")
        due = datetime.fromisoformat(prediction.decision_at) + timedelta(seconds=prediction.horizon_seconds)
        if datetime.fromisoformat(label.realized_at) < due:
            raise ValueError("label realized before prediction horizon")
        if label.available_at <= T and label.recorded_at <= T:
            visible.append({**label.to_dict(), "identity": label.identity})
    visible.sort(key=lambda row: (row["recorded_at"], row["label_id"], row["identity"]))
    return {"prediction": prediction.to_dict(), "prediction_hash": prediction.identity,
            "label_state": "AVAILABLE" if visible else "PENDING", "labels": visible}
