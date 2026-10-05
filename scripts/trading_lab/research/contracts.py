"""Versioned research contracts extending, without changing, CONTRACTS_V1."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import ClassVar, Mapping

from scripts.trading_lab.platform.contracts import Contract, digest, positive, timestamp
from scripts.trading_lab.sources.canonical import sha256_canonical

TRIAL_STATES = {'PREPARED', 'RUNNING', 'COMPLETE', 'FAILED', 'ABANDONED', 'BLOCKED'}
OUTCOMES = {'POSITIVE', 'NULL', 'NEGATIVE', 'ABANDONED', 'ERROR', 'PENDING'}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('numeric observations must be finite')
    result = float(value)
    if not isfinite(result):
        raise ValueError('numeric observations must be finite')
    return result


@dataclass(frozen=True, kw_only=True)
class Hypothesis(Contract):
    schema: ClassVar[str] = 'research-hypothesis-v1'
    hypothesis_id: str
    version: str
    statement: str
    mechanism: str
    falsification: str
    sources: tuple
    features: Mapping
    target: str
    horizon_seconds: int
    population: Mapping
    baselines: tuple
    splits: Mapping
    transformations: Mapping
    costs: Mapping
    calendar: Mapping
    availability: Mapping
    decision_criteria: Mapping
    budgets: Mapping
    scope: str
    synthetic: bool
    protocol_hash: str | None = None

    def validate(self):
        positive(self.horizon_seconds)
        if any(not isinstance(x, str) or not x.strip() for x in (self.statement, self.mechanism, self.falsification, self.target)):
            raise ValueError('economic statement, mechanism, falsification and target required')
        if self.scope not in {'EXPLORATORY', 'PREREGISTERED_PROTOCOL'}:
            raise ValueError('invalid research scope')
        for field in ('sources', 'features', 'population', 'baselines', 'splits', 'transformations', 'costs',
                      'calendar', 'availability', 'decision_criteria', 'budgets'):
            if not getattr(self, field):
                raise ValueError('complete research plan required')
        if self.transformations.get('fit_on') != 'TRAIN_ONLY':
            raise ValueError('transformations must fit on train only')
        for key in ('max_trials', 'max_rows', 'wall_seconds'):
            positive(self.budgets.get(key))
        if self.budgets['max_trials'] > 8 or self.budgets['max_rows'] > 10000 or self.budgets['wall_seconds'] > 180:
            raise ValueError('research budget exceeds local bounds')
        if self.protocol_hash is not None:
            digest(self.protocol_hash)
        if self.scope == 'PREREGISTERED_PROTOCOL' and not self.protocol_hash:
            raise ValueError('protocol scope requires its frozen identity')

    @property
    def criteria_hash(self):
        return sha256_canonical(self.to_dict()['decision_criteria'])


@dataclass(frozen=True, kw_only=True)
class Trial(Contract):
    schema: ClassVar[str] = 'research-trial-v1'
    trial_id: str
    hypothesis_hash: str
    experiment_hash: str
    criteria_hash: str
    state: str
    outcome: str
    recorded_at: str
    evidence: Mapping

    def validate(self):
        for value in (self.hypothesis_hash, self.experiment_hash, self.criteria_hash):
            digest(value)
        object.__setattr__(self, 'recorded_at', timestamp(self.recorded_at))
        if self.state not in TRIAL_STATES or self.outcome not in OUTCOMES:
            raise ValueError('invalid trial state or outcome')
        if self.state in {'PREPARED', 'RUNNING', 'BLOCKED'} and self.outcome != 'PENDING':
            raise ValueError('unfinished trial cannot claim a result')
        if self.state == 'COMPLETE' and self.outcome not in {'POSITIVE', 'NULL', 'NEGATIVE'}:
            raise ValueError('completed trials preserve measured outcomes')
        if self.state == 'ABANDONED' and self.outcome != 'ABANDONED':
            raise ValueError('abandoned trials must be kept')
        if self.state == 'FAILED' and self.outcome != 'ERROR':
            raise ValueError('failed trial requires an error outcome')


@dataclass(frozen=True, kw_only=True)
class PredictionEvidence(Contract):
    schema: ClassVar[str] = 'prediction-evidence-v1'
    prediction_id: str
    prediction_hash: str
    recorded_at: str
    features: tuple | None
    snapshot: Mapping
    baselines: Mapping
    input_quality: Mapping
    split: str
    provenance: Mapping

    def validate(self):
        digest(self.prediction_hash)
        object.__setattr__(self, 'recorded_at', timestamp(self.recorded_at))
        if not self.snapshot or not self.input_quality or not self.provenance:
            raise ValueError('snapshot, quality and provenance required')
        for value in self.baselines.values():
            number(value)
        if self.features is not None:
            names = []
            for name, value in self.features:
                if not isinstance(name, str) or not name:
                    raise ValueError('feature names required')
                names.append(name)
                number(value)
            if len(names) != len(set(names)):
                raise ValueError('duplicate feature names')


@dataclass(frozen=True, kw_only=True)
class ExecutionObservation(Contract):
    schema: ClassVar[str] = 'prediction-execution-v1'
    observation_id: str
    prediction_id: str
    prediction_hash: str
    available_at: str
    recorded_at: str
    state: str
    executed_position: Mapping | None
    costs: Mapping | None
    proposal_gap: Mapping | None
    errors: tuple
    provenance: Mapping

    def validate(self):
        digest(self.prediction_hash)
        for field in ('available_at', 'recorded_at'):
            object.__setattr__(self, field, timestamp(getattr(self, field)))
        if self.available_at > self.recorded_at or self.state not in {'PENDING', 'FILLED', 'NO_FILL', 'EXPIRED', 'ERROR'}:
            raise ValueError('invalid execution observation')
        if not self.provenance:
            raise ValueError('execution provenance required')
        if self.costs:
            for value in self.costs.values():
                if number(value) < 0:
                    raise ValueError('costs cannot be negative')


@dataclass(frozen=True, kw_only=True)
class InferenceObservation(Contract):
    schema: ClassVar[str] = 'inference-observation-v1'
    observation_id: str
    model_id: str
    model_contract_hash: str
    artifact_hash: str
    horizon_seconds: int
    product: str
    decision_at: str
    recorded_at: str
    status: str
    latency_ms: object
    errors: tuple
    synthetic: bool

    def validate(self):
        for field in ('decision_at', 'recorded_at'):
            object.__setattr__(self, field, timestamp(getattr(self, field)))
        digest(self.model_contract_hash)
        digest(self.artifact_hash)
        positive(self.horizon_seconds)
        if self.recorded_at < self.decision_at or self.status not in {'SUCCESS', 'ERROR', 'UNAVAILABLE'}:
            raise ValueError('invalid inference observation')
        if self.latency_ms is not None and number(self.latency_ms) < 0:
            raise ValueError('latency cannot be negative')


def due_at(prediction):
    return timestamp((datetime.fromisoformat(prediction.decision_at) + timedelta(seconds=prediction.horizon_seconds)).isoformat())


@dataclass(frozen=True, kw_only=True)
class DecisionObservation(Contract):
    schema: ClassVar[str] = 'prediction-decision-v1'
    observation_id: str
    prediction_id: str
    prediction_hash: str
    available_at: str
    recorded_at: str
    signal: Mapping
    risk: Mapping
    proposed_position: Mapping
    provenance: Mapping

    def validate(self):
        digest(self.prediction_hash)
        for field in ('available_at', 'recorded_at'):
            object.__setattr__(self, field, timestamp(getattr(self, field)))
        if self.available_at > self.recorded_at or not all((self.signal, self.risk, self.proposed_position, self.provenance)):
            raise ValueError('complete causal decision observation required')
