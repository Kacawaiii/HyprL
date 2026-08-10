"""Phase 5A: turning a predicted return into a signal, and nothing more.

A model that outputs `forward_return` does not yet say what to do. The gap
between "the model predicts +0.4 %" and "buy" is where most backtests quietly
acquire their edge, so it gets its own layer, its own contract and its own
hash rather than living as an `if prediction > 0` somewhere in a loop.

This layer stops deliberately early. It produces a `SignalDecision` -- a
direction and a descriptive intensity -- and it knows nothing about capital,
position size, orders, brokers, fees, slippage or profit. Those belong to
later phases, and mixing them in here would make it impossible to tell which
component was responsible for a number.

Three properties are enforced rather than intended:

* **No labels.** `generate_signal` has no parameter through which an actual
  forward return could arrive, and the record adapter reads only the
  timestamp and the prediction. A decision is made before its outcome exists.
* **No calibration.** The thresholds are fixed constants, declared in advance.
  Nothing is a quantile, a z-score, a rolling standard deviation or a "top
  10 %": every one of those makes the signal at T depend on observations that
  come after T, or on the distribution of the very data being scored.
* **No clock.** The decision timestamp comes from the prediction, so replaying
  a historical record produces the identical decision, today or next year.

The thresholds were NOT tuned. They were picked as a plain symmetric rule
before this module scored anything, and specifically not from the observed V1
or V2 prediction distributions -- that corpus is spent, and calibrating on it
would launder an already-observed result into what looks like a design choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
import hashlib
import json

SIGNAL_SCHEMA_VERSION = "trading-lab.signal-engine.v1"
SIGNAL_RULE_VERSION = "static-symmetric-threshold-v1"
SIGNAL_PRECISION = 34

# Fixed in advance, symmetric, and explicitly not optimised for anything.
LONG_THRESHOLD_V1 = Decimal("0.0025")
SHORT_THRESHOLD_V1 = Decimal("-0.0025")
# Excess beyond the threshold at which strength saturates: +0.25 % threshold
# plus 1.00 % of further predicted move reads as full intensity.
SIGNAL_FULL_STRENGTH_EXCESS_V1 = Decimal("0.01")
# Boundary semantics are part of the contract: a prediction sitting exactly on
# a threshold is FLAT. Strict inequalities, tested at the boundary.
BOUNDARY_SEMANTICS_V1 = "strict"

# Read this as a fact about the contract, not modesty about it.
SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED = True

MAX_SIGNAL_BATCH = 1_000_000


class SignalEngineError(RuntimeError):
    """Raised when a signal cannot be produced deterministically and causally."""


class SignalDirection:
    """The three states this layer can express. Deliberately not an exposure."""

    LONG = "LONG"
    FLAT = "FLAT"
    SHORT = "SHORT"

    ALL = ("LONG", "FLAT", "SHORT")


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _require_decimal(value: object, *, field: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise SignalEngineError(f"{field} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise SignalEngineError(f"{field} must be finite, got {value}")
    return value


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not all(
        character in "0123456789abcdef" for character in value
    ):
        raise SignalEngineError(f"{field} must be a 64-character hex digest, got {value!r}")
    return value


def _require_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SignalEngineError(f"timestamp must be a non-empty string, got {value!r}")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise SignalEngineError(f"timestamp is not ISO-8601: {value!r}") from error
    return value


@dataclass(frozen=True)
class SignalSpec:
    """The DEFINITION of the mapping. Fixed before any signal was produced."""

    name: str = SIGNAL_RULE_VERSION
    version: str = SIGNAL_SCHEMA_VERSION
    prediction_horizon: int = 4
    long_threshold: Decimal = LONG_THRESHOLD_V1
    short_threshold: Decimal = SHORT_THRESHOLD_V1
    full_strength_excess: Decimal = SIGNAL_FULL_STRENGTH_EXCESS_V1
    boundary_semantics: str = BOUNDARY_SEMANTICS_V1

    def validate(self) -> None:
        for field in ("long_threshold", "short_threshold", "full_strength_excess"):
            _require_decimal(getattr(self, field), field=field)
        if self.long_threshold <= 0:
            raise SignalEngineError("long_threshold must be strictly positive")
        if self.short_threshold >= 0:
            raise SignalEngineError("short_threshold must be strictly negative")
        if self.full_strength_excess <= 0:
            raise SignalEngineError("full_strength_excess must be strictly positive")
        if type(self.prediction_horizon) is not int or self.prediction_horizon < 1:
            raise SignalEngineError("prediction_horizon must be a positive int")
        if self.boundary_semantics != "strict":
            raise SignalEngineError(
                f"unsupported boundary semantics {self.boundary_semantics!r}")

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "prediction_horizon": self.prediction_horizon,
            "long_threshold": str(self.long_threshold),
            "short_threshold": str(self.short_threshold),
            "full_strength_excess": str(self.full_strength_excess),
            "boundary_semantics": self.boundary_semantics,
            "optimized": not SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED,
        }

    @property
    def spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


SIGNAL_SPEC_V1 = SignalSpec()


@dataclass(frozen=True)
class SignalDecision:
    """One decision, and where it came from. No capital, no order, no cost."""

    timestamp: str
    prediction: Decimal
    direction: str
    strength: Decimal
    signal_spec_hash: str
    source_model_spec_hash: str
    source_fitted_hash: str
    source_benchmark_spec_hash: str
    reason: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "prediction": str(self.prediction),
            "direction": self.direction,
            "strength": str(self.strength),
            "signal_spec_hash": self.signal_spec_hash,
            "source_model_spec_hash": self.source_model_spec_hash,
            "source_fitted_hash": self.source_fitted_hash,
            "source_benchmark_spec_hash": self.source_benchmark_spec_hash,
        }

    @property
    def decision_hash(self) -> str:
        return _sha256_canonical(self.canonical())


def _strength(prediction: Decimal, threshold: Decimal, spec: SignalSpec) -> Decimal:
    """Descriptive intensity in [0, 1]. NOT a capital fraction.

    `strength=0.8` means the prediction sits 80 % of the way to the saturation
    excess. It does not mean 80 % of a portfolio, and translating one into the
    other is the risk engine's job in a later phase.
    """
    with localcontext() as context:
        context.prec = SIGNAL_PRECISION
        excess = abs(prediction) - abs(threshold)
        if excess <= 0:
            return Decimal(0)
        return min(Decimal(1), excess / spec.full_strength_excess)


def generate_signal(*, timestamp: str, prediction: Decimal, model_spec_hash: str,
                    fitted_hash: str, benchmark_spec_hash: str,
                    signal_spec: SignalSpec = SIGNAL_SPEC_V1) -> SignalDecision:
    """Map one prediction to one decision. A pure function of its arguments.

    There is no parameter here through which an actual forward return could
    arrive, and that is the point: a decision is formed before its outcome
    exists, so no test has to be trusted on the matter.
    """
    signal_spec.validate()
    value = _require_decimal(prediction, field="prediction")
    stamp = _require_timestamp(timestamp)
    model = _require_hash(model_spec_hash, field="model_spec_hash")
    fitted = _require_hash(fitted_hash, field="fitted_hash")
    benchmark = _require_hash(benchmark_spec_hash, field="benchmark_spec_hash")

    # Strict inequalities: a prediction exactly on a threshold is FLAT.
    if value > signal_spec.long_threshold:
        direction = SignalDirection.LONG
        strength = _strength(value, signal_spec.long_threshold, signal_spec)
        reason = "prediction above long threshold"
    elif value < signal_spec.short_threshold:
        direction = SignalDirection.SHORT
        strength = _strength(value, signal_spec.short_threshold, signal_spec)
        reason = "prediction below short threshold"
    else:
        direction = SignalDirection.FLAT
        strength = Decimal(0)
        reason = "prediction within the neutral band"

    return SignalDecision(
        timestamp=stamp,
        prediction=value,
        direction=direction,
        strength=strength,
        signal_spec_hash=signal_spec.spec_hash,
        source_model_spec_hash=model,
        source_fitted_hash=fitted,
        source_benchmark_spec_hash=benchmark,
        reason=reason,
    )


def signal_from_prediction_record(record, *, model_spec_hash: str, fitted_hash: str,
                                  benchmark_spec_hash: str,
                                  signal_spec: SignalSpec = SIGNAL_SPEC_V1
                                  ) -> SignalDecision:
    """Adapt a stored PredictionRecord, reading ONLY its timestamp and prediction.

    A `PredictionRecord` also carries `actual_forward_return`, because it was
    built for scoring after the fact. This function never touches it -- a test
    passes a record whose `actual_forward_return` raises on access, and a
    signal still comes out.
    """
    return generate_signal(
        timestamp=record.bar_open_at,
        prediction=record.prediction,
        model_spec_hash=model_spec_hash,
        fitted_hash=fitted_hash,
        benchmark_spec_hash=benchmark_spec_hash,
        signal_spec=signal_spec,
    )


@dataclass(frozen=True)
class SignalSeries:
    """An ordered run of decisions. Still no trading logic of any kind."""

    signal_spec_hash: str
    decisions: tuple[SignalDecision, ...]

    @property
    def count(self) -> int:
        return len(self.decisions)

    @property
    def first_timestamp(self) -> str | None:
        return self.decisions[0].timestamp if self.decisions else None

    @property
    def last_timestamp(self) -> str | None:
        return self.decisions[-1].timestamp if self.decisions else None

    @property
    def series_hash(self) -> str:
        return _sha256_canonical({
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "signal_spec_hash": self.signal_spec_hash,
            "decisions": [decision.canonical() for decision in self.decisions],
        })


def generate_signals(records, *, model_spec_hash: str, fitted_hash: str,
                     benchmark_spec_hash: str,
                     signal_spec: SignalSpec = SIGNAL_SPEC_V1) -> SignalSeries:
    """Map a sequence of predictions, preserving order and refusing to repair it.

    Out-of-order or duplicated timestamps fail closed rather than being sorted
    or de-duplicated: silently reordering an input would hide the very defect
    worth knowing about, and each decision is independent anyway, so sorting
    would buy nothing.
    """
    entries = tuple(records)
    if len(entries) > MAX_SIGNAL_BATCH:
        raise SignalEngineError(f"at most {MAX_SIGNAL_BATCH} predictions per batch")
    stamps = [_require_timestamp(record.bar_open_at) for record in entries]
    if stamps != sorted(stamps):
        raise SignalEngineError("predictions are not in ascending timestamp order")
    if len(set(stamps)) != len(stamps):
        raise SignalEngineError("predictions contain duplicate timestamps")
    decisions = tuple(
        signal_from_prediction_record(
            record, model_spec_hash=model_spec_hash, fitted_hash=fitted_hash,
            benchmark_spec_hash=benchmark_spec_hash, signal_spec=signal_spec)
        for record in entries
    )
    return SignalSeries(signal_spec_hash=signal_spec.spec_hash, decisions=decisions)
