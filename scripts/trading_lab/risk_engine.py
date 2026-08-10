"""Phase 5B: turning a signal into a target exposure, and still not into an order.

Phase 5A stopped at a direction and an intensity, and said explicitly that
intensity is not a capital fraction. This module is where that translation is
written down -- once, in a hashed contract, instead of being improvised at the
point of use.

What comes out is a `PositionTarget`: a desired exposure as a signed fraction
of NAV. It is not an order, not a quantity, not a notional. No equity, no
price, no cash and no portfolio ever enters here, so the layer is stateless
and a historical signal replays to the identical target forever.

Two deliberate omissions:

* **No volatility scaling in V1.** It is the obvious next knob, and that is
  exactly why it is not here. Adding it would mean freezing a volatility
  measure, a window, a target, a floor, a cap and gap semantics all at once --
  a second experimental axis tangled with the first. `risk_scale` exists in
  the contract and is fixed at 1, so a later spec can vary it without
  reshaping anything.
* **No portfolio state.** Current position, cash, margin and unrealised P&L
  belong to the economic engine in a later phase. A risk layer that owned
  mutable state could not be replayed, and every number downstream would
  become a function of call order.

The 25 % cap was not optimised. It was picked as a conservative infrastructure
limit before this module met a single real signal, and specifically not from
the V1 or V2 benchmark results -- those observations are spent, and sizing
chosen against them would inherit their selection bias while looking like
engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
import hashlib
import json

RISK_SCHEMA_VERSION = "trading-lab.risk-engine.v1"
RISK_STRENGTH_MAPPING_VERSION = "linear-strength-to-exposure-v1"
RISK_SCALE_RULE_VERSION = "constant-unit-scale-v1"
RISK_PRECISION = 34

# A conservative infrastructure ceiling, declared in advance. Not an optimum.
MAX_LONG_EXPOSURE_V1 = Decimal("0.25")
MAX_SHORT_EXPOSURE_V1 = Decimal("0.25")
# Fixed at 1 in V1: no data-dependent scaling of any kind.
RISK_SCALE_V1 = Decimal("1")
VOLATILITY_SCALING_ENABLED_V1 = False

# Read as a fact about the contract, not modesty about it.
RISK_LIMIT_V1_IS_NOT_OPTIMIZED = True

MAX_TARGET_BATCH = 1_000_000
_ZERO = Decimal(0)
_ONE = Decimal(1)


class RiskEngineError(RuntimeError):
    """Raised when a position target cannot be derived safely and deterministically."""


class PositionSide:
    """The three exposures this layer can express. Still not an order."""

    LONG = "LONG"
    FLAT = "FLAT"
    SHORT = "SHORT"

    ALL = ("LONG", "FLAT", "SHORT")


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _canonical_decimal(value: Decimal) -> Decimal:
    """One textual form per numeric value.

    Identities here are hashes of `str(...)`, and Decimal keeps trailing zeros:
    `Decimal("-0")`, `Decimal("0.000")` and `Decimal("0")` are all equal yet
    render differently, and `Decimal("0.50") * Decimal("0.25")` renders as
    "0.1250" rather than "0.125". Left alone, two numerically identical targets
    would carry different hashes -- which would quietly break every downstream
    comparison. Exposures live in [-1, 1], so `normalize()` is safe here; the
    integral guard keeps the endpoints out of exponent notation.
    """
    if value == 0:
        return _ZERO
    value = value.normalize()
    if value == value.to_integral_value():
        value = value.quantize(_ONE)
    return value


def _require_decimal(value: object, *, field: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise RiskEngineError(f"{field} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise RiskEngineError(f"{field} must be finite, got {value}")
    return value


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not all(
        character in "0123456789abcdef" for character in value
    ):
        raise RiskEngineError(f"{field} must be a 64-character hex digest, got {value!r}")
    return value


def _require_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RiskEngineError(f"timestamp must be a non-empty string, got {value!r}")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise RiskEngineError(f"timestamp is not ISO-8601: {value!r}") from error
    return value


@dataclass(frozen=True)
class RiskSpec:
    """The DEFINITION of the sizing rule. Frozen before it met a real signal."""

    protocol_version: str = RISK_SCHEMA_VERSION
    max_long_exposure: Decimal = MAX_LONG_EXPOSURE_V1
    max_short_exposure: Decimal = MAX_SHORT_EXPOSURE_V1
    strength_mapping_version: str = RISK_STRENGTH_MAPPING_VERSION
    risk_scale_rule_version: str = RISK_SCALE_RULE_VERSION
    volatility_scaling_enabled: bool = VOLATILITY_SCALING_ENABLED_V1

    def validate(self) -> None:
        for field in ("max_long_exposure", "max_short_exposure"):
            value = _require_decimal(getattr(self, field), field=field)
            if not 0 < value <= _ONE:
                raise RiskEngineError(
                    f"{field} must sit in (0, 1], got {value}")
        if self.volatility_scaling_enabled:
            # V1 has no volatility measure, window, target, floor or cap frozen,
            # so enabling the flag could only mean an undefined rule.
            raise RiskEngineError(
                "volatility scaling is not defined in risk spec v1; a later spec must "
                "freeze its measure, window, target, floor and gap semantics first")
        if self.strength_mapping_version != RISK_STRENGTH_MAPPING_VERSION:
            raise RiskEngineError(
                f"unsupported strength mapping {self.strength_mapping_version!r}")
        if self.risk_scale_rule_version != RISK_SCALE_RULE_VERSION:
            raise RiskEngineError(
                f"unsupported risk scale rule {self.risk_scale_rule_version!r}")

    @property
    def max_abs_exposure(self) -> Decimal:
        return max(self.max_long_exposure, self.max_short_exposure)

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": RISK_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "max_long_exposure": str(self.max_long_exposure),
            "max_short_exposure": str(self.max_short_exposure),
            "strength_mapping_version": self.strength_mapping_version,
            "risk_scale_rule_version": self.risk_scale_rule_version,
            "volatility_scaling_enabled": self.volatility_scaling_enabled,
            "optimized": not RISK_LIMIT_V1_IS_NOT_OPTIMIZED,
        }

    @property
    def risk_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


RISK_SPEC_V1 = RiskSpec()


@dataclass(frozen=True)
class PositionTarget:
    """A desired exposure as a signed fraction of NAV. NOT an order.

    `target_exposure = +0.125` means "hold a long position worth 12.5 % of
    NAV". Converting that into a quantity requires an equity figure and a
    price, neither of which this layer has ever seen.
    """

    timestamp: str
    side: str
    target_exposure: Decimal
    signal_strength: Decimal
    raw_target_exposure: Decimal
    risk_scale: Decimal
    risk_spec_hash: str
    source_signal_spec_hash: str
    source_signal_decision_hash: str
    reason: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": RISK_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "side": self.side,
            "target_exposure": str(self.target_exposure),
            "signal_strength": str(self.signal_strength),
            "raw_target_exposure": str(self.raw_target_exposure),
            "risk_scale": str(self.risk_scale),
            "risk_spec_hash": self.risk_spec_hash,
            "source_signal_spec_hash": self.source_signal_spec_hash,
            "source_signal_decision_hash": self.source_signal_decision_hash,
        }

    @property
    def position_target_hash(self) -> str:
        return _sha256_canonical(self.canonical())


def _validate_signal(signal) -> tuple[str, Decimal]:
    """Consume a valid SignalDecision, or refuse. Never silently repair one."""
    direction = getattr(signal, "direction", None)
    if direction not in ("LONG", "FLAT", "SHORT"):
        raise RiskEngineError(f"signal direction {direction!r} is not recognised")
    strength = _require_decimal(getattr(signal, "strength", None),
                                field="signal strength")
    if not _ZERO <= strength <= _ONE:
        # A forged strength is refused rather than clamped: clamping would let a
        # malformed upstream produce a plausible-looking target.
        raise RiskEngineError(f"signal strength must sit in [0, 1], got {strength}")
    if direction == "FLAT" and strength != 0:
        raise RiskEngineError(
            f"a FLAT signal must carry zero strength, got {strength}")
    _require_hash(getattr(signal, "signal_spec_hash", None), field="signal_spec_hash")
    _require_hash(getattr(signal, "decision_hash", None), field="decision_hash")
    _require_timestamp(getattr(signal, "timestamp", None))
    return direction, strength


def generate_position_target(*, signal, risk_spec: RiskSpec = RISK_SPEC_V1
                             ) -> PositionTarget:
    """Map one signal to one target exposure. A pure function of its arguments.

    No equity, price, cash, portfolio, cost or outcome is accepted, and with
    the V1 spec no market data is needed at all.
    """
    risk_spec.validate()
    direction, strength = _validate_signal(signal)

    with localcontext() as context:
        context.prec = RISK_PRECISION
        risk_scale = RISK_SCALE_V1
        if direction == PositionSide.LONG:
            raw = strength * risk_spec.max_long_exposure * risk_scale
            reason = "long signal scaled by the long exposure limit"
        elif direction == PositionSide.SHORT:
            raw = -(strength * risk_spec.max_short_exposure * risk_scale)
            reason = "short signal scaled by the short exposure limit"
        else:
            raw = _ZERO
            reason = "flat signal carries no exposure"
        # Defence in depth: the mapping already respects the caps, but a future
        # risk_scale must never be able to push a target past them.
        capped = min(max(raw, -risk_spec.max_short_exposure),
                     risk_spec.max_long_exposure)

    raw = _canonical_decimal(raw)
    capped = _canonical_decimal(capped)
    strength = _canonical_decimal(strength)
    risk_scale = _canonical_decimal(risk_scale)
    if capped > 0:
        side = PositionSide.LONG
    elif capped < 0:
        side = PositionSide.SHORT
    else:
        side = PositionSide.FLAT

    return PositionTarget(
        timestamp=signal.timestamp,
        side=side,
        target_exposure=capped,
        signal_strength=strength,
        raw_target_exposure=raw,
        risk_scale=risk_scale,
        risk_spec_hash=risk_spec.risk_spec_hash,
        source_signal_spec_hash=signal.signal_spec_hash,
        source_signal_decision_hash=signal.decision_hash,
        reason=reason,
    )


@dataclass(frozen=True)
class PositionTargetSeries:
    """An ordered run of targets. Still stateless, still not a portfolio."""

    risk_spec_hash: str
    targets: tuple[PositionTarget, ...]

    @property
    def count(self) -> int:
        return len(self.targets)

    @property
    def first_timestamp(self) -> str | None:
        return self.targets[0].timestamp if self.targets else None

    @property
    def last_timestamp(self) -> str | None:
        return self.targets[-1].timestamp if self.targets else None

    @property
    def series_hash(self) -> str:
        return _sha256_canonical({
            "schema_version": RISK_SCHEMA_VERSION,
            "risk_spec_hash": self.risk_spec_hash,
            "targets": [target.canonical() for target in self.targets],
        })


def generate_position_targets(signals, *, risk_spec: RiskSpec = RISK_SPEC_V1
                              ) -> PositionTargetSeries:
    """Map a run of signals, preserving order and refusing to repair it.

    Out-of-order or duplicated timestamps fail closed. Each target depends only
    on its own signal, so sorting would buy nothing and would hide a real
    upstream defect.
    """
    entries = tuple(getattr(signals, "decisions", signals))
    if len(entries) > MAX_TARGET_BATCH:
        raise RiskEngineError(f"at most {MAX_TARGET_BATCH} signals per batch")
    stamps = [_require_timestamp(getattr(signal, "timestamp", None))
              for signal in entries]
    if stamps != sorted(stamps):
        raise RiskEngineError("signals are not in ascending timestamp order")
    if len(set(stamps)) != len(stamps):
        raise RiskEngineError("signals contain duplicate timestamps")
    targets = tuple(
        generate_position_target(signal=signal, risk_spec=risk_spec)
        for signal in entries
    )
    return PositionTargetSeries(risk_spec_hash=risk_spec.risk_spec_hash, targets=targets)
