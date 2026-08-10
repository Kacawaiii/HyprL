"""Phase 5B: signal to target exposure, and the limits that must hold.

This layer is deliberately thin, so most of what follows checks what it is
unable to do: read an outcome, own a portfolio, consult a clock, exceed its
own cap, or repair a malformed input rather than refusing it.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
import ast
import importlib
import inspect
import pathlib

import pytest


HOUR = timedelta(hours=1)
GRID = datetime(2026, 5, 1, tzinfo=timezone.utc)
SPEC_HASH = "a" * 64
DECISION_HASH = "b" * 64


@pytest.fixture
def risk():
    return importlib.import_module("scripts.trading_lab.risk_engine")


@pytest.fixture
def signals():
    return importlib.import_module("scripts.trading_lab.signal_engine")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat()


_UNSET = object()


class Signal:
    """A minimal stand-in carrying exactly what the risk engine may read.

    `timestamp` uses a sentinel so a deliberately malformed value -- including
    None -- reaches the engine untouched instead of being replaced by the
    default, which would make the validation tests assert nothing.
    """

    def __init__(self, direction, strength, *, timestamp=_UNSET,
                 signal_spec_hash=SPEC_HASH, decision_hash=DECISION_HASH):
        self.timestamp = _iso(GRID) if timestamp is _UNSET else timestamp
        self.direction = direction
        self.strength = strength if isinstance(strength, Decimal) else Decimal(strength)
        self.signal_spec_hash = signal_spec_hash
        self.decision_hash = decision_hash


class OutcomeTrap(Signal):
    """A signal carrying an outcome that detonates if anything reads it."""

    @property
    def actual_forward_return(self):
        raise AssertionError("the risk engine read a realized outcome")

    @property
    def pnl(self):
        raise AssertionError("the risk engine read a profit figure")


# --- the frozen contract ---------------------------------------------------


def test_the_risk_spec_v1_is_the_declared_conservative_cap(risk):
    spec = risk.RISK_SPEC_V1
    assert spec.max_long_exposure == Decimal("0.25")
    assert spec.max_short_exposure == Decimal("0.25")
    assert spec.max_abs_exposure == Decimal("0.25")
    assert spec.volatility_scaling_enabled is False
    assert spec.strength_mapping_version == "linear-strength-to-exposure-v1"
    assert spec.risk_scale_rule_version == "constant-unit-scale-v1"
    assert risk.RISK_LIMIT_V1_IS_NOT_OPTIMIZED is True
    assert spec.canonical()["optimized"] is False
    assert risk.RISK_SCALE_V1 == Decimal(1)


@pytest.mark.parametrize("direction,strength,exposure", [
    ("FLAT", "0", "0"),
    ("LONG", "0.01", "0.0025"),
    ("LONG", "0.25", "0.0625"),
    ("LONG", "0.50", "0.125"),
    ("LONG", "1", "0.25"),
    ("SHORT", "0.01", "-0.0025"),
    ("SHORT", "0.50", "-0.125"),
    ("SHORT", "1", "-0.25"),
])
def test_the_mapping_on_hand_computed_values(risk, direction, strength, exposure):
    target = risk.generate_position_target(signal=Signal(direction, strength))
    assert target.target_exposure == Decimal(exposure)
    assert str(target.target_exposure) == exposure      # one canonical text form


@pytest.mark.parametrize("strength", ["0.01", "0.2", "0.5", "0.75", "1"])
def test_the_contract_is_symmetric_under_risk_spec_v1(risk, strength):
    long = risk.generate_position_target(signal=Signal("LONG", strength))
    short = risk.generate_position_target(signal=Signal("SHORT", strength))
    assert long.target_exposure == -short.target_exposure
    assert abs(long.target_exposure) == abs(short.target_exposure)


def test_an_asymmetric_spec_is_representable_even_though_v1_is_symmetric(risk):
    """V1 is symmetric by choice, not because the design cannot express otherwise."""
    spec = replace(risk.RISK_SPEC_V1, max_short_exposure=Decimal("0.10"))
    long = risk.generate_position_target(signal=Signal("LONG", "1"), risk_spec=spec)
    short = risk.generate_position_target(signal=Signal("SHORT", "1"), risk_spec=spec)
    assert long.target_exposure == Decimal("0.25")
    assert short.target_exposure == Decimal("-0.10")
    assert spec.risk_spec_hash != risk.RISK_SPEC_V1.risk_spec_hash


def test_the_target_never_exceeds_the_cap(risk):
    for strength in ("0.9", "0.99", "1"):
        for direction in ("LONG", "SHORT"):
            target = risk.generate_position_target(signal=Signal(direction, strength))
            assert abs(target.target_exposure) <= risk.RISK_SPEC_V1.max_abs_exposure
    assert risk.generate_position_target(
        signal=Signal("LONG", "1")).target_exposure == Decimal("0.25")


def test_the_risk_scale_is_a_constant_one_in_v1(risk):
    for direction, strength in (("LONG", "0.4"), ("SHORT", "0.4"), ("FLAT", "0")):
        target = risk.generate_position_target(signal=Signal(direction, strength))
        assert target.risk_scale == Decimal(1)
    # the raw value is recorded alongside the capped one, and in V1 they agree
    target = risk.generate_position_target(signal=Signal("LONG", "0.4"))
    assert target.raw_target_exposure == target.target_exposure


def test_volatility_scaling_cannot_be_switched_on_without_a_defined_rule(risk):
    """Enabling a flag whose measure and window are unfrozen would mean nothing."""
    spec = replace(risk.RISK_SPEC_V1, volatility_scaling_enabled=True)
    with pytest.raises(risk.RiskEngineError, match="volatility scaling is not defined"):
        risk.generate_position_target(signal=Signal("LONG", "0.5"), risk_spec=spec)


# --- side consistency and zero -------------------------------------------


@pytest.mark.parametrize("direction,strength", [
    ("LONG", "0.6"), ("SHORT", "0.6"), ("FLAT", "0"), ("LONG", "0"), ("SHORT", "0"),
])
def test_the_side_always_agrees_with_the_sign(risk, direction, strength):
    target = risk.generate_position_target(signal=Signal(direction, strength))
    assert target.side in risk.PositionSide.ALL
    if target.target_exposure > 0:
        assert target.side == "LONG"
    elif target.target_exposure < 0:
        assert target.side == "SHORT"
    else:
        assert target.side == "FLAT"


def test_a_zero_strength_signal_collapses_to_a_canonical_flat(risk):
    """A short of zero size is flat, and must not hash as something else."""
    short_zero = risk.generate_position_target(signal=Signal("SHORT", "0"))
    long_zero = risk.generate_position_target(signal=Signal("LONG", "0"))
    flat = risk.generate_position_target(signal=Signal("FLAT", "0"))
    for target in (short_zero, long_zero, flat):
        assert target.side == "FLAT"
        assert target.target_exposure == Decimal(0)
        assert str(target.target_exposure) == "0"
    assert short_zero.position_target_hash == long_zero.position_target_hash == \
        flat.position_target_hash


def test_numerically_equal_targets_share_one_identity(risk):
    """Trailing zeros must not fork the hash of an identical exposure."""
    coarse = risk.generate_position_target(signal=Signal("LONG", Decimal("0.5")))
    padded = risk.generate_position_target(signal=Signal("LONG", Decimal("0.50")))
    assert coarse.target_exposure == padded.target_exposure
    assert str(coarse.target_exposure) == str(padded.target_exposure) == "0.125"
    assert coarse.position_target_hash == padded.position_target_hash


# --- refusing bad input ----------------------------------------------------


@pytest.mark.parametrize("direction", ["BUY", "SELL", "long", "", None, 1])
def test_an_unrecognised_direction_is_refused(risk, direction):
    with pytest.raises(risk.RiskEngineError, match="direction"):
        risk.generate_position_target(signal=Signal(direction, "0.5"))


@pytest.mark.parametrize("strength", [
    Decimal("1.0001"), Decimal("-0.1"), Decimal("2"), Decimal("NaN"),
    Decimal("Infinity"), 0.5, "0.5", None,
])
def test_a_forged_strength_fails_closed_rather_than_being_clamped(risk, strength):
    """Clamping would let a malformed upstream produce a plausible target."""
    signal = Signal("LONG", "0.5")
    signal.strength = strength
    with pytest.raises(risk.RiskEngineError, match="strength"):
        risk.generate_position_target(signal=signal)


def test_a_flat_signal_carrying_strength_is_refused(risk):
    with pytest.raises(risk.RiskEngineError, match="FLAT signal must carry zero"):
        risk.generate_position_target(signal=Signal("FLAT", "0.4"))


@pytest.mark.parametrize("field", ["signal_spec_hash", "decision_hash"])
@pytest.mark.parametrize("value", ["", "abc", "Z" * 64, "a" * 63, None])
def test_a_malformed_provenance_hash_is_refused(risk, field, value):
    signal = Signal("LONG", "0.5")
    setattr(signal, field, value)
    with pytest.raises(risk.RiskEngineError, match=field):
        risk.generate_position_target(signal=signal)


@pytest.mark.parametrize("timestamp", ["", "not-a-date", None, 20260501])
def test_a_malformed_timestamp_is_refused(risk, timestamp):
    with pytest.raises(risk.RiskEngineError, match="timestamp"):
        risk.generate_position_target(signal=Signal("LONG", "0.5", timestamp=timestamp))


@pytest.mark.parametrize("overrides,fragment", [
    ({"max_long_exposure": Decimal("0")}, "max_long_exposure"),
    ({"max_long_exposure": Decimal("-0.1")}, "max_long_exposure"),
    ({"max_long_exposure": Decimal("1.5")}, "max_long_exposure"),
    ({"max_short_exposure": Decimal("0")}, "max_short_exposure"),
    ({"max_long_exposure": 0.25}, "max_long_exposure"),
    ({"strength_mapping_version": "other-v9"}, "strength mapping"),
    ({"risk_scale_rule_version": "other-v9"}, "risk scale rule"),
])
def test_a_malformed_risk_spec_is_refused(risk, overrides, fragment):
    spec = replace(risk.RISK_SPEC_V1, **overrides)
    with pytest.raises(risk.RiskEngineError, match=fragment):
        risk.generate_position_target(signal=Signal("LONG", "0.5"), risk_spec=spec)


# --- what this layer is not -----------------------------------------------


def test_the_api_accepts_no_portfolio_price_or_outcome(risk):
    """Structural: there is no channel for equity, price, cost or an outcome."""
    parameters = inspect.signature(risk.generate_position_target).parameters
    assert set(parameters) == {"signal", "risk_spec"}
    forbidden = ("equity", "cash", "price", "nav", "portfolio", "position_size",
                 "quantity", "fee", "slippage", "actual", "label", "future", "pnl",
                 "broker", "order")
    for name in parameters:
        assert not any(word in name.lower() for word in forbidden), name


def test_a_target_carries_no_execution_or_accounting_concept(risk):
    target = risk.generate_position_target(signal=Signal("LONG", "0.5"))
    for name in ("cash", "equity", "nav", "quantity", "notional", "order", "fee",
                 "fees", "slippage", "pnl", "realized", "unrealized", "margin",
                 "fill", "broker"):
        assert not hasattr(target, name), name
    for name in dir(risk):
        if not name.startswith("_"):
            assert not any(word in name.lower() for word in
                           ("fee", "slippage", "broker", "pnl", "equity", "fill",
                            "portfolio")), name


def test_the_engine_never_reads_an_outcome(risk):
    """A signal whose outcome explodes still produces a target."""
    target = risk.generate_position_target(signal=OutcomeTrap("LONG", "0.5"))
    assert target.side == "LONG"
    assert target.target_exposure == Decimal("0.125")


def test_the_module_holds_no_mutable_state_and_no_clock(risk):
    source = pathlib.Path(risk.__file__).read_text()
    assert "datetime.now" not in source and "utcnow" not in source
    assert "time.time" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(
                        decorator.func, "id", "") == "dataclass":
                    frozen = {kw.arg: getattr(kw.value, "value", None)
                              for kw in decorator.keywords}
                    assert frozen.get("frozen") is True, node.name
    first = risk.generate_position_target(signal=Signal("LONG", "0.5"))
    second = risk.generate_position_target(signal=Signal("LONG", "0.5"))
    assert first.position_target_hash == second.position_target_hash


def test_the_module_never_reaches_for_a_benchmark_result(risk):
    source = pathlib.Path(risk.__file__).read_text()
    for forbidden in ("benchmark_results", "oos_records", "rank_ic", "PredictionRecord",
                      "real_benchmark", "walk_forward"):
        assert forbidden not in source, forbidden


# --- identity --------------------------------------------------------------


def test_the_risk_spec_hash_covers_every_part_of_the_contract(risk):
    baseline = risk.RISK_SPEC_V1.risk_spec_hash
    assert len(baseline) == 64
    assert risk.RiskSpec().risk_spec_hash == baseline
    for overrides in ({"max_long_exposure": Decimal("0.5")},
                      {"max_short_exposure": Decimal("0.5")},
                      {"strength_mapping_version": "other-v9"},
                      {"risk_scale_rule_version": "other-v9"},
                      {"volatility_scaling_enabled": True},
                      {"protocol_version": "trading-lab.risk-engine.v9"}):
        assert replace(risk.RISK_SPEC_V1, **overrides).risk_spec_hash != baseline


def test_the_target_hash_covers_the_sizing_and_its_provenance(risk):
    target = risk.generate_position_target(signal=Signal("LONG", "0.5"))
    baseline = target.position_target_hash
    assert len(baseline) == 64
    assert risk.generate_position_target(
        signal=Signal("LONG", "0.5")).position_target_hash == baseline
    assert risk.generate_position_target(
        signal=Signal("LONG", "0.6")).position_target_hash != baseline
    assert risk.generate_position_target(
        signal=Signal("SHORT", "0.5")).position_target_hash != baseline
    assert risk.generate_position_target(
        signal=Signal("LONG", "0.5", timestamp=_iso(GRID + HOUR))
    ).position_target_hash != baseline
    for field in ("signal_spec_hash", "decision_hash"):
        signal = Signal("LONG", "0.5")
        setattr(signal, field, "9" * 64)
        assert risk.generate_position_target(
            signal=signal).position_target_hash != baseline
    spec = replace(risk.RISK_SPEC_V1, max_long_exposure=Decimal("0.5"))
    assert risk.generate_position_target(
        signal=Signal("LONG", "0.5"), risk_spec=spec).position_target_hash != baseline


def test_the_target_records_where_it_came_from(risk):
    target = risk.generate_position_target(signal=Signal("LONG", "0.5"))
    assert target.risk_spec_hash == risk.RISK_SPEC_V1.risk_spec_hash
    assert target.source_signal_spec_hash == SPEC_HASH
    assert target.source_signal_decision_hash == DECISION_HASH
    assert target.signal_strength == Decimal("0.5")
    assert "long signal" in target.reason


def test_the_target_is_indifferent_to_the_callers_decimal_context(risk):
    original = getcontext().prec
    seen = set()
    try:
        for precision in (6, 28, 34, 60):
            getcontext().prec = precision
            target = risk.generate_position_target(signal=Signal("LONG", "0.37"))
            seen.add((str(target.target_exposure), target.position_target_hash))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


# --- batches ---------------------------------------------------------------


def test_a_batch_preserves_order_and_hashes_the_whole_run(risk):
    entries = [Signal(direction, strength, timestamp=_iso(GRID + HOUR * index))
               for index, (direction, strength) in enumerate(
                   [("LONG", "1"), ("FLAT", "0"), ("SHORT", "0.5"), ("LONG", "0.2")])]
    series = risk.generate_position_targets(entries)
    assert series.count == 4
    assert [t.side for t in series.targets] == ["LONG", "FLAT", "SHORT", "LONG"]
    assert [str(t.target_exposure) for t in series.targets] == [
        "0.25", "0", "-0.125", "0.05"]
    assert series.first_timestamp == _iso(GRID)
    assert series.last_timestamp == _iso(GRID + HOUR * 3)
    assert series.risk_spec_hash == risk.RISK_SPEC_V1.risk_spec_hash
    assert risk.generate_position_targets(entries).series_hash == series.series_hash
    reordered = risk.PositionTargetSeries(risk_spec_hash=series.risk_spec_hash,
                                          targets=tuple(reversed(series.targets)))
    assert reordered.series_hash != series.series_hash


def test_out_of_order_or_duplicated_signals_fail_closed(risk):
    scrambled = [Signal("LONG", "0.5", timestamp=_iso(GRID + HOUR)),
                 Signal("LONG", "0.5", timestamp=_iso(GRID))]
    with pytest.raises(risk.RiskEngineError, match="ascending"):
        risk.generate_position_targets(scrambled)
    duplicated = [Signal("LONG", "0.5"), Signal("SHORT", "0.5")]
    with pytest.raises(risk.RiskEngineError, match="duplicate"):
        risk.generate_position_targets(duplicated)


def test_the_batch_never_sorts_its_input(risk):
    source = pathlib.Path(risk.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            assert not (isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "sorted"), ast.dump(node)[:80]
    assert "stamps != sorted(stamps)" in source


def test_a_batch_does_not_mutate_its_source(risk):
    entries = [Signal("LONG", "0.5", timestamp=_iso(GRID + HOUR * index))
               for index in range(4)]
    before = [(s.timestamp, s.direction, s.strength) for s in entries]
    risk.generate_position_targets(entries)
    assert [(s.timestamp, s.direction, s.strength) for s in entries] == before


def test_an_empty_batch_is_an_empty_series(risk):
    series = risk.generate_position_targets([])
    assert series.count == 0
    assert series.first_timestamp is None and series.last_timestamp is None
    assert len(series.series_hash) == 64


def test_later_signals_cannot_change_earlier_targets(risk):
    early = [Signal("LONG", "0.4", timestamp=_iso(GRID + HOUR * index))
             for index in range(5)]
    extended = early + [Signal("SHORT", "1", timestamp=_iso(GRID + HOUR * index))
                        for index in range(5, 12)]
    short = risk.generate_position_targets(early)
    long = risk.generate_position_targets(extended)
    assert long.count > short.count
    assert long.targets[:short.count] == short.targets
    for earlier, later in zip(short.targets, long.targets):
        assert earlier.position_target_hash == later.position_target_hash


def test_a_target_depends_only_on_its_own_signal(risk):
    entries = [Signal(d, s, timestamp=_iso(GRID + HOUR * index))
               for index, (d, s) in enumerate(
                   [("LONG", "0.3"), ("SHORT", "1"), ("FLAT", "0")])]
    series = risk.generate_position_targets(entries)
    for signal, target in zip(entries, series.targets):
        assert risk.generate_position_target(signal=signal) == target


# --- integration with the frozen signal engine ----------------------------


@pytest.mark.parametrize("prediction,side,exposure", [
    ("0", "FLAT", "0"),
    ("0.0025", "FLAT", "0"),
    ("0.0075", "LONG", "0.125"),
    ("-0.0075", "SHORT", "-0.125"),
    ("0.0125", "LONG", "0.25"),
    ("-0.0125", "SHORT", "-0.25"),
    ("5", "LONG", "0.25"),
])
def test_prediction_to_signal_to_target_end_to_end(risk, signals, prediction, side,
                                                   exposure):
    """The whole causal chain, with no model and no market data required."""
    model_hash = "d" * 64
    signal = signals.generate_signal(
        timestamp=_iso(GRID), prediction=Decimal(prediction),
        model_spec_hash=model_hash, fitted_hash=model_hash,
        benchmark_spec_hash=model_hash)
    target = risk.generate_position_target(signal=signal)
    assert target.side == side
    assert target.target_exposure == Decimal(exposure)
    assert target.source_signal_spec_hash == signals.SIGNAL_SPEC_V1.spec_hash
    assert target.source_signal_decision_hash == signal.decision_hash
    assert target.timestamp == signal.timestamp


def test_a_signal_series_flows_straight_into_a_target_series(risk, signals):
    model_hash = "e" * 64

    class Record:
        def __init__(self, timestamp, prediction):
            self.bar_open_at = timestamp
            self.prediction = Decimal(prediction)
            self.actual_forward_return = Decimal("0.9")

    records = [Record(_iso(GRID + HOUR * index), value) for index, value in
               enumerate(["0.0075", "0", "-0.0125", "0.004"])]
    signal_series = signals.generate_signals(
        records, model_spec_hash=model_hash, fitted_hash=model_hash,
        benchmark_spec_hash=model_hash)
    target_series = risk.generate_position_targets(signal_series)
    assert target_series.count == signal_series.count == 4
    assert [t.side for t in target_series.targets] == ["LONG", "FLAT", "SHORT", "LONG"]
    assert [str(t.target_exposure) for t in target_series.targets] == [
        "0.125", "0", "-0.25", "0.0375"]
    assert risk.generate_position_targets(signal_series).series_hash == \
        target_series.series_hash


def test_the_cap_still_holds_when_a_future_risk_scale_exceeds_one(risk, monkeypatch):
    """The clamp is unreachable under V1's own mapping -- and that is the point.

    With strength in [0, 1] and the mapping already multiplying by the limit,
    `raw` can never exceed the cap today, so a test written only against V1
    cannot tell a capped engine from an uncapped one. The clamp exists to
    guarantee that a LATER spec supplying `risk_scale != 1` still cannot push a
    target past the limit, so that is the scenario exercised here.

    It also shows why both fields exist: `raw_target_exposure` keeps what the
    mapping asked for, `target_exposure` keeps what the risk contract allows.
    """
    monkeypatch.setattr(risk, "RISK_SCALE_V1", Decimal("2"))
    for direction, expected in (("LONG", Decimal("0.25")), ("SHORT", Decimal("-0.25"))):
        target = risk.generate_position_target(signal=Signal(direction, "1"))
        assert abs(target.raw_target_exposure) == Decimal("0.5")   # what it wanted
        assert target.target_exposure == expected                   # what it may have
        assert abs(target.target_exposure) <= risk.RISK_SPEC_V1.max_abs_exposure
        assert target.side == direction

    # an asymmetric spec must be clamped on the correct side, not the larger one
    spec = replace(risk.RISK_SPEC_V1, max_short_exposure=Decimal("0.10"))
    short = risk.generate_position_target(signal=Signal("SHORT", "1"), risk_spec=spec)
    assert short.target_exposure == Decimal("-0.10")
    long = risk.generate_position_target(signal=Signal("LONG", "1"), risk_spec=spec)
    assert long.target_exposure == Decimal("0.25")
