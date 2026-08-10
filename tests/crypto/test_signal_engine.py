"""Phase 5A: the prediction-to-signal contract, and the proofs it stays honest.

This layer is small, so most of these tests are about what it must NOT be able
to do: read an outcome, learn a threshold from the data it is scoring, consult
a clock, or quietly repair an input that arrived in the wrong order.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import inspect

import pytest


HOUR = timedelta(hours=1)
GRID = datetime(2026, 4, 1, tzinfo=timezone.utc)
MODEL_HASH = "1" * 64
FITTED_HASH = "2" * 64
BENCH_HASH = "3" * 64


@pytest.fixture
def engine():
    return importlib.import_module("scripts.trading_lab.signal_engine")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat()


_UNSET = object()


def _signal(engine, prediction, *, timestamp=_UNSET, spec=None, **overrides):
    """Pass arguments through untouched: coercing them here would hide defects."""
    kwargs = {
        "timestamp": _iso(GRID) if timestamp is _UNSET else timestamp,
        "prediction": Decimal(prediction) if isinstance(prediction, str) else prediction,
        "model_spec_hash": MODEL_HASH,
        "fitted_hash": FITTED_HASH,
        "benchmark_spec_hash": BENCH_HASH,
    }
    kwargs.update(overrides)
    if spec is not None:
        kwargs["signal_spec"] = spec
    return engine.generate_signal(**kwargs)


class Record:
    """A stored prediction, as walk_forward produces them."""

    def __init__(self, timestamp, prediction, actual="0.5"):
        self.bar_open_at = timestamp
        self.prediction = Decimal(prediction)
        self.actual_forward_return = Decimal(actual)


class LabelTrap:
    """A record whose outcome detonates if anything reads it."""

    def __init__(self, timestamp, prediction):
        self.bar_open_at = timestamp
        self.prediction = Decimal(prediction)

    @property
    def actual_forward_return(self):
        raise AssertionError(f"the signal engine read the label of {self.bar_open_at}")


# --- the frozen contract ---------------------------------------------------


def test_the_thresholds_are_the_declared_symmetric_pair(engine):
    spec = engine.SIGNAL_SPEC_V1
    assert spec.long_threshold == Decimal("0.0025")
    assert spec.short_threshold == Decimal("-0.0025")
    assert spec.long_threshold == -spec.short_threshold
    assert spec.full_strength_excess == Decimal("0.01")
    assert spec.prediction_horizon == 4
    assert spec.boundary_semantics == "strict"
    assert engine.SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED is True
    assert spec.canonical()["optimized"] is False


@pytest.mark.parametrize("prediction,direction", [
    ("0", "FLAT"),
    ("0.0025", "FLAT"),      # exactly on the boundary
    ("-0.0025", "FLAT"),     # exactly on the boundary
    ("0.0026", "LONG"),
    ("-0.0026", "SHORT"),
    ("0.0125", "LONG"),
    ("-0.0125", "SHORT"),
    ("0.00249999", "FLAT"),
    ("-0.00249999", "FLAT"),
])
def test_the_direction_rule_at_and_around_the_boundaries(engine, prediction, direction):
    """Strict inequalities: sitting exactly on a threshold is FLAT."""
    assert _signal(engine, prediction).direction == direction


@pytest.mark.parametrize("prediction,strength", [
    ("0", "0"),
    ("0.0025", "0"),
    ("-0.0025", "0"),
    ("0.0026", "0.01"),
    ("-0.0026", "0.01"),
    ("0.0075", "0.5"),
    ("0.0125", "1"),
    ("-0.0125", "1"),
])
def test_the_strength_rule_on_hand_computed_values(engine, prediction, strength):
    assert _signal(engine, prediction).strength == Decimal(strength)


@pytest.mark.parametrize("prediction", ["0.0126", "0.05", "1", "1000"])
def test_the_strength_is_clamped_at_one(engine, prediction):
    """A wild prediction is still a signal of bounded intensity."""
    decision = _signal(engine, prediction)
    assert decision.strength == Decimal(1)
    assert decision.direction == "LONG"
    assert _signal(engine, f"-{prediction}").strength == Decimal(1)


@pytest.mark.parametrize("prediction", ["0.003", "0.0075", "0.0125", "0.5"])
def test_the_contract_is_symmetric(engine, prediction):
    positive = _signal(engine, prediction)
    negative = _signal(engine, f"-{prediction}")
    assert positive.direction == "LONG" and negative.direction == "SHORT"
    assert positive.strength == negative.strength


def test_flat_decisions_never_carry_intensity(engine):
    for prediction in ("0", "0.001", "-0.001", "0.0025", "-0.0025"):
        decision = _signal(engine, prediction)
        assert decision.direction == "FLAT"
        assert decision.strength == Decimal(0)
        assert "neutral band" in decision.reason


def test_the_three_directions_are_the_only_ones(engine):
    assert engine.SignalDirection.ALL == ("LONG", "FLAT", "SHORT")
    seen = {_signal(engine, p).direction
            for p in ("-1", "-0.003", "0", "0.003", "1")}
    assert seen == {"LONG", "FLAT", "SHORT"}


# --- what this layer is not ------------------------------------------------


def test_a_decision_carries_no_capital_or_execution_concept(engine):
    decision = _signal(engine, "0.01")
    forbidden = ("cash", "quantity", "size", "leverage", "order", "fee", "fees",
                 "slippage", "notional", "capital", "pnl", "position")
    for name in forbidden:
        assert not hasattr(decision, name), name
    for name in dir(engine):
        if not name.startswith("_"):
            assert not any(word in name.lower() for word in
                           ("fee", "slippage", "broker", "pnl", "equity", "order")), name


def test_strength_is_documented_as_not_being_a_position_size(engine):
    import pathlib
    source = pathlib.Path(engine.__file__).read_text()
    assert "NOT a capital fraction" in source
    assert "does not mean 80 % of a portfolio" in source


# --- label isolation -------------------------------------------------------


def test_the_generator_has_no_parameter_for_an_outcome(engine):
    """Structural: there is no channel through which a label could arrive."""
    parameters = inspect.signature(engine.generate_signal).parameters
    assert set(parameters) == {"timestamp", "prediction", "model_spec_hash",
                               "fitted_hash", "benchmark_spec_hash", "signal_spec"}
    for name in parameters:
        assert not any(word in name.lower()
                       for word in ("actual", "label", "future", "outcome", "realized"))
    fields = {field for field in engine.SignalDecision.__dataclass_fields__}
    for name in fields:
        assert not any(word in name.lower()
                       for word in ("actual", "label", "realized", "outcome"))


def test_the_record_adapter_never_reads_the_outcome(engine):
    """A record whose label explodes still produces a signal."""
    trapped = LabelTrap(_iso(GRID), "0.0075")
    decision = engine.signal_from_prediction_record(
        trapped, model_spec_hash=MODEL_HASH, fitted_hash=FITTED_HASH,
        benchmark_spec_hash=BENCH_HASH)
    assert decision.direction == "LONG"
    assert decision.strength == Decimal("0.5")
    assert decision.prediction == Decimal("0.0075")


def test_changing_the_outcome_cannot_change_the_decision(engine):
    """The same prediction with a different future is the same decision."""
    first = engine.signal_from_prediction_record(
        Record(_iso(GRID), "0.004", actual="0.9"), model_spec_hash=MODEL_HASH,
        fitted_hash=FITTED_HASH, benchmark_spec_hash=BENCH_HASH)
    second = engine.signal_from_prediction_record(
        Record(_iso(GRID), "0.004", actual="-0.9"), model_spec_hash=MODEL_HASH,
        fitted_hash=FITTED_HASH, benchmark_spec_hash=BENCH_HASH)
    assert first == second
    assert first.decision_hash == second.decision_hash


# --- no calibration, no clock ----------------------------------------------


def test_nothing_in_the_module_calibrates_on_the_data(engine):
    """A learned threshold would make the signal at T depend on the whole series.

    Checked on the parsed identifiers rather than the raw text: the module's
    own docstring names these techniques in order to disclaim them, and a
    substring scan would flag that prose as though it were code.
    """
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(engine.__file__).read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    for forbidden in ("quantile", "percentile", "zscore", "z_score", "stdev",
                      "variance", "mean", "median", "isotonic", "platt", "fit",
                      "sorted"):
        assert forbidden not in called or forbidden == "sorted", forbidden
    # `sorted` may appear only inside a comparison, never bound to a name and
    # used as the output: the batch must refuse a bad order, not repair it.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            assert not (isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "sorted"), ast.dump(node)[:80]
    source = pathlib.Path(engine.__file__).read_text()
    assert "stamps != sorted(stamps)" in source


def test_the_module_never_consults_a_clock(engine):
    import pathlib
    source = pathlib.Path(engine.__file__).read_text()
    assert "datetime.now" not in source and "utcnow" not in source
    assert "time.time" not in source
    first = _signal(engine, "0.01")
    second = _signal(engine, "0.01")
    assert first.decision_hash == second.decision_hash


def test_a_decision_depends_only_on_its_own_prediction(engine):
    """Batch context must not leak into an individual decision."""
    records = [Record(_iso(GRID + HOUR * index), value)
               for index, value in enumerate(["0.004", "-0.02", "0.0001", "0.5"])]
    series = engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                                     fitted_hash=FITTED_HASH,
                                     benchmark_spec_hash=BENCH_HASH)
    for record, decision in zip(records, series.decisions):
        alone = engine.signal_from_prediction_record(
            record, model_spec_hash=MODEL_HASH, fitted_hash=FITTED_HASH,
            benchmark_spec_hash=BENCH_HASH)
        assert alone == decision


def test_later_predictions_cannot_change_earlier_decisions(engine):
    """No global normalisation: history is stable under extension."""
    early = [Record(_iso(GRID + HOUR * index), "0.004") for index in range(5)]
    extended = early + [Record(_iso(GRID + HOUR * index), "9.9")
                        for index in range(5, 12)]
    short = engine.generate_signals(early, model_spec_hash=MODEL_HASH,
                                    fitted_hash=FITTED_HASH,
                                    benchmark_spec_hash=BENCH_HASH)
    long = engine.generate_signals(extended, model_spec_hash=MODEL_HASH,
                                   fitted_hash=FITTED_HASH,
                                   benchmark_spec_hash=BENCH_HASH)
    assert long.count > short.count
    assert long.decisions[:short.count] == short.decisions
    for earlier, later in zip(short.decisions, long.decisions):
        assert earlier.decision_hash == later.decision_hash


# --- input validation ------------------------------------------------------


@pytest.mark.parametrize("prediction", [
    Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), 0.004, "0.004", None, 1,
])
def test_a_prediction_that_is_not_a_finite_decimal_is_refused(engine, prediction):
    with pytest.raises(engine.SignalEngineError, match="prediction"):
        engine.generate_signal(timestamp=_iso(GRID), prediction=prediction,
                               model_spec_hash=MODEL_HASH, fitted_hash=FITTED_HASH,
                               benchmark_spec_hash=BENCH_HASH)


@pytest.mark.parametrize("field", ["model_spec_hash", "fitted_hash",
                                   "benchmark_spec_hash"])
@pytest.mark.parametrize("value", ["", "abc", "Z" * 64, "a" * 63, None, 123])
def test_a_malformed_provenance_hash_is_refused(engine, field, value):
    with pytest.raises(engine.SignalEngineError, match=field):
        _signal(engine, "0.01", **{field: value})


@pytest.mark.parametrize("timestamp", ["", "not-a-date", None, 20260401])
def test_a_malformed_timestamp_is_refused(engine, timestamp):
    with pytest.raises(engine.SignalEngineError, match="timestamp"):
        _signal(engine, "0.01", timestamp=timestamp)


@pytest.mark.parametrize("overrides,fragment", [
    ({"long_threshold": Decimal("-0.001")}, "long_threshold"),
    ({"short_threshold": Decimal("0.001")}, "short_threshold"),
    ({"full_strength_excess": Decimal("0")}, "full_strength_excess"),
    ({"prediction_horizon": 0}, "prediction_horizon"),
    ({"boundary_semantics": "inclusive"}, "boundary semantics"),
    ({"long_threshold": 0.0025}, "long_threshold"),
])
def test_a_malformed_signal_spec_is_refused(engine, overrides, fragment):
    spec = replace(engine.SIGNAL_SPEC_V1, **overrides)
    with pytest.raises(engine.SignalEngineError, match=fragment):
        _signal(engine, "0.01", spec=spec)


# --- identity --------------------------------------------------------------


def test_the_spec_hash_covers_every_part_of_the_contract(engine):
    baseline = engine.SIGNAL_SPEC_V1.spec_hash
    assert len(baseline) == 64
    assert engine.SignalSpec().spec_hash == baseline
    for overrides in ({"long_threshold": Decimal("0.005")},
                      {"short_threshold": Decimal("-0.005")},
                      {"full_strength_excess": Decimal("0.02")},
                      {"prediction_horizon": 8},
                      {"name": "other-rule-v9"},
                      {"version": "trading-lab.signal-engine.v9"}):
        assert replace(engine.SIGNAL_SPEC_V1, **overrides).spec_hash != baseline


def test_the_decision_hash_covers_the_prediction_and_the_provenance(engine):
    decision = _signal(engine, "0.01")
    baseline = decision.decision_hash
    assert len(baseline) == 64
    assert _signal(engine, "0.01").decision_hash == baseline
    assert _signal(engine, "0.011").decision_hash != baseline
    assert _signal(engine, "0.01", timestamp=_iso(GRID + HOUR)).decision_hash != baseline
    for field in ("model_spec_hash", "fitted_hash", "benchmark_spec_hash"):
        assert _signal(engine, "0.01", **{field: "9" * 64}).decision_hash != baseline
    spec = replace(engine.SIGNAL_SPEC_V1, long_threshold=Decimal("0.005"))
    assert _signal(engine, "0.01", spec=spec).decision_hash != baseline
    # the same decision under a different model is a different decision
    assert decision.source_model_spec_hash == MODEL_HASH
    assert decision.source_fitted_hash == FITTED_HASH
    assert decision.source_benchmark_spec_hash == BENCH_HASH
    assert decision.signal_spec_hash == engine.SIGNAL_SPEC_V1.spec_hash


def test_the_decision_is_indifferent_to_the_callers_decimal_context(engine):
    original = getcontext().prec
    seen = set()
    try:
        for precision in (6, 28, 34, 60):
            getcontext().prec = precision
            decision = _signal(engine, "0.0071")
            seen.add((str(decision.strength), decision.decision_hash))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


# --- batches ---------------------------------------------------------------


def test_a_batch_preserves_order_and_hashes_the_whole_run(engine):
    records = [Record(_iso(GRID + HOUR * index), value) for index, value in
               enumerate(["0.004", "-0.004", "0", "0.02", "-0.02"])]
    series = engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                                     fitted_hash=FITTED_HASH,
                                     benchmark_spec_hash=BENCH_HASH)
    assert series.count == 5
    assert [d.direction for d in series.decisions] == [
        "LONG", "SHORT", "FLAT", "LONG", "SHORT"]
    assert series.first_timestamp == _iso(GRID)
    assert series.last_timestamp == _iso(GRID + HOUR * 4)
    assert series.signal_spec_hash == engine.SIGNAL_SPEC_V1.spec_hash
    assert len(series.series_hash) == 64
    again = engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                                    fitted_hash=FITTED_HASH,
                                    benchmark_spec_hash=BENCH_HASH)
    assert again.series_hash == series.series_hash
    # the run identity depends on the order, not just on the multiset
    reordered = engine.SignalSeries(signal_spec_hash=series.signal_spec_hash,
                                    decisions=tuple(reversed(series.decisions)))
    assert reordered.series_hash != series.series_hash


def test_out_of_order_or_duplicated_timestamps_fail_closed(engine):
    """Sorting the input would hide exactly the defect worth surfacing."""
    scrambled = [Record(_iso(GRID + HOUR), "0.004"), Record(_iso(GRID), "0.004")]
    with pytest.raises(engine.SignalEngineError, match="ascending"):
        engine.generate_signals(scrambled, model_spec_hash=MODEL_HASH,
                                fitted_hash=FITTED_HASH, benchmark_spec_hash=BENCH_HASH)
    duplicated = [Record(_iso(GRID), "0.004"), Record(_iso(GRID), "0.005")]
    with pytest.raises(engine.SignalEngineError, match="duplicate"):
        engine.generate_signals(duplicated, model_spec_hash=MODEL_HASH,
                                fitted_hash=FITTED_HASH, benchmark_spec_hash=BENCH_HASH)


def test_a_batch_does_not_mutate_its_source(engine):
    records = [Record(_iso(GRID + HOUR * index), "0.004") for index in range(4)]
    before = [(r.bar_open_at, r.prediction, r.actual_forward_return) for r in records]
    engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                            fitted_hash=FITTED_HASH, benchmark_spec_hash=BENCH_HASH)
    after = [(r.bar_open_at, r.prediction, r.actual_forward_return) for r in records]
    assert before == after


def test_an_empty_batch_is_an_empty_series(engine):
    series = engine.generate_signals([], model_spec_hash=MODEL_HASH,
                                     fitted_hash=FITTED_HASH,
                                     benchmark_spec_hash=BENCH_HASH)
    assert series.count == 0
    assert series.first_timestamp is None and series.last_timestamp is None
    assert len(series.series_hash) == 64


# --- integration with real stored records (read-only) ----------------------


def test_stored_prediction_records_can_be_replayed_into_signals(engine, tmp_path):
    """Parsing, provenance and determinism only.

    Deliberately no direction counts are asserted or reported: this contract
    was frozen before it met any real prediction, and reading a distribution
    off it here is the first step toward quietly tuning the thresholds.
    """
    import json
    import pathlib
    results = (pathlib.Path(__file__).resolve().parent.parent.parent
               / "data" / "crypto" / "benchmark_results_v1" / "BTC-USD.json")
    if not results.is_file():
        pytest.skip("the V1 benchmark has not been run in this checkout")
    stored = json.loads(results.read_bytes().decode("utf-8"))
    records = [Record(r["bar_open_at"], r["prediction"], r["actual_forward_return"])
               for r in stored["oos_records"][:200]]
    series = engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                                     fitted_hash=FITTED_HASH,
                                     benchmark_spec_hash=stored["benchmark_spec_hash"])
    assert series.count == len(records)
    assert all(d.direction in engine.SignalDirection.ALL for d in series.decisions)
    assert all(Decimal(0) <= d.strength <= Decimal(1) for d in series.decisions)
    assert all(d.source_benchmark_spec_hash == stored["benchmark_spec_hash"]
               for d in series.decisions)
    again = engine.generate_signals(records, model_spec_hash=MODEL_HASH,
                                    fitted_hash=FITTED_HASH,
                                    benchmark_spec_hash=stored["benchmark_spec_hash"])
    assert again.series_hash == series.series_hash
