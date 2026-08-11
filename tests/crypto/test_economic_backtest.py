"""Phase 5C: the economic simulator, on synthetic markets only.

Every fixture here is hand-built so the expected answer can be worked out on
paper. No real corpus is scored in this file: the engine has to be shown
correct before it is allowed to produce a number anybody might quote.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
import importlib

import pytest


HOUR = timedelta(hours=1)
START = datetime(2026, 1, 5, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    return importlib.import_module("scripts.trading_lab.economic_backtest")


@dataclass(frozen=True)
class FakeTarget:
    """Stands in for a risk-engine PositionTarget; only two fields are read."""

    timestamp: str
    target_exposure: Decimal

    @property
    def position_target_hash(self) -> str:
        return f"target-{self.timestamp}-{self.target_exposure}"


def _iso(index: int) -> str:
    return (START + HOUR * index).isoformat()


def _bars(prices, *, skip=()):
    return [{"bar_open_at": _iso(index), "open": str(price)}
            for index, price in enumerate(prices) if index not in skip]


def _targets(pairs):
    return [FakeTarget(timestamp=_iso(index), target_exposure=Decimal(str(exposure)))
            for index, exposure in pairs]


def _spec(engine, execution_spec=None):
    execution = execution_spec or engine.EXECUTION_SPEC_V1
    return engine.EconomicBacktestSpec(
        protocol_version=engine.ECONOMIC_BACKTEST_SCHEMA_VERSION,
        product="BTC-USD", timeframe="1h",
        source_benchmark_protocol="trading-lab.real-benchmark.v2",
        source_benchmark_spec_hash="a" * 64,
        source_benchmark_results_hash="b" * 64,
        signal_spec_hash="c" * 64, risk_spec_hash="d" * 64,
        execution_spec_hash=execution.execution_spec_hash,
        market_corpus_spec_hash="e" * 64, market_corpus_content_hash="f" * 64,
        result_schema_version=engine.ECONOMIC_RESULT_SCHEMA_VERSION)


def _run(engine, targets, bars, execution_spec=None, bars_product="BTC-USD"):
    execution = execution_spec or engine.EXECUTION_SPEC_V1
    class _Series:
        def __init__(self, entries): self.targets = tuple(entries)
        series_hash = "series-hash"
    return engine.run_economic_backtest(
        product="BTC-USD", targets=_Series(targets), bars=bars,
        spec=_spec(engine, execution), signal_series_hash="signal-hash",
        execution_spec=execution, bars_product=bars_product)


# --- A: doing nothing costs nothing ---------------------------------------


def test_a_permanently_flat_target_never_trades_and_never_loses(engine):
    bars = _bars([100, 110, 90, 105, 100])
    targets = _targets([(0, 0), (1, 0), (2, 0)])
    result = _run(engine, targets, bars)
    assert result.metrics.fill_count == 0
    assert result.metrics.total_fees == Decimal(0)
    assert result.metrics.total_slippage_cost == Decimal(0)
    assert result.metrics.turnover_ratio == Decimal(0)
    assert result.metrics.final_equity == engine.INITIAL_EQUITY_V1
    assert result.metrics.net_return == Decimal(0)
    assert result.metrics.max_drawdown == Decimal(0)
    assert result.metrics.exposure_time_fraction == Decimal(0)
    assert all(point.equity == engine.INITIAL_EQUITY_V1 for point in result.equity_curve)


# --- B..D: direction and sign ---------------------------------------------


def test_a_long_in_a_rising_market_makes_money_before_costs(engine):
    bars = _bars([100, 100, 110, 110])
    result = _run(engine, _targets([(0, "0.25")]), bars,
                  engine.GROSS_EXECUTION_SPEC_V1)
    assert result.metrics.fill_count == 2            # entry, then liquidation
    assert result.metrics.net_pnl > 0
    # 25 % of 100k at 100 = 250 units; +10 % move on a quarter of NAV = +2.5 %
    assert result.metrics.net_return == Decimal("0.025")


def test_a_short_in_a_falling_market_makes_money_before_costs(engine):
    bars = _bars([100, 100, 90, 90])
    result = _run(engine, _targets([(0, "-0.25")]), bars,
                  engine.GROSS_EXECUTION_SPEC_V1)
    assert result.metrics.net_pnl > 0
    assert result.metrics.net_return == Decimal("0.025")
    assert result.equity_curve[0].position_quantity < 0


def test_a_long_in_a_falling_market_loses_money(engine):
    bars = _bars([100, 100, 90, 90])
    result = _run(engine, _targets([(0, "0.25")]), bars,
                  engine.GROSS_EXECUTION_SPEC_V1)
    assert result.metrics.net_pnl < 0
    assert result.metrics.net_return == Decimal("-0.025")
    assert result.metrics.max_drawdown < 0


# --- E, F: costs only ever subtract ---------------------------------------


def test_fees_and_slippage_can_only_reduce_the_result(engine):
    bars = _bars([100, 100, 110, 110])
    targets = _targets([(0, "0.25")])
    gross = _run(engine, targets, bars, engine.GROSS_EXECUTION_SPEC_V1)
    net = _run(engine, targets, bars)
    assert net.metrics.net_return < gross.metrics.net_return
    assert net.metrics.total_fees > 0 and net.metrics.total_slippage_cost > 0
    assert net.metrics.total_execution_cost == \
        net.metrics.total_fees + net.metrics.total_slippage_cost
    # the same run reports the free path as its gross view
    assert net.metrics.gross_return == gross.metrics.net_return


def test_fees_alone_and_slippage_alone_each_cost_something(engine):
    bars = _bars([100, 100, 110, 110])
    targets = _targets([(0, "0.25")])
    free = _run(engine, targets, bars, engine.GROSS_EXECUTION_SPEC_V1)
    fee_only = _run(engine, targets, bars, replace(
        engine.EXECUTION_SPEC_V1, slippage_rate=Decimal(0)))
    slip_only = _run(engine, targets, bars, replace(
        engine.EXECUTION_SPEC_V1, fee_rate=Decimal(0)))
    assert fee_only.metrics.net_return < free.metrics.net_return
    assert slip_only.metrics.net_return < free.metrics.net_return
    assert fee_only.metrics.total_slippage_cost == 0
    assert slip_only.metrics.total_fees == 0


def test_a_buy_pays_up_and_a_sell_receives_less(engine):
    """Slippage always moves the price against the trade, in both directions."""
    bars = _bars([100, 100, 100, 100])
    long_run = _run(engine, _targets([(0, "0.25")]), bars)
    short_run = _run(engine, _targets([(0, "-0.25")]), bars)
    buy = long_run.fills[0]
    sell = short_run.fills[0]
    assert buy.side == "buy" and buy.fill_price > buy.reference_price
    assert sell.side == "sell" and sell.fill_price < sell.reference_price
    assert buy.fill_price == Decimal(100) * (Decimal(1) + engine.SLIPPAGE_RATE_V1)
    assert sell.fill_price == Decimal(100) * (Decimal(1) - engine.SLIPPAGE_RATE_V1)


# --- G: no pointless trading ----------------------------------------------


def test_repeating_the_same_target_in_a_still_market_does_not_trade_again(engine):
    """Only well defined at zero cost: exposure is a fraction of NAV, so once
    fees have moved NAV even a repeated target implies a genuine rebalance."""
    bars = _bars([100, 100, 100, 100, 100, 100])
    targets = _targets([(0, "0.25"), (1, "0.25"), (2, "0.25"), (3, "0.25")])
    result = _run(engine, targets, bars, engine.GROSS_EXECUTION_SPEC_V1)
    assert result.metrics.rebalance_count == 4       # four targets became executable
    assert result.metrics.fill_count == 2            # one entry, one liquidation
    assert [fill.side for fill in result.fills] == ["buy", "sell"]

    # with costs the NAV drifts, so the same exposure needs a small correction
    with_costs = _run(engine, targets, bars)
    assert with_costs.metrics.fill_count > 2


# --- H: reversals ----------------------------------------------------------


@pytest.mark.parametrize("first,second,sides", [
    ("0.25", "-0.10", ["buy", "sell", "buy"]),       # long -> short -> flat
    ("-0.25", "0.10", ["sell", "buy", "sell"]),      # short -> long -> flat
    ("0.25", "0", ["buy", "sell"]),                  # long -> flat
    ("-0.25", "0", ["sell", "buy"]),                 # short -> flat
])
def test_a_reversal_is_one_net_fill(engine, first, second, sides):
    bars = _bars([100, 100, 100, 100, 100])
    result = _run(engine, _targets([(0, first), (1, second)]), bars)
    assert [fill.side for fill in result.fills] == sides
    if second != "0":
        crossing = result.fills[1]
        # a single delta carries the close and the re-open, and pays cost on both
        assert crossing.notional > 0
        assert (Decimal(first) > 0) != (crossing.position_after > 0)
    assert result.equity_curve[-1].position_quantity == 0


# --- I, J: gaps ------------------------------------------------------------


def test_a_target_whose_next_bar_is_missing_expires(engine):
    """Executing it later would trade on an intention nobody had at that price."""
    bars = _bars([100, 100, 100, 100, 100, 100], skip={3})
    result = _run(engine, _targets([(0, "0.25"), (2, "-0.25")]), bars)
    assert result.metrics.expired_target_count == 1
    expired = result.expired_targets[0]
    assert expired.timestamp == _iso(2)
    assert expired.expected_fill_at == _iso(3)
    assert "contiguous" in expired.reason
    assert [fill.timestamp for fill in result.fills][0] == _iso(1)


def test_a_position_is_carried_across_a_gap_not_liquidated_before_it(engine):
    """Closing before the hole would use the knowledge that the hole is coming."""
    bars = _bars([100, 100, 100, 120, 120, 120, 120], skip={2})
    result = _run(engine, _targets([(0, "0.25"), (3, "0.25")]), bars)
    stamps = [point.timestamp for point in result.equity_curve]
    assert _iso(2) not in stamps                     # no invented bar in the gap
    entry = result.fills[0]
    assert entry.timestamp == _iso(1)
    carried = next(point for point in result.equity_curve if point.timestamp == _iso(3))
    assert carried.position_quantity == entry.position_after
    assert carried.mark_price == Decimal(120)        # revalued when trading resumes


# --- K, L: liquidation and the causal boundary ----------------------------


def test_the_position_is_liquidated_at_the_next_observable_open(engine):
    bars = _bars([100, 100, 100, 130])
    result = _run(engine, _targets([(1, "0.25")]), bars)
    last = result.fills[-1]
    assert last.timestamp == _iso(3) == result.window["liquidation_at"]
    assert last.reference_price == Decimal(130)
    assert result.equity_curve[-1].position_quantity == 0
    assert result.equity_curve[-1].timestamp == _iso(3)


def test_a_corpus_that_ends_at_the_last_fill_refuses_to_invent_a_price(engine):
    bars = _bars([100, 100])
    with pytest.raises(engine.EconomicBacktestError, match="final liquidation price"):
        _run(engine, _targets([(0, "0.25")]), bars)


def test_a_decision_can_never_be_filled_on_its_own_bar(engine):
    """The candle behind the prediction is already finished when it is known."""
    bars = _bars([100, 500, 500, 500])
    result = _run(engine, _targets([(0, "0.25")]), bars)
    entry = result.fills[0]
    assert entry.timestamp == _iso(1)
    assert entry.reference_price == Decimal(500)     # the NEXT open, not 100
    assert all(fill.timestamp != _iso(0) for fill in result.fills)
    assert all(point.timestamp != _iso(0) for point in result.equity_curve)


# --- M, and no future target -----------------------------------------------


def test_bars_added_after_the_liquidation_change_nothing(engine):
    targets = _targets([(0, "0.25")])
    short = _run(engine, targets, _bars([100, 100, 110, 115]))
    long = _run(engine, targets, _bars([100, 100, 110, 115, 900, 5, 4000]))
    assert [fill.canonical() for fill in short.fills] == \
           [fill.canonical() for fill in long.fills]
    assert [p.canonical() for p in short.equity_curve] == \
           [p.canonical() for p in long.equity_curve]
    assert short.metrics.canonical() == long.metrics.canonical()
    assert short.economic_results_hash == long.economic_results_hash


def test_a_later_target_cannot_move_an_earlier_snapshot(engine):
    bars = _bars([100, 100, 100, 100, 100, 100, 100])
    base = _run(engine, _targets([(0, "0.25")]), bars)
    extended = _run(engine, _targets([(0, "0.25"), (4, "-1")]), bars)
    # Compare only before the earlier run's liquidation: the final liquidation is
    # defined relative to the END of the target stream, so extending the stream
    # legitimately moves it. What must not move is the position itself.
    boundary = base.window["liquidation_at"]
    early = [p.canonical() for p in base.equity_curve if p.timestamp < boundary]
    later = [p.canonical() for p in extended.equity_curve if p.timestamp < boundary]
    assert early == later and early


# --- accounting invariants -------------------------------------------------


def test_every_snapshot_balances_and_costs_only_accumulate(engine):
    bars = _bars([100, 104, 97, 111, 88, 120, 95, 130])
    result = _run(engine, _targets([(0, "0.25"), (1, "-0.20"), (3, "0.15"), (5, "0")]),
                  bars)
    previous_fees = Decimal(0)
    previous_slippage = Decimal(0)
    from decimal import localcontext
    for point in result.equity_curve:
        with localcontext() as context:
            # the engine balances the books at ITS precision, not the caller's
            context.prec = engine.ECONOMIC_PRECISION
            assert point.equity == point.cash + point.position_quantity * point.mark_price
            assert point.position_value == point.position_quantity * point.mark_price
        assert point.cumulative_fees >= previous_fees
        assert point.cumulative_slippage_cost >= previous_slippage
        assert point.equity > 0
        assert point.equity.is_finite() and point.cash.is_finite()
        previous_fees = point.cumulative_fees
        previous_slippage = point.cumulative_slippage_cost
    for fill in result.fills:
        assert fill.notional > 0 and fill.fee >= 0 and fill.slippage_cost >= 0
        assert (fill.quantity_delta > 0) == (fill.side == "buy")
    assert result.metrics.total_fees == result.equity_curve[-1].cumulative_fees


def test_an_insolvent_account_fails_closed_instead_of_dividing_by_nothing(engine):
    """A blown-up synthetic account stops the run; it does not keep trading."""
    # a fully short book against a tenfold rally wipes the synthetic account out
    bars = _bars([100, 100, 1000, 1000])
    with pytest.raises(engine.EconomicBacktestError, match="insolvent"):
        _run(engine, _targets([(0, "-1")]), bars)


def test_the_turnover_ratio_sums_traded_notional_over_pre_trade_equity(engine):
    bars = _bars([100, 100, 100, 100])
    result = _run(engine, _targets([(0, "0.25")]), bars,
                  engine.GROSS_EXECUTION_SPEC_V1)
    # entry moves 25 % of equity, the liquidation moves it back
    assert result.metrics.turnover_ratio == Decimal("0.50")


def test_the_maximum_drawdown_is_negative_and_measured_from_the_running_peak(engine):
    # the position must still be held when the market falls, so the stream keeps
    # going past the peak rather than liquidating at it
    bars = _bars([100, 100, 200, 100, 150])
    result = _run(engine, _targets([(0, "1"), (2, "1")]), bars,
                  engine.GROSS_EXECUTION_SPEC_V1)
    from decimal import localcontext
    assert result.metrics.max_drawdown < 0
    with localcontext() as context:
        context.prec = engine.ECONOMIC_PRECISION
        worst = Decimal(0)
        peak = None
        for point in result.equity_curve:
            peak = point.equity if peak is None or point.equity > peak else peak
            worst = min(worst, point.equity / peak - Decimal(1))
    assert result.metrics.max_drawdown == worst


def test_the_sharpe_is_undefined_when_there_is_nothing_to_measure(engine):
    flat = _bars([100, 100, 100, 100])
    result = _run(engine, _targets([(0, 0)]), flat)
    assert result.metrics.annualized_sharpe is None   # zero variance
    assert result.metrics.canonical()["periods_per_year"] == 8760


# --- determinism and identity ----------------------------------------------


def test_the_result_hash_is_deterministic_and_covers_the_fills(engine):
    bars = _bars([100, 103, 97, 111, 120])
    targets = _targets([(0, "0.25"), (2, "-0.10")])
    first = _run(engine, targets, bars)
    second = _run(engine, targets, bars)
    assert first.economic_results_hash == second.economic_results_hash
    assert replace(first, fills=first.fills[:-1]).economic_results_hash != \
        first.economic_results_hash
    assert replace(first, equity_curve=first.equity_curve[:-1]).economic_results_hash != \
        first.economic_results_hash
    assert first.economic_results_hash != first.spec.economic_backtest_spec_hash


def test_the_simulation_ignores_the_callers_decimal_context(engine):
    bars = _bars([100, 103, 97, 111, 120])
    targets = _targets([(0, "0.25"), (2, "-0.10")])
    original = getcontext().prec
    seen = set()
    try:
        for precision in (7, 28, 34, 60):
            getcontext().prec = precision
            seen.add(_run(engine, targets, bars).economic_results_hash)
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_the_execution_spec_is_the_frozen_v1_contract(engine):
    spec = engine.EXECUTION_SPEC_V1
    spec.validate()
    assert spec.fee_rate == Decimal("0.0010")
    assert spec.slippage_rate == Decimal("0.0005")
    assert spec.initial_equity == Decimal("100000")
    assert spec.currency == "USD"
    assert spec.fill_policy == "next-contiguous-bar-open-after-decision-v1"
    assert spec.mark_policy == "next-observable-open-v1"
    assert spec.instrument_model == "synthetic-linear-usd-notional-v1"
    assert spec.allow_long and spec.allow_short
    assert spec.funding_rate == 0 and spec.borrow_rate == 0
    assert len(spec.execution_spec_hash) == 64
    assert engine.ExecutionSpec().execution_spec_hash == spec.execution_spec_hash
    assert spec.canonical()["cost_model"] == "synthetic"
    assert spec.canonical()["optimized"] is False
    assert spec.canonical()["exchange_account_specific"] is False


def test_a_funding_or_borrow_rate_is_refused_because_v1_models_neither(engine):
    for field in ("funding_rate", "borrow_rate"):
        with pytest.raises(engine.EconomicBacktestError, match="funding or borrow"):
            replace(engine.EXECUTION_SPEC_V1, **{field: Decimal("0.01")}).validate()


def test_the_execution_spec_hash_moves_with_every_cost(engine):
    baseline = engine.EXECUTION_SPEC_V1.execution_spec_hash
    for field, value in (("fee_rate", Decimal("0.002")),
                         ("slippage_rate", Decimal("0.001")),
                         ("initial_equity", Decimal("50000")),
                         ("currency", "EUR"),
                         ("allow_short", False)):
        assert replace(engine.EXECUTION_SPEC_V1,
                       **{field: value}).execution_spec_hash != baseline


def test_the_backtest_spec_hash_covers_every_upstream_contract(engine):
    spec = _spec(engine)
    baseline = spec.economic_backtest_spec_hash
    for field in ("product", "source_benchmark_spec_hash", "source_benchmark_results_hash",
                  "signal_spec_hash", "risk_spec_hash", "execution_spec_hash",
                  "market_corpus_content_hash", "market_corpus_spec_hash"):
        assert replace(spec, **{field: "z" * 64}).economic_backtest_spec_hash != baseline


# --- label isolation --------------------------------------------------------


def test_signal_generation_cannot_read_the_realised_label(engine):
    """Profit must come from prices, never from the label used to score models."""
    view = engine.PredictionView({"bar_open_at": _iso(0), "prediction": "0.01",
                                  "fold_index": "0"})
    assert view.prediction == Decimal("0.01")
    with pytest.raises(AssertionError, match="realised label"):
        _ = view.actual_forward_return

    signals, targets = engine.build_position_targets(
        [{"bar_open_at": _iso(index), "prediction": "0.01", "fold_index": "0"}
         for index in range(3)],
        model_spec_hash="a" * 64, fitted_hash="b" * 64, benchmark_spec_hash="c" * 64)
    assert len(targets.targets) == 3
    assert all(target.target_exposure > 0 for target in targets.targets)


def test_malformed_inputs_are_refused(engine):
    bars = _bars([100, 100, 100])
    with pytest.raises(engine.EconomicBacktestError, match="ascending"):
        engine.simulate_targets(_targets([(1, "0.1"), (0, "0.1")]), bars)
    with pytest.raises(engine.EconomicBacktestError, match="duplicate"):
        engine.simulate_targets([FakeTarget(_iso(0), Decimal("0.1")),
                                 FakeTarget(_iso(0), Decimal("0.2"))], bars)
    with pytest.raises(engine.EconomicBacktestError, match="no market bars"):
        engine.simulate_targets(_targets([(0, "0.1")]), [])
    with pytest.raises(engine.EconomicBacktestError, match="became executable"):
        engine.simulate_targets(_targets([(9, "0.1")]), bars)
    with pytest.raises(engine.EconomicBacktestError, match="ascending order"):
        engine.simulate_targets(_targets([(0, "0.1")]), list(reversed(_bars([1, 2, 3]))))


def test_a_backtest_refuses_bars_from_another_instrument(engine):
    """Targets carry spec hashes and timestamps but no instrument, and a bar is
    six numbers. Nothing said which market either belonged to, so one target
    stream ran against another instrument's prices and produced a complete,
    plausible, differently-valued result labelled with the first product."""
    from scripts.trading_lab.identity import InstrumentMismatchError

    bars = _bars([100, 110, 90, 105, 100])
    targets = _targets([(0, 1), (1, 0), (2, 0)])
    with pytest.raises(InstrumentMismatchError):
        _run(engine, targets, bars, bars_product="ETH-USD")


def test_a_backtest_accepts_any_spelling_of_the_matching_instrument(engine):
    bars = _bars([100, 110, 90, 105, 100])
    targets = _targets([(0, 1), (1, 0), (2, 0)])
    canonical = _run(engine, targets, bars, bars_product="BTC-USD")
    for alias in ("btc-usd", "coinbase:BTC-USD", "BTCUSD"):
        assert _run(engine, targets, bars, bars_product=alias).metrics.final_equity \
            == canonical.metrics.final_equity
