"""Phase 6B: the multi-instrument portfolio engine, on synthetic data.

Not marked `ml`: shared-capital accounting must be provable without the model
stack.

Two properties carry most of the weight here.

**Order independence.** If BTC is filled before ETH and ETH is then sized from
the changed equity, the portfolio depends on the order two instruments happen
to appear in a list. That is not a rounding difference -- it is a different
portfolio, and it would be invisible in any single run. Several tests permute
the inputs and demand an identical result hash.

**Attribution reconciliation.** Per-instrument contributions that do not add
up to the change in portfolio equity are a story about the result rather than
a decomposition of it.
"""

from __future__ import annotations

import itertools
from decimal import Decimal, getcontext, localcontext

import pytest

BTC = "coinbase:BTC-USD"
ETH = "coinbase:ETH-USD"


@pytest.fixture
def portfolio():
    import importlib
    return importlib.import_module("scripts.trading_lab.portfolio")


def _frame(portfolio, timestamp, **prices):
    return portfolio.PortfolioMarketFrame(
        timestamp=timestamp,
        prices=tuple((name, Decimal(str(value))) for name, value in prices.items()))


def _targets(portfolio, timestamp, **exposures):
    return portfolio.PortfolioTargetSet(
        timestamp=timestamp,
        portfolio_spec_hash=portfolio.PORTFOLIO_SPEC_V1.portfolio_spec_hash,
        targets=tuple(
            portfolio.InstrumentPositionTarget(
                instrument_id=name, target_exposure=Decimal(str(value)),
                source_position_target_hash=f"{name}-{timestamp}")
            for name, value in exposures.items()))


def _run(portfolio, batches, **kwargs):
    return portfolio.run_portfolio_backtest(batches, **kwargs)


def _series(portfolio, prices, exposures):
    """(frame, target_set) pairs from parallel price and exposure tables."""
    batches = []
    for index, (timestamp, quotes) in enumerate(prices):
        frame = _frame(portfolio, timestamp, **quotes)
        wanted = exposures[index] if index < len(exposures) else None
        batches.append((frame, _targets(portfolio, timestamp, **wanted)
                        if wanted is not None else None))
    return batches


# --- A: doing nothing costs nothing ---------------------------------------


def test_two_flat_instruments_leave_equity_untouched(portfolio):
    prices = [(f"2026-01-01T{hour:02d}:00:00+00:00",
               {BTC: 60000 + hour * 100, ETH: 3000 - hour * 10})
              for hour in range(6)]
    result = _run(portfolio, _series(portfolio, prices, [{BTC: 0, ETH: 0}] * 6))
    assert result.metrics.fill_count == 0
    assert result.metrics.final_equity == portfolio.PORTFOLIO_SPEC_V1.initial_equity
    assert result.metrics.total_execution_cost == Decimal(0)
    assert result.metrics.net_return == Decimal(0)
    assert all(snapshot.gross_exposure == Decimal(0)
               for snapshot in result.equity_curve)


# --- B-E: positions in combination ----------------------------------------


def test_one_long_and_one_flat_moves_equity_with_that_one_instrument(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 50}),
              ("2026-01-01T02:00:00+00:00", {BTC: 110, ETH: 50})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: 0}, None, None]))
    assert result.metrics.fill_count == 1
    # 25 % of 100 000 long into a +10 % move
    assert result.metrics.gross_pnl > Decimal(2000)
    assert result.metrics.net_pnl < result.metrics.gross_pnl


def test_a_long_and_a_short_can_be_held_at_once(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45}),
              ("2026-01-01T02:00:00+00:00", {BTC: 110, ETH: 45})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: "-0.25"}, None, None]))
    assert result.metrics.fill_count == 2
    positions = {name: quantity
                 for name, quantity, _, _ in result.equity_curve[0].positions}
    assert positions[BTC] > 0 and positions[ETH] < 0
    # both moves were favourable, so gross P&L is positive on both legs
    contributions = {record.instrument_id: record.gross_pnl
                     for record in result.attribution}
    assert contributions[BTC] > 0 and contributions[ETH] > 0


def test_opposite_returns_partly_offset_inside_one_portfolio(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 120, ETH: 40}),
              ("2026-01-01T02:00:00+00:00", {BTC: 120, ETH: 40})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: "0.25"}, None, None]))
    contributions = {record.instrument_id: record.gross_pnl
                     for record in result.attribution}
    assert contributions[BTC] > 0 and contributions[ETH] < 0
    assert result.metrics.gross_pnl == contributions[BTC] + contributions[ETH]


# --- F-G: shared capital, one pre-trade equity ----------------------------


def test_equity_is_cash_plus_every_marked_position(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 105, ETH: 55}),
              ("2026-01-01T02:00:00+00:00", {BTC: 95, ETH: 60})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.2", ETH: "0.2"}] * 3))
    with localcontext() as context:
        context.prec = portfolio.ECONOMIC_PRECISION
        for snapshot in result.equity_curve:
            rebuilt = snapshot.cash + sum(
                (quantity * price
                 for _, quantity, price, _ in snapshot.positions), Decimal(0))
            assert rebuilt == snapshot.equity


def test_there_is_one_cash_ledger_not_one_per_instrument(portfolio):
    """Buying BTC must reduce the cash available to ETH."""
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 100, ETH: 50})]
    only_btc = _run(portfolio, _series(portfolio, prices,
                                       [{BTC: "0.25", ETH: 0}, None]))
    both = _run(portfolio, _series(portfolio, prices,
                                   [{BTC: "0.25", ETH: "0.25"}, None]))
    assert both.equity_curve[0].cash < only_btc.equity_curve[0].cash


def test_every_target_in_a_batch_is_sized_from_the_same_pre_trade_equity(portfolio):
    """The bias this prevents: ETH sized from equity that BTC's fill moved."""
    spec = portfolio.PORTFOLIO_SPEC_V1
    frame = _frame(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: 100, ETH: 50})
    outcome = portfolio.rebalance(
        portfolio.initial_portfolio_state(spec),
        target_set=_targets(portfolio, "2026-01-01T00:00:00+00:00",
                            **{BTC: "0.25", ETH: "0.25"}),
        frame=frame, spec=spec)
    assert outcome.pre_trade_equity == spec.initial_equity
    with localcontext() as context:
        context.prec = portfolio.ECONOMIC_PRECISION
        for fill in outcome.fills:
            expected = (Decimal("0.25") * spec.initial_equity) / fill.reference_price
            assert fill.quantity_delta == expected, fill.instrument_id


# --- H: order independence ------------------------------------------------


@pytest.mark.parametrize("order", list(itertools.permutations([BTC, ETH])))
def test_the_input_order_of_targets_changes_nothing(portfolio, order):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45}),
              ("2026-01-01T02:00:00+00:00", {BTC: 105, ETH: 47})]
    wanted = {BTC: "0.25", ETH: "-0.2"}
    baseline = _run(portfolio, _series(portfolio, prices, [wanted] * 3))

    permuted = []
    for timestamp, quotes in prices:
        frame = portfolio.PortfolioMarketFrame(
            timestamp=timestamp,
            prices=tuple((name, Decimal(str(quotes[name]))) for name in order))
        target_set = portfolio.PortfolioTargetSet(
            timestamp=timestamp,
            portfolio_spec_hash=portfolio.PORTFOLIO_SPEC_V1.portfolio_spec_hash,
            targets=tuple(
                portfolio.InstrumentPositionTarget(
                    instrument_id=name, target_exposure=Decimal(str(wanted[name])),
                    source_position_target_hash=f"{name}-{timestamp}")
                for name in order))
        permuted.append((frame, target_set))
    assert _run(portfolio, permuted).result_hash == baseline.result_hash


def test_permuting_market_frame_order_changes_nothing(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45})]
    wanted = {BTC: "0.25", ETH: "0.25"}
    forward = _run(portfolio, _series(portfolio, prices, [wanted, None]))
    reversed_prices = [(timestamp, dict(reversed(list(quotes.items()))))
                       for timestamp, quotes in prices]
    assert _run(portfolio, _series(portfolio, reversed_prices,
                                   [wanted, None])).result_hash == forward.result_hash


def test_fills_are_serialised_in_canonical_instrument_order(portfolio):
    frame = _frame(portfolio, "2026-01-01T00:00:00+00:00", **{ETH: 50, BTC: 100})
    outcome = portfolio.rebalance(
        portfolio.initial_portfolio_state(),
        target_set=_targets(portfolio, "2026-01-01T00:00:00+00:00",
                            **{ETH: "0.2", BTC: "0.2"}),
        frame=frame)
    assert [fill.instrument_id for fill in outcome.fills] == [BTC, ETH]


# --- I-J: the caps --------------------------------------------------------


def test_a_batch_over_the_gross_cap_is_scaled_proportionally(portfolio):
    """Never 'BTC first, ETH gets what is left'."""
    spec = portfolio.PORTFOLIO_SPEC_V1
    allocated, scale, requested = portfolio.allocate(
        _targets(portfolio, "2026-01-01T00:00:00+00:00",
                 **{BTC: "0.25", ETH: "-0.25"}), spec)
    assert requested == Decimal("0.5") and scale == Decimal(1)

    # a hypothetical three-instrument request would scale, so build one by
    # relaxing only the per-instrument cap, not the gross cap
    from dataclasses import replace
    loose = replace(spec, max_instrument_abs_exposure=Decimal("0.40"))
    allocated, scale, requested = portfolio.allocate(
        _targets(portfolio, "2026-01-01T00:00:00+00:00",
                 **{BTC: "0.40", ETH: "0.40"}), loose)
    assert requested == Decimal("0.8")
    assert scale == Decimal("0.5") / Decimal("0.8")
    exposures = {name: exposure for name, exposure, _ in allocated}
    # equal requests stay equal after scaling; neither is favoured
    assert exposures[BTC] == exposures[ETH]
    assert sum(abs(value) for value in exposures.values()) == Decimal("0.5")


def test_scaling_preserves_the_ratio_between_unequal_requests(portfolio):
    from dataclasses import replace
    spec = replace(portfolio.PORTFOLIO_SPEC_V1,
                   max_instrument_abs_exposure=Decimal("0.40"))
    allocated, scale, _ = portfolio.allocate(
        _targets(portfolio, "2026-01-01T00:00:00+00:00",
                 **{BTC: "0.30", ETH: "0.10"}), spec)
    exposures = {name: exposure for name, exposure, _ in allocated}
    assert exposures[BTC] / exposures[ETH] == Decimal(3)


def test_an_instrument_over_its_own_cap_fails_closed(portfolio):
    with pytest.raises(portfolio.PortfolioLimitBreached):
        portfolio.allocate(_targets(portfolio, "2026-01-01T00:00:00+00:00",
                                    **{BTC: "0.30", ETH: "0.10"}))


def test_a_net_cap_that_cannot_be_met_fails_rather_than_inventing_a_rule(portfolio):
    from dataclasses import replace
    spec = replace(portfolio.PORTFOLIO_SPEC_V1,
                   max_net_abs_exposure=Decimal("0.10"))
    with pytest.raises(portfolio.PortfolioLimitBreached) as error:
        portfolio.allocate(_targets(portfolio, "2026-01-01T00:00:00+00:00",
                                    **{BTC: "0.25", ETH: "0.25"}), spec)
    assert "cannot satisfy both caps" in str(error.value)


def test_realised_exposure_may_drift_from_the_cap_but_only_by_the_cost(portfolio):
    """The cap governs what is allocated and executed, not what prices do next.

    Sizing uses pre-trade equity; paying the execution cost lowers equity, so
    realised exposure lands a hair above the cap on every trade. Enforcing the
    cap on realised exposure would report a breach on every single fill, and
    enforcing it continuously afterwards would be a different strategy that
    trades on price moves alone.
    """
    result = _run(portfolio, _series(
        portfolio,
        [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
         ("2026-01-01T01:00:00+00:00", {BTC: 130, ETH: 51})],
        [{BTC: "0.25", ETH: "0.25"}, None]))
    spec = portfolio.PORTFOLIO_SPEC_V1
    at_trade = result.equity_curve[0].gross_exposure
    assert at_trade > spec.max_gross_exposure          # by the cost, not more
    assert at_trade - spec.max_gross_exposure < Decimal("0.001")
    # and a later price move is free to carry it further, without a rebalance
    assert result.equity_curve[1].gross_exposure > at_trade
    assert result.metrics.fill_count == 2


# --- K-L: identity guards before any arithmetic ---------------------------


def test_a_target_cannot_be_executed_against_another_instruments_price(portfolio):
    """BTC target x ETH open. The 6A-R failure, at a new boundary."""
    frame = _frame(portfolio, "2026-01-01T00:00:00+00:00", **{ETH: 50})
    with pytest.raises(portfolio.PortfolioValuationUnavailable):
        portfolio.rebalance(
            portfolio.initial_portfolio_state(),
            target_set=_targets(portfolio, "2026-01-01T00:00:00+00:00",
                                **{BTC: "0.25"}),
            frame=frame)


def test_an_unregistered_instrument_never_reaches_the_accounting(portfolio):
    with pytest.raises(portfolio.PortfolioIdentityMismatch):
        _targets(portfolio, "2026-01-01T00:00:00+00:00", **{"coinbase:SOL-USD": "0.1"})
    with pytest.raises(portfolio.PortfolioIdentityMismatch):
        _frame(portfolio, "2026-01-01T00:00:00+00:00", **{"nasdaq:AAPL": 100})


@pytest.mark.parametrize("alias", ["BTC-USD", "btc-usd", "BTCUSD", "BTC/USD"])
def test_any_spelling_resolves_to_the_one_canonical_instrument(portfolio, alias):
    target_set = _targets(portfolio, "2026-01-01T00:00:00+00:00", **{alias: "0.1"})
    assert target_set.instruments == (BTC,)


def test_the_same_instrument_twice_in_a_batch_fails_closed(portfolio):
    with pytest.raises(portfolio.PortfolioIdentityMismatch):
        portfolio.PortfolioTargetSet(
            timestamp="2026-01-01T00:00:00+00:00",
            portfolio_spec_hash=portfolio.PORTFOLIO_SPEC_V1.portfolio_spec_hash,
            targets=(
                portfolio.InstrumentPositionTarget(
                    instrument_id="BTC-USD", target_exposure=Decimal("0.1"),
                    source_position_target_hash="a"),
                portfolio.InstrumentPositionTarget(
                    instrument_id="btc-usd", target_exposure=Decimal("0.2"),
                    source_position_target_hash="b")))


def test_the_same_instrument_priced_twice_fails_closed(portfolio):
    with pytest.raises(portfolio.PortfolioIdentityMismatch):
        portfolio.PortfolioMarketFrame(
            timestamp="2026-01-01T00:00:00+00:00",
            prices=(("BTC-USD", Decimal(100)), ("btcusd", Decimal(200))))


def test_a_batch_cannot_execute_against_another_timestamps_prices(portfolio):
    with pytest.raises(portfolio.PortfolioError):
        portfolio.rebalance(
            portfolio.initial_portfolio_state(),
            target_set=_targets(portfolio, "2026-01-01T00:00:00+00:00",
                                **{BTC: "0.25"}),
            frame=_frame(portfolio, "2026-01-01T01:00:00+00:00", **{BTC: 100}))


# --- M-N: valuation gaps --------------------------------------------------


def test_an_unpriceable_open_position_produces_no_snapshot(portfolio):
    """No forward-fill. An equity nobody can compute is not reported."""
    batches = [
        (_frame(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: 100, ETH: 50}),
         _targets(portfolio, "2026-01-01T00:00:00+00:00",
                  **{BTC: "0.25", ETH: "0.25"})),
        (_frame(portfolio, "2026-01-01T01:00:00+00:00", **{BTC: 110}), None),
        (_frame(portfolio, "2026-01-01T02:00:00+00:00", **{BTC: 120, ETH: 55}), None),
    ]
    result = _run(portfolio, batches)
    assert result.unavailable_valuations == ("2026-01-01T01:00:00+00:00",)
    timestamps = [snapshot.timestamp for snapshot in result.equity_curve]
    assert "2026-01-01T01:00:00+00:00" not in timestamps
    assert len(timestamps) == 2


def test_a_missing_price_is_never_replaced_by_the_previous_one(portfolio):
    """The gap is skipped, and the next real mark uses the real price."""
    batches = [
        (_frame(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: 100, ETH: 50}),
         _targets(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: "0.25", ETH: 0})),
        (_frame(portfolio, "2026-01-01T01:00:00+00:00", **{ETH: 50}), None),
        (_frame(portfolio, "2026-01-01T02:00:00+00:00", **{BTC: 200, ETH: 50}), None),
    ]
    result = _run(portfolio, batches)
    last = result.equity_curve[-1]
    marks = {name: price for name, _, price, _ in last.positions}
    assert marks[BTC] == Decimal(200)


def test_a_flat_position_does_not_require_a_price(portfolio):
    """Nothing held is worth nothing at any price, so no quote is needed."""
    batches = [
        (_frame(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: 100, ETH: 50}),
         _targets(portfolio, "2026-01-01T00:00:00+00:00", **{BTC: "0.25", ETH: 0})),
        (_frame(portfolio, "2026-01-01T01:00:00+00:00", **{BTC: 110}), None),
    ]
    result = _run(portfolio, batches)
    assert result.unavailable_valuations == ()
    assert len(result.equity_curve) == 2


# --- O-P: costs and attribution -------------------------------------------


def test_execution_costs_attach_to_the_instrument_that_incurred_them(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 100, ETH: 50})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: 0}, None]))
    records = {record.instrument_id: record for record in result.attribution}
    assert records[BTC].fees > 0 and records[BTC].slippage_cost > 0
    assert records[ETH].fees == Decimal(0) and records[ETH].slippage_cost == Decimal(0)
    assert records[BTC].fees + records[ETH].fees == result.metrics.total_fees


def test_net_contributions_reconcile_with_the_change_in_portfolio_equity(portfolio):
    """Attribution that does not add up is a story, not a decomposition."""
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 108, ETH: 46}),
              ("2026-01-01T02:00:00+00:00", {BTC: 103, ETH: 52}),
              ("2026-01-01T03:00:00+00:00", {BTC: 111, ETH: 49})]
    result = _run(portfolio, _series(
        portfolio, prices,
        [{BTC: "0.25", ETH: "-0.2"}, {BTC: "0.1", ETH: "0.15"},
         {BTC: "-0.2", ETH: "0.25"}, None]))
    assert result.attribution_reconciles()
    with localcontext() as context:
        context.prec = portfolio.ECONOMIC_PRECISION
        # the residual is last-digit rounding at 34 significant figures, not a
        # misplaced fee: orders of magnitude below anything meaningful
        residual = abs(result.reconciliation_residual()) / result.metrics.final_equity
        assert residual < Decimal("1E-30")
        # Gross P&L is derived two ways -- bottom-up from per-instrument
        # accruals, top-down as net P&L plus costs -- and they round
        # independently at the last digit. Both are correct; neither is a
        # rounding of the other.
        gross = sum((record.gross_pnl for record in result.attribution), Decimal(0))
        assert abs(gross - result.metrics.gross_pnl) / result.metrics.final_equity \
            < portfolio.ATTRIBUTION_RELATIVE_TOLERANCE


def test_the_gross_view_is_the_same_decisions_without_the_costs(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45}),
              ("2026-01-01T02:00:00+00:00", {BTC: 105, ETH: 48})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.2", ETH: "0.2"}] * 3))
    assert result.gross_metrics.total_execution_cost == Decimal(0)
    assert result.gross_metrics.fill_count == result.metrics.fill_count
    with localcontext() as context:
        context.prec = portfolio.ECONOMIC_PRECISION
        assert result.gross_metrics.final_equity == \
            result.metrics.final_equity + result.metrics.total_execution_cost


# --- Q-U: metrics, isolation, determinism ---------------------------------


def test_drawdown_is_measured_from_the_running_peak(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 140, ETH: 50}),
              ("2026-01-01T02:00:00+00:00", {BTC: 70, ETH: 50}),
              ("2026-01-01T03:00:00+00:00", {BTC: 90, ETH: 50})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: 0}, None, None, None]))
    assert result.metrics.max_drawdown < Decimal(0)
    with localcontext() as context:
        context.prec = portfolio.ECONOMIC_PRECISION
        peak = max(snapshot.equity for snapshot in result.equity_curve)
        trough = min(snapshot.equity for snapshot in result.equity_curve
                     if snapshot.timestamp >= "2026-01-01T01:00:00+00:00")
        assert result.metrics.max_drawdown == (trough - peak) / peak


def test_turnover_counts_traded_notional_against_initial_equity(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 100, ETH: 50})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: 0}, {BTC: 0, ETH: 0}]))
    assert result.metrics.portfolio_turnover > Decimal(0)
    assert result.metrics.fill_count == 2


def test_a_later_batch_cannot_alter_an_earlier_snapshot(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45}),
              ("2026-01-01T02:00:00+00:00", {BTC: 90, ETH: 60})]
    short = _run(portfolio, _series(portfolio, prices[:2],
                                    [{BTC: "0.25", ETH: "0.25"}, None]))
    long = _run(portfolio, _series(portfolio, prices,
                                   [{BTC: "0.25", ETH: "0.25"}, None, None]))
    assert long.equity_curve[0].canonical() == short.equity_curve[0].canonical()
    assert long.equity_curve[1].canonical() == short.equity_curve[1].canonical()


def test_the_result_does_not_depend_on_the_callers_decimal_context(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 107, ETH: 46})]
    wanted = [{BTC: "0.25", ETH: "-0.2"}, None]
    baseline = _run(portfolio, _series(portfolio, prices, wanted))
    original = getcontext().prec
    try:
        getcontext().prec = 6
        cramped = _run(portfolio, _series(portfolio, prices, wanted))
    finally:
        getcontext().prec = original
    assert cramped.result_hash == baseline.result_hash


def test_the_same_run_twice_produces_the_same_result_hash(portfolio):
    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45}),
              ("2026-01-01T02:00:00+00:00", {BTC: 105, ETH: 47})]
    wanted = [{BTC: "0.25", ETH: "-0.2"}] * 3
    first = _run(portfolio, _series(portfolio, prices, wanted))
    second = _run(portfolio, _series(portfolio, prices, wanted))
    assert first.result_hash == second.result_hash


def test_the_result_hash_covers_every_fill_and_snapshot(portfolio):
    import dataclasses

    prices = [("2026-01-01T00:00:00+00:00", {BTC: 100, ETH: 50}),
              ("2026-01-01T01:00:00+00:00", {BTC: 110, ETH: 45})]
    result = _run(portfolio, _series(portfolio, prices,
                                     [{BTC: "0.25", ETH: "0.25"}, None]))
    baseline = result.result_hash
    assert dataclasses.replace(result, fills=result.fills[:-1]).result_hash != baseline
    # and the hash must cover each fill's economics, not merely how many there
    # are: a hash that counts fills cannot notice a repriced one
    altered = dataclasses.replace(
        result.fills[0], fee=result.fills[0].fee + Decimal("1"))
    assert dataclasses.replace(
        result, fills=(altered,) + result.fills[1:]).result_hash != baseline
    moved = dataclasses.replace(
        result.fills[0], quantity_delta=result.fills[0].quantity_delta * 2)
    assert dataclasses.replace(
        result, fills=(moved,) + result.fills[1:]).result_hash != baseline
    assert dataclasses.replace(
        result, equity_curve=result.equity_curve[:-1]).result_hash != baseline
    assert dataclasses.replace(
        result, attribution=result.attribution[:-1]).result_hash != baseline


def test_the_specification_is_frozen_and_declares_it_is_not_optimised(portfolio):
    spec = portfolio.PORTFOLIO_SPEC_V1
    assert spec.optimized is False
    assert portfolio.PORTFOLIO_LIMIT_V1_IS_NOT_OPTIMIZED is True
    assert spec.initial_equity == Decimal("100000")
    assert spec.max_instrument_abs_exposure == Decimal("0.25")
    assert spec.max_gross_exposure == Decimal("0.50")
    assert spec.allocation_rule == "proportional-gross-cap-v1"
    assert spec.cash_model == "shared-cash-v1"
    with pytest.raises(Exception):
        spec.max_gross_exposure = Decimal("1")


def test_the_gross_cap_is_what_the_existing_risk_rules_already_allow(portfolio):
    """50 % is two instruments at RiskSpec V1's 25 % each, not a fitted number."""
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1

    spec = portfolio.PORTFOLIO_SPEC_V1
    assert spec.max_instrument_abs_exposure == RISK_SPEC_V1.max_long_exposure
    assert spec.max_gross_exposure == 2 * spec.max_instrument_abs_exposure
