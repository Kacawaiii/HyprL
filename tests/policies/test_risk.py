from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
from scripts.trading_lab.policies.demo import bar
from scripts.trading_lab.policies.risk import ProtectionPlan, protection_plan, simulate
from scripts.trading_lab.policies.spec import SPEC_HASH

ENTRY = "2026-06-25T00:00:00+00:00"
END = "2026-06-25T04:00:00+00:00"


def plan(**changes):
    args = dict(product="BTC-USD", mode="PAPER", side="LONG", entry_at=ENTRY, entry_fill="100",
                horizon_seconds=14400, source_prediction_hash="a" * 64, synthetic=True)
    return protection_plan(**{**args, **changes})


def run(candles, *, position=None, as_of=END):
    return simulate(position or plan(), candles, bar_seconds=3600, as_of=as_of)


def test_strategy_precedence_per_level_and_origin_method_identity():
    strategy = {"take_profit": {"value": "105", "source_hash": "b" * 64,
                               "method": "synthetic-range-v1", "provided_at": ENTRY}}
    result = plan(strategy=strategy)
    assert result.levels["take_profit"]["value"] == "105"
    assert result.levels["take_profit"]["origin"] == "STRATEGY"
    assert result.levels["take_profit"]["method"] == "synthetic-range-v1"
    assert result.levels["stop_loss"]["value"] == "99.00"
    assert result.levels["stop_loss"]["origin"] == "POLICY"
    assert result.levels["stop_loss"]["source_hash"] == SPEC_HASH
    assert plan(side="SHORT").levels["take_profit"]["value"] == "98.00"
    assert plan(side="SHORT").levels["stop_loss"]["value"] == "101.00"
    assert ProtectionPlan.from_dict(result.to_dict()).identity == result.identity


@pytest.mark.parametrize("side,ohlc,reason,reference", [
    ("LONG", (97, 100, 96, 98), "STOP_GAP", "97"),
    ("LONG", (104, 105, 100, 101), "TARGET_GAP", "102"),
    ("SHORT", (104, 105, 100, 101), "STOP_GAP", "104"),
    ("SHORT", (96, 100, 95, 99), "TARGET_GAP", "98"),
    ("LONG", (100, 102, 100, 101), "TARGET_TOUCH", "102"),
    ("SHORT", (100, 100, 98, 99), "TARGET_TOUCH", "98"),
])
def test_gaps_touch_equality_and_adverse_execution_costs(side, ohlc, reason, reference):
    result = run([bar(ENTRY, *ohlc)], position=plan(side=side))
    fill = result["exit"]
    assert result["state"] == "CLOSED"
    assert fill["reason"] == reason
    assert Decimal(fill["reference_price"]) == Decimal(reference)
    expected = Decimal(reference) * (1 - (Decimal(1) if side == "LONG" else Decimal(-1)) * EXECUTION_SPEC_V1.slippage_rate)
    assert Decimal(fill["fill_price"]) == expected
    assert Decimal(fill["net_pnl"]) < Decimal(fill["gross_pnl"])
    assert Decimal(fill["fees"]) == (Decimal(100) + expected) * EXECUTION_SPEC_V1.fee_rate
    assert fill["available_at"] == "2026-06-25T01:00:00+00:00"


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_intrabar_ambiguity_stop_first_with_target_sensitivity(side):
    result = run([bar(ENTRY, 100, 103, 97, 100)], position=plan(side=side))
    assert result["ambiguous"] is True
    assert result["exit"]["reason"] == "STOP_TOUCH"
    assert result["target_first_sensitivity"]["reason"] == "TARGET_FIRST_SENSITIVITY"
    assert Decimal(result["exit"]["net_pnl"]) < Decimal(result["target_first_sensitivity"]["net_pnl"])


def test_missing_bar_stops_without_later_price_or_pnl_claim():
    result = run([bar("2026-06-25T01:00:00Z", 95, 96, 94, 95)])
    assert result["state"] == "NOT_OBSERVED"
    assert result["missing_open_at"] == ENTRY
    assert result["exit"] is None
    assert result["bars_observed"] == 0


def test_future_bar_stays_pending_and_horizon_close_uses_complete_bars():
    candles = [bar((datetime.fromisoformat(ENTRY) + timedelta(hours=i)).isoformat(), 100, 101, 99.5, 100.5) for i in range(4)]
    assert run(candles, as_of=ENTRY)["state"] == "PENDING"
    pending = run(candles, as_of="2026-06-25T02:30:00Z")
    assert pending["state"] == "PENDING" and pending["exit"] is None and pending["bars_observed"] == 2
    closed = run(candles)
    assert closed["exit"]["reason"] == "HORIZON_CLOSE"
    assert closed["exit"]["available_at"] == END
    assert run(candles[:2])["state"] == "NOT_OBSERVED"


def test_invalid_strategy_does_not_silently_fall_back():
    for value, method, provided in (("98", "test-v1", ENTRY), ("105", "", ENTRY),
                                    ("105", "test-v1", END), ("nan", "test-v1", ENTRY)):
        with pytest.raises(ValueError):
            plan(strategy={"take_profit": {"value": value, "method": method, "provided_at": provided, "source_hash": "b" * 64}})
    with pytest.raises(ValueError):
        plan(strategy={"take_profit": {"value": "105"}})


def test_invalid_bars_and_non_paper_or_real_simulation_are_refused():
    for changes in ({"mode": "LIVE"}, {"synthetic": False}, {"side": "FLAT"}, {"entry_fill": True}):
        with pytest.raises(ValueError):
            plan(**changes)
    with pytest.raises(ValueError):
        run([bar(ENTRY, 100, 98, 97, 100)])
    with pytest.raises(ValueError):
        simulate(plan(), [], bar_seconds=5000, as_of=END)
    with pytest.raises(ValueError):
        replace(plan(), policy_hash="b" * 64)


def test_high_precision_loaded_plan_and_protected_range():
    position = plan(entry_fill="100.123456789012345678901234567890")
    assert ProtectionPlan.from_dict(position.to_dict()).identity == position.identity
    from scripts.trading_lab.research_protection import protection_table
    with pytest.raises(ValueError, match="protected"):
        plan(entry_at=protection_table()["BTC-USD"][0]["start"])
