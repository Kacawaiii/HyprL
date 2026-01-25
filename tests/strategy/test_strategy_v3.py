"""Tests for Strategy V3 components."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from hyprl.strategy.sizing_v2 import (
    SizingConfig,
    compute_dynamic_size,
    compute_conviction_multiplier,
)
from hyprl.strategy.trailing_v2 import (
    TrailingConfig,
    create_initial_state,
    update_trailing_stop,
)
from hyprl.strategy.exits_v2 import (
    ExitConfig,
    create_position_state,
    evaluate_exit,
    compute_r_multiple,
)
from hyprl.strategy.signal_quality import (
    QualityConfig,
    evaluate_signal_quality,
    check_volume,
    check_time_of_day,
)
from hyprl.strategy.strategy_v3 import (
    StrategyV3,
    StrategyV3Config,
)


class TestConvictionSizing:
    """Tests for conviction-based position sizing."""

    def test_high_conviction_increases_size(self):
        config = SizingConfig()
        mult = compute_conviction_multiplier(0.80, 0.60, config)
        assert mult == config.max_conviction_mult

    def test_low_conviction_decreases_size(self):
        config = SizingConfig()
        mult = compute_conviction_multiplier(0.52, 0.60, config)
        assert mult == config.min_conviction_mult

    def test_dynamic_size_respects_risk(self):
        result = compute_dynamic_size(
            equity=100_000,
            entry_price=100.0,
            stop_price=98.0,  # $2 risk per share
            probability=0.65,
            threshold=0.60,
            current_atr_pct=0.02,
            regime_mult=1.0,
        )
        assert result.shares > 0
        assert result.risk_amount <= 100_000 * 0.02  # Max 2% risk


class TestTrailingStop:
    """Tests for ATR-based trailing stop."""

    def test_trailing_stop_moves_up_for_long(self):
        state = create_initial_state(
            entry_price=100.0,
            stop_price=98.0,
            is_long=True,
        )

        # Price moves up
        state.highest_price = 105.0
        update = update_trailing_stop(
            state=state,
            current_price=105.0,
            current_atr=1.5,
        )

        assert update.new_stop >= state.current_stop
        assert not update.should_close

    def test_trailing_stop_triggers_close(self):
        state = create_initial_state(
            entry_price=100.0,
            stop_price=98.0,
            is_long=True,
        )

        # Price drops to stop
        update = update_trailing_stop(
            state=state,
            current_price=97.5,
            current_atr=1.5,
        )

        assert update.should_close


class TestExitManagement:
    """Tests for exit rules."""

    def test_r_multiple_calculation(self):
        state = create_position_state(
            symbol="TEST",
            entry_price=100.0,
            stop_price=98.0,  # $2 risk
            qty=10,
            is_long=True,
        )

        # At 1R profit (entry + risk)
        r = compute_r_multiple(state, 102.0)
        assert abs(r - 1.0) < 0.01

    def test_partial_profit_triggers(self):
        state = create_position_state(
            symbol="TEST",
            entry_price=100.0,
            stop_price=98.0,
            qty=10,
            is_long=True,
        )
        config = ExitConfig(partial_r_levels=(1.5,))

        # At 1.5R profit
        signal = evaluate_exit(state, 103.0, config=config)
        assert signal.action == "close_partial"

    def test_stagnant_trade_exit(self):
        entry_time = datetime.now(timezone.utc) - timedelta(hours=8)
        state = create_position_state(
            symbol="TEST",
            entry_price=100.0,
            stop_price=98.0,
            qty=10,
            is_long=True,
            entry_time=entry_time,
        )
        config = ExitConfig(stagnant_exit_hours=6, stagnant_threshold_r=0.3)

        # No progress after 8 hours
        signal = evaluate_exit(state, 100.0, config=config)
        assert signal.action == "close_all"
        assert signal.reason == "stagnant_trade"


class TestSignalQuality:
    """Tests for signal quality filter."""

    def test_low_volume_rejected(self):
        config = QualityConfig(volume_min_ratio=0.8)
        check = check_volume(80, 200, config)  # 0.4x average
        assert not check.passed

    def test_high_volume_approved(self):
        config = QualityConfig(volume_min_ratio=0.8)
        check = check_volume(300, 200, config)  # 1.5x average
        assert check.passed

    def test_overall_quality_evaluation(self):
        result = evaluate_signal_quality(
            signal_direction="long",
            current_price=100.0,
            current_volume=150,
            avg_volume=100,
            atr_pct=0.02,
        )
        assert result.score > 0
        assert result.recommendation in ("trade", "reduce_size", "skip")


class TestStrategyV3:
    """Tests for unified Strategy V3."""

    def test_signal_evaluation(self):
        strategy = StrategyV3()

        result = strategy.evaluate_signal(
            symbol="NVDA",
            probability=0.72,
            threshold_long=0.60,
            threshold_short=0.40,
            entry_price=150.0,
            stop_price=147.0,
            take_profit_price=156.0,
            equity=100_000,
            current_volume=1_000_000,
            avg_volume=800_000,
            atr=3.0,
        )

        assert result.direction == "long"
        assert result.quality_score > 0

    def test_position_tracking(self):
        strategy = StrategyV3()

        strategy.register_position(
            symbol="NVDA",
            entry_price=150.0,
            stop_price=147.0,
            qty=10,
            is_long=True,
        )

        assert "NVDA" in strategy.get_all_positions()

        update = strategy.update_position(
            symbol="NVDA",
            current_price=155.0,  # In profit
            current_atr=3.0,
        )

        assert update.action in ("hold", "update_stop", "close_partial")

    def test_cooldown_enforced(self):
        strategy = StrategyV3()
        strategy._last_trade_time["NVDA"] = datetime.now(timezone.utc)

        result = strategy.evaluate_signal(
            symbol="NVDA",
            probability=0.75,
            threshold_long=0.60,
            threshold_short=0.40,
            entry_price=150.0,
            stop_price=147.0,
            take_profit_price=156.0,
            equity=100_000,
            current_volume=1_000_000,
            avg_volume=800_000,
            atr=3.0,
        )

        assert not result.should_trade
        assert any("cooldown" in r for r in result.reasons)
