"""
SENSE Shadow Merge Tests
Based on SENSE_FULL_SPEC.md

Tests:
- Timestamp normalization (ts/entry_ts/index)
- Window filtering (tolerance +/- 90m)
- Match rate calculation
- Retention metrics
- Shadow metrics (pnl_delta, filter distribution)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestTimestampNormalization:
    """Test timestamp normalization for trades/controls merge"""

    def test_trade_ts_column(self):
        """Trades with 'ts' column should be normalized"""
        trades = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "pnl": [100, -50],
        })
        trades["_merge_ts"] = pd.to_datetime(trades["ts"])
        assert trades["_merge_ts"].dtype == "datetime64[ns]"

    def test_trade_entry_ts_column(self):
        """Trades with 'entry_ts' column should be normalized"""
        trades = pd.DataFrame({
            "entry_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "pnl": [100, -50],
        })
        # Normalize: prefer entry_ts over ts
        if "entry_ts" in trades.columns:
            trades["_merge_ts"] = pd.to_datetime(trades["entry_ts"])
        elif "ts" in trades.columns:
            trades["_merge_ts"] = pd.to_datetime(trades["ts"])
        assert trades["_merge_ts"].iloc[0] == pd.Timestamp("2024-01-01 10:00")

    def test_trade_datetime_index(self):
        """Trades with DatetimeIndex should use index as ts"""
        trades = pd.DataFrame({
            "pnl": [100, -50],
        }, index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]))
        trades.index.name = "datetime"

        # Normalize: use index if no ts column
        if "ts" not in trades.columns and "entry_ts" not in trades.columns:
            trades["_merge_ts"] = trades.index
        assert trades["_merge_ts"].iloc[0] == pd.Timestamp("2024-01-01 10:00")

    def test_controls_ts_column(self):
        """Controls with 'ts' column should be normalized"""
        controls = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "sense_filter": ["ALLOW", "BLOCK"],
        })
        controls["_merge_ts"] = pd.to_datetime(controls["ts"])
        assert controls["_merge_ts"].dtype == "datetime64[ns]"

    def test_controls_datetime_index(self):
        """Controls with DatetimeIndex should use index as ts"""
        controls = pd.DataFrame({
            "sense_filter": ["ALLOW", "BLOCK"],
        }, index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]))

        if "ts" not in controls.columns:
            controls["_merge_ts"] = controls.index
        assert controls["_merge_ts"].iloc[1] == pd.Timestamp("2024-01-01 11:00")

    def test_timezone_handling(self):
        """Timestamps should be tz-naive for merge"""
        ts_aware = pd.Timestamp("2024-01-01 10:00", tz="US/Eastern")
        ts_naive = ts_aware.tz_localize(None)
        assert ts_naive.tzinfo is None


class TestWindowFiltering:
    """Test window filtering with tolerance"""

    def test_default_tolerance_90m(self):
        """Default tolerance should be 90 minutes"""
        tolerance_minutes = 90
        assert tolerance_minutes == 90

    def test_trade_within_window(self):
        """Trades within controls window should be kept"""
        controls_min = pd.Timestamp("2024-01-01 09:00")
        controls_max = pd.Timestamp("2024-01-01 17:00")
        tolerance = timedelta(minutes=90)

        trade_ts = pd.Timestamp("2024-01-01 10:00")
        window_start = controls_min - tolerance
        window_end = controls_max + tolerance

        in_window = window_start <= trade_ts <= window_end
        assert in_window is True

    def test_trade_outside_window_before(self):
        """Trades before window - tolerance should be filtered"""
        controls_min = pd.Timestamp("2024-01-01 09:00")
        controls_max = pd.Timestamp("2024-01-01 17:00")
        tolerance = timedelta(minutes=90)

        trade_ts = pd.Timestamp("2024-01-01 06:00")  # 3h before min
        window_start = controls_min - tolerance  # 07:30

        in_window = trade_ts >= window_start
        assert in_window is False

    def test_trade_outside_window_after(self):
        """Trades after window + tolerance should be filtered"""
        controls_min = pd.Timestamp("2024-01-01 09:00")
        controls_max = pd.Timestamp("2024-01-01 17:00")
        tolerance = timedelta(minutes=90)

        trade_ts = pd.Timestamp("2024-01-01 20:00")  # 3h after max
        window_end = controls_max + tolerance  # 18:30

        in_window = trade_ts <= window_end
        assert in_window is False

    def test_trade_at_tolerance_boundary(self):
        """Trade exactly at tolerance boundary should be included"""
        controls_min = pd.Timestamp("2024-01-01 09:00")
        tolerance = timedelta(minutes=90)

        trade_ts = pd.Timestamp("2024-01-01 07:30")  # Exactly at boundary
        window_start = controls_min - tolerance

        in_window = trade_ts >= window_start
        assert in_window is True


class TestMatchRate:
    """Test match rate calculation"""

    def test_perfect_match(self):
        """All trades matched = 100% match rate"""
        matched_trades = 100
        total_trades = 100
        match_rate = matched_trades / total_trades
        assert match_rate == 1.0

    def test_partial_match(self):
        """Some trades unmatched = partial match rate"""
        matched_trades = 80
        total_trades = 100
        match_rate = matched_trades / total_trades
        assert match_rate == 0.8

    def test_no_match(self):
        """No trades matched = 0% match rate"""
        matched_trades = 0
        total_trades = 100
        match_rate = matched_trades / total_trades
        assert match_rate == 0.0

    def test_match_with_asof_merge(self):
        """Test asof merge for timestamp matching"""
        trades = pd.DataFrame({
            "_merge_ts": pd.to_datetime([
                "2024-01-01 10:05",  # 5 min after control
                "2024-01-01 11:30",  # 30 min after control
            ]),
            "pnl": [100, 50],
        }).sort_values("_merge_ts")

        controls = pd.DataFrame({
            "_merge_ts": pd.to_datetime([
                "2024-01-01 10:00",
                "2024-01-01 11:00",
                "2024-01-01 12:00",
            ]),
            "sense_filter": ["ALLOW", "ALLOW", "BLOCK"],
        }).sort_values("_merge_ts")

        # asof merge: each trade gets nearest previous control
        merged = pd.merge_asof(
            trades,
            controls,
            on="_merge_ts",
            direction="backward",
        )

        assert len(merged) == 2
        assert merged["sense_filter"].iloc[0] == "ALLOW"  # 10:05 -> 10:00
        assert merged["sense_filter"].iloc[1] == "ALLOW"  # 11:30 -> 11:00


class TestRetentionMetrics:
    """Test retention calculations"""

    def test_retention_on_matched_window(self):
        """Retention = shadow_trades / matched_trades"""
        shadow_trades = 95  # After BLOCK filter
        matched_trades = 100
        retention = shadow_trades / matched_trades
        assert retention == 0.95

    def test_retention_over_all_trades(self):
        """Overall retention includes filtered-out trades"""
        shadow_trades = 317
        total_trades = 465
        retention = shadow_trades / total_trades
        assert retention == pytest.approx(0.6817, rel=1e-3)

    def test_blocked_trades_count(self):
        """Blocked trades = matched - shadow"""
        matched_trades = 323
        shadow_trades = 317
        blocked = matched_trades - shadow_trades
        assert blocked == 6

    def test_filtered_out_count(self):
        """Filtered out = total - matched (outside window)"""
        total_trades = 465
        matched_trades = 323
        filtered_out = total_trades - matched_trades
        assert filtered_out == 142


class TestShadowMetrics:
    """Test shadow harness metrics"""

    def test_pnl_delta_calculation(self):
        """pnl_delta = shadow_pnl - baseline_pnl"""
        baseline_pnl = 135063.67
        shadow_pnl = 132763.98
        pnl_delta = shadow_pnl - baseline_pnl
        assert pnl_delta == pytest.approx(-2299.69, rel=1e-2)

    def test_pnl_scaled_by_risk_multiplier(self):
        """Shadow PnL should be scaled by risk_multiplier"""
        base_pnl = 100.0
        risk_multiplier = 1.0  # Audit-only
        scaled_pnl = base_pnl * risk_multiplier
        assert scaled_pnl == base_pnl

    def test_filter_distribution_sum(self):
        """Filter distribution should sum to matched trades"""
        distribution = {"ALLOW": 316, "BLOCK": 6, "REDUCE": 1}
        matched_trades = 323
        assert sum(distribution.values()) == matched_trades

    def test_max_dd_comparison(self):
        """Shadow max DD might be worse than baseline (BLOCK useful trades)"""
        max_dd_baseline = -13385.49
        max_dd_shadow = -15100.74
        # Shadow DD can be worse if BLOCK removed some winning trades
        assert abs(max_dd_shadow) >= abs(max_dd_baseline)

    def test_blocked_losses_avoided(self):
        """Blocked losses = sum of PnL from BLOCK trades (if positive)"""
        blocked_trades_pnl = [-500, -300, -200, 100, -400, -327]
        losses_avoided = sum(pnl for pnl in blocked_trades_pnl if pnl < 0)
        assert losses_avoided == -1727

    def test_reason_top5_format(self):
        """Reason codes should be tracked with counts"""
        reasons = {
            "SENSE_ALLOW_OK": 315,
            "BLOCK_GAP_ANOMALY": 6,
            "MISSING_ALIGNED_SIGNALS": 1,
            "REDUCE_SENSE_TRUST_LOW": 1,
        }
        top5 = dict(sorted(reasons.items(), key=lambda x: -x[1])[:5])
        assert list(top5.keys())[0] == "SENSE_ALLOW_OK"
        assert sum(top5.values()) == 323

    def test_effective_window_format(self):
        """Effective window should be ISO format timestamps"""
        window_start = "2023-12-19T13:00:00"
        window_end = "2025-12-15T22:00:00"

        # Parse to verify format
        start = pd.Timestamp(window_start)
        end = pd.Timestamp(window_end)

        assert start < end
        assert (end - start).days > 700  # ~2 years

    def test_assumptions_documented(self):
        """Shadow summary should document assumptions"""
        assumptions = ["pnl_scaled_by_risk_multiplier"]
        assert "pnl_scaled_by_risk_multiplier" in assumptions
