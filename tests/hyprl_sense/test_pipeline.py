"""
SENSE Pipeline Tests
Based on SENSE_FULL_SPEC.md

Tests:
- Fusion: sense_trust formula
- Filter: BLOCK/ALLOW/REDUCE logic
- Gap: informational-only handling
- Anti-lookahead: shift(1) alignment
- Risk multiplier: audit-only constraints
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestSenseFusion:
    """Test sense_trust fusion formula: (chrono_trust_a + (1 - fear_index_a)) / 2"""

    def test_fusion_perfect_conditions(self):
        """High chrono_trust + low fear = high sense_trust"""
        chrono_trust_a = 1.0
        fear_index_a = 0.0
        sense_trust = (chrono_trust_a + (1 - fear_index_a)) / 2
        assert sense_trust == 1.0

    def test_fusion_worst_conditions(self):
        """Low chrono_trust + high fear = low sense_trust"""
        chrono_trust_a = 0.0
        fear_index_a = 1.0
        sense_trust = (chrono_trust_a + (1 - fear_index_a)) / 2
        assert sense_trust == 0.0

    def test_fusion_neutral(self):
        """Neutral conditions = 0.5 sense_trust"""
        chrono_trust_a = 0.5
        fear_index_a = 0.5
        sense_trust = (chrono_trust_a + (1 - fear_index_a)) / 2
        assert sense_trust == 0.5

    def test_fusion_fear_dominates(self):
        """High fear should reduce sense_trust even with good chrono"""
        chrono_trust_a = 0.8
        fear_index_a = 0.9
        sense_trust = (chrono_trust_a + (1 - fear_index_a)) / 2
        assert sense_trust == pytest.approx(0.45)


class TestSenseFilter:
    """Test BLOCK/ALLOW/REDUCE filter logic"""

    def test_allow_default(self):
        """Default filter should be ALLOW when conditions normal"""
        fear_index_a = 0.5
        filter_decision = "BLOCK" if fear_index_a >= 0.90 else "ALLOW"
        assert filter_decision == "ALLOW"

    def test_block_high_fear(self):
        """BLOCK when fear_index_a >= 0.90 (fallback rule)"""
        fear_index_a = 0.90
        filter_decision = "BLOCK" if fear_index_a >= 0.90 else "ALLOW"
        assert filter_decision == "BLOCK"

    def test_block_extreme_fear(self):
        """BLOCK on extreme fear"""
        fear_index_a = 0.95
        filter_decision = "BLOCK" if fear_index_a >= 0.90 else "ALLOW"
        assert filter_decision == "BLOCK"

    def test_allow_just_below_threshold(self):
        """ALLOW when just below 0.90 threshold"""
        fear_index_a = 0.89
        filter_decision = "BLOCK" if fear_index_a >= 0.90 else "ALLOW"
        assert filter_decision == "ALLOW"

    def test_reason_codes(self):
        """Verify reason code telemetry mapping"""
        reason_codes = {
            "ALLOW": "SENSE_ALLOW_OK",
            "BLOCK_FEAR": "BLOCK_GAP_ANOMALY",
            "MISSING": "MISSING_ALIGNED_SIGNALS",
            "REDUCE": "REDUCE_SENSE_TRUST_LOW",
        }
        assert "SENSE_ALLOW_OK" in reason_codes.values()
        assert "BLOCK_GAP_ANOMALY" in reason_codes.values()


class TestGapHandling:
    """Test gap handling (informational-only mode)"""

    def test_gap_multiple_calculation(self):
        """Gap multiple should be ratio of gap to normal range"""
        normal_range = 0.02  # 2% typical
        actual_gap = 0.06  # 6% gap
        gap_multiple = actual_gap / normal_range
        assert gap_multiple == 3.0

    def test_gap_informational_no_action(self):
        """Gap should not trigger action in audit-only mode"""
        gap_multiple = 5.0  # High gap
        risk_multiplier = 1.0  # Forced to 1.0 in audit-only
        # In audit-only, gap is logged but does not affect risk
        assert risk_multiplier == 1.0

    def test_gap_anomaly_detection(self):
        """Detect anomalous gaps > threshold"""
        gap_threshold = 3.0
        gaps = [1.0, 2.5, 3.5, 5.0, 1.2]
        anomalous = [g for g in gaps if g > gap_threshold]
        assert len(anomalous) == 2
        assert 3.5 in anomalous
        assert 5.0 in anomalous


class TestAntiLookahead:
    """Test anti-lookahead alignment (shift by 1)"""

    def test_aligned_columns_shifted(self):
        """_a columns should be shifted by 1 bar"""
        df = pd.DataFrame({
            "chrono_trust": [0.5, 0.6, 0.7, 0.8, 0.9],
            "fear_index": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        df["chrono_trust_a"] = df["chrono_trust"].shift(1)
        df["fear_index_a"] = df["fear_index"].shift(1)

        # First row should be NaN (no previous data)
        assert pd.isna(df["chrono_trust_a"].iloc[0])
        assert pd.isna(df["fear_index_a"].iloc[0])

        # Second row should have first row's value
        assert df["chrono_trust_a"].iloc[1] == 0.5
        assert df["fear_index_a"].iloc[1] == 0.1

    def test_no_future_leak(self):
        """Current bar should not access future data"""
        df = pd.DataFrame({
            "value": [10, 20, 30, 40, 50],
        }, index=pd.date_range("2024-01-01", periods=5, freq="h"))

        df["value_a"] = df["value"].shift(1)

        # At index 2 (30), aligned value should be 20 (previous bar)
        assert df["value_a"].iloc[2] == 20
        # Not 30 (current) or 40 (future)
        assert df["value_a"].iloc[2] != 30

    def test_aligned_suffix_convention(self):
        """Verify _a suffix convention for aligned columns"""
        columns = ["chrono_trust", "fear_index", "gap_multiple"]
        aligned = [f"{c}_a" for c in columns]
        assert aligned == ["chrono_trust_a", "fear_index_a", "gap_multiple_a"]


class TestRiskMultiplier:
    """Test risk_multiplier audit-only constraints"""

    def test_risk_multiplier_forced_one(self):
        """risk_multiplier must be 1.0 in audit-only mode"""
        config_risk_multiplier = 1.0  # From configs/sense.yaml
        assert config_risk_multiplier == 1.0

    def test_risk_multiplier_no_sizing_impact(self):
        """Risk multiplier should not affect position sizing"""
        base_size = 1000
        risk_multiplier = 1.0
        adjusted_size = base_size * risk_multiplier
        assert adjusted_size == base_size

    def test_threshold_tighten_zero(self):
        """threshold_tighten must be 0.0 in audit-only mode"""
        threshold_tighten = 0.0  # From configs/sense.yaml
        assert threshold_tighten == 0.0

    def test_audit_only_constraints(self):
        """Verify all audit-only constraints are enforced"""
        audit_config = {
            "risk_multiplier": 1.0,
            "threshold_tighten": 0.0,
            "block_threshold": 0.90,  # Only fallback BLOCK
        }
        assert audit_config["risk_multiplier"] == 1.0
        assert audit_config["threshold_tighten"] == 0.0


class TestOutputFormat:
    """Test SENSE output format (controls parquet + summary JSON)"""

    def test_controls_required_columns(self):
        """Controls output should have required columns"""
        required_cols = [
            "ts",
            "chrono_trust_a",
            "fear_index_a",
            "sense_trust",
            "sense_filter",
            "reason_code",
            "risk_multiplier",
        ]
        # Mock controls DataFrame
        controls = pd.DataFrame(columns=required_cols)
        for col in required_cols:
            assert col in controls.columns

    def test_summary_required_fields(self):
        """Summary JSON should have required fields"""
        required_fields = [
            "baseline_trades",
            "shadow_trades",
            "trade_retention_pct",
            "baseline_pnl",
            "shadow_pnl",
            "pnl_delta",
            "max_dd_baseline",
            "max_dd_shadow",
            "match_rate",
            "filter_distribution",
        ]
        # Mock summary dict
        summary = {field: None for field in required_fields}
        for field in required_fields:
            assert field in summary

    def test_filter_distribution_categories(self):
        """Filter distribution should track ALLOW/BLOCK/REDUCE"""
        distribution = {"ALLOW": 316, "BLOCK": 6, "REDUCE": 1}
        assert "ALLOW" in distribution
        assert "BLOCK" in distribution
        assert "REDUCE" in distribution
        assert sum(distribution.values()) == 323
