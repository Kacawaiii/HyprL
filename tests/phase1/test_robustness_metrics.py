"""
Tests for Phase 1 robustness analysis.
"""
from __future__ import annotations

import math
import pytest

from hyprl.analysis.phase1 import compute_phase1_comparison


def test_robustness_metrics_perfect_match():
    """Test robustness metrics when live matches backtest exactly."""
    backtest = {
        'pf': 2.0,
        'sharpe': 1.5,
        'maxdd': 20.0,  # Percentage
        'expectancy': 50.0,
        'trades': 100,
        'win_rate': 0.60,
        'equity_vol': 0.15,
    }
    
    live = backtest.copy()  # Perfect match
    
    result = compute_phase1_comparison(backtest, live)
    
    # All ratios should be 1.0 (perfect robustness)
    assert abs(result['pf_ratio'] - 1.0) < 0.001, f"pf_ratio={result['pf_ratio']}"
    assert abs(result['sharpe_ratio'] - 1.0) < 0.001, f"sharpe_ratio={result['sharpe_ratio']}"
    assert abs(result['dd_ratio'] - 1.0) < 0.001, f"dd_ratio={result['dd_ratio']}"
    assert abs(result['expectancy_ratio'] - 1.0) < 0.001, f"expectancy_ratio={result['expectancy_ratio']}"
    assert abs(result['equity_vol_ratio'] - 1.0) < 0.001, f"equity_vol_ratio={result['equity_vol_ratio']}"
    assert abs(result['winrate_delta']) < 0.001, f"winrate_delta={result['winrate_delta']}"
    
    # robustness_score should be positive (exact value depends on formula)
    assert result['robustness_score'] > 0.5, (
        f"robustness_score={result['robustness_score']} should be high for perfect match"
    )


def test_robustness_metrics_degradation():
    """Test robustness metrics when live degrades vs backtest."""
    backtest = {
        'pf': 2.0,
        'sharpe': 1.5,
        'maxdd': 20.0,
        'expectancy': 50.0,
        'trades': 100,
        'win_rate': 0.60,
        'equity_vol': 0.15,
    }
    
    # Live degrades: PF and Sharpe drop, DD increases
    live = {
        'pf': 1.6,        # 80% of backtest
        'sharpe': 1.2,    # 80% of backtest
        'maxdd': 25.0,    # 125% of backtest (worse)
        'expectancy': 40.0,  # 80% of backtest
        'trades': 95,
        'win_rate': 0.55,  # -5% delta
        'equity_vol': 0.18,  # 120% of backtest (higher vol)
    }
    
    result = compute_phase1_comparison(backtest, live)
    
    # Check ratios
    assert abs(result['pf_ratio'] - 0.80) < 0.01, f"pf_ratio={result['pf_ratio']}"
    assert abs(result['sharpe_ratio'] - 0.80) < 0.01, f"sharpe_ratio={result['sharpe_ratio']}"
    assert abs(result['dd_ratio'] - 1.25) < 0.01, f"dd_ratio={result['dd_ratio']}"
    assert abs(result['expectancy_ratio'] - 0.80) < 0.01, f"expectancy_ratio={result['expectancy_ratio']}"
    assert abs(result['equity_vol_ratio'] - 1.20) < 0.01, f"equity_vol_ratio={result['equity_vol_ratio']}"
    assert abs(result['winrate_delta'] - (-0.05)) < 0.01, f"winrate_delta={result['winrate_delta']}"
    
    # robustness_score should be lower than perfect match
    assert result['robustness_score'] < 0.7, (
        f"robustness_score={result['robustness_score']} should indicate degradation"
    )


def test_robustness_metrics_improvement():
    """Test robustness metrics when live IMPROVES vs backtest (rare but possible)."""
    backtest = {
        'pf': 1.5,
        'sharpe': 1.0,
        'maxdd': 25.0,
        'expectancy': 30.0,
        'trades': 80,
        'win_rate': 0.55,
        'equity_vol': 0.18,
    }
    
    # Live improves (e.g., adaptive system kicked in, or backtest was pessimistic)
    live = {
        'pf': 1.8,        # 120% of backtest
        'sharpe': 1.2,    # 120% of backtest
        'maxdd': 20.0,    # 80% of backtest (better)
        'expectancy': 40.0,  # 133% of backtest
        'trades': 85,
        'win_rate': 0.60,  # +5% delta
        'equity_vol': 0.15,  # 83% of backtest (lower vol is good)
    }
    
    result = compute_phase1_comparison(backtest, live)
    
    # Check ratios
    assert result['pf_ratio'] > 1.0, f"pf_ratio={result['pf_ratio']} should be > 1.0"
    assert result['sharpe_ratio'] > 1.0, f"sharpe_ratio={result['sharpe_ratio']} should be > 1.0"
    assert result['dd_ratio'] < 1.0, f"dd_ratio={result['dd_ratio']} should be < 1.0 (improved)"
    assert result['expectancy_ratio'] > 1.0, f"expectancy_ratio={result['expectancy_ratio']} should be > 1.0"
    assert result['winrate_delta'] > 0, f"winrate_delta={result['winrate_delta']} should be positive"
    
    # robustness_score should be high (strategy exceeded expectations)
    assert result['robustness_score'] > 0.7, (
        f"robustness_score={result['robustness_score']} should be high for improvement"
    )


def test_robustness_metrics_zero_backtest():
    """Test graceful handling of zero/invalid backtest metrics."""
    backtest = {
        'pf': 0.0,  # Invalid
        'sharpe': 0.0,
        'maxdd': 0.0,
        'expectancy': 0.0,
        'trades': 0,
        'win_rate': 0.0,
        'equity_vol': 0.0,
    }
    
    live = {
        'pf': 1.5,
        'sharpe': 1.0,
        'maxdd': 10.0,
        'expectancy': 20.0,
        'trades': 50,
        'win_rate': 0.55,
        'equity_vol': 0.12,
    }
    
    result = compute_phase1_comparison(backtest, live)
    
    # Ratios should be NaN (division by zero)
    assert math.isnan(result['pf_ratio']), "pf_ratio should be NaN for zero backtest"
    assert math.isnan(result['sharpe_ratio']), "sharpe_ratio should be NaN for zero backtest"
    # dd_ratio might be NaN or inf depending on implementation
    
    # robustness_score should handle NaN gracefully (clamp to [0, 1])
    assert 0.0 <= result['robustness_score'] <= 1.0 or math.isnan(result['robustness_score']), (
        f"robustness_score={result['robustness_score']} should be in [0, 1] or NaN"
    )


def test_robustness_metrics_columns_present():
    """Test that all required columns are present in output."""
    backtest = {
        'pf': 2.0,
        'sharpe': 1.5,
        'maxdd': 20.0,
        'expectancy': 50.0,
        'trades': 100,
        'win_rate': 0.60,
        'equity_vol': 0.15,
    }
    
    live = backtest.copy()
    result = compute_phase1_comparison(backtest, live)
    
    # Check all required columns exist
    required_cols = [
        'pf_ratio',
        'sharpe_ratio',
        'dd_ratio',
        'expectancy_ratio',
        'equity_vol_ratio',
        'winrate_delta',
        'robustness_score',
    ]
    
    for col in required_cols:
        assert col in result, f"Missing required column: {col}"


def test_robustness_metrics_dd_interpretation():
    """Test that DD ratio is correctly interpreted (lower is better)."""
    # Scenario 1: DD improved live (good)
    backtest_good = {'pf': 2.0, 'sharpe': 1.5, 'maxdd': 30.0, 'expectancy': 50.0, 
                     'trades': 100, 'win_rate': 0.6, 'equity_vol': 0.15}
    live_good = backtest_good.copy()
    live_good['maxdd'] = 20.0  # Improved (lower DD)
    
    result_good = compute_phase1_comparison(backtest_good, live_good)
    assert result_good['dd_ratio'] < 1.0, "DD ratio should be < 1.0 when DD improves live"
    
    # Scenario 2: DD worsened live (bad)
    backtest_bad = backtest_good.copy()
    live_bad = backtest_good.copy()
    live_bad['maxdd'] = 40.0  # Worsened (higher DD)
    
    result_bad = compute_phase1_comparison(backtest_bad, live_bad)
    assert result_bad['dd_ratio'] > 1.0, "DD ratio should be > 1.0 when DD worsens live"


def test_robustness_score_formula():
    """Test that robustness_score formula is reasonable."""
    # Perfect match case
    backtest = {
        'pf': 2.0,
        'sharpe': 1.5,
        'maxdd': 20.0,
        'expectancy': 50.0,
        'trades': 100,
        'win_rate': 0.60,
        'equity_vol': 0.15,
    }
    live = backtest.copy()
    
    result = compute_phase1_comparison(backtest, live)
    
    # robustness_score should be in [0, 1]
    assert 0.0 <= result['robustness_score'] <= 1.0, (
        f"robustness_score={result['robustness_score']} should be in [0, 1]"
    )
    
    # For perfect match, should be relatively high (>0.5)
    assert result['robustness_score'] > 0.5, (
        f"robustness_score={result['robustness_score']} should be > 0.5 for perfect match"
    )
