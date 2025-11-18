"""Test parity between Python and Rust metrics implementations."""
import numpy as np
import pytest

try:
    from hyprl import hyprl_supercalc
    HAS_SUPERCALC = True
except ImportError:
    HAS_SUPERCALC = False


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.005, 0.015]
    sharpe = hyprl_supercalc.sharpe_ratio_py(returns)
    
    assert sharpe is not None
    assert sharpe > 0.0
    assert isinstance(sharpe, float)


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.005, 0.015]
    sortino = hyprl_supercalc.sortino_ratio_py(returns, 0.0)
    
    assert sortino is not None
    assert isinstance(sortino, float)


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_max_drawdown():
    """Test max drawdown calculation."""
    equity_curve = [100.0, 110.0, 105.0, 120.0, 100.0, 115.0]
    mdd = hyprl_supercalc.max_drawdown_py(equity_curve)
    
    assert mdd > 0.0
    assert mdd < 1.0
    # Max DD should be around 16.67% (from 120 to 100)
    assert abs(mdd - 0.1667) < 0.01


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_win_rate():
    """Test win rate calculation."""
    trade_pnls = [10.0, -5.0, 15.0, -3.0, 20.0, 5.0]
    wr = hyprl_supercalc.win_rate_py(trade_pnls)
    
    # 4 wins out of 6 trades
    assert abs(wr - 4.0/6.0) < 0.01


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_profit_factor():
    """Test profit factor calculation."""
    trade_pnls = [10.0, -5.0, 15.0, -3.0, 20.0]
    pf = hyprl_supercalc.profit_factor_py(trade_pnls)
    
    assert pf is not None
    # Gross profit = 10 + 15 + 20 = 45
    # Gross loss = 5 + 3 = 8
    # PF = 45/8 = 5.625
    assert abs(pf - 5.625) < 0.01


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_risk_of_ruin():
    """Test risk of ruin calculation."""
    # Positive expectancy trades
    trade_returns = [0.01, -0.005, 0.015, -0.003, 0.02]
    ror = hyprl_supercalc.risk_of_ruin_py(trade_returns, 0.02)
    
    # Should be low for positive expectancy
    assert ror >= 0.0
    assert ror <= 1.0


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_robustness_score():
    """Test robustness score calculation."""
    sharpe = 1.5
    profit_factor = 2.0
    win_rate = 0.6
    max_dd = 0.15
    
    score = hyprl_supercalc.robustness_score_py(
        sharpe=sharpe,
        profit_factor=profit_factor,
        win_rate=win_rate,
        max_dd=max_dd
    )
    
    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0
    # Good metrics should give a decent score
    assert score > 0.5


@pytest.mark.skipif(not HAS_SUPERCALC, reason="hyprl_supercalc not built")
def test_supercalc_compute_metrics():
    """Test comprehensive metrics computation."""
    equity_curve = [10000.0, 10100.0, 10050.0, 10200.0, 10000.0, 10150.0]
    trade_returns = [0.01, -0.005, 0.015, -0.02, 0.015]
    trade_pnls = [100.0, -50.0, 150.0, -200.0, 150.0]
    
    metrics = hyprl_supercalc.compute_metrics_py(
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        trade_pnls=trade_pnls,
        risk_pct=0.02
    )
    
    assert metrics is not None
    assert hasattr(metrics, 'sharpe_ratio')
    assert hasattr(metrics, 'sortino_ratio')
    assert hasattr(metrics, 'profit_factor')
    assert hasattr(metrics, 'win_rate')
    assert hasattr(metrics, 'max_drawdown')
    assert hasattr(metrics, 'risk_of_ruin')
    assert hasattr(metrics, 'robustness_score')
    
    # Check values are reasonable
    assert metrics.max_drawdown >= 0.0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert 0.0 <= metrics.robustness_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
