from hyprl.risk.guards import RiskGuardConfig, RiskGuardMonitor


def test_guard_triggers_on_drawdown() -> None:
    guard = RiskGuardMonitor(RiskGuardConfig(max_drawdown_pct=0.2, min_pf=0.5, lookback_trades=10, max_consec_losses=None), equity_start=100.0)
    # Apply losses to reach >20% DD
    guard.on_new_trade(-10.0)
    guard.on_new_trade(-15.0)  # peak=100, equity=75 => 25% DD
    assert guard.should_stop_trading() is True
    assert guard.state.reason == "max_drawdown_exceeded"


def test_guard_triggers_on_min_pf() -> None:
    guard = RiskGuardMonitor(RiskGuardConfig(max_drawdown_pct=1.0, min_pf=1.0, lookback_trades=5, max_consec_losses=None), equity_start=100.0)
    guard.on_new_trade(10.0)
    guard.on_new_trade(-30.0)  # PF = 10/30 < 1.0
    assert guard.should_stop_trading() is True
    assert guard.state.reason == "min_pf_breached"


def test_guard_triggers_on_consec_losses() -> None:
    guard = RiskGuardMonitor(RiskGuardConfig(max_drawdown_pct=1.0, min_pf=0.0, lookback_trades=10, max_consec_losses=3), equity_start=100.0)
    guard.on_new_trade(-1.0)
    guard.on_new_trade(-2.0)
    guard.on_new_trade(-3.0)
    assert guard.should_stop_trading() is True
    assert guard.state.reason == "max_consec_losses"
