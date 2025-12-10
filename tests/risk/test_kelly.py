from hyprl.risk.kelly import KellyParams, KellySizer, compute_kelly_risk_pct


def test_kelly_sizer_warmup_returns_base() -> None:
    params = KellyParams(lookback_trades=50, base_risk_pct=0.01, min_trades=10, max_multiplier=2.0, min_multiplier=0.25)
    ks = KellySizer(params)
    assert ks.get_risk_pct() == 0.01


def test_compute_kelly_from_trades_scales_between_bounds() -> None:
    pnls = [1.0, 1.2, -0.5, 0.8, 1.1, -0.4, 0.9, 1.3, -0.6, 0.7]
    params = KellyParams(lookback_trades=10, base_risk_pct=0.01, min_trades=5, max_multiplier=2.0, min_multiplier=0.25)
    risk_pct = compute_kelly_risk_pct(pnls, params)
    assert 0.0025 <= risk_pct <= 0.02


def test_kelly_respects_min_trades_warmup() -> None:
    params = KellyParams(lookback_trades=5, base_risk_pct=0.02, min_trades=5, max_multiplier=2.0, min_multiplier=0.25)
    ks = KellySizer(params)
    ks.update_from_pnls([0.5, -0.2])  # below min_trades
    assert ks.get_risk_pct() == params.base_risk_pct
