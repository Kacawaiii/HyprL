import pandas as pd

from hyprl.backtest.runner import BacktestConfig, RiskConfig, SupercalcRow, _row_sentiment_permits, _row_trend_permits


def _row(prob_up: float = 0.3, rr: float = -0.001, sentiment: float = 0.0) -> SupercalcRow:
    return SupercalcRow(
        price_index=0,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        probability_up=prob_up,
        rolling_return=rr,
        atr_value=1.0,
        sentiment_score=sentiment,
        extreme_fear_flag=0,
        extreme_greed_flag=0,
    )


def _cfg(enable_trend: bool = False, trend_short_min: float = 0.0, sentiment_regime: str = "off") -> BacktestConfig:
    return BacktestConfig(
        ticker="TEST",
        period="1mo",
        interval="1h",
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=10_000.0, risk_pct=0.01, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=1),
        enable_trend_filter=enable_trend,
        trend_short_min=trend_short_min,
        trend_long_min=0.0,
        sentiment_regime=sentiment_regime,
    )


def test_trend_filter_pass_through_when_disabled():
    row = _row()
    cfg = _cfg(enable_trend=False)
    assert _row_trend_permits(row, "short", cfg) is True
    assert _row_trend_permits(row, "long", cfg) is True


def test_trend_filter_blocks_short_when_enabled_and_rr_positive():
    row = _row(rr=0.002)
    cfg = _cfg(enable_trend=True, trend_short_min=0.0)
    assert _row_trend_permits(row, "short", cfg) is False


def test_sentiment_filter_pass_through_when_disabled():
    row = _row(sentiment=0.9)
    cfg = _cfg(sentiment_regime="off")
    assert _row_sentiment_permits(row, cfg) is True
