import pandas as pd

from hyprl.backtest.runner import BacktestConfig, RiskConfig, SupercalcDataset, SupercalcRow
from hyprl.supercalc import _build_signal_series


def test_signal_series_emits_short_below_threshold():
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.2, 101.2, 102.2],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        },
        index=index,
    )
    row = SupercalcRow(
        price_index=1,
        timestamp=index[1],
        probability_up=0.3,
        rolling_return=-0.001,
        atr_value=1.0,
        sentiment_score=0.0,
        extreme_fear_flag=0,
        extreme_greed_flag=0,
    )
    dataset = SupercalcDataset(
        rows=[row],
        prices=prices,
        benchmark_return_pct=0.0,
        initial_balance=10_000.0,
    )
    cfg = BacktestConfig(
        ticker="TEST",
        period="1mo",
        interval="1h",
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=10_000.0, risk_pct=0.01, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=1),
        enable_trend_filter=False,
        sentiment_regime="off",
    )

    signal, fear, greed = _build_signal_series(dataset, cfg)

    assert signal[1] == -1.0
    assert fear == 0 and greed == 0
