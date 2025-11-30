from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from hyprl.backtest.runner import TradeRecord
from hyprl.model.probability import ProbabilityModel
import hyprl.model.probability as probability_module
from hyprl.metrics.calibration import brier_score, log_loss_score, trade_calibration_metrics


def _toy_dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    feature = rng.normal(size=(n, 2))
    target = (feature[:, 0] + feature[:, 1] + rng.normal(scale=0.2, size=n) > 0).astype(int)
    df = pd.DataFrame(feature, columns=["f1", "f2"])
    series = pd.Series(target)
    return df, series


def test_probability_model_supports_platt_calibration():
    features, target = _toy_dataset()
    model = ProbabilityModel.create(model_type="logistic", calibration="platt", random_state=7)
    model.fit(features, target)
    probs = model.predict_proba(features)
    assert model.calibrator.method == "platt"
    assert model.calibrator.is_fitted
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_probability_model_random_forest_backend():
    features, target = _toy_dataset()
    model = ProbabilityModel.create(model_type="random_forest", calibration="none", random_state=11)
    model.fit(features, target)
    probs = model.predict_proba(features.head(5))
    assert probs.shape[0] == 5
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_probability_model_xgboost_backend(monkeypatch: pytest.MonkeyPatch):
    features, target = _toy_dataset()

    class _FakeXGB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, *_args, **_kwargs):
            return self

        def predict_proba(self, feature_matrix):
            n = len(feature_matrix)
            up = np.full(n, 0.7)
            down = 1.0 - up
            return np.column_stack([down, up])

    monkeypatch.setattr(probability_module, "XGBClassifier", _FakeXGB)
    model = probability_module.ProbabilityModel.create(model_type="xgboost", calibration="none", random_state=13)
    model.fit(features, target)
    probs = model.predict_proba(features.head(3))
    assert np.allclose(probs, 0.7)


def test_probability_model_xgboost_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(probability_module, "XGBClassifier", None)
    with pytest.raises(ImportError):
        probability_module.ProbabilityModel.create(model_type="xgboost", calibration="none", random_state=0)


def test_calibration_metrics_basic():
    probs = np.array([0.8, 0.2, 0.6, 0.4])
    targets = np.array([1, 0, 1, 0])
    brier = brier_score(probs, targets)
    logloss = log_loss_score(probs, targets)
    assert brier <= 0.1
    assert logloss < 0.5


def test_trade_calibration_metrics_from_trades():
    trades = [
        TradeRecord(
            entry_timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            exit_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            direction="long",
            probability_up=0.7,
            threshold=0.55,
            entry_price=100,
            exit_price=102,
            position_size=1,
            pnl=2.0,
            return_pct=0.02,
            equity_after=1002.0,
            risk_amount=1.0,
            expected_pnl=1.0,
        ),
        TradeRecord(
            entry_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            exit_timestamp=pd.Timestamp("2024-01-03", tz="UTC"),
            direction="short",
            probability_up=0.3,
            threshold=0.45,
            entry_price=101,
            exit_price=100,
            position_size=1,
            pnl=1.0,
            return_pct=0.01,
            equity_after=1003.0,
            risk_amount=1.0,
            expected_pnl=0.5,
        ),
    ]
    metrics = trade_calibration_metrics(trades)
    assert metrics["brier"] is not None
    assert metrics["log_loss"] is not None
