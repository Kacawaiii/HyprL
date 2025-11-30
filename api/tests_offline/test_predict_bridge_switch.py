import asyncio
import os

import pytest

from core_bridge import service


def test_predict_bridge_stub_default(monkeypatch):
    monkeypatch.delenv("HYPRL_PREDICT_IMPL", raising=False)
    results = service.predict_batch(
        symbols=["AAPL", "MSFT"],
        interval="1m",
        features=["sma"],
        threshold=0.6,
        risk_pct=0.1,
    )
    assert results == [
        {
            "symbol": "AAPL",
            "prob_up": 0.512,
            "direction": "UP",
            "threshold": 0.6,
            "tp": 123.45,
            "sl": 120.12,
        },
        {
            "symbol": "MSFT",
            "prob_up": 0.431,
            "direction": "DOWN",
            "threshold": 0.6,
            "tp": 234.56,
            "sl": 230.01,
        },
    ]


def test_predict_bridge_real_fallback(monkeypatch, caplog):
    monkeypatch.setenv("HYPRL_PREDICT_IMPL", "real")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_predict_real", boom)
    caplog.set_level("WARNING")

    results = service.predict_batch(
        symbols=["AAPL"],
        interval="1m",
        features=["sma"],
        threshold=0.6,
        risk_pct=0.1,
    )
    assert results[0]["symbol"] == "AAPL"
    assert "fallback stub" in caplog.text


def test_predict_bridge_real_success(monkeypatch):
    monkeypatch.setenv("HYPRL_PREDICT_IMPL", "real")
    results = service.predict_batch(
        symbols=["AAPL", "MSFT"],
        interval="1m",
        features=["sma"],
        threshold=0.55,
        risk_pct=0.2,
    )
    assert len(results) == 2
    for row in results:
        assert {"symbol", "prob_up", "direction", "threshold", "tp", "sl"} <= set(row.keys())
        assert isinstance(row["prob_up"], float)
