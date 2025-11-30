from __future__ import annotations

from typing import Any

import pytest

from portal.config import PortalSettings
from portal.hyprl_client import PortalHyprlClient
from portal.predict_monitor import append_predict_history


def test_append_predict_history(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = PortalSettings(api_base="http://hyprl.local", api_token="tok_test", title="Portal")
    client = PortalHyprlClient(settings)

    def fake_predict(
        self: PortalHyprlClient, symbols: list[str], threshold: float | None = None, risk_pct: float | None = None
    ) -> dict[str, Any]:
        assert symbols == ["AAPL", "MSFT"]
        assert threshold == 0.55
        assert risk_pct is None
        return {
            "results": [
                {
                    "prediction_id": "pred_1",
                    "symbol": "aapl",
                    "prob_up": 0.51,
                    "direction": "up",
                    "threshold": 0.55,
                    "tp": 123.4,
                    "sl": 120.0,
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "closed": True,
                    "outcome": "WIN",
                    "pnl": 1.0,
                },
                {
                    "prediction_id": "pred_2",
                    "symbol": "msft",
                    "prob_up": 0.44,
                    "direction": "down",
                    "threshold": 0.55,
                    "tp": None,
                    "sl": None,
                    "closed": False,
                    "outcome": None,
                    "pnl": None,
                },
            ]
        }

    monkeypatch.setattr(PortalHyprlClient, "predict", fake_predict, raising=False)
    response = client.predict(["AAPL", "MSFT"], threshold=0.55)
    history: list[dict] = []
    updated = append_predict_history(history, response, timestamp=1234.0)

    assert len(updated) == 2
    assert updated[0]["symbol"] == "AAPL"
    assert updated[0]["prob_up"] == pytest.approx(0.51)
    assert updated[0]["direction"] == "UP"
    assert updated[0]["threshold"] == 0.55
    assert updated[0]["prediction_id"] == "pred_1"
    assert updated[0]["pnl"] == pytest.approx(1.0)
    assert updated[0]["ts"] == pytest.approx(1735689600.0)
    assert updated[1]["symbol"] == "MSFT"
    assert updated[1]["ts"] > updated[0]["ts"]

    client.close()
