from __future__ import annotations

from fastapi.testclient import TestClient

from hyprl_api.main import app

client = TestClient(app)
DISCORD_SECRET_HEADER = {"X-Discord-Secret": "dev-discord-secret"}


def _issue_token(discord_id: str = "123456789012345678") -> str:
    resp = client.post(
        "/v2/token/resolve-discord",
        json={"discord_id": discord_id},
        headers=DISCORD_SECRET_HEADER,
    )
    assert resp.status_code == 200
    token = resp.json()["token"]
    assert "." in token
    return token


def test_predict_requires_token() -> None:
    payload = {
        "ticker": "AAPL",
        "period": "1y",
        "interval": "1h",
        "preset": "research_minimal",
    }
    resp = client.post("/v2/predict", json=payload)
    assert resp.status_code == 401


def test_resolve_token_and_usage_flow() -> None:
    token = _issue_token()

    usage_resp = client.get(
        "/v2/usage",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert usage_resp.status_code == 200
    usage_data = usage_resp.json()
    assert usage_data["daily_calls"] == 0

    payload = {
        "ticker": "AAPL",
        "period": "1y",
        "interval": "1h",
        "preset": "research_minimal",
    }
    predict_resp = client.post(
        "/v2/predict",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert predict_resp.status_code == 200
    data = predict_resp.json()
    for key in (
        "final_balance",
        "profit_factor",
        "sharpe",
        "max_drawdown_pct",
        "risk_of_ruin",
        "robustness_score",
        "pnl_10k",
    ):
        assert key in data
    assert data["ticker"] == "AAPL"

    usage_after = client.get(
        "/v2/usage",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert usage_after.status_code == 200
    assert usage_after.json()["daily_calls"] >= 1
