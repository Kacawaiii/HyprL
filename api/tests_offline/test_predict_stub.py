import asyncio

import pytest
from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_predict import predict
from api.schemas import PredictRequest
from api.tests_offline.helpers import build_request


def test_predict_stub_contract(sqlite_session):
    SessionLocal = sqlite_session
    with SessionLocal() as session:
        token, _ = repo.create_token(
            session,
            account_id="acc_test",
            scopes=["read:predict", "read:usage"],
            label="test",
            credits_total=10,
        )
        session.commit()

    payload = PredictRequest(
        symbols=["AAPL", "MSFT"],
        interval="1m",
        features=["sma", "rsi"],
        threshold=0.6,
        risk_pct=0.1,
    )
    response = Response()
    auth_ctx = AuthContext(
        account_id="acc_test",
        token_id=token.id,
        scopes={"read:predict", "read:usage"},
    )

    async def _run():
        request = build_request(include_auth=False)
        request.state.token_id = auth_ctx.token_id
        with SessionLocal() as session:
            result = await predict(
                payload=payload,
                request=request,
                response=response,
                auth_ctx=auth_ctx,
                db=session,
            )
            session.commit()
            return result

    result = asyncio.run(_run())

    assert result.meta == {"model": "prob_bridge_v2", "version": 2}
    payloads = [item.model_dump() for item in result.results]
    assert payloads[0]["symbol"] == "AAPL"
    assert payloads[0]["prediction_id"].startswith("pred_")
    assert payloads[0]["closed"] is True
    assert payloads[0]["outcome"] == "WIN"
    assert payloads[0]["pnl"] == pytest.approx(1.0)
    assert payloads[0]["tp"] == 123.45
    assert payloads[1]["outcome"] == "LOSS"
    assert payloads[1]["pnl"] == pytest.approx(-1.0)
    with SessionLocal() as session:
        account = repo.get_account(session, "acc_test")
        assert account is not None
        assert account.credits_remaining == 8
