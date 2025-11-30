import asyncio

import pytest
from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_predict import predict, predict_summary, update_prediction_outcome
from api.schemas import PredictRequest, PredictionOutcomeUpdateRequest
from api.tests_offline.helpers import build_request


@pytest.mark.usefixtures("sqlite_session")
def test_prediction_outcome_update_and_summary(sqlite_session):
    SessionLocal = sqlite_session
    with SessionLocal() as session:
        token, _ = repo.create_token(
            session,
            account_id="acc_predict",
            scopes=["read:predict", "read:usage"],
            label="predict",
            credits_total=50,
        )
        session.commit()

    payload = PredictRequest(
        symbols=["AAPL"],
        interval="1m",
        features=["sma"],
        threshold=0.55,
        risk_pct=0.1,
    )
    response = Response()
    auth_ctx = AuthContext(
        account_id="acc_predict",
        token_id=token.id,
        scopes={"read:predict", "read:usage"},
    )

    async def _invoke_predict():
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

    result = asyncio.run(_invoke_predict())
    prediction = result.results[0]
    assert prediction.closed is True

    async def _summary_call():
        with SessionLocal() as session:
            return await predict_summary(auth_ctx=auth_ctx, db=session)

    summary = asyncio.run(_summary_call())
    assert summary.total_predictions == 1
    assert summary.closed_predictions == 1
    assert summary.pnl_total != 0

    update_payload = PredictionOutcomeUpdateRequest(closed=True, pnl=2.5, outcome="WIN")

    async def _update_call():
        with SessionLocal() as session:
            return await update_prediction_outcome(
                prediction_id=prediction.prediction_id,
                payload=update_payload,
                auth_ctx=auth_ctx,
                db=session,
            )

    asyncio.run(_update_call())

    summary_after = asyncio.run(_summary_call())
    assert summary_after.pnl_total == pytest.approx(2.5)
    assert summary_after.win_predictions == 1
