import asyncio

import pytest
from fastapi import HTTPException
from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_predict import predict
from api.routers.v2_usage import usage
from api.schemas import PredictRequest
from api.tests_offline.helpers import build_request


def test_quota_persistence(sqlite_session):
    SessionLocal = sqlite_session
    with SessionLocal() as session:
        token, _ = repo.create_token(
            session,
            account_id="acc_quota",
            scopes=["read:predict", "read:usage"],
            label="quota",
            credits_total=5,
        )
        session.commit()

    auth_ctx = AuthContext(
        account_id="acc_quota",
        token_id=token.id,
        scopes={"read:predict", "read:usage"},
    )
    payload = PredictRequest(
        symbols=["AAPL", "MSFT", "GOOGL"],
        interval="1m",
        features=["sma"],
        threshold=0.55,
        risk_pct=0.1,
    )
    async def _first_call():
        request = build_request(include_auth=False)
        request.state.token_id = auth_ctx.token_id
        with SessionLocal() as session:
            await predict(
                payload=payload,
                request=request,
                response=Response(),
                auth_ctx=auth_ctx,
                db=session,
            )
            session.commit()

    asyncio.run(_first_call())

    with SessionLocal() as session:
        account = repo.get_account(session, "acc_quota")
        assert account is not None
        assert account.credits_remaining == 2
        summary = repo.usage_by_endpoint(session, "acc_quota")
        assert summary["v2/predict"] == 3
        account.credits_remaining = 1
        session.commit()

    async def _second_call():
        request = build_request(include_auth=False)
        request.state.token_id = auth_ctx.token_id
        with SessionLocal() as session:
            await predict(
                payload=payload,
                request=request,
                response=Response(),
                auth_ctx=auth_ctx,
                db=session,
            )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_second_call())
    assert excinfo.value.status_code == 402

    async def _usage():
        with SessionLocal() as session:
            return await usage(
                auth_ctx=auth_ctx,
                db=session,
            )

    usage_response = asyncio.run(_usage())
    assert usage_response.credits_total == 5
    assert usage_response.credits_remaining == 1
    assert usage_response.by_endpoint["v2/predict"] >= 3
