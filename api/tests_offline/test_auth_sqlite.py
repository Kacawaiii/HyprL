import asyncio

import pytest
from fastapi import HTTPException

from api import repo
from api.auth import require_scopes
from api.tests_offline.helpers import build_request


def test_auth_dependency(sqlite_session):
    SessionLocal = sqlite_session
    with SessionLocal() as session:
        ok_token, ok_plain = repo.create_token(
            session,
            account_id="acc_ok",
            scopes=["read:predict", "read:usage"],
            label="ok",
            credits_total=50,
        )
        limited_token, limited_plain = repo.create_token(
            session,
            account_id="acc_limited",
            scopes=["read:predict"],
            label="limited",
            credits_total=5,
        )
        session.commit()

    dep_predict = require_scopes({"read:predict"})
    dep_usage = require_scopes({"read:usage"})

    request = build_request(token=ok_plain)
    async def _call_dep(req):
        with SessionLocal() as session:
            return await dep_predict(req, session)

    ctx = asyncio.run(_call_dep(request))
    assert ctx.account_id == "acc_ok"
    assert ctx.token_id == ok_token.id

    no_header_request = build_request(include_auth=False)
    async def _missing():
        with SessionLocal() as session:
            await dep_predict(no_header_request, session)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_missing())
    assert excinfo.value.status_code == 401

    bad_token_request = build_request(token="tok_bad.invalid")
    async def _bad():
        with SessionLocal() as session:
            await dep_predict(bad_token_request, session)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_bad())
    assert excinfo.value.status_code == 401

    limited_request = build_request(token=limited_plain)
    async def _limited():
        with SessionLocal() as session:
            await dep_usage(limited_request, session)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_limited())
    assert excinfo.value.status_code == 403
