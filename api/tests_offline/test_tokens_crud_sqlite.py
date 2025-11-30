import asyncio

import pytest
from fastapi import HTTPException

from api import repo
from api.auth import AuthContext, require_scopes
from api.models import Token
from api.routers.v2_tokens import create_token, revoke_token
from api.schemas import TokenCreateRequest
from api.tests_offline.helpers import build_request


def test_tokens_crud(sqlite_session):
    SessionLocal = sqlite_session
    admin_ctx = AuthContext(
        account_id=repo.ADMIN_ACCOUNT_ID,
        token_id="tok_admin",
        scopes=set(repo.ADMIN_SCOPES),
    )

    payload = TokenCreateRequest(
        account_id="acc_clientA",
        scopes=["read:predict", "read:usage"],
        label="clientA_bot",
        credits_total=100_000,
        expires_at=None,
    )

    async def _create():
        with SessionLocal() as session:
            return await create_token(payload=payload, db=session, _=admin_ctx)

    resp = asyncio.run(_create())

    assert resp.token_plain.startswith(resp.token_id)
    assert resp.scopes == ["read:predict", "read:usage"]

    with SessionLocal() as session:
        token_row = session.get(Token, resp.token_id)
        assert token_row is not None
        assert token_row.hash.startswith("$argon2id$")
        assert resp.token_plain not in token_row.hash
        account = repo.get_account(session, "acc_clientA")
        assert account is not None
        assert account.credits_total == 100_000
        assert account.credits_remaining == 100_000

    async def _revoke():
        with SessionLocal() as session:
            await revoke_token(token_id=resp.token_id, db=session, _=admin_ctx)

    asyncio.run(_revoke())

    request = build_request(token=resp.token_plain)
    dep = require_scopes({"read:predict"})
    async def _check():
        with SessionLocal() as session:
            await dep(request, session)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_check())
    assert excinfo.value.status_code == 401
