import asyncio

from fastapi import HTTPException
from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_sessions import delete_session, session_status, start_session
from api.schemas import StartSessionRequest
from api.session_manager import session_manager
from api.tests_offline.helpers import build_request


def test_sessions_delete_flow(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_delete",
            scopes=["write:session", "read:usage", "read:session"],
            label="test_delete",
            credits_total=500,
        )
        db.commit()

    payload = StartSessionRequest(
        symbols=["AAPL"],
        interval="1m",
        threshold=0.55,
        risk_pct=0.2,
        kill_switch_dd=0.4,
        resume_session=None,
        enable_paper=False,
    )

    async def run_flow():
        request = build_request(token=token_plain, method="POST", path="/v2/sessions")
        request.state.token_id = token.id
        response = Response()
        auth_ctx = AuthContext(account_id="acc_delete", token_id=token.id, scopes={"write:session", "read:usage"})
        with SessionLocal() as db:
            start_resp = await start_session(payload, request, response, auth_ctx=auth_ctx, db=db)
        session_id = start_resp.session_id
        await asyncio.sleep(0.15)

        delete_req = build_request(token=token_plain, method="DELETE", path=f"/v2/sessions/{session_id}")
        delete_req.state.token_id = token.id
        delete_ctx = AuthContext(account_id="acc_delete", token_id=token.id, scopes={"write:session"})
        delete_resp = Response()
        await delete_session(session_id, delete_req, delete_resp, auth_ctx=delete_ctx)

        status_req = build_request(token=token_plain, method="GET", path=f"/v2/sessions/{session_id}")
        status_req.state.token_id = token.id
        status_ctx = AuthContext(account_id="acc_delete", token_id=token.id, scopes={"read:usage"})
        status_resp = Response()
        try:
            result = await session_status(session_id, status_req, status_resp, auth_ctx=status_ctx)
            assert result.status != "running"
        except HTTPException as exc:
            assert exc.status_code == 404

    try:
        asyncio.run(run_flow())
    finally:
        asyncio.run(session_manager.shutdown())
