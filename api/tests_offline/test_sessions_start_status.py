import asyncio

from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_sessions import delete_session, session_status, start_session
from api.schemas import StartSessionRequest
from api.session_manager import session_manager
from api.tests_offline.helpers import build_request


def test_sessions_start_and_status(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_sessions",
            scopes=["write:session", "read:usage", "read:session"],
            label="test_sessions",
            credits_total=500,
        )
        db.commit()

    payload = StartSessionRequest(
        symbols=["AAPL", "MSFT"],
        interval="1m",
        threshold=0.6,
        risk_pct=0.1,
        kill_switch_dd=0.5,
        resume_session=None,
        enable_paper=False,
    )

    async def run_flow():
        request = build_request(token=token_plain, method="POST", path="/v2/sessions")
        request.state.token_id = token.id
        response = Response()
        auth_ctx = AuthContext(account_id="acc_sessions", token_id=token.id, scopes={"write:session", "read:usage"})
        with SessionLocal() as db:
            start_resp = await start_session(payload, request, response, auth_ctx=auth_ctx, db=db)
        assert start_resp.meta["debited"] == 50
        assert response.headers.get("X-RateLimit-Limit") is not None
        session_id = start_resp.session_id

        await asyncio.sleep(0.2)
        status_request = build_request(token=token_plain, method="GET", path=f"/v2/sessions/{session_id}")
        status_request.state.token_id = token.id
        status_response = Response()
        status_ctx = AuthContext(account_id="acc_sessions", token_id=token.id, scopes={"read:usage"})
        status_payload = await session_status(session_id, status_request, status_response, auth_ctx=status_ctx)
        assert status_payload.counters.bars > 0
        assert status_payload.counters.predictions > 0
        assert status_payload.status in {"running", "finished"}

        delete_request = build_request(token=token_plain, method="DELETE", path=f"/v2/sessions/{session_id}")
        delete_request.state.token_id = token.id
        delete_response = Response()
        delete_ctx = AuthContext(account_id="acc_sessions", token_id=token.id, scopes={"write:session"})
        resp = await delete_session(session_id, delete_request, delete_response, auth_ctx=delete_ctx)
        assert resp.status_code == 204

    try:
        asyncio.run(run_flow())
    finally:
        asyncio.run(session_manager.shutdown())
