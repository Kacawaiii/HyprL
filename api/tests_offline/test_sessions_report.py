import asyncio

from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.routers.v2_sessions import delete_session, session_report, start_session
from api.schemas import StartSessionRequest
from api.session_manager import session_manager
from api.tests_offline.helpers import build_request


def test_sessions_report(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_report",
            scopes=["write:session", "read:session", "read:usage"],
            label="test_report",
            credits_total=500,
        )
        db.commit()

    payload = StartSessionRequest(
        symbols=["AAPL", "NVDA"],
        interval="1m",
        threshold=0.58,
        risk_pct=0.15,
        kill_switch_dd=0.45,
        resume_session=None,
        enable_paper=False,
    )

    async def run_flow():
        request = build_request(token=token_plain, method="POST", path="/v2/sessions")
        request.state.token_id = token.id
        response = Response()
        auth_ctx = AuthContext(account_id="acc_report", token_id=token.id, scopes={"write:session", "read:usage"})
        with SessionLocal() as db:
            start_resp = await start_session(payload, request, response, auth_ctx=auth_ctx, db=db)
        session_id = start_resp.session_id

        await asyncio.sleep(0.3)
        report_request = build_request(token=token_plain, method="GET", path=f"/v2/sessions/{session_id}/report")
        report_request.state.token_id = token.id
        report_ctx = AuthContext(account_id="acc_report", token_id=token.id, scopes={"read:session"})
        report_response = Response()
        report = await session_report(session_id, report_request, report_response, auth_ctx=report_ctx)
        assert report.metrics.pf == report.metrics.pf
        assert report.metrics.sharpe == report.metrics.sharpe
        assert report.metrics.dd >= 0
        assert report.metrics.winrate >= 0
        assert report.metrics.exposure >= 0
        assert report.metrics.avg_hold_bars >= 0
        assert isinstance(report.top_rejections, list)
        assert report.duration_s >= 0

        delete_req = build_request(token=token_plain, method="DELETE", path=f"/v2/sessions/{session_id}")
        delete_req.state.token_id = token.id
        delete_ctx = AuthContext(account_id="acc_report", token_id=token.id, scopes={"write:session"})
        await delete_session(session_id, delete_req, Response(), auth_ctx=delete_ctx)

    try:
        asyncio.run(run_flow())
    finally:
        asyncio.run(session_manager.shutdown())
