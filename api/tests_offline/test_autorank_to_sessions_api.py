import asyncio
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.responses import Response

from api import repo
from api.auth import AuthContext
from api.autorank_manager import autorank_manager
from api.routers.v2_autorank import autorank_start, autorank_status
from api.schemas import AutorankSessionConfig, AutorankStartRequest
from api.session_manager import session_manager
from api.tests_offline.helpers import build_request


def _write_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "tickers": "AAPL,MSFT",
                "config_index": 0,
                "base_score_normalized": 0.8,
                "pf_backtest": 1.2,
                "portfolio_sharpe": 0.45,
                "sharpe_backtest": 0.45,
                "maxdd_backtest": 0.18,
                "portfolio_dd": 18,
                "n_trades": 60,
                "trades_backtest": 60,
            },
            {
                "tickers": "MSFT",
                "config_index": 2,
                "base_score_normalized": 0.7,
                "pf_backtest": 1.15,
                "portfolio_sharpe": 0.4,
                "sharpe_backtest": 0.4,
                "maxdd_backtest": 0.2,
                "portfolio_dd": 20,
                "n_trades": 40,
                "trades_backtest": 40,
            },
            {
                "tickers": "NVDA",
                "config_index": 4,
                "base_score_normalized": 0.3,
                "pf_backtest": 0.9,
                "portfolio_sharpe": 0.1,
                "sharpe_backtest": 0.1,
                "maxdd_backtest": 0.5,
                "portfolio_dd": 50,
                "n_trades": 12,
                "trades_backtest": 12,
            },
        ]
    )
    df.to_csv(path, index=False)
    return path


def _session_config() -> AutorankSessionConfig:
    return AutorankSessionConfig(
        interval="1m",
        threshold=0.6,
        risk_pct=0.1,
        kill_switch_dd=0.30,
        enable_paper=False,
    )


def test_autorank_api_start_and_status(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    csv_path = _write_csv(Path("data/experiments/test_autorank_api.csv"))
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_autorank_api",
            scopes=["write:session", "read:usage", "read:session"],
            label="autorank_api",
            credits_total=500,
        )
        db.commit()

    payload = AutorankStartRequest(
        csv_paths=[str(csv_path)],
        top_k=2,
        meta_model=None,
        meta_weight=0.2,
        constraints={"min_pf": 1.0, "max_dd": 0.5, "min_trades": 20},
        session=_session_config(),
        seed=42,
        dry_run=False,
    )

    async def _run():
        req = build_request(token=token_plain, method="POST", path="/v2/autorank/start")
        req.state.token_id = token.id
        resp = Response()
        auth_ctx = AuthContext(account_id="acc_autorank_api", token_id=token.id, scopes={"write:session"})
        with SessionLocal() as db:
            start = await autorank_start(payload, req, resp, auth_ctx=auth_ctx, db=db)
        assert start.debited_credits == 110
        assert len(start.sessions) == 2
        assert Path(start.autoranked_csv).exists()
        assert Path(start.summary_txt).exists()

        status_req = build_request(token=token_plain, method="GET", path=f"/v2/autorank/{start.autorank_id}")
        status_req.state.token_id = token.id
        status_resp = Response()
        status_ctx = AuthContext(account_id="acc_autorank_api", token_id=token.id, scopes={"read:usage"})
        status = await autorank_status(start.autorank_id, status_req, status_resp, auth_ctx=status_ctx)
        assert status.status == "finished"
        assert len(status.sessions) == 2

        for session_info in start.sessions:
            if session_info.session_id:
                await session_manager.stop_session(
                    session_info.session_id,
                    account_id="acc_autorank_api",
                    scopes={"write:session"},
                )

    try:
        asyncio.run(_run())
    finally:
        autorank_manager.reset()


def test_autorank_api_insufficient_credits(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    csv_path = _write_csv(Path("data/experiments/test_autorank_api_low.csv"))
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_autorank_low",
            scopes=["write:session", "read:usage"],
            label="autorank_low",
            credits_total=20,
        )
        db.commit()

    payload = AutorankStartRequest(
        csv_paths=[str(csv_path)],
        top_k=2,
        meta_model=None,
        meta_weight=0.2,
        constraints={},
        session=_session_config(),
        seed=7,
        dry_run=False,
    )

    async def _run():
        req = build_request(token=token_plain, method="POST", path="/v2/autorank/start")
        req.state.token_id = token.id
        resp = Response()
        auth_ctx = AuthContext(account_id="acc_autorank_low", token_id=token.id, scopes={"write:session"})
        with SessionLocal() as db:
            with pytest.raises(HTTPException) as exc:
                await autorank_start(payload, req, resp, auth_ctx=auth_ctx, db=db)
        assert exc.value.status_code == 402

    try:
        asyncio.run(_run())
    finally:
        autorank_manager.reset()


def test_autorank_api_path_guard(sqlite_session):
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_autorank_path",
            scopes=["write:session"],
            label="autorank_path",
            credits_total=200,
        )
        db.commit()

    payload = AutorankStartRequest(
        csv_paths=["../outside.csv"],
        top_k=1,
        meta_model=None,
        meta_weight=0.2,
        constraints={},
        session=_session_config(),
        seed=1,
        dry_run=True,
    )

    async def _run():
        req = build_request(token=token_plain, method="POST", path="/v2/autorank/start")
        req.state.token_id = token.id
        resp = Response()
        auth_ctx = AuthContext(account_id="acc_autorank_path", token_id=token.id, scopes={"write:session"})
        with SessionLocal() as db:
            with pytest.raises(HTTPException) as exc:
                await autorank_start(payload, req, resp, auth_ctx=auth_ctx, db=db)
        assert exc.value.status_code == 400

    asyncio.run(_run())
    autorank_manager.reset()


def test_autorank_api_idempotence(sqlite_session, monkeypatch):
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    csv_path = _write_csv(Path("data/experiments/test_autorank_api_idem.csv"))
    SessionLocal = sqlite_session
    with SessionLocal() as db:
        token, token_plain = repo.create_token(
            db,
            account_id="acc_autorank_idem",
            scopes=["write:session", "read:usage"],
            label="autorank_idem",
            credits_total=500,
        )
        db.commit()

    payload = AutorankStartRequest(
        csv_paths=[str(csv_path)],
        top_k=1,
        meta_model=None,
        meta_weight=0.3,
        constraints={},
        session=_session_config(),
        seed=99,
        dry_run=True,
    )

    async def _run():
        req = build_request(token=token_plain, method="POST", path="/v2/autorank/start")
        req.state.token_id = token.id
        resp = Response()
        auth_ctx = AuthContext(account_id="acc_autorank_idem", token_id=token.id, scopes={"write:session"})
        with SessionLocal() as db:
            await autorank_start(payload, req, resp, auth_ctx=auth_ctx, db=db)
        req2 = build_request(token=token_plain, method="POST", path="/v2/autorank/start")
        req2.state.token_id = token.id
        resp2 = Response()
        with SessionLocal() as db:
            with pytest.raises(HTTPException) as exc:
                await autorank_start(payload, req2, resp2, auth_ctx=auth_ctx, db=db)
        assert exc.value.status_code == 409

    try:
        asyncio.run(_run())
    finally:
        autorank_manager.reset()
