from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

PORTFOLIO_HEALTH_PATH = Path(os.getenv("HYPRL_PORTFOLIO_HEALTH", "live/logs/portfolio_live/health_asc_v2.json"))
LIVE_ROOT = Path(os.getenv("HYPRL_LIVE_LOG_ROOT", "live/logs"))

router = APIRouter(prefix="/v2", tags=["portfolio"])


def _ticker_path(ticker: str) -> Path:
    sym = ticker.upper()
    return LIVE_ROOT / f"live_{sym.lower()}" / f"trades_{sym}_live_all.csv"


def _read_last_trade(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception:
        return None
    if not rows:
        return None
    last = rows[-1]
    return {
        "ticker": last.get("ticker") or path.stem.split("_")[1].upper(),
        "entry_timestamp": last.get("entry_timestamp"),
        "exit_timestamp": last.get("exit_timestamp"),
        "direction": last.get("direction"),
        "pnl": float(last["pnl"]) if last.get("pnl") else None,
        "position_size": float(last["position_size"]) if last.get("position_size") else None,
    }


@router.get("/signal")
async def latest_signal(ticker: str = Query(..., description="Ticker symbol, e.g. NVDA")) -> dict:
    """Return the last trade info for a ticker from live trade logs."""
    path = _ticker_path(ticker)
    trade = _read_last_trade(path)
    if not trade:
        raise HTTPException(status_code=404, detail={"error": "not_found", "ticker": ticker})
    return trade


@router.get("/portfolio")
async def portfolio_health() -> dict:
    """Return the current portfolio health summary (PF/DD/Sharpe/status)."""
    if not PORTFOLIO_HEALTH_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "path": str(PORTFOLIO_HEALTH_PATH)},
        )
    try:
        import json

        return json.loads(PORTFOLIO_HEALTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail={"error": "read_error", "path": str(PORTFOLIO_HEALTH_PATH), "msg": str(exc)}
        )
