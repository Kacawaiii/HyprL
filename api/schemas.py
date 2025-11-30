"""Pydantic schemas for HyprL V2 endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1)
    interval: str
    features: List[str]
    threshold: float = Field(..., ge=0.0, le=1.0)
    risk_pct: float = Field(..., ge=0.0)


class PredictItem(BaseModel):
    prediction_id: str
    symbol: str
    prob_up: float
    direction: str
    threshold: float
    risk_pct: Optional[float] = None
    tp: Optional[float] = None
    sl: Optional[float] = None
    closed: Optional[bool] = None
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class PredictResponse(BaseModel):
    results: List[PredictItem]
    meta: dict


class PredictionOutcomeUpdateRequest(BaseModel):
    closed: bool = True
    pnl: Optional[float] = None
    outcome: Optional[str] = None


class PredictSummaryResponse(BaseModel):
    total_predictions: int
    closed_predictions: int
    pending_predictions: int
    win_predictions: int
    winrate_real: Optional[float] = None
    avg_pnl: Optional[float] = None
    pnl_total: float


class TokenCreateRequest(BaseModel):
    account_id: str
    scopes: List[str]
    label: Optional[str] = None
    credits_total: Optional[int] = Field(default=None, ge=0)
    expires_at: Optional[datetime] = None


class TokenCreateResponse(BaseModel):
    token_id: str
    token_plain: str
    scopes: List[str]


class UsageResponse(BaseModel):
    account_id: str
    credits_total: int
    credits_remaining: int
    by_endpoint: Dict[str, int]


class StartSessionRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1)
    interval: Literal["1m", "5m", "15m", "1h"]
    threshold: float = Field(..., ge=0.0, le=1.0)
    risk_pct: float = Field(..., ge=0.0)
    kill_switch_dd: Optional[float] = Field(default=0.30, ge=0.0, le=1.0)
    resume_session: Optional[str] = None
    enable_paper: bool = False


class StartSessionResponse(BaseModel):
    session_id: str
    log_dir: str
    impl: str
    meta: Dict[str, int]


class SessionCounters(BaseModel):
    bars: int = 0
    predictions: int = 0
    fills: int = 0


class SessionStatusMetrics(BaseModel):
    pf: Optional[float] = None
    sharpe: Optional[float] = None
    dd: Optional[float] = Field(default=None, description="Session drawdown ratio")


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    last_event_ts: Optional[float]
    counters: SessionCounters
    kill_switch_triggered: bool
    metrics: Optional[SessionStatusMetrics] = None


class SessionReportMetrics(BaseModel):
    pf: float
    sharpe: float
    dd: float
    winrate: float
    exposure: float
    avg_hold_bars: float


class SessionReportResponse(BaseModel):
    session_id: str
    metrics: SessionReportMetrics
    top_rejections: List[Tuple[str, int]]
    duration_s: float


class AutorankSessionConfig(BaseModel):
    interval: Literal["1m", "5m", "15m", "1h"]
    threshold: float = Field(..., ge=0.0, le=1.0)
    risk_pct: float = Field(..., ge=0.0)
    kill_switch_dd: Optional[float] = Field(default=0.30, ge=0.0, le=1.0)
    enable_paper: bool = False


class AutorankStartRequest(BaseModel):
    csv_paths: List[str] = Field(..., min_length=1)
    top_k: int = Field(..., ge=1)
    meta_model: Optional[str] = None
    meta_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    constraints: Dict[str, float] | None = None
    session: AutorankSessionConfig
    seed: int = 42
    dry_run: bool = False


class AutorankSessionLaunch(BaseModel):
    rank: int
    session_id: Optional[str] = None
    source_csv: str
    config_index: int


class AutorankStartResponse(BaseModel):
    autorank_id: str
    artifacts_dir: str
    autoranked_csv: str
    summary_txt: str
    sessions: List[AutorankSessionLaunch]
    debited_credits: int


class AutorankSessionStatus(BaseModel):
    session_id: str
    status: str


class AutorankStatusResponse(BaseModel):
    autorank_id: str
    status: str
    sessions: List[AutorankSessionStatus]
    artifacts: Dict[str, str]
