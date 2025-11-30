"""Autorank → sessions orchestrator for HyprL V2."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from api import repo
from api.schemas import AutorankStartRequest, AutorankSessionConfig, StartSessionRequest
from api.session_manager import SessionAccessError, SessionNotFoundError, session_manager
from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.autorank import (
    AutorankConstraints,
    apply_autorank_filters,
    load_meta_info,
    write_summary,
)
from hyprl.meta.model import MetaRobustnessModel


logger = logging.getLogger(__name__)

ALLOWED_CSV_ROOTS = [Path("data/experiments").resolve(), Path("docs/experiments").resolve()]
JOB_ROOT = Path("data/autorank_jobs").resolve()
IDEMPOTENCE_WINDOW_SECONDS = 600
JOB_COST = 10
SESSION_UNIT_COST = 50


class AutorankJobExistsError(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"Autorank job already exists: {job_id}")
        self.job_id = job_id


class AutorankJobNotFoundError(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"Autorank job not found: {job_id}")
        self.job_id = job_id


class AutorankPathError(ValueError):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class AutorankSessionRecord:
    rank: int
    session_id: Optional[str]
    source_csv: str
    config_index: int


@dataclass
class AutorankJob:
    job_id: str
    account_id: str
    token_id: str
    params_hash: str
    artifacts_dir: Path
    autoranked_csv: Path
    summary_path: Path
    created_at: float = field(default_factory=lambda: time.time())
    status: str = "running"
    sessions: List[AutorankSessionRecord] = field(default_factory=list)
    debited_credits: int = 0
    dry_run: bool = False
    requested_top_k: int = 0
    filters: Dict[str, int] | None = None
    error: str | None = None


class AutorankManager:
    def __init__(self) -> None:
        self._jobs: dict[str, AutorankJob] = {}
        self._hash_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = asyncio.Lock()
        JOB_ROOT.mkdir(parents=True, exist_ok=True)

    async def start_autorank(
        self,
        payload: AutorankStartRequest,
        *,
        account_id: str,
        token_id: str,
        db: Session,
    ) -> AutorankJob:
        csv_paths = self._sanitize_csvs(payload.csv_paths)
        params_hash = self._fingerprint(account_id, payload)
        cache_key = (account_id, params_hash)
        async with self._lock:
            existing = self._check_recent(account_id, params_hash)
            if existing:
                raise AutorankJobExistsError(existing)
            job_id = self._generate_job_id()
            job_dir = (JOB_ROOT / job_id).resolve()
            job_dir.mkdir(parents=True, exist_ok=True)
            autoranked_csv = job_dir / "autoranked.csv"
            summary_path = job_dir / "autoranked.SUMMARY.txt"
            job = AutorankJob(
                job_id=job_id,
                account_id=account_id,
                token_id=token_id,
                params_hash=params_hash,
                artifacts_dir=job_dir,
                autoranked_csv=autoranked_csv,
                summary_path=summary_path,
                dry_run=payload.dry_run,
                requested_top_k=payload.top_k,
            )
            self._jobs[job_id] = job
            self._hash_cache[cache_key] = (job_id, time.time())

        try:
            diag = self._build_diag(csv_paths, payload)
            filtered, stats = apply_autorank_filters(diag, self._build_constraints(payload.constraints))
            filtered = self._rank_filtered(filtered, payload.seed)
            filtered.to_csv(autoranked_csv, index=False)
            meta_path = Path(payload.meta_model).resolve() if payload.meta_model else None
            meta_info = load_meta_info(meta_path) if meta_path else {
                "meta_model_type": "",
                "meta_dataset_hash": "",
                "meta_trained_at": "",
            }
            write_summary(
                summary_path,
                meta_path,
                meta_info,
                filtered,
                float(payload.meta_weight),
                payload.seed,
                payload.top_k,
                stats,
            )
            session_targets = filtered.head(payload.top_k)
            job.filters = stats
            actual_sessions = min(payload.top_k, len(session_targets))
            total_cost = 0 if payload.dry_run else JOB_COST + SESSION_UNIT_COST * actual_sessions
            if not payload.dry_run and total_cost > 0:
                try:
                    repo.debit_for_autorank(
                        db,
                        account_id=account_id,
                        token_id=token_id,
                        total_cost=total_cost,
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                    raise
            job.debited_credits = total_cost
            if not payload.dry_run and actual_sessions:
                session_records = await self._launch_sessions(
                    session_targets.head(actual_sessions),
                    payload.session,
                    account_id,
                    token_id,
                    db,
                )
                job.sessions.extend(session_records)
            job.status = "finished"
            return job
        except Exception as exc:
            logger.exception("Autorank job %s failed: %s", job_id, exc)
            job.status = "failed"
            job.error = str(exc)
            async with self._lock:
                self._hash_cache.pop(cache_key, None)
                self._jobs.pop(job_id, None)
            raise

    async def get_status(
        self,
        job_id: str,
        *,
        account_id: str,
    ) -> AutorankJob:
        job = self._jobs.get(job_id)
        if not job or job.account_id != account_id:
            raise AutorankJobNotFoundError(job_id)
        return job

    def reset(self) -> None:
        self._jobs.clear()
        self._hash_cache.clear()

    def _sanitize_csvs(self, paths: Sequence[str]) -> List[Path]:
        resolved: List[Path] = []
        for raw in paths:
            path = Path(raw).resolve()
            if not any(str(path).startswith(str(root)) for root in ALLOWED_CSV_ROOTS):
                raise AutorankPathError(f"CSV path outside allowed roots: {raw}")
            if not path.exists():
                raise AutorankPathError(f"CSV not found: {raw}")
            resolved.append(path)
        return resolved

    def _build_diag(self, csv_paths: List[Path], payload: AutorankStartRequest) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            frames.append(df)
        if not frames:
            raise ValueError("No CSV data provided")
        super_df = pd.concat(frames, ignore_index=True)
        model = None
        meta_path = payload.meta_model
        if meta_path:
            resolved = Path(meta_path).resolve()
            if not resolved.exists():
                raise AutorankPathError(f"Meta model not found: {meta_path}")
            model = MetaRobustnessModel.load(resolved)
        diag = build_meta_diag_frame(
            super_df,
            meta_weight=float(payload.meta_weight),
            model=model,
            calibrator=None,
        )
        return diag

    def _build_constraints(self, data: Optional[Dict[str, float]]) -> AutorankConstraints:
        data = data or {}
        return AutorankConstraints(
            min_pf=self._maybe_float(data.get("min_pf")),
            min_sharpe=self._maybe_float(data.get("min_sharpe")),
            max_dd=self._maybe_float(data.get("max_dd")),
            max_corr=self._maybe_float(data.get("max_corr")),
            min_trades=self._maybe_int(data.get("min_trades")),
            min_weight=self._maybe_float(data.get("min_weight")),
            max_weight=self._maybe_float(data.get("max_weight")),
        )

    def _rank_filtered(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        if df.empty:
            return df
        rng = np.random.default_rng(seed)
        df = df.copy()
        df["_tie"] = rng.random(len(df))
        ranked = df.sort_values(["final_score", "base_score_normalized", "_tie"], ascending=[False, False, True])
        return ranked.drop(columns=["_tie"])

    async def _launch_sessions(
        self,
        frame: pd.DataFrame,
        session_cfg: AutorankSessionConfig,
        account_id: str,
        token_id: str,
        db: Session,
    ) -> List[AutorankSessionRecord]:
        sessions: List[AutorankSessionRecord] = []
        for idx, row in enumerate(frame.itertuples(), start=1):
            symbols = self._extract_symbols(getattr(row, "tickers", ""))
            if not symbols:
                continue
            session_payload = StartSessionRequest(
                symbols=symbols,
                interval=session_cfg.interval,
                threshold=session_cfg.threshold,
                risk_pct=session_cfg.risk_pct,
                kill_switch_dd=session_cfg.kill_switch_dd,
                resume_session=None,
                enable_paper=session_cfg.enable_paper,
            )
            session_info = await session_manager.start_session(
                session_payload,
                account_id=account_id,
                token_id=token_id,
                db=db,
                billable=False,
            )
            sessions.append(
                AutorankSessionRecord(
                    rank=idx,
                    session_id=session_info["session_id"],
                    source_csv=str(getattr(row, "source_csv", "")),
                    config_index=int(getattr(row, "config_index", idx - 1)),
                )
            )
        return sessions

    def _extract_symbols(self, tickers: str | Iterable[str]) -> List[str]:
        if isinstance(tickers, str):
            parts = [token.strip().upper() for token in tickers.replace(";", ",").split(",") if token.strip()]
            return parts
        if isinstance(tickers, Iterable):
            return [str(item).upper() for item in tickers if str(item).strip()]
        return []

    def _generate_job_id(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        suffix = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:4]
        return f"ar_{ts}_{suffix}"

    def _fingerprint(self, account_id: str, payload: AutorankStartRequest) -> str:
        normalized = {
            "account_id": account_id,
            "csv_paths": sorted([str(Path(path).resolve()) for path in payload.csv_paths]),
            "top_k": payload.top_k,
            "meta_model": payload.meta_model or "",
            "meta_weight": float(payload.meta_weight),
            "constraints": payload.constraints or {},
            "session": payload.session.model_dump(),
            "seed": payload.seed,
            "dry_run": payload.dry_run,
        }
        serialized = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _check_recent(self, account_id: str, params_hash: str) -> Optional[str]:
        key = (account_id, params_hash)
        record = self._hash_cache.get(key)
        if not record:
            return None
        job_id, timestamp = record
        if time.time() - timestamp <= IDEMPOTENCE_WINDOW_SECONDS:
            return job_id
        self._hash_cache.pop(key, None)
        return None

    @staticmethod
    def _maybe_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _maybe_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


autorank_manager = AutorankManager()
