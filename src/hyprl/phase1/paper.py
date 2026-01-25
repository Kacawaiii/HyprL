from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from hyprl.backtest.runner import BacktestResult, run_backtest
from hyprl.execution.broker import PaperBroker
from hyprl.execution.engine import run_paper_trading_session
from hyprl.execution.logging import LiveLogger
from hyprl.execution.paper import PaperBuildParams, build_backtest_config_from_row


def _parse_ticker_list(payload: object) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        tokens = [token.strip() for token in payload.replace(";", ",").split(",") if token.strip()]
        return [token.upper() for token in tokens]
    if isinstance(payload, (list, tuple)):
        return [str(item).upper() for item in payload if str(item).strip()]
    return []


def _coerce_str(value: object, fallback: str | None = None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if pd.notna(value):
        return str(value)
    return fallback


@dataclass(slots=True)
class Phase1ExecutionConfig:
    period: str = "1y"
    interval: str = "1h"
    start: str | None = None
    end: str | None = None
    initial_balance: float = 10_000.0
    engine: str = "auto"
    model_type: str = "logistic"
    calibration: str = "none"
    default_long_threshold: float = 0.6
    default_short_threshold: float = 0.4
    session_prefix: str = "phase1"
    clock: Callable[[], float] | None = None


def run_phase1_sessions(panel_df: pd.DataFrame, config: Phase1ExecutionConfig) -> pd.DataFrame:
    if panel_df.empty:
        raise ValueError("Panel vide, aucun run Phase 1 à exécuter.")
    sessions: list[dict[str, object]] = []
    per_strategy_capital = config.initial_balance
    for _, entry in panel_df.iterrows():
        strat_id = entry.get("strat_id", "STRAT")
        tickers = _parse_ticker_list(entry.get("tickers"))
        if not tickers:
            raise ValueError(f"Tickers manquants pour {strat_id}")
        period = _coerce_str(entry.get("period"), config.period) or config.period
        interval = _coerce_str(entry.get("interval"), config.interval) or config.interval
        config_index = int(entry.get("config_index", 0))
        csv_path = Path(str(entry.get("source_csv")))
        df = pd.read_csv(csv_path)
        if config_index < 0 or config_index >= len(df):
            raise IndexError(f"config_index {config_index} invalide pour {csv_path}")
        row = df.iloc[config_index]
        capital_share = config.initial_balance / max(len(tickers), 1)
        params = PaperBuildParams(
            period=period,
            start=config.start,
            end=config.end,
            interval=interval,
            model_type=config.model_type,
            calibration=config.calibration,
            default_long_threshold=config.default_long_threshold,
            default_short_threshold=config.default_short_threshold,
        )
        ticker_results: dict[str, BacktestResult] = {}
        for ticker in tickers:
            bt_config = build_backtest_config_from_row(row, ticker=ticker, capital_share=capital_share, params=params)
            ticker_results[ticker] = run_backtest(bt_config)
        broker = PaperBroker(config.initial_balance)
        clock = config.clock or time.time
        session_id = f"{config.session_prefix}_{strat_id}_{int(clock())}"
        logger = LiveLogger(session_id)
        equity = run_paper_trading_session(ticker_results, broker, logger)
        sessions.append(
            {
                "session_id": session_id,
                "strat_id": strat_id,
                "source_csv": str(csv_path),
                "config_index": config_index,
                "tickers": ",".join(tickers),
                "interval": interval,
                "period": period,
                "initial_balance": config.initial_balance,
                "engine": config.engine,
                "log_dir": str(logger.session_dir),
                "n_points": len(equity),
            }
        )
    return pd.DataFrame(sessions)
