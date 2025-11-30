from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hyprl.backtest.runner import TradeRecord


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    diff = probs - targets
    return float(np.mean(diff * diff))


def log_loss_score(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-9) -> float:
    probs = np.clip(probs, eps, 1 - eps)
    loss = -targets * np.log(probs) - (1 - targets) * np.log(1 - probs)
    return float(np.mean(loss))


def trade_calibration_metrics(trades: Sequence["TradeRecord"]) -> dict[str, float | None]:
    if not trades:
        return {"brier": None, "log_loss": None}
    p_win: list[float] = []
    outcomes: list[float] = []
    for trade in trades:
        prob = trade.probability_up
        if trade.direction == "short":
            prob = 1.0 - prob
        p_win.append(prob)
        outcomes.append(1.0 if trade.pnl > 0 else 0.0)
    probs_arr = np.array(p_win, dtype=float)
    targets_arr = np.array(outcomes, dtype=float)
    return {
        "brier": brier_score(probs_arr, targets_arr),
        "log_loss": log_loss_score(probs_arr, targets_arr),
    }
