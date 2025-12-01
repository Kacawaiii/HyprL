from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List

from hyprl.live.broker import PaperBrokerImpl


@dataclass(slots=True)
class PortfolioRunStats:
    """Accumulates portfolio-level execution stats for a run."""

    blocked: List[dict] = field(default_factory=list)
    scaled: List[dict] = field(default_factory=list)
    executed: List[dict] = field(default_factory=list)

    def record_blocked(self, *, symbol: str, reason: str, risk_amount: float) -> None:
        self.blocked.append(
            {"symbol": symbol.upper(), "reason": reason, "risk_amount": float(risk_amount)}
        )

    def record_scaled(
        self,
        *,
        symbol: str,
        scale: float,
        risk_before: float,
        risk_after: float,
    ) -> None:
        self.scaled.append(
            {
                "symbol": symbol.upper(),
                "scale": float(scale),
                "risk_before": float(risk_before),
                "risk_after": float(risk_after),
            }
        )

    def record_executed(
        self,
        *,
        symbol: str,
        expected_pnl: float,
        probability_up: float,
        direction: str,
        threshold: float,
    ) -> None:
        self.executed.append(
            {
                "symbol": symbol.upper(),
                "expected_pnl": float(expected_pnl),
                "probability_up": float(probability_up),
                "direction": direction,
                "threshold": float(threshold),
            }
        )

    def record_risk(
        self,
        *,
        open_risk: Dict[str, float],
        equity_ref: float,
    ) -> None:
        # Deprecated placeholder kept for signature compatibility; no-op to avoid affecting execution.
        return None


def _compute_skews(executed: Iterable[dict]) -> Dict[str, float]:
    accum: Dict[str, tuple[float, int]] = {}
    for entry in executed:
        symbol = entry["symbol"]
        prob = float(entry["probability_up"])
        thresh = float(entry["threshold"])
        direction = entry["direction"]
        skew = prob - thresh if direction == "long" else thresh - prob
        total, count = accum.get(symbol, (0.0, 0))
        accum[symbol] = (total + skew, count + 1)
    return {symbol: total / count for symbol, (total, count) in accum.items() if count > 0}


def build_portfolio_summary(
    *,
    broker: PaperBrokerImpl,
    tickers: Iterable[str],
    stats: PortfolioRunStats,
    summary_path: Path,
    guard_state: dict | None = None,
    equity_ref: float,
    portfolio_limits: "PortfolioRiskLimits",
) -> dict:
    equity = broker.get_balance()
    expected_portfolio_pnl = sum(item["expected_pnl"] for item in stats.executed)
    per_ticker_skew = _compute_skews(stats.executed)
    open_risk = broker.get_open_risk_amounts()
    total_open = sum(open_risk.values())
    risk_used_pct_total = total_open / equity if equity > 0 else 0.0
    risk_used_pct_by_ticker = {k: (v / equity if equity > 0 else 0.0) for k, v in open_risk.items()}

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tickers": [t.upper() for t in tickers],
        "equity": equity,
        "risk_used_pct_total": risk_used_pct_total,
        "risk_used_pct_by_ticker": risk_used_pct_by_ticker,
        "blocked_trades": stats.blocked,
        "scaled_trades": stats.scaled,
        "executed_trades": stats.executed,
        "expected_portfolio_pnl": expected_portfolio_pnl,
        "per_ticker_skew": per_ticker_skew,
        "counts": {
            "blocked": len(stats.blocked),
            "scaled": len(stats.scaled),
            "executed": len(stats.executed),
        },
        "equity_ref": equity_ref,
    }
    if guard_state:
        payload["guard"] = guard_state
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload
