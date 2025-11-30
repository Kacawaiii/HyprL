from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


class LiveLogger:
    """Writes paper-trading events under data/live/sessions/<session_id>."""

    def __init__(self, session_id: str, base_dir: Path | str = Path("data/live/sessions")) -> None:
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.trades_path = self.session_dir / "trades.csv"
        self.equity_path = self.session_dir / "equity.csv"
        self._trade_header_written = self.trades_path.exists()
        self._equity_header_written = self.equity_path.exists()

    def log_trade(
        self,
        *,
        timestamp,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        realized_pnl: float,
        order_id: str,
        cash_after: float,
    ) -> None:
        headers = [
            "timestamp",
            "ticker",
            "side",
            "quantity",
            "price",
            "realized_pnl",
            "order_id",
            "cash_after",
        ]
        with self.trades_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            if not self._trade_header_written:
                writer.writeheader()
                self._trade_header_written = True
            writer.writerow(
                {
                    "timestamp": str(timestamp),
                    "ticker": ticker,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "realized_pnl": realized_pnl,
                    "order_id": order_id,
                    "cash_after": cash_after,
                }
            )

    def log_equity(
        self,
        *,
        timestamp,
        equity: float,
        cash: float,
        positions: Dict[str, Dict[str, float]],
    ) -> None:
        headers = ["timestamp", "equity", "cash", "positions"]
        with self.equity_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            if not self._equity_header_written:
                writer.writeheader()
                self._equity_header_written = True
            writer.writerow(
                {
                    "timestamp": str(timestamp),
                    "equity": equity,
                    "cash": cash,
                    "positions": json.dumps(positions),
                }
            )

    def session_files(self) -> dict[str, Path]:
        return {"trades": self.trades_path, "equity": self.equity_path}
