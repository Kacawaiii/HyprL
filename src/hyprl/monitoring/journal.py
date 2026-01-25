"""
Trade Journal for HyprL
========================
- Log tous les trades avec détails complets
- Analyse post-mortem automatique
- Export CSV/JSON pour analyse externe
- Stats par symbole, par jour, par stratégie
"""

import json
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import statistics


@dataclass
class TradeRecord:
    """Single trade record."""
    id: str
    timestamp_entry: str
    timestamp_exit: Optional[str]
    symbol: str
    direction: str  # long/short
    shares: float
    entry_price: float
    exit_price: Optional[float]
    stop_price: float
    tp_price: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    exit_reason: Optional[str]  # stop/tp/trailing/manual/eod
    strategy: str
    rsi_at_entry: Optional[float]
    momentum_at_entry: Optional[float]
    bars_held: Optional[int]
    notes: str = ""


class TradeJournal:
    """Manage trade journal with analysis capabilities."""

    def __init__(self, journal_dir: str = "live/journal"):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        self._open_trades: Dict[str, TradeRecord] = {}
        self._trade_counter = 0

    def _get_journal_file(self, date: Optional[datetime] = None) -> Path:
        """Get journal file path for a specific date."""
        if date is None:
            date = datetime.now(timezone.utc)
        return self.journal_dir / f"trades_{date.strftime('%Y-%m')}.jsonl"

    def _generate_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{self._trade_counter:04d}"

    def open_trade(
        self,
        symbol: str,
        direction: str,
        shares: float,
        entry_price: float,
        stop_price: float,
        tp_price: float,
        strategy: str = "mvp",
        rsi: Optional[float] = None,
        momentum: Optional[float] = None,
        notes: str = ""
    ) -> str:
        """Record a new trade entry. Returns trade ID."""
        trade_id = self._generate_id()
        
        record = TradeRecord(
            id=trade_id,
            timestamp_entry=datetime.now(timezone.utc).isoformat(),
            timestamp_exit=None,
            symbol=symbol,
            direction=direction,
            shares=shares,
            entry_price=entry_price,
            exit_price=None,
            stop_price=stop_price,
            tp_price=tp_price,
            pnl=None,
            pnl_pct=None,
            exit_reason=None,
            strategy=strategy,
            rsi_at_entry=rsi,
            momentum_at_entry=momentum,
            bars_held=None,
            notes=notes
        )
        
        self._open_trades[trade_id] = record
        return trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        bars_held: int = 0,
        notes: str = ""
    ) -> Optional[TradeRecord]:
        """Close an open trade and save to journal."""
        if trade_id not in self._open_trades:
            return None

        record = self._open_trades.pop(trade_id)
        record.timestamp_exit = datetime.now(timezone.utc).isoformat()
        record.exit_price = exit_price
        record.exit_reason = exit_reason
        record.bars_held = bars_held
        
        # Calculate PnL
        if record.direction == "long":
            record.pnl_pct = (exit_price - record.entry_price) / record.entry_price
        else:
            record.pnl_pct = (record.entry_price - exit_price) / record.entry_price
        
        record.pnl = record.pnl_pct * record.shares * record.entry_price
        
        if notes:
            record.notes = f"{record.notes} | {notes}".strip(" | ")

        # Append to journal file
        self._append_to_journal(record)
        
        return record

    def _append_to_journal(self, record: TradeRecord) -> None:
        """Append trade record to journal file."""
        journal_file = self._get_journal_file()
        
        with open(journal_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def get_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[TradeRecord]:
        """Load trades from journal with optional filters."""
        trades = []
        
        # Find all journal files in range
        for journal_file in sorted(self.journal_dir.glob("trades_*.jsonl")):
            with open(journal_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        record = TradeRecord(**data)
                        
                        # Apply filters
                        if symbol and record.symbol != symbol:
                            continue
                        if strategy and record.strategy != strategy:
                            continue
                        if start_date:
                            entry_dt = datetime.fromisoformat(record.timestamp_entry)
                            if entry_dt < start_date:
                                continue
                        if end_date:
                            entry_dt = datetime.fromisoformat(record.timestamp_entry)
                            if entry_dt > end_date:
                                continue
                        
                        trades.append(record)
                    except Exception:
                        continue
        
        return trades

    def get_stats(
        self,
        trades: Optional[List[TradeRecord]] = None,
        **filters
    ) -> Dict[str, Any]:
        """Calculate statistics for trades."""
        if trades is None:
            trades = self.get_trades(**filters)

        if not trades:
            return {"error": "No trades found"}

        completed = [t for t in trades if t.pnl is not None]
        
        if not completed:
            return {"error": "No completed trades"}

        wins = [t for t in completed if t.pnl > 0]
        losses = [t for t in completed if t.pnl <= 0]
        
        pnls = [t.pnl for t in completed]
        pnl_pcts = [t.pnl_pct for t in completed]
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        return {
            "total_trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed),
            "total_pnl": sum(pnls),
            "avg_pnl": statistics.mean(pnls),
            "avg_pnl_pct": statistics.mean(pnl_pcts),
            "max_win": max(pnls),
            "max_loss": min(pnls),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            "avg_win": statistics.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": statistics.mean([t.pnl for t in losses]) if losses else 0,
            "avg_bars_held": statistics.mean([t.bars_held for t in completed if t.bars_held]) if any(t.bars_held for t in completed) else 0,
            "by_exit_reason": self._count_by(completed, "exit_reason"),
            "by_symbol": self._count_by(completed, "symbol"),
            "by_direction": self._count_by(completed, "direction")
        }

    def _count_by(self, trades: List[TradeRecord], field: str) -> Dict[str, int]:
        """Count trades by a specific field."""
        counts = {}
        for t in trades:
            key = getattr(t, field, "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def export_csv(self, output_path: str, **filters) -> str:
        """Export trades to CSV."""
        trades = self.get_trades(**filters)
        
        if not trades:
            return "No trades to export"

        fieldnames = [
            "id", "timestamp_entry", "timestamp_exit", "symbol", "direction",
            "shares", "entry_price", "exit_price", "stop_price", "tp_price",
            "pnl", "pnl_pct", "exit_reason", "strategy", "rsi_at_entry",
            "momentum_at_entry", "bars_held", "notes"
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow(asdict(trade))

        return f"Exported {len(trades)} trades to {output_path}"

    def print_summary(self, **filters) -> None:
        """Print a summary of trading stats."""
        stats = self.get_stats(**filters)
        
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return

        print("\n" + "=" * 50)
        print("TRADE JOURNAL SUMMARY")
        print("=" * 50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win/Loss: {stats['wins']}/{stats['losses']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Total PnL: ${stats['total_pnl']:,.0f}")
        print(f"Avg PnL: ${stats['avg_pnl']:,.0f} ({stats['avg_pnl_pct']:.2%})")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Win: ${stats['max_win']:,.0f}")
        print(f"Max Loss: ${stats['max_loss']:,.0f}")
        print(f"Avg Bars Held: {stats['avg_bars_held']:.1f}")
        
        print("\nBy Exit Reason:")
        for reason, count in stats['by_exit_reason'].items():
            print(f"  {reason}: {count}")
        
        print("\nBy Symbol:")
        for sym, count in stats['by_symbol'].items():
            print(f"  {sym}: {count}")


# Global instance
_journal: Optional[TradeJournal] = None

def get_journal() -> TradeJournal:
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal
