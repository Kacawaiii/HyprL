from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from hyprl.backtest.runner import BacktestResult, TradeRecord
from hyprl.execution.broker import PaperBroker
from hyprl.execution.logging import LiveLogger


@dataclass(order=True)
class TradeEvent:
    timestamp: pd.Timestamp
    priority: int
    ticker: str
    side: str
    quantity: float
    price: float
    description: str


def _build_events(trades_by_ticker: Mapping[str, Iterable[TradeRecord]]) -> List[TradeEvent]:
    events: List[TradeEvent] = []
    for ticker, trades in trades_by_ticker.items():
        for trade in trades:
            entry_ts = pd.Timestamp(trade.entry_timestamp)
            exit_ts = pd.Timestamp(trade.exit_timestamp)
            qty = float(trade.position_size)
            if qty <= 0:
                continue
            if trade.direction == "long":
                entry_side, exit_side = "buy", "sell"
            else:
                entry_side, exit_side = "sell", "buy"
            events.append(
                TradeEvent(
                    timestamp=entry_ts,
                    priority=0,
                    ticker=ticker,
                    side=entry_side,
                    quantity=qty,
                    price=float(trade.entry_price),
                    description="entry",
                )
            )
            events.append(
                TradeEvent(
                    timestamp=exit_ts,
                    priority=1,
                    ticker=ticker,
                    side=exit_side,
                    quantity=qty,
                    price=float(trade.exit_price),
                    description="exit",
                )
            )
    events.sort()
    return events


def _compute_equity(broker: PaperBroker, last_prices: Dict[str, float]) -> float:
    equity = broker.get_balance()
    for ticker, pos in broker.get_positions().items():
        price = last_prices.get(ticker, pos.get("avg_price", 0.0))
        equity += pos.get("quantity", 0.0) * price
    return float(equity)


def replay_trades(
    trades_by_ticker: Mapping[str, Iterable[TradeRecord]],
    broker: PaperBroker,
    logger: LiveLogger | None = None,
) -> pd.Series:
    events = _build_events(trades_by_ticker)
    if not events:
        return pd.Series(dtype=float)
    last_prices: Dict[str, float] = {}
    equity_records: List[tuple[pd.Timestamp, float]] = []
    for event in events:
        last_prices[event.ticker] = event.price
        record = broker.place_order(
            ticker=event.ticker,
            side=event.side, 
            quantity=event.quantity,
            price=event.price,
        )
        equity = _compute_equity(broker, last_prices)
        equity_records.append((event.timestamp, equity))
        if logger:
            logger.log_trade(
                timestamp=event.timestamp,
                ticker=event.ticker,
                side=event.side,
                quantity=event.quantity,
                price=event.price,
                realized_pnl=record.get("realized_pnl", 0.0),
                order_id=record.get("order_id", "paper"),
                cash_after=record.get("cash_after", broker.get_balance()),
            )
            logger.log_equity(
                timestamp=event.timestamp,
                equity=equity,
                cash=broker.get_balance(),
                positions=broker.get_positions(),
            )
    index = [ts for ts, _ in equity_records]
    values = [val for _, val in equity_records]
    return pd.Series(values, index=index).sort_index()


def run_paper_trading_session(
    ticker_results: Mapping[str, BacktestResult],
    broker: PaperBroker,
    logger: LiveLogger | None = None,
) -> pd.Series:
    trades_by_ticker = {ticker: result.trades for ticker, result in ticker_results.items()}
    return replay_trades(trades_by_ticker, broker, logger)
