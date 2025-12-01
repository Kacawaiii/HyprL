#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from hyprl.backtest.runner import BacktestConfig
from hyprl.live.market_data import YFinanceSource
from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.risk import LiveRiskConfig, LiveRiskManager
from hyprl.live.strategy_engine import StrategyEngine
from hyprl.live.types import Bar


def _default_trade_path(symbol: str, interval: str) -> Path:
    sanitized_interval = interval.replace("/", "-")
    return Path("data") / "live" / f"trades_{symbol.upper()}_{sanitized_interval}.csv"


def _live_execution_enabled() -> bool:
    raw = os.getenv("LIVE_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _wait_until_next_bar(interval: str) -> None:
    """MVP sleep strategy; replace with proper scheduling when wiring real feeds."""
    if interval.endswith("h"):
        time.sleep(60)
    else:
        time.sleep(10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HyprL strategy in paper live mode.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--lookback", default="180d")
    parser.add_argument("--trades-csv", type=Path, help="Optional path for live trade logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    interval = args.interval

    data_source = YFinanceSource()
    trade_path = args.trades_csv or _default_trade_path(symbol, interval)
    bt_config = BacktestConfig(
        ticker=symbol,
        interval=interval,
        period=args.lookback,
        initial_balance=args.initial_balance,
    )
    broker = PaperBrokerImpl(
        cash=args.initial_balance,
        commission_pct=bt_config.commission_pct,
        slippage_pct=bt_config.slippage_pct,
        trade_log_path=trade_path,
    )
    engine = StrategyEngine(config=bt_config)
    risk_manager = LiveRiskManager(
        LiveRiskConfig(
            max_daily_loss_pct=0.03,
            max_position_notional_pct=0.2,
            max_gross_exposure_pct=1.0,
        ),
        broker=broker,
    )

    history = data_source.get_history(symbol, interval, args.lookback)
    engine.warmup(history)

    live_flag = "ON" if _live_execution_enabled() else "OFF"
    print(f"[{datetime.utcnow().isoformat()}] Live paper runner started for {symbol}@{interval}")
    print(f"[INFO] Live trades will be appended to {trade_path}")
    print(f"[INFO] LIVE_ENABLED={live_flag} (toggle via environment variable)")

    while True:
        try:
            bar: Bar = data_source.get_latest_bar(symbol, interval)
            broker.mark_to_market(bar)
            open_positions = broker.get_positions()
            equity = broker.get_balance()
            signals = engine.on_bar(bar, open_positions, equity=equity)

            for signal in signals:
                if not _live_execution_enabled():
                    print(f"[{bar.timestamp}] LIVE_ENABLED=0 → skip execution for {signal.reason}")
                    continue
                price = bar.close
                decision = risk_manager.allow_trade(signal, price)
                if not decision.allowed:
                    print(
                        f"[{bar.timestamp}] Trade blocked by risk manager: {signal.reason}"
                        + (f" (reason={decision.reason})" if decision.reason else "")
                    )
                    continue
                broker.submit_signal(signal, bar)
                print(f"[{bar.timestamp}] EXECUTED: {signal}")

            print(
                f"[{bar.timestamp}] Equity={broker.get_balance():.2f} | Positions={len(open_positions)}"
            )
        except Exception as exc:  # pragma: no cover - live loop guard
            print(f"[ERROR] {exc}")

        _wait_until_next_bar(interval)


if __name__ == "__main__":
    main()
