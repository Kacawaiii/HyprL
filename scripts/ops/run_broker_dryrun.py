#!/usr/bin/env python3
"""Dry-run broker orchestrator using HyprL API signals.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
    python scripts/ops/run_broker_dryrun.py \
      --tickers NVDA,MSFT,AMD,META,QQQ \
      --api-base http://127.0.0.1:8000 \
      --state-file live/logs/broker_state_dryrun.json \
      --audit-file live/logs/audit_trades_dryrun.jsonl

Notes:
- State is persisted in --state-file; audit decisions append to --audit-file (JSONL).
- No real orders are placed; relies on /v2/signal outputs that already include risk sizing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyprl.broker import OrderSide, OrderType, TimeInForce  # noqa: E402
from hyprl.broker.dryrun import DryRunBroker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate broker orders from HyprL API signals (dry-run).")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. NVDA,MSFT,AMD")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="HyprL API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("live/logs/broker_state_dryrun.json"),
        help="Path to broker state JSON (default: live/logs/broker_state_dryrun.json)",
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=Path("live/logs/audit_trades_dryrun.jsonl"),
        help="Append-only JSONL audit file (default: live/logs/audit_trades_dryrun.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not persist broker state (still logs audit).",
    )
    return parser.parse_args()


def _http_get_json(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"[WARN] HTTP {exc.code} for {url}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
    except URLError as exc:
        print(f"[WARN] URL error for {url}: {exc}", file=sys.stderr)
    except json.JSONDecodeError as exc:
        print(f"[WARN] JSON decode error for {url}: {exc}", file=sys.stderr)
    return None


def fetch_signal(api_base: str, ticker: str) -> dict[str, Any] | None:
    return _http_get_json(f"{api_base}/v2/signal?ticker={ticker}")


def fetch_portfolio(api_base: str) -> dict[str, Any] | None:
    return _http_get_json(f"{api_base}/v2/portfolio")


def signal_id(signal: dict[str, Any]) -> str:
    for key in ("exit_timestamp", "entry_timestamp"):
        if signal.get(key):
            return str(signal[key])
    return signal.get("direction") or str(hash(json.dumps(signal, sort_keys=True)))


def write_audit_line(path: Path, obj: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
    except OSError as exc:
        print(f"[WARN] Failed to write audit line to {path}: {exc}", file=sys.stderr)


def build_order_payload(signal: dict[str, Any]) -> dict[str, Any]:
    direction = (signal.get("direction") or "").upper()
    if direction == "LONG":
        side = OrderSide.BUY
    elif direction == "SHORT":
        side = OrderSide.SELL
    else:
        side = OrderSide.BUY
    qty = float(signal.get("position_size") or signal.get("qty") or 0.0)
    price = signal.get("entry_price") or signal.get("exit_price")
    return {
        "side": side,
        "qty": qty,
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.DAY,
        "limit_price": float(price) if price is not None else None,
    }


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    broker = DryRunBroker(args.state_file, persist=not args.dry_run)
    portfolio_snapshot = fetch_portfolio(args.api_base)

    for ticker in tickers:
        signal = fetch_signal(args.api_base, ticker)
        if not signal:
            audit = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": ticker,
                "action": "skip",
                "reason": "not_found",
                "signal": None,
                "portfolio": portfolio_snapshot,
                "order": None,
                "broker_result": None,
            }
            write_audit_line(args.audit_file, audit)
            continue

        sid = signal_id(signal)
        last_sid = broker.last_signals.get(ticker)
        if last_sid == sid:
            audit = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": ticker,
                "action": "skip",
                "reason": "no_change",
                "signal": signal,
                "portfolio": portfolio_snapshot,
                "order": None,
                "broker_result": None,
            }
            write_audit_line(args.audit_file, audit)
            continue

        order_payload = build_order_payload(signal)
        action_reason = "new_signal"
        broker_result = None

        if order_payload["qty"] <= 0:
            action = "skip"
            action_reason = "no_qty"
        else:
            action = "submit"
            broker_result = broker.submit_order(
                symbol=ticker,
                side=order_payload["side"],
                qty=order_payload["qty"],
                order_type=order_payload["order_type"],
                time_in_force=order_payload["time_in_force"],
                limit_price=order_payload["limit_price"],
            )
            broker.set_last_signal(ticker, sid)

        audit = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": ticker,
            "action": action,
            "reason": action_reason,
            "signal": signal,
            "portfolio": portfolio_snapshot,
            "order": order_payload if action == "submit" else None,
            "broker_result": asdict(broker_result) if broker_result else None,
        }
        write_audit_line(args.audit_file, audit)
        time.sleep(0.05)  # small pacing for API friendliness


if __name__ == "__main__":
    main()
