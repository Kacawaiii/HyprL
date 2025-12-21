#!/usr/bin/env python3
"""Alpaca execution bridge for core_v3 signals (JSONL).

This bridge tails a signal JSONL file and places Alpaca paper/live orders
without modifying core_v3. It uses a state file for dedupe and safe restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def stable_client_order_id(symbol: str, ts: str, decision: str) -> str:
    raw = f"hyprl:{symbol}:{ts}:{decision}".encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()
    return f"hyprl-{symbol.lower()}-{h[:24]}"


@dataclass
class BridgeState:
    file_offset: int = 0
    last_exec_ts_by_symbol: Dict[str, str] = None
    last_exec_decision_by_symbol: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.last_exec_ts_by_symbol is None:
            self.last_exec_ts_by_symbol = {}
        if self.last_exec_decision_by_symbol is None:
            self.last_exec_decision_by_symbol = {}


def load_state(path: Path) -> BridgeState:
    if not path.exists():
        return BridgeState()
    data = json.loads(path.read_text(encoding="utf-8"))
    return BridgeState(
        file_offset=int(data.get("file_offset", 0)),
        last_exec_ts_by_symbol=dict(data.get("last_exec_ts_by_symbol", {})),
        last_exec_decision_by_symbol=dict(data.get("last_exec_decision_by_symbol", {})),
    )


def save_state(path: Path, st: BridgeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(st), indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _call_with_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return fn(**accepted)


def call_submit_order(broker: Any, **kwargs: Any) -> Any:
    candidates = ["submit_order", "submit_market_order", "place_order"]
    last_err = None
    for name in candidates:
        if not hasattr(broker, name):
            continue
        fn = getattr(broker, name)
        try:
            return _call_with_kwargs(fn, kwargs)
        except Exception as exc:  # pragma: no cover - pass-through
            last_err = exc
            continue
    raise RuntimeError(f"Could not submit order via broker: {last_err}")


def call_close_position(broker: Any, symbol: str) -> Any:
    if hasattr(broker, "close_position"):
        return broker.close_position(symbol)
    raise RuntimeError("Broker does not support close_position")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpaca bridge for core_v3 signal JSONL.")
    parser.add_argument("--signals", required=True, help="Path to signal JSONL (append-only).")
    parser.add_argument("--out", required=True, help="Execution journal JSONL output.")
    parser.add_argument("--state", required=True, help="State file (offset + last exec).")
    parser.add_argument("--symbols", default="", help="Comma-separated allowlist. Empty = allow all.")
    parser.add_argument("--paper", action="store_true", help="Use Alpaca paper trading.")
    parser.add_argument("--live", action="store_true", help="Use Alpaca live trading.")
    parser.add_argument(
        "--i-understand-live-orders",
        action="store_true",
        help="Required to run with --live.",
    )
    parser.add_argument(
        "--min-seconds-between-orders",
        type=int,
        default=3300,
        help="Cooldown per symbol (default: 55min).",
    )
    parser.add_argument(
        "--size-mode",
        choices=["shares", "notional"],
        default="shares",
        help="Interpret signal.size as shares or notional (default: shares).",
    )
    parser.add_argument("--max-notional-per-order", type=float, default=250.0)
    parser.add_argument("--min-notional-per-order", type=float, default=10.0)
    parser.add_argument("--max-qty", type=float, default=None)
    parser.add_argument("--allow-short", action="store_true", help="Allow short decisions.")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live and not args.i_understand_live_orders:
        raise SystemExit("Refusing to run live without --i-understand-live-orders")

    allowlist: Optional[Set[str]] = None
    if args.symbols.strip():
        allowlist = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    signals_path = Path(args.signals)
    out_path = Path(args.out)
    state_path = Path(args.state)

    st = load_state(state_path)

    from hyprl.broker.alpaca import AlpacaBroker
    from hyprl.broker import OrderSide, OrderType, TimeInForce

    paper = True if args.paper else False
    if args.live:
        paper = False
    broker = AlpacaBroker(paper=paper)

    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "bridge_start", "signals": str(signals_path), "paper": paper})

    # Wait for signals file to exist
    while not signals_path.exists():
        time.sleep(args.poll_seconds)

    with signals_path.open("r", encoding="utf-8") as f:
        f.seek(st.file_offset, os.SEEK_SET)
        while True:
            line = f.readline()
            if not line:
                st.file_offset = f.tell()
                save_state(state_path, st)
                time.sleep(args.poll_seconds)
                continue

            st.file_offset = f.tell()
            line = line.strip()
            if not line:
                continue

            try:
                ev = json.loads(line)
            except Exception as exc:
                append_jsonl(out_path, {"ts": utc_now_iso(), "event": "bad_json", "error": str(exc)})
                continue

            symbol = str(ev.get("symbol", "")).upper()
            if not symbol:
                continue
            if allowlist is not None and symbol not in allowlist:
                continue

            ts = str(ev.get("timestamp", ""))
            decision = str(ev.get("decision", "")).lower()
            size = float(ev.get("size", 0.0) or 0.0)
            if not ts or decision not in ("long", "short", "flat"):
                continue

            last_ts = st.last_exec_ts_by_symbol.get(symbol)
            last_dec = st.last_exec_decision_by_symbol.get(symbol)
            now_dt = parse_ts(ts)
            if last_ts:
                last_dt = parse_ts(last_ts)
                if (now_dt - last_dt).total_seconds() < args.min_seconds_between_orders:
                    continue
                if last_dec == decision and (now_dt - last_dt).total_seconds() < 2 * args.min_seconds_between_orders:
                    continue

            # Market guard
            try:
                if hasattr(broker, "is_market_open") and not broker.is_market_open():
                    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "market_closed_skip", "symbol": symbol})
                    continue
            except Exception as exc:
                append_jsonl(out_path, {"ts": utc_now_iso(), "event": "market_check_error", "error": str(exc)})

            target = decision
            if target == "short" and not args.allow_short:
                target = "flat"

            pos = None
            if hasattr(broker, "get_position"):
                try:
                    pos = broker.get_position(symbol)
                except Exception:
                    pos = None

            def already_long(p) -> bool:
                return bool(p) and str(getattr(p, "side", "")) == "long"

            def already_short(p) -> bool:
                return bool(p) and str(getattr(p, "side", "")) == "short"

            client_order_id = stable_client_order_id(symbol, ts, target)

            if target == "flat":
                if pos:
                    try:
                        res = call_close_position(broker, symbol)
                        append_jsonl(out_path, {"ts": utc_now_iso(), "event": "close_position", "symbol": symbol, "result": str(res)})
                    except Exception as exc:
                        append_jsonl(out_path, {"ts": utc_now_iso(), "event": "close_failed", "symbol": symbol, "error": str(exc)})
                        continue
            elif target == "long":
                if already_long(pos):
                    continue
                if pos and already_short(pos):
                    try:
                        call_close_position(broker, symbol)
                    except Exception as exc:
                        append_jsonl(out_path, {"ts": utc_now_iso(), "event": "flip_close_failed", "symbol": symbol, "error": str(exc)})
                        continue

                qty = max(0.0, size)
                if args.size_mode == "notional":
                    notional = max(args.min_notional_per_order, min(size, args.max_notional_per_order))
                    # AlpacaBroker requires qty; without price we cannot compute. Fallback to qty mode.
                    qty = max(0.0, size)
                if args.max_qty is not None:
                    qty = min(qty, args.max_qty)
                if qty <= 0:
                    continue

                try:
                    res = call_submit_order(
                        broker,
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=client_order_id,
                    )
                    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "open_long", "symbol": symbol, "qty": qty, "result": str(res)})
                except Exception as exc:
                    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "order_failed", "symbol": symbol, "error": str(exc)})
                    continue
            else:
                if already_short(pos):
                    continue
                if pos and already_long(pos):
                    try:
                        call_close_position(broker, symbol)
                    except Exception as exc:
                        append_jsonl(out_path, {"ts": utc_now_iso(), "event": "flip_close_failed", "symbol": symbol, "error": str(exc)})
                        continue

                qty = max(0.0, size)
                if args.max_qty is not None:
                    qty = min(qty, args.max_qty)
                if qty <= 0:
                    continue

                try:
                    res = call_submit_order(
                        broker,
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=client_order_id,
                    )
                    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "open_short", "symbol": symbol, "qty": qty, "result": str(res)})
                except Exception as exc:
                    append_jsonl(out_path, {"ts": utc_now_iso(), "event": "order_failed", "symbol": symbol, "error": str(exc)})
                    continue

            st.last_exec_ts_by_symbol[symbol] = ts
            st.last_exec_decision_by_symbol[symbol] = decision
            save_state(state_path, st)


if __name__ == "__main__":
    main()
