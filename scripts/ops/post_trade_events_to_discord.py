#!/usr/bin/env python3
"""Post new live trades to Discord using session channel mappings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Ensure repository root is on sys.path for bot.session_store imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.session_store import (  # noqa: E402
    get_session_channels,
    get_sessions,
    resolve_session,
)
from hyprl.discord_dlq import DiscordDLQ  # noqa: E402
from hyprl.discord_idempotence import (  # noqa: E402
    DiscordIdempotenceStore,
    DiscordMessageRecord,
)
from hyprl.discord_poster_common import (  # noqa: E402
    IDEMP_STATUS_FAILED_PERM,
    IDEMP_STATUS_FAILED_TEMP,
    IDEMP_STATUS_SENT,
    TRANSIENT_STATUS_CODES,
    post_discord_message,
)
from hyprl.discord_templates import build_trade_embed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post live trade events to Discord channels.")
    parser.add_argument("--session", required=True, help="Session slug (e.g., client-a-nvda).")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument(
        "--tickers",
        default="NVDA,MSFT,AMD,META,QQQ",
        help="Comma-separated tickers to process (default: NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Optional state file path (default: live/logs/discord_trade_state_<session>.json).",
    )
    parser.add_argument(
        "--health",
        type=Path,
        help="Optional health JSON path (default: <log-root>/portfolio_live/health_asc_v2.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log payloads without sending to Discord.",
    )
    return parser.parse_args()


def load_state(path: Path) -> dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        print(f"[WARN] Failed to read state {path}: {exc}", file=sys.stderr)
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        print(f"[WARN] Invalid JSON state {path}: {exc}", file=sys.stderr)
        return {}


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=path.name,
        suffix=".tmp",
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)


def load_sessions(path: Path) -> dict[str, dict[str, Any]]:
    sessions = get_sessions(path)
    if not isinstance(sessions, dict):
        return {}
    return sessions


def resolve_session_slug(slug: str, sessions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return resolve_session(slug, sessions)


def get_trades_csv_path(log_root: Path, ticker: str) -> Path:
    return log_root / f"live_{ticker.lower()}" / f"trades_{ticker.upper()}_live_all.csv"


def load_trades(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
    except FileNotFoundError:
        return []
    except OSError as exc:
        print(f"[WARN] Failed reading trades {path}: {exc}", file=sys.stderr)
        return []
    return rows


def _trade_identifier(row: dict[str, str], fallback_index: int) -> str:
    for key in ("trade_id", "id"):
        value = row.get(key)
        if value:
            return str(value)
    for key in ("exit_timestamp", "entry_timestamp"):
        value = row.get(key)
        if value:
            return str(value)
    return str(fallback_index)


def _sort_trades(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows
    if "exit_timestamp" in rows[0]:
        sorted_rows = sorted(
            rows,
            key=lambda r: r.get("exit_timestamp") or "",
        )
        return sorted_rows
    return rows


def _build_message_key(session: str, fallback_ticker: str, trade: dict[str, str]) -> str:
    ticker = (trade.get("ticker") or trade.get("symbol") or fallback_ticker).upper()
    bar_ts = (
        trade.get("exit_timestamp")
        or trade.get("entry_timestamp")
        or trade.get("bar_time")
        or trade.get("bar_ts")
        or ""
    )
    trade_id = _trade_identifier(trade, 0)
    return f"{session}:{ticker}:{bar_ts}:{trade_id}"


def find_new_trades(rows: list[dict[str, str]], last_id: str | None) -> tuple[list[dict[str, str]], str | None]:
    if not rows:
        return [], last_id
    ordered = _sort_trades(rows)
    trade_ids = [_trade_identifier(row, idx) for idx, row in enumerate(ordered)]

    if last_id is None:
        # First run: only send the latest trade to avoid spam.
        return [ordered[-1]], trade_ids[-1]

    try:
        last_idx = trade_ids.index(last_id)
    except ValueError:
        print(f"[WARN] last_id {last_id} not found; defaulting to last trade only.", file=sys.stderr)
        return [ordered[-1]], trade_ids[-1]

    new_rows = ordered[last_idx + 1 :]
    if not new_rows:
        return [], last_id
    return new_rows, trade_ids[-1]


def load_health(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"[WARN] Failed reading health {path}: {exc}", file=sys.stderr)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Invalid health JSON {path}: {exc}", file=sys.stderr)
        return None
    return data


def _extract_first(summary: dict[str, Any], candidates: Iterable[str]) -> Any:
    lowered = {k.lower(): v for k, v in summary.items()}
    for key in candidates:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def extract_health_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"pf": None, "maxdd": None, "sharpe": None, "trades": None, "status": None}
    metrics = summary.get("metrics") or summary
    pf = _extract_first(metrics, ("pf", "profit_factor"))
    maxdd = _extract_first(metrics, ("maxdd", "max_dd_pct", "max_drawdown_pct", "dd"))
    sharpe = _extract_first(metrics, ("sharpe",))
    trades = _extract_first(metrics, ("trades", "n_trades", "num_trades"))
    status = _extract_first(summary, ("status",))
    return {"pf": pf, "maxdd": maxdd, "sharpe": sharpe, "trades": trades, "status": status}


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def normalize_trade_event(trade_row: dict[str, str], fallback_ticker: str) -> dict[str, Any]:
    ticker = (trade_row.get("ticker") or trade_row.get("symbol") or fallback_ticker or "???").upper()
    side = (trade_row.get("side") or trade_row.get("direction") or "N/A").upper()
    event_type = (trade_row.get("event_type") or "EXIT").upper()
    pnl = _parse_float(trade_row.get("pnl") or trade_row.get("pnl_abs"))
    pnl_pct = _parse_float(trade_row.get("pnl_pct") or trade_row.get("pnl_percent"))
    entry_price = _parse_float(trade_row.get("entry_price"))
    exit_price = _parse_float(trade_row.get("exit_price"))
    entry_ts = trade_row.get("entry_timestamp") or trade_row.get("entry_time")
    exit_ts = trade_row.get("exit_timestamp") or trade_row.get("exit_time")
    exit_reason = trade_row.get("exit_reason") or trade_row.get("reason") or "n/a"
    prediction_id = trade_row.get("prediction_id")

    return {
        "event_type": event_type,
        "symbol": ticker,
        "side": side,
        "entry_time": entry_ts,
        "exit_time": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "prediction_id": prediction_id,
    }


def main() -> None:
    args = parse_args()
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("[ERROR] DISCORD_BOT_TOKEN is required", file=sys.stderr)
        sys.exit(1)

    session_slug = args.session
    log_root: Path = args.log_root
    state_path = args.state_file or (log_root / f"discord_trade_state_{session_slug}.json")
    sessions_file = Path(os.getenv("DISCORD_SESSIONS_FILE") or (log_root / "discord_sessions.json"))
    health_path = args.health or (log_root / "portfolio_live" / "health_asc_v2.json")
    idempotence_store = DiscordIdempotenceStore(log_root / "discord_poster.db")
    dlq = DiscordDLQ(log_root / "discord_dlq.jsonl")

    sessions = load_sessions(sessions_file)
    session = resolve_session_slug(session_slug, sessions)
    if not session:
        print(f"[ERROR] Session '{session_slug}' not found in {sessions_file}", file=sys.stderr)
        sys.exit(1)
    channels = get_session_channels(session)
    trades_channel = channels.get("trades")
    if trades_channel is None:
        print(f"[ERROR] Session '{session_slug}' has no 'trades' channel configured", file=sys.stderr)
        sys.exit(1)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    state = load_state(state_path)
    health_summary = load_health(health_path)
    health_ctx = extract_health_metrics(health_summary)

    state_updated = False
    session_name = session.get("name", session_slug)
    plan = session.get("plan")
    for ticker in tickers:
        csv_path = get_trades_csv_path(log_root, ticker)
        trades = load_trades(csv_path)
        if not trades:
            continue
        last_id = state.get(ticker)
        new_trades, new_last_id = find_new_trades(trades, last_id)
        if not new_trades:
            continue
        for trade in new_trades:
            message_key = _build_message_key(session_slug, ticker, trade)
            record: DiscordMessageRecord | None = idempotence_store.get(session_slug, message_key)
            if record and record.status == IDEMP_STATUS_SENT:
                continue
            trade_event = normalize_trade_event(trade, ticker)
            payload = build_trade_embed(
                session_name=session_name,
                plan=plan,
                trade=trade_event,
                health=health_ctx,
            )
            trade_id = _trade_identifier(trade, 0)
            if args.dry_run:
                print(f"[DRY] ticker={ticker} trade_id={trade_id} payload={json.dumps(payload)}")
            else:
                status_code, message_id, error_body = post_discord_message(
                    token, int(trades_channel), payload, dry_run=False
                )
                if status_code is None:
                    idempotence_store.upsert(session_slug, message_key, IDEMP_STATUS_FAILED_TEMP)
                    dlq.push(
                        session=session_slug,
                        message_key=message_key,
                        reason="network_error",
                        payload={"payload": payload},
                    )
                    continue
                if 200 <= status_code < 300:
                    idempotence_store.upsert(
                        session_slug,
                        message_key,
                        IDEMP_STATUS_SENT,
                        discord_message_id=message_id,
                    )
                    continue
                if status_code in TRANSIENT_STATUS_CODES:
                    idempotence_store.upsert(session_slug, message_key, IDEMP_STATUS_FAILED_TEMP)
                    dlq.push(
                        session=session_slug,
                        message_key=message_key,
                        reason=f"discord_{status_code}",
                        payload={"payload": payload, "body": error_body},
                    )
                    continue
                idempotence_store.upsert(session_slug, message_key, IDEMP_STATUS_FAILED_PERM)
                dlq.push(
                    session=session_slug,
                    message_key=message_key,
                    reason=f"discord_{status_code}_perm",
                    payload={"payload": payload, "body": error_body},
                )
        state[ticker] = new_last_id or state.get(ticker, "")
        state_updated = True

    if state_updated:
        save_state(state_path, state)


if __name__ == "__main__":
    main()
