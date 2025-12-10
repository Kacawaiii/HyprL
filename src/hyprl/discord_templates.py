"""Discord embed templates for live trade and health alerts."""

from __future__ import annotations

from typing import Any, Mapping


def _fmt_number(val: Any, *, pct: bool = False) -> str:
    if val is None:
        return "n/a"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "n/a"
    if pct:
        return f"{num:.2f}%"
    return f"{num:.2f}"


def build_trade_embed(
    *,
    session_name: str,
    plan: str | None,
    trade: Mapping[str, object],
    health: Mapping[str, object] | None,
) -> dict:
    """Return a Discord embed payload for a trade + optional health context."""
    symbol = str(trade.get("symbol", "???")).upper()
    side = str(trade.get("side", "N/A")).upper()
    event_type = str(trade.get("event_type", "EVENT")).upper()
    exit_reason = trade.get("exit_reason") or "n/a"
    pnl = trade.get("pnl")
    pnl_pct = trade.get("pnl_pct")
    entry_price = trade.get("entry_price")
    exit_price = trade.get("exit_price")
    entry_time = trade.get("entry_time")
    exit_time = trade.get("exit_time")
    prediction_id = trade.get("prediction_id")

    color = 0x00FF00 if side == "LONG" else 0xFF0000 if side == "SHORT" else 0x808080
    title = f"ALERT {symbol} {side}"

    signal_lines = [
        f"**Type d'event :** {event_type}",
        f"**Direction :** {side}",
        f"**Raison :** {exit_reason}",
    ]
    if prediction_id:
        signal_lines.append(f"**Prediction ID :** {prediction_id}")

    trade_lines = []
    if entry_price is not None or exit_price is not None:
        trade_lines.append(f"**Entry/Exit :** {entry_price or 'n/a'} → {exit_price or 'n/a'}")
    if entry_time or exit_time:
        trade_lines.append(f"**Time :** {entry_time or 'n/a'} → {exit_time or 'n/a'}")
    if pnl is not None or pnl_pct is not None:
        pnl_str = _fmt_number(pnl)
        pnl_pct_str = _fmt_number(pnl_pct, pct=True) if pnl_pct is not None else None
        if pnl_pct_str and pnl_pct_str != "n/a":
            trade_lines.append(f"**PnL :** {pnl_str} ({pnl_pct_str})")
        else:
            trade_lines.append(f"**PnL :** {pnl_str}")

    health_lines = []
    if health:
        pf = health.get("pf") or health.get("profit_factor")
        maxdd = health.get("maxdd") or health.get("max_dd") or health.get("max_drawdown_pct")
        sharpe = health.get("sharpe")
        trades = health.get("trades")
        status = health.get("status")
        health_lines.append(
            f"**Portfolio :** PF={_fmt_number(pf)}, DD={_fmt_number(maxdd, pct=True)}, "
            f"Sharpe={_fmt_number(sharpe)}, Trades={trades if trades is not None else 'n/a'}"
        )
        health_lines.append(f"**Status :** {status or 'n/a'}")
    else:
        health_lines.append("_Health indisponible_")

    footer_parts = [session_name]
    if plan:
        footer_parts.append(f"Plan: {plan}")
    footer_parts.append("Research-only, not financial advice")
    footer_text = " • ".join(footer_parts)

    return {
        "embeds": [
            {
                "title": title,
                "description": "Signal temps réel HyprL v2",
                "color": color,
                "fields": [
                    {"name": "Signal", "value": "\n".join(signal_lines), "inline": False},
                    {
                        "name": "Trade",
                        "value": "\n".join(trade_lines) if trade_lines else "_N/A_",
                        "inline": False,
                    },
                    {"name": "Health", "value": "\n".join(health_lines), "inline": False},
                ],
                "footer": {"text": footer_text},
            }
        ]
    }


__all__ = ["build_trade_embed"]
