#!/usr/bin/env python3
"""
Alert on portfolio health JSON (PF / MaxDD / Sharpe / trades) with webhook.

Example:
    python scripts/ops/alert_portfolio_health.py \
      --health live/logs/portfolio_live/health_asc_v2.json \
      --pf-alert 1.3 --dd-alert 20 --sharpe-alert 1.5 \
      --webhook $SLACK_WEBHOOK_URL
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send webhook alert based on portfolio health JSON (PF/MaxDD/Sharpe/trades)."
    )
    parser.add_argument(
        "--health",
        type=Path,
        default=Path("live/logs/portfolio_live/health_asc_v2.json"),
        help="Path to portfolio health summary JSON (default: live/logs/portfolio_live/health_asc_v2.json).",
    )
    parser.add_argument(
        "--pf-alert",
        type=float,
        default=1.3,
        help="Alert if PF < this value (<=0 to disable PF threshold). Default: 1.3.",
    )
    parser.add_argument(
        "--dd-alert",
        type=float,
        default=20.0,
        help="Alert if MaxDD%% > this value (<=0 to disable DD threshold). Default: 20.",
    )
    parser.add_argument(
        "--sharpe-alert",
        type=float,
        default=1.5,
        help="Alert if Sharpe < this value (<=0 to disable Sharpe threshold). Default: 1.5.",
    )
    parser.add_argument(
        "--webhook",
        help="Webhook URL (Slack/Discord/other JSON endpoint). If omitted, only prints to stdout/stderr.",
    )
    parser.add_argument(
        "--send-ok",
        action="store_true",
        help="Also send a webhook notification when status is OK (heartbeat mode).",
    )
    return parser.parse_args()


def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    """POST payload as JSON to webhook. Best-effort (logs warning on failure)."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if "text" in payload and "content" not in payload:
        payload["content"] = payload["text"]
    req = urllib.request.Request(webhook_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except urllib.error.URLError as exc:
        print(f"[WARN] Failed to send webhook: {exc}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Unexpected error sending webhook: {exc}", file=sys.stderr)


def load_summary(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def find_metric(source: Dict[str, Any], *candidates: str) -> Optional[float]:
    """Try several key names, case-insensitive; return first numeric value."""
    for key in candidates:
        if key in source and isinstance(source[key], (int, float)):
            return float(source[key])
        for variant in (key.lower(), key.upper(), key.capitalize()):
            if variant in source and isinstance(source[variant], (int, float)):
                return float(source[variant])
    return None


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics_src: Dict[str, Any]
    if isinstance(summary.get("metrics"), dict):
        metrics_src = summary["metrics"]
    else:
        metrics_src = summary
    pf = find_metric(metrics_src, "pf", "profit_factor")
    max_dd_pct = find_metric(metrics_src, "max_dd_pct", "max_drawdown_pct", "max_dd")
    sharpe = find_metric(metrics_src, "sharpe")
    trades = find_metric(metrics_src, "trades", "num_trades", "n_trades")
    as_of = (
        summary.get("as_of")
        or summary.get("timestamp")
        or summary.get("ts")
        or summary.get("ts_iso")
    )
    return {
        "pf": pf,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "trades": trades,
        "as_of": as_of,
    }


def main() -> None:
    args = parse_args()
    pf_threshold = args.pf_alert if args.pf_alert > 0 else None
    dd_threshold = args.dd_alert if args.dd_alert > 0 else None
    sharpe_threshold = args.sharpe_alert if args.sharpe_alert > 0 else None

    if not args.health.is_file():
        msg = f"[ALERT] Health file missing: {args.health}"
        print(msg, file=sys.stderr)
        payload = {
            "text": msg,
            "kind": "portfolio_health",
            "severity": "ERROR",
            "path": str(args.health),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.webhook:
            send_webhook(args.webhook, payload)
        sys.exit(1)

    try:
        summary = load_summary(args.health)
    except Exception as exc:
        msg = f"[ALERT] Failed to load/parse health JSON at {args.health}: {exc}"
        print(msg, file=sys.stderr)
        payload = {
            "text": msg,
            "kind": "portfolio_health",
            "severity": "ERROR",
            "path": str(args.health),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.webhook:
            send_webhook(args.webhook, payload)
        sys.exit(1)

    status_raw = summary.get("status", "UNKNOWN")
    status = str(status_raw).upper() if status_raw is not None else "UNKNOWN"
    metrics = extract_metrics(summary)
    pf = metrics["pf"]
    max_dd_pct = metrics["max_dd_pct"]
    sharpe = metrics["sharpe"]
    trades = metrics["trades"]
    as_of = metrics["as_of"] or datetime.now(timezone.utc).isoformat()

    reasons: list[str] = []
    manual_alert = False
    if pf_threshold is not None and pf is not None and pf < pf_threshold:
        manual_alert = True
        reasons.append(f"PF<{pf_threshold} (PF={pf:.3f})")
    if dd_threshold is not None and max_dd_pct is not None and max_dd_pct > dd_threshold:
        manual_alert = True
        reasons.append(f"DD>{dd_threshold}% (DD={max_dd_pct:.2f}%)")
    if sharpe_threshold is not None and sharpe is not None and sharpe < sharpe_threshold:
        manual_alert = True
        reasons.append(f"Sharpe<{sharpe_threshold} (Sharpe={sharpe:.3f})")

    status_alert = status in {"ALERT", "AMBER", "ERROR"}
    alert = manual_alert or status_alert

    base_msg = (
        f"[Asc v2] status={status} PF={pf} DD%={max_dd_pct} "
        f"Sharpe={sharpe} Trades={trades} as_of={as_of}"
    )

    if alert:
        reason_str = "; ".join(reasons) if reasons else f"status={status}"
        text = f"[ALERT] {base_msg} | reasons: {reason_str}"
        print(text, file=sys.stderr)
        payload = {
            "text": text,
            "kind": "portfolio_health",
            "severity": "ALERT",
            "status": status,
            "pf": pf,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "trades": trades,
            "as_of": as_of,
            "reasons": reasons,
            "raw": summary,
        }
        if args.webhook:
            send_webhook(args.webhook, payload)
        sys.exit(1)
    else:
        text = f"[OK] {base_msg}"
        print(text)
        if args.webhook and args.send_ok:
            payload = {
                "text": text,
                "kind": "portfolio_health",
                "severity": "OK",
                "status": status,
                "pf": pf,
                "max_dd_pct": max_dd_pct,
                "sharpe": sharpe,
                "trades": trades,
                "as_of": as_of,
                "raw": summary,
            }
            send_webhook(args.webhook, payload)
        sys.exit(0)


if __name__ == "__main__":
    main()
