#!/usr/bin/env python3
"""
Quick Palier 1/2 status check using existing live/paper outputs (no recompute).

Uses:
- NVDA trade log for count, optional NVDA health JSON if present.
- Portfolio health JSON for PF/DD/Sharpe; optionally sums per-ticker trade logs for count.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LIVE_ROOT = Path("live/logs")
DEFAULT_P2_TICKERS = ("NVDA", "MSFT", "AMD", "META", "QQQ")

P1_DEFAULT_MIN_TRADES = 120
P1_DEFAULT_TARGET_TRADES = 200
P1_PF_MIN = 1.5
P1_MAXDD_MAX = 20.0
P1_SHARPE_MIN = 1.5

P2_DEFAULT_MIN_TRADES = 300
P2_DEFAULT_TARGET_TRADES = 400
P2_PF_MIN = 1.5
P2_MAXDD_MAX = 20.0
P2_SHARPE_MIN = 1.5


@dataclass
class GateConfig:
    min_trades: int
    target_trades: int
    pf_min: float
    maxdd_max: float
    sharpe_min: float


@dataclass
class PalierState:
    name: str
    status: str
    trades: Optional[int]
    pf: Optional[float]
    maxdd: Optional[float]
    sharpe: Optional[float]
    gates: GateConfig
    reasons: List[str]
    notes: List[str]
    sources: Dict[str, str]
    health_status: Optional[str]
    as_of: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Palier 1 (NVDA) and Palier 2 (Asc v2) status from live outputs.")
    parser.add_argument(
        "--live-root",
        type=Path,
        default=DEFAULT_LIVE_ROOT,
        help="Root directory containing live/logs (default: live/logs).",
    )
    parser.add_argument(
        "--nvda-trades",
        type=Path,
        help="Path to NVDA trade log CSV (default: <live-root>/live_nvda/trades_NVDA_live_all.csv).",
    )
    parser.add_argument(
        "--nvda-health",
        type=Path,
        help="Optional NVDA health JSON (default: <live-root>/live_nvda/health_NVDA_live.json).",
    )
    parser.add_argument(
        "--portfolio-health",
        type=Path,
        help="Portfolio health JSON (default: <live-root>/portfolio_live/health_asc_v2.json).",
    )
    parser.add_argument(
        "--portfolio-trades",
        type=str,
        help="Comma-separated per-ticker trade logs for Palier 2 (default: NVDA/MSFT/AMD/META/QQQ under <live-root>).",
    )
    parser.add_argument(
        "--p1-min-trades",
        type=int,
        default=P1_DEFAULT_MIN_TRADES,
        help=f"Minimum trades for Palier 1 gate (default: {P1_DEFAULT_MIN_TRADES}).",
    )
    parser.add_argument(
        "--p1-target-trades",
        type=int,
        default=P1_DEFAULT_TARGET_TRADES,
        help=f"Target trades for Palier 1 progression note (default: {P1_DEFAULT_TARGET_TRADES}).",
    )
    parser.add_argument(
        "--p2-min-trades",
        type=int,
        default=P2_DEFAULT_MIN_TRADES,
        help=f"Minimum trades for Palier 2 gate (default: {P2_DEFAULT_MIN_TRADES}).",
    )
    parser.add_argument(
        "--p2-target-trades",
        type=int,
        default=P2_DEFAULT_TARGET_TRADES,
        help=f"Target trades for Palier 2 progression note (default: {P2_DEFAULT_TARGET_TRADES}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print condensed JSON instead of text.",
    )
    return parser.parse_args()


def default_nvda_paths(live_root: Path) -> Tuple[Path, Path]:
    trades = live_root / "live_nvda" / "trades_NVDA_live_all.csv"
    health = live_root / "live_nvda" / "health_NVDA_live.json"
    return trades, health


def default_portfolio_health(live_root: Path) -> Path:
    return live_root / "portfolio_live" / "health_asc_v2.json"


def default_portfolio_trade_logs(live_root: Path) -> List[Path]:
    paths: List[Path] = []
    for ticker in DEFAULT_P2_TICKERS:
        sym_lower = ticker.lower()
        paths.append(live_root / f"live_{sym_lower}" / f"trades_{ticker}_live_all.csv")
    return paths


def parse_trade_logs_arg(arg: Optional[str]) -> List[Path]:
    if not arg:
        return []
    return [Path(item.strip()) for item in arg.split(",") if item.strip()]


def coerce_number(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def find_metric(source: Dict[str, Any], *candidates: str) -> Optional[float]:
    for key in candidates:
        for variant in (key, key.lower(), key.upper(), key.capitalize()):
            if variant in source:
                num = coerce_number(source[variant])
                if num is not None:
                    return num
    return None


def extract_metrics(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[str], Optional[str]]:
    metrics_src: Dict[str, Any]
    if isinstance(summary.get("metrics"), dict):
        metrics_src = summary["metrics"]
    else:
        metrics_src = summary
    pf = find_metric(metrics_src, "pf", "profit_factor")
    maxdd = find_metric(metrics_src, "maxdd", "max_dd_pct", "max_drawdown_pct", "maxdd_pct", "max_dd")
    sharpe = find_metric(metrics_src, "sharpe")
    trades = find_metric(metrics_src, "trades", "num_trades", "n_trades")
    trades_int = int(trades) if trades is not None else None
    as_of = (
        summary.get("as_of")
        or summary.get("timestamp")
        or summary.get("ts")
        or summary.get("ts_iso")
    )
    health_status = str(summary.get("status")).upper() if summary.get("status") else None
    return pf, maxdd, sharpe, trades_int, as_of, health_status


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_trades(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows:
            return 0
        # Assume first row is header
        return max(len(rows) - 1, 0)
    except Exception:
        return None


def evaluate_palier(
    name: str,
    trades: Optional[int],
    pf: Optional[float],
    maxdd: Optional[float],
    sharpe: Optional[float],
    gates: GateConfig,
    sources: Dict[str, str],
    health_status: Optional[str],
    as_of: Optional[str],
    missing_trade_reason: Optional[str],
) -> PalierState:
    status = "INCOMPLETE"
    reasons: List[str] = []
    notes: List[str] = []

    if missing_trade_reason:
        reasons.append(missing_trade_reason)

    if trades is None:
        reasons.append("missing trade count")
        return PalierState(name, status, trades, pf, maxdd, sharpe, gates, reasons, notes, sources, health_status, as_of)

    if trades < gates.min_trades:
        reasons.append(f"trades<{gates.min_trades}")
        return PalierState(name, status, trades, pf, maxdd, sharpe, gates, reasons, notes, sources, health_status, as_of)

    if pf is None or maxdd is None or sharpe is None:
        reasons.append("missing metrics (pf/maxdd/sharpe)")
        return PalierState(name, status, trades, pf, maxdd, sharpe, gates, reasons, notes, sources, health_status, as_of)

    status = "OK"
    if pf < gates.pf_min:
        status = "FAIL"
        reasons.append(f"pf<{gates.pf_min}")
    if maxdd > gates.maxdd_max:
        status = "FAIL"
        reasons.append(f"maxdd>{gates.maxdd_max}")
    if sharpe < gates.sharpe_min:
        status = "FAIL"
        reasons.append(f"sharpe<{gates.sharpe_min}")

    if status == "OK" and trades < gates.target_trades:
        notes.append(f"trades below target ({trades}/{gates.target_trades})")

    return PalierState(name, status, trades, pf, maxdd, sharpe, gates, reasons, notes, sources, health_status, as_of)


def state_to_dict(state: PalierState) -> Dict[str, Any]:
    return {
        "name": state.name,
        "status": state.status,
        "trades": state.trades,
        "pf": state.pf,
        "maxdd": state.maxdd,
        "sharpe": state.sharpe,
        "gates": {
            "min_trades": state.gates.min_trades,
            "target_trades": state.gates.target_trades,
            "pf_min": state.gates.pf_min,
            "maxdd_max": state.gates.maxdd_max,
            "sharpe_min": state.gates.sharpe_min,
        },
        "reasons": state.reasons,
        "notes": state.notes,
        "sources": state.sources,
        "health_status": state.health_status,
        "as_of": state.as_of,
    }


def fmt_float(val: Optional[float], digits: int = 3) -> str:
    return f"{val:.{digits}f}" if val is not None else "n/a"


def fmt_status_line(state: PalierState) -> List[str]:
    lines = [f"{state.name}: status={state.status}"]
    lines.append(
        f"  trades={state.trades if state.trades is not None else 'n/a'} "
        f"(min={state.gates.min_trades}, target={state.gates.target_trades})"
    )
    lines.append(
        "  metrics: "
        f"PF={fmt_float(state.pf)} | "
        f"MaxDD%={fmt_float(state.maxdd, 2)} | "
        f"Sharpe={fmt_float(state.sharpe)}"
        + (f" | health_status={state.health_status}" if state.health_status else "")
        + (f" | as_of={state.as_of}" if state.as_of else "")
    )
    lines.append(
        "  gates: "
        f"PF>={state.gates.pf_min} "
        f"DD<={state.gates.maxdd_max} "
        f"Sharpe>={state.gates.sharpe_min}"
    )
    if state.reasons:
        lines.append(f"  reasons: {', '.join(state.reasons)}")
    if state.notes:
        lines.append(f"  notes: {', '.join(state.notes)}")
    return lines


def main() -> None:
    args = parse_args()

    live_root = args.live_root
    nvda_trades_path, nvda_health_path = default_nvda_paths(live_root)
    if args.nvda_trades:
        nvda_trades_path = args.nvda_trades
    if args.nvda_health:
        nvda_health_path = args.nvda_health

    portfolio_health_path = args.portfolio_health or default_portfolio_health(live_root)
    portfolio_trade_paths = parse_trade_logs_arg(args.portfolio_trades) or default_portfolio_trade_logs(live_root)

    p1_gates = GateConfig(
        min_trades=args.p1_min_trades,
        target_trades=args.p1_target_trades,
        pf_min=P1_PF_MIN,
        maxdd_max=P1_MAXDD_MAX,
        sharpe_min=P1_SHARPE_MIN,
    )
    p2_gates = GateConfig(
        min_trades=args.p2_min_trades,
        target_trades=args.p2_target_trades,
        pf_min=P2_PF_MIN,
        maxdd_max=P2_MAXDD_MAX,
        sharpe_min=P2_SHARPE_MIN,
    )

    nvda_trades_count = count_trades(nvda_trades_path)
    nvda_missing_reason = None if nvda_trades_count is not None else f"missing trades file ({nvda_trades_path})"

    nvda_pf = nvda_maxdd = nvda_sharpe = None
    nvda_as_of = nvda_health_status = None
    if nvda_health_path and nvda_health_path.is_file():
        nvda_summary = load_json(nvda_health_path)
        if nvda_summary:
            nvda_pf, nvda_maxdd, nvda_sharpe, nvda_health_trades, nvda_as_of, nvda_health_status = extract_metrics(nvda_summary)
            if nvda_health_trades is not None:
                nvda_trades_count = nvda_health_trades
    p1_state = evaluate_palier(
        name="Palier 1 (NVDA)",
        trades=nvda_trades_count,
        pf=nvda_pf,
        maxdd=nvda_maxdd,
        sharpe=nvda_sharpe,
        gates=p1_gates,
        sources={"trades": str(nvda_trades_path), "health": str(nvda_health_path)},
        health_status=nvda_health_status,
        as_of=nvda_as_of,
        missing_trade_reason=nvda_missing_reason,
    )

    port_pf = port_maxdd = port_sharpe = None
    port_trades = None
    port_as_of = port_health_status = None
    portfolio_missing_reason = None

    if portfolio_health_path.is_file():
        portfolio_summary = load_json(portfolio_health_path)
        if portfolio_summary:
            port_pf, port_maxdd, port_sharpe, port_trades, port_as_of, port_health_status = extract_metrics(portfolio_summary)
    else:
        portfolio_missing_reason = f"missing portfolio health JSON ({portfolio_health_path})"

    if port_trades is None:
        trade_counts: List[int] = []
        missing: List[str] = []
        for path in portfolio_trade_paths:
            count = count_trades(path)
            if count is None:
                missing.append(str(path))
            else:
                trade_counts.append(count)
        if trade_counts:
            port_trades = sum(trade_counts)
        if missing and not portfolio_missing_reason:
            portfolio_missing_reason = f"incomplete trade logs: {', '.join(missing)}"

    p2_state = evaluate_palier(
        name="Palier 2 (Asc v2 portfolio)",
        trades=port_trades,
        pf=port_pf,
        maxdd=port_maxdd,
        sharpe=port_sharpe,
        gates=p2_gates,
        sources={
            "health": str(portfolio_health_path),
            "trade_logs": ", ".join(str(p) for p in portfolio_trade_paths),
        },
        health_status=port_health_status,
        as_of=port_as_of,
        missing_trade_reason=portfolio_missing_reason,
    )

    if args.json:
        payload = {
            "palier1": state_to_dict(p1_state),
            "palier2": state_to_dict(p2_state),
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        return

    for line in fmt_status_line(p1_state):
        print(line)
    print("")
    for line in fmt_status_line(p2_state):
        print(line)


if __name__ == "__main__":
    main()
