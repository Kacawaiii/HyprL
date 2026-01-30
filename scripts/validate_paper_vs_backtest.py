#!/usr/bin/env python3
"""
Paper Trading vs Backtest Validation
=====================================

Compares live paper trading session metrics against backtest expectations
from track_record_latest.json. Flags significant deviations early.

Usage:
    python scripts/validate_paper_vs_backtest.py
    python scripts/validate_paper_vs_backtest.py --session sess_20260130_...
    python scripts/validate_paper_vs_backtest.py --session latest --min-trades 10
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SESSIONS_DIR = Path("data/live/sessions")
TRACK_RECORD_PATH = Path("docs/reports/track_record/track_record_latest.json")

# Tolerance thresholds (relative to backtest baseline)
WIN_RATE_WARN = 0.08       # 8% below backtest → warning
WIN_RATE_CRIT = 0.15       # 15% below → critical
PF_WARN_RATIO = 0.70       # PF < 70% of backtest → warning
PF_CRIT_RATIO = 0.50       # PF < 50% of backtest → critical
DD_WARN_RATIO = 1.30       # DD > 130% of backtest → warning
DD_CRIT_RATIO = 1.80       # DD > 180% of backtest → critical
SHARPE_WARN_RATIO = 0.60   # Sharpe < 60% of backtest → warning
SHARPE_CRIT_RATIO = 0.35   # Sharpe < 35% of backtest → critical


@dataclass
class ValidationResult:
    metric: str
    paper_value: float
    backtest_value: float
    status: str  # "pass", "warning", "critical"
    message: str


def load_track_record() -> dict:
    if not TRACK_RECORD_PATH.exists():
        print(f"ERROR: Track record not found: {TRACK_RECORD_PATH}")
        sys.exit(1)
    return json.loads(TRACK_RECORD_PATH.read_text())


def find_latest_session() -> Path | None:
    if not SESSIONS_DIR.exists():
        return None
    sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    for s in sessions:
        manifest = s / "session_manifest.json"
        if manifest.exists():
            return s
    return None


def load_session_fills(session_dir: Path) -> list[dict]:
    fills_path = session_dir / "fills.jsonl"
    if not fills_path.exists():
        return []
    fills = []
    with open(fills_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    fills.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return fills


def load_session_equity(session_dir: Path) -> list[dict]:
    eq_path = session_dir / "equity.jsonl"
    if not eq_path.exists():
        return []
    records = []
    with open(eq_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_paper_metrics(
    session_dir: Path,
    manifest: dict,
) -> dict:
    fills = load_session_fills(session_dir)
    equity_records = load_session_equity(session_dir)

    initial_equity = manifest.get("initial_balance", 10000)
    equity_peak_manifest = manifest.get("equity_peak", initial_equity)

    # Win rate from fills with PnL
    pnl_fills = [f for f in fills if f.get("pnl") is not None]
    wins = sum(1 for f in pnl_fills if f["pnl"] > 0)
    n_trades = len(pnl_fills)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = sum(f["pnl"] for f in pnl_fills if f["pnl"] > 0)
    gross_loss = abs(sum(f["pnl"] for f in pnl_fills if f["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve from equity records
    equities = [r.get("equity", initial_equity) for r in equity_records if r.get("equity") is not None]
    if not equities:
        equities = [initial_equity]

    final_equity = equities[-1]
    return_pct = (final_equity / initial_equity - 1) * 100

    # Max drawdown
    peak = initial_equity
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Sharpe (annualized from equity changes)
    if len(equities) > 2:
        returns = np.diff(equities) / np.array(equities[:-1])
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7)  # ~7 bars/day for 1H
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "return_pct": return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "final_equity": final_equity,
        "initial_equity": initial_equity,
    }


def validate(paper: dict, backtest: dict, min_trades: int = 10) -> list[ValidationResult]:
    results = []

    n = paper["n_trades"]
    if n < min_trades:
        results.append(ValidationResult(
            metric="n_trades",
            paper_value=n,
            backtest_value=backtest.get("n_trades", 0),
            status="warning",
            message=f"Only {n} trades (need {min_trades}+ for statistical validity)"
        ))
        return results

    # Win rate
    bt_wr = backtest.get("win_rate", 0.54)
    p_wr = paper["win_rate"]
    diff = bt_wr - p_wr
    if diff > WIN_RATE_CRIT:
        status = "critical"
    elif diff > WIN_RATE_WARN:
        status = "warning"
    else:
        status = "pass"
    results.append(ValidationResult(
        metric="win_rate",
        paper_value=p_wr,
        backtest_value=bt_wr,
        status=status,
        message=f"Win rate: {p_wr:.1%} vs backtest {bt_wr:.1%} (delta: {-diff:+.1%})"
    ))

    # Profit factor
    bt_pf = backtest.get("profit_factor", 1.0)
    p_pf = paper["profit_factor"]
    if bt_pf > 0:
        ratio = p_pf / bt_pf
        if ratio < PF_CRIT_RATIO:
            status = "critical"
        elif ratio < PF_WARN_RATIO:
            status = "warning"
        else:
            status = "pass"
    else:
        status = "pass"
    results.append(ValidationResult(
        metric="profit_factor",
        paper_value=p_pf,
        backtest_value=bt_pf,
        status=status,
        message=f"Profit factor: {p_pf:.2f} vs backtest {bt_pf:.2f}"
    ))

    # Max drawdown
    bt_dd = abs(backtest.get("max_drawdown_pct", 0))
    p_dd = abs(paper["max_drawdown_pct"])
    if bt_dd > 0:
        ratio = p_dd / bt_dd
        if ratio > DD_CRIT_RATIO:
            status = "critical"
        elif ratio > DD_WARN_RATIO:
            status = "warning"
        else:
            status = "pass"
    else:
        status = "pass" if p_dd < 10 else "warning"
    results.append(ValidationResult(
        metric="max_drawdown",
        paper_value=p_dd,
        backtest_value=bt_dd,
        status=status,
        message=f"Max drawdown: {p_dd:.1f}% vs backtest {bt_dd:.1f}%"
    ))

    # Sharpe ratio
    bt_sharpe = backtest.get("sharpe_ratio", 0)
    p_sharpe = paper["sharpe_ratio"]
    if bt_sharpe > 0:
        ratio = p_sharpe / bt_sharpe
        if ratio < SHARPE_CRIT_RATIO:
            status = "critical"
        elif ratio < SHARPE_WARN_RATIO:
            status = "warning"
        else:
            status = "pass"
    else:
        status = "pass"
    results.append(ValidationResult(
        metric="sharpe_ratio",
        paper_value=p_sharpe,
        backtest_value=bt_sharpe,
        status=status,
        message=f"Sharpe: {p_sharpe:.2f} vs backtest {bt_sharpe:.2f}"
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate paper trading vs backtest expectations")
    parser.add_argument("--session", type=str, default="latest",
                        help="Session ID or 'latest'")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum trades for statistical validity")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Output validation results as JSON")
    args = parser.parse_args()

    # Load backtest baseline
    backtest = load_track_record()
    print("=" * 60)
    print("  PAPER vs BACKTEST VALIDATION")
    print("=" * 60)
    print(f"  Backtest baseline: {backtest['asof_date']}")
    print(f"  Strategy: {backtest.get('strategy_mode', 'unknown')}")
    print(f"  Symbols: {backtest.get('symbols', [])}")
    print()

    # Find session
    if args.session == "latest":
        session_dir = find_latest_session()
        if not session_dir:
            print("ERROR: No sessions found in data/live/sessions/")
            sys.exit(1)
    else:
        session_dir = SESSIONS_DIR / args.session
        if not session_dir.exists():
            print(f"ERROR: Session not found: {session_dir}")
            sys.exit(1)

    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest in {session_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"  Session: {session_dir.name}")
    print(f"  Status: {manifest.get('status', 'unknown')}")
    print(f"  Symbols: {manifest.get('symbols', [])}")
    print()

    # Compute paper metrics
    paper = compute_paper_metrics(session_dir, manifest)
    print(f"  Paper trades: {paper['n_trades']}")
    print(f"  Paper return: {paper['return_pct']:+.2f}%")
    print(f"  Paper equity: ${paper['final_equity']:,.2f}")
    print()

    # Validate
    results = validate(paper, backtest, min_trades=args.min_trades)

    has_critical = False
    has_warning = False

    print("  VALIDATION RESULTS")
    print("  " + "-" * 56)
    for r in results:
        icon = {"pass": "[OK]", "warning": "[WARN]", "critical": "[CRIT]"}[r.status]
        print(f"  {icon} {r.message}")
        if r.status == "critical":
            has_critical = True
        elif r.status == "warning":
            has_warning = True

    print()
    print("  " + "=" * 56)
    if has_critical:
        verdict = "FAIL - Critical deviations detected. Investigate before continuing."
        print(f"  VERDICT: {verdict}")
    elif has_warning:
        verdict = "CAUTION - Some metrics outside expected range. Monitor closely."
        print(f"  VERDICT: {verdict}")
    else:
        verdict = "PASS - Paper trading metrics within expected range."
        print(f"  VERDICT: {verdict}")
    print("  " + "=" * 56)

    # JSON output
    if args.json_out:
        out = {
            "session": session_dir.name,
            "backtest_date": backtest["asof_date"],
            "paper_metrics": paper,
            "backtest_metrics": {
                "win_rate": backtest.get("win_rate"),
                "profit_factor": backtest.get("profit_factor"),
                "sharpe_ratio": backtest.get("sharpe_ratio"),
                "max_drawdown_pct": backtest.get("max_drawdown_pct"),
                "n_trades": backtest.get("n_trades"),
            },
            "results": [
                {
                    "metric": r.metric,
                    "paper": r.paper_value,
                    "backtest": r.backtest_value,
                    "status": r.status,
                    "message": r.message,
                }
                for r in results
            ],
            "verdict": "fail" if has_critical else "caution" if has_warning else "pass",
            "timestamp": datetime.now().isoformat(),
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  JSON saved: {args.json_out}")

    sys.exit(2 if has_critical else 1 if has_warning else 0)


if __name__ == "__main__":
    main()
