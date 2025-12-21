#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List



@dataclass
class SnapshotResult:
    payload: Dict[str, Any]
    path: Path
    sha_path: Path
    latest_path: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _max_drawdown_pct(equity: Iterable[float]) -> float:
    peak = None
    max_dd = 0.0
    for val in equity:
        if peak is None or val > peak:
            peak = val
        if peak and peak > 0:
            dd = (val - peak) / peak * 100.0
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _safe_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_snapshot(
    broker: Any,
    *,
    mode: str,
    period: str,
    timeframe: str,
) -> Dict[str, Any]:
    history = broker.get_portfolio_history(period=period, timeframe=timeframe)
    equity = _safe_float_list(history.get("equity", []))
    timestamps = list(history.get("timestamps", []))
    profit_loss = _safe_float_list(history.get("profit_loss", []))
    profit_loss_pct = _safe_float_list(history.get("profit_loss_pct", []))

    start_equity = equity[0] if equity else 0.0
    end_equity = equity[-1] if equity else 0.0
    return_pct = ((end_equity / start_equity - 1.0) * 100.0) if start_equity > 0 else 0.0
    max_dd_pct = _max_drawdown_pct(equity)

    return {
        "snapshot_ts": _utc_now().isoformat(),
        "mode": mode,
        "period": period,
        "timeframe": timeframe,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "return_pct": return_pct,
        "max_drawdown_pct": max_dd_pct,
        "equity_curve": {
            "timestamps": timestamps,
            "equity": equity,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
        },
    }


def write_snapshot(payload: Dict[str, Any], out_dir: Path) -> SnapshotResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    stamp = _utc_now().date().isoformat()
    path = snapshots_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    sha_path = snapshots_dir / f"{stamp}.sha256"
    digest = _hash_file(path)
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")

    latest_path = out_dir / "latest.json"
    shutil.copyfile(path, latest_path)

    return SnapshotResult(payload=payload, path=path, sha_path=sha_path, latest_path=latest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Alpaca track-record snapshot.")
    parser.add_argument("--paper", action="store_true", help="Use Alpaca paper trading.")
    parser.add_argument("--live", action="store_true", help="Use Alpaca live trading.")
    parser.add_argument("--out-dir", default="docs/reports/track_record", help="Output base directory.")
    parser.add_argument("--period", default="3M", help="Portfolio history period (1D|1W|1M|3M|1A).")
    parser.add_argument("--timeframe", default="1D", help="Portfolio history timeframe (1Min|5Min|15Min|1H|1D).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live and args.paper:
        raise SystemExit("Choose only one of --paper or --live")

    mode = "paper" if args.paper or not args.live else "live"
    from hyprl.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(paper=(mode == "paper"))

    payload = build_snapshot(broker, mode=mode, period=args.period, timeframe=args.timeframe)
    result = write_snapshot(payload, Path(args.out_dir))

    print(json.dumps({
        "snapshot": str(result.path),
        "sha256": str(result.sha_path),
        "latest": str(result.latest_path),
        "mode": mode,
        "period": args.period,
        "timeframe": args.timeframe,
    }, indent=2))


if __name__ == "__main__":
    main()
