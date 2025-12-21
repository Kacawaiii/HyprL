#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshots(snap_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not snap_dir.exists():
        return items
    for path in sorted(snap_dir.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data["_path"] = path.as_posix()
        data["_date"] = path.stem
        items.append(data)
    return items


def load_orders_summary(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    total = 0
    opens = 0
    closes = 0
    first_ts = None
    last_ts = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        ev = str(obj.get("event", ""))
        if ev.startswith("open_") or ev.startswith("would_open"):
            opens += 1
        if ev in {"close_position", "would_close"}:
            closes += 1
        ts = obj.get("ts")
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts
    return {
        "orders_total": total,
        "orders_open": opens,
        "orders_close": closes,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "source": path.as_posix(),
    }


def write_report(out_dir: Path, payload: Dict[str, Any]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    asof_date = payload.get("asof_date", "unknown")
    report_json = out_dir / "track_record_latest.json"
    report_md = out_dir / f"TRACK_RECORD_{asof_date}.md"
    latest_md = out_dir / "TRACK_RECORD_latest.md"

    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# Track Record Report",
        "",
        f"- Generated: {payload.get('generated_ts')}",
        f"- Mode: {payload.get('mode')}",
        f"- Period: {payload.get('period')}",
        f"- Timeframe: {payload.get('timeframe')}",
        "",
        "## Summary",
        f"- Start equity: {payload.get('start_equity')}",
        f"- End equity: {payload.get('end_equity')}",
        f"- Return (%): {payload.get('return_pct')}",
        f"- Max drawdown (%): {payload.get('max_drawdown_pct')}",
        f"- Snapshots: {payload.get('snapshots_count')}",
    ]

    orders = payload.get("orders", {})
    if orders:
        md_lines += [
            "",
            "## Orders (bridge log)",
            f"- Total: {orders.get('orders_total')}",
            f"- Opens: {orders.get('orders_open')}",
            f"- Closes: {orders.get('orders_close')}",
            f"- First ts: {orders.get('first_ts')}",
            f"- Last ts: {orders.get('last_ts')}",
        ]

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    shutil.copyfile(report_md, latest_md)

    return {"json": report_json, "md": report_md, "latest_md": latest_md}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate track-record report.")
    parser.add_argument("--in-dir", default="docs/reports/track_record")
    parser.add_argument("--orders-log", default="live/execution/alpaca/orders.jsonl")
    parser.add_argument("--out-dir", default="docs/reports/track_record")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    snap_dir = in_dir / "snapshots"
    snaps = load_snapshots(snap_dir)
    if not snaps:
        print(f"No snapshots found in {snap_dir}", file=sys.stderr)
        raise SystemExit(2)

    snaps_sorted = sorted(snaps, key=lambda x: x.get("_date", ""))
    latest = snaps_sorted[-1]

    payload = {
        "generated_ts": _utc_now(),
        "mode": latest.get("mode"),
        "period": latest.get("period"),
        "timeframe": latest.get("timeframe"),
        "asof_date": latest.get("_date"),
        "start_equity": latest.get("start_equity"),
        "end_equity": latest.get("end_equity"),
        "return_pct": latest.get("return_pct"),
        "max_drawdown_pct": latest.get("max_drawdown_pct"),
        "latest_snapshot": latest.get("_path"),
        "snapshots_count": len(snaps_sorted),
        "snapshots": [
            {
                "date": snap.get("_date"),
                "end_equity": snap.get("end_equity"),
                "return_pct": snap.get("return_pct"),
                "max_drawdown_pct": snap.get("max_drawdown_pct"),
                "path": snap.get("_path"),
            }
            for snap in snaps_sorted
        ],
    }

    orders = load_orders_summary(Path(args.orders_log) if args.orders_log else None)
    if orders:
        payload["orders"] = orders

    write_report(Path(args.out_dir), payload)


if __name__ == "__main__":
    main()
