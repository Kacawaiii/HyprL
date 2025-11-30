from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import shutil

from hyprl.backtest.runner import BacktestConfig, BacktestResult, TradeRecord

BACKUP_ROOT = Path("backups")


def _serialize_config(config: BacktestConfig) -> dict[str, Any]:
    data = asdict(config)
    return data


def _serialize_result(result: BacktestResult) -> dict[str, Any]:
    result_dict = asdict(result)
    result_dict.pop("equity_curve", None)
    trades = result_dict.pop("trades", [])
    result_dict["trades_summary"] = {
        "count": len(trades),
        "total_pnl": sum(trade["pnl"] for trade in trades),
        "long_trades": result.long_trades,
        "short_trades": result.short_trades,
    }
    return result_dict


def save_snapshot(
    config: BacktestConfig,
    result: BacktestResult,
    trades_path: Path | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{config.ticker}"
    snapshot_dir = BACKUP_ROOT / folder_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    config_json = snapshot_dir / "config.json"
    config_json.write_text(json.dumps(_serialize_config(config), indent=2), encoding="utf-8")

    result_json = snapshot_dir / "result.json"
    result_json.write_text(json.dumps(_serialize_result(result), indent=2), encoding="utf-8")

    if trades_path and trades_path.exists():
        shutil.copy(trades_path, snapshot_dir / trades_path.name)

    return snapshot_dir


def make_backup_zip(snapshot_dir: Path) -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = BACKUP_ROOT / snapshot_dir.name
    zip_path = Path(shutil.make_archive(str(archive_path), "zip", root_dir=snapshot_dir))
    return zip_path
