from __future__ import annotations

import json
from pathlib import Path

import importlib.util
import sys


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


snapshot = _load_module(Path("scripts/ops/alpaca_track_record_snapshot.py"), "alpaca_track_record_snapshot")
report = _load_module(Path("scripts/ops/make_track_record_report.py"), "make_track_record_report")


class FakeBroker:
    def get_portfolio_history(self, period: str = "3M", timeframe: str = "1D"):
        return {
            "timestamps": ["t1", "t2", "t3"],
            "equity": [10000, 10500, 9800],
            "profit_loss": [0, 500, -200],
            "profit_loss_pct": [0.0, 5.0, -2.0],
        }


def test_snapshot_write_creates_files(tmp_path: Path) -> None:
    broker = FakeBroker()
    payload = snapshot.build_snapshot(broker, mode="paper", period="3M", timeframe="1D")
    result = snapshot.write_snapshot(payload, tmp_path, "2025-12-20")

    assert result.path.exists()
    assert result.sha_path.exists()
    assert result.latest_path.exists()

    loaded = json.loads(result.path.read_text(encoding="utf-8"))
    assert loaded["mode"] == "paper"
    assert loaded["start_equity"] == 10000


def test_report_generator_from_snapshots(tmp_path: Path, monkeypatch) -> None:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "mode": "paper",
        "period": "3M",
        "timeframe": "1D",
        "start_equity": 10000,
        "end_equity": 10200,
        "return_pct": 2.0,
        "max_drawdown_pct": -3.0,
        "equity_curve": {"timestamps": [], "equity": []},
    }

    (snap_dir / "2025-12-01.json").write_text(json.dumps(data), encoding="utf-8")
    (snap_dir / "2025-12-08.json").write_text(json.dumps(data), encoding="utf-8")

    out_dir = tmp_path / "reports"

    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setattr(report, "Path", Path)
    monkeypatch.setattr(report, "parse_args", lambda: type("Args", (), {
        "in_dir": str(tmp_path),
        "orders_log": "",
        "out_dir": str(out_dir),
    })())

    report.main()

    assert (out_dir / "track_record_latest.json").exists()
    assert (out_dir / "TRACK_RECORD_2025-12-08.md").exists()
    assert (out_dir / "TRACK_RECORD_latest.md").exists()
