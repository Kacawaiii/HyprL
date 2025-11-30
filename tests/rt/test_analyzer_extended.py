from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_analyzer_extended_metrics(tmp_path: Path) -> None:
    session_dir = tmp_path / "an-session"
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(session_dir / "equity.jsonl", [{"ts": 0, "equity": 10000.0}, {"ts": 1, "equity": 10100.0}, {"ts": 2, "equity": 9800.0}])
    _write_jsonl(
        session_dir / "predictions.jsonl",
        [
            {"ts": 0, "prob_up": 0.6, "threshold": 0.5, "reason": "signal", "direction": "UP"},
            {"ts": 1, "event": "tuner", "delta": {"threshold": 0.02}, "after": {"threshold": 0.52, "risk_pct": 0.19}},
            {"ts": 2, "event": "kill_switch", "dd": 0.2},
            {"ts": 3, "event": "oco_close", "reason": "tp"},
        ],
    )
    _write_jsonl(session_dir / "orders.jsonl", [{"symbol": "AAA", "price_ref": 100.0}])
    _write_jsonl(session_dir / "fills.jsonl", [{"symbol": "AAA", "price": 100.5}])
    _write_jsonl(session_dir / "bars.jsonl", [{"ts": 0, "symbol": "AAA", "close": 100.0}])
    _write_jsonl(session_dir / "events.jsonl", [{"event": "resume"}])
    manifest = {
        "session_id": "an-session",
        "created_at": 0.0,
        "resumed_from": None,
        "last_bar_ts": 2.0,
        "equity_peak": 10100.0,
        "killswitch": {
            "enabled": True,
            "dd_limit": 0.3,
            "triggered": True,
            "triggered_at_ts": 2.0,
            "dd_at_trigger": 0.2,
        },
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_csv = tmp_path / "live_report.csv"
    subprocess.run(
        ["python", "scripts/analyze_live_session.py", "--session", str(session_dir), "--output", str(output_csv)],
        check=True,
    )
    df = pd.read_csv(output_csv)
    for column in ["avg_slippage", "tuner_updates", "killswitch_triggered", "dd_at_trigger"]:
        assert column in df.columns
    summary_text = output_csv.with_suffix(".SUMMARY.txt").read_text()
    assert "TunerUpdates" in summary_text
    assert "KillSwitch" in summary_text
