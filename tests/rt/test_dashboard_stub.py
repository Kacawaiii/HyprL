from __future__ import annotations

import json
from pathlib import Path

from scripts.hyprl_live_dashboard import load_session_frames


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dashboard_loader(tmp_path: Path) -> None:
    session_dir = tmp_path / "dash-session"
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(session_dir / "bars.jsonl", [{"ts": 0, "symbol": "AAA", "close": 100.0}])
    _write_jsonl(session_dir / "equity.jsonl", [{"ts": 0, "equity": 10000.0}])
    _write_jsonl(
        session_dir / "predictions.jsonl",
        [{"ts": 0, "prob_up": 0.6, "threshold": 0.5, "reason": "signal"}],
    )
    _write_jsonl(session_dir / "orders.jsonl", [{"symbol": "AAA", "price_ref": 100.0}])
    _write_jsonl(session_dir / "fills.jsonl", [{"symbol": "AAA", "qty": 1, "price": 100.5}])
    _write_jsonl(session_dir / "events.jsonl", [{"event": "resume"}])
    (session_dir / "session_manifest.json").write_text(
        json.dumps(
            {
                "session_id": "dash-session",
                "created_at": 0.0,
                "resumed_from": None,
                "last_bar_ts": 0.0,
                "equity_peak": 10000.0,
                "killswitch": {"enabled": False, "dd_limit": None, "triggered": False, "triggered_at_ts": None, "dd_at_trigger": None},
            }
        ),
        encoding="utf-8",
    )

    frames = load_session_frames(session_dir)
    assert "bars" in frames and isinstance(frames["bars"], object)
    assert not frames["bars"].empty
    assert not frames["equity"].empty
    assert frames["manifest"]["session_id"] == "dash-session"
