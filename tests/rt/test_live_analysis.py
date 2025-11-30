from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_live_session import compute_metrics


def test_analyze_live_session_metrics(tmp_path: Path) -> None:
    equity_path = tmp_path / "equity.jsonl"
    records = [
        {"ts": idx, "equity": 10000 + idx * 10}
        for idx in range(1, 6)
    ]
    equity_path.write_text("\n".join(json.dumps(rec) for rec in records), encoding="utf-8")
    import pandas as pd

    df = pd.DataFrame(records)
    metrics = compute_metrics(df)
    assert metrics["pf"] == metrics["pf"]  # not NaN
