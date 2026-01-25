from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_live_session import compute_metrics


def test_analyze_live_session_metrics(tmp_path: Path) -> None:
    equity_path = tmp_path / "equity.jsonl"
    # Include both gains and losses to get valid profit factor
    records = [
        {"ts": 1, "equity": 10000},
        {"ts": 2, "equity": 10100},  # gain
        {"ts": 3, "equity": 10050},  # loss
        {"ts": 4, "equity": 10150},  # gain
        {"ts": 5, "equity": 10100},  # loss
    ]
    equity_path.write_text("\n".join(json.dumps(rec) for rec in records), encoding="utf-8")
    import pandas as pd

    df = pd.DataFrame(records)
    metrics = compute_metrics(df)
    # pf should be computed when there are both gains and losses
    assert pd.notna(metrics["pf"]), "pf should not be NaN when there are gains and losses"
    assert metrics["pf"] > 0, "pf should be positive"
