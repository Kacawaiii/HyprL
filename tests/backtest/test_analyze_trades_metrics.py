from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_trades.py"
SPEC = importlib.util.spec_from_file_location("analyze_trades", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_analyze_trades_additional_stats():
    df = pd.DataFrame(
        {
            "pnl": [100, -50, 200, -25, -10],
            "return_pct": [0.01, -0.005, 0.02, -0.0025, -0.001],
            "probability_up": [0.7, 0.6, 0.8, 0.55, 0.45],
            "expected_pnl": [80, -30, 150, -20, -15],
        }
    )
    stats = MODULE.compute_basic_stats(df)
    assert stats["profit_factor"] > 0
    assert stats["best_trade"] == 200
    assert stats["worst_trade"] == -50
    assert stats["max_consec_losses"] >= 1
    custom_bins = [0.4, 0.5, 0.7, 1.01]
    calibration = MODULE.compute_calibration(df, custom_bins)
    assert calibration[0]["bin"].startswith("[0.40")
