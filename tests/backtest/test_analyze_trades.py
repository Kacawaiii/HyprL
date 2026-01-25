from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_trades.py"
_SPEC = importlib.util.spec_from_file_location("analyze_trades", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)

analyze_trades = _MODULE.analyze_trades
load_trades = _MODULE.load_trades


def _sample_trades() -> pd.DataFrame:
    entries = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    exits = entries + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "entry_timestamp": entries,
            "exit_timestamp": exits,
            "direction": ["long", "short", "long", "short"],
            "probability_up": [0.62, 0.58, 0.74, 0.81],
            "threshold": [0.4, 0.4, 0.4, 0.4],
            "entry_price": [100.0, 101.0, 102.5, 103.0],
            "exit_price": [101.0, 100.0, 103.5, 102.0],
            "position_size": [10, 12, 8, 9],
            "pnl": [10.0, -12.0, 8.0, -9.0],
            "return_pct": [0.01, -0.011, 0.008, -0.009],
            "equity_after": [1010.0, 998.0, 1006.0, 997.0],
        }
    )


def test_analyze_trades_outputs(tmp_path: Path) -> None:
    df = _sample_trades()
    csv_path = tmp_path / "trades.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_trades(csv_path)
    results = analyze_trades(loaded)

    stats = results["basic_stats"]
    assert 0 < stats["win_rate"] < 1
    assert np.isfinite(stats["expectancy"])
    calibration = results["calibration"]
    assert calibration
    assert all(entry["count"] > 0 for entry in calibration)
