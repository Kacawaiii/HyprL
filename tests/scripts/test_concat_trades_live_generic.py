from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.concat_trades_live_generic import concat_trades


def test_concat_dedup_sort(tmp_path: Path) -> None:
    root = tmp_path / "live_nvda"
    root.mkdir()
    f1 = root / "trades_NVDA_live_2025-01-01.csv"
    f2 = root / "sub" / "trades_NVDA_live_2025-01-02.csv"
    f2.parent.mkdir()

    data1 = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T10:00:00+00:00",
                "direction": "long",
                "position_size": 1.0,
                "entry_price": 100.0,
                "pnl": 10.0,
            },
            {
                "entry_timestamp": "2025-01-01T10:00:00+00:00",
                "direction": "short",
                "position_size": 2.0,
                "entry_price": 90.0,
                "pnl": -5.0,
            },
        ]
    )
    data2 = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T10:00:00+00:00",  # duplicate of first row in data1
                "direction": "long",
                "position_size": 1.0,
                "entry_price": 100.0,
                "pnl": 12.0,  # different pnl to ensure dedupe keeps last
            },
            {
                "entry_timestamp": "2025-01-03T10:00:00+00:00",
                "direction": "long",
                "position_size": 1.5,
                "entry_price": 105.0,
                "pnl": 8.0,
            },
        ]
    )
    data1.to_csv(f1, index=False)
    data2.to_csv(f2, index=False)

    output = tmp_path / "trades_NVDA_live_all.csv"
    concat_trades("NVDA", root, output)

    result = pd.read_csv(output)
    # Expect 3 rows: dedup removed duplicate entry_timestamp+direction+size+price
    assert len(result) == 3
    # Sorted by entry_timestamp ascending
    assert list(result["entry_timestamp"]) == [
        "2025-01-01T10:00:00+00:00",
        "2025-01-02T10:00:00+00:00",
        "2025-01-03T10:00:00+00:00",
    ]
    # Duplicate kept last (from data2)
    assert float(result.loc[result["entry_timestamp"] == "2025-01-02T10:00:00+00:00", "pnl"].iloc[0]) == 12.0


def test_concat_pattern_filter(tmp_path: Path) -> None:
    root = tmp_path / "live"
    root.mkdir()
    keep = root / "trades_NVDA_live_2025-01-01.csv"
    drop = root / "trades_NVDA_live_extra.csv"
    alt = root / "trades_MSFT_live_2025-01-01.csv"
    pd.DataFrame([{"entry_timestamp": "2025-01-01T10:00:00+00:00", "direction": "long", "position_size": 1.0, "entry_price": 10.0, "pnl": 1.0, "exit_price": 11.0}]).to_csv(keep, index=False)
    pd.DataFrame([{"entry_timestamp": "2025-01-02T10:00:00+00:00", "direction": "long", "position_size": 1.0, "entry_price": 12.0, "pnl": 1.0, "exit_price": 13.0}]).to_csv(drop, index=False)
    pd.DataFrame([{"entry_timestamp": "2025-01-01T11:00:00+00:00", "direction": "long", "position_size": 1.0, "entry_price": 20.0, "pnl": 2.0, "exit_price": 22.0}]).to_csv(alt, index=False)

    output = tmp_path / "filtered.csv"
    concat_trades("NVDA", root, output, pattern="trades_NVDA_live_2025-01-01.csv")
    df = pd.read_csv(output)
    assert len(df) == 1
    assert df["entry_price"].iloc[0] == 10.0


def test_concat_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "empty.csv"
    concat_trades("NVDA", tmp_path, output)
    df = pd.read_csv(output)
    expected_cols = [
        "ticker",
        "entry_timestamp",
        "exit_timestamp",
        "direction",
        "probability_up",
        "threshold",
        "entry_price",
        "stop_price",
        "take_profit_price",
        "trailing_stop_activation_price",
        "trailing_stop_distance_price",
        "exit_price",
        "exit_reason",
        "position_size",
        "pnl",
        "return_pct",
        "equity_after",
        "risk_amount",
        "expected_pnl",
        "risk_profile",
        "effective_long_threshold",
        "effective_short_threshold",
        "regime_name",
    ]
    assert list(df.columns) == expected_cols
    assert df.empty


def test_concat_dedupe_trade_id_precedence(tmp_path: Path) -> None:
    root = tmp_path / "live_tradeid"
    root.mkdir()
    f1 = root / "trades_NVDA_live_a.csv"
    f2 = root / "trades_NVDA_live_b.csv"
    pd.DataFrame(
        [
            {"trade_id": "t1", "entry_timestamp": "2025-01-01T10:00:00+00:00", "direction": "long", "position_size": 1.0, "entry_price": 100.0, "exit_price": 105.0, "pnl": 5.0}
        ]
    ).to_csv(f1, index=False)
    pd.DataFrame(
        [
            {"trade_id": "t1", "entry_timestamp": "2025-01-01T10:05:00+00:00", "direction": "long", "position_size": 1.0, "entry_price": 101.0, "exit_price": 104.0, "pnl": 3.0},
            {"trade_id": "t2", "entry_timestamp": "2025-01-02T10:00:00+00:00", "direction": "short", "position_size": 1.0, "entry_price": 90.0, "exit_price": 85.0, "pnl": 5.0},
        ]
    ).to_csv(f2, index=False)
    output = root / "out.csv"
    concat_trades("NVDA", root, output)
    df = pd.read_csv(output)
    # trade_id dedupe keeps last occurrence
    assert len(df) == 2
    assert df.loc[df["trade_id"] == "t1", "entry_price"].iloc[0] == 101.0
