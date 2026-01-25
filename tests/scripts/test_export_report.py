from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.export_report import export_report, parse_weights


def make_trades(path: Path, entries: list[dict]) -> None:
    pd.DataFrame(entries).to_csv(path, index=False)


def test_export_report_single_ticker(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades_NVDA_live.csv"
    make_trades(
        trades_path,
        [
            {"exit_timestamp": "2025-01-01T10:00:00+00:00", "pnl": 10.0},
            {"exit_timestamp": "2025-01-02T10:00:00+00:00", "pnl": -5.0},
            {"exit_timestamp": "2025-01-03T10:00:00+00:00", "pnl": 7.0},
        ],
    )
    out = tmp_path / "report.md"
    export_report(
        trade_logs=[trades_path],
        weights={"NVDA": 1.0},
        initial_equity=10000.0,
        annualization=1638,
        output_path=out,
        fmt="md",
    )
    content = out.read_text()
    assert out.exists()
    assert "PF_portfolio" in content
    assert "MaxDD_portfolio" in content
    assert "Trades_total" in content
    assert "NVDA" in content
    assert "Généré le" in content


def test_export_report_two_tickers(tmp_path: Path) -> None:
    root = tmp_path
    t1 = root / "trades_NVDA_live.csv"
    t2 = root / "trades_MSFT_live.csv"
    make_trades(
        t1,
        [
            {"exit_timestamp": "2025-01-01T10:00:00+00:00", "pnl": 10.0},
            {"exit_timestamp": "2025-01-02T10:00:00+00:00", "pnl": -5.0},
        ],
    )
    make_trades(
        t2,
        [
            {"exit_timestamp": "2025-01-01T12:00:00+00:00", "pnl": 3.0},
            {"exit_timestamp": "2025-01-03T12:00:00+00:00", "pnl": -1.0},
        ],
    )
    out = root / "report2.md"
    export_report(
        trade_logs=[t1, t2],
        weights=parse_weights("NVDA=0.6,MSFT=0.4"),
        initial_equity=10000.0,
        annualization=1638,
        output_path=out,
        fmt="md",
    )
    txt = out.read_text()
    assert "NVDA" in txt and "MSFT" in txt
    assert "Trades_total" in txt
    assert abs(sum(parse_weights("NVDA=0.6,MSFT=0.4").values()) - 1.0) < 1e-6


def test_export_report_missing_weight_errors(tmp_path: Path) -> None:
    t1 = tmp_path / "trades_NVDA_live.csv"
    t2 = tmp_path / "trades_MSFT_live.csv"
    make_trades(
        t1,
        [
            {"exit_timestamp": "2025-01-01T10:00:00+00:00", "pnl": 1.0},
        ],
    )
    make_trades(
        t2,
        [
            {"exit_timestamp": "2025-01-02T10:00:00+00:00", "pnl": 2.0},
        ],
    )
    out = tmp_path / "report_missing.md"
    with pytest.raises(SystemExit, match="Missing weight"):
        export_report(
            trade_logs=[t1, t2],
            weights={"NVDA": 1.0},
            initial_equity=10000.0,
            annualization=1638,
            output_path=out,
            fmt="md",
        )


def test_export_report_html_format(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades_NVDA_live.csv"
    make_trades(
        trades_path,
        [
            {"exit_timestamp": "2025-01-01T10:00:00+00:00", "pnl": 5.0},
            {"exit_timestamp": "2025-01-02T10:00:00+00:00", "pnl": 5.0},
        ],
    )
    out = tmp_path / "report.html"
    try:
        import markdown  # type: ignore

        markdown_available = True
    except Exception:
        markdown_available = False

    if markdown_available:
        export_report(
            trade_logs=[trades_path],
            weights={"NVDA": 1.0},
            initial_equity=10000.0,
            annualization=1638,
            output_path=out,
            fmt="html",
        )
        html = out.read_text()
        assert out.exists()
        assert "<h1>" in html or "<p>" in html
    else:
        with pytest.raises(SystemExit, match="markdown package"):
            export_report(
                trade_logs=[trades_path],
                weights={"NVDA": 1.0},
                initial_equity=10000.0,
                annualization=1638,
                output_path=out,
                fmt="html",
            )


def test_export_report_cli(tmp_path: Path, monkeypatch) -> None:
    trades_path = tmp_path / "trades_NVDA_live.csv"
    make_trades(
        trades_path,
        [
            {"exit_timestamp": "2025-01-01T10:00:00+00:00", "pnl": 5.0},
            {"exit_timestamp": "2025-01-02T10:00:00+00:00", "pnl": 5.0},
        ],
    )
    out = tmp_path / "cli_report.md"
    cmd = [
        sys.executable,
        "scripts/reporting/export_report.py",
        "--trade-logs",
        str(trades_path),
        "--weights",
        "NVDA=1.0",
        "--initial-equity",
        "10000",
        "--annualization",
        "1638",
        "--output",
        str(out),
        "--format",
        "md",
    ]
    res = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert out.exists()
    txt = out.read_text()
    assert "PF_portfolio" in txt and "NVDA" in txt
