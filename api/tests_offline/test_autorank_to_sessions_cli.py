import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_cli_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "tickers": "BTC,ETH",
                "config_index": 10,
                "base_score_normalized": 0.9,
                "pf_backtest": 1.3,
                "portfolio_sharpe": 0.5,
                "sharpe_backtest": 0.5,
                "maxdd_backtest": 0.1,
                "portfolio_dd": 10,
                "n_trades": 80,
                "trades_backtest": 80,
            }
        ]
    )
    df.to_csv(path, index=False)
    return path


def test_autorank_cli_dry_run(monkeypatch):
    monkeypatch.setenv("HYPRL_DB_URL", "sqlite:///./test_hyprl_v2.db")
    monkeypatch.setenv("HYPRL_ADMIN_TOKEN", "hyprl_admin_dev_123")
    monkeypatch.setenv("HYPRL_RT_IMPL", "stub")
    project_root = Path(__file__).resolve().parents[2]
    csv_path = _write_cli_csv(project_root / "data/experiments/test_autorank_cli.csv")
    job_root = (project_root / "data/autorank_jobs")
    before = set(job_root.glob("ar_*")) if job_root.exists() else set()
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "scripts/autorank_to_sessions.py",
        "--csv",
        str(csv_path),
        "--top-k",
        "1",
        "--dry-run",
        "--session-threshold",
        "0.6",
        "--session-risk-pct",
        "0.1",
    ]
    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job_root.mkdir(parents=True, exist_ok=True)
    after = set(job_root.glob("ar_*"))
    new_dirs = after - before
    assert new_dirs, "Expected a new autorank job folder"
    latest = max(new_dirs, key=lambda p: p.stat().st_mtime)
    assert (latest / "autoranked.csv").exists()
    assert (latest / "autoranked.SUMMARY.txt").exists()
