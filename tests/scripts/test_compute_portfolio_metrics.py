from pathlib import Path

import json
import pandas as pd
import runpy


def test_compute_portfolio_metrics_runs(tmp_path, monkeypatch) -> None:
    # Build synthetic equity for two tickers
    idx = pd.date_range("2025-01-01", periods=10, freq="h")
    eq1 = pd.DataFrame({"timestamp": idx, "equity": 1000 + (idx.view(int) % 10)})
    eq2 = pd.DataFrame({"timestamp": idx, "equity": 1000 + ((idx.view(int) % 10) * 0.5)})
    f1 = tmp_path / "t1.csv"
    f2 = tmp_path / "t2.csv"
    eq1.to_csv(f1, index=False)
    eq2.to_csv(f2, index=False)

    out = tmp_path / "out.json"
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    args = [
        "scripts/compute_portfolio_metrics.py",
        "--equity-logs",
        str(f1),
        str(f2),
        "--tickers",
        "T1",
        "T2",
        "--weights",
        "0.6",
        "0.4",
        "--output",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", args)
    runpy.run_path("scripts/compute_portfolio_metrics.py", run_name="__main__")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "pf" in payload
    assert "sharpe" in payload
    assert "maxdd" in payload
