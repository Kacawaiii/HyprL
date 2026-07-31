#!/usr/bin/env python3
"""Reproduce the frozen ETH base-strategy discovery backtest.

Signals use data through day T and become the close-to-close position for day
T+1.  One-way turnover costs are deducted whenever that lagged position changes.
News is intentionally absent: its incremental value is measured separately by
the forward paper A/B test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eth_paper.strategy import (  # noqa: E402
    EthStrategyConfig,
    compute_base_signal,
)


@dataclass(frozen=True)
class BacktestMetrics:
    start: str
    end: str
    observations: int
    cagr: float
    sharpe: float
    max_drawdown: float
    average_exposure: float
    turnover: float
    final_multiple: float


def build_target_series(
    closes: pd.Series,
    *,
    config: EthStrategyConfig | None = None,
) -> pd.Series:
    cfg = config or EthStrategyConfig()
    values = pd.Series(closes, copy=True).astype(float).dropna().sort_index()
    required = max(
        max(cfg.trend_windows),
        max(cfg.momentum_windows) + 1,
        cfg.volatility_window + 1,
    )
    targets = pd.Series(0.0, index=values.index)
    for offset in range(required - 1, len(values)):
        targets.iloc[offset] = compute_base_signal(
            values.iloc[: offset + 1],
            as_of=values.index[offset],
            config=cfg,
        ).target_exposure
    return targets


def backtest_series(
    closes: pd.Series,
    *,
    cost_bps: float = 30.0,
    config: EthStrategyConfig | None = None,
) -> pd.DataFrame:
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    values = pd.Series(closes, copy=True).astype(float).dropna().sort_index()
    targets = build_target_series(values, config=config)
    returns = values.pct_change().fillna(0.0)
    # A target formed with close T becomes the position for T -> T+1.
    position = targets.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    net_return = position * returns - turnover * (cost_bps / 10_000.0)
    equity = (1.0 + net_return).cumprod()
    return pd.DataFrame(
        {
            "close": values,
            "target": targets,
            "position": position,
            "turnover": turnover,
            "gross_return": position * returns,
            "net_return": net_return,
            "equity": equity,
        }
    )


def summarize(frame: pd.DataFrame) -> BacktestMetrics:
    if len(frame) < 2:
        raise ValueError("backtest frame is too short")
    returns = frame["net_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    years = len(frame) / 365.25
    final_multiple = float(equity.iloc[-1])
    cagr = final_multiple ** (1.0 / years) - 1.0
    volatility = float(returns.std(ddof=0))
    sharpe = (
        float(returns.mean() / volatility * math.sqrt(365.25))
        if volatility > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    return BacktestMetrics(
        start=str(pd.Timestamp(frame.index[0]).date()),
        end=str(pd.Timestamp(frame.index[-1]).date()),
        observations=len(frame),
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        average_exposure=float(frame["position"].mean()),
        turnover=float(frame["turnover"].sum()),
        final_multiple=final_multiple,
    )


def load_yahoo(start: str, end: str | None) -> pd.Series:
    import yfinance as yf

    data = yf.download(
        "ETH-USD",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or "Close" not in data:
        raise RuntimeError("Yahoo returned no ETH-USD daily closes")
    return data["Close"].dropna().astype(float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-11-09")
    parser.add_argument("--end")
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    closes = load_yahoo(args.start, args.end)
    frame = backtest_series(closes, cost_bps=args.cost_bps)
    periods = {
        "full": frame,
        "since_2022": frame.loc["2022-01-01":],
        "since_2023": frame.loc["2023-01-01":],
    }
    report = {name: asdict(summarize(part)) for name, part in periods.items()}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, metrics in report.items():
            print(
                f"{name:10} CAGR={metrics['cagr']:.2%} "
                f"Sharpe={metrics['sharpe']:.2f} "
                f"MaxDD={metrics['max_drawdown']:.2%} "
                f"AvgExp={metrics['average_exposure']:.1%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
