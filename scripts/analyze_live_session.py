#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hyprl.rt.logging import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse une session realtime (logs JSONL).")
    parser.add_argument("--session", required=True, help="Chemin vers data/live/sessions/<id>.")
    parser.add_argument("--output", type=Path, default=Path("live_report.csv"))
    return parser.parse_args()


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compute_metrics(equity_df: pd.DataFrame) -> dict[str, float]:
    if equity_df.empty:
        return {"pf": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    equity_df = equity_df.sort_values("ts")
    returns = equity_df["equity"].pct_change().dropna()
    positives = returns[returns > 0].sum()
    negatives = returns[returns < 0].sum()
    pf = positives / abs(negatives) if negatives < 0 else np.nan
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(max(len(returns), 1)) if len(returns) > 1 else np.nan
    equity = equity_df["equity"].to_numpy()
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(abs(dd.min())) if len(dd) else np.nan
    return {"pf": float(pf), "sharpe": float(sharpe), "max_dd": max_dd}


def compute_slippage(orders_df: pd.DataFrame, fills_df: pd.DataFrame) -> float:
    if orders_df.empty or fills_df.empty:
        return float("nan")
    order_prices = orders_df.get("price_ref")
    fill_prices = fills_df.get("price")
    if order_prices is None or fill_prices is None:
        return float("nan")
    merged = pd.DataFrame({"order": order_prices.dropna().to_numpy(), "fill": fill_prices.dropna().to_numpy()})
    if merged.empty:
        return float("nan")
    count = min(len(merged["order"]), len(merged["fill"]))
    merged = merged.iloc[:count]
    with np.errstate(divide="ignore", invalid="ignore"):
        slippage = np.abs(merged["fill"] - merged["order"]) / merged["order"].replace(0, np.nan)
    return float(slippage.mean())


def summarize_manifest(session_dir: Path) -> dict[str, float | bool | None]:
    manifest_path = session_dir / "session_manifest.json"
    manifest = load_manifest(manifest_path)
    killswitch = manifest.get("killswitch", {}) if manifest else {}
    return {
        "killswitch_triggered": bool(killswitch.get("triggered")),
        "dd_at_trigger": killswitch.get("dd_at_trigger"),
    }


def main() -> None:
    args = parse_args()
    session_dir = Path(args.session)
    equity_df = _load_jsonl(session_dir / "equity.jsonl")
    metrics = compute_metrics(equity_df)

    predictions_df = _load_jsonl(session_dir / "predictions.jsonl")
    reason_series = predictions_df.get("reason") if not predictions_df.empty else None
    signals = predictions_df[reason_series == "signal"] if reason_series is not None else pd.DataFrame()
    if reason_series is not None:
        avg_hold_bars = len(predictions_df) / max(len(signals), 1)
        exposure = len(signals) / len(predictions_df)
        reason_counts = reason_series.value_counts().head(5).to_dict()
    else:
        avg_hold_bars = np.nan
        exposure = np.nan
        reason_counts = {}

    winrate = np.nan
    if not signals.empty:
        wins = 0
        evals = 0
        signals = signals.reset_index(drop=True)
        for idx, row in signals.iterrows():
            next_idx = row.name + 1
            if next_idx >= len(predictions_df):
                continue
            next_close = predictions_df.iloc[next_idx].get("close")
            if next_close is None:
                continue
            current_close = row.get("close")
            if current_close is None:
                continue
            change = next_close - current_close
            direction = row.get("direction", "UP")
            evals += 1
            if (direction == "UP" and change >= 0) or (direction != "UP" and change <= 0):
                wins += 1
        if evals:
            winrate = wins / evals

    orders_df = _load_jsonl(session_dir / "orders.jsonl")
    fills_df = _load_jsonl(session_dir / "fills.jsonl")
    slippage = compute_slippage(orders_df, fills_df)
    tuner_updates = 0
    if not predictions_df.empty and "event" in predictions_df.columns:
        tuner_updates = int((predictions_df["event"] == "tuner").sum())

    manifest_summary = summarize_manifest(session_dir)

    metrics.update(
        {
            "avg_hold_bars": float(avg_hold_bars) if avg_hold_bars == avg_hold_bars else np.nan,
            "exposure": float(exposure) if exposure == exposure else np.nan,
            "winrate": float(winrate) if winrate == winrate else np.nan,
            "avg_slippage": slippage,
            "tuner_updates": tuner_updates,
            "killswitch_triggered": manifest_summary["killswitch_triggered"],
            "dd_at_trigger": manifest_summary["dd_at_trigger"],
        }
    )

    output_df = pd.DataFrame([metrics])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    summary_path = args.output.with_suffix(".SUMMARY.txt")
    reasons_str = ", ".join(f"{k}:{v}" for k, v in reason_counts.items()) or "none"
    summary_lines = [
        f"PF={metrics['pf']:.3f} Sharpe={metrics['sharpe']:.3f} MaxDD={metrics['max_dd']:.3f}",
        f"AvgHoldBars={metrics['avg_hold_bars']:.2f} Exposure={metrics['exposure']:.2%} Winrate={metrics['winrate']:.2%}",
        f"Slippage={metrics['avg_slippage']:.4f} TunerUpdates={metrics['tuner_updates']} "
        f"KillSwitch={metrics['killswitch_triggered']} DD_at_trigger={metrics['dd_at_trigger']}",
        f"Top reasons: {reasons_str}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] Rapport live écrit dans {args.output}")


if __name__ == "__main__":
    main()
