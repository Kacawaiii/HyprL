#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_session_frames(session_path: str | Path) -> Dict[str, pd.DataFrame | dict]:
    session_dir = Path(session_path)
    frames: Dict[str, pd.DataFrame | dict] = {
        "bars": _read_jsonl(session_dir / "bars.jsonl"),
        "equity": _read_jsonl(session_dir / "equity.jsonl"),
        "predictions": _read_jsonl(session_dir / "predictions.jsonl"),
        "orders": _read_jsonl(session_dir / "orders.jsonl"),
        "fills": _read_jsonl(session_dir / "fills.jsonl"),
        "events": _read_jsonl(session_dir / "events.jsonl"),
    }
    manifest_path = session_dir / "session_manifest.json"
    frames["manifest"] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return frames


def _compute_equity_metrics(equity_df: pd.DataFrame) -> dict:
    if equity_df.empty or "equity" not in equity_df:
        return {"pf": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    equity_df = equity_df.sort_values("ts")
    returns = equity_df["equity"].pct_change().dropna()
    positives = returns[returns > 0].sum()
    negatives = returns[returns < 0].sum()
    pf = positives / abs(negatives) if negatives < 0 else np.nan
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)) if len(returns) > 1 else np.nan
    equity = equity_df["equity"].to_numpy()
    running = np.maximum.accumulate(equity)
    dd = (equity - running) / running
    max_dd = float(abs(dd.min())) if len(dd) else np.nan
    return {"pf": float(pf), "sharpe": float(sharpe), "max_dd": max_dd}


def _top_symbols(fills_df: pd.DataFrame) -> pd.DataFrame:
    if fills_df.empty or "symbol" not in fills_df:
        return pd.DataFrame(columns=["symbol", "pf"])
    fills_df = fills_df.dropna(subset=["symbol", "qty", "price"])
    if fills_df.empty:
        return pd.DataFrame(columns=["symbol", "pf"])
    fills_df["value"] = fills_df["qty"].astype(float) * fills_df["price"].astype(float)
    grouped = fills_df.groupby("symbol")["value"]
    pos = grouped.apply(lambda s: s[s > 0].sum())
    neg = grouped.apply(lambda s: s[s < 0].sum())
    pf = pos / neg.abs().replace(0, np.nan)
    table = pf.reset_index(name="pf")
    return table.sort_values("pf", ascending=False).head(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyprL Live Session Dashboard (Streamlit).")
    parser.add_argument("--session", required=True, help="Path to data/live/sessions/<id>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_session_frames(args.session)
    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Install streamlit to run this dashboard.") from exc

    st.set_page_config(page_title="HyprL Live Dashboard", layout="wide")
    st.title("HyprL Live Session Dashboard")
    if st.button("Reload"):
        st.experimental_rerun()

    equity_df = frames["equity"]
    metrics = _compute_equity_metrics(equity_df if isinstance(equity_df, pd.DataFrame) else pd.DataFrame())
    cols = st.columns(3)
    cols[0].metric("Profit Factor", f"{metrics['pf']:.2f}" if metrics["pf"] == metrics["pf"] else "n/a")
    cols[1].metric("Sharpe", f"{metrics['sharpe']:.2f}" if metrics["sharpe"] == metrics["sharpe"] else "n/a")
    cols[2].metric("Max DD", f"{metrics['max_dd']:.2%}" if metrics["max_dd"] == metrics["max_dd"] else "n/a")

    st.subheader("Equity Curve")
    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
        chart_df = equity_df.copy()
        chart_df["datetime"] = pd.to_datetime(chart_df["ts"], unit="s")
        chart_df = chart_df.set_index("datetime")
        st.line_chart(chart_df["equity"])
    else:
        st.info("No equity data yet.")

    fills_df = frames["fills"]
    top_symbols_df = _top_symbols(fills_df if isinstance(fills_df, pd.DataFrame) else pd.DataFrame())
    st.subheader("Top Symbols (PF proxy)")
    st.table(top_symbols_df)

    predictions_df = frames["predictions"]
    st.subheader("Probability Scatter")
    if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
        scatter_df = predictions_df.dropna(subset=["prob_up", "threshold"])
        if not scatter_df.empty:
            st.vega_lite_chart(
                scatter_df,
                {
                    "mark": {"type": "circle", "tooltip": True},
                    "encoding": {
                        "x": {"field": "prob_up", "type": "quantitative"},
                        "y": {"field": "threshold", "type": "quantitative"},
                        "color": {"field": "reason", "type": "nominal"},
                    },
                },
            )
        else:
            st.info("Scatter requires probability data.")
    else:
        st.info("No prediction data yet.")

    st.subheader("Event Feed")
    if isinstance(predictions_df, pd.DataFrame) and "event" in predictions_df:
        feed = predictions_df[predictions_df["event"].isin(["resume", "kill_switch", "tuner", "oco_close"])]
        st.dataframe(feed.tail(20))
    else:
        st.info("No event data available.")


if __name__ == "__main__":
    main()
