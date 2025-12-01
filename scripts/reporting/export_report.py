#!/usr/bin/env python3
"""
Export a simple performance report (Markdown or HTML) from trade logs and weights.

Example:
    python scripts/reporting/export_report.py \
      --trade-logs path/to/trades_NVDA.csv path/to/trades_MSFT.csv \
      --weights NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00 \
      --initial-equity 10000 \
      --annualization 1638 \
      --output reports/report_asc_v2_W0.md \
      --format md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class SeriesMetrics:
    pf: float
    maxdd: float
    trades: int
    win_rate: float
    sharpe: float
    equity_end: float
    equity_start: float
    start_ts: datetime
    end_ts: datetime


def compute_metrics(df: pd.DataFrame, annualization: int, initial_equity: float) -> SeriesMetrics:
    """Compute per-series metrics from a trade dataframe.

    Expects at least `pnl` and `exit_timestamp` columns. Returns PF, MaxDD (%),
    trade count, win rate (%), Sharpe, equity start/end, and window bounds.
    """
    pnl = df["pnl"].astype(float)
    timestamps = pd.to_datetime(df["exit_timestamp"])
    equity = initial_equity + pnl.cumsum()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    dd = (equity.cummax() - equity) / equity.cummax()
    maxdd = float(dd.max() * 100) if not dd.empty else 0.0
    trades = len(df)
    win_rate = float((pnl > 0).mean() * 100) if trades else 0.0
    returns = equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std(ddof=0) * (annualization**0.5)) if not returns.empty and returns.std(ddof=0) > 0 else 0.0
    equity_end = float(equity.iloc[-1]) if not equity.empty else initial_equity
    equity_start = float(equity.iloc[0]) if not equity.empty else initial_equity
    start_ts = timestamps.min().to_pydatetime() if not timestamps.empty else datetime.min
    end_ts = timestamps.max().to_pydatetime() if not timestamps.empty else datetime.min
    return SeriesMetrics(
        pf=float(pf),
        maxdd=maxdd,
        trades=trades,
        win_rate=win_rate,
        sharpe=sharpe,
        equity_end=equity_end,
        equity_start=equity_start,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def load_trade_logs(trade_logs: Iterable[Path]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in trade_logs:
        if not path.is_file():
            raise SystemExit(f"Trade log not found: {path}")
        df = pd.read_csv(path)
        if "exit_timestamp" not in df.columns or "pnl" not in df.columns:
            raise SystemExit(f"Trade log {path} missing required columns.")
        # Infer ticker from filename: trades_<TICKER>_*.csv
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            ticker = parts[1].upper()
        else:
            ticker = stem.upper()
        out[ticker] = df
    return out


def portfolio_metrics(per_ticker: Dict[str, pd.DataFrame], weights: Dict[str, float], annualization: int, initial_equity: float) -> Tuple[SeriesMetrics, Dict[str, SeriesMetrics]]:
    per_stats: Dict[str, SeriesMetrics] = {}
    merged_events: List[pd.DataFrame] = []
    for ticker, df in per_ticker.items():
        w = weights.get(ticker, 0.0)
        stats = compute_metrics(df, annualization, initial_equity * w)
        per_stats[ticker] = stats
        scaled_pnl = df["pnl"].astype(float) * w
        merged_events.append(
            pd.DataFrame(
                {
                    "exit_timestamp": pd.to_datetime(df["exit_timestamp"]),
                    "pnl_scaled": scaled_pnl,
                }
            )
        )
    if not merged_events:
        empty = pd.DataFrame(columns=["exit_timestamp", "pnl_scaled"])
        merged_events = [empty]
    merged = pd.concat(merged_events, ignore_index=True).sort_values("exit_timestamp")
    merged["equity"] = initial_equity + merged["pnl_scaled"].cumsum()
    gross_profit = merged["pnl_scaled"][merged["pnl_scaled"] > 0].sum()
    gross_loss = -merged["pnl_scaled"][merged["pnl_scaled"] < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    dd = (merged["equity"].cummax() - merged["equity"]) / merged["equity"].cummax()
    maxdd = float(dd.max() * 100) if not dd.empty else 0.0
    trades = len(merged_events[0]) if len(per_ticker) == 1 else sum(len(df) for df in per_ticker.values())
    returns = merged["equity"].pct_change().dropna()
    sharpe = float(returns.mean() / returns.std(ddof=0) * (annualization**0.5)) if not returns.empty and returns.std(ddof=0) > 0 else 0.0
    equity_end = float(merged["equity"].iloc[-1]) if not merged.empty else initial_equity
    start_ts = merged["exit_timestamp"].min().to_pydatetime() if not merged.empty else datetime.min
    end_ts = merged["exit_timestamp"].max().to_pydatetime() if not merged.empty else datetime.min
    # Win rate portfolio not directly defined; approximate using summed positive pnl events vs total events
    wins = (merged["pnl_scaled"] > 0).sum()
    win_rate = float(wins / len(merged) * 100) if len(merged) else 0.0
    port_stats = SeriesMetrics(
        pf=float(pf),
        maxdd=maxdd,
        trades=trades,
        win_rate=win_rate,
        sharpe=sharpe,
        equity_end=equity_end,
        equity_start=initial_equity,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return port_stats, per_stats


def render_report(
    portfolio: SeriesMetrics,
    per_ticker: Dict[str, SeriesMetrics],
    weights: Dict[str, float],
    output_path: Path,
    fmt: str = "md",
    notes: Iterable[str] | None = None,
) -> None:
    date_str = datetime.now(timezone.utc).isoformat()
    start = portfolio.start_ts.date() if portfolio.start_ts != datetime.min else "N/A"
    end = portfolio.end_ts.date() if portfolio.end_ts != datetime.min else "N/A"
    extra_notes = list(notes or [])

    def fmt_line(line: str) -> str:
        return line

    lines: List[str] = []
    lines.append(f"# HyprL Ascendant v2 – Rapport de performance")
    lines.append(f"_Généré le {date_str}_")
    lines.append(f"Fenêtre: {start} → {end}")
    lines.append("")
    lines.append("## Résumé global")
    lines.append(f"- PF_portfolio: {portfolio.pf:.3f}")
    lines.append(f"- MaxDD_portfolio: {portfolio.maxdd:.2f}%")
    lines.append(f"- Sharpe_portfolio: {portfolio.sharpe:.2f}")
    lines.append(f"- Trades_total: {portfolio.trades}")
    lines.append(f"- Equity initiale: {portfolio.equity_start:.2f} | Equity finale: {portfolio.equity_end:.2f}")
    lines.append("")
    lines.append("## Détail par ticker")
    lines.append("| Ticker | Weight | PF | MaxDD% | Trades | Win% | Sharpe | Equity_end |")
    lines.append("|--------|--------|----|--------|--------|------|--------|------------|")
    for ticker, stats in per_ticker.items():
        w = weights.get(ticker, 0.0)
        lines.append(
            f"| {ticker} | {w:.4f} | {stats.pf:.3f} | {stats.maxdd:.2f} | {stats.trades} | "
            f"{stats.win_rate:.2f}% | {stats.sharpe:.2f} | {stats.equity_end:.2f} |"
        )
    lines.append("")
    lines.append("## Section métriques de risque")
    lines.append("- Kelly dynamique (borne par caps).")
    lines.append("- Caps typiques: max_total_risk_pct ~5%, max_ticker_risk_pct ~3%, max_group_risk_pct ~4%, max_positions ~5.")
    lines.append("- Guards: max_drawdown_pct, min_pf_live, max_consecutive_losses (voir configs Ascendant v2).")
    lines.append("- Gates visées: PF ≥ 1.5, MaxDD ≤ 15–20%, Sharpe ≥ 1.5 sur fenêtre roulante.")
    lines.append("")
    lines.append("## Notes / Limitations")
    lines.append("- Résultats issus de backtests/replays/live-paper selon la source des logs.")
    lines.append("- Les performances passées ne préjugent pas des performances futures.")
    for note in extra_notes:
        lines.append(f"- {note}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(fmt_line(l) for l in lines)
    if fmt.lower() == "html":
        try:
            import markdown
        except Exception:
            raise SystemExit("HTML output requires markdown package (pip install markdown).")
        html = markdown.markdown(content)
        output_path.write_text(html, encoding="utf-8")
    else:
        output_path.write_text(content, encoding="utf-8")


def parse_weights(weights_arg: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in weights_arg.split(","):
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid weight entry: {item}")
        ticker, val = item.split("=", 1)
        weights[ticker.strip().upper()] = float(val)
    return weights


def export_report(
    trade_logs: List[Path],
    weights: Dict[str, float],
    initial_equity: float,
    annualization: int,
    output_path: Path,
    fmt: str = "md",
) -> None:
    """Generate and write a performance report from trade logs.

    Args:
        trade_logs: Paths to trade CSVs (must contain exit_timestamp, pnl).
        weights: Mapping ticker -> weight (expected to roughly sum to 1).
        initial_equity: Starting equity for metrics.
        annualization: Periods per year for Sharpe scaling.
        output_path: Destination file (parent dirs created).
        fmt: "md" or "html".
    """
    if not trade_logs:
        raise SystemExit("At least one trade log is required.")
    fmt_lower = fmt.lower()
    if fmt_lower not in {"md", "html"}:
        raise SystemExit(f"Unsupported format '{fmt}'. Use 'md' or 'html'.")
    per_ticker_logs = load_trade_logs(trade_logs)
    if not per_ticker_logs:
        raise SystemExit("No trade logs loaded.")
    missing_weights = [ticker for ticker in per_ticker_logs if ticker not in weights]
    if missing_weights:
        missing = ", ".join(sorted(missing_weights))
        raise SystemExit(f"Missing weight for ticker(s): {missing}")
    notes: List[str] = []
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        warn = f"Weights sum to {weight_sum:.4f}, expected ~1.0."
        print(f"[WARN] {warn}")
        notes.append(warn)
    port_stats, per_stats = portfolio_metrics(per_ticker_logs, weights, annualization, initial_equity)
    render_report(portfolio=port_stats, per_ticker=per_stats, weights=weights, output_path=output_path, fmt=fmt_lower, notes=notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a performance report from trade logs.")
    parser.add_argument("--trade-logs", nargs="+", required=True, help="Trade CSVs (trades_<TICKER>_*.csv).")
    parser.add_argument("--weights", required=True, help="Comma-separated weights e.g. NVDA=0.30,MSFT=0.27,...")
    parser.add_argument("--initial-equity", type=float, required=True)
    parser.add_argument("--annualization", type=int, default=1638)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["md", "html"], default="md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)
    export_report(
        trade_logs=[Path(p) for p in args.trade_logs],
        weights=weights,
        initial_equity=args.initial_equity,
        annualization=args.annualization,
        output_path=args.output,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
