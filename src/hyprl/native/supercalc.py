from __future__ import annotations

from typing import Mapping, Sequence

import math
from pathlib import Path

try:  # pragma: no cover - optional dependency imported lazily at runtime
    import polars as pl
except ImportError:  # pragma: no cover - polars is required for the wrapper to operate
    pl = None  # type: ignore

try:  # pragma: no cover - pandas only needed for convenience conversions
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

import numpy as np

from hyprl.backtest.runner import BacktestConfig, BacktestResult

try:  # pragma: no cover - hyprl_supercalc may be absent on some hosts
    import hyprl_supercalc as _native
except ImportError:  # pragma: no cover
    _native = None  # type: ignore

_REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "volume")
_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
_MS_IN_YEAR = 365.25 * 24 * 3600 * 1000.0


class NativeEngineUnavailable(RuntimeError):
    pass


def native_available() -> bool:
    """Return True when the compiled hyprl_supercalc module can be imported."""

    return _native is not None


def _ensure_polars_dataframe(frame: pl.DataFrame | "pd.DataFrame") -> pl.DataFrame:
    if pl is None:  # pragma: no cover - enforced during CI via optional dependency checks
        raise NativeEngineUnavailable("polars is required to use the native engine wrapper")

    if isinstance(frame, pl.DataFrame):
        df = frame.clone()
    elif pd is not None and isinstance(frame, pd.DataFrame):
        pdf = frame.copy()
        if "ts" not in pdf.columns:
            ts_values = _timestamp_series_from_index(pdf.index)
            pdf = pdf.assign(ts=ts_values)
        pdf = pdf.reset_index(drop=True)
        df = pl.from_pandas(pdf)
    else:
        raise TypeError("df must be a polars.DataFrame or pandas.DataFrame")

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename({"timestamp": "ts"})

    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Native engine DataFrame missing required columns: {missing}")

    df = df.sort("ts")
    df = df.with_columns(
        pl.col("ts").cast(pl.Int64),
        *(pl.col(col).cast(pl.Float64) for col in _PRICE_COLUMNS),
    )
    return df.select(_REQUIRED_COLUMNS)


def _timestamp_series_from_index(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return (index.view("int64") // 1_000_000).astype(np.int64)
    return np.arange(len(index), dtype=np.int64)


def _normalize_signal(signal: Sequence[float], length: int) -> list[float]:
    if len(signal) != length:
        raise ValueError(f"Signal length {len(signal)} must match dataframe height {length}")
    return [float(value) for value in signal]


def _collect_label(cfg: BacktestConfig) -> str:
    label = getattr(cfg.label, "mode", None)
    if isinstance(label, str):
        return label
    return cfg.ticker


def _build_trade_rows(
    report_trades: list[dict[str, object]],
    df: pl.DataFrame,
    cfg: BacktestConfig,
    *,
    strategy_id: str | None,
    strategy_label: str | None,
    session_id: str | None,
    source_type: str = "backtest",
) -> list[dict[str, object]]:
    if not report_trades:
        return []
    def _normalize_exit_reason(reason: str) -> str:
        mapping = {
            "stop_or_take": "stop_loss",
            "stop_loss": "stop_loss",
            "take_profit": "take_profit",
            "trailing_stop": "trailing_stop",
            "end_of_data": "time_exit",
        }
        return mapping.get(reason, reason or "unknown")
    ts_series = df["ts"].to_list()
    initial_balance = float(cfg.initial_balance or 1.0)
    equity = initial_balance
    symbol = cfg.ticker.upper()
    rows: list[dict[str, object]] = []
    for trade in report_trades:
        reason_raw = str(trade.get("exit_reason", "unknown"))
        if reason_raw == "entry":
            continue
        entry_idx = int(trade.get("entry_idx", 0))
        exit_idx = int(trade.get("exit_idx", entry_idx))
        entry_ts_raw = ts_series[entry_idx] if entry_idx < len(ts_series) else ts_series[0]
        exit_ts_raw = ts_series[exit_idx] if exit_idx < len(ts_series) else ts_series[-1]
        entry_ts = pd.to_datetime(entry_ts_raw, unit="ms", utc=True)
        exit_ts = pd.to_datetime(exit_ts_raw, unit="ms", utc=True)
        direction = str(trade.get("direction", "flat")).upper()
        side = "LONG" if direction == "LONG" else "SHORT" if direction == "SHORT" else "FLAT"
        entry_price = float(trade.get("entry_price", 0.0))
        exit_price = float(trade.get("exit_price", 0.0))
        pnl_pct = float(trade.get("return_pct", 0.0))
        if not np.isfinite(pnl_pct):
            pnl_pct = 0.0
        pnl = equity * pnl_pct
        equity += pnl
        exit_reason = _normalize_exit_reason(reason_raw)
        rows.append(
            {
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "symbol": symbol,
                "side": side,
                "qty": 1.0,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "strategy_id": strategy_id or "",
                "strategy_label": strategy_label or "",
                "source_type": source_type,
                "session_id": session_id or "",
            }
        )
    return rows


def _config_to_native_dict(
    cfg: BacktestConfig,
    *,
    use_atr_position_sizing: bool | None = None,
    label_override: str | None = None,
) -> dict[str, object]:
    risk_cfg = cfg.risk
    risk_pct = float(getattr(risk_cfg, "risk_pct", 0.01))
    atr_mult_stop = float(getattr(risk_cfg, "atr_multiplier", 2.0))
    reward_multiple = float(getattr(risk_cfg, "reward_multiple", 2.0))
    atr_mult_tp = atr_mult_stop * reward_multiple
    allow_short = bool(cfg.short_threshold > 0.0)
    trailing_activation = getattr(risk_cfg, "trailing_stop_activation", 0.0)
    trailing_distance = getattr(risk_cfg, "trailing_stop_distance", 0.0)
    return {
        "risk_pct": risk_pct,
        "commission_pct": float(cfg.commission_pct),
        "slippage_pct": float(cfg.slippage_pct),
        "max_leverage": 1.0,
        "params": [],
        "allow_short": allow_short,
        "label": label_override if label_override is not None else _collect_label(cfg),
        "atr_window": int(cfg.atr_window),
        "atr_mult_stop": atr_mult_stop,
        "atr_mult_tp": atr_mult_tp,
        "use_atr_position_sizing": bool(
            risk_pct > 0.0 if use_atr_position_sizing is None else use_atr_position_sizing
        ),
        "trailing_activation_r": float(0.0 if trailing_activation is None else trailing_activation),
        "trailing_distance_r": float(0.0 if trailing_distance is None else trailing_distance),
    }


def _normalize_constraints(constraints: Mapping[str, float] | None) -> dict[str, float]:
    base = {
        "min_trades": 50,
        "min_profit_factor": 1.2,
        "min_sharpe": 0.8,
        "max_drawdown": 0.35,
        "max_risk_of_ruin": 0.1,
        "min_expectancy": 0.0,
        "min_robustness": 0.0,
        "max_maxdd_p95": 0.35,
        "min_pnl_p05": float("-inf"),
        "min_pnl_p50": float("-inf"),
        "min_pnl_p95": float("-inf"),
    }
    if not constraints:
        return base
    payload = dict(base)
    for key, value in constraints.items():
        if key not in payload or value is None:
            continue
        payload[key] = float(value)
    return payload


def _equity_curve_values(points: Sequence[dict[str, float]], scale: float) -> list[float]:
    curve: list[float] = []
    for point in points:
        equity = float(point.get("equity", 0.0))
        curve.append(equity * scale)
    return curve


def _equity_returns(equity_curve: Sequence[float]) -> list[float]:
    returns: list[float] = []
    for prev, curr in zip(equity_curve, equity_curve[1:]):
        if prev > 0.0:
            returns.append(curr / prev - 1.0)
        else:
            returns.append(0.0)
    return returns


def _benchmark_stats(df: pl.DataFrame, initial_balance: float) -> tuple[float, float, float | None]:
    if initial_balance <= 0:
        return float(initial_balance), 0.0, None
    close_series = df.get_column("close")
    ts_series = df.get_column("ts")
    first_close = float(close_series[0])
    last_close = float(close_series[len(close_series) - 1])
    if first_close <= 0:
        return float(initial_balance), 0.0, None
    shares = initial_balance / first_close
    final_balance = shares * last_close
    percent_return = (final_balance / initial_balance - 1.0) * 100.0
    ts_start = float(ts_series[0])
    ts_end = float(ts_series[len(ts_series) - 1])
    years = max((ts_end - ts_start) / _MS_IN_YEAR, 1e-9)
    cagr = None
    if final_balance > 0:
        try:
            cagr = (final_balance / initial_balance) ** (1.0 / years) - 1.0
        except OverflowError:
            cagr = None
    return final_balance, percent_return, cagr


def _annualized_volatility(returns: Sequence[float], ts_start: float, ts_end: float) -> float | None:
    if len(returns) < 2:
        return None
    duration_years = max((ts_end - ts_start) / _MS_IN_YEAR, 1e-9)
    bars_per_year = len(returns) / duration_years if duration_years > 0 else 0.0
    if bars_per_year <= 0:
        return None
    arr = np.array(returns, dtype=float)
    if arr.std(ddof=1) == 0:
        return None
    return float(arr.std(ddof=1) * math.sqrt(bars_per_year))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _build_backtest_result(
    report: dict[str, object],
    cfg: BacktestConfig,
    df: pl.DataFrame,
    *,
    strategy_id: str | None = None,
    strategy_label: str | None = None,
    session_id: str | None = None,
    source_type: str = "backtest",
    export_trades_path: str | Path | None = None,
) -> BacktestResult:
    metrics = report.get("metrics", {})
    equity_points: list[dict[str, float]] = list(report.get("equity_curve", []))
    if not equity_points:
        raise RuntimeError("Native engine returned an empty equity curve")
    initial_balance = float(cfg.initial_balance or 1.0)
    scale = initial_balance if initial_balance > 0 else 1.0
    equity_curve = _equity_curve_values(equity_points, scale)
    ts_start = float(equity_points[0]["ts"])
    ts_end = float(equity_points[-1]["ts"])
    returns = _equity_returns(equity_curve)
    benchmark_final, benchmark_return_pct, benchmark_cagr = _benchmark_stats(df, initial_balance)
    sharpe = float(metrics.get("sharpe", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    expectancy = float(metrics.get("expectancy", 0.0))
    annualized_return = float(metrics.get("cagr", 0.0))
    annualized_volatility = _annualized_volatility(returns, ts_start, ts_end)
    sortino_metric = _optional_float(metrics.get("sortino"))
    risk_of_ruin_metric = _optional_float(metrics.get("risk_of_ruin"))
    maxdd_p95_metric = _optional_float(metrics.get("maxdd_p95"))
    maxdd_p05_metric = _optional_float(metrics.get("maxdd_p05"))
    pnl_p05_metric = _optional_float(metrics.get("pnl_p05"))
    pnl_p50_metric = _optional_float(metrics.get("pnl_p50"))
    pnl_p95_metric = _optional_float(metrics.get("pnl_p95"))
    robustness_metric = _optional_float(metrics.get("robustness_score"))

    trade_rows = _build_trade_rows(
        list(report.get("trades", [])),
        df,
        cfg,
        strategy_id=strategy_id,
        strategy_label=strategy_label,
        session_id=session_id,
        source_type=source_type,
    )
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    long_pnl = 0.0
    short_pnl = 0.0
    for trade in trade_rows:
        side = str(trade.get("side", "")).upper()
        pnl_val = float(trade.get("pnl", 0.0))
        if side == "LONG":
            long_trades += 1
            long_pnl += pnl_val
            if pnl_val > 0:
                long_wins += 1
        elif side == "SHORT":
            short_trades += 1
            short_pnl += pnl_val
            if pnl_val > 0:
                short_wins += 1
    long_win_rate = float(long_wins) / long_trades if long_trades > 0 else 0.0
    short_win_rate = float(short_wins) / short_trades if short_trades > 0 else 0.0

    if export_trades_path and trade_rows:
        out_path = Path(export_trades_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trade_rows).to_csv(out_path, index=False)

    return BacktestResult(
        final_balance=float(equity_curve[-1]),
        equity_curve=[float(value) for value in equity_curve],
        n_trades=int(report.get("n_trades", 0)),
        win_rate=float(metrics.get("win_rate", 0.0)),
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        trades=trade_rows,
        benchmark_final_balance=float(benchmark_final),
        benchmark_return=float(benchmark_return_pct),
        annualized_return=annualized_return,
        annualized_benchmark_return=float(benchmark_cagr) if benchmark_cagr is not None else None,
        annualized_volatility=annualized_volatility,
        sortino_ratio=sortino_metric,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_r_multiple=0.0,
        avg_expected_pnl=0.0,
        long_trades=long_trades,
        short_trades=short_trades,
        long_win_rate=long_win_rate,
        short_win_rate=short_win_rate,
        long_total_pnl=long_pnl,
        short_total_pnl=short_pnl,
        brier_score=None,
        log_loss=None,
        final_risk_profile=getattr(cfg, "risk_profile", "native"),
        final_long_threshold=float(cfg.long_threshold),
        final_short_threshold=float(cfg.short_threshold),
        adaptive_profile_changes=0,
        regime_usage={},
        regime_transitions=[],
        exit_reason_counts={},
        sentiment_stats={"trades_in_fear": 0, "trades_in_greed": 0},
        trade_returns=[float(value) for value in returns],
        risk_of_ruin=risk_of_ruin_metric,
        maxdd_p95=maxdd_p95_metric,
        maxdd_p05=maxdd_p05_metric,
        pnl_p05=pnl_p05_metric,
        pnl_p50=pnl_p50_metric,
        pnl_p95=pnl_p95_metric,
        robustness_score=robustness_metric,
        native_metrics=dict(metrics),
    )


def run_backtest_native(
    df: pl.DataFrame | "pd.DataFrame",
    signal: Sequence[float],
    cfg: BacktestConfig,
    *,
    use_atr_position_sizing: bool | None = None,
    require_native: bool = True,
    export_trades_path: str | Path | None = None,
    strategy_id: str | None = None,
    strategy_label: str | None = None,
    session_id: str | None = None,
    source_type: str = "backtest",
) -> BacktestResult:
    """Run the Rust native engine for a single configuration and return a BacktestResult."""

    if require_native and not native_available():
        raise NativeEngineUnavailable(
            "hyprl_supercalc is not installed. Build it via scripts/build_supercalc.sh before using the native engine."
        )
    if not native_available():
        raise NativeEngineUnavailable("hyprl_supercalc is not installed.")
    polars_df = _ensure_polars_dataframe(df)
    signal_vec = _normalize_signal(signal, polars_df.height)
    config_dict = _config_to_native_dict(cfg, use_atr_position_sizing=use_atr_position_sizing)
    report = _native.run_backtest_py(polars_df, signal_vec, config_dict)  # type: ignore[attr-defined]
    return _build_backtest_result(
        report,
        cfg,
        polars_df,
        strategy_id=strategy_id,
        strategy_label=strategy_label,
        session_id=session_id,
        source_type=source_type,
        export_trades_path=export_trades_path,
    )


def run_native_search_batch(
    df: pl.DataFrame | "pd.DataFrame",
    signal: Sequence[float],
    cfgs: Sequence[BacktestConfig],
    constraints: Mapping[str, float] | None,
    top_k: int,
    *,
    labels: Sequence[str] | None = None,
    use_atr_position_sizing: bool | None = None,
):
    """Run the native search kernel on a batch of configs sharing the same dataframe + signal."""

    if not native_available():
        raise NativeEngineUnavailable(
            "hyprl_supercalc is not installed; build it via scripts/build_supercalc.sh to enable native search."
        )
    if not cfgs:
        return []
    polars_df = _ensure_polars_dataframe(df)
    signal_vec = _normalize_signal(signal, polars_df.height)
    if labels is not None and len(labels) != len(cfgs):
        raise ValueError("labels length must match cfgs length")
    config_dicts = []
    for idx, cfg in enumerate(cfgs):
        label_override = labels[idx] if labels is not None else None
        config_dicts.append(
            _config_to_native_dict(
                cfg,
                use_atr_position_sizing=use_atr_position_sizing,
                label_override=label_override,
            )
        )
    constraint_dict = _normalize_constraints(constraints)
    max_k = max(int(top_k), 0)
    if max_k == 0:
        return []
    reports = _native.run_native_search_py(  # type: ignore[attr-defined]
        polars_df,
        signal_vec,
        config_dicts,
        constraint_dict,
        max_k,
    )
    return reports


def run_backtest_native_batch(
    df: pl.DataFrame | "pd.DataFrame",
    signal: Sequence[float],
    cfgs: Sequence[BacktestConfig],
    *,
    use_atr_position_sizing: bool | None = None,
) -> list[BacktestResult]:
    """Run the native engine for a batch of BacktestConfig objects sharing the same dataframe + signal."""

    if not native_available():
        raise NativeEngineUnavailable("hyprl_supercalc module unavailable; cannot run native batch")
    if not cfgs:
        return []
    polars_df = _ensure_polars_dataframe(df)
    signal_vec = _normalize_signal(signal, polars_df.height)
    config_dicts = [
        _config_to_native_dict(cfg, use_atr_position_sizing=use_atr_position_sizing)
        for cfg in cfgs
    ]
    reports = _native.run_batch_backtest_py(polars_df, signal_vec, config_dicts)  # type: ignore[attr-defined]
    if len(reports) != len(cfgs):
        raise RuntimeError("Native engine batch size mismatch between configs and reports")
    return [
        _build_backtest_result(rep, cfg, polars_df)
        for rep, cfg in zip(reports, cfgs)
    ]
    def _normalize_exit_reason(reason: str) -> str:
        mapping = {
            "stop_or_take": "stop_loss",
            "stop_loss": "stop_loss",
            "take_profit": "take_profit",
            "trailing_stop": "trailing_stop",
            "end_of_data": "time_exit",
        }
        return mapping.get(reason, reason or "unknown")
