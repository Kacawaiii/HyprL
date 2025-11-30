#!/usr/bin/env python
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import streamlit as st

from hyprl.backtest.runner import BacktestConfig, BacktestResult, run_backtest
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.data.market import MarketDataFetcher
from hyprl.risk.manager import RiskConfig
from hyprl.snapshots import save_snapshot, make_backup_zip


def _prices_for_config(config: BacktestConfig) -> pd.DataFrame:
    fetcher = MarketDataFetcher(config.ticker)
    period = config.period if not (config.start or config.end) else None
    prices = fetcher.get_prices(
        interval=config.interval,
        period=period,
        start=config.start,
        end=config.end,
    )
    return prices.sort_index()


def _equity_series(result: BacktestResult, prices: pd.DataFrame) -> pd.Series:
    if not result.trades:
        times = [prices.index[0]]
    else:
        times = [prices.index[0]]
        for trade in result.trades:
            times.append(trade.exit_timestamp)
    equity = pd.Series(result.equity_curve, index=pd.Index(times[: len(result.equity_curve)], name="timestamp"))
    return equity.sort_index()


def _store_replay_state(prices: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series, config: BacktestConfig, result: BacktestResult):
    st.session_state["replay_prices"] = prices
    st.session_state["replay_trades"] = trades
    st.session_state["replay_equity"] = equity
    st.session_state["replay_config"] = config
    st.session_state["replay_result"] = result
    st.session_state["replay_index"] = len(equity) - 1
    st.session_state["replay_zip"] = None
    st.session_state["replay_trades_path"] = None


def _display_metrics(result: BacktestResult, initial_balance: float) -> None:
    st.subheader("Strategy Metrics")
    strategy_return = (result.final_balance / initial_balance - 1.0) * 100.0 if initial_balance > 0 else 0.0
    st.write(
        f"Strategy return: {strategy_return:.2f}% "
        f"(Annualized: {(result.annualized_return or 0.0) * 100.0:.2f}%)"
    )
    st.write(
        f"Benchmark return: {result.benchmark_return:.2f}% "
        f"(Annualized: {(result.annualized_benchmark_return or 0.0) * 100.0:.2f}%)"
    )
    st.write(f"Sharpe: {result.sharpe_ratio or 'n/a'} | Sortino: {result.sortino_ratio or 'n/a'}")
    st.write(f"Max DD: {result.max_drawdown * 100.0:.2f}% | Annualized vol: {(result.annualized_volatility or 0.0) * 100.0:.2f}%")
    st.write(f"Profit factor: {result.profit_factor or 'n/a'} | Expectancy: {result.expectancy:.2f}")
    st.write(
        f"Trades: {result.n_trades} (Win rate {result.win_rate * 100.0:.2f}%) | "
        f"Long {result.long_trades} ({result.long_win_rate * 100.0:.2f}%) | "
        f"Short {result.short_trades} ({result.short_win_rate * 100.0:.2f}%)"
    )
    st.write(f"Average R-multiple: {result.avg_r_multiple:.3f} | Avg expected pnl: {result.avg_expected_pnl:.2f}")
    st.write(
        f"Final regime: {result.final_risk_profile or 'n/a'} | "
        f"Adaptive changes: {result.adaptive_profile_changes} | "
        f"Long threshold: {result.final_long_threshold:.2f} | Short threshold: {result.final_short_threshold:.2f}"
    )
    if result.regime_usage:
        total = sum(result.regime_usage.values())
        usage_rows = [
            {"regime": name, "trades": count, "pct_trades": (count / total * 100.0) if total else 0.0}
            for name, count in sorted(result.regime_usage.items(), key=lambda item: (-item[1], item[0]))
        ]
        st.subheader("Regime usage")
        st.dataframe(pd.DataFrame(usage_rows))
    if result.regime_transitions:
        st.write(
            "Transitions: "
            + ", ".join(f"{item['regime']}@{item['trade']}" for item in result.regime_transitions)
        )


def _run_simulation_tab():
    st.sidebar.subheader("Simulation parameters")
    ticker = st.sidebar.text_input("Ticker", "NVDA")
    period = st.sidebar.text_input("Period (yfinance)", "1y")
    interval = st.sidebar.text_input("Interval", "1h")
    settings = load_ticker_settings(ticker, interval)
    if settings.get("tradable") is False:
        st.sidebar.warning(settings.get("note", "Preset marked non-tradable (no edge)."))
    adaptive_settings = get_adaptive_config(settings, None)
    initial_balance = st.sidebar.number_input("Initial balance", value=10_000.0, step=1_000.0)
    seed = st.sidebar.number_input("Seed", value=42, step=1)
    model_options = ["logistic", "random_forest"]
    default_model = settings.get("model_type", "logistic")
    if default_model not in model_options:
        default_model = "logistic"
    model_type = st.sidebar.selectbox("Model type", model_options, index=model_options.index(default_model))
    calibration_options = ["none", "platt", "isotonic"]
    default_calibration = settings.get("calibration", "none")
    if default_calibration not in calibration_options:
        default_calibration = "none"
    calibration = st.sidebar.selectbox(
        "Calibration", calibration_options, index=calibration_options.index(default_calibration)
    )
    profile_options = list((settings.get("risk_profiles") or {}).keys())
    default_profile = settings.get("default_risk_profile")
    if profile_options:
        if default_profile not in profile_options:
            default_profile = profile_options[0]
        risk_profile = st.sidebar.selectbox(
            "Risk profile (ticker presets)", profile_options, index=profile_options.index(default_profile)
        )
        risk_params = get_risk_settings(settings, risk_profile)
    else:
        risk_profile = None
        risk_params = get_risk_settings(settings, None)
    long_default = float(load_long_threshold(settings, default=0.55))
    long_threshold = st.sidebar.slider("Long threshold", 0.4, 0.9, long_default, 0.01)
    short_default = float(load_short_threshold(settings))
    short_threshold = st.sidebar.slider("Short threshold", 0.1, 0.6, short_default, 0.01)
    risk_pct = st.sidebar.number_input("Risk pct", value=float(risk_params["risk_pct"]), step=0.005, format="%.3f")
    atr_multiplier = st.sidebar.number_input(
        "ATR multiplier", value=float(risk_params["atr_multiplier"]), step=0.1, format="%.2f"
    )
    reward_multiple = st.sidebar.number_input(
        "Reward multiple", value=float(risk_params["reward_multiple"]), step=0.1, format="%.2f"
    )
    min_position_size = st.sidebar.number_input(
        "Min position size", value=int(risk_params["min_position_size"]), step=1
    )
    adaptive_enabled = st.sidebar.checkbox("Adaptive mode", value=adaptive_settings.enable)
    adaptive_lookback = st.sidebar.slider(
        "Adaptive lookback (trades)", min_value=5, max_value=100, value=int(adaptive_settings.lookback_trades), step=1
    )
    regime_options = list(adaptive_settings.regimes.keys())
    if regime_options:
        default_regime = (
            adaptive_settings.default_regime if adaptive_settings.default_regime in regime_options else regime_options[0]
        )
        adaptive_default = st.sidebar.selectbox(
            "Adaptive default regime", regime_options, index=regime_options.index(default_regime)
        )
    else:
        adaptive_default = adaptive_settings.default_regime
    adaptive_cfg = replace(
        adaptive_settings,
        enable=adaptive_enabled,
        lookback_trades=int(adaptive_lookback),
        default_regime=adaptive_default,
    )
    run_button = st.sidebar.button("Run simulation")

    if run_button:
        risk_cfg = RiskConfig(
            balance=initial_balance,
            risk_pct=risk_pct,
            atr_multiplier=atr_multiplier,
            reward_multiple=reward_multiple,
            min_position_size=int(min_position_size),
        )
        config = BacktestConfig(
            ticker=ticker,
            period=period,
            interval=interval,
            initial_balance=initial_balance,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            model_type=model_type,
            calibration=calibration,
            risk=risk_cfg,
            risk_profile=risk_profile or "normal",
            risk_profiles=settings.get("risk_profiles", {}),
            adaptive=adaptive_cfg,
            random_state=int(seed),
        )
        with st.spinner("Running backtest..."):
            result = run_backtest(config)
            prices = _prices_for_config(config)
            trades_df = pd.DataFrame([asdict(t) for t in result.trades])
            equity = _equity_series(result, prices)
        _store_replay_state(prices, trades_df, equity, config, result)
        st.success("Simulation completed.")
        _display_metrics(result, initial_balance)
        if not trades_df.empty:
            st.dataframe(trades_df.head())


def _build_equity_from_trades(trades_df: pd.DataFrame, initial_balance: float) -> pd.Series:
    trades_df = trades_df.sort_values("exit_timestamp")
    equity = [initial_balance]
    times = [trades_df["exit_timestamp"].iloc[0] if not trades_df.empty else pd.Timestamp.utcnow()]
    balance = initial_balance
    for _, row in trades_df.iterrows():
        balance += row["pnl"]
        equity.append(balance)
        times.append(row["exit_timestamp"])
    if len(times) != len(equity):
        times = times[: len(equity)]
    return pd.Series(equity, index=pd.Index(times, name="timestamp"))


def _run_replay_tab():
    st.subheader("Replay from trades CSV")
    uploaded = st.file_uploader("Trades CSV", type=["csv"])
    if uploaded is None:
        return
    ticker = st.text_input("Ticker for replay", "NVDA")
    period = st.text_input("Period (yfinance)", "1y")
    interval = st.text_input("Interval", "1h")
    initial_balance = st.number_input("Initial balance for replay", value=10_000.0, step=1_000.0)
    trades_df = pd.read_csv(uploaded)
    required_cols = {"entry_timestamp", "exit_timestamp"}
    if not required_cols.issubset(trades_df.columns):
        st.error("Trades CSV must contain 'entry_timestamp' and 'exit_timestamp' columns.")
        return
    trades_df["entry_timestamp"] = pd.to_datetime(trades_df["entry_timestamp"])
    trades_df["exit_timestamp"] = pd.to_datetime(trades_df["exit_timestamp"])
    if trades_df.empty:
        st.warning("Trades CSV is empty.")
        return
    fetcher = MarketDataFetcher(ticker)
    prices = fetcher.get_prices(interval=interval, period=period).sort_index()
    equity = _build_equity_from_trades(trades_df, initial_balance)
    _store_replay_state(prices, trades_df, equity, None, None)
    st.success("Replay data loaded.")
    st.dataframe(trades_df.head())


def _display_replay_controls():
    if "replay_equity" not in st.session_state:
        return
    equity = st.session_state["replay_equity"]
    prices = st.session_state["replay_prices"]
    trades_df = st.session_state["replay_trades"]
    max_step = len(equity) - 1
    index = st.slider("Replay position", 0, max_step, value=st.session_state.get("replay_index", max_step))
    st.session_state["replay_index"] = index
    current_time = equity.index[index]
    st.write(f"Current timestamp: {current_time}")
    equity_until = equity.iloc[: index + 1]
    price_until = prices.loc[: current_time]
    st.line_chart(equity_until, height=200)
    st.line_chart(price_until["close"], height=200)
    trades_until = trades_df[trades_df["exit_timestamp"] <= current_time]
    if not trades_until.empty:
        st.write("Trades executed up to current time:")
        st.dataframe(trades_until.tail())
    config = st.session_state.get("replay_config")
    result = st.session_state.get("replay_result")
    if config and result:
        export_dir = Path("backups/replay_gui")
        if st.button("Save snapshot for this run"):
            export_dir.mkdir(parents=True, exist_ok=True)
            trades_path = export_dir / f"trades_{config.ticker}_{config.period or 'custom'}_{config.model_type}.csv"
            trades_df.to_csv(trades_path, index=False)
            snapshot_dir = save_snapshot(config, result, trades_path=trades_path)
            zip_path = make_backup_zip(snapshot_dir)
            st.session_state["replay_zip"] = zip_path
            st.session_state["replay_trades_path"] = trades_path
            st.success(f"Snapshot saved to {snapshot_dir}")
        if st.session_state.get("replay_zip"):
            with open(st.session_state["replay_zip"], "rb") as f:
                st.download_button(
                    label="Download backup zip",
                    data=f,
                    file_name=Path(st.session_state["replay_zip"]).name,
                    mime="application/zip",
                )


def main() -> None:
    st.title("HyprL – Market Replay")
    tabs = st.tabs(["Simulate run", "Replay CSV", "Replay Viewer"])
    with tabs[0]:
        _run_simulation_tab()
    with tabs[1]:
        _run_replay_tab()
    with tabs[2]:
        _display_replay_controls()


if __name__ == "__main__":
    main()
