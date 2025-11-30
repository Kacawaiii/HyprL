#!/usr/bin/env python
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import streamlit as st

from hyprl.backtest.runner import BacktestConfig, run_backtest
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.risk.manager import RiskConfig
from hyprl.snapshots import save_snapshot, make_backup_zip


def _run_backtest(config: BacktestConfig):
    return run_backtest(config)


def main() -> None:
    st.title("HyprL – Backtest GUI")
    st.sidebar.header("Parameters")
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
    long_default = float(load_long_threshold(settings, default=0.55))
    long_threshold = st.sidebar.slider("Long threshold", 0.4, 0.9, long_default, 0.01)
    short_threshold_default = float(load_short_threshold(settings))
    short_threshold = st.sidebar.slider("Short threshold", 0.1, 0.6, short_threshold_default, 0.01)
    risk_profiles = list((settings.get("risk_profiles") or {}).keys())
    default_profile = settings.get("default_risk_profile")
    if risk_profiles:
        if default_profile not in risk_profiles:
            default_profile = risk_profiles[0]
        profile_index = risk_profiles.index(default_profile)
        risk_profile = st.sidebar.selectbox("Risk profile", risk_profiles, index=profile_index)
    else:
        risk_profile = None
    risk_params = get_risk_settings(settings, risk_profile)
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
    export_dir = Path(st.sidebar.text_input("Export directory", "backups/gui_runs"))
    run_button = st.sidebar.button("Run backtest")

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
            result = _run_backtest(config)
        st.success("Backtest completed.")

        st.subheader("Strategy Metrics")
        st.write(
            f"Strategy return: { (result.final_balance / initial_balance - 1.0) * 100.0:.2f}% "
            f"(Annualized: { (result.annualized_return or 0.0) * 100.0:.2f}%)"
        )
        st.write(
            f"Benchmark return: { result.benchmark_return:.2f}% "
            f"(Annualized: { (result.annualized_benchmark_return or 0.0) * 100.0:.2f}%)"
        )
        st.write(f"Alpha: { (result.benchmark_return and ( (result.final_balance / initial_balance - 1) * 100.0 - result.benchmark_return)) :.2f}%")
        st.write(f"Sharpe: {result.sharpe_ratio or 'n/a'} | Sortino: {result.sortino_ratio or 'n/a'}")
        st.write(f"Max DD: {result.max_drawdown * 100.0:.2f}% | Annualized vol: {(result.annualized_volatility or 0.0) * 100.0:.2f}%")
        st.write(f"Profit factor: {result.profit_factor or 'n/a'} | Expectancy: {result.expectancy:.2f}")
        st.write(
            f"Trades: {result.n_trades} (Win rate {result.win_rate * 100.0:.2f}%), "
            f"Long {result.long_trades} ({result.long_win_rate * 100.0:.2f}%), "
            f"Short {result.short_trades} ({result.short_win_rate * 100.0:.2f}%)"
        )
        st.write(f"Average R-multiple: {result.avg_r_multiple:.3f} | Avg expected pnl: {result.avg_expected_pnl:.2f}")
        st.write(
            f"Final regime: {result.final_risk_profile or 'n/a'} | "
            f"Adaptive changes: {result.adaptive_profile_changes} | "
            f"Long threshold: {result.final_long_threshold:.2f} | Short threshold: {result.final_short_threshold:.2f}"
        )
        if result.regime_usage:
            st.subheader("Regime usage")
            total = sum(result.regime_usage.values())
            usage_rows = [
                {"regime": name, "trades": count, "pct_trades": (count / total * 100.0) if total else 0.0}
                for name, count in sorted(result.regime_usage.items(), key=lambda item: (-item[1], item[0]))
            ]
            st.dataframe(pd.DataFrame(usage_rows))
        if result.regime_transitions:
            st.write(
                "Transitions: "
                + ", ".join(f"{item['regime']}@{item['trade']}" for item in result.regime_transitions)
            )

        trades_df = pd.DataFrame([asdict(trade) for trade in result.trades])
        if not trades_df.empty:
            st.subheader("Trades Preview")
            st.dataframe(trades_df.head())

        export_dir.mkdir(parents=True, exist_ok=True)
        trades_path = export_dir / f"trades_{ticker}_{period}_{model_type}.csv"
        trades_df.to_csv(trades_path, index=False)
        snapshot_dir = save_snapshot(config, result, trades_path=trades_path)
        zip_path = make_backup_zip(snapshot_dir)

        st.write(f"Snapshot saved to {snapshot_dir}")
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download backup zip",
                data=f,
                file_name=zip_path.name,
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
