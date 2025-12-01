#!/usr/bin/env python3
from __future__ import annotations

"""
HyprL Control Panel (internal)

Objectifs :
- Superviser les stratégies et backtests HyprL (Python + Rust supercalc).
- Tester rapidement des réglages (thresholds, risk_pct, presets…).
- Visualiser equity curve, points d'entrée/sortie, et profil de risque.
- Préparer le futur branchement avec l'API/Discord sans dupliquer la logique.

Ce fichier est volontairement structuré en couches :
- Section 1 : Imports + constantes UI.
- Section 2 : Helpers HyprL (config, backtest, risk).
- Section 3 : Composants UI (onglets).
- Section 4 : main() + layout Streamlit.

Les fonctions marquées TODO sont parfaites pour Copilot.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- HyprL imports (ajuster si nécessaire en fonction de ton arbre réel) ---

from hyprl.backtest import runner
from hyprl.backtest.runner import BacktestConfig
from hyprl.native.supercalc import native_available, run_backtest_native
from hyprl.risk.manager import RiskConfig
from hyprl.risk.metrics import (
    bootstrap_equity_drawdowns,
    compute_risk_of_ruin,
)
from hyprl.supercalc import (
    _build_signal_series,
    prepare_supercalc_dataset,
)
from hyprl.search.configs import load_supersearch_presets


# ======================================================================
# 1. CONST UI / TYPES
# ======================================================================


DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "SPY"]
DEFAULT_INTERVALS: List[str] = ["1h", "1d"]
DEFAULT_PERIODS: List[str] = ["6mo", "1y", "3y"]


@dataclass
class ControlPanelSelection:
    ticker: str
    period: str
    interval: str
    preset_name: Optional[str]
    long_threshold: float
    short_threshold: float
    risk_pct: float
    initial_balance: float
    trend_filter: bool
    sentiment_min: float
    sentiment_max: float
    sentiment_regime: str
    bootstrap_runs: int
    seed: int
    trailing_stop_activation: Optional[float] = None
    trailing_stop_distance: Optional[float] = None


@dataclass
class MonteCarloSummary:
    horizon_trades: int
    runs: int
    prob_loss_any: float
    prob_loss_gt_threshold: float
    loss_threshold_pct: float
    return_p05: float
    return_p50: float
    return_p95: float
    maxdd_p05: float
    maxdd_p50: float
    maxdd_p95: float


# ======================================================================
# 2. HELPERS HYPERL : CONFIG, BACKTEST, RISK
# ======================================================================


def build_backtest_config(sel: ControlPanelSelection) -> BacktestConfig:
    """
    Construit un BacktestConfig à partir de la sélection UI.
    On reste volontairement simple : on ne gère ici que les options clés
    (thresholds, risk_pct, trend_filter, sentiment).
    """
    risk_cfg = RiskConfig(
        balance=sel.initial_balance,
        risk_pct=sel.risk_pct,
        atr_multiplier=2.0,  # TODO: exposer dans l'UI si besoin
        reward_multiple=2.0,  # TODO: exposer dans l'UI si besoin
        min_position_size=1,
        trailing_stop_activation=sel.trailing_stop_activation,
        trailing_stop_distance=sel.trailing_stop_distance,
    )

    cfg = BacktestConfig(
        ticker=sel.ticker,
        period=sel.period,
        interval=sel.interval,
        initial_balance=sel.initial_balance,
        long_threshold=sel.long_threshold,
        short_threshold=sel.short_threshold,
        risk=risk_cfg,
        commission_pct=0.0005,
        slippage_pct=0.0005,
        enable_trend_filter=sel.trend_filter,
        sentiment_min=sel.sentiment_min,
        sentiment_max=sel.sentiment_max,
        sentiment_regime=sel.sentiment_regime,
        random_state=sel.seed,
    )
    return cfg


@st.cache_data(show_spinner=True)
def load_presets_cached() -> Dict[str, Dict[str, Any]]:
    """Charge les presets supersearch depuis YAML (côté HyprL)."""
    return load_supersearch_presets()


@st.cache_data(show_spinner=True)
def load_supersearch_results(
    ticker: str,
    period: str,
    interval: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Charge les résultats de supersearch pour un ticker donné.
    Hypothèse: fichiers type docs/experiments/SUPERCALC_NATIVE_<TICKER>_<PERIOD>*.csv
    """
    base = Path("docs/experiments")
    if not base.exists():
        return pd.DataFrame()
    pattern = f"SUPERCALC_NATIVE_{ticker}_{period}*.csv"
    candidates = sorted(base.glob(pattern))
    if not candidates:
        return pd.DataFrame()

    df = pd.read_csv(candidates[-1])
    sort_cols = [
        col for col in ["robustness_score", "sharpe", "profit_factor"] if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] + [False] * (len(sort_cols) - 1))
    return df.head(top_n)


@st.cache_data(show_spinner=True)
def load_dataset_for_cfg(cfg: BacktestConfig) -> Any:
    """
    Charge le dataset supercalc pour un BacktestConfig donné.
    On renvoie l'objet dataset comme tel (SupercalcDataset).
    """
    return prepare_supercalc_dataset(cfg)


def run_backtest_with_native_or_python(
    cfg: BacktestConfig, bootstrap_runs: int
) -> Dict[str, Any]:
    """
    Exécute un backtest en préférant le moteur natif (Rust supercalc),
    avec fallback sur le runner Python si natif indisponible.

    Retourne un dict normalisé avec :
    - metrics -> métriques globales
    - equity_df -> DataFrame equity/time
    - trades_df -> DataFrame trades (TODO: à compléter selon ton format réel)
    """
    dataset = load_dataset_for_cfg(cfg)
    signal, _, _ = _build_signal_series(dataset, cfg)
    position_df = _build_position_series_from_signal(dataset, signal)
    python_stats = None

    if native_available():
        result = run_backtest_native(dataset.prices, signal, cfg)
        # Hypothèse : résultat avec attributs final_balance, equity_curve, native_metrics…
        native_metrics = result.native_metrics or {}

        # Construit equity_df à partir de result.equity_curve (liste d'objets ou tuples)
        equity_df = _equity_curve_to_df(result.equity_curve)

        metrics = {
            "engine": "native",
            "final_balance": result.final_balance,
            "profit_factor": result.profit_factor,
            "sharpe": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown * 100.0,
            "expectancy": result.expectancy,
            "win_rate": result.win_rate,
            "risk_of_ruin": result.risk_of_ruin,
            "pnl_p05": native_metrics.get("pnl_p05"),
            "pnl_p95": native_metrics.get("pnl_p95"),
            "maxdd_p95": native_metrics.get("maxdd_p95"),
            "robustness_score": native_metrics.get("robustness_score"),
        }
        trades_df = _extract_trades_df_from_result(result)
    else:
        stats = runner.simulate_from_dataset(dataset, cfg)
        python_stats = stats
        equity_df = _equity_history_to_df(stats.equity_history)
        metrics = _compute_python_metrics_with_bootstrap(
            stats, cfg, bootstrap_runs=bootstrap_runs
        )
        metrics["engine"] = "python"
        trades_df = _extract_trades_df_from_stats(stats)

    return {
        "metrics": metrics,
        "equity_df": equity_df,
        "trades_df": trades_df,
        "dataset": dataset,
        "config": cfg,
        "position_df": position_df,
        "python_stats": python_stats,
    }


def _equity_curve_to_df(equity_curve: Any) -> pd.DataFrame:
    """
    Convertit l'equity_curve natif en DataFrame.
    TODO: adapter en fonction de la vraie structure EquityPoint.
    """
    if not equity_curve:
        return pd.DataFrame(columns=["ts", "equity"])

    # Hypothèse : equity_curve = [{"ts": int_ms, "equity": float}, ...]
    records = [
        {"ts": pt["ts"], "equity": pt["equity"]} for pt in equity_curve  # type: ignore[index]
    ]
    df = pd.DataFrame(records)
    # TODO: convertir ts (ms) en datetime si besoin
    return df


def _equity_history_to_df(equity_history: List[Tuple[Any, float]]) -> pd.DataFrame:
    if not equity_history:
        return pd.DataFrame(columns=["ts", "equity"])
    ts, eq = zip(*equity_history)
    return pd.DataFrame({"ts": ts, "equity": eq})


def _extract_trades_df_from_result(result: Any) -> pd.DataFrame:
    """
    Extraction des trades depuis le résultat natif.
    TODO: implémenter en fonction de la structure reale (result.trades ?).
    """
    # Placeholder : Copilot pourra remplir avec le bon mapping.
    return pd.DataFrame()


def _extract_trades_df_from_stats(stats: Any) -> pd.DataFrame:
    """
    Extraction des trades depuis StrategyStats Python.
    TODO: implémenter à partir de stats.trades ou trades_export si disponible.
    """
    return pd.DataFrame()
def _build_position_series_from_signal(dataset: Any, signal: np.ndarray) -> pd.DataFrame:
    """Approche simplifiée : long=1 si signal >=0, sinon 0 (TODO: gérer short)."""
    if signal is None or len(signal) == 0:
        return pd.DataFrame(columns=["ts", "position"])
    if hasattr(dataset, "prices") and hasattr(dataset.prices, "index"):
        ts = dataset.prices.index
    else:
        ts = np.arange(len(signal))
    pos = np.where(signal >= 0.0, 1.0, 0.0)
    ts_aligned = ts[: len(pos)]
    return pd.DataFrame({"ts": ts_aligned, "position": pos})


def _compute_python_metrics_with_bootstrap(
    stats: Any, cfg: BacktestConfig, bootstrap_runs: int
) -> Dict[str, Any]:
    trade_returns = stats.trade_returns or []
    equity_vol = float(
        np.std([val for _, val in stats.equity_history], ddof=1)
    ) if stats.equity_history else 0.0

    risk_per_trade = max(
        1e-6, float((cfg.initial_balance or 1.0) * cfg.risk.risk_pct)
    )
    ror = compute_risk_of_ruin(
        trade_returns, cfg.initial_balance or 1.0, risk_per_trade
    )
    _, _, boot = bootstrap_equity_drawdowns(
        trade_returns, n_runs=bootstrap_runs, seed=cfg.random_state
    )

    total_return = (stats.final_balance / (cfg.initial_balance or 1.0)) - 1.0

    # Tu peux aussi réutiliser _robustness_from_python_metrics que tu as
    # déjà dans ton script de comparaison Python vs natif si tu veux converger.
    robustness_score = 0.5  # TODO: calculer à partir des métriques python

    return {
        "final_balance": stats.final_balance,
        "profit_factor": stats.profit_factor,
        "sharpe": stats.sharpe_ratio,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "expectancy": stats.expectancy,
        "win_rate": stats.win_rate,
        "risk_of_ruin": ror,
        "pnl_p05": boot.pnl_p05,
        "pnl_p95": boot.pnl_p95,
        "maxdd_p95": boot.maxdd_p95 * 100.0,
        "equity_vol": equity_vol,
        "robustness_score": robustness_score,
    }


def compute_risk_profile_from_metrics(
    metrics: Dict[str, Any], initial_balance: float
) -> Dict[str, Any]:
    """
    Prépare un petit résumé "lisible" de risque pour l'onglet Risk Lab.
    """
    strat_return_pct = 100.0 * (
        (metrics.get("final_balance", initial_balance) / initial_balance) - 1.0
    )
    return {
        "return_pct": strat_return_pct,
        "profit_factor": metrics.get("profit_factor"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "risk_of_ruin": metrics.get("risk_of_ruin"),
        "robustness_score": metrics.get("robustness_score"),
    }


def run_monte_carlo_from_stats(
    stats: runner.SimulationStats,
    cfg: BacktestConfig,
    horizon_trades: int,
    n_runs: int,
    loss_threshold_pct: float,
    seed: Optional[int] = None,
) -> MonteCarloSummary:
    rng = np.random.default_rng(seed or cfg.random_state or 42)
    trade_returns = np.asarray(stats.trade_returns or [], dtype=float)
    if trade_returns.size == 0:
        nan = float("nan")
        return MonteCarloSummary(
            horizon_trades=horizon_trades,
            runs=n_runs,
            prob_loss_any=nan,
            prob_loss_gt_threshold=nan,
            loss_threshold_pct=loss_threshold_pct,
            return_p05=nan,
            return_p50=nan,
            return_p95=nan,
            maxdd_p05=nan,
            maxdd_p50=nan,
            maxdd_p95=nan,
        )

    final_returns: list[float] = []
    max_dds: list[float] = []
    threshold_ratio = -abs(loss_threshold_pct) / 100.0

    for _ in range(n_runs):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for _ in range(horizon_trades):
            r = float(rng.choice(trade_returns))
            equity *= 1.0 + r
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown
        final_returns.append(equity - 1.0)
        max_dds.append(max_dd)

    final_arr = np.asarray(final_returns)
    maxdd_arr = np.asarray(max_dds)

    prob_loss_any = float(np.mean(final_arr < 0.0))
    prob_loss_gt_threshold = float(np.mean(final_arr <= threshold_ratio))

    return MonteCarloSummary(
        horizon_trades=horizon_trades,
        runs=n_runs,
        prob_loss_any=prob_loss_any,
        prob_loss_gt_threshold=prob_loss_gt_threshold,
        loss_threshold_pct=loss_threshold_pct,
        return_p05=float(np.percentile(final_arr, 5) * 100.0),
        return_p50=float(np.percentile(final_arr, 50) * 100.0),
        return_p95=float(np.percentile(final_arr, 95) * 100.0),
        maxdd_p05=float(np.percentile(maxdd_arr, 5) * 100.0),
        maxdd_p50=float(np.percentile(maxdd_arr, 50) * 100.0),
        maxdd_p95=float(np.percentile(maxdd_arr, 95) * 100.0),
    )


# ======================================================================
# 3. COMPOSANTS UI (ONGLETS)
# ======================================================================


def render_overview_tab(sel: ControlPanelSelection, result: Dict[str, Any]) -> None:
    st.header("📊 HyprL Overview")

    st.write(
        f"Ticker **{sel.ticker}**, {sel.period} @ {sel.interval} – "
        f"Preset: `{sel.preset_name or 'none'}`"
    )

    metrics = result["metrics"]
    risk_profile = compute_risk_profile_from_metrics(
        metrics, sel.initial_balance
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Return % / 10k", f"{risk_profile['return_pct']:.2f} %")
        st.metric("Profit Factor", f"{risk_profile['profit_factor']:.2f}")
    with col2:
        st.metric("Sharpe", f"{risk_profile['sharpe']:.2f}")
        st.metric("Max DD %", f"{risk_profile['max_drawdown_pct']:.2f}")
    with col3:
        st.metric("Risk of Ruin", f"{100.0 * (risk_profile['risk_of_ruin'] or 0.0):.2f} %")
        st.metric("Robustness", f"{100.0 * (risk_profile['robustness_score'] or 0.0):.1f} / 100")

    st.subheader("Equity Curve (stratégie)")
    eq_df: pd.DataFrame = result["equity_df"]
    if not eq_df.empty:
        st.line_chart(eq_df.set_index("ts")["equity"])
    else:
        st.info("Pas d'equity curve disponible (aucun trade ou dataset vide).")


def render_best_settings_tab(sel: ControlPanelSelection) -> None:
    st.header("🏆 Meilleures configurations (Presets & Supersearch)")

    presets = load_presets_cached()
    st.subheader("Presets supersearch")
    if presets:
        st.json(presets, expanded=False)
    else:
        st.info("Aucun preset supersearch chargé.")

    st.markdown("---")
    st.subheader("Top résultats Supersearch (fichiers natifs)")

    df = load_supersearch_results(sel.ticker, sel.period, sel.interval, top_n=20)
    if df.empty:
        st.info(
            "Aucun CSV supersearch trouvé pour cette combinaison. Lance `run_supersearch.py` "
            "avec --engine native/auto pour peupler docs/experiments/."
        )
        return

    st.dataframe(df)

    st.markdown("### Appliquer une configuration")
    row_idx = st.number_input(
        "Index de ligne à appliquer (0 = meilleure stratégie)",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )
    if st.button("Charger ces paramètres dans le panel"):
        row = df.iloc[int(row_idx)]
        st.session_state["long_threshold"] = float(row.get("long_threshold", sel.long_threshold))
        st.session_state["short_threshold"] = float(row.get("short_threshold", sel.short_threshold))
        st.session_state["risk_pct"] = float(row.get("risk_pct", sel.risk_pct))
        st.success("Configuration appliquée (les sliders refléteront ces valeurs).")
        st.experimental_rerun()


def render_risk_lab_tab(sel: ControlPanelSelection) -> None:
    st.header("🧪 Risk Lab – Simulation de profil de risque")

    st.write(
        "Ici tu peux jouer avec les paramètres de risque (risk_pct, horizon, "
        "bootstrap_runs) et voir l'effet sur la probabilité de ruine, le DD, etc."
    )

    col1, col2 = st.columns(2)
    with col1:
        risk_pct = st.slider(
            "risk_pct par trade",
            min_value=0.001,
            max_value=0.05,
            value=sel.risk_pct,
            step=0.001,
        )
        bootstrap_runs = st.slider(
            "Nombre de runs bootstrap",
            min_value=64,
            max_value=2048,
            value=sel.bootstrap_runs,
            step=64,
        )
    with col2:
        horizon_trades = st.slider(
            "Horizon (nombre de trades simulés)",
            min_value=20,
            max_value=500,
            value=100,
            step=10,
        )
        target_dd = st.slider(
            "DD max toléré (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
        )

    updated_sel = ControlPanelSelection(
        **{**sel.__dict__, "risk_pct": risk_pct, "bootstrap_runs": bootstrap_runs}
    )
    cfg = build_backtest_config(updated_sel)

    with st.spinner("Recalcul du backtest pour Risk Lab…"):
        result = run_backtest_with_native_or_python(cfg, updated_sel.bootstrap_runs)

    metrics = result["metrics"]
    risk_profile = compute_risk_profile_from_metrics(
        metrics, updated_sel.initial_balance
    )
    stats_cache: Dict[Tuple[Any, ...], runner.SimulationStats] = st.session_state.setdefault(
        "risk_lab_stats_cache", {}
    )
    stats_key = (
        updated_sel.ticker,
        updated_sel.period,
        updated_sel.interval,
        updated_sel.long_threshold,
        updated_sel.short_threshold,
        updated_sel.risk_pct,
        updated_sel.trend_filter,
        updated_sel.sentiment_min,
        updated_sel.sentiment_max,
        updated_sel.sentiment_regime,
        updated_sel.trailing_stop_activation,
        updated_sel.trailing_stop_distance,
    )
    python_stats = result.get("python_stats")
    if python_stats is not None:
        stats_cache[stats_key] = python_stats
    else:
        python_stats = stats_cache.get(stats_key)
        if python_stats is None:
            python_stats = runner.simulate_from_dataset(result["dataset"], result["config"])
            stats_cache[stats_key] = python_stats

    st.subheader("Résumé risque (à partir des trades réels)")
    st.write(
        f"- Return %: **{risk_profile['return_pct']:.2f} %**\n"
        f"- Profit Factor: **{risk_profile['profit_factor']:.2f}**\n"
        f"- Sharpe: **{risk_profile['sharpe']:.2f}**\n"
        f"- Max DD réel: **{risk_profile['max_drawdown_pct']:.2f} %**\n"
        f"- Risk of Ruin (approx): **{100.0 * (risk_profile['risk_of_ruin'] or 0.0):.2f} %**\n"
        f"- Robustness: **{100.0 * (risk_profile['robustness_score'] or 0.0):.1f} / 100**"
    )

    st.subheader("Interprétation rapide")
    ror = float(risk_profile["risk_of_ruin"] or 0.0)
    dd = float(risk_profile["max_drawdown_pct"] or 0.0)
    st.write(
        f"- Probabilité approx. de ruine (horizon simulé) : **{100.0 * ror:.2f} %**\n"
        f"- Drawdown max observé : **{dd:.2f} %**\n"
        f"- Avec risk_pct = **{risk_pct:.3f}**, un capital de **{updated_sel.initial_balance:,.0f} $** "
        f"implique une perte potentielle d'environ **{dd / 100 * updated_sel.initial_balance:,.0f} $**"
    )

    st.markdown("### Monte Carlo (risque de perte)")
    mc_runs = st.slider(
        "Nombre de scénarios Monte Carlo",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="risk_lab_mc_runs",
    )

    cfg_fingerprint = stats_key + (int(horizon_trades), float(target_dd), int(mc_runs))
    if st.session_state.get("risk_lab_last_cfg") != cfg_fingerprint:
        st.session_state["risk_lab_last_cfg"] = cfg_fingerprint
        st.session_state.pop("risk_lab_mc_summary", None)

    if st.button("Lancer la simulation Monte Carlo"):
        summary = run_monte_carlo_from_stats(
            python_stats,
            result["config"],
            horizon_trades=int(horizon_trades),
            n_runs=int(mc_runs),
            loss_threshold_pct=float(target_dd),
        )
        st.session_state["risk_lab_mc_summary"] = summary

    mc_summary: Optional[MonteCarloSummary] = st.session_state.get("risk_lab_mc_summary")
    if mc_summary is not None:
        if np.isnan(mc_summary.prob_loss_any):
            st.warning("Monte Carlo indisponible : pas assez de trades pour estimer la distribution.")
        else:
            st.write(
                f"- Probabilité de finir en perte ({mc_summary.horizon_trades} trades) : **{mc_summary.prob_loss_any * 100:.1f}%**"
            )
            st.write(
                f"- Probabilité de perdre ≥ {mc_summary.loss_threshold_pct:.1f}% : **{mc_summary.prob_loss_gt_threshold * 100:.1f}%**"
            )
            st.write(
                "- Δ equity simulé (en %) : "
                f"p05 = **{mc_summary.return_p05:.1f}%**, "
                f"median = **{mc_summary.return_p50:.1f}%**, "
                f"p95 = **{mc_summary.return_p95:.1f}%**"
            )
            st.write(
                "- Drawdown max simulé (en %) : "
                f"p50 = **{mc_summary.maxdd_p50:.1f}%**, "
                f"p95 = **{mc_summary.maxdd_p95:.1f}%**"
            )
    else:
        st.info("Lance la simulation Monte Carlo pour visualiser la distribution des pertes.")

    st.info(
        "TODO: compléter avec une vraie simulation forward (par ex. tirer des trades "
        "au hasard selon la distribution observée et tracer la distribution de DD "
        "pour l'horizon choisi)."
    )


def render_strategy_sandbox_tab(sel: ControlPanelSelection, result: Dict[str, Any]) -> None:
    st.header("🧬 Strategy Sandbox – Timeline & entrées/sorties")

    st.write(
        "But : voir à quel moment la stratégie entre/sort, comment évolue l'equity, "
        "et tester rapidement d'autres réglages pour améliorer l'IA."
    )

    equity_df: pd.DataFrame = result["equity_df"]
    trades_df: pd.DataFrame = result["trades_df"]
    position_df: pd.DataFrame = result.get("position_df", pd.DataFrame())

    st.subheader("Equity curve")
    if not equity_df.empty:
        st.line_chart(equity_df.set_index("ts")["equity"])
    else:
        st.info("Pas d'equity disponible.")

    st.subheader("Position (investi / flat / short)")
    if not position_df.empty:
        st.line_chart(position_df.set_index("ts")["position"])
    else:
        st.info("Série de position non disponible (TODO mapping signal → position).")

    st.subheader("Trades (placeholder)")
    if not trades_df.empty:
        st.dataframe(trades_df)
    else:
        st.info("Extraction des trades non implémentée. TODO: mapper result.trades ou stats.trades → DataFrame.")

    st.markdown("---")
    st.subheader("Suggestions d'amélioration (TODO)")
    st.info(
        "Ici tu peux plus tard calculer des 'what-if' (ex: augmenter le long_threshold, "
        "réduire risk_pct) et proposer automatiquement des variantes qui améliorent PF "
        "ou réduisent DD, en s'appuyant sur supercalc/native search."
    )


# ======================================================================
# 4. MAIN STREAMLIT APP
# ======================================================================


def build_initial_selection() -> ControlPanelSelection:
    presets = load_presets_cached()
    preset_names = ["(aucun)"] + sorted(presets.keys())

    st.sidebar.header("HyprL – Filtres globaux")

    for key, default in {
        "long_threshold": 0.6,
        "short_threshold": 0.4,
        "risk_pct": 0.02,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    ticker = st.sidebar.selectbox("Ticker", DEFAULT_TICKERS, index=0)
    period = st.sidebar.selectbox("Période", DEFAULT_PERIODS, index=1)
    interval = st.sidebar.selectbox("Intervalle", DEFAULT_INTERVALS, index=0)
    preset_choice = st.sidebar.selectbox("Preset Supersearch", preset_names, index=0)
    preset_name: Optional[str] = (
        None if preset_choice == "(aucun)" else preset_choice
    )

    long_threshold = st.sidebar.slider(
        "Long threshold",
        min_value=0.4,
        max_value=0.8,
        value=float(st.session_state["long_threshold"]),
        step=0.01,
        key="long_threshold",
    )
    short_threshold = st.sidebar.slider(
        "Short threshold",
        min_value=0.2,
        max_value=0.5,
        value=float(st.session_state["short_threshold"]),
        step=0.01,
        key="short_threshold",
    )
    risk_pct = st.sidebar.slider(
        "risk_pct",
        min_value=0.001,
        max_value=0.05,
        value=float(st.session_state["risk_pct"]),
        step=0.001,
        key="risk_pct",
    )
    trailing_enabled = st.sidebar.checkbox("Activer trailing stop", value=False)
    trailing_activation_value = st.sidebar.slider(
        "Trailing activation (R)",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="trailing_stop_activation",
    )
    trailing_distance_value = st.sidebar.slider(
        "Trailing distance (R)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        key="trailing_stop_distance",
    )
    trailing_stop_activation = trailing_activation_value if trailing_enabled else None
    trailing_stop_distance = trailing_distance_value if trailing_enabled else None
    initial_balance = st.sidebar.number_input(
        "Capital initial (pour Perf / 10k)",
        min_value=1_000.0,
        max_value=1_000_000.0,
        value=10_000.0,
        step=1_000.0,
    )

    trend_filter = st.sidebar.checkbox("Trend filter (SMA)", value=False)
    sentiment_min = st.sidebar.slider(
        "Sentiment min", min_value=-1.0, max_value=0.0, value=-1.0, step=0.05
    )
    sentiment_max = st.sidebar.slider(
        "Sentiment max", min_value=0.0, max_value=1.0, value=1.0, step=0.05
    )
    sentiment_regime = st.sidebar.selectbox(
        "Sentiment regime", ["off", "fear_only", "greed_only", "neutral_only"], index=0
    )

    bootstrap_runs = st.sidebar.slider(
        "Bootstrap runs (risk metrics)",
        min_value=64,
        max_value=1024,
        value=256,
        step=64,
    )
    seed = st.sidebar.number_input(
        "Seed (random_state)", min_value=0, max_value=1_000_000, value=42, step=1
    )

    return ControlPanelSelection(
        ticker=ticker,
        period=period,
        interval=interval,
        preset_name=preset_name,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        risk_pct=risk_pct,
        initial_balance=initial_balance,
        trend_filter=trend_filter,
        sentiment_min=sentiment_min,
        sentiment_max=sentiment_max,
        sentiment_regime=sentiment_regime,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
        trailing_stop_activation=trailing_stop_activation,
        trailing_stop_distance=trailing_stop_distance,
    )


def main() -> None:
    st.set_page_config(
        page_title="HyprL Control Panel",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("HyprL Control Panel (internal)")
    st.caption(
        "Supervision des stratégies, réglages de risque, et sandbox pour améliorer l'IA."
    )

    sel = build_initial_selection()
    base_cfg = build_backtest_config(sel)
    with st.spinner("Exécution du backtest de référence…"):
        base_result = run_backtest_with_native_or_python(base_cfg, sel.bootstrap_runs)

    tabs = st.tabs(
        [
            "📊 Overview",
            "🏆 Best Settings",
            "🧪 Risk Lab",
            "🧬 Strategy Sandbox",
        ]
    )

    with tabs[0]:
        render_overview_tab(sel, base_result)
    with tabs[1]:
        render_best_settings_tab(sel)
    with tabs[2]:
        render_risk_lab_tab(sel)
    with tabs[3]:
        render_strategy_sandbox_tab(sel, base_result)


if __name__ == "__main__":
    main()
