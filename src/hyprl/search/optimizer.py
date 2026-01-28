from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Any, Literal

import json
import math
import joblib
import numpy as np
import pandas as pd

from hyprl.adaptive.engine import AdaptiveConfig
from hyprl.backtest.runner import BacktestConfig, StrategyStats, SupercalcDataset, prepare_supercalc_dataset
from hyprl.configs import get_risk_settings, get_supersearch_preset, load_ticker_settings
from hyprl.labels.amplitude import LabelConfig
from hyprl.risk.manager import RiskConfig
from hyprl.native.supercalc import native_available as native_engine_available, run_native_search_batch
from hyprl.supercalc import evaluate_candidates, _build_signal_series
from hyprl.risk.metrics import (
    compute_basic_stats,
    compute_risk_of_ruin,
    bootstrap_equity_drawdowns,
)
from hyprl.risk.statistics import deflated_sharpe_ratio, SharpeStatistics
from hyprl.portfolio.core import (
    build_portfolio_equity,
    compute_portfolio_stats,
    compute_portfolio_weights,
    compute_correlation_matrix,
)
from hyprl.meta.features import build_feature_frame_from_records, select_meta_features
from hyprl.meta.model import MetaRobustnessModel
import joblib


@dataclass(slots=True)
class SearchConfig:
    ticker: str
    tickers: list[str] = field(default_factory=list)
    period: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1h"
    initial_balance: float = 10_000.0
    seed: int = 42
    use_presets: bool = True
    constraint_preset: Optional[str] = None
    long_thresholds: list[float] = field(default_factory=list)
    short_thresholds: list[float] = field(default_factory=list)
    risk_pcts: list[float] = field(default_factory=list)
    min_ev_multiples: Optional[list[float]] = None
    trend_filter_flags: Optional[list[bool]] = None
    sentiment_min_values: list[float] = field(default_factory=list)
    sentiment_max_values: list[float] = field(default_factory=list)
    sentiment_regimes: list[str] = field(default_factory=list)
    engine: str = "auto"
    min_trades: int = 50
    min_profit_factor: float = 1.2
    min_sharpe: float = 0.8
    max_drawdown_pct: float = 0.35
    max_risk_of_ruin: float = 0.1
    min_expectancy: float = 0.0
    bootstrap_runs: int = 256
    min_portfolio_profit_factor: float = 0.0
    min_portfolio_sharpe: float = -10.0
    max_portfolio_drawdown_pct: float = 1.0
    max_portfolio_risk_of_ruin: float = 1.0
    max_correlation: float = 1.0
    min_robustness_score: float = 0.0
    weighting_scheme: Literal["equal", "inv_vol"] = "equal"
    meta_robustness_model_path: Optional[str] = None
    meta_weight: float = 0.4
    meta_registry_key: Optional[str] = None
    meta_calibration_path: Optional[str] = None
    meta_calibration_registry_key: Optional[str] = None
    label: LabelConfig = field(default_factory=LabelConfig)
    debug: bool = False

    def validate(self) -> None:
        if not self.period and not (self.start and self.end):
            raise ValueError("SearchConfig requires --period or both --start and --end.")


@dataclass(slots=True)
class CandidateConfig:
    long_threshold: float
    short_threshold: float
    risk_pct: float
    min_ev_multiple: float
    trend_filter: bool
    sentiment_min: float
    sentiment_max: float
    sentiment_regime: str


@dataclass(slots=True)
class SearchResult:
    config: CandidateConfig
    strategy_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    profit_factor: float
    sharpe: Optional[float]
    max_drawdown_pct: float
    expectancy: float
    n_trades: int
    win_rate_pct: float
    sentiment_stats: dict[str, int] = field(default_factory=dict)
    expectancy_per_trade: float = 0.0
    risk_of_ruin: float = 1.0
    maxdd_p95: float = 0.0
    pnl_p05: float = 0.0
    pnl_p95: float = 0.0
    portfolio_return_pct: float = 0.0
    portfolio_profit_factor: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0
    portfolio_risk_of_ruin: float = 1.0
    portfolio_maxdd_p95: float = 0.0
    portfolio_pnl_p05: float = 0.0
    correlation_mean: float = 0.0
    correlation_max: float = 0.0
    per_ticker_details: dict[str, dict[str, float]] = field(default_factory=dict)
    portfolio_equity_vol: float = 0.0
    robustness_score: float = 0.0
    base_score: float = 0.0
    meta_prediction: Optional[float] = None
    final_score: float = 0.0
    portfolio_weights: dict[str, float] = field(default_factory=dict)
    deflated_sharpe: Optional[float] = None  # Sharpe corrected for multiple testing
    pbo: float = 0.5  # Probability of backtest overfitting


@dataclass
class SearchDebugTracker:
    enabled: bool = False
    grid_estimate: int = 0
    grid_candidates: int = 0
    invalid_combos: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hard_rejects: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dataset_failures: int = 0
    evaluation_failures: int = 0
    survivors: int = 0

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[DEBUG] {message}")

    def record_invalid(self, reason: str) -> None:
        if self.enabled:
            self.invalid_combos[reason] += 1

    def record_hard_reject(self, reason: str) -> None:
        if self.enabled:
            self.hard_rejects[reason] += 1

    def record_dataset_failure(self) -> None:
        if self.enabled:
            self.dataset_failures += 1

    def record_eval_failure(self) -> None:
        if self.enabled:
            self.evaluation_failures += 1

    def set_grid_estimate(self, estimate: int) -> None:
        if self.enabled:
            self.grid_estimate = int(estimate)

    def increment_candidates(self) -> None:
        if self.enabled:
            self.grid_candidates += 1

    def set_survivors(self, count: int) -> None:
        if self.enabled:
            self.survivors = int(count)

    def _format_counts(self, payload: dict[str, int]) -> str:
        if not payload:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(payload.items()))

    def log_summary(self) -> None:
        if not self.enabled:
            return
        invalid = self._format_counts(self.invalid_combos)
        rejects = self._format_counts(self.hard_rejects)
        self.log(
            "Search stats → grid_estimate=%s validated=%s survivors=%s dataset_failures=%s eval_failures=%s"
            % (
                self.grid_estimate,
                self.grid_candidates,
                self.survivors,
                self.dataset_failures,
                self.evaluation_failures,
            )
        )
        self.log(f"Invalid combos: {invalid}")
        self.log(f"Hard rejects: {rejects}")


def _preview_list(values: Iterable[Any], limit: int = 4) -> str:
    seq = list(values)
    if not seq:
        return "[]"
    head = ", ".join(str(item) for item in seq[:limit])
    if len(seq) > limit:
        head += f", ... (x{len(seq)})"
    return f"[{head}]"


def _resolve_list(values: list[float], fallback: float) -> list[float]:
    target = [float(v) for v in values if math.isfinite(float(v))] if values else []
    if not target:
        target = [float(fallback)]
    return target


def _resolve_bool_list(values: Optional[list[bool]], fallback: bool) -> list[bool]:
    if values is None or not values:
        return [bool(fallback)]
    return [bool(v) for v in values]


def _resolve_str_list(values: Optional[list[str]], fallback: str) -> list[str]:
    if values is None or not values:
        return [str(fallback)]
    return [str(item) if item else str(fallback) for item in values]


def _score_tuple(result: SearchResult) -> tuple[float, float, float]:
    """
    Compute the scoring tuple for lexicographic ranking of search results.

    IMPORTANT: Uses DEFLATED Sharpe ratio to correct for multiple testing bias.
    Raw Sharpe is inflated when testing many parameter combinations.

    Ranking Logic:
    --------------
    Results are sorted by this tuple in ASCENDING order (lower is better).
    Python's tuple comparison is lexicographic: the first element is compared first;
    if equal, the second is compared, and so on.

    Tuple Structure (in priority order):
    ------------------------------------
    1. Primary score: -PF + RoR*2 + sentiment_ratio + PBO
       - Rewards high profit_factor (negated, so high PF → low score)
       - Penalizes high risk_of_ruin (RoR > 0.1 is fragile)
       - Penalizes high probability of backtest overfitting (PBO)
       - Penalizes overconcentration in extreme sentiment trades

    2. Risk-adjusted score: -DSR + RoR
       - Rewards high DEFLATED Sharpe ratio (corrected for multiple testing)
       - Penalizes high RoR again (tail risk)

    3. Drawdown/expectancy score: DD + RoR*100 - expectancy*100
       - Penalizes high drawdown (both portfolio and individual)
       - Penalizes high RoR (heavily weighted)
       - Rewards positive expectancy (avg profit per trade)

    Why Deflated Sharpe?
    --------------------
    When testing N strategies, the "best" Sharpe is inflated by selection bias.
    E[max(Sharpe)] ≈ sqrt(2*ln(N)) * SE(Sharpe)
    DSR = (Sharpe - E[max]) / SE corrects for this.

    Example:
    --------
    Testing 1000 configs with raw Sharpe 1.29, 98 trades:
    - SE(Sharpe) ≈ 0.14
    - E[max] ≈ 3.7 * 0.14 ≈ 0.52
    - DSR ≈ (1.29 - 0.52) / 0.14 ≈ 5.5 (still significant)

    But with raw Sharpe 0.8:
    - DSR ≈ (0.8 - 0.52) / 0.14 ≈ 2.0 (barely significant)

    Args:
        result: SearchResult containing backtest metrics

    Returns:
        tuple: Scoring tuple for lexicographic ranking (lower is better)
    """
    base_pf = result.portfolio_profit_factor or result.profit_factor
    pf = base_pf if math.isfinite(base_pf) else 0.0

    # Use DEFLATED Sharpe (corrected for multiple testing) instead of raw
    dsr = result.deflated_sharpe
    if dsr is None or not math.isfinite(dsr):
        # Fall back to raw sharpe if deflated not computed yet
        base_sharpe = result.portfolio_sharpe if result.portfolio_sharpe else result.sharpe
        dsr = base_sharpe if (base_sharpe is not None and math.isfinite(base_sharpe)) else float("-inf")

    primary_dd = max(
        result.portfolio_max_drawdown_pct,
        result.portfolio_maxdd_p95,
        result.max_drawdown_pct,
        result.maxdd_p95,
    )
    sentiment_stats = result.sentiment_stats or {}
    extreme_total = float(sentiment_stats.get("trades_in_fear", 0) + sentiment_stats.get("trades_in_greed", 0))
    sentiment_ratio = extreme_total / result.n_trades if result.n_trades else 0.0
    ror = result.portfolio_risk_of_ruin if math.isfinite(result.portfolio_risk_of_ruin) else result.risk_of_ruin
    if not math.isfinite(ror):
        ror = 1.0
    expectancy = result.expectancy_per_trade

    # Probability of backtest overfitting (0 = no overfitting, 1 = likely overfit)
    pbo = result.pbo if math.isfinite(result.pbo) else 0.5

    # Lower tuple values are better
    return (
        -pf + ror * 2.0 + sentiment_ratio + pbo,  # Added PBO penalty
        -dsr + ror,  # Use deflated Sharpe instead of raw
        primary_dd + ror * 100.0 - expectancy * 100.0,
    )


def _passes_hard_constraints(
    cfg: SearchConfig,
    result: SearchResult,
    tracker: SearchDebugTracker | None = None,
) -> bool:
    def reject(reason: str) -> bool:
        if tracker is not None:
            tracker.record_hard_reject(reason)
        return False

    if result.n_trades < cfg.min_trades:
        return reject("min_trades")
    if not math.isfinite(result.profit_factor) or result.profit_factor < cfg.min_profit_factor:
        return reject("profit_factor")
    if cfg.min_sharpe is not None and cfg.min_sharpe > 0.0:
        if result.sharpe is None or not math.isfinite(result.sharpe) or result.sharpe < cfg.min_sharpe:
            return reject("sharpe")
    max_dd_limit = cfg.max_drawdown_pct * 100.0
    if result.max_drawdown_pct > max_dd_limit or result.maxdd_p95 > max_dd_limit:
        return reject("drawdown")
    if not math.isfinite(result.risk_of_ruin) or result.risk_of_ruin > cfg.max_risk_of_ruin:
        return reject("risk_of_ruin")
    if result.expectancy_per_trade < cfg.min_expectancy:
        return reject("expectancy")
    if result.win_rate_pct > 85.0 and result.profit_factor < 1.3:
        return reject("winrate_anomaly")
    if cfg.min_portfolio_profit_factor > 0.0 and (
        result.portfolio_profit_factor < cfg.min_portfolio_profit_factor
    ):
        return reject("portfolio_profit_factor")
    if cfg.min_portfolio_sharpe is not None and cfg.min_portfolio_sharpe > 0.0:
        sharpe = result.portfolio_sharpe
        if sharpe is None or not math.isfinite(sharpe) or sharpe < cfg.min_portfolio_sharpe:
            return reject("portfolio_sharpe")
    if result.portfolio_max_drawdown_pct > cfg.max_portfolio_drawdown_pct * 100.0:
        return reject("portfolio_drawdown")
    if result.portfolio_risk_of_ruin > cfg.max_portfolio_risk_of_ruin:
        return reject("portfolio_risk_of_ruin")
    if cfg.max_correlation < 1.0 and result.correlation_max > cfg.max_correlation:
        return reject("correlation")
    if cfg.min_robustness_score > 0.0:
        score = result.robustness_score
        if not math.isfinite(score) or score < cfg.min_robustness_score:
            return reject("robustness_score")
    return True


def _series_from_history(history: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    if not history:
        return pd.Series(dtype=float)
    data: dict[pd.Timestamp, float] = {}
    for ts, value in history:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        data[ts] = float(value)
    return pd.Series(data).sort_index()


def _correlation_summary(matrix: pd.DataFrame) -> tuple[float, float]:
    if matrix.empty or matrix.shape[0] < 2:
        return 0.0, 0.0
    arr = matrix.to_numpy()
    n = arr.shape[0]
    mask = np.triu_indices(n, k=1)
    off_diag = arr[mask]
    if off_diag.size == 0:
        return 0.0, 0.0
    return float(np.nanmean(off_diag)), float(np.nanmax(off_diag))


def _assign_base_scores(results: list[SearchResult], n_trials_override: int | None = None) -> None:
    """
    Assign normalized rank-based scores to search results.

    Scoring Process:
    ----------------
    1. Compute deflated Sharpe ratio for each result (corrects for multiple testing)
    2. Sort all results by _score_tuple (lexicographic ranking, lower is better).
    3. Assign base_score as a linear interpolation of rank:
       - Best result (rank 0): base_score = 1.0
       - Worst result (rank N-1): base_score = 0.0
       - Intermediate: base_score = 1.0 - (rank / (N-1))

    The deflated Sharpe uses Bailey & López de Prado (2014) correction:
    - Accounts for the fact that testing N strategies inflates the best Sharpe
    - DSR > 2 typically indicates genuine skill (not just luck)
    - DSR < 0 suggests the strategy is likely overfit

    Args:
        results: List of SearchResult objects (modified in-place)
        n_trials_override: Override for number of trials (default: len(results))
    """
    if not results:
        return

    # Step 1: Compute deflated Sharpe for each result
    n_trials = n_trials_override if n_trials_override else len(results)
    for res in results:
        sharpe = res.portfolio_sharpe if res.portfolio_sharpe else res.sharpe
        if sharpe is None or not math.isfinite(sharpe):
            res.deflated_sharpe = float("-inf")
            res.pbo = 1.0
            continue

        n_obs = res.n_trades
        if n_obs < 10:
            res.deflated_sharpe = float("-inf")
            res.pbo = 1.0
            continue

        # Compute deflated Sharpe
        res.deflated_sharpe = deflated_sharpe_ratio(sharpe, n_obs, n_trials)
        # Probability of backtest overfitting
        from hyprl.risk.statistics import probability_of_backtest_overfitting
        res.pbo = probability_of_backtest_overfitting(sharpe, n_obs, n_trials)

    # Step 2: Sort by scoring tuple
    ordering = sorted(range(len(results)), key=lambda idx: _score_tuple(results[idx]))
    total = len(ordering)
    if total == 1:
        results[ordering[0]].base_score = 1.0
        results[ordering[0]].final_score = 1.0
        return

    # Step 3: Assign rank-based scores
    for rank, idx in enumerate(ordering):
        score = 1.0 - (rank / (total - 1))
        results[idx].base_score = score
        results[idx].final_score = score


def _apply_meta_scores(
    results: list[SearchResult],
    model: MetaRobustnessModel,
    meta_weight: float,
    weighting_scheme: str,
    calibrator: Any | None = None,
) -> None:
    if not results:
        return
    records: list[dict[str, object]] = []
    for res in results:
        cfg = res.config
        records.append(
            {
                "long_threshold": cfg.long_threshold,
                "short_threshold": cfg.short_threshold,
                "risk_pct": cfg.risk_pct,
                "min_ev_multiple": cfg.min_ev_multiple,
                "trend_filter": cfg.trend_filter,
                "sentiment_min": cfg.sentiment_min,
                "sentiment_max": cfg.sentiment_max,
                "sentiment_regime": cfg.sentiment_regime,
                "weighting_scheme": weighting_scheme,
                "pf_backtest": res.portfolio_profit_factor,
                "sharpe_backtest": res.portfolio_sharpe,
                "maxdd_backtest": res.portfolio_max_drawdown_pct,
                "winrate_backtest": (res.win_rate_pct or 0.0) / 100.0,
                "equity_vol_backtest": res.portfolio_equity_vol,
                "trades_backtest": res.n_trades,
                "correlation_mean": res.correlation_mean,
                "correlation_max": res.correlation_max,
            }
        )
    feature_df = build_feature_frame_from_records(records)
    feature_matrix, _ = select_meta_features(feature_df)
    # Be robust to DummyModel implementations that don't accept a 'calibrator' kwarg
    try:
        preds = model.predict(feature_matrix, calibrator=calibrator)
    except TypeError:
        preds = model.predict(feature_matrix)
    weight = float(np.clip(meta_weight, 0.0, 1.0))
    for res, pred in zip(results, preds):
        res.meta_prediction = float(pred)
        res.final_score = (1.0 - weight) * res.base_score + weight * float(pred)


_CONSTRAINT_FIELD_MAP: dict[str, tuple[str, type]] = {
    "min_trades": ("min_trades", int),
    "min_pf": ("min_profit_factor", float),
    "min_profit_factor": ("min_profit_factor", float),
    "min_sharpe": ("min_sharpe", float),
    "max_dd": ("max_drawdown_pct", float),
    "max_drawdown_pct": ("max_drawdown_pct", float),
    "max_ror": ("max_risk_of_ruin", float),
    "max_risk_of_ruin": ("max_risk_of_ruin", float),
    "min_expectancy": ("min_expectancy", float),
    "min_portfolio_pf": ("min_portfolio_profit_factor", float),
    "min_portfolio_profit_factor": ("min_portfolio_profit_factor", float),
    "min_portfolio_sharpe": ("min_portfolio_sharpe", float),
    "max_portfolio_dd": ("max_portfolio_drawdown_pct", float),
    "max_portfolio_drawdown_pct": ("max_portfolio_drawdown_pct", float),
    "max_portfolio_ror": ("max_portfolio_risk_of_ruin", float),
    "max_portfolio_risk_of_ruin": ("max_portfolio_risk_of_ruin", float),
    "max_correlation": ("max_correlation", float),
    "min_robustness_score": ("min_robustness_score", float),
}


def _apply_constraint_preset(cfg: SearchConfig) -> SearchConfig:
    if not cfg.constraint_preset:
        return cfg
    try:
        preset = get_supersearch_preset(cfg.constraint_preset)
    except KeyError as exc:  # pragma: no cover - invalid preset name should surface to CLI/tests
        raise ValueError(f"Preset '{cfg.constraint_preset}' introuvable") from exc
    overrides: dict[str, Any] = {}
    for key, (attr, caster) in _CONSTRAINT_FIELD_MAP.items():
        if key not in preset or preset[key] is None:
            continue
        try:
            overrides[attr] = caster(preset[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Valeur invalide pour '{key}' dans le preset '{cfg.constraint_preset}': {preset[key]!r}"
            ) from exc
    if not overrides:
        return cfg
    return replace(cfg, **overrides)


def run_search(search_cfg: SearchConfig) -> list[SearchResult]:
    search_cfg.validate()
    search_cfg = _apply_constraint_preset(search_cfg)
    tracker = SearchDebugTracker(enabled=search_cfg.debug)
    weighting_scheme = (search_cfg.weighting_scheme or "equal").lower()
    if weighting_scheme not in {"equal", "inv_vol"}:
        weighting_scheme = "equal"
    meta_model: MetaRobustnessModel | None = None
    meta_model_path = search_cfg.meta_robustness_model_path
    if search_cfg.meta_registry_key and not meta_model_path:
        from hyprl.meta.registry import resolve_model

        meta_model_path = str(resolve_model(search_cfg.meta_registry_key))
    if meta_model_path:
        try:
            meta_model = MetaRobustnessModel.load(meta_model_path)
        except Exception as exc:  # pragma: no cover - load errors
            raise RuntimeError(f"Impossible de charger le modèle meta: {exc}") from exc
    settings = load_ticker_settings(search_cfg.ticker, search_cfg.interval) if search_cfg.use_presets else {}
    base_long = float(settings.get("long_threshold", 0.6))
    base_short = float(settings.get("short_threshold", 0.4))
    risk_profile = settings.get("default_risk_profile") if search_cfg.use_presets else None
    risk_params = get_risk_settings(settings, profile=risk_profile)
    active_tickers = [t.strip() for t in (search_cfg.tickers or []) if t.strip()]
    if not active_tickers:
        active_tickers = [search_cfg.ticker]
    per_ticker_capital = search_cfg.initial_balance / max(len(active_tickers), 1)
    tracker.log(
        "Search start → tickers=%s interval=%s period=%s start=%s end=%s engine=%s weighting=%s label=%s"
        % (
            ",".join(active_tickers),
            search_cfg.interval,
            search_cfg.period or "",
            search_cfg.start or "",
            search_cfg.end or "",
            search_cfg.engine,
            weighting_scheme,
            search_cfg.label.mode,
        )
    )

    base_risk = RiskConfig(
        balance=per_ticker_capital,
        risk_pct=float(risk_params["risk_pct"]),
        atr_multiplier=float(risk_params["atr_multiplier"]),
        reward_multiple=float(risk_params["reward_multiple"]),
        min_position_size=int(risk_params["min_position_size"]),
    )
    min_ev_default = float(settings.get("min_ev_multiple", 0.0)) if search_cfg.use_presets else 0.0
    trend_default = bool(settings.get("enable_trend_filter", False)) if search_cfg.use_presets else False
    trend_long_min = float(settings.get("trend_long_min", 0.0)) if search_cfg.use_presets else 0.0
    trend_short_min = float(settings.get("trend_short_min", 0.0)) if search_cfg.use_presets else 0.0
    model_type = str(settings.get("model_type", "logistic")) if search_cfg.use_presets else "logistic"
    calibration = str(settings.get("calibration", "none")) if search_cfg.use_presets else "none"
    sentiment_min_default = float(settings.get("sentiment_min", -1.0)) if search_cfg.use_presets else -1.0
    sentiment_max_default = float(settings.get("sentiment_max", 1.0)) if search_cfg.use_presets else 1.0
    sentiment_regime_default = str(settings.get("sentiment_regime", "off")) if search_cfg.use_presets else "off"

    long_values = _resolve_list(search_cfg.long_thresholds, base_long)
    short_values = _resolve_list(search_cfg.short_thresholds, base_short)
    risk_values = _resolve_list(search_cfg.risk_pcts, base_risk.risk_pct)
    min_ev_values = (
        _resolve_list(search_cfg.min_ev_multiples or [], min_ev_default)
        if search_cfg.min_ev_multiples is not None
        else [min_ev_default]
    )
    trend_values = _resolve_bool_list(search_cfg.trend_filter_flags, trend_default)
    sentiment_min_values = _resolve_list(search_cfg.sentiment_min_values, sentiment_min_default)
    sentiment_max_values = _resolve_list(search_cfg.sentiment_max_values, sentiment_max_default)
    sentiment_regime_values = _resolve_str_list(search_cfg.sentiment_regimes, sentiment_regime_default)
    grid_estimate = (
        len(long_values)
        * len(short_values)
        * len(risk_values)
        * len(min_ev_values)
        * len(trend_values)
        * len(sentiment_min_values)
        * len(sentiment_max_values)
        * len(sentiment_regime_values)
    )
    tracker.set_grid_estimate(grid_estimate)
    tracker.log(
        "Grid axes | long=%s short=%s risk=%s min_ev=%s trend=%s sent_min=%s sent_max=%s regimes=%s estimate=%d"
        % (
            _preview_list(long_values),
            _preview_list(short_values),
            _preview_list(risk_values),
            _preview_list(min_ev_values),
            _preview_list(trend_values),
            _preview_list(sentiment_min_values),
            _preview_list(sentiment_max_values),
            _preview_list(sentiment_regime_values),
            grid_estimate,
        )
    )

    commission_pct = float(settings.get("commission_pct", 0.0005)) if search_cfg.use_presets else 0.0005
    slippage_pct = float(settings.get("slippage_pct", 0.0005)) if search_cfg.use_presets else 0.0005
    adaptive_cfg = AdaptiveConfig(enable=False, lookback_trades=20, default_regime="normal", regimes={})
    base_bt_config = BacktestConfig(
        ticker=active_tickers[0],
        period=search_cfg.period,
        start=search_cfg.start,
        end=search_cfg.end,
        interval=search_cfg.interval,
        initial_balance=per_ticker_capital,
        long_threshold=base_long,
        short_threshold=base_short,
        model_type=model_type,
        calibration=calibration,
        risk=base_risk,
        risk_profile="",
        risk_profiles={},
        adaptive=adaptive_cfg,
        random_state=search_cfg.seed,
        min_ev_multiple=min_ev_default,
        enable_trend_filter=trend_default,
        trend_long_min=trend_long_min,
        trend_short_min=trend_short_min,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        sentiment_min=sentiment_min_default,
        sentiment_max=sentiment_max_default,
        sentiment_regime=sentiment_regime_default,
        label=search_cfg.label,
    )
    datasets_by_ticker: dict[str, SupercalcDataset] = {}
    for ticker in active_tickers:
        cfg_for_dataset = replace(base_bt_config, ticker=ticker)
        try:
            datasets_by_ticker[ticker] = prepare_supercalc_dataset(cfg_for_dataset)
        except Exception as exc:
            tracker.record_dataset_failure()
            tracker.log(f"Dataset preload failed for {ticker}: {exc}")
            datasets_by_ticker = {}
            break

    candidate_entries: list[tuple[CandidateConfig, BacktestConfig]] = []

    for long_threshold in long_values:
        for short_threshold in short_values:
            if not (0.0 < short_threshold <= long_threshold < 1.0):
                tracker.record_invalid("threshold_order")
                continue
            for risk_pct in risk_values:
                if risk_pct <= 0:
                    tracker.record_invalid("risk_pct")
                    continue
                risk_template = replace(base_risk, risk_pct=float(risk_pct))
                for min_ev_multiple in min_ev_values:
                    if min_ev_multiple < 0:
                        tracker.record_invalid("min_ev_multiple")
                        continue
                    for trend_flag in trend_values:
                        for sentiment_min in sentiment_min_values:
                            for sentiment_max in sentiment_max_values:
                                if sentiment_min > sentiment_max:
                                    tracker.record_invalid("sentiment_bounds")
                                    continue
                                for sentiment_regime in sentiment_regime_values:
                                    candidate = CandidateConfig(
                                        long_threshold=float(long_threshold),
                                        short_threshold=float(short_threshold),
                                        risk_pct=float(risk_pct),
                                        min_ev_multiple=float(min_ev_multiple),
                                        trend_filter=bool(trend_flag),
                                        sentiment_min=float(sentiment_min),
                                        sentiment_max=float(sentiment_max),
                                        sentiment_regime=str(sentiment_regime),
                                    )
                                    bt_config = replace(
                                        base_bt_config,
                                        long_threshold=candidate.long_threshold,
                                        short_threshold=candidate.short_threshold,
                                        risk=risk_template,
                                        min_ev_multiple=candidate.min_ev_multiple,
                                        enable_trend_filter=candidate.trend_filter,
                                        sentiment_min=candidate.sentiment_min,
        sentiment_max=candidate.sentiment_max,
        sentiment_regime=candidate.sentiment_regime,
    )
                                    candidate_entries.append((candidate, bt_config))
                                    tracker.increment_candidates()

    if not candidate_entries:
        tracker.log("Aucune combinaison valide après filtrage initial.")
        tracker.log_summary()
        return []
    tracker.log(
        "Combinaisons validées=%d (estimation brute=%d, invalid=%d)"
        % (
            len(candidate_entries),
            tracker.grid_estimate or len(candidate_entries),
            sum(tracker.invalid_combos.values()),
        )
    )

    engine_mode = (search_cfg.engine or "auto").lower()
    dataset = None
    use_dataset = engine_mode in {"auto", "python", "native"}
    if use_dataset:
        try:
            dataset = prepare_supercalc_dataset(base_bt_config)
        except Exception as exc:
            if engine_mode == "native":
                raise
            dataset = None
            tracker.record_dataset_failure()
            tracker.log(f"Dataset global build failed: {exc}")

    precomputed_stats: dict[tuple[int, str], StrategyStats] = {}
    precomputed_metrics: dict[tuple[int, str], dict[str, Any]] = {}
    if datasets_by_ticker and engine_mode in {"auto", "native"}:
        try:
            precomputed_stats, precomputed_metrics = _precompute_native_stats(
                active_tickers,
                datasets_by_ticker,
                candidate_entries,
                per_ticker_capital,
            )
        except Exception as exc:  # pragma: no cover - logged, fallback handled below
            if engine_mode == "native":
                raise
            tracker.log(f"Native search precompute failed, falling back to Python engine: {exc}")
            precomputed_stats = {}
            precomputed_metrics = {}

    results: list[SearchResult] = []

    for candidate_idx, (candidate, bt_config) in enumerate(candidate_entries):
        per_ticker_equities: dict[str, pd.Series] = {}
        per_ticker_returns: dict[str, pd.Series] = {}
        per_ticker_details: dict[str, dict[str, float]] = {}
        benchmark_values: list[float] = []
        stats_collection: dict[str, StrategyStats] = {}
        valid = True

        for ticker in active_tickers:
            cfg_for_ticker = replace(bt_config, ticker=ticker)
            dataset = datasets_by_ticker.get(ticker)
            if dataset is None:
                try:
                    dataset = prepare_supercalc_dataset(cfg_for_ticker)
                    datasets_by_ticker[ticker] = dataset
                except Exception as exc:
                    valid = False
                    tracker.record_dataset_failure()
                    tracker.log(f"Dataset build failed for {ticker}: {exc}")
                    break
            benchmark_values.append(dataset.benchmark_return_pct)
            stats = precomputed_stats.get((candidate_idx, ticker))
            if stats is None:
                try:
                    stats = evaluate_candidates(
                        dataset,
                        [cfg_for_ticker],
                        engine=engine_mode,
                        require_trade_returns=True,
                    )[0]
                except Exception as exc:
                    valid = False
                    tracker.record_eval_failure()
                    tracker.log(f"Evaluation failed for {ticker}: {exc}")
                    break
            series = _series_from_history(stats.equity_history)
            if series.empty:
                idx = dataset.prices.index
                if len(idx) >= 2:
                    timestamps = [idx[0], idx[-1]]
                else:
                    timestamps = [
                        pd.Timestamp("1970-01-01"),
                        pd.Timestamp("1970-01-02"),
                    ]
                series = pd.Series(
                    [per_ticker_capital, float(stats.final_balance)],
                    index=timestamps,
                )
            stats_collection[ticker] = stats
            per_ticker_equities[ticker] = series
            per_ticker_returns[ticker] = series.pct_change().dropna()
            metrics_entry = precomputed_metrics.get((candidate_idx, ticker), {})
            native_ror = stats.risk_of_ruin
            if native_ror is None:
                native_ror = _as_opt_float(metrics_entry.get("risk_of_ruin"))
            if native_ror is None and stats.trade_returns:
                native_ror = compute_risk_of_ruin(
                    stats.trade_returns,
                    initial_capital=per_ticker_capital,
                    risk_per_trade=max(per_ticker_capital * candidate.risk_pct, 1e-6),
                )
            if native_ror is None:
                native_ror = 1.0
            per_ticker_details[ticker] = {
                "final_balance": stats.final_balance,
                "profit_factor": float(stats.profit_factor or 0.0),
                "sharpe": float(stats.sharpe_ratio or 0.0) if stats.sharpe_ratio is not None else 0.0,
                "max_drawdown_pct": float(stats.max_drawdown_pct),
                "n_trades": int(stats.n_trades),
                "risk_of_ruin": float(native_ror),
                "sortino": float(stats.sortino_ratio) if stats.sortino_ratio is not None else 0.0,
                "maxdd_p95": float(stats.maxdd_p95) if stats.maxdd_p95 is not None else 0.0,
                "pnl_p05": float(stats.pnl_p05) if stats.pnl_p05 is not None else 0.0,
                "pnl_p50": float(stats.pnl_p50) if stats.pnl_p50 is not None else 0.0,
                "pnl_p95": float(stats.pnl_p95) if stats.pnl_p95 is not None else 0.0,
                "robustness_score": float(stats.robustness_score) if stats.robustness_score is not None else 0.0,
            }

        if not valid or not per_ticker_equities:
            if valid and not per_ticker_equities:
                tracker.record_eval_failure()
            continue

        portfolio_weights = compute_portfolio_weights(
            per_ticker_equities,
            scheme=weighting_scheme,
        )
        portfolio_equity = build_portfolio_equity(
            per_ticker_equities,
            total_capital=search_cfg.initial_balance,
            weights=portfolio_weights,
        )
        if portfolio_equity.empty:
            continue

        portfolio_stats = compute_portfolio_stats(
            portfolio_equity,
            initial_balance=search_cfg.initial_balance,
            seed=search_cfg.seed,
            bootstrap_runs=search_cfg.bootstrap_runs,
        )

        equity_returns = portfolio_equity.pct_change().dropna()
        portfolio_equity_vol = float(equity_returns.std(ddof=1)) if len(equity_returns) >= 2 else 0.0

        corr_matrix = compute_correlation_matrix(per_ticker_returns)
        corr_mean, corr_max = _correlation_summary(corr_matrix)

        total_trades = sum(stats.n_trades for stats in stats_collection.values())
        wins_total = sum(stats.win_rate * stats.n_trades for stats in stats_collection.values())
        sentiment_fear = sum(stats.trades_in_fear for stats in stats_collection.values())
        sentiment_greed = sum(stats.trades_in_greed for stats in stats_collection.values())
        expectancy_per_trade = (
            (portfolio_stats["return_pct"] / 100.0 * search_cfg.initial_balance) / total_trades
            if total_trades
            else 0.0
        )
        benchmark_return_pct = float(np.mean(benchmark_values)) if benchmark_values else 0.0
        alpha_pct = portfolio_stats["return_pct"] - benchmark_return_pct
        robustness_values: list[float] = []
        for detail in per_ticker_details.values():
            value = detail.get("robustness_score")
            if value is None:
                continue
            if math.isfinite(value) and value > 0.0:
                robustness_values.append(value)
        aggregate_robustness = float(np.mean(robustness_values)) if robustness_values else 0.0

        result_row = SearchResult(
            config=candidate,
            strategy_return_pct=portfolio_stats["return_pct"],
            benchmark_return_pct=benchmark_return_pct,
            alpha_pct=alpha_pct,
            profit_factor=float(portfolio_stats["profit_factor"]),
            sharpe=float(portfolio_stats["sharpe"]),
            max_drawdown_pct=float(portfolio_stats["max_drawdown_pct"]),
            expectancy=expectancy_per_trade,
            n_trades=total_trades,
            win_rate_pct=float((wins_total / total_trades) * 100.0) if total_trades else 0.0,
            sentiment_stats={"trades_in_fear": sentiment_fear, "trades_in_greed": sentiment_greed},
            expectancy_per_trade=expectancy_per_trade,
            risk_of_ruin=float(portfolio_stats["risk_of_ruin"]),
            maxdd_p95=float(portfolio_stats["maxdd_p95"]),
            pnl_p05=float(portfolio_stats["pnl_p05"]),
            pnl_p95=float(portfolio_stats.get("pnl_p95", 0.0)),
            portfolio_return_pct=portfolio_stats["return_pct"],
            portfolio_profit_factor=float(portfolio_stats["profit_factor"]),
            portfolio_sharpe=float(portfolio_stats["sharpe"]),
            portfolio_max_drawdown_pct=float(portfolio_stats["max_drawdown_pct"]),
            portfolio_risk_of_ruin=float(portfolio_stats["risk_of_ruin"]),
            portfolio_maxdd_p95=float(portfolio_stats["maxdd_p95"]),
            portfolio_pnl_p05=float(portfolio_stats["pnl_p05"]),
            correlation_mean=corr_mean,
            correlation_max=corr_max,
            per_ticker_details=per_ticker_details,
            portfolio_weights=portfolio_weights,
            portfolio_equity_vol=portfolio_equity_vol,
            robustness_score=aggregate_robustness,
        )
        if not _passes_hard_constraints(search_cfg, result_row, tracker=tracker):
            continue
        results.append(result_row)

    _assign_base_scores(results)
    if meta_model is not None:
        calibrator = None
        calibrator_path = search_cfg.meta_calibration_path
        if search_cfg.meta_calibration_registry_key and not calibrator_path:
            from hyprl.meta.registry import resolve_model

            calibrator_path = str(resolve_model(search_cfg.meta_calibration_registry_key))
        if calibrator_path:
            calibrator = joblib.load(calibrator_path)
        _apply_meta_scores(results, meta_model, search_cfg.meta_weight, weighting_scheme, calibrator)
        results.sort(key=lambda res: (res.final_score, res.base_score), reverse=True)
    else:
        results.sort(key=lambda res: res.base_score, reverse=True)
    tracker.set_survivors(len(results))
    tracker.log_summary()
    return results


_RELAXED_NATIVE_CONSTRAINTS: dict[str, float] = {
    "min_trades": 0,
    "min_profit_factor": -1e6,
    "min_sharpe": -1e6,
    "max_drawdown": 1e6,
    "max_risk_of_ruin": 1.0,
    "min_expectancy": -1e6,
    "min_robustness": -1e6,
    "max_maxdd_p95": 1e6,
    "min_pnl_p05": -1e6,
    "min_pnl_p50": -1e6,
    "min_pnl_p95": -1e6,
}


def _group_candidates_by_signal(
    candidate_entries: list[tuple[CandidateConfig, BacktestConfig]]
) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for idx, (candidate, _) in enumerate(candidate_entries):
        key = (
            float(candidate.long_threshold),
            float(candidate.short_threshold),
            bool(candidate.trend_filter),
            float(candidate.sentiment_min),
            float(candidate.sentiment_max),
            str(candidate.sentiment_regime),
        )
        groups.setdefault(key, []).append(idx)
    return groups


def _as_opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _report_to_strategy_stats(
    report: Mapping[str, Any],
    *,
    initial_balance: float,
    trades_in_fear: int,
    trades_in_greed: int,
) -> StrategyStats:
    equity_points = report.get("equity_curve", []) or []
    equity_history: list[tuple[pd.Timestamp, float]] = []
    for point in equity_points:
        ts_raw = point.get("ts", 0)
        ts = pd.to_datetime(int(ts_raw), unit="ms", utc=False)
        equity_value = float(point.get("equity", 1.0)) * initial_balance
        equity_history.append((ts, equity_value))
    if not equity_history:
        now = pd.Timestamp.utcnow()
        equity_history = [(now, float(initial_balance))]
    final_balance = float(equity_history[-1][1])
    metrics = report.get("metrics", {})
    profit_factor = _as_opt_float(metrics.get("profit_factor"))
    sharpe = _as_opt_float(metrics.get("sharpe"))
    sortino = _as_opt_float(metrics.get("sortino"))
    risk_of_ruin = _as_opt_float(metrics.get("risk_of_ruin"))
    max_dd_ratio = float(metrics.get("max_drawdown", 0.0))
    expectancy = float(metrics.get("expectancy", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    n_trades = int(report.get("n_trades", 0))
    return StrategyStats(
        final_balance=final_balance,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=abs(max_dd_ratio) * 100.0,
        expectancy=expectancy,
        n_trades=n_trades,
        win_rate=win_rate,
        trades_in_fear=trades_in_fear,
        trades_in_greed=trades_in_greed,
        trade_returns=[],
        equity_history=equity_history,
        sortino_ratio=sortino,
        risk_of_ruin=risk_of_ruin,
        maxdd_p05=_as_opt_float(metrics.get("maxdd_p05")),
        maxdd_p95=_as_opt_float(metrics.get("maxdd_p95")),
        pnl_p05=_as_opt_float(metrics.get("pnl_p05")),
        pnl_p50=_as_opt_float(metrics.get("pnl_p50")),
        pnl_p95=_as_opt_float(metrics.get("pnl_p95")),
        robustness_score=_as_opt_float(metrics.get("robustness_score")),
    )


def _precompute_native_stats(
    tickers: list[str],
    datasets_by_ticker: Mapping[str, SupercalcDataset],
    candidate_entries: list[tuple[CandidateConfig, BacktestConfig]],
    per_ticker_capital: float,
) -> tuple[dict[tuple[int, str], StrategyStats], dict[tuple[int, str], dict[str, Any]]]:
    if not candidate_entries or not native_engine_available():
        return {}, {}
    stats_map: dict[tuple[int, str], StrategyStats] = {}
    metrics_map: dict[tuple[int, str], dict[str, Any]] = {}
    groups = _group_candidates_by_signal(candidate_entries)
    for ticker in tickers:
        dataset = datasets_by_ticker.get(ticker)
        if dataset is None:
            continue
        for idx_list in groups.values():
            template_cfg = replace(candidate_entries[idx_list[0]][1], ticker=ticker)
            signal, trades_in_fear, trades_in_greed = _build_signal_series(dataset, template_cfg)
            if not signal:
                signal = [0.0] * len(dataset.prices)
            cfg_batch = [replace(candidate_entries[idx][1], ticker=ticker) for idx in idx_list]
            labels = [str(idx) for idx in idx_list]
            reports = run_native_search_batch(
                dataset.prices,
                signal,
                cfg_batch,
                constraints=_RELAXED_NATIVE_CONSTRAINTS,
                top_k=len(cfg_batch),
                labels=labels,
            )
            for report in reports:
                label = report.get("config", {}).get("label")
                if label is None:
                    continue
                try:
                    candidate_idx = int(label)
                except (TypeError, ValueError):
                    continue
                stats_map[(candidate_idx, ticker)] = _report_to_strategy_stats(
                    report,
                    initial_balance=per_ticker_capital,
                    trades_in_fear=trades_in_fear,
                    trades_in_greed=trades_in_greed,
                )
                metrics = report.get("metrics", {})
                if isinstance(metrics, dict):
                    metrics_map[(candidate_idx, ticker)] = metrics
    return stats_map, metrics_map


def save_results_csv(
    results: Iterable[SearchResult],
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    base_fields = [
        "long_threshold",
        "short_threshold",
        "risk_pct",
        "min_ev_multiple",
        "trend_filter",
        "sentiment_min",
        "sentiment_max",
        "sentiment_regime",
        "strategy_return_pct",
        "benchmark_return_pct",
        "alpha_pct",
        "profit_factor",
        "sharpe",
        "max_drawdown_pct",
        "expectancy",
        "n_trades",
        "win_rate_pct",
        "trades_in_fear",
        "trades_in_greed",
        "expectancy_per_trade",
        "risk_of_ruin",
        "maxdd_p95",
        "pnl_p05",
        "pnl_p95",
        "portfolio_return_pct",
        "portfolio_profit_factor",
        "portfolio_sharpe",
        "portfolio_max_drawdown_pct",
        "portfolio_risk_of_ruin",
        "portfolio_maxdd_p95",
        "portfolio_pnl_p05",
        "correlation_mean",
        "correlation_max",
        "tickers",
        "portfolio_weights",
        "portfolio_equity_vol",
        "robustness_score",
        "base_score",
        "meta_prediction",
        "final_score",
    ]
    rows: list[dict[str, Any]] = []
    for res in results:
        tickers = sorted(res.per_ticker_details.keys())
        row = {
            "long_threshold": res.config.long_threshold,
            "short_threshold": res.config.short_threshold,
            "risk_pct": res.config.risk_pct,
            "min_ev_multiple": res.config.min_ev_multiple,
            "trend_filter": res.config.trend_filter,
            "sentiment_min": res.config.sentiment_min,
            "sentiment_max": res.config.sentiment_max,
            "sentiment_regime": res.config.sentiment_regime,
            "strategy_return_pct": res.strategy_return_pct,
            "benchmark_return_pct": res.benchmark_return_pct,
            "alpha_pct": res.alpha_pct,
            "profit_factor": res.profit_factor,
            "sharpe": res.sharpe,
            "max_drawdown_pct": res.max_drawdown_pct,
            "expectancy": res.expectancy,
            "n_trades": res.n_trades,
            "win_rate_pct": res.win_rate_pct,
            "trades_in_fear": res.sentiment_stats.get("trades_in_fear", 0),
            "trades_in_greed": res.sentiment_stats.get("trades_in_greed", 0),
            "expectancy_per_trade": res.expectancy_per_trade,
            "risk_of_ruin": res.risk_of_ruin,
            "maxdd_p95": res.maxdd_p95,
            "pnl_p05": res.pnl_p05,
            "pnl_p95": res.pnl_p95,
            "portfolio_return_pct": res.portfolio_return_pct,
            "portfolio_profit_factor": res.portfolio_profit_factor,
            "portfolio_sharpe": res.portfolio_sharpe,
            "portfolio_max_drawdown_pct": res.portfolio_max_drawdown_pct,
            "portfolio_risk_of_ruin": res.portfolio_risk_of_ruin,
            "portfolio_maxdd_p95": res.portfolio_maxdd_p95,
            "portfolio_pnl_p05": res.portfolio_pnl_p05,
            "correlation_mean": res.correlation_mean,
            "correlation_max": res.correlation_max,
            "tickers": ",".join(tickers),
            "portfolio_weights": json.dumps(res.portfolio_weights, sort_keys=True),
            "portfolio_equity_vol": res.portfolio_equity_vol,
            "robustness_score": res.robustness_score,
            "base_score": res.base_score,
            "meta_prediction": res.meta_prediction if res.meta_prediction is not None else "",
            "final_score": res.final_score,
        }
        if metadata:
            for key, value in metadata.items():
                row.setdefault(key, value)
        rows.append(row)
    if rows:
        df = pd.DataFrame.from_records(rows)
    else:
        columns = list(base_fields)
        if metadata:
            columns.extend(key for key in metadata.keys() if key not in columns)
        df = pd.DataFrame(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
