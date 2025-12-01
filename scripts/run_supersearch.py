#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from hyprl.labels.amplitude import LabelConfig
from hyprl.search.optimizer import (
    CandidateConfig,
    SearchConfig,
    SearchResult,
    run_search,
    save_results_csv,
)

_SEARCH_CFG_DEFAULTS = SearchConfig.__dataclass_fields__
_DEFAULT_MIN_SHARPE = _SEARCH_CFG_DEFAULTS["min_sharpe"].default
_DEFAULT_MAX_DD = _SEARCH_CFG_DEFAULTS["max_drawdown_pct"].default
_DEFAULT_MAX_PORTFOLIO_DD = _SEARCH_CFG_DEFAULTS["max_portfolio_drawdown_pct"].default
_DEFAULT_MAX_ROR = _SEARCH_CFG_DEFAULTS["max_risk_of_ruin"].default
_DEFAULT_MAX_PORTFOLIO_ROR = _SEARCH_CFG_DEFAULTS["max_portfolio_risk_of_ruin"].default


def _parse_float_list(payload: str | None) -> list[float]:
    if not payload:
        return []
    values: list[float] = []
    for token in payload.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _parse_bool_list(payload: str | None) -> list[bool] | None:
    if not payload:
        return None
    mapping = {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}
    values: list[bool] = []
    for token in payload.split(","):
        token = token.strip().lower()
        if token not in mapping:
            continue
        values.append(mapping[token])
    return values or None


def _parse_str_list(payload: str | None) -> list[str]:
    if not payload:
        return []
    return [token.strip() for token in payload.split(",") if token.strip()]


def _synthetic_results(mode: str) -> list[SearchResult]:
    if mode == "fail":
        pf, sharpe, maxdd, expectancy = 0.95, 0.2, 25.0, -15.0
        n_trades, win_rate = 18, 41.0
        fear, greed = 6, 9
    else:  # "ok"
        pf, sharpe, maxdd, expectancy = 1.35, 1.05, 11.0, 42.0
        n_trades, win_rate = 56, 58.0
        fear, greed = 3, 4
    candidate = CandidateConfig(
        long_threshold=0.58,
        short_threshold=0.42,
        risk_pct=0.015,
        min_ev_multiple=0.1,
        trend_filter=True,
        sentiment_min=-0.4,
        sentiment_max=0.6,
        sentiment_regime="neutral_only",
    )
    return [
        SearchResult(
            config=candidate,
            strategy_return_pct=18.4 if mode != "fail" else -3.1,
            benchmark_return_pct=11.2,
            alpha_pct=7.2 if mode != "fail" else -14.3,
            profit_factor=pf,
            sharpe=sharpe,
            max_drawdown_pct=maxdd,
            expectancy=expectancy,
            n_trades=n_trades,
            win_rate_pct=win_rate,
            sentiment_stats={"trades_in_fear": fear, "trades_in_greed": greed},
        )
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HyprL supersearch – grille multi-paramètres avec filtres de sentiment."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--tickers", help="Liste de tickers (portefeuille) séparés par des virgules.")
    parser.add_argument("--period", help="Fenêtre simple type 1y/6mo (optionnel si start/end).")
    parser.add_argument("--start", help="Date de début ISO (optionnel).")
    parser.add_argument("--end", help="Date de fin ISO (optionnel).")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-presets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Active les presets YAML (désactiver avec --no-use-presets).",
    )
    parser.add_argument(
        "--constraint-preset",
        help="Nom du preset de contraintes (configs/supersearch_presets.yaml).",
    )
    parser.add_argument("--long-thresholds", help="Liste ex: 0.55,0.6,0.65")
    parser.add_argument("--short-thresholds", help="Liste ex: 0.35,0.4")
    parser.add_argument("--risk-pcts", help="Liste ex: 0.01,0.015")
    parser.add_argument("--min-ev-multiples", help="Liste ex: 0.0,0.25")
    parser.add_argument("--trend-filter-flags", help="Liste bool ex: true,false")
    parser.add_argument("--sentiment-min-values", help="Liste bornes min (score).")
    parser.add_argument("--sentiment-max-values", help="Liste bornes max (score).")
    parser.add_argument(
        "--sentiment-regimes",
        help="Comma list (off,fear_only,greed_only,neutral_only).",
        default="off",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "python", "native"],
        default="auto",
        help="Force l'engine (auto=fallback natif->python).",
    )
    parser.add_argument("--min-trades", type=int, default=50, help="Trades minimum pour valider une config.")
    parser.add_argument("--min-pf", type=float, default=1.2, help="Profit factor minimal.")
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=None,
        help="Sharpe minimal (laisser vide pour le défaut SearchConfig, passer 0.0 pour désactiver).",
    )
    parser.add_argument(
        "--max-dd",
        type=float,
        default=None,
        help="Max drawdown autorisé (ratio, ex 0.35).",
    )
    parser.add_argument(
        "--max-ror",
        type=float,
        default=None,
        help="Risque de ruine maximal (0-1).",
    )
    parser.add_argument(
        "--min-robustness-score",
        type=float,
        default=0.0,
        help="Score de robustesse minimal (0-1).",
    )
    parser.add_argument("--min-expectancy", type=float, default=0.0, help="Espérance minimale par trade.")
    parser.add_argument("--bootstrap-runs", type=int, default=256, help="Tirs Monte Carlo pour le stress test.")
    parser.add_argument("--min-portfolio-pf", type=float, default=0.0, help="Profit factor minimal du portefeuille.")
    parser.add_argument("--min-portfolio-sharpe", type=float, default=-10.0, help="Sharpe minimal du portefeuille.")
    parser.add_argument(
        "--max-portfolio-dd",
        type=float,
        default=None,
        help="Max drawdown portefeuille (ratio). Par défaut suit --max-dd ou reste à %.2f."
        % _DEFAULT_MAX_PORTFOLIO_DD,
    )
    parser.add_argument(
        "--max-portfolio-ror",
        type=float,
        default=None,
        help="RoR portefeuille max (0-1). Par défaut suit --max-ror ou reste à %.2f."
        % _DEFAULT_MAX_PORTFOLIO_ROR,
    )
    parser.add_argument("--max-correlation", type=float, default=1.0, help="Corrélation max entre tickers (0-1).")
    parser.add_argument(
        "--weighting-scheme",
        choices=["equal", "inv_vol"],
        default="equal",
        help="Schéma de pondération portefeuille (equal ou inverse volatilité).",
    )
    parser.add_argument(
        "--meta-robustness",
        help="Chemin vers le modèle Meta-ML (joblib) utilisé pour re-ranker les configurations.",
    )
    parser.add_argument(
        "--meta-registry",
        help="Clé@Stage ou alias dans le registre (ex: robustness@Production, robustness@stable).",
    )
    parser.add_argument(
        "--meta-weight",
        type=float,
        default=0.4,
        help="Poids [0-1] accordé au score Meta-ML dans le ranking final.",
    )
    parser.add_argument(
        "--meta-calibration",
        help="Calibrateur joblib (optionnel).",
    )
    parser.add_argument(
        "--meta-calibration-registry",
        help="Clé@Stage ou alias pour le calibrateur.",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV de sortie.")
    parser.add_argument("--top", type=int, default=5, help="Nombre de lignes à afficher.")
    parser.add_argument(
        "--drysim",
        choices=["off", "ok", "fail"],
        default="off",
        help="Mode synthétique pour tests hors-ligne.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "amplitude"],
        default="binary",
        help="Mode de label (binary/amplitude).",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=4,
        help="Horizon en barres pour labels amplitude.",
    )
    parser.add_argument(
        "--label-threshold-pct",
        type=float,
        default=1.5,
        help="Mouvement %% requis pour BIG_UP/BIG_DOWN (default 1.5).",
    )
    parser.add_argument(
        "--label-neutral-strategy",
        choices=["ignore", "keep"],
        default="ignore",
        help="Gestion des labels NEUTRAL (default ignore).",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=40,
        help="Samples minimum BIG_UP/BIG_DOWN avant entrainement.",
    )
    parser.add_argument(
        "--debug-search",
        action="store_true",
        help="Logge les paramètres et le comptage de grilles pour debug.",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> SearchConfig:
    trend_flags = _parse_bool_list(args.trend_filter_flags)
    sentiment_regimes = _parse_str_list(args.sentiment_regimes)
    tickers = _parse_str_list(args.tickers)
    label_cfg = LabelConfig(
        mode=args.label_mode,
        horizon=args.label_horizon,
        threshold_pct=args.label_threshold_pct,
        neutral_strategy=args.label_neutral_strategy,
        min_samples_per_class=args.min_samples_per_class,
    )
    max_dd_arg = args.max_dd
    max_dd_value = max_dd_arg if max_dd_arg is not None else _DEFAULT_MAX_DD
    if args.max_portfolio_dd is not None:
        max_portfolio_dd_value = args.max_portfolio_dd
    elif max_dd_arg is not None:
        max_portfolio_dd_value = max_dd_value
    else:
        max_portfolio_dd_value = _DEFAULT_MAX_PORTFOLIO_DD

    max_ror_arg = args.max_ror
    max_ror_value = max_ror_arg if max_ror_arg is not None else _DEFAULT_MAX_ROR
    if args.max_portfolio_ror is not None:
        max_portfolio_ror_value = args.max_portfolio_ror
    elif max_ror_arg is not None:
        max_portfolio_ror_value = max_ror_value
    else:
        max_portfolio_ror_value = _DEFAULT_MAX_PORTFOLIO_ROR

    min_sharpe_value = args.min_sharpe if args.min_sharpe is not None else _DEFAULT_MIN_SHARPE

    return SearchConfig(
        ticker=args.ticker,
        tickers=tickers,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        seed=args.seed,
        use_presets=args.use_presets,
        constraint_preset=args.constraint_preset,
        long_thresholds=_parse_float_list(args.long_thresholds),
        short_thresholds=_parse_float_list(args.short_thresholds),
        risk_pcts=_parse_float_list(args.risk_pcts),
        min_ev_multiples=_parse_float_list(args.min_ev_multiples),
        trend_filter_flags=trend_flags,
        sentiment_min_values=_parse_float_list(args.sentiment_min_values),
        sentiment_max_values=_parse_float_list(args.sentiment_max_values),
        sentiment_regimes=sentiment_regimes,
        engine=args.engine,
        min_trades=args.min_trades,
        min_profit_factor=args.min_pf,
        min_sharpe=min_sharpe_value,
        max_drawdown_pct=max_dd_value,
        max_risk_of_ruin=max_ror_value,
        min_robustness_score=args.min_robustness_score,
        min_expectancy=args.min_expectancy,
        bootstrap_runs=args.bootstrap_runs,
        min_portfolio_profit_factor=args.min_portfolio_pf,
        min_portfolio_sharpe=args.min_portfolio_sharpe,
        max_portfolio_drawdown_pct=max_portfolio_dd_value,
        max_portfolio_risk_of_ruin=max_portfolio_ror_value,
        max_correlation=args.max_correlation,
        weighting_scheme=args.weighting_scheme,
        meta_robustness_model_path=args.meta_robustness,
        meta_weight=args.meta_weight,
        meta_registry_key=args.meta_registry,
        meta_calibration_path=args.meta_calibration,
        meta_calibration_registry_key=args.meta_calibration_registry,
        label=label_cfg,
        debug=args.debug_search,
    )


def _print_summary(results: Iterable[SearchResult], top: int) -> None:
    print("[INFO] Top configurations (triées par PF/Sharpe/DD):")
    for rank, res in enumerate(results, start=1):
        if rank > top:
            break
        cfg = res.config
        stats = res.sentiment_stats or {}
        print(
            f"#{rank} PF={res.portfolio_profit_factor:.2f} Sharpe={res.portfolio_sharpe:.2f} "
            f"PortDD={res.portfolio_max_drawdown_pct:.2f}% Trades={res.n_trades} "
            f"Sentiment=[{cfg.sentiment_min:.2f},{cfg.sentiment_max:.2f}] "
            f"Regime={cfg.sentiment_regime} "
            f"fear={stats.get('trades_in_fear', 0)} greed={stats.get('trades_in_greed', 0)} "
            f"Exp={res.expectancy_per_trade:.4f} ROR={res.portfolio_risk_of_ruin:.2%} "
            f"DD95={res.portfolio_maxdd_p95:.2f}% CorrMax={res.correlation_max:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.drysim != "off":
        results = _synthetic_results(args.drysim)
    else:
        config = _build_config(args)
        if args.debug_search:
            print(
                "[DEBUG] SearchConfig core: "
                f"ticker={config.ticker} tickers={config.tickers or [config.ticker]} "
                f"period={config.period or ''} start={config.start or ''} end={config.end or ''} "
                f"interval={config.interval} engine={config.engine} presets={config.use_presets} "
                f"constraint_preset={config.constraint_preset or 'none'}"
            )
            min_ev = config.min_ev_multiples if config.min_ev_multiples is not None else "[preset]"
            print(
                "[DEBUG] Grid inputs: "
                f"longs={config.long_thresholds or '[preset]'} "
                f"shorts={config.short_thresholds or '[preset]'} "
                f"risks={config.risk_pcts or '[preset]'} "
                f"min_ev={min_ev} trend_flags={config.trend_filter_flags or '[default]'} "
                f"sent_min={config.sentiment_min_values or '[preset]'} "
                f"sent_max={config.sentiment_max_values or '[preset]'} "
                f"regimes={config.sentiment_regimes or '[preset]'}"
            )
            print(f"[DEBUG] Label config: {asdict(config.label)}")
            print(
                "[DEBUG] Hard constraints: "
                f"min_trades={config.min_trades} "
                f"min_pf={config.min_profit_factor:.2f} "
                f"min_sharpe={config.min_sharpe:.2f} "
                f"max_dd={config.max_drawdown_pct:.3f} "
                f"max_ror={config.max_risk_of_ruin:.3f} "
                f"min_robustness={config.min_robustness_score:.2f}"
            )
            print(f"[DEBUG] Output target: {args.output}")
        results = run_search(config)
        if not results:
            print(
                "[WARN] Supersearch n'a produit aucune configuration valide. "
                "Assouplissez --min-pf/--min-sharpe ou élargissez les grilles."
            )
            metadata = {
                "primary_ticker": args.ticker,
                "search_interval": args.interval,
                "search_period": args.period or "",
                "search_start": args.start or "",
                "search_end": args.end or "",
                "search_tickers": args.tickers or args.ticker,
                "search_initial_balance": args.initial_balance,
                "search_weighting_scheme": args.weighting_scheme,
                "filters_active": (
                    f"min_pf={args.min_pf}, min_sharpe={args.min_sharpe}, "
                    f"max_dd={args.max_dd}, max_ror={args.max_ror}"
                ),
            }
            save_results_csv([], args.output, metadata=metadata)
            print(f"[INFO] Fichier vide écrit dans {args.output}")
            return 0

    metadata = {
        "primary_ticker": args.ticker,
        "search_interval": args.interval,
        "search_period": args.period or "",
        "search_start": args.start or "",
        "search_end": args.end or "",
        "search_tickers": args.tickers or args.ticker,
        "search_initial_balance": args.initial_balance,
        "search_weighting_scheme": args.weighting_scheme,
    }
    save_results_csv(results, args.output, metadata=metadata)
    _print_summary(results, args.top)
    print(f"[OK] Résultats exportés vers {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
