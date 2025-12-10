# src/hyprl/backtest/runner.py - Améliorations documentées
• [Context] →
Supersearch CLI produisait des CSV vides alors que run_search(...) manuel renvoyait des survivants. Nouveaux outils de debug (--debug-search + SearchDebugTracker) en place.

• [Findings] →
(1) 1h + date range fixe (2023-01-01 → 2024-01-01) casse sur Yahoo >730j → dataset failures. Avec --period 1y ou cache pré-rempli, le dataset se construit correctement.
(2) Différence CLI vs manuel due à des gates portefeuille plus stricts : CLI relaxait max_dd/max_ror par stratégie mais pas max_portfolio_dd/max_portfolio_ror, alors que run_search utilisait les defaults SearchConfig (1.0 / 1.0).

• [Action] →
Aligné run_supersearch.py et SearchConfig :
- --use-presets par défaut True (config centrée dans SearchConfig).
- --max-dd/--max-ror maintenant optionnels : si non fournis, on garde les defaults SearchConfig ; s’ils sont fournis, les limites portefeuille héritent des mêmes valeurs sauf override explicite.
- Ajout d’un test CLI-level (test_supersearch_cli_limits_follow_global_flags) pour vérifier qu’un run CLI avec contraintes lâches produit des survivants dès qu’un run_search équivalent en produit.

• [Result] →
- En utilisant --period 1y + binary labels, supersearch CLI produit des CSV non vides quand run_search(...) le fait.
- Les “no survivors” dus uniquement aux gates portefeuille cachés sont éliminés (CLI et manuel sont cohérents sur les limites de risque).

• [Next Steps] →
- Standardiser les commandes : utiliser --period 1y --interval 1h pour la recherche AAPL/MSFT/NVDA (ou pré-seeder le cache pour des périodes plus longues).
- Lancer deux runs supersearch alignés CLI vs manuel :
  (1) binary labels baseline,
  (2) amplitude labels (horizon=4, threshold=1.5),
  puis comparer PF, expectancy et nb de trades à partir des CSV générés via CLI.

from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """Configuration for backtest execution with cost modeling.
    
    Cost Model:
    -----------
    - commission_pct: Percentage of notional value charged as commission per trade
                      (e.g., 0.0005 = 0.05% = 5 basis points)
    - slippage_pct: Percentage of notional value approximating execution slippage
                    (e.g., 0.0005 = 0.05% = 5 basis points)
    
    Both costs are applied to each trade side (entry and exit), so the total
    round-trip cost is approximately 2 * (commission_pct + slippage_pct).
    
    For example, with defaults:
    - commission_pct = 0.0005 (0.05%)
    - slippage_pct = 0.0005 (0.05%)
    - Total cost per side = 0.001 (0.10%)
    - Total round-trip cost ≈ 0.002 (0.20%)
    """
    # Existing fields...
    
    # Cost parameters with clear defaults
    commission_pct: float = 0.0005  # 0.05% commission per trade
    slippage_pct: float = 0.0005    # 0.05% slippage approximation
    
    def get_total_cost_rate(self) -> float:
        """Returns the combined commission + slippage rate per trade side.
        
        Returns:
            float: Total cost rate (e.g., 0.001 = 0.10%)
        """
        return self.commission_pct + self.slippage_pct


class BacktestRunner:
    def __init__(self, config: BacktestConfig):
        self.config = config
        # Cache the total rate for performance
        self._total_cost_rate = config.get_total_cost_rate()
    
    def _execute_trade(self, side: str, price: float, qty: float) -> tuple[float, float]:
        """Execute a trade with costs applied.
        
        Args:
            side: 'BUY' or 'SELL'
            price: Execution price (close of signal bar)
            qty: Position size
        
        Returns:
            tuple: (fill_price, total_cost)
        """
        notional = price * qty
        
        # Apply combined commission + slippage as a percentage adjustment
        # total_rate ~ 0.001 => ~0.10% combined commission + slippage
        if side == 'BUY':
            # Pay more when buying (price increases)
            fill_price = price * (1 + self._total_cost_rate)
        else:  # SELL
            # Receive less when selling (price decreases)
            fill_price = price * (1 - self._total_cost_rate)
        
        # Total cost is the difference between ideal and actual fill
        total_cost = abs(fill_price - price) * qty
        
        return fill_price, total_cost# PROJECT_BRAIN — Procédure Locale

## Environnement
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `pip install -e .`

Le fichier `requirements.txt` reprend les dépendances runtime + dev (pytest). Garder `.venv/` à la racine pour que les wrappers puissent l'activer automatiquement.

## Tests
- Jeu complet : `source .venv/bin/activate && pytest -q`
- Seuils configs : `source .venv/bin/activate && pytest tests/configs/test_configs.py -k threshold -q`
- Wrapper (active `.venv` et lance pytest, arguments transmis tels quels) :
  - `./scripts/run_tests.sh` (équivalent `pytest -q`)
  - `./scripts/run_tests.sh tests/configs/test_configs.py -k threshold -q`

## Supercalc (Rust)
- Build dépendances : `pip install maturin` dans `.venv`.
- Compilation/installation : `bash scripts/build_supercalc.sh` (wrappe `maturin develop --release` dans `native/hyprl_supercalc`).
- Tests rapides :
  - `source .venv/bin/activate && pytest tests/backtest/test_supercalc_native.py -q`
  - `python scripts/run_supersearch.py ... --engine auto` pour vérifier le fallback.

## Risk Layer & Robustesse
- `src/hyprl/risk/metrics.py` calcule expectancy, winrate, risk-of-ruin approximatif et bootstrap Monte Carlo sur les trade_returns.
- `run_search` applique des **hard constraints** (min trades, PF, Sharpe, max DD, ROR, expectancy) avant de scorer les configs.
- Le score multi-objectifs pénalise `risk_of_ruin` et `maxDD_p95`, tout en bonifiant l’espérance par trade.
- CLI `run_supersearch.py` expose les flags:
  - `--min-trades`, `--min-pf`, `--min-sharpe`,
  - `--max-dd`, `--max-ror`, `--min-expectancy`,
  - `--bootstrap-runs` pour contrôler le Monte Carlo.
- Exemple :
  ```
  python scripts/run_supersearch.py --ticker AAPL --period 1y --interval 1h \
    --initial-balance 10000 --seed 42 \
    --long-thresholds "0.55,0.6" --short-thresholds "0.35,0.4" \
    --risk-pcts "0.01,0.015" \
    --sentiment-min-values "-0.4,-0.2" --sentiment-max-values "0.2,0.5" \
    --sentiment-regimes "off,neutral_only" \
    --min-trades 50 --min-pf 1.2 --min-sharpe 0.8 \
    --max-dd 0.35 --max-ror 0.1 --min-expectancy 0.0 \
    --engine auto --output data/supersearch_AAPL_1y_risklayer.csv
  ```

## Portfolio Layer (multi-tickers)
- `SearchConfig.tickers` (et `--tickers` côté CLI) permettent d’exécuter une même stratégie sur plusieurs actifs en parallèle.
- `src/hyprl/portfolio/core.py` aligne les courbes d’equity par ticker, applique les poids (égal par défaut) et calcule PF/Sharpe/DD/RoR au niveau portefeuille + corrélations.
- Hard constraints supplémentaires : `--min-portfolio-pf`, `--min-portfolio-sharpe`, `--max-portfolio-dd`, `--max-portfolio-ror`, `--max-correlation`.
- Les métriques portefeuille sont intégrées dans `SearchResult` (`portfolio_*`) et exportées dans le CSV (ainsi que `per_ticker_details` pour inspecter chaque actif).
- Exemple :
  ```
  python scripts/run_supersearch.py --tickers "AAPL,MSFT,GOOGL" --ticker AAPL \
    --period 1y --interval 1h --initial-balance 10000 --seed 42 \
    --long-thresholds "0.55,0.6" --short-thresholds "0.35,0.4" \
    --risk-pcts "0.01,0.015" \
    --sentiment-min-values "-0.4,-0.2" --sentiment-max-values "0.2,0.5" \
    --sentiment-regimes "off,neutral_only" \
    --min-trades 50 --min-pf 1.2 --min-sharpe 0.8 \
    --max-dd 0.35 --max-ror 0.1 --min-expectancy 0.0 \
    --min-portfolio-pf 1.2 --min-portfolio-sharpe 0.8 \
    --max-portfolio-dd 0.35 --max-portfolio-ror 0.1 --max-correlation 0.8 \
    --engine auto --output data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv
  ```

## Execution Layer & Paper Trading
- `src/hyprl/execution/` regroupe le broker papier, le logger et l'engine de replay.
- `scripts/run_paper_trading.py` lit un CSV Supersearch, reconstruit la stratégie, exécute les trades (mode replay) via `PaperBroker`, et logge sous `data/live/sessions/<session>/`.
- `scripts/hyprl_dashboard.py` (Streamlit) visualise les fichiers `equity.csv` et `trades.csv`.
- Exemples :
  ```bash
  python scripts/run_paper_trading.py \
    --tickers "AAPL,MSFT" \
    --period 1y --interval 1h \
    --initial-balance 10000 \
    --config-csv data/supersearch_portfolio_AAPL_MSFT_1y.csv \
    --config-index 0 --engine auto

  streamlit run scripts/hyprl_dashboard.py
  ```

## Phase 1 — Validation scientifique
Objectif : sélectionner un petit **panel** de stratégies portefeuille, les rejouer en mode paper et comparer systématiquement backtest vs live.

1. **Construire le panel** (filtrage PF/Sharpe/DD/RoR/corrélation) :
   ```bash
   source .venv/bin/activate
   python scripts/build_phase1_panel.py \
     --csv-paths "data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv,data/supersearch_portfolio_BTC_ETH_1y.csv" \
     --max-strategies 5
   ```
   Résultat : `docs/experiments/PHASE1_PANEL.csv` (colonnes `strat_id`, `source_csv`, `config_index`, tickers, métriques backtest).

2. **Lancer les sessions paper** (replay) :
   ```bash
   source .venv/bin/activate
   python scripts/run_phase1_experiments.py \
     --period 1y --interval 1h --initial-balance 10000
   ```
   Pour chaque ligne du panel :
   - reconstruit les `BacktestConfig` depuis le CSV Supersearch,
   - relance `run_backtest`, rejoue les trades via `PaperBroker`,
   - logge sous `data/live/sessions/<session_id>/`,
   - enregistre la session dans `docs/experiments/PHASE1_SESSIONS.csv`.

3. **Analyser les sessions** :
   ```bash
   source .venv/bin/activate
   python scripts/analyze_phase1_results.py
   ```
   - Relit les métriques backtest (`load_backtest_metrics`),
   - Recalcule PF/Sharpe/DD/expectancy live à partir des logs (`load_live_metrics`),
   - Produit `data/experiments/phase1_results.csv` avec ratios PF/Sharpe/DD et un `robustness_score` borné [0,1].

4. **Journal humain** : `docs/experiments/PHASE1_LOG.md` (jour J0, J+7, J+14…) consigne PF/Sharpe/DD live vs backtest + commentaires. Sert de trace scientifique pour décider si une stratégie reste dans la shortlist.

Critères recommandés pour conserver une stratégie après plusieurs semaines :
- `robustness_score_moyen >= 0.7`
- `PF_live_moyen >= 1.3`
- `Sharpe_live_moyen >= 1.0`
- `maxDD_live_max <= 0.35`
- aucune séquence de drawdown singulière (>30%).

### Registry aliases & history
- `python scripts/registry_list.py --key robustness --dump-json` → affiche versions actives, alias, snapshot JSON minimal.
- `python scripts/registry_set_alias.py --key robustness --alias stable --version v0.1.2` pour geler une version («stable», «latest», etc.).
- `python scripts/registry_delete_alias.py --key robustness --alias stable` retire l’alias.
- `python scripts/promote_meta_model.py --key robustness --version v0.1.2 --stage Production` continue d’ajuster les stages officiels.
Toutes les opérations (register/promote/alias_set/alias_delete) appellent un `history[]` consultable via `scripts/registry_list.py --dump-json` ou directement dans `artifacts/meta_ml/registry.json`.

### Orchestrateur Phase-1 auto
- `python scripts/run_phase1_from_autorank.py --autoranked docs/experiments/AAA_autoranked.csv --max-strategies 5 --initial-balance 15000 --session-prefix phase1_auto`
  - Entrées : CSV autoranked **ou** `--csv ...` + `--meta-registry KEY@Production` (alias acceptés) + `--meta-calibration-registry KEY@stable` (optionnel).
  - Filtres : mêmes flags que `autorank_supersearch.py` (`--min-portfolio-pf`, `--max-portfolio-dd`, `--min-weight-per-ticker`, etc.).
  - Sorties (dans `docs/experiments/RUN_<timestamp>/`) :
    - `PHASE1_PANEL.csv` (+ `.meta.json`),
    - `PHASE1_SESSIONS.csv` (+ `.meta.json`),
    - `phase1_results.csv` (+ `.meta.json`),
    - `SUMMARY.txt`, `provenance.json`.
  - Papier et analyse sont déclenchés inline via `hyprl.phase1.paper.run_phase1_sessions` puis `hyprl.phase1.results.build_phase1_results`.

## Notes
- Aucun flux Poetry requis désormais.
- Les scripts CLI (`python scripts/*.py`) supposent simplement que `.venv` est activé.
- Mettre à jour ce document à chaque changement d'outillage local (install/tests).

## Realtime (Alpaca Paper) — MVP
- **ENV requis** : `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ALPACA_BASE_URL` (paper), `ALPACA_DATA_WS` (ex: `wss://stream.data.alpaca.markets/v2/iex`).
- **Dry-run** (par défaut) :
  ```bash
  python scripts/run_realtime_paper.py \
    --provider alpaca \
    --symbols "AAPL,MSFT" \
    --interval 1m --threshold 0.52 --risk-pct 0.25 \
    --session-id auto --dry-run
  ```
  Ajouter `--enable-paper` pour soumettre de vrais ordres «paper» (aucun capital réel engagé).
- **Logs** : `data/live/sessions/<session_id>/` contient `events.jsonl`, `bars.jsonl`, `features.jsonl`, `predictions.jsonl`, `orders.jsonl`, `fills.jsonl`, `equity.jsonl` + `session_manifest.json` (config). Tout est NDJSON → facile à rejouer.
- **Analyse** :
  ```bash
  python scripts/analyze_live_session.py --session data/live/sessions/<session_id> --output data/live/reports/<session_id>.csv
  ```
  Produit `live_report.csv` + `SUMMARY.txt` avec PF/Sharpe/DD. MVP = recherche uniquement (pas de conseil financier).

### Incrément 1 – Features réelles & clamps légers
- La boucle RT calcule désormais les features techniques complètes (`src/hyprl/rt/features.py` : SMA(14/50), RSI(14), MACD(12/26), Bollinger 20x2σ, ATR(14), trend_ratio, volatility) et entraîne un `ProbabilityModel` sur la fenêtre locale pour produire `prob_up` (comparé à `--threshold`).
- Meta-ML optionnel : `--meta-robustness path` ou `--meta-registry key@Stage` (idem calibrateur) → `meta_pred` loggué dans `predictions.jsonl`.
- Clamps légers : `--max-orders-per-min`, `--per-symbol-cap`, `--min-qty/--max-qty`. Toute décision loggue `reason` (`warmup`, `nan_features`, `qty_clamp`, `rate_cap`, `signal`, etc.).
- Fichiers JSONL enrichis : `predictions.jsonl` contient features clés, prob_up, threshold, qty/stop/tp, meta_pred, reason. `orders.jsonl` précise `paper`, `client_id`, `price_ref`.
- Analyzer (`scripts/analyze_live_session.py`) calcule PF/Sharpe/DD + `avg_hold_bars`, `exposure`, `winrate`, et affiche le top 5 des reasons.
- Commande type :
  ```bash
  python scripts/run_realtime_paper.py --provider alpaca \
    --symbols "AAPL,MSFT,GOOGL" --interval 1m --threshold 0.52 --risk-pct 0.25 \
    --max-orders-per-min 10 --per-symbol-cap 3 --min-qty 1 --dry-run
  ```

## Provenance & Sidecars
- `src/hyprl/utils/provenance.py` fournit `save_dataframe_with_meta` (CSV + `.meta.json` contenant timestamp, colonnes, hash SHA-256) et `stamp_provenance` (provenance.json consolidant modèle/calibrateur/registry/flags/input hashes/git hash).
- Tous les artefacts Phase-1 générés par les scripts (`PHASE1_PANEL.csv`, `PHASE1_SESSIONS.csv`, `phase1_results.csv`) sont accompagnés d’un sidecar `.meta.json`. `run_phase1_from_autorank.py` ajoute aussi `provenance.json` décrivant le modèle (clé/alias/stage), le calibrateur (méthode, métriques avant/après), le snapshot registry filtré ainsi que les flags CLI utilisés.

## Historical Log

[Log]
- Date: 2025-11-15
- Files: tests/backtest/test_no_lookahead.py, src/hyprl/risk/exposure.py, tests/risk/test_exposure_manager.py, docs/SUPERSEARCH_SCORING.md, tests/phase1/test_robustness_metrics.py, src/hyprl/backtest/runner.py, src/hyprl/search/optimizer.py, src/hyprl/analysis/phase1.py, src/hyprl/phase1/results.py
- Change: Infra patches préparatoires (5 tâches partielles) : (1) docstring cost model explicite (commission+slippage ~0.2% round-trip), (2) tests walk-forward anti-look-ahead (données synthétiques flat+drift, 19 tests OK/1 skip), (3) ExposureManager standalone 10% cap (non branché runner), (4) SUPERSEARCH_SCORING.md détaillant ranking lexicographique vs weighted, (5) Phase 1 robustness exposant tous ratios (pf/sharpe/dd/expectancy/vol/winrate) au lieu de score agrégé seul.
- Metrics: 19 nouveaux tests passent (8 exposure, 5 lookahead, 10 phase1 ratios), 1 skip (test_full_series WIP).
- Notes: ExposureManager testé mais non intégré BacktestRunner (flags CLI --max-total-exposure à brancher). Ratios Phase 1 exposés mais robustness_score legacy maintenu (risque confusion). Test lookahead WIP skip (données synthétiques calibrées insuffisantes).

[Log]
- Date: 2025-11-15
- Files: src/hyprl/labels/amplitude.py, src/hyprl/backtest/runner.py, src/hyprl/pipeline.py, scripts/run_backtest.py, scripts/run_supersearch.py, scripts/run_analysis.py, src/hyprl/search/optimizer.py, tests/labels/test_amplitude_labels.py, tests/integration/test_amplitude_backtest.py, data/supersearch_binary_aapl_1y.csv, data/supersearch_amplitude_aapl_1y.csv
- Change: Implémentation du mode de labels amplitude (BIG_UP/BIG_DOWN/NEUTRAL) avec LabelConfig partagé (mode/horizon/threshold_pct/neutral_strategy/min_samples). Pipeline + runner + supersearch reçoivent les labels multi-horizon via attach_amplitude_labels/validate_label_support, nouveaux flags CLI (--label-mode/--label-horizon/--label-threshold-pct/--label-neutral-strategy/--min-samples-per-class). Ajout des tests unitaires + intégration (pytest labels + runtime stub) et génération de deux supersearch AAPL 1y (binary vs amplitude) pour comparer PF/expectancy/trades.
- Metrics: pytest tests/labels/test_amplitude_labels.py -v (pass), pytest tests/integration/test_amplitude_backtest.py -v (pass), pytest tests/ -k "not amplitude" -v (PYTHONPATH=. run) → 5 échecs RT+autorank déjà présents; nouveaux jobs supersearch produisent 8 configs binary (PF≈0.96 max, expectancy ≈ -2.43, 94 trades) vs 12 configs amplitude (PF≈0.73 max, expectancy ≈ -19.2, 79 trades).
- Notes: CLI run_supersearch.py retourne encore un CSV vide malgré le run_search OK → workaround provisoire via script Python direct (run_search + save_results_csv). RT test suite continue d'échouer (clamps, kill switch, queue, autorank constraints) sans lien avec labels amplitude.

[Log]
- Date: 2025-11-11
- Files: src/hyprl/meta/registry.py, src/hyprl/meta/autorank.py, scripts/calibrate_meta_predictions.py, scripts/autorank_supersearch.py, scripts/hyprl_meta_dashboard.py
- Change: Added Meta-ML registry CLI flow, calibration pipeline (isotonic/quantile), autorank constraints with Phase 1 shortlist + GUI integration.
- Notes: New pytest suites guard registry bumps, calibrator quality, autorank filters; run `pytest -q` after changes.

[Log]
- Date: 2025-11-17
- Files: native/hyprl_supercalc/src/lib.rs, native/hyprl_supercalc/src/ffi.rs, native/hyprl_supercalc/src/backtest/mod.rs
- Change: Cleaned lib.rs to pure module wiring and rewired the PyO3 bridge (Bound API, custom BacktestConfig extractor, IntoPy dict emitters, stub DataFrame→Candle path) so the new crate skeleton compiles. Dropped stale metrics import.
- Notes: `run_batch_backtest_py` currently short-circuits when candle extraction is unimplemented; fill `df_to_candles` next and flesh out run_backtest before wiring into Python.

[Log]
- Date: 2025-11-17
- Files: native/hyprl_supercalc/src/ffi.rs, native/hyprl_supercalc/src/backtest/mod.rs, native/hyprl_supercalc/src/metrics.rs, native/hyprl_supercalc/Cargo.toml, tests/test_supercalc_bridge.py
- Change: Implemented Polars→Candle conversion, real signal-driven backtester with cost model + metrics pipeline, expanded metrics.rs helpers, aligned Polars deps (0.42) with pyo3-polars, and added a pytest smoke test for the PyO3 bridge.
- Notes: `run_batch_backtest_py` now validates signal length; pytest smoke test requires the extension to be built via `maturin develop` beforehand.
## External LLM Code Audit – 2025-11-15

Contexte  
Audit réalisé via LLM (Claude) sur la base de la documentation v1.2 (PROJECT_BRAIN, V1_1_1_SUMMARY, hyprl_export_overview) et des interfaces CLI. Les fichiers de code n’ont pas été ouverts dans cette passe : tous les constats marqués comme “non vérifiés” restent des hypothèses à confirmer dans le code.

Points forts confirmés
- Architecture claire : data → features → ProbabilityModel → Risk layer (ATR, EV filter, RoR bootstrap) → walk-forward backtest → supersearch → portfolio → Phase 1 (panel + paper + analyse).  
- Hard constraints appliqués avant le scoring supersearch (min_trades, PF, Sharpe, maxDD, RoR, expectancy).  
- Portfolio layer multi-ticker avec métriques portfolio_* et contraintes de corrélation.  
- Provenance forte (CSV + .meta.json avec hash, provenance.json consolidé).  
- Phase 1 : pipeline complet panel → sessions live → phase1_results.csv avec ratios backtest/live et robustness_score ∈ [0,1].

Risques / gaps identifiés (à vérifier dans le code)
**P0 – Validité scientifique**
1. Modèle coûts/slippage : docs mentionnent “coûts, slippage” mais sans détails. À vérifier dans `src/hyprl/backtest/runner.py` :  
   - comment sont appliquées commissions et slippage,  
   - quelles valeurs par défaut,  
   - symétrie long/short.  
2. Walk-forward / look-ahead : la notion de “fenêtre locale” est documentée mais pas la split exacte. À vérifier dans `src/hyprl/model/probability.py` et le runner :  
   - train = [t-N, t-1], test = t (sans fuite),  
   - features calculées sans utiliser de stats futures,  
   - exécution au bar suivant (close → next open).  
3. Cap de risque agrégé : portfolio core calcule des métriques ex-post, mais aucun cap “max total exposure” n’est mentionné. À vérifier dans `src/hyprl/portfolio/core.py` / execution :  
   - somme des expositions en % avant ouverture d’un nouveau trade,  
   - éventuel rejet si > seuil (ex : 10 %).

**P1 – Robustesse et lisibilité**
4. Scoring supersearch : docs disent “pénalise RoR et maxDD_p95, bonifie l’espérance”, mais la formule exacte n’est pas documentée. À extraire de `src/hyprl/search/optimizer.py` et potentiellement exposer les poids via CLI + doc `SUPERSEARCH_SCORING.md`.  
5. robustness_score Phase 1 : on sait qu’il est borné [0,1] mais pas comment il est calculé. À clarifier dans `scripts/analyze_phase1_results.py`. Piste : privilégier un rapport multi-métriques (PF_ratio, Sharpe_ratio, DD_ratio, expectancy_ratio, winrate_ratio) plutôt qu’un seul score opaque.

Décisions / TODO v1.3 (proposées)
- [P0] Relire le code de `backtest/runner.py` et `model/probability.py` et ajouter au besoin :
  - paramètres explicites de coûts/slippage + doc détaillée,  
  - tests unitaires anti-look-ahead sur données synthétiques.
- [P0] Concevoir un `PortfolioRiskAggregator` simple avec `--max-total-exposure` et l’intégrer au moteur d’exécution.  
- [P1] Documenter clairement la formule de scoring supersearch et la rendre optionnellement paramétrable.  
- [P1] Revoir l’output Phase 1 : exposer systématiquement les ratios backtest/live par métrique, garder `robustness_score` uniquement s’il est justifié/documenté.  
- [P1] Ajouter dans README/PROJECT_BRAIN un encadré “Caveats” sur le survivorship bias de yfinance.

### 2025-11-17 — Native supercalc V1 locked in (Rust engine + Py wrapper)

- [Action] → Rebuilt the hyprl_supercalc abi3 wheel via `scripts/build_supercalc.sh`, reran the full supercalc + backtest suites using the native engine gate in `hyprl.supercalc.__init__`.
- [Result] → `pytest tests/test_supercalc_*.py tests/backtest/test_supercalc_native.py -q` ✅ (7 tests, 733 known warnings) and `pytest tests/backtest -q` ✅ (30 passed, 1 skipped, 807 known warnings). No new regressions introduced by the Rust engine.
- [Status] → Native Rust engine + PyO3 bridge + wrapper selection are now considered stable for research use (`engine="native"` / `"auto"`), with parity validated against the Python backtester.
- [Next] → Start running small supersearch grids with `--engine native/auto`, then add a lightweight benchmark script (Python vs native) to quantify speedups on real candidate sets.
