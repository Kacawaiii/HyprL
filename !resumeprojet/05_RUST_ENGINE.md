# Calculateur Rust HyprL (hyprl_supercalc)

## Vue d'ensemble

HyprL utilise un moteur de calcul haute performance en Rust (via PyO3) pour:
- Calculer les indicateurs techniques 10-50x plus vite que Python
- Exécuter des backtests complets avec gestion ATR des positions
- Optimiser les paramètres en parallèle (grid search)
- Calculer des métriques de risque (Monte Carlo bootstrap)

## Architecture

```
native/hyprl_supercalc/
├── Cargo.toml              # Configuration et dépendances
└── src/
    ├── lib.rs              # Point d'entrée, exports PyO3
    ├── core.rs             # Types: Candle, BacktestConfig, BacktestReport
    ├── indicators/
    │   └── mod.rs          # Indicateurs techniques (SMA, EMA, RSI, MACD, ATR, BB)
    ├── backtest/
    │   └── mod.rs          # Moteur de backtest single-asset
    ├── batch.rs            # Évaluation parallèle (grid search)
    ├── metrics.rs          # Calcul des métriques de performance
    └── ffi.rs              # Interface Python (PyO3)
```

## Dépendances Rust

```toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }
pyo3-polars = "0.16"        # Intégration Polars DataFrames
polars = { version = "0.42", features = ["lazy", "parquet", "ndarray"] }
rayon = "1.10"              # Parallélisation
ndarray = "0.15"            # Arrays N-dimensionnels
statrs = "0.16"             # Statistiques
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = { version = "0.8", features = ["std"] }
rand_pcg = "0.3"            # RNG déterministe pour Monte Carlo
```

## Indicateurs Techniques

### Fonctions disponibles

| Fonction | Description | Warmup |
|----------|-------------|--------|
| `sma(series, window)` | Simple Moving Average | window |
| `ema(series, window)` | Exponential Moving Average (pandas compatible) | 0 |
| `rsi(series, window)` | Relative Strength Index | window |
| `atr(high, low, close, window)` | Average True Range | window |
| `bollinger_bands(series, window, num_std)` | Bollinger Bands (mid, upper, lower) | window |
| `macd_full(series)` | MACD (12, 26, 9) avec signal et histogram | 26 |
| `trend_ratio(series, short, long)` | Ratio SMA court/long | long |
| `rolling_volatility(series, window)` | Volatilité (log returns) | window |

### IndicatorSet complet

```rust
pub struct IndicatorSet {
    pub sma_20: Vec<f64>,
    pub ema_20: Vec<f64>,
    pub rsi_14: Vec<f64>,
    pub macd: Vec<f64>,
    pub macd_signal: Vec<f64>,
    pub macd_hist: Vec<f64>,
    pub bb_upper_20: Vec<f64>,
    pub bb_mid_20: Vec<f64>,
    pub bb_lower_20: Vec<f64>,
    pub atr_14: Vec<f64>,
    pub trend_ratio_50_200: Vec<f64>,
    pub rolling_vol_20: Vec<f64>,
}
```

## Moteur de Backtest

### Configuration

```rust
pub struct BacktestConfig {
    pub risk_pct: f64,              // Risk par trade (ex: 0.02 = 2%)
    pub commission_pct: f64,        // Commission (ex: 0.001 = 0.1%)
    pub slippage_pct: f64,          // Slippage (ex: 0.0005)
    pub max_leverage: f64,          // Levier max (ex: 1.8)
    pub allow_short: bool,          // Autoriser les shorts

    // ATR-based position sizing
    pub use_atr_position_sizing: bool,
    pub atr_window: usize,          // Période ATR (ex: 14)
    pub atr_mult_stop: f64,         // Multiplicateur stop (ex: 2.5 ATR)
    pub atr_mult_tp: f64,           // Multiplicateur TP (ex: 5.0 ATR)

    // Trailing stop
    pub trailing_activation_r: f64, // R-multiple pour activer (ex: 1.0)
    pub trailing_distance_r: f64,   // Distance trailing (ex: 0.5 R)
}
```

### Fonctionnalités

- **Position sizing ATR-based**: Calcul automatique de la taille de position basée sur ATR
- **Stop Loss dynamique**: Calculé à partir de l'ATR
- **Take Profit**: Target basé sur le ratio risque/récompense
- **Trailing Stop**: S'active après N x R de profit, suit le prix à distance configurable
- **Coûts de transaction**: Commission + slippage inclus
- **Exit reasons**: `stop_loss`, `take_profit`, `trailing_stop`, `entry`

### Rapport de backtest

```rust
pub struct BacktestReport {
    pub config: BacktestConfig,
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<EquityPoint>,
    pub n_trades: usize,
    pub trades: Vec<TradeOutcome>,
    pub debug_info: Option<String>,
}
```

## Métriques de Performance

### Métriques calculées

| Métrique | Description |
|----------|-------------|
| `total_return` | Return total (last/first - 1) |
| `cagr` | Compound Annual Growth Rate |
| `sharpe` | Sharpe Ratio (simple, non-annualisé) |
| `sortino` | Sortino Ratio (annualisé) |
| `calmar` | CAGR / Max Drawdown |
| `max_drawdown` | Drawdown maximum (négatif) |
| `max_drawdown_duration` | Durée en bars |
| `profit_factor` | Gross profit / |Gross loss| |
| `win_rate` | % de trades gagnants |
| `expectancy` | Espérance par trade |
| `risk_of_ruin` | Probabilité de ruine |

### Bootstrap Monte Carlo

Simulation de 512 paths pour estimer la distribution:

```rust
pub struct BootstrapStats {
    pub maxdd_p05: f64,   // 5e percentile drawdown
    pub maxdd_p95: f64,   // 95e percentile drawdown
    pub maxdd_p99: f64,   // 99e percentile drawdown
    pub pnl_p05: f64,     // 5e percentile P/L
    pub pnl_p50: f64,     // Médiane P/L
    pub pnl_p95: f64,     // 95e percentile P/L
}
```

### Score de Robustesse

Score composite (0-1) basé sur:
- Ratio P/L bootstrap vs backtest (30%)
- Stabilité du Sharpe (30%)
- Ratio drawdown bootstrap/réel (20%)
- Volatilité du P/L (10%)
- Delta win rate vs 50% (10%)

## Optimisation Parallèle (Grid Search)

### Batch processing avec Rayon

```rust
pub fn evaluate_batch(
    candles: &[Candle],
    signal: &[f64],
    configs: &[BacktestConfig],
) -> Vec<BacktestReport> {
    configs
        .par_iter()  // Parallélisation automatique
        .map(|cfg| run_backtest(candles, signal, cfg))
        .collect()
}
```

### Contraintes de recherche

```rust
pub struct SearchConstraint {
    pub min_trades: usize,          // 50
    pub min_profit_factor: f64,     // 1.2
    pub min_sharpe: f64,            // 0.8
    pub max_drawdown: f64,          // 0.35 (35%)
    pub max_risk_of_ruin: f64,      // 0.1 (10%)
    pub min_expectancy: f64,        // 0.0
    pub min_robustness: f64,        // 0.0
    pub max_maxdd_p95: f64,         // 0.35
}
```

## Benchmarks de Performance

### Indicateurs (10,000 barres)

| Opération | Python (ms) | Rust (ms) | Speedup |
|-----------|-------------|-----------|---------|
| ATR 14 | 45 | 1.2 | **37x** |
| RSI 14 | 38 | 0.9 | **42x** |
| MACD (12,26,9) | 62 | 1.8 | **34x** |
| Bollinger Bands 20 | 55 | 1.5 | **36x** |
| SMA 20 | 12 | 0.3 | **40x** |
| EMA 20 | 15 | 0.4 | **37x** |
| All indicators | 180 | 5.2 | **35x** |

### Backtest (2 ans de données 1h, ~4,400 barres)

| Opération | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Single backtest | 450ms | 12ms | **37x** |
| Grid search (1000 configs) | 7.5min | 8sec | **56x** |
| Bootstrap 512 runs | 2.3sec | 45ms | **51x** |

### Throughput

| Opération | Performance |
|-----------|-------------|
| Backtests séquentiels | ~80/sec |
| Backtests parallèles (8 cores) | ~450/sec |
| Indicateurs (10k bars) | ~200 sets/sec |

## Binding Python

### Chemin: `src/hyprl/native/supercalc.py`

```python
"""Python bindings pour le calculateur Rust."""

import numpy as np

try:
    from hyprl_supercalc import (
        compute_atr_fast,
        compute_rsi_fast,
        compute_bollinger_fast,
        compute_macd_fast,
        compute_indicators,
        run_backtest,
        run_native_search,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcule l'ATR avec Rust si disponible."""
    if RUST_AVAILABLE:
        return np.array(compute_atr_fast(
            high.tolist(),
            low.tolist(),
            close.tolist(),
            period
        ))
    else:
        return _compute_atr_python(high, low, close, period)
```

## Build et Installation

### Compilation

```bash
cd native/hyprl_supercalc
cargo build --release
```

### Installation comme module Python

```bash
pip install maturin
cd native/hyprl_supercalc
maturin develop --release
```

### Vérification

```python
>>> import hyprl_supercalc
>>> hyprl_supercalc.__version__
'0.1.0'
>>> from hyprl_supercalc import compute_indicators
>>> indicators = compute_indicators(candles)
```

## Utilisation dans HyprL

### Indicateurs rapides

```python
from src.hyprl.native.supercalc import compute_atr, compute_rsi, compute_macd

# Calcul rapide des indicateurs
atr = compute_atr(df['high'].values, df['low'].values, df['close'].values, 14)
rsi = compute_rsi(df['close'].values, 14)
macd, signal, hist = compute_macd(df['close'].values, 12, 26, 9)
```

### Grid Search natif

```python
from hyprl_supercalc import run_native_search

# Optimisation parallèle
best_configs = run_native_search(
    candles=candles,
    signal=signal,
    configs=param_grid,
    constraints=SearchConstraint(
        min_trades=50,
        min_sharpe=0.8,
        max_drawdown=0.25
    ),
    top_k=10
)
```

## Notes de développement

- Le code Rust est compatible Python 3.9+ (ABI stable)
- Les NaN sont propagés correctement pour le warmup des indicateurs
- Le RNG Monte Carlo est déterministe (seed 42) pour reproductibilité
- Rayon utilise automatiquement tous les cores disponibles
