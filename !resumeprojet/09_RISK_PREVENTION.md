# Prévention des Risques et Catastrophes - HyprL

## Pourquoi les Bots de Trading Échouent

### Statistiques Alarmantes

- **44% des stratégies** publiées échouent à répliquer leurs performances sur de nouvelles données
- **R² < 0.025** entre Sharpe backtest et Sharpe live (quasi aucune corrélation)
- **Knight Capital**: $440M perdus en 45 minutes (code obsolète activé par erreur)
- **Mai 2025**: Flash crash crypto, $2B vendus par des bots AI en cascade

### Causes Principales d'Échec

| Cause | Fréquence | Impact |
|-------|-----------|--------|
| Overfitting | 40% | Élevé |
| Regime change | 25% | Critique |
| Bugs/erreurs techniques | 15% | Variable |
| Problèmes d'exécution | 10% | Moyen |
| Événements black swan | 10% | Catastrophique |

---

## 1. DRIFT DETECTION - Dégradation du Modèle

### Le Problème

Les modèles ML perdent leur efficacité quand:
- La distribution des features change (Feature Drift)
- La relation input/output change (Concept Drift)
- Le régime de marché change (Regime Change)

### Solution: Module de Détection de Drift

```python
# src/hyprl/monitoring/drift_detector.py

from dataclasses import dataclass
from scipy import stats
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class DriftAlert:
    """Alerte de drift détectée."""
    drift_type: str  # "feature", "prediction", "performance"
    feature_name: Optional[str]
    severity: str  # "warning", "critical"
    psi_score: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov
    p_value: float
    message: str
    timestamp: datetime


class DriftDetector:
    """Détecte le drift des features et des prédictions."""

    # Seuils PSI (Population Stability Index)
    PSI_WARNING = 0.1   # Drift modéré
    PSI_CRITICAL = 0.25  # Drift sévère

    # Seuils KS
    KS_WARNING = 0.1
    KS_CRITICAL = 0.2

    def __init__(self, baseline_data: dict[str, np.ndarray]):
        """
        Args:
            baseline_data: Dict de {feature_name: array} du training set
        """
        self.baseline = baseline_data
        self.baseline_stats = self._compute_stats(baseline_data)
        self.alerts_history = []

    def _compute_stats(self, data: dict) -> dict:
        """Calcule les stats de référence."""
        stats = {}
        for name, values in data.items():
            stats[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "quantiles": np.percentile(values, [10, 25, 50, 75, 90])
            }
        return stats

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        PSI < 0.1: Pas de changement significatif
        PSI 0.1-0.25: Changement modéré
        PSI > 0.25: Changement significatif
        """
        # Créer les bins à partir de la distribution de référence
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]

        # Éviter division par zéro
        expected_percents = (expected_counts + 1) / (len(expected) + bins)
        actual_percents = (actual_counts + 1) / (len(actual) + bins)

        psi = np.sum((actual_percents - expected_percents) *
                     np.log(actual_percents / expected_percents))
        return psi

    def check_feature_drift(self, current_data: dict[str, np.ndarray]) -> list[DriftAlert]:
        """Vérifie le drift de chaque feature."""
        alerts = []

        for feature_name, current_values in current_data.items():
            if feature_name not in self.baseline:
                continue

            baseline_values = self.baseline[feature_name]

            # PSI
            psi = self.calculate_psi(baseline_values, current_values)

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)

            # Déterminer la sévérité
            if psi > self.PSI_CRITICAL or ks_stat > self.KS_CRITICAL:
                severity = "critical"
            elif psi > self.PSI_WARNING or ks_stat > self.KS_WARNING:
                severity = "warning"
            else:
                continue  # Pas de drift significatif

            alert = DriftAlert(
                drift_type="feature",
                feature_name=feature_name,
                severity=severity,
                psi_score=psi,
                ks_statistic=ks_stat,
                p_value=p_value,
                message=f"Feature '{feature_name}' drift detected: PSI={psi:.3f}, KS={ks_stat:.3f}",
                timestamp=datetime.now()
            )
            alerts.append(alert)

        self.alerts_history.extend(alerts)
        return alerts

    def check_prediction_drift(
        self,
        baseline_preds: np.ndarray,
        current_preds: np.ndarray
    ) -> Optional[DriftAlert]:
        """Vérifie si la distribution des prédictions a changé."""
        psi = self.calculate_psi(baseline_preds, current_preds)
        ks_stat, p_value = stats.ks_2samp(baseline_preds, current_preds)

        if psi > self.PSI_CRITICAL:
            severity = "critical"
        elif psi > self.PSI_WARNING:
            severity = "warning"
        else:
            return None

        return DriftAlert(
            drift_type="prediction",
            feature_name=None,
            severity=severity,
            psi_score=psi,
            ks_statistic=ks_stat,
            p_value=p_value,
            message=f"Prediction drift detected: PSI={psi:.3f}",
            timestamp=datetime.now()
        )

    def check_performance_drift(
        self,
        recent_win_rate: float,
        baseline_win_rate: float = 0.54,
        recent_sharpe: float = None,
        baseline_sharpe: float = 1.42,
        n_trades: int = 20
    ) -> Optional[DriftAlert]:
        """Vérifie si les performances se dégradent."""

        # Test binomial pour le win rate
        from scipy.stats import binom_test
        wins = int(recent_win_rate * n_trades)
        p_value = binom_test(wins, n_trades, baseline_win_rate, alternative='less')

        if p_value < 0.01:  # 99% confidence que c'est pire
            severity = "critical"
        elif p_value < 0.05:  # 95% confidence
            severity = "warning"
        else:
            return None

        return DriftAlert(
            drift_type="performance",
            feature_name=None,
            severity=severity,
            psi_score=0,
            ks_statistic=0,
            p_value=p_value,
            message=f"Performance degradation: Win rate {recent_win_rate:.1%} vs expected {baseline_win_rate:.1%}",
            timestamp=datetime.now()
        )
```

### Utilisation

```python
# Dans le bridge de trading
from src.hyprl.monitoring.drift_detector import DriftDetector

# Initialiser avec les données de training
detector = DriftDetector(baseline_data=training_features)

# Vérifier périodiquement (toutes les 24h)
alerts = detector.check_feature_drift(last_24h_features)
for alert in alerts:
    if alert.severity == "critical":
        logger.critical(f"DRIFT CRITICAL: {alert.message}")
        # Envoyer notification Discord/Telegram
        # Réduire la taille des positions de 50%
    elif alert.severity == "warning":
        logger.warning(f"DRIFT WARNING: {alert.message}")
```

---

## 2. CIRCUIT BREAKERS - Protection Multi-Niveaux

### Le Problème

Sans circuit breakers, une erreur peut causer des pertes en cascade.

### Solution: Circuit Breakers Multi-Niveaux

```python
# src/hyprl/risk/circuit_breakers.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import threading


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration d'un circuit breaker."""
    name: str
    threshold: float
    window_minutes: int
    cooldown_minutes: int
    action: str  # "reduce_size", "close_positions", "halt_trading"


class CircuitBreakerManager:
    """
    Gère les circuit breakers à plusieurs niveaux.
    Inspiré du modèle NYSE (7%, 13%, 20%).
    """

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.session_start_equity = initial_equity
        self.current_equity = initial_equity
        self.state = CircuitState.CLOSED
        self.last_trigger_time: Optional[datetime] = None
        self.triggers_today = 0
        self._lock = threading.Lock()

        # Configuration des niveaux
        self.levels = [
            CircuitBreakerConfig(
                name="LEVEL_1_WARNING",
                threshold=-0.02,  # -2% daily
                window_minutes=1440,
                cooldown_minutes=15,
                action="reduce_size"
            ),
            CircuitBreakerConfig(
                name="LEVEL_2_CAUTION",
                threshold=-0.05,  # -5% daily
                window_minutes=1440,
                cooldown_minutes=60,
                action="close_new_positions"
            ),
            CircuitBreakerConfig(
                name="LEVEL_3_CRITICAL",
                threshold=-0.10,  # -10% daily
                window_minutes=1440,
                cooldown_minutes=1440,  # Rest of day
                action="halt_trading"
            ),
            CircuitBreakerConfig(
                name="LEVEL_4_CATASTROPHIC",
                threshold=-0.15,  # -15% total
                window_minutes=0,  # From start
                cooldown_minutes=10080,  # 1 week
                action="close_all_and_halt"
            ),
        ]

        # Métriques de trading
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.trades_today = 0
        self.max_trades_per_day = 20

    def update_equity(self, new_equity: float) -> list[str]:
        """Met à jour l'equity et vérifie les circuit breakers."""
        with self._lock:
            self.current_equity = new_equity
            actions = []

            daily_return = (new_equity - self.session_start_equity) / self.session_start_equity
            total_return = (new_equity - self.initial_equity) / self.initial_equity

            for level in self.levels:
                threshold_return = daily_return if level.window_minutes > 0 else total_return

                if threshold_return <= level.threshold:
                    if self._can_trigger(level):
                        actions.append(self._trigger_breaker(level))

            return actions

    def _can_trigger(self, level: CircuitBreakerConfig) -> bool:
        """Vérifie si le breaker peut être déclenché."""
        if self.last_trigger_time is None:
            return True

        elapsed = datetime.now() - self.last_trigger_time
        return elapsed > timedelta(minutes=level.cooldown_minutes)

    def _trigger_breaker(self, level: CircuitBreakerConfig) -> str:
        """Déclenche un circuit breaker."""
        self.state = CircuitState.OPEN
        self.last_trigger_time = datetime.now()
        self.triggers_today += 1

        action_msg = f"CIRCUIT BREAKER {level.name} TRIGGERED: {level.action}"
        return action_msg

    def record_trade_result(self, is_win: bool) -> Optional[str]:
        """Enregistre le résultat d'un trade."""
        self.trades_today += 1

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        # Breaker sur pertes consécutives
        if self.consecutive_losses >= self.max_consecutive_losses:
            return f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses - halting"

        # Breaker sur nombre de trades
        if self.trades_today >= self.max_trades_per_day:
            return f"CIRCUIT BREAKER: Max trades ({self.max_trades_per_day}) reached - halting"

        return None

    def can_trade(self) -> tuple[bool, str]:
        """Vérifie si le trading est autorisé."""
        if self.state == CircuitState.OPEN:
            return False, "Circuit breaker is OPEN"
        if self.trades_today >= self.max_trades_per_day:
            return False, "Max daily trades reached"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, "Max consecutive losses reached"
        return True, "OK"

    def reset_daily(self):
        """Reset quotidien (appelé à l'ouverture du marché)."""
        with self._lock:
            self.session_start_equity = self.current_equity
            self.triggers_today = 0
            self.trades_today = 0
            self.consecutive_losses = 0
            if self.state != CircuitState.OPEN:  # Ne pas reset si Level 4
                self.state = CircuitState.CLOSED


class KillSwitch:
    """
    Kill switch d'urgence - arrêt total immédiat.
    Stocké dans un fichier externe pour éviter modification par le bot.
    """

    KILL_FILE = "/home/kyo/HyprL/.kill_switch"

    @classmethod
    def is_killed(cls) -> bool:
        """Vérifie si le kill switch est activé."""
        try:
            with open(cls.KILL_FILE, 'r') as f:
                content = f.read().strip().lower()
                return content in ('1', 'true', 'kill', 'stop')
        except FileNotFoundError:
            return False

    @classmethod
    def activate(cls, reason: str = "Manual activation"):
        """Active le kill switch."""
        with open(cls.KILL_FILE, 'w') as f:
            f.write(f"1\n{datetime.now().isoformat()}\n{reason}")

    @classmethod
    def deactivate(cls):
        """Désactive le kill switch."""
        with open(cls.KILL_FILE, 'w') as f:
            f.write("0")
```

### Intégration dans le Bridge

```python
# Dans run_strategy_bridge.py

from src.hyprl.risk.circuit_breakers import CircuitBreakerManager, KillSwitch

# Initialisation
circuit_breaker = CircuitBreakerManager(initial_equity=100000)

# Avant chaque trade
if KillSwitch.is_killed():
    logger.critical("KILL SWITCH ACTIVE - No trading allowed")
    sys.exit(1)

can_trade, reason = circuit_breaker.can_trade()
if not can_trade:
    logger.warning(f"Trading blocked: {reason}")
    continue

# Après chaque mise à jour d'equity
actions = circuit_breaker.update_equity(current_equity)
for action in actions:
    logger.critical(action)
    send_discord_alert(action)
```

---

## 3. REGIME DETECTION - Adaptation au Marché

### Le Problème

Un modèle entraîné en bull market échoue en bear market.

### Solution: Détection de Régime

```python
# src/hyprl/regime/detector.py

from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional


class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"      # Forte tendance haussière
    BULL_VOLATILE = "bull_volatile"      # Hausse volatile
    BEAR_TRENDING = "bear_trending"      # Forte tendance baissière
    BEAR_VOLATILE = "bear_volatile"      # Baisse volatile
    SIDEWAYS_LOW_VOL = "sideways_low"    # Range, faible volatilité
    SIDEWAYS_HIGH_VOL = "sideways_high"  # Range, haute volatilité
    CRASH = "crash"                       # Effondrement rapide
    MELT_UP = "melt_up"                  # Hausse parabolique


@dataclass
class RegimeConfig:
    """Configuration par régime."""
    regime: MarketRegime
    position_size_mult: float  # Multiplicateur de taille
    confidence_threshold: float  # Seuil de confiance requis
    allow_longs: bool
    allow_shorts: bool
    max_holding_hours: int


REGIME_CONFIGS = {
    MarketRegime.BULL_TRENDING: RegimeConfig(
        regime=MarketRegime.BULL_TRENDING,
        position_size_mult=1.2,
        confidence_threshold=0.52,
        allow_longs=True,
        allow_shorts=False,  # Pas de shorts en bull
        max_holding_hours=48
    ),
    MarketRegime.BEAR_TRENDING: RegimeConfig(
        regime=MarketRegime.BEAR_TRENDING,
        position_size_mult=0.8,
        confidence_threshold=0.58,  # Plus strict
        allow_longs=False,
        allow_shorts=True,
        max_holding_hours=24
    ),
    MarketRegime.SIDEWAYS_HIGH_VOL: RegimeConfig(
        regime=MarketRegime.SIDEWAYS_HIGH_VOL,
        position_size_mult=0.5,  # Réduire exposition
        confidence_threshold=0.60,
        allow_longs=True,
        allow_shorts=True,
        max_holding_hours=12
    ),
    MarketRegime.CRASH: RegimeConfig(
        regime=MarketRegime.CRASH,
        position_size_mult=0.0,  # NO TRADING
        confidence_threshold=1.0,
        allow_longs=False,
        allow_shorts=False,
        max_holding_hours=0
    ),
    MarketRegime.MELT_UP: RegimeConfig(
        regime=MarketRegime.MELT_UP,
        position_size_mult=0.3,  # Très prudent
        confidence_threshold=0.65,
        allow_longs=True,
        allow_shorts=False,
        max_holding_hours=8
    ),
}


class RegimeDetector:
    """Détecte le régime de marché actuel."""

    def __init__(self):
        self.current_regime = MarketRegime.SIDEWAYS_LOW_VOL
        self.regime_history = []

    def detect_regime(
        self,
        returns_20d: np.ndarray,
        volatility_20d: float,
        volatility_60d: float,
        sma_50: float,
        sma_200: float,
        current_price: float,
        vix: Optional[float] = None
    ) -> MarketRegime:
        """
        Détecte le régime de marché basé sur plusieurs indicateurs.
        """
        # Calculs
        cumulative_return = np.sum(returns_20d)
        daily_vol = np.std(returns_20d)
        trend_strength = (sma_50 - sma_200) / sma_200
        vol_ratio = volatility_20d / volatility_60d if volatility_60d > 0 else 1.0

        # Détection CRASH
        if cumulative_return < -0.15 or (vix and vix > 40):
            regime = MarketRegime.CRASH

        # Détection MELT_UP
        elif cumulative_return > 0.15 and vol_ratio > 1.5:
            regime = MarketRegime.MELT_UP

        # BULL TRENDING
        elif trend_strength > 0.05 and cumulative_return > 0.03:
            if vol_ratio > 1.3:
                regime = MarketRegime.BULL_VOLATILE
            else:
                regime = MarketRegime.BULL_TRENDING

        # BEAR TRENDING
        elif trend_strength < -0.05 and cumulative_return < -0.03:
            if vol_ratio > 1.3:
                regime = MarketRegime.BEAR_VOLATILE
            else:
                regime = MarketRegime.BEAR_TRENDING

        # SIDEWAYS
        else:
            if daily_vol > 0.02:  # >2% daily vol
                regime = MarketRegime.SIDEWAYS_HIGH_VOL
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL

        self.current_regime = regime
        self.regime_history.append((datetime.now(), regime))

        return regime

    def get_config(self) -> RegimeConfig:
        """Retourne la configuration pour le régime actuel."""
        return REGIME_CONFIGS.get(
            self.current_regime,
            REGIME_CONFIGS[MarketRegime.SIDEWAYS_HIGH_VOL]
        )

    def should_trade(self, signal_direction: str) -> tuple[bool, str]:
        """Vérifie si le trade est autorisé dans le régime actuel."""
        config = self.get_config()

        if config.position_size_mult == 0:
            return False, f"No trading in {self.current_regime.value} regime"

        if signal_direction == "long" and not config.allow_longs:
            return False, f"Longs not allowed in {self.current_regime.value} regime"

        if signal_direction == "short" and not config.allow_shorts:
            return False, f"Shorts not allowed in {self.current_regime.value} regime"

        return True, "OK"
```

---

## 4. CORRELATION MONITOR - Risque de Portfolio

### Le Problème

NVDA, MSFT, QQQ sont corrélés. Une chute du Nasdaq impacte tous les trades.

### Solution: Monitoring de Corrélation

```python
# src/hyprl/risk/correlation_monitor.py

import numpy as np
from typing import Dict, List


class CorrelationMonitor:
    """Surveille les corrélations et ajuste l'exposition."""

    # Seuils
    HIGH_CORRELATION = 0.7
    CRITICAL_CORRELATION = 0.85

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.correlation_matrix = np.eye(len(symbols))

    def update_correlations(self, returns_dict: Dict[str, np.ndarray]):
        """Met à jour la matrice de corrélation."""
        n = len(self.symbols)
        for i, sym1 in enumerate(self.symbols):
            for j, sym2 in enumerate(self.symbols):
                if i < j and sym1 in returns_dict and sym2 in returns_dict:
                    corr = np.corrcoef(returns_dict[sym1], returns_dict[sym2])[0, 1]
                    self.correlation_matrix[i, j] = corr
                    self.correlation_matrix[j, i] = corr

    def get_max_positions(self, current_positions: Dict[str, str]) -> Dict[str, int]:
        """
        Calcule le nombre max de positions par direction.
        Si corrélation élevée, limite l'exposition.
        """
        # Compter les positions longues et courtes
        longs = [s for s, d in current_positions.items() if d == "long"]
        shorts = [s for s, d in current_positions.items() if d == "short"]

        # Vérifier corrélation entre positions longues
        if len(longs) >= 2:
            for i, sym1 in enumerate(longs):
                for sym2 in longs[i+1:]:
                    idx1 = self.symbols.index(sym1)
                    idx2 = self.symbols.index(sym2)
                    corr = self.correlation_matrix[idx1, idx2]

                    if corr > self.CRITICAL_CORRELATION:
                        return {"max_longs": 1, "max_shorts": 1,
                                "reason": f"Critical correlation {sym1}/{sym2}: {corr:.2f}"}
                    elif corr > self.HIGH_CORRELATION:
                        return {"max_longs": 2, "max_shorts": 2,
                                "reason": f"High correlation {sym1}/{sym2}: {corr:.2f}"}

        return {"max_longs": 3, "max_shorts": 3, "reason": "Normal"}

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],  # symbol -> dollar exposure
        returns_dict: Dict[str, np.ndarray],
        confidence: float = 0.95
    ) -> float:
        """Calcule la Value at Risk du portfolio."""
        weights = []
        returns_matrix = []

        total_exposure = sum(abs(v) for v in positions.values())
        if total_exposure == 0:
            return 0

        for symbol in self.symbols:
            if symbol in positions and symbol in returns_dict:
                weights.append(positions[symbol] / total_exposure)
                returns_matrix.append(returns_dict[symbol])

        if not weights:
            return 0

        weights = np.array(weights)
        returns_matrix = np.array(returns_matrix)

        # Portfolio returns
        portfolio_returns = np.dot(weights, returns_matrix)

        # VaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)

        return var * total_exposure
```

---

## 5. DATA QUALITY MONITOR - Données Corrompues

### Le Problème

Mauvaises données = mauvaises décisions.

### Solution: Validation des Données

```python
# src/hyprl/monitoring/data_quality.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


@dataclass
class DataQualityReport:
    """Rapport de qualité des données."""
    symbol: str
    timestamp: datetime
    is_valid: bool
    issues: list[str]
    staleness_seconds: int
    missing_fields: list[str]
    outliers_detected: int


class DataQualityMonitor:
    """Vérifie la qualité des données en temps réel."""

    # Seuils
    MAX_STALENESS_SECONDS = 120  # Données trop vieilles
    MAX_PRICE_CHANGE_PCT = 0.15  # 15% en une bougie = suspect
    MIN_VOLUME_RATIO = 0.1  # Volume < 10% de la moyenne = suspect

    def __init__(self):
        self.last_prices = {}
        self.volume_history = {}

    def validate_candle(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: datetime
    ) -> DataQualityReport:
        """Valide une bougie."""
        issues = []
        missing = []
        outliers = 0

        # 1. Vérifier les champs manquants/invalides
        if open_ <= 0 or np.isnan(open_):
            missing.append("open")
        if high <= 0 or np.isnan(high):
            missing.append("high")
        if low <= 0 or np.isnan(low):
            missing.append("low")
        if close <= 0 or np.isnan(close):
            missing.append("close")
        if volume < 0 or np.isnan(volume):
            missing.append("volume")

        # 2. Vérifier la cohérence OHLC
        if high < low:
            issues.append(f"High ({high}) < Low ({low})")
        if high < open_ or high < close:
            issues.append("High is not the highest")
        if low > open_ or low > close:
            issues.append("Low is not the lowest")

        # 3. Vérifier les outliers de prix
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            change_pct = abs(close - last_price) / last_price
            if change_pct > self.MAX_PRICE_CHANGE_PCT:
                issues.append(f"Price change {change_pct:.1%} exceeds threshold")
                outliers += 1
        self.last_prices[symbol] = close

        # 4. Vérifier le volume
        if symbol in self.volume_history:
            avg_volume = np.mean(self.volume_history[symbol][-20:])
            if volume < avg_volume * self.MIN_VOLUME_RATIO:
                issues.append(f"Volume {volume} < 10% of average {avg_volume:.0f}")

        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > 100:
            self.volume_history[symbol] = self.volume_history[symbol][-100:]

        # 5. Vérifier la fraîcheur
        staleness = (datetime.now() - timestamp).total_seconds()
        if staleness > self.MAX_STALENESS_SECONDS:
            issues.append(f"Data is {staleness:.0f}s old")

        return DataQualityReport(
            symbol=symbol,
            timestamp=timestamp,
            is_valid=len(missing) == 0 and len(issues) == 0,
            issues=issues,
            staleness_seconds=int(staleness),
            missing_fields=missing,
            outliers_detected=outliers
        )

    def validate_features(self, features: dict) -> tuple[bool, list[str]]:
        """Valide les features avant prédiction."""
        issues = []

        required_features = [
            'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',
            'atr_14', 'rsi_14', 'macd', 'bb_width'
        ]

        for feat in required_features:
            if feat not in features:
                issues.append(f"Missing feature: {feat}")
            elif np.isnan(features[feat]) or np.isinf(features[feat]):
                issues.append(f"Invalid value for {feat}: {features[feat]}")

        # Sanity checks
        if 'rsi_14' in features:
            if features['rsi_14'] < 0 or features['rsi_14'] > 100:
                issues.append(f"RSI out of range: {features['rsi_14']}")

        if 'atr_14' in features:
            if features['atr_14'] < 0:
                issues.append(f"Negative ATR: {features['atr_14']}")

        return len(issues) == 0, issues
```

---

## 6. HEALTH MONITORING - Surveillance Continue

### Dashboard de Santé

```python
# src/hyprl/monitoring/health_dashboard.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json


@dataclass
class HealthStatus:
    component: str
    status: str  # "healthy", "degraded", "critical", "unknown"
    last_check: datetime
    message: str
    metrics: Dict


class HealthMonitor:
    """Surveillance de la santé du système."""

    def __init__(self):
        self.components: Dict[str, HealthStatus] = {}

    def check_all(self) -> Dict[str, HealthStatus]:
        """Vérifie tous les composants."""
        checks = {
            "alpaca_api": self._check_alpaca(),
            "data_feed": self._check_data_feed(),
            "model": self._check_model(),
            "drift": self._check_drift(),
            "circuit_breakers": self._check_circuit_breakers(),
            "positions": self._check_positions(),
            "database": self._check_database(),
        }
        self.components = checks
        return checks

    def _check_alpaca(self) -> HealthStatus:
        """Vérifie la connexion Alpaca."""
        try:
            from alpaca.trading.client import TradingClient
            import os

            client = TradingClient(
                os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_SECRET_KEY"),
                paper=True
            )
            account = client.get_account()

            return HealthStatus(
                component="alpaca_api",
                status="healthy",
                last_check=datetime.now(),
                message="Connected",
                metrics={
                    "equity": float(account.equity),
                    "buying_power": float(account.buying_power),
                    "day_trade_count": account.daytrade_count
                }
            )
        except Exception as e:
            return HealthStatus(
                component="alpaca_api",
                status="critical",
                last_check=datetime.now(),
                message=str(e),
                metrics={}
            )

    def _check_model(self) -> HealthStatus:
        """Vérifie que les modèles sont chargés."""
        try:
            import joblib
            from pathlib import Path

            models = list(Path("models").glob("*_v3.joblib"))

            if len(models) == 0:
                return HealthStatus(
                    component="model",
                    status="critical",
                    last_check=datetime.now(),
                    message="No models found",
                    metrics={}
                )

            # Charger un modèle pour vérifier
            model = joblib.load(models[0])

            return HealthStatus(
                component="model",
                status="healthy",
                last_check=datetime.now(),
                message=f"{len(models)} models loaded",
                metrics={"model_count": len(models)}
            )
        except Exception as e:
            return HealthStatus(
                component="model",
                status="critical",
                last_check=datetime.now(),
                message=str(e),
                metrics={}
            )

    def get_overall_status(self) -> str:
        """Retourne le status global."""
        statuses = [c.status for c in self.components.values()]

        if "critical" in statuses:
            return "critical"
        elif "degraded" in statuses:
            return "degraded"
        elif all(s == "healthy" for s in statuses):
            return "healthy"
        else:
            return "unknown"

    def to_json(self) -> str:
        """Export JSON pour API."""
        return json.dumps({
            "overall": self.get_overall_status(),
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: {
                    "status": status.status,
                    "message": status.message,
                    "metrics": status.metrics
                }
                for name, status in self.components.items()
            }
        }, indent=2)
```

---

## 7. RECOVERY PROCEDURES - Procédures de Récupération

### Playbook de Récupération

| Situation | Détection | Action Automatique | Action Manuelle |
|-----------|-----------|-------------------|-----------------|
| Drift critique | PSI > 0.25 | Réduire positions 50% | Retrain modèle |
| 5 pertes consécutives | Circuit breaker | Stop trading 1h | Analyser trades |
| -10% daily | Circuit breaker | Fermer tout | Review stratégie |
| API Alpaca down | Health check | Retry 3x puis alert | Check status.alpaca.markets |
| Données stale > 5min | Data quality | Pas de nouveau trade | Vérifier feed |
| Corrélation > 0.85 | Correlation monitor | Max 1 position | Diversifier |
| Régime CRASH détecté | Regime detector | 0 trading | Attendre stabilisation |

### Script de Récupération d'Urgence

```bash
#!/bin/bash
# scripts/ops/emergency_recovery.sh

echo "=== HYPRL EMERGENCY RECOVERY ==="

# 1. Activer le kill switch
echo "1" > /home/kyo/HyprL/.kill_switch
echo "[X] Kill switch activated"

# 2. Fermer toutes les positions (via API)
python -c "
from alpaca.trading.client import TradingClient
import os

client = TradingClient(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    paper=True
)
client.close_all_positions(cancel_orders=True)
print('[X] All positions closed')
"

# 3. Arrêter les services
sudo systemctl stop hyprl-aggressive hyprl-normal hyprl-mix
echo "[X] Services stopped"

# 4. Sauvegarder les logs
cp /var/log/hyprl/*.log /home/kyo/HyprL/recovery_$(date +%Y%m%d_%H%M%S)/
echo "[X] Logs backed up"

echo "=== Recovery complete. Manual review required. ==="
```

---

## 8. CHECKLIST DE MISE EN PRODUCTION

### Avant Go-Live

- [ ] Drift detector initialisé avec baseline data
- [ ] Circuit breakers configurés (-2%, -5%, -10%, -15%)
- [ ] Kill switch file créé et testé
- [ ] Régime detector calibré sur 1 an de données
- [ ] Correlation matrix initialisée
- [ ] Data quality thresholds validés
- [ ] Health monitor endpoints exposés
- [ ] Alertes Discord/Telegram configurées
- [ ] Recovery scripts testés
- [ ] Backups automatiques en place

### Monitoring Quotidien

| Check | Fréquence | Alerte si |
|-------|-----------|-----------|
| Health status | 5 min | != healthy |
| Drift PSI | 1h | > 0.1 |
| Performance | 24h | Win rate < 45% |
| Correlation | 24h | > 0.8 |
| Equity curve | Continu | > -2% daily |
| Data freshness | 1 min | > 120s |

---

## Sources

- [Lessons from Algo Trading Failures](https://www.luxalgo.com/blog/lessons-from-algo-trading-failures/)
- [Why Most Trading Bots Lose Money](https://www.fortraders.com/blog/trading-bots-lose-money)
- [Why AI Trading Bots Fail](https://www.amplework.com/blog/ai-trading-bots-failures-how-to-build-profitable-bot/)
- [Systemic failures in algorithmic trading](https://pmc.ncbi.nlm.nih.gov/articles/PMC8978471/)
- [Trading System Kill Switch](https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box)
- [FIA Best Practices for Automated Trading](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf)
- [ML Monitoring and Drift Detection](https://www.bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection)
- [Evidently AI - Model Monitoring](https://www.evidentlyai.com/ml-in-production/model-monitoring)
