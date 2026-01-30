# Stratégie et Filtres HyprL

## Pipeline de décision

```
Model Prediction → Smart Filter → Sentiment Filter → Risk Gates → Final Signal
```

## 1. Smart Filter

### Chemin: `src/hyprl/strategy/smart_filter.py`

Le Smart Filter bloque les signaux quand les conditions de marché sont défavorables.

### Règles

```python
class SmartFilter:
    """Filtre intelligent basé sur les indicateurs techniques."""

    # Seuils
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MOMENTUM_THRESHOLD = 0.02  # 2%

    def should_block(self, signal: str, features: dict) -> tuple[bool, str]:
        """Retourne (blocked, reason)."""

        rsi = features.get('rsi_14', 50)
        ret_1h = features.get('ret_1h', 0)

        # Bloquer LONG si RSI suracheté
        if signal == 'long' and rsi > self.RSI_OVERBOUGHT:
            return True, "rsi_overbought"

        # Bloquer SHORT si RSI survendu
        if signal == 'short' and rsi < self.RSI_OVERSOLD:
            return True, "rsi_oversold"

        # Bloquer si momentum contre le signal
        if signal == 'long' and ret_1h < -self.MOMENTUM_THRESHOLD:
            return True, "momentum_against_long"

        if signal == 'short' and ret_1h > self.MOMENTUM_THRESHOLD:
            return True, "momentum_against_short"

        return False, ""
```

## 2. Sentiment Filter

### Chemin: `src/hyprl/sentiment/trading_filter.py`

Le Sentiment Filter utilise les news et les réseaux sociaux.

### Sources

| Source | Fiabilité | Méthode |
|--------|-----------|---------|
| Finviz News | 65% | Scraping headlines |
| Reddit WSB | 45% | API Reddit |
| Reddit Stocks | 55% | API Reddit |
| StockTwits | 50% | API StockTwits |
| Fear & Greed | 70% | CNN Index |

### Règles

```python
class SentimentFilter:
    """Filtre basé sur le sentiment de marché."""

    BULLISH_THRESHOLD = 0.3
    BEARISH_THRESHOLD = -0.3
    MIN_CONFIDENCE = 0.4

    def should_block(self, signal: str, sentiment: AggregatedSentiment) -> tuple[bool, str]:
        """Bloquer si sentiment contradictoire."""

        if sentiment.final_confidence < self.MIN_CONFIDENCE:
            return False, "low_confidence"  # Pas assez de données

        # Bloquer LONG si sentiment bearish
        if signal == 'long' and sentiment.final_score < self.BEARISH_THRESHOLD:
            return True, "sentiment_bearish"

        # Bloquer SHORT si sentiment bullish
        if signal == 'short' and sentiment.final_score > self.BULLISH_THRESHOLD:
            return True, "sentiment_bullish"

        return False, ""
```

### Chemin: `src/hyprl/sentiment/multi_source.py`

```python
class MultiSourceSentiment:
    """Agrège le sentiment de plusieurs sources."""

    def get_sentiment(self, symbol: str) -> AggregatedSentiment:
        """Retourne le sentiment agrégé avec score et confiance."""
        # Scrape toutes les sources
        # Calcule moyenne pondérée par fiabilité
        # Retourne score -1 à +1
```

## 3. Risk Gates

### Chemin: `src/hyprl/live/risk.py`

Gates de sécurité avant d'exécuter un trade.

### Gates implémentées

```python
RISK_GATES = {
    'max_daily_loss': 0.05,        # -5% max par jour
    'max_position_size': 0.40,     # 40% max par position
    'max_total_exposure': 1.50,    # 150% max d'exposition totale
    'min_cash_reserve': 0.05,      # 5% cash minimum
    'max_drawdown': 0.15,          # -15% drawdown = pause
}

class RiskGates:
    def check_all(self, account, signal) -> tuple[bool, str]:
        """Vérifie toutes les gates."""

        if self.daily_loss_exceeded(account):
            return False, "daily_loss_limit"

        if self.position_too_large(signal):
            return False, "position_size_limit"

        if self.exposure_too_high(account):
            return False, "exposure_limit"

        return True, "passed"
```

## 4. Position-Aware Exit Rules

### Chemin: `src/hyprl/strategy/position_aware.py`

Règles de sortie dynamiques basées sur l'état de la position.

### Règles

| Situation | Action | Urgence |
|-----------|--------|---------|
| Profit > 1% + sentiment retourne | CLOSE | Soon |
| Profit > 2% | CLOSE | Soon |
| Perte + sentiment empire | CLOSE | Immediate |
| Perte > 1.5% | CLOSE | Immediate |
| Position > 24h + profit | CLOSE | Soon |
| Perte + sentiment s'améliore | HOLD | Monitor |

```python
class DynamicExitRules:
    PROFIT_TAKE_PCT = 2.0
    SENTIMENT_REVERSAL_THRESHOLD = 0.3
    MAX_HOLD_HOURS = 24
    LOSS_CUT_PCT = -1.5

    def evaluate(self, position: Position) -> ExitSignal:
        # Profit + Sentiment Reversal
        if position.unrealized_pnl_pct > 1.0:
            if position.sentiment_direction == "deteriorating":
                return ExitSignal(action="close", reason="profit_sentiment_reversal")

        # Max Loss
        if position.unrealized_pnl_pct < self.LOSS_CUT_PCT:
            return ExitSignal(action="close", reason="max_loss", urgency="immediate")

        # ...
```

## 5. Configurations par stratégie

### Normal (conservateur)

```yaml
strategy:
  name: "normal"

capital:
  position_size_pct: 0.20      # 20% par trade
  max_position_pct: 0.40       # 40% max par symbole
  max_total_exposure: 1.00     # 100% pas de levier

risk:
  stop_loss_atr: 3.0           # 3 ATR stop
  take_profit_atr: 6.0         # 6 ATR target

filters:
  enable_smart_filter: true
  enable_sentiment: true
```

### Aggressive (risqué)

```yaml
strategy:
  name: "aggressive"

capital:
  position_size_pct: 0.40      # 40% par trade
  max_position_pct: 0.60       # 60% max par symbole
  max_total_exposure: 1.80     # 180% levier

risk:
  stop_loss_atr: 2.5           # 2.5 ATR stop (plus serré)
  take_profit_atr: 5.0         # 5 ATR target

filters:
  enable_smart_filter: true
  enable_sentiment: true
```

### Mix (70/30)

```yaml
strategy:
  name: "mix"

capital:
  position_size_pct: 0.30      # 30% par trade
  max_position_pct: 0.50       # 50% max par symbole
  max_total_exposure: 1.40     # 140% levier modéré

filters:
  enable_smart_filter: true
  enable_sentiment: true
```

## 6. Signal final

### Chemin: `src/hyprl/strategy/core.py`

```python
class CoreStrategy:
    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        """Génère le signal final après tous les filtres."""

        # 1. Prédiction ML
        prob = self.model.predict(features)
        raw_signal = self.prob_to_signal(prob)

        # 2. Smart Filter
        if self.smart_filter.should_block(raw_signal, features):
            return Signal(decision="flat", reason="smart_filter_blocked")

        # 3. Sentiment Filter
        sentiment = self.sentiment.get_sentiment(symbol)
        if self.sentiment_filter.should_block(raw_signal, sentiment):
            return Signal(decision="flat", reason="sentiment_blocked")

        # 4. Risk Gates
        if not self.risk_gates.check_all():
            return Signal(decision="flat", reason="risk_gate_blocked")

        # 5. Signal validé
        return Signal(
            decision=raw_signal,
            probability=prob,
            sentiment=sentiment.final_score
        )
```
