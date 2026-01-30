# HyprL Strategy 2.0 - Plan d'Améliorations
> Version: 2026-01-09 | Priorisation des features manquantes

## Résumé Exécutif

| Priorité | Feature | Effort | Impact Estimé | Status |
|----------|---------|--------|---------------|--------|
| 1 | Calendrier Events | 2-3h | Évite 50% pertes surprises | TODO |
| 2 | Corrélation Portfolio | 3-4h | Survie (-5% → -2% max) | TODO |
| 3 | Régime Macro (VIX) | 4h | +15-25% perf | TODO |
| 4 | Liquidity Sizing | 3h | Meilleure exécution | TODO |
| 5 | Options Flow | 1 jour | +10-20% perf | TODO |

---

## 1. Calendrier Events (Priorité Critique)

### 1.1 Problème
- Earnings surprises causent des gaps de 5-15%
- FOMC days = volatilité imprévisible
- Triple witching (OpEx) = mouvements erratiques

### 1.2 Solution
Bloquer les trades N jours avant/après ces events.

### 1.3 Features à Implémenter

| Feature | Type | Description |
|---------|------|-------------|
| `days_to_earnings` | int | Jours jusqu'au prochain earnings (-1 si passé) |
| `days_from_earnings` | int | Jours depuis le dernier earnings |
| `is_earnings_week` | bool | True si earnings dans 5 jours |
| `is_fomc_day` | bool | True si jour de décision Fed |
| `is_fomc_week` | bool | True si semaine FOMC |
| `is_opex_day` | bool | True si 3ème vendredi du mois |
| `is_opex_week` | bool | True si semaine OpEx |

### 1.4 Règles de Trading

```python
# Dans strategy/core.py ou bridge

def should_skip_event_risk(symbol: str, event_data: dict) -> tuple[bool, str]:
    """Return (skip, reason) based on event calendar."""

    # Earnings blackout: 3 jours avant, 1 jour après
    if event_data.get("days_to_earnings", 999) <= 3:
        return True, f"earnings_blackout_{event_data['days_to_earnings']}d"
    if event_data.get("days_from_earnings", 999) <= 1:
        return True, "post_earnings_blackout"

    # FOMC: pas de nouvelles positions le jour J
    if event_data.get("is_fomc_day", False):
        return True, "fomc_day_blackout"

    # OpEx: réduire taille ou skip
    if event_data.get("is_opex_day", False):
        return True, "opex_day_blackout"

    return False, ""
```

### 1.5 Sources de Données

| Data | Source | Coût | Fiabilité |
|------|--------|------|-----------|
| Earnings dates | Yahoo Finance API | Gratuit | ⭐⭐⭐ |
| Earnings dates | Alpha Vantage | Gratuit (500/jour) | ⭐⭐⭐⭐ |
| Earnings dates | Finnhub | Gratuit (60/min) | ⭐⭐⭐⭐ |
| FOMC dates | federalreserve.gov | Gratuit | ⭐⭐⭐⭐⭐ |
| OpEx dates | Calcul (3ème vendredi) | N/A | ⭐⭐⭐⭐⭐ |

### 1.6 Implémentation

```python
# src/hyprl/calendar/events.py

from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

class EventCalendar:
    """Track market events that affect trading decisions."""

    # FOMC 2026 dates (à mettre à jour annuellement)
    FOMC_DATES_2026 = [
        "2026-01-28", "2026-01-29",  # Jan
        "2026-03-17", "2026-03-18",  # Mar
        "2026-05-05", "2026-05-06",  # May
        "2026-06-16", "2026-06-17",  # Jun
        "2026-07-28", "2026-07-29",  # Jul
        "2026-09-15", "2026-09-16",  # Sep
        "2026-11-03", "2026-11-04",  # Nov
        "2026-12-15", "2026-12-16",  # Dec
    ]

    def __init__(self):
        self._earnings_cache = {}

    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Fetch next earnings date for symbol."""
        if symbol in self._earnings_cache:
            cached_date, cached_at = self._earnings_cache[symbol]
            if datetime.now() - cached_at < timedelta(hours=6):
                return cached_date

        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and "Earnings Date" in cal:
                earnings_date = cal["Earnings Date"][0]
                self._earnings_cache[symbol] = (earnings_date, datetime.now())
                return earnings_date
        except Exception:
            pass
        return None

    def days_to_earnings(self, symbol: str) -> int:
        """Return days until next earnings, or 999 if unknown."""
        earnings = self.get_earnings_date(symbol)
        if earnings is None:
            return 999
        delta = (earnings - datetime.now()).days
        return max(delta, -30)  # Cap at -30 for past earnings

    def is_fomc_day(self, date: datetime = None) -> bool:
        """Check if date is a FOMC decision day."""
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        return date_str in self.FOMC_DATES_2026

    def is_fomc_week(self, date: datetime = None) -> bool:
        """Check if date is in a FOMC week."""
        date = date or datetime.now()
        for fomc_str in self.FOMC_DATES_2026:
            fomc_date = datetime.strptime(fomc_str, "%Y-%m-%d")
            if abs((date - fomc_date).days) <= 2:
                return True
        return False

    @staticmethod
    def get_opex_date(year: int, month: int) -> datetime:
        """Get options expiration (3rd Friday) for given month."""
        # Find first day of month
        first_day = datetime(year, month, 1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        # Third Friday
        return first_friday + timedelta(weeks=2)

    def is_opex_day(self, date: datetime = None) -> bool:
        """Check if date is options expiration day."""
        date = date or datetime.now()
        opex = self.get_opex_date(date.year, date.month)
        return date.date() == opex.date()

    def is_opex_week(self, date: datetime = None) -> bool:
        """Check if date is in options expiration week."""
        date = date or datetime.now()
        opex = self.get_opex_date(date.year, date.month)
        return abs((date - opex).days) <= 2

    def get_event_data(self, symbol: str) -> dict:
        """Get all event data for a symbol."""
        now = datetime.now()
        return {
            "days_to_earnings": self.days_to_earnings(symbol),
            "is_fomc_day": self.is_fomc_day(now),
            "is_fomc_week": self.is_fomc_week(now),
            "is_opex_day": self.is_opex_day(now),
            "is_opex_week": self.is_opex_week(now),
        }
```

### 1.7 Intégration Bridge

```python
# Dans run_alpaca_bridge.py, après les guards existants

from hyprl.calendar.events import EventCalendar

event_calendar = EventCalendar()

# Dans la boucle principale, avant de soumettre un ordre:
event_data = event_calendar.get_event_data(symbol)
skip, reason = should_skip_event_risk(symbol, event_data)
if skip:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "event_calendar_skip",
        "symbol": symbol,
        "reason": reason,
        "event_data": event_data,
    })
    continue
```

### 1.8 Tests

```python
# tests/calendar/test_events.py

def test_opex_january_2026():
    cal = EventCalendar()
    opex = cal.get_opex_date(2026, 1)
    assert opex.day == 16  # 3rd Friday of Jan 2026
    assert opex.weekday() == 4  # Friday

def test_fomc_detection():
    cal = EventCalendar()
    from datetime import datetime
    assert cal.is_fomc_day(datetime(2026, 1, 29))
    assert not cal.is_fomc_day(datetime(2026, 1, 30))
```

---

## 2. Corrélation Portfolio (Priorité Critique)

### 2.1 Problème

Portfolio actuel:
- NVDA: 40%
- MSFT: 30%
- QQQ: 30%

Corrélations typiques:
```
        NVDA    MSFT    QQQ
NVDA    1.00    0.75    0.85
MSFT    0.75    1.00    0.90
QQQ     0.85    0.90    1.00
```

**Risque:** Une journée "tech down" = toutes les positions perdent ensemble.

### 2.2 Solution

Limiter l'exposition corrélée totale.

### 2.3 Features à Implémenter

| Feature | Type | Description |
|---------|------|-------------|
| `portfolio_correlation` | float | Corrélation moyenne pondérée |
| `sector_exposure` | dict | Exposition par secteur (tech, etc.) |
| `max_correlated_positions` | int | Limite positions corrélées |
| `correlation_adjusted_size` | float | Taille ajustée par corrélation |

### 2.4 Règles de Trading

```python
# Règles de corrélation

MAX_CORRELATED_NOTIONAL = 15000  # Max $15k en positions corrélées (>0.7)
MAX_SECTOR_EXPOSURE = 0.6        # Max 60% dans un secteur
MAX_SINGLE_NAME = 0.4            # Max 40% sur une action

def check_correlation_limits(
    new_position: dict,
    current_positions: list,
    correlations: dict
) -> tuple[bool, str, float]:
    """
    Check if new position respects correlation limits.
    Returns (allowed, reason, adjusted_size_multiplier)
    """
    symbol = new_position["symbol"]
    notional = new_position["notional"]

    # Calculer exposition corrélée
    correlated_exposure = 0
    for pos in current_positions:
        corr = correlations.get((symbol, pos["symbol"]), 0)
        if corr > 0.7:
            correlated_exposure += pos["notional"] * corr

    # Si ajout dépasse limite
    if correlated_exposure + notional > MAX_CORRELATED_NOTIONAL:
        # Réduire taille
        allowed_notional = MAX_CORRELATED_NOTIONAL - correlated_exposure
        if allowed_notional <= 0:
            return False, "correlation_limit_reached", 0
        multiplier = allowed_notional / notional
        return True, "correlation_size_reduced", multiplier

    return True, "", 1.0
```

### 2.5 Matrice de Corrélation

```python
# src/hyprl/risk/correlation.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class CorrelationTracker:
    """Track and update correlation matrix."""

    def __init__(self, symbols: list[str], lookback_days: int = 60):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self._matrix = None
        self._last_update = None

    def update(self) -> pd.DataFrame:
        """Fetch fresh correlation matrix."""
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        data = {}
        for sym in self.symbols:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, end=end)
            if not hist.empty:
                data[sym] = hist["Close"].pct_change().dropna()

        df = pd.DataFrame(data)
        self._matrix = df.corr()
        self._last_update = datetime.now()
        return self._matrix

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if self._matrix is None or self._stale():
            self.update()

        if sym1 == sym2:
            return 1.0

        try:
            return self._matrix.loc[sym1, sym2]
        except KeyError:
            return 0.5  # Default moderate correlation

    def _stale(self) -> bool:
        """Check if matrix needs refresh (>6h old)."""
        if self._last_update is None:
            return True
        return (datetime.now() - self._last_update).seconds > 21600

    def get_portfolio_risk(self, positions: list[dict]) -> dict:
        """
        Calculate portfolio-level risk metrics.
        positions: [{"symbol": "NVDA", "notional": 5000, "side": "long"}, ...]
        """
        if not positions:
            return {"weighted_correlation": 0, "concentration_risk": 0}

        total_notional = sum(p["notional"] for p in positions)

        # Weighted average correlation
        weighted_corr = 0
        pairs = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                corr = self.get_correlation(p1["symbol"], p2["symbol"])
                weight = (p1["notional"] * p2["notional"]) / (total_notional ** 2)
                weighted_corr += corr * weight
                pairs += 1

        # Concentration (Herfindahl index)
        hhi = sum((p["notional"] / total_notional) ** 2 for p in positions)

        return {
            "weighted_correlation": weighted_corr,
            "concentration_risk": hhi,
            "num_positions": len(positions),
            "total_notional": total_notional,
        }
```

### 2.6 Intégration Bridge

```python
# Dans run_alpaca_bridge.py

from hyprl.risk.correlation import CorrelationTracker

correlation_tracker = CorrelationTracker(["NVDA", "MSFT", "QQQ"])

# Avant d'ouvrir une position:
current_positions = [
    {"symbol": p.symbol, "notional": float(p.market_value), "side": p.side}
    for p in broker.list_positions()
]

portfolio_risk = correlation_tracker.get_portfolio_risk(current_positions)

# Log risk metrics
append_jsonl(out_path, {
    "ts": utc_now_iso(),
    "event": "portfolio_risk_check",
    "risk_metrics": portfolio_risk,
})

# Check if new position allowed
new_pos = {"symbol": symbol, "notional": entry_price * qty}
allowed, reason, size_mult = check_correlation_limits(
    new_pos, current_positions,
    lambda s1, s2: correlation_tracker.get_correlation(s1, s2)
)

if not allowed:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "correlation_limit_reject",
        "symbol": symbol,
        "reason": reason,
    })
    continue

if size_mult < 1.0:
    qty = int(qty * size_mult)
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "correlation_size_adjustment",
        "symbol": symbol,
        "multiplier": size_mult,
        "new_qty": qty,
    })
```

---

## 3. Régime Macro (VIX)

### 3.1 Problème

Le modèle actuel ignore le contexte macro:
- VIX à 12 vs VIX à 30 = stratégies différentes
- Fed hawkish = tech souffre
- Risk-off days = corrélations montent à 1

### 3.2 Solution

Ajouter des features macro et adapter la stratégie selon le régime.

### 3.3 Features à Implémenter

| Feature | Type | Description |
|---------|------|-------------|
| `vix_level` | float | Niveau VIX actuel |
| `vix_percentile_52w` | float | Percentile VIX sur 52 semaines |
| `vix_change_1d` | float | Variation VIX 1 jour |
| `vix_term_structure` | float | VIX - VIX3M (contango/backwardation) |
| `spy_trend_20d` | float | Trend SPY 20 jours |
| `spy_above_200ma` | bool | SPY au-dessus de sa MM200 |
| `rate_10y` | float | Taux 10 ans US |
| `rate_10y_change_5d` | float | Variation taux 5 jours |

### 3.4 Régimes de Marché

```python
# src/hyprl/regime/macro.py

from enum import Enum
from dataclasses import dataclass

class MarketRegime(Enum):
    RISK_ON = "risk_on"           # VIX < 15, SPY trending up
    NEUTRAL = "neutral"           # VIX 15-20, mixed signals
    CAUTIOUS = "cautious"         # VIX 20-25, uncertainty
    RISK_OFF = "risk_off"         # VIX > 25, panic mode
    CRISIS = "crisis"             # VIX > 35, extreme fear

@dataclass
class RegimeConfig:
    """Trading parameters per regime."""
    regime: MarketRegime
    position_size_mult: float     # Multiplier on position size
    threshold_adjustment: float   # Add to probability thresholds
    allow_shorts: bool
    max_positions: int

REGIME_CONFIGS = {
    MarketRegime.RISK_ON: RegimeConfig(
        regime=MarketRegime.RISK_ON,
        position_size_mult=1.0,
        threshold_adjustment=0.0,
        allow_shorts=True,
        max_positions=3,
    ),
    MarketRegime.NEUTRAL: RegimeConfig(
        regime=MarketRegime.NEUTRAL,
        position_size_mult=0.8,
        threshold_adjustment=0.02,  # +2% on thresholds
        allow_shorts=True,
        max_positions=2,
    ),
    MarketRegime.CAUTIOUS: RegimeConfig(
        regime=MarketRegime.CAUTIOUS,
        position_size_mult=0.5,
        threshold_adjustment=0.05,  # +5% on thresholds
        allow_shorts=True,
        max_positions=2,
    ),
    MarketRegime.RISK_OFF: RegimeConfig(
        regime=MarketRegime.RISK_OFF,
        position_size_mult=0.3,
        threshold_adjustment=0.10,  # +10% on thresholds
        allow_shorts=False,         # Pas de shorts en panic
        max_positions=1,
    ),
    MarketRegime.CRISIS: RegimeConfig(
        regime=MarketRegime.CRISIS,
        position_size_mult=0.0,     # Pas de nouveaux trades
        threshold_adjustment=0.20,
        allow_shorts=False,
        max_positions=0,
    ),
}
```

### 3.5 Détection du Régime

```python
# src/hyprl/regime/detector.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class RegimeDetector:
    """Detect current market regime based on VIX and macro indicators."""

    def __init__(self):
        self._cache = {}
        self._cache_time = None

    def get_vix_data(self) -> dict:
        """Fetch VIX data."""
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1y")

        if hist.empty:
            return {"vix_level": 20, "vix_percentile": 50}

        current = hist["Close"].iloc[-1]
        percentile = (hist["Close"] < current).mean() * 100
        change_1d = (current / hist["Close"].iloc[-2] - 1) * 100 if len(hist) > 1 else 0

        # Term structure (VIX vs VIX3M)
        try:
            vix3m = yf.Ticker("^VIX3M")
            vix3m_hist = vix3m.history(period="5d")
            vix3m_current = vix3m_hist["Close"].iloc[-1] if not vix3m_hist.empty else current
            term_structure = current - vix3m_current
        except:
            term_structure = 0

        return {
            "vix_level": current,
            "vix_percentile_52w": percentile,
            "vix_change_1d": change_1d,
            "vix_term_structure": term_structure,
        }

    def get_spy_data(self) -> dict:
        """Fetch SPY trend data."""
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")

        if hist.empty:
            return {"spy_trend_20d": 0, "spy_above_200ma": True}

        current = hist["Close"].iloc[-1]
        ma20 = hist["Close"].rolling(20).mean().iloc[-1]
        ma200 = hist["Close"].rolling(200).mean().iloc[-1]

        trend_20d = (current / ma20 - 1) * 100

        return {
            "spy_level": current,
            "spy_trend_20d": trend_20d,
            "spy_above_200ma": current > ma200,
            "spy_ma200": ma200,
        }

    def get_rates_data(self) -> dict:
        """Fetch interest rate data."""
        try:
            tnx = yf.Ticker("^TNX")  # 10-year yield
            hist = tnx.history(period="1mo")

            if hist.empty:
                return {"rate_10y": 4.0, "rate_10y_change_5d": 0}

            current = hist["Close"].iloc[-1]
            change_5d = current - hist["Close"].iloc[-6] if len(hist) >= 6 else 0

            return {
                "rate_10y": current,
                "rate_10y_change_5d": change_5d,
            }
        except:
            return {"rate_10y": 4.0, "rate_10y_change_5d": 0}

    def detect_regime(self) -> tuple[MarketRegime, dict]:
        """
        Detect current market regime.
        Returns (regime, all_data)
        """
        # Check cache (valid for 5 minutes)
        if self._cache_time and (datetime.now() - self._cache_time).seconds < 300:
            return self._cache["regime"], self._cache["data"]

        vix = self.get_vix_data()
        spy = self.get_spy_data()
        rates = self.get_rates_data()

        all_data = {**vix, **spy, **rates}

        # Determine regime primarily from VIX
        vix_level = vix["vix_level"]

        if vix_level > 35:
            regime = MarketRegime.CRISIS
        elif vix_level > 25:
            regime = MarketRegime.RISK_OFF
        elif vix_level > 20:
            regime = MarketRegime.CAUTIOUS
        elif vix_level > 15:
            regime = MarketRegime.NEUTRAL
        else:
            regime = MarketRegime.RISK_ON

        # Adjust based on VIX spike
        if vix["vix_change_1d"] > 20:  # VIX up 20%+ in a day
            # Upgrade risk level
            regime_order = list(MarketRegime)
            idx = regime_order.index(regime)
            if idx < len(regime_order) - 1:
                regime = regime_order[idx + 1]

        # Adjust based on SPY trend
        if not spy["spy_above_200ma"] and regime == MarketRegime.RISK_ON:
            regime = MarketRegime.NEUTRAL

        # Cache results
        self._cache = {"regime": regime, "data": all_data}
        self._cache_time = datetime.now()

        return regime, all_data
```

### 3.6 Intégration Bridge

```python
# Dans run_alpaca_bridge.py

from hyprl.regime.detector import RegimeDetector
from hyprl.regime.macro import REGIME_CONFIGS, MarketRegime

regime_detector = RegimeDetector()

# Au début de chaque cycle de trading:
current_regime, macro_data = regime_detector.detect_regime()
regime_config = REGIME_CONFIGS[current_regime]

append_jsonl(out_path, {
    "ts": utc_now_iso(),
    "event": "regime_check",
    "regime": current_regime.value,
    "config": {
        "position_size_mult": regime_config.position_size_mult,
        "threshold_adjustment": regime_config.threshold_adjustment,
        "allow_shorts": regime_config.allow_shorts,
        "max_positions": regime_config.max_positions,
    },
    "macro_data": macro_data,
})

# Skip si CRISIS
if current_regime == MarketRegime.CRISIS:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "regime_crisis_skip",
    })
    continue

# Ajuster les thresholds
adjusted_long_threshold = config["thresholds"]["long"] + regime_config.threshold_adjustment
adjusted_short_threshold = config["thresholds"]["short"] + regime_config.threshold_adjustment

# Ajuster la taille
adjusted_qty = int(qty * regime_config.position_size_mult)

# Vérifier shorts autorisés
if decision == "short" and not regime_config.allow_shorts:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "regime_short_blocked",
        "regime": current_regime.value,
    })
    continue
```

---

## 4. Liquidity-Adjusted Sizing

### 4.1 Problème

- Taille fixe ignore la liquidité du moment
- First/last 30 min = spreads plus larges, slippage plus élevé
- Faible volume = market impact

### 4.2 Solution

Adapter la taille selon la liquidité disponible.

### 4.3 Features à Implémenter

| Feature | Type | Description |
|---------|------|-------------|
| `volume_20d_avg` | int | Volume moyen 20 jours |
| `volume_today_pct` | float | Volume aujourd'hui vs moyenne |
| `spread_percentile` | float | Spread actuel vs historique |
| `time_of_day_liquidity` | float | Multiplicateur selon l'heure |
| `market_impact_estimate` | float | Impact estimé de notre ordre |

### 4.4 Règles

```python
# src/hyprl/risk/liquidity.py

from datetime import datetime
from zoneinfo import ZoneInfo

class LiquidityManager:
    """Manage position sizing based on liquidity."""

    # Liquidity multipliers by time of day (ET)
    TIME_MULTIPLIERS = {
        (9, 30): 0.5,   # First 30 min: reduce 50%
        (10, 0): 0.7,   # 10:00-10:30: reduce 30%
        (10, 30): 1.0,  # Normal liquidity
        (15, 30): 0.7,  # Last 30 min warning
        (15, 45): 0.5,  # Last 15 min: reduce 50%
    }

    MAX_VOLUME_PCT = 0.01  # Max 1% of daily volume

    def __init__(self):
        self._volume_cache = {}

    def get_time_multiplier(self) -> float:
        """Get position size multiplier based on time of day."""
        et = ZoneInfo("America/New_York")
        now = datetime.now(et)
        hour, minute = now.hour, now.minute

        # Before market open
        if hour < 9 or (hour == 9 and minute < 30):
            return 0.5

        # First 30 minutes
        if hour == 9 and minute < 60:
            return 0.5

        # 10:00-10:30
        if hour == 10 and minute < 30:
            return 0.7

        # Last 30 minutes
        if hour == 15 and minute >= 30:
            return 0.5

        # Normal hours
        return 1.0

    def get_volume_limit(self, symbol: str, avg_volume: int) -> int:
        """
        Get max shares based on volume limit.
        We don't want to be more than 1% of daily volume.
        """
        return int(avg_volume * self.MAX_VOLUME_PCT)

    def get_spread_multiplier(self, current_spread_pct: float, avg_spread_pct: float) -> float:
        """
        Reduce size if spread is wider than usual.
        """
        if avg_spread_pct <= 0:
            return 1.0

        spread_ratio = current_spread_pct / avg_spread_pct

        if spread_ratio > 3.0:
            return 0.3  # Spread 3x normal: reduce 70%
        elif spread_ratio > 2.0:
            return 0.5  # Spread 2x normal: reduce 50%
        elif spread_ratio > 1.5:
            return 0.7  # Spread 1.5x normal: reduce 30%
        else:
            return 1.0

    def adjust_position_size(
        self,
        requested_qty: int,
        symbol: str,
        avg_volume: int,
        current_spread_pct: float,
        avg_spread_pct: float,
    ) -> tuple[int, dict]:
        """
        Adjust position size based on liquidity factors.
        Returns (adjusted_qty, factors)
        """
        time_mult = self.get_time_multiplier()
        volume_limit = self.get_volume_limit(symbol, avg_volume)
        spread_mult = self.get_spread_multiplier(current_spread_pct, avg_spread_pct)

        # Apply all multipliers
        adjusted = requested_qty
        adjusted = int(adjusted * time_mult)
        adjusted = min(adjusted, volume_limit)
        adjusted = int(adjusted * spread_mult)

        factors = {
            "time_multiplier": time_mult,
            "volume_limit": volume_limit,
            "spread_multiplier": spread_mult,
            "original_qty": requested_qty,
            "final_qty": adjusted,
        }

        return adjusted, factors
```

### 4.5 Intégration Bridge

```python
# Dans run_alpaca_bridge.py

from hyprl.risk.liquidity import LiquidityManager

liquidity_mgr = LiquidityManager()

# Avant de soumettre l'ordre:
# Fetch quote et volume
quote = broker.get_latest_quote(symbol)
avg_volume = get_avg_volume(symbol, 20)  # À implémenter
current_spread = (quote.ask - quote.bid) / quote.bid * 100
avg_spread = get_avg_spread(symbol, 20)  # À implémenter

adjusted_qty, liquidity_factors = liquidity_mgr.adjust_position_size(
    requested_qty=qty,
    symbol=symbol,
    avg_volume=avg_volume,
    current_spread_pct=current_spread,
    avg_spread_pct=avg_spread,
)

append_jsonl(out_path, {
    "ts": utc_now_iso(),
    "event": "liquidity_adjustment",
    "symbol": symbol,
    "factors": liquidity_factors,
})

if adjusted_qty < min_position_size:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "liquidity_skip",
        "symbol": symbol,
        "reason": "adjusted_qty_below_minimum",
        "adjusted_qty": adjusted_qty,
        "minimum": min_position_size,
    })
    continue

qty = adjusted_qty
```

---

## 5. Options Flow (Bonus)

### 5.1 Problème

- Institutionnels tradent souvent via options avant le spot
- Unusual options activity = signal leading indicator
- Put/call ratio = sentiment institutionnel

### 5.2 Solution

Intégrer des données options comme features.

### 5.3 Features à Implémenter

| Feature | Type | Description |
|---------|------|-------------|
| `put_call_ratio` | float | Ratio puts/calls (volume) |
| `put_call_oi_ratio` | float | Ratio puts/calls (open interest) |
| `iv_rank` | float | IV percentile 52 semaines |
| `iv_skew` | float | Skew put-call IV |
| `unusual_calls` | bool | Volume calls > 2x moyenne |
| `unusual_puts` | bool | Volume puts > 2x moyenne |
| `max_pain` | float | Prix max pain options |

### 5.4 Sources de Données

| Source | Coût | Features |
|--------|------|----------|
| Yahoo Finance | Gratuit | Options chains basiques |
| CBOE | $$ | Put/call ratio officiel |
| Unusual Whales | $50/mois | Unusual activity, flow |
| TradingView | $15-60/mois | IV, charts |
| Polygon.io | $29/mois | Options data complètes |

### 5.5 Implémentation (Yahoo Finance - Gratuit)

```python
# src/hyprl/options/flow.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class OptionsFlowAnalyzer:
    """Analyze options flow for trading signals."""

    def __init__(self):
        self._cache = {}

    def get_options_data(self, symbol: str) -> dict:
        """Fetch and analyze options data for symbol."""
        try:
            ticker = yf.Ticker(symbol)

            # Get nearest expiration
            expirations = ticker.options
            if not expirations:
                return self._default_data()

            # Use nearest weekly/monthly
            nearest = expirations[0]
            chain = ticker.option_chain(nearest)

            calls = chain.calls
            puts = chain.puts

            # Volume-based put/call ratio
            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0

            # Open interest ratio
            call_oi = calls["openInterest"].sum()
            put_oi = puts["openInterest"].sum()
            pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0

            # IV analysis (average)
            avg_call_iv = calls["impliedVolatility"].mean()
            avg_put_iv = puts["impliedVolatility"].mean()
            iv_skew = avg_put_iv - avg_call_iv

            # Unusual activity detection
            call_vol_mean = calls["volume"].mean()
            put_vol_mean = puts["volume"].mean()
            unusual_calls = (calls["volume"] > 2 * call_vol_mean).any()
            unusual_puts = (puts["volume"] > 2 * put_vol_mean).any()

            # Max pain calculation
            max_pain = self._calculate_max_pain(calls, puts)

            return {
                "put_call_ratio": pc_ratio,
                "put_call_oi_ratio": pc_oi_ratio,
                "avg_iv": (avg_call_iv + avg_put_iv) / 2,
                "iv_skew": iv_skew,
                "unusual_calls": unusual_calls,
                "unusual_puts": unusual_puts,
                "max_pain": max_pain,
                "expiration": nearest,
            }

        except Exception as e:
            return self._default_data()

    def _calculate_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame) -> float:
        """Calculate max pain price (where options expire worthless)."""
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))

            min_pain = float("inf")
            max_pain_strike = strikes[len(strikes) // 2]

            for strike in strikes:
                # Pain for call holders if price at strike
                call_pain = calls[calls["strike"] < strike].apply(
                    lambda row: (strike - row["strike"]) * row["openInterest"], axis=1
                ).sum()

                # Pain for put holders if price at strike
                put_pain = puts[puts["strike"] > strike].apply(
                    lambda row: (row["strike"] - strike) * row["openInterest"], axis=1
                ).sum()

                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike

            return max_pain_strike
        except:
            return 0

    def _default_data(self) -> dict:
        """Return default values when data unavailable."""
        return {
            "put_call_ratio": 1.0,
            "put_call_oi_ratio": 1.0,
            "avg_iv": 0.3,
            "iv_skew": 0,
            "unusual_calls": False,
            "unusual_puts": False,
            "max_pain": 0,
            "expiration": None,
        }

    def get_signal_adjustment(self, options_data: dict, decision: str) -> tuple[float, str]:
        """
        Adjust signal based on options flow.
        Returns (confidence_adjustment, reason)

        Positive adjustment = more confident in signal
        Negative adjustment = less confident
        """
        adjustments = []

        pc_ratio = options_data["put_call_ratio"]

        if decision == "long":
            # Bullish if put/call ratio high (contrarian)
            if pc_ratio > 1.5:
                adjustments.append((0.02, "high_put_call_contrarian"))
            # Bearish if unusual puts
            if options_data["unusual_puts"]:
                adjustments.append((-0.03, "unusual_put_activity"))
            # Bullish if unusual calls
            if options_data["unusual_calls"]:
                adjustments.append((0.02, "unusual_call_activity"))

        elif decision == "short":
            # Bearish if put/call ratio low (contrarian)
            if pc_ratio < 0.7:
                adjustments.append((0.02, "low_put_call_contrarian"))
            # Bullish if unusual calls (bad for short)
            if options_data["unusual_calls"]:
                adjustments.append((-0.03, "unusual_call_activity"))
            # Bearish if unusual puts
            if options_data["unusual_puts"]:
                adjustments.append((0.02, "unusual_put_activity"))

        # IV skew adjustment
        iv_skew = options_data["iv_skew"]
        if iv_skew > 0.1:  # Puts more expensive
            if decision == "short":
                adjustments.append((0.01, "put_skew_confirms"))
            else:
                adjustments.append((-0.01, "put_skew_warns"))

        total_adj = sum(a[0] for a in adjustments)
        reasons = [a[1] for a in adjustments]

        return total_adj, ",".join(reasons) if reasons else "none"
```

### 5.6 Intégration Bridge

```python
# Dans run_alpaca_bridge.py

from hyprl.options.flow import OptionsFlowAnalyzer

options_analyzer = OptionsFlowAnalyzer()

# Après avoir reçu le signal, avant de trader:
options_data = options_analyzer.get_options_data(symbol)
options_adj, options_reason = options_analyzer.get_signal_adjustment(options_data, decision)

append_jsonl(out_path, {
    "ts": utc_now_iso(),
    "event": "options_flow_check",
    "symbol": symbol,
    "options_data": options_data,
    "adjustment": options_adj,
    "reason": options_reason,
})

# Ajuster la probabilité
adjusted_prob = probability + options_adj

# Re-vérifier le threshold avec la prob ajustée
if decision == "long" and adjusted_prob < long_threshold:
    append_jsonl(out_path, {
        "ts": utc_now_iso(),
        "event": "options_flow_reject",
        "symbol": symbol,
        "original_prob": probability,
        "adjusted_prob": adjusted_prob,
        "threshold": long_threshold,
    })
    continue
```

---

## 6. Plan d'Implémentation

### Phase 1: Protection (Semaine 1)
- [ ] Calendrier Events (2-3h)
- [ ] Corrélation Portfolio (3-4h)
- [ ] Tests unitaires
- [ ] Déploiement en paper

### Phase 2: Optimisation (Semaine 2)
- [ ] Régime Macro VIX (4h)
- [ ] Liquidity Sizing (3h)
- [ ] Intégration dans bridge
- [ ] Monitoring dashboards

### Phase 3: Alpha (Semaine 3+)
- [ ] Options Flow (1 jour)
- [ ] Backtesting avec nouvelles features
- [ ] Validation statistique
- [ ] Fine-tuning parameters

---

## 7. Métriques de Succès

| Métrique | Avant | Objectif |
|----------|-------|----------|
| Max Drawdown Daily | -5% | -2% |
| Pertes surprise earnings | Fréquentes | 0 |
| Trades en VIX > 25 | Normal | -70% taille |
| Corrélation portfolio | ~0.85 | < 0.7 effective |
| Win rate | ~55% | ~60% |

---

## 8. Risques et Mitigations

| Risque | Mitigation |
|--------|------------|
| Overfitting aux features | Validation out-of-sample |
| API rate limits (yfinance) | Caching agressif, fallbacks |
| Données earnings incorrectes | Multiple sources, manual override |
| VIX manipulation | Combiner avec d'autres indicateurs |
| Latence features | Calcul async, prefetch |

---

*Document créé le 2026-01-09. À réviser après chaque phase d'implémentation.*
