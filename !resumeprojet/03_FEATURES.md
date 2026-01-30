# Feature Engineering HyprL

## Vue d'ensemble

Le feature engineering transforme les données brutes OHLCV en features prédictives pour les modèles ML.

## Pipeline de features

```
OHLCV Data → Returns → Technical Indicators → Normalization → Final Features
```

## Fichiers principaux

| Fichier | Rôle |
|---------|------|
| `src/hyprl/features/enhanced_v3.py` | Feature engine V3 |
| `src/hyprl/indicators/technical.py` | Indicateurs techniques |

## Features complètes

### 1. Returns (Momentum)

```python
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les returns sur différentes périodes."""
    df['ret_1h'] = df['close'].pct_change(1)
    df['ret_3h'] = df['close'].pct_change(3)
    df['ret_6h'] = df['close'].pct_change(6)
    df['ret_12h'] = df['close'].pct_change(12)
    df['ret_24h'] = df['close'].pct_change(24)
    df['ret_5d'] = df['close'].pct_change(5 * 24)  # 5 jours
    return df
```

### 2. ATR (Average True Range)

```python
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR pour mesurer la volatilité."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

# Features ATR
df['atr_14'] = compute_atr(df, 14)
df['atr_72'] = compute_atr(df, 72)
df['atr_14_norm'] = df['atr_14'] / df['close']  # Normalisé par prix
df['atr_72_norm'] = df['atr_72'] / df['close']
```

### 3. RSI (Relative Strength Index)

```python
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI pour mesurer le momentum."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = compute_rsi(df, 14)
df['rsi_7'] = compute_rsi(df, 7)
```

### 4. MACD

```python
def compute_macd(df: pd.DataFrame) -> tuple:
    """MACD pour détecter les tendances."""
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist

df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df)
```

### 5. Bollinger Bands

```python
def compute_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0):
    """Bollinger Bands pour la volatilité."""
    sma = df['close'].rolling(period).mean()
    rolling_std = df['close'].rolling(period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    width = (upper - lower) / sma
    return upper, lower, width

df['bb_upper'], df['bb_lower'], df['bb_width'] = compute_bollinger(df)
```

### 6. Volume features

```python
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features basées sur le volume."""
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_change'] = df['volume'].pct_change()
    return df
```

### 7. Price position

```python
def compute_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Position du prix dans sa range."""
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
    return df
```

## Liste complète des features V3

```python
FEATURE_COLUMNS_V3 = [
    # Returns
    'ret_1h',
    'ret_3h',
    'ret_6h',
    'ret_24h',

    # ATR
    'atr_14',
    'atr_72',
    'atr_14_norm',
    'atr_72_norm',

    # RSI
    'rsi_14',

    # MACD
    'macd',
    'macd_signal',
    'macd_hist',

    # Bollinger
    'bb_width',

    # Volume
    'volume_ratio',
]
```

## Normalisation

```python
def normalize_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Normalise les features pour le ML."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
```

## Feature importance (NVDA V3)

| Feature | Importance |
|---------|------------|
| ret_1h | 0.23 |
| ret_24h | 0.18 |
| atr_14_norm | 0.15 |
| rsi_14 | 0.12 |
| macd_hist | 0.10 |
| volume_ratio | 0.08 |
| bb_width | 0.07 |
| ret_3h | 0.04 |
| ret_6h | 0.03 |

## Code source

### Chemin: `src/hyprl/features/enhanced_v3.py`

```python
class FeatureEngineV3:
    """Feature engine pour les modèles V3."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_columns = config.get('feature_columns', FEATURE_COLUMNS_V3)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule toutes les features."""
        df = compute_returns(df)
        df = compute_technical_indicators(df)
        df = compute_volume_features(df)
        df = df.dropna()
        return df[self.feature_columns]
```

## Utilisation

```python
from src.hyprl.features.enhanced_v3 import FeatureEngineV3

# Créer l'engine
engine = FeatureEngineV3(config)

# Calculer les features
features_df = engine.compute(ohlcv_df)

# Prédire
probability = model.predict(features_df.iloc[[-1]])
```
