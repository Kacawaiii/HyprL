"""Crypto Signal Generator.

Generates trading signals for cryptocurrencies using ML or policy logic.
Works 24/7 unlike stock markets.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from pathlib import Path
import joblib
import pandas as pd

from .trader import CryptoConfig, CryptoSignal, CryptoTrader
from .policy import PolicyConfig, simulate_policy, compute_policy_frame

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1h",
    "ret_4h",
    "ret_12h",
    "ret_24h",
    "ret_72h",
    "volatility_12h",
    "volatility_24h",
    "volatility_72h",
    "rsi_6",
    "rsi_14",
    "rsi_21",
    "sma_ratio_12",
    "sma_ratio_24",
    "sma_ratio_72",
    "volume_ratio_12",
    "volume_ratio_24",
    "volume_zscore",
    "atr_14",
    "atr_14_norm",
    "high_low_range",
    "true_range",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_asia_session",
    "is_europe_session",
    "is_us_session",
)

LEGACY_FEATURE_COLUMNS: tuple[str, ...] = (
    "returns_1",
    "returns_4",
    "returns_12",
    "returns_24",
    "volatility_12",
    "volatility_24",
    "sma_ratio_12",
    "sma_ratio_24",
    "sma_ratio_72",
    "rsi_14",
    "rsi_6",
    "volume_ratio_12",
    "volume_ratio_24",
    "atr_14",
    "high_low_range",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
)

LEGACY_FEATURE_ALIASES: dict[str, str] = {
    "returns_1": "ret_1h",
    "returns_4": "ret_4h",
    "returns_12": "ret_12h",
    "returns_24": "ret_24h",
    "volatility_12": "volatility_12h",
    "volatility_24": "volatility_24h",
}

TIMEFRAME_MINUTES: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "1Hour": 60,
    "1Day": 1440,
}


def _bars_per_hour(timeframe: str) -> int:
    minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
    if minutes <= 0:
        return 1
    if minutes >= 60:
        return 1
    return max(1, int(round(60 / minutes)))


@dataclass
class CryptoFeatures:
    """Technical features for crypto trading."""
    # Returns
    ret_1h: float = 0.0
    ret_4h: float = 0.0
    ret_12h: float = 0.0
    ret_24h: float = 0.0
    ret_72h: float = 0.0

    # Volatility
    volatility_12h: float = 0.0
    volatility_24h: float = 0.0
    volatility_72h: float = 0.0

    # RSI
    rsi_6: float = 50.0
    rsi_14: float = 50.0
    rsi_21: float = 50.0

    # SMA Ratios
    sma_ratio_12: float = 1.0
    sma_ratio_24: float = 1.0
    sma_ratio_72: float = 1.0

    # Volume
    volume_ratio_12: float = 1.0
    volume_ratio_24: float = 1.0
    volume_zscore: float = 0.0

    # ATR / Range
    atr_14: float = 0.0
    atr_14_norm: float = 0.0
    high_low_range: float = 0.0
    true_range: float = 0.0

    # Temporal
    hour_of_day: float = 0.0
    day_of_week: float = 0.0
    is_weekend: float = 0.0
    is_asia_session: float = 0.0
    is_europe_session: float = 0.0
    is_us_session: float = 0.0


class CryptoSignalGenerator:
    """Generates ML-based signals for crypto trading."""

    def __init__(self, config: Optional[CryptoConfig] = None, base_dir: str = "."):
        self.config = config or CryptoConfig()
        self.base_dir = Path(base_dir)
        self.trader = CryptoTrader(config, base_dir)
        self._models: dict = {}
        self._scalers: dict = {}
        self._feature_cols: dict = {}
        self._policy_config = self._resolve_policy_config(self.config)

    @staticmethod
    def _resolve_policy_config(config: CryptoConfig) -> PolicyConfig:
        policy_cfg = config.policy_config or PolicyConfig()
        if policy_cfg.timeframe != config.timeframe:
            policy_cfg = replace(policy_cfg, timeframe=config.timeframe)
        return policy_cfg

    def _use_policy(self, symbol: str) -> bool:
        if self.config.signal_mode == "policy":
            return True
        if self.config.signal_mode == "ml":
            return symbol in self.config.policy_symbols
        return False

    def _load_model(self, symbol: str) -> bool:
        """Load model for a symbol."""
        clean_symbol = symbol.replace("/", "_").lower()
        model_path = self.base_dir / self.config.models_dir / f"{clean_symbol}_xgb.joblib"

        if model_path.exists():
            try:
                bundle = joblib.load(model_path)
                self._models[symbol] = bundle.get("model")
                self._scalers[symbol] = bundle.get("scaler")
                feature_cols = bundle.get("feature_columns")
                if isinstance(feature_cols, (list, tuple)) and feature_cols:
                    self._feature_cols[symbol] = list(feature_cols)
                return True
            except Exception as e:
                print(f"Error loading model for {symbol}: {e}")

        return False

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _true_ranges(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate true range series."""
        if len(closes) < 2:
            return np.array([])

        prev_close = closes[:-1]
        high_low = highs[1:] - lows[1:]
        high_close = np.abs(highs[1:] - prev_close)
        low_close = np.abs(lows[1:] - prev_close)
        return np.maximum.reduce([high_low, high_close, low_close])

    def _rolling_std(self, values: np.ndarray, window: int) -> float:
        if values.size < window or window < 2:
            return 0.0
        return float(np.std(values[-window:], ddof=1))

    def extract_features(self, bars: list[dict]) -> Optional[CryptoFeatures]:
        """Extract features from bar data."""
        if not bars:
            return None

        bars_sorted = sorted(bars, key=lambda b: b["timestamp"])
        bars_per_hour = _bars_per_hour(self.config.timeframe)

        def _bars(hours: int) -> int:
            return max(1, int(hours * bars_per_hour))

        min_bars = max(_bars(72), _bars(24), _bars(21), _bars(14)) + 1
        if len(bars_sorted) < min_bars:
            return None

        closes = np.array([b["close"] for b in bars_sorted], dtype=float)
        highs = np.array([b["high"] for b in bars_sorted], dtype=float)
        lows = np.array([b["low"] for b in bars_sorted], dtype=float)
        volumes = np.array([b["volume"] for b in bars_sorted], dtype=float)

        # Current bar timestamp
        now = bars_sorted[-1]["timestamp"]
        if isinstance(now, str):
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))

        # Returns
        def _return(horizon: int) -> float:
            if len(closes) <= horizon:
                return 0.0
            prior = closes[-1 - horizon]
            return float(closes[-1] / prior - 1.0) if prior != 0 else 0.0

        ret_1h = _return(_bars(1))
        ret_4h = _return(_bars(4))
        ret_12h = _return(_bars(12))
        ret_24h = _return(_bars(24))
        ret_72h = _return(_bars(72))

        returns = np.diff(closes) / closes[:-1]
        volatility_12h = self._rolling_std(returns, _bars(12))
        volatility_24h = self._rolling_std(returns, _bars(24))
        volatility_72h = self._rolling_std(returns, _bars(72))

        # SMA ratios
        sma_12_window = _bars(12)
        sma_24_window = _bars(24)
        sma_72_window = _bars(72)
        sma_12 = np.mean(closes[-sma_12_window:]) if len(closes) >= sma_12_window else closes[-1]
        sma_24 = np.mean(closes[-sma_24_window:]) if len(closes) >= sma_24_window else closes[-1]
        sma_72 = np.mean(closes[-sma_72_window:]) if len(closes) >= sma_72_window else closes[-1]

        # Volume ratios
        vol_12_window = _bars(12)
        vol_24_window = _bars(24)
        vol_avg_12 = np.mean(volumes[-vol_12_window:]) if len(volumes) >= vol_12_window else volumes[-1]
        vol_avg_24 = np.mean(volumes[-vol_24_window:]) if len(volumes) >= vol_24_window else volumes[-1]
        volume_zscore = 0.0
        if len(volumes) >= vol_24_window:
            vol_window = volumes[-vol_24_window:]
            vol_std = np.std(vol_window, ddof=1) if len(vol_window) > 1 else 0.0
            if vol_std > 0:
                volume_zscore = float((volumes[-1] - float(np.mean(vol_window))) / vol_std)

        tr_values = self._true_ranges(highs, lows, closes)
        true_range = float(tr_values[-1]) if tr_values.size else 0.0
        atr_window = _bars(14)
        atr_14 = float(tr_values[-atr_window:].mean()) if tr_values.size >= atr_window else 0.0
        atr_14_norm = atr_14 / closes[-1] if closes[-1] > 0 else 0.0
        high_low_range = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0

        hour = now.hour
        day = now.weekday()
        is_weekend = float(day >= 5)
        is_asia_session = float(0 <= hour < 8)
        is_europe_session = float(8 <= hour < 16)
        is_us_session = float(14 <= hour < 22)

        return CryptoFeatures(
            ret_1h=ret_1h,
            ret_4h=ret_4h,
            ret_12h=ret_12h,
            ret_24h=ret_24h,
            ret_72h=ret_72h,
            volatility_12h=volatility_12h,
            volatility_24h=volatility_24h,
            volatility_72h=volatility_72h,
            rsi_6=self._calculate_rsi(closes, max(2, _bars(6))),
            rsi_14=self._calculate_rsi(closes, max(2, _bars(14))),
            rsi_21=self._calculate_rsi(closes, max(2, _bars(21))),
            sma_ratio_12=closes[-1] / sma_12 if sma_12 > 0 else 1.0,
            sma_ratio_24=closes[-1] / sma_24 if sma_24 > 0 else 1.0,
            sma_ratio_72=closes[-1] / sma_72 if sma_72 > 0 else 1.0,
            volume_ratio_12=volumes[-1] / vol_avg_12 if vol_avg_12 > 0 else 1.0,
            volume_ratio_24=volumes[-1] / vol_avg_24 if vol_avg_24 > 0 else 1.0,
            volume_zscore=volume_zscore,
            atr_14=atr_14,
            atr_14_norm=atr_14_norm,
            high_low_range=high_low_range,
            true_range=true_range,
            hour_of_day=hour / 24.0,
            day_of_week=day / 7.0,
            is_weekend=is_weekend,
            is_asia_session=is_asia_session,
            is_europe_session=is_europe_session,
            is_us_session=is_us_session,
        )

    def _features_to_dict(self, features: CryptoFeatures) -> dict[str, float]:
        feature_map = {
            "ret_1h": features.ret_1h,
            "ret_4h": features.ret_4h,
            "ret_12h": features.ret_12h,
            "ret_24h": features.ret_24h,
            "ret_72h": features.ret_72h,
            "volatility_12h": features.volatility_12h,
            "volatility_24h": features.volatility_24h,
            "volatility_72h": features.volatility_72h,
            "rsi_6": features.rsi_6,
            "rsi_14": features.rsi_14,
            "rsi_21": features.rsi_21,
            "sma_ratio_12": features.sma_ratio_12,
            "sma_ratio_24": features.sma_ratio_24,
            "sma_ratio_72": features.sma_ratio_72,
            "volume_ratio_12": features.volume_ratio_12,
            "volume_ratio_24": features.volume_ratio_24,
            "volume_zscore": features.volume_zscore,
            "atr_14": features.atr_14,
            "atr_14_norm": features.atr_14_norm,
            "high_low_range": features.high_low_range,
            "true_range": features.true_range,
            "hour_of_day": features.hour_of_day,
            "day_of_week": features.day_of_week,
            "is_weekend": features.is_weekend,
            "is_asia_session": features.is_asia_session,
            "is_europe_session": features.is_europe_session,
            "is_us_session": features.is_us_session,
        }

        for legacy, current in LEGACY_FEATURE_ALIASES.items():
            feature_map[legacy] = feature_map[current]

        return feature_map

    def features_to_array(self, features: CryptoFeatures, feature_columns: Optional[list[str]] = None) -> np.ndarray:
        """Convert features to numpy array for model."""
        columns = feature_columns or list(FEATURE_COLUMNS)
        feature_map = self._features_to_dict(features)
        missing = [col for col in columns if col not in feature_map]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        values = np.array([feature_map[col] for col in columns], dtype=float).reshape(1, -1)
        if np.isnan(values).any():
            raise ValueError("NaN in feature vector")
        return values

    def predict_probability(self, symbol: str, features: CryptoFeatures) -> Optional[float]:
        """Predict probability using loaded model."""
        if symbol not in self._models:
            if not self._load_model(symbol):
                return None

        model = self._models.get(symbol)
        scaler = self._scalers.get(symbol)
        feature_cols = self._feature_cols.get(symbol)

        if model is None:
            return None

        if feature_cols is None:
            expected = getattr(model, "n_features_in_", None)
            if expected == len(LEGACY_FEATURE_COLUMNS):
                feature_cols = list(LEGACY_FEATURE_COLUMNS)
            else:
                feature_cols = list(FEATURE_COLUMNS)

        try:
            X = self.features_to_array(features, feature_cols)
        except ValueError as e:
            print(f"Feature error for {symbol}: {e}")
            return None

        if scaler:
            try:
                X = scaler.transform(X)
            except Exception:
                pass  # Use unscaled if scaler fails

        try:
            proba = model.predict_proba(X)[0]
            # Return probability of positive class
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")

        return None

    def generate_signal(self, symbol: str) -> Optional[CryptoSignal]:
        """Generate a trading signal for a crypto symbol."""
        # Fetch bars
        bars = self.trader.fetch_bars(symbol)
        if not bars:
            return None

        if self._use_policy(symbol):
            return self._generate_policy_signal(symbol, bars)

        # Extract features
        features = self.extract_features(bars)
        if not features:
            return None

        # Get current price
        current_price = bars[-1]["close"]

        # Predict probability (or use rule-based if no model)
        probability = None if self.config.signal_mode == "rule" else self.predict_probability(symbol, features)

        if probability is None:
            probability = self._rule_based_probability(features)

        # Determine direction
        if probability >= self.config.threshold_long:
            direction = "long"
        elif probability <= self.config.threshold_short:
            direction = "short"
        else:
            direction = "neutral"

        # Mean-reversion overlay: buy oversold, sell/exit overbought
        overlay = None
        oversold = (
            features.rsi_14 <= 35
            and features.sma_ratio_24 <= 0.995
            and features.ret_12h <= -0.01
        )
        overbought = (
            features.rsi_14 >= 65
            and features.sma_ratio_24 >= 1.005
            and features.ret_12h >= 0.01
        )
        aggressive_long = (
            probability >= 0.20
            and features.rsi_14 <= 60
            and features.sma_ratio_24 <= 1.01
            and features.ret_4h <= 0.01
        )
        aggressive_short = (
            probability <= 0.45
            and features.rsi_14 >= 60
            and features.sma_ratio_24 >= 1.01
            and features.ret_4h >= 0.01
        )
        if overbought and direction == "long":
            direction = "flat"
            overlay = "overbought_exit"
        elif overbought and direction == "neutral":
            direction = "short"
            probability = max(0.0, min(probability, 0.35))
            overlay = "overbought_short"
        elif oversold and direction in ("neutral", "short"):
            direction = "long"
            probability = min(1.0, max(probability, self.config.threshold_long + 0.10))
            overlay = "oversold_entry"
        elif aggressive_long and direction == "neutral":
            direction = "long"
            probability = min(1.0, max(probability, 0.62))
            overlay = "aggressive_long"
        elif aggressive_short and direction == "neutral":
            direction = "short"
            probability = max(0.0, min(probability, 0.38))
            overlay = "aggressive_short"

        # Calculate confidence
        if direction == "long":
            confidence = (probability - 0.5) * 2
        elif direction == "short":
            confidence = (0.5 - probability) * 2
        else:
            confidence = 0.0

        # Size based on confidence
        if direction == "neutral":
            size_pct = 0.0
        else:
            base_size = self.config.max_position_pct
            size_pct = base_size * min(1.0, confidence / 0.3)

        # Calculate stop loss and take profit
        atr = features.atr_14
        if direction == "long":
            stop_loss = current_price * (1 - self.config.stop_loss_pct)
            take_profit = current_price * (1 + self.config.take_profit_pct)
        elif direction == "short":
            stop_loss = current_price * (1 + self.config.stop_loss_pct)
            take_profit = current_price * (1 - self.config.take_profit_pct)
        else:
            stop_loss = current_price
            take_profit = current_price

        # Build reason string
        reason_parts = [
            f"prob={probability:.2f}",
            f"rsi={features.rsi_14:.0f}",
            f"sma_ratio={features.sma_ratio_24:.3f}",
            f"vol={features.volatility_24h:.4f}",
        ]
        if overlay:
            reason_parts.append(f"overlay={overlay}")

        return CryptoSignal(
            symbol=symbol,
            direction=direction,
            probability=probability,
            confidence=confidence,
            size_pct=size_pct,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=", ".join(reason_parts),
        )

    def _generate_policy_signal(self, symbol: str, bars: list[dict]) -> Optional[CryptoSignal]:
        frame = pd.DataFrame(bars)
        if frame.empty:
            return None
        policy_frame = compute_policy_frame(frame, self._policy_config)
        if policy_frame.empty:
            return None

        decision, state = simulate_policy(policy_frame, self._policy_config)
        if decision is None:
            return None

        current_price = float(policy_frame["close"].iloc[-1])
        direction = "long" if state.in_position else "neutral"
        if decision.action == "SELL":
            direction = "flat"

        size_pct = decision.size_pct if direction == "long" else 0.0
        stop_loss = decision.stop_loss if direction == "long" else current_price
        take_profit = decision.take_profit if direction == "long" else current_price
        probability = decision.probability
        confidence = decision.confidence if direction == "long" else 0.0

        return CryptoSignal(
            symbol=symbol,
            direction=direction,
            probability=probability,
            confidence=confidence,
            size_pct=size_pct,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=decision.reason,
        )

    def _rule_based_probability(self, features: CryptoFeatures) -> float:
        """Simple rule-based probability when no model available."""
        score = 0.5

        # RSI
        if features.rsi_14 < 30:
            score += 0.15  # Oversold = bullish
        elif features.rsi_14 > 70:
            score -= 0.15  # Overbought = bearish

        # Trend (SMA ratio)
        if features.sma_ratio_24 > 1.02:
            score += 0.1  # Above SMA = bullish
        elif features.sma_ratio_24 < 0.98:
            score -= 0.1  # Below SMA = bearish

        # Short-term momentum
        if features.ret_4h > 0.02:
            score += 0.05
        elif features.ret_4h < -0.02:
            score -= 0.05

        # Volume confirmation
        if features.volume_ratio_12 > 1.5:
            if features.ret_1h > 0:
                score += 0.05  # High volume + up = bullish
            else:
                score -= 0.05  # High volume + down = bearish

        return max(0.0, min(1.0, score))

    def scan_all(self) -> list[CryptoSignal]:
        """Generate signals for all configured symbols."""
        signals = []

        for symbol in self.config.symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)
                self.trader.log_signal(signal)

        return signals


def format_crypto_scan(signals: list[CryptoSignal]) -> str:
    """Format scan results as readable text."""
    if not signals:
        return "No crypto signals generated"

    lines = [
        "=" * 50,
        f"Crypto Scan - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        "=" * 50,
        "",
    ]

    # Group by direction
    longs = [s for s in signals if s.direction == "long"]
    shorts = [s for s in signals if s.direction == "short"]
    neutrals = [s for s in signals if s.direction in ("neutral", "flat")]

    if longs:
        lines.append("LONG SIGNALS:")
        for s in sorted(longs, key=lambda x: x.probability, reverse=True):
            lines.append(f"  {s.symbol:8} prob={s.probability:.2f} size={s.size_pct:.1%} @ ${s.entry_price:,.2f}")
        lines.append("")

    if shorts:
        lines.append("SHORT SIGNALS:")
        for s in sorted(shorts, key=lambda x: x.probability):
            lines.append(f"  {s.symbol:8} prob={s.probability:.2f} size={s.size_pct:.1%} @ ${s.entry_price:,.2f}")
        lines.append("")

    if neutrals:
        lines.append("NEUTRAL/FLAT:")
        for s in neutrals:
            reason = f" reason={s.reason}" if s.reason else ""
            lines.append(
                f"  {s.symbol:8} {s.direction:6} prob={s.probability:.2f} @ ${s.entry_price:,.2f}{reason}"
            )

    return "\n".join(lines)
