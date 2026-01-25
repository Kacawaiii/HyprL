#!/usr/bin/env python3
"""
HyprL Multi-Account Signal Generator v2.0
==========================================
Version optimisée avec:
1. Paramètres de risque dynamiques (Kelly, volatilité)
2. Indicateurs améliorés (ADX, Stoch RSI, VWAP, multi-timeframe)
3. Stops/TP optimisés (trailing, partial profits, time-based)

Usage:
    python scripts/run_multi_account_v2.py
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment
from dotenv import load_dotenv
CONFIG_DIR = Path(__file__).parent.parent / "configs" / "runtime"
load_dotenv(CONFIG_DIR / ".env.discord")
load_dotenv(Path(__file__).parent.parent / ".env.ops")

import pytz
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from hyprl.monitoring.discord_notifier import (
    HyprLDiscordNotifier,
    PaperAccount,
    get_discord_notifier,
)

# Setup logging
LOG_DIR = Path.home() / ".hyprl" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "multi_account_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HyprL.v2")

# Constants
NY_TZ = pytz.timezone("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Symbols
SYMBOLS = [
    "NVDA", "META", "MSFT", "GOOGL", "NFLX",
    "AMD", "GS", "V", "QQQ", "TSLA",
    "LLY", "PLTR", "SHOP", "SMCI", "UBER", "RBLX"
]

# =============================================================================
# 1. PARAMÈTRES DE RISQUE OPTIMISÉS
# =============================================================================

@dataclass
class RiskConfig:
    """Configuration de risque dynamique."""
    # Base risk
    base_risk_per_trade: float = 0.015  # 1.5% base

    # Kelly criterion bounds
    kelly_fraction: float = 0.25  # Use 25% of Kelly
    max_kelly_risk: float = 0.03  # Cap at 3%
    min_kelly_risk: float = 0.005  # Floor at 0.5%

    # Volatility adjustment
    vol_scale_factor: float = 1.0  # Reduce risk in high vol

    # Win rate tracking
    lookback_trades: int = 20

    # Position limits
    max_positions: int = 6
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.40


@dataclass
class AccountConfigV2:
    """Configuration compte v2."""
    name: str
    paper_account: PaperAccount
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"

    # Risk profile
    risk_profile: str = "normal"  # normal, aggressive, conservative
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Signal thresholds
    min_signal_score: int = 4  # Minimum score to trade
    min_confidence: float = 0.55

    # Trade history for Kelly
    trade_history: List[dict] = field(default_factory=list)


# Account configurations
def get_accounts() -> Dict[PaperAccount, AccountConfigV2]:
    """Load account configurations."""

    def load_keys(env_file: str) -> Tuple[str, str]:
        env_path = CONFIG_DIR / env_file
        if not env_path.exists():
            return "", ""
        env_vars = {}
        for line in env_path.read_text().split("\n"):
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
        return env_vars.get("APCA_API_KEY_ID", ""), env_vars.get("APCA_API_SECRET_KEY", "")

    accounts = {}

    # Normal - Conservative Kelly
    key, secret = load_keys(".env.normal")
    if key and secret:
        accounts[PaperAccount.NORMAL] = AccountConfigV2(
            name="Normal",
            paper_account=PaperAccount.NORMAL,
            api_key=key,
            api_secret=secret,
            risk_profile="normal",
            risk_config=RiskConfig(
                base_risk_per_trade=0.015,
                kelly_fraction=0.20,
                max_positions=5,
            ),
            min_signal_score=5,
            min_confidence=0.58,
        )

    # Aggressive - Higher Kelly
    key, secret = load_keys(".env.aggressive")
    if key and secret:
        accounts[PaperAccount.AGGRESSIVE] = AccountConfigV2(
            name="Aggressive",
            paper_account=PaperAccount.AGGRESSIVE,
            api_key=key,
            api_secret=secret,
            risk_profile="aggressive",
            risk_config=RiskConfig(
                base_risk_per_trade=0.025,
                kelly_fraction=0.35,
                max_kelly_risk=0.05,
                max_positions=8,
            ),
            min_signal_score=3,
            min_confidence=0.50,
        )

    # Mix - Balanced
    key, secret = load_keys(".env.mix")
    if key and secret:
        accounts[PaperAccount.MIX] = AccountConfigV2(
            name="Mix",
            paper_account=PaperAccount.MIX,
            api_key=key,
            api_secret=secret,
            risk_profile="balanced",
            risk_config=RiskConfig(
                base_risk_per_trade=0.018,
                kelly_fraction=0.25,
                max_positions=6,
            ),
            min_signal_score=4,
            min_confidence=0.55,
        )

    return accounts


def calculate_kelly_risk(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion optimal bet size.

    Kelly % = W - [(1-W) / R]
    Where:
        W = Win rate
        R = Win/Loss ratio
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.01

    win_loss_ratio = abs(avg_win / avg_loss)
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Apply fraction (never bet full Kelly)
    kelly_adjusted = kelly * fraction

    return max(0.005, min(0.05, kelly_adjusted))


def calculate_dynamic_risk(
    config: AccountConfigV2,
    current_volatility: float,
    avg_volatility: float,
) -> float:
    """
    Calculate dynamic risk based on:
    - Kelly criterion from trade history
    - Current vs average volatility
    """
    risk = config.risk_config.base_risk_per_trade

    # Kelly adjustment if we have trade history
    if len(config.trade_history) >= 10:
        wins = [t for t in config.trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in config.trade_history if t.get("pnl", 0) <= 0]

        if wins and losses:
            win_rate = len(wins) / len(config.trade_history)
            avg_win = np.mean([t["pnl"] for t in wins])
            avg_loss = np.mean([abs(t["pnl"]) for t in losses])

            kelly_risk = calculate_kelly_risk(
                win_rate, avg_win, avg_loss,
                config.risk_config.kelly_fraction
            )

            # Blend base risk with Kelly
            risk = 0.5 * risk + 0.5 * kelly_risk

    # Volatility adjustment - reduce risk in high vol
    if avg_volatility > 0:
        vol_ratio = current_volatility / avg_volatility
        if vol_ratio > 1.5:  # High volatility
            risk *= 0.7
        elif vol_ratio < 0.7:  # Low volatility
            risk *= 1.2

    # Apply bounds
    risk = max(config.risk_config.min_kelly_risk,
               min(config.risk_config.max_kelly_risk, risk))

    return risk


# =============================================================================
# 2. INDICATEURS AMÉLIORÉS
# =============================================================================

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength)."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def calculate_stoch_rsi(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic RSI."""
    # First calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Then apply Stochastic formula to RSI
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()

    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    stoch_rsi_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(smooth_d).mean()

    return stoch_rsi_k, stoch_rsi_d


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap


def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate price momentum."""
    return df['Close'].pct_change(period) * 100


def calculate_indicators_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators (enhanced version)."""

    # === Original indicators ===
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close'] * 100

    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # Volume
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']

    # === New indicators ===
    # ADX (trend strength)
    df['adx'] = calculate_adx(df)

    # Stochastic RSI
    df['stoch_rsi_k'], df['stoch_rsi_d'] = calculate_stoch_rsi(df)

    # VWAP
    df['vwap'] = calculate_vwap(df)

    # Momentum
    df['momentum_10'] = calculate_momentum(df, 10)
    df['momentum_20'] = calculate_momentum(df, 20)

    # Price position relative to range
    df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                           (df['High'].rolling(20).max() - df['Low'].rolling(20).min())

    # Trend alignment
    df['trend_aligned'] = (
        (df['ema_9'] > df['ema_21']) &
        (df['ema_21'] > df['sma_50']) &
        (df['Close'] > df['vwap'])
    ).astype(int)

    # Volatility regime
    df['vol_regime'] = df['atr_pct'].rolling(20).mean()
    df['vol_expanding'] = df['atr_pct'] > df['vol_regime']

    return df


# =============================================================================
# 3. STOPS/TP OPTIMISÉS
# =============================================================================

@dataclass
class StopConfig:
    """Configuration des stops optimisés."""
    # ATR multipliers
    stop_atr_mult: float = 1.5  # Tighter stop
    tp1_atr_mult: float = 2.5   # First target (partial) - better R:R
    tp2_atr_mult: float = 4.0   # Second target (full)

    # Partial profit taking
    partial_exit_pct: float = 0.50  # Exit 50% at TP1

    # Trailing stop
    use_trailing: bool = True
    trail_activation_r: float = 1.0  # Activate after 1R profit
    trail_distance_atr: float = 1.2  # Trail distance in ATR

    # Time-based exit
    max_hold_hours: int = 48  # Close if no movement after 48h

    # Volatility adjustment
    vol_stop_adjust: bool = True  # Widen stops in high vol


def calculate_optimal_stops(
    price: float,
    atr: float,
    direction: str,
    volatility_ratio: float = 1.0,
    config: StopConfig = None,
) -> Tuple[float, float, float]:
    """
    Calculate optimal stop loss and take profit levels.

    Returns:
        (stop_price, tp1_price, tp2_price)
    """
    if config is None:
        config = StopConfig()

    # Adjust ATR multipliers based on volatility
    stop_mult = config.stop_atr_mult
    tp1_mult = config.tp1_atr_mult
    tp2_mult = config.tp2_atr_mult

    if config.vol_stop_adjust:
        if volatility_ratio > 1.3:  # High volatility
            stop_mult *= 1.2  # Wider stop
            tp1_mult *= 0.9   # Closer TP1
        elif volatility_ratio < 0.7:  # Low volatility
            stop_mult *= 0.85  # Tighter stop
            tp2_mult *= 1.2    # Further TP2

    if direction == "long":
        stop = price - (atr * stop_mult)
        tp1 = price + (atr * tp1_mult)
        tp2 = price + (atr * tp2_mult)
    else:  # short
        stop = price + (atr * stop_mult)
        tp1 = price - (atr * tp1_mult)
        tp2 = price - (atr * tp2_mult)

    return round(stop, 2), round(tp1, 2), round(tp2, 2)


def calculate_rr_ratio(entry: float, stop: float, target: float) -> float:
    """Calculate risk/reward ratio."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return reward / risk if risk > 0 else 0


# =============================================================================
# SIGNAL GENERATION V2
# =============================================================================

@dataclass
class SignalV2:
    """Enhanced trading signal."""
    symbol: str
    direction: str
    score: int
    confidence: float
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    atr: float
    volatility_ratio: float
    reasons: List[str]
    indicators: Dict[str, float]


def detect_market_regime(df: pd.DataFrame, i: int) -> str:
    """
    Detect market regime at index i.

    Returns: "trending", "ranging", "volatile", or "neutral"
    """
    if i < 50:
        return "neutral"

    adx = df.iloc[i]['adx']
    atr_pct = df.iloc[i]['atr_pct']
    avg_atr = df['atr_pct'].iloc[i-50:i].mean()
    vol_ratio = atr_pct / avg_atr if avg_atr > 0 else 1
    bb_width = df.iloc[i]['bb_width']
    avg_bb = df['bb_width'].iloc[i-50:i].mean()

    if vol_ratio > 1.5:
        return "volatile"
    elif adx > 30 and vol_ratio < 1.3:
        return "trending"
    elif adx < 20 and bb_width < avg_bb:
        return "ranging"
    else:
        return "neutral"


def generate_signal_v2(symbol: str) -> Optional[SignalV2]:
    """Generate enhanced trading signal."""
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval="1h")

        if len(df) < 50:
            return None

        # Calculate indicators
        df = calculate_indicators_v2(df)

        # Check regime - skip volatile markets
        regime = detect_market_regime(df, len(df) - 1)
        if regime == "volatile":
            logger.debug(f"{symbol}: Skipping - volatile regime")
            return None

        # Get values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = latest['Close']
        atr = latest['atr']

        # Volatility ratio
        avg_vol = df['atr_pct'].rolling(50).mean().iloc[-1]
        vol_ratio = latest['atr_pct'] / avg_vol if avg_vol > 0 else 1.0

        # Signal scoring
        score = 0
        reasons = []

        # === RSI (weight: 2) ===
        if latest['rsi'] < 30:
            score += 3
            reasons.append("RSI <30 (oversold)")
        elif latest['rsi'] < 40:
            score += 1
            reasons.append("RSI bullish zone")
        elif latest['rsi'] > 70:
            score -= 3
            reasons.append("RSI >70 (overbought)")
        elif latest['rsi'] > 60:
            score -= 1
            reasons.append("RSI bearish zone")

        # === Stochastic RSI (weight: 2) ===
        if latest['stoch_rsi_k'] < 20 and latest['stoch_rsi_k'] > latest['stoch_rsi_d']:
            score += 2
            reasons.append("StochRSI bullish cross from oversold")
        elif latest['stoch_rsi_k'] > 80 and latest['stoch_rsi_k'] < latest['stoch_rsi_d']:
            score -= 2
            reasons.append("StochRSI bearish cross from overbought")

        # === MACD (weight: 3) ===
        if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            score += 3
            reasons.append("MACD bullish crossover")
        elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            score -= 3
            reasons.append("MACD bearish crossover")

        # MACD histogram momentum
        if latest['macd_hist'] > 0 and latest['macd_hist'] > prev['macd_hist']:
            score += 1
            reasons.append("MACD histogram expanding")
        elif latest['macd_hist'] < 0 and latest['macd_hist'] < prev['macd_hist']:
            score -= 1
            reasons.append("MACD histogram contracting")

        # === Bollinger Bands (weight: 2) ===
        if price < latest['bb_lower']:
            score += 2
            reasons.append("Below BB lower")
        elif price > latest['bb_upper']:
            score -= 2
            reasons.append("Above BB upper")

        # === Trend (weight: 2) ===
        if latest['trend_aligned'] == 1:
            score += 2
            reasons.append("Full trend alignment (EMA9>21>SMA50, above VWAP)")
        elif latest['ema_9'] < latest['ema_21'] < latest['sma_50']:
            score -= 2
            reasons.append("Bearish trend alignment")

        # === ADX - Trend strength (weight: 1-2) ===
        if latest['adx'] > 25:
            # Strong trend - go with momentum
            if latest['momentum_10'] > 5:
                score += 2
                reasons.append(f"Strong uptrend (ADX={latest['adx']:.0f})")
            elif latest['momentum_10'] < -5:
                score -= 2
                reasons.append(f"Strong downtrend (ADX={latest['adx']:.0f})")
        elif latest['adx'] < 20:
            # Weak trend - mean reversion signals
            if latest['rsi'] < 35:
                score += 1
                reasons.append("Mean reversion setup (low ADX)")
            elif latest['rsi'] > 65:
                score -= 1

        # === Volume confirmation (multiplier) ===
        if latest['volume_ratio'] > 1.5:
            score = int(score * 1.3)
            reasons.append(f"High volume ({latest['volume_ratio']:.1f}x)")
        elif latest['volume_ratio'] < 0.5:
            score = int(score * 0.7)
            reasons.append("Low volume warning")

        # === VWAP (weight: 1) ===
        if price > latest['vwap'] and score > 0:
            score += 1
            reasons.append("Above VWAP")
        elif price < latest['vwap'] and score < 0:
            score -= 1
            reasons.append("Below VWAP")

        # === Momentum confirmation ===
        if latest['momentum_10'] > 3 and score > 0:
            score += 1
        elif latest['momentum_10'] < -3 and score < 0:
            score -= 1

        # === Strong mean-reversion bonus ===
        signal_type = "mixed"
        if latest['rsi'] < 25 and latest['stoch_rsi_k'] < 10:
            score += 3
            reasons.append("Extreme oversold (mean reversion)")
            signal_type = "mean_reversion"
        elif latest['rsi'] > 75 and latest['stoch_rsi_k'] > 90:
            score -= 3
            reasons.append("Extreme overbought (mean reversion)")
            signal_type = "mean_reversion"

        # Detect signal type from main contributors
        if "MACD" in str(reasons) or "momentum" in str(reasons).lower():
            signal_type = "momentum"
        elif "oversold" in str(reasons).lower() or "overbought" in str(reasons).lower():
            signal_type = "mean_reversion"

        # === Regime + Signal alignment ===
        if regime == "trending" and signal_type == "mean_reversion":
            score = int(score * 0.7)  # Reduce mean reversion signals in trends
            reasons.append("(regime penalty)")
        elif regime == "ranging" and signal_type == "momentum":
            score = int(score * 0.7)  # Reduce momentum signals in ranges
            reasons.append("(regime penalty)")
        elif regime == "trending" and signal_type == "momentum":
            score = int(score * 1.2)  # Boost momentum in trends
            reasons.append("(regime bonus)")

        # Minimum threshold
        if abs(score) < 3:
            return None

        # Direction and confidence
        direction = "long" if score > 0 else "short"
        confidence = min(0.45 + abs(score) * 0.05, 0.85)

        # Calculate stops
        stop_price, tp1_price, tp2_price = calculate_optimal_stops(
            price, atr, direction, vol_ratio
        )

        # R:R check - minimum 1.3:1 (relaxed)
        rr = calculate_rr_ratio(price, stop_price, tp1_price)
        if rr < 1.3:
            return None

        return SignalV2(
            symbol=symbol,
            direction=direction,
            score=abs(score),
            confidence=confidence,
            entry_price=price,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            atr=atr,
            volatility_ratio=vol_ratio,
            reasons=reasons,
            indicators={
                "rsi": latest['rsi'],
                "adx": latest['adx'],
                "macd_hist": latest['macd_hist'],
                "volume_ratio": latest['volume_ratio'],
                "momentum": latest['momentum_10'],
            }
        )

    except Exception as e:
        logger.error(f"{symbol}: Signal error - {e}")
        return None


# =============================================================================
# EXECUTION
# =============================================================================

def is_market_open() -> bool:
    """Check if market is open."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


async def execute_signal_v2(
    api: tradeapi.REST,
    config: AccountConfigV2,
    signal: SignalV2,
    discord: HyprLDiscordNotifier,
) -> bool:
    """Execute signal with v2 logic."""

    # Check thresholds
    if signal.score < config.min_signal_score:
        return False
    if signal.confidence < config.min_confidence:
        return False

    try:
        # Get account
        account = api.get_account()
        equity = float(account.equity)

        # Check positions
        positions = api.list_positions()
        if len(positions) >= config.risk_config.max_positions:
            return False

        # Already in position?
        for p in positions:
            if p.symbol == signal.symbol:
                return False

        # Calculate dynamic risk
        avg_vol = 1.5  # Would calculate from historical data
        risk_pct = calculate_dynamic_risk(config, signal.volatility_ratio, avg_vol)

        # Position sizing
        risk_amount = equity * risk_pct
        risk_per_share = abs(signal.entry_price - signal.stop_price)

        if risk_per_share <= 0:
            return False

        shares = int(risk_amount / risk_per_share)

        # Limits
        position_value = shares * signal.entry_price
        max_position = equity * 0.15
        if position_value > max_position:
            shares = int(max_position / signal.entry_price)

        if shares < 1:
            return False

        # Submit bracket order
        side = "buy" if signal.direction == "long" else "sell"

        order = api.submit_order(
            symbol=signal.symbol,
            qty=shares,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            stop_loss={"stop_price": str(signal.stop_price)},
            take_profit={"limit_price": str(signal.tp1_price)},  # Use TP1
        )

        rr = calculate_rr_ratio(signal.entry_price, signal.stop_price, signal.tp1_price)

        logger.info(
            f"{config.name}: {signal.direction.upper()} {signal.symbol} "
            f"x{shares} @ ${signal.entry_price:.2f} | "
            f"Risk: {risk_pct:.1%} | R:R {rr:.1f} | Score: {signal.score}"
        )

        # Discord
        await discord.send_trade_entry(
            account=config.paper_account,
            symbol=signal.symbol,
            direction=signal.direction,
            shares=shares,
            price=signal.entry_price,
            stop_price=signal.stop_price,
            tp_price=signal.tp1_price,
            confidence=signal.confidence,
        )

        return True

    except Exception as e:
        logger.error(f"{config.name}: Order failed - {e}")
        return False


async def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("HyprL Multi-Account v2.0 - Signal Generator")
    logger.info("=" * 60)

    if not is_market_open():
        logger.info("Market closed")
        return

    logger.info("Market OPEN - generating signals...")

    # Init
    discord = get_discord_notifier()
    accounts = get_accounts()

    # Connect APIs
    apis = {}
    for acc_type, config in accounts.items():
        try:
            api = tradeapi.REST(config.api_key, config.api_secret, config.base_url)
            api.get_account()
            apis[acc_type] = (api, config)
            logger.info(f"✓ {config.name} connected")
        except Exception as e:
            logger.error(f"✗ {config.name}: {e}")

    if not apis:
        logger.error("No accounts connected")
        return

    # Generate signals
    logger.info(f"\nAnalyzing {len(SYMBOLS)} symbols...")
    signals: List[SignalV2] = []

    for symbol in SYMBOLS:
        signal = generate_signal_v2(symbol)
        if signal:
            signals.append(signal)
            logger.info(
                f"  {symbol}: {signal.direction.upper()} "
                f"(score={signal.score}, conf={signal.confidence:.0%}) | "
                f"{', '.join(signal.reasons[:3])}"
            )

    logger.info(f"\n{len(signals)} signals generated")

    # Sort by score
    signals.sort(key=lambda s: s.score, reverse=True)

    # Execute
    if signals:
        total = 0
        for signal in signals:
            for acc_type, (api, config) in apis.items():
                if await execute_signal_v2(api, config, signal, discord):
                    total += 1
        logger.info(f"\n{total} trades executed")

    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
