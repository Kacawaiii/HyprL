"""Sentiment-Based Trading Filter.

Filters/adjusts trading signals based on news sentiment.
Uses multiple sources: Alpaca News, Yahoo Finance, Fear & Greed Index.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import os


class SentimentLevel(Enum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class NewsItem:
    """A single news item."""
    headline: str
    source: str
    timestamp: datetime
    symbols: list[str]
    sentiment_score: float  # -1 to 1
    relevance: float  # 0 to 1
    url: Optional[str] = None


@dataclass
class SentimentResult:
    """Sentiment analysis result for a symbol."""
    symbol: str
    level: SentimentLevel
    score: float  # -1 (very bearish) to 1 (very bullish)
    news_count: int
    avg_sentiment: float
    recent_headlines: list[str]
    fear_greed_index: Optional[int] = None
    signal_modifier: float = 1.0
    should_trade: bool = True
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SentimentConfig:
    """Sentiment filter configuration."""
    # Thresholds for classification
    very_bearish_threshold: float = -0.5
    bearish_threshold: float = -0.2
    bullish_threshold: float = 0.2
    very_bullish_threshold: float = 0.5

    # Signal modifiers per sentiment level
    very_bearish_mult: float = 0.0  # Block trade
    bearish_mult: float = 0.5
    neutral_mult: float = 1.0
    bullish_mult: float = 1.2
    very_bullish_mult: float = 1.5

    # Filtering options
    min_news_for_signal: int = 2
    news_lookback_hours: int = 24
    block_on_very_bearish: bool = True
    reduce_on_sentiment_conflict: bool = True

    # Keywords
    bullish_keywords: list[str] = field(default_factory=lambda: [
        "beat", "beats", "exceeds", "surge", "soar", "rally", "upgrade",
        "outperform", "buy", "bullish", "growth", "profit", "gain",
        "positive", "strong", "record", "high", "breakthrough", "innovation",
    ])
    bearish_keywords: list[str] = field(default_factory=lambda: [
        "miss", "misses", "below", "plunge", "crash", "selloff", "downgrade",
        "underperform", "sell", "bearish", "decline", "loss", "drop",
        "negative", "weak", "low", "warning", "concern", "fear", "risk",
        "lawsuit", "investigation", "recall", "layoff", "cut", "fraud",
    ])


class TradingSentimentFilter:
    """Filters trading signals based on sentiment."""

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._cache: dict[str, tuple[SentimentResult, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)
        self._vader = None

    def _get_vader(self):
        """Lazy load VADER sentiment analyzer."""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                pass
        return self._vader

    def _score_headline(self, headline: str) -> float:
        """Score a headline for sentiment."""
        # Try VADER first
        vader = self._get_vader()
        if vader:
            return vader.polarity_scores(headline)["compound"]

        # Fallback to keyword matching
        headline_lower = headline.lower()
        score = 0.0

        for word in self.config.bullish_keywords:
            if word in headline_lower:
                score += 0.15

        for word in self.config.bearish_keywords:
            if word in headline_lower:
                score -= 0.15

        return max(-1.0, min(1.0, score))

    def _fetch_alpaca_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news from Alpaca."""
        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest

            api_key = os.environ.get("APCA_API_KEY_ID")
            secret_key = os.environ.get("APCA_API_SECRET_KEY")

            if not api_key or not secret_key:
                return []

            client = NewsClient(api_key=api_key, secret_key=secret_key)
            request = NewsRequest(symbols=symbol, limit=limit)
            news = client.get_news(request)

            items = []
            for article in news.news:
                sentiment = self._score_headline(article.headline)
                items.append(NewsItem(
                    headline=article.headline,
                    source=article.source,
                    timestamp=article.created_at,
                    symbols=list(article.symbols),
                    sentiment_score=sentiment,
                    relevance=1.0 if symbol in article.symbols else 0.5,
                    url=article.url,
                ))
            return items
        except Exception as e:
            return []

    def _fetch_yahoo_news(self, symbol: str) -> list[NewsItem]:
        """Fetch news from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            items = []
            for article in news[:15]:
                headline = article.get("title", "")
                sentiment = self._score_headline(headline)

                pub_time = article.get("providerPublishTime", 0)
                timestamp = datetime.fromtimestamp(pub_time, tz=timezone.utc) if pub_time else datetime.now(timezone.utc)

                items.append(NewsItem(
                    headline=headline,
                    source=article.get("publisher", "Yahoo"),
                    timestamp=timestamp,
                    symbols=[symbol],
                    sentiment_score=sentiment,
                    relevance=1.0,
                    url=article.get("link"),
                ))
            return items
        except Exception:
            return []

    def _classify_sentiment(self, score: float) -> SentimentLevel:
        """Classify sentiment score."""
        if score <= self.config.very_bearish_threshold:
            return SentimentLevel.VERY_BEARISH
        elif score <= self.config.bearish_threshold:
            return SentimentLevel.BEARISH
        elif score >= self.config.very_bullish_threshold:
            return SentimentLevel.VERY_BULLISH
        elif score >= self.config.bullish_threshold:
            return SentimentLevel.BULLISH
        return SentimentLevel.NEUTRAL

    def _get_signal_modifier(self, level: SentimentLevel) -> float:
        """Get signal modifier for sentiment level."""
        return {
            SentimentLevel.VERY_BEARISH: self.config.very_bearish_mult,
            SentimentLevel.BEARISH: self.config.bearish_mult,
            SentimentLevel.NEUTRAL: self.config.neutral_mult,
            SentimentLevel.BULLISH: self.config.bullish_mult,
            SentimentLevel.VERY_BULLISH: self.config.very_bullish_mult,
        }.get(level, 1.0)

    def analyze(self, symbol: str, force_refresh: bool = False) -> SentimentResult:
        """Analyze sentiment for a symbol."""
        # Check cache
        if not force_refresh and symbol in self._cache:
            cached, cache_time = self._cache[symbol]
            if datetime.now(timezone.utc) - cache_time < self._cache_ttl:
                return cached

        # Fetch news
        all_news = []
        all_news.extend(self._fetch_alpaca_news(symbol))
        all_news.extend(self._fetch_yahoo_news(symbol))

        # Filter by recency
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.news_lookback_hours)
        recent_news = [n for n in all_news if n.timestamp >= cutoff]

        # Calculate average sentiment
        if recent_news:
            weighted_sum = sum(n.sentiment_score * n.relevance for n in recent_news)
            total_weight = sum(n.relevance for n in recent_news)
            avg_sentiment = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            avg_sentiment = 0

        level = self._classify_sentiment(avg_sentiment)
        modifier = self._get_signal_modifier(level)

        should_trade = True
        reason = ""

        if level == SentimentLevel.VERY_BEARISH and self.config.block_on_very_bearish:
            should_trade = False
            reason = "Blocked: very bearish sentiment"

        headlines = [n.headline for n in sorted(recent_news, key=lambda x: x.timestamp, reverse=True)[:5]]

        result = SentimentResult(
            symbol=symbol,
            level=level,
            score=avg_sentiment,
            news_count=len(recent_news),
            avg_sentiment=avg_sentiment,
            recent_headlines=headlines,
            signal_modifier=modifier,
            should_trade=should_trade,
            reason=reason,
        )

        self._cache[symbol] = (result, datetime.now(timezone.utc))
        return result

    def filter_signal(
        self,
        symbol: str,
        signal_direction: str,
        base_size: float,
        base_probability: float,
    ) -> tuple[float, float, bool, str]:
        """Filter/adjust signal based on sentiment.

        Args:
            symbol: Stock symbol
            signal_direction: "long" or "short"
            base_size: Original position size
            base_probability: Model probability

        Returns:
            (adjusted_size, adjusted_probability, should_trade, reason)
        """
        sentiment = self.analyze(symbol)

        if not sentiment.should_trade:
            return 0, base_probability, False, sentiment.reason

        adjusted_size = base_size * sentiment.signal_modifier

        # Reduce size if sentiment conflicts with signal direction
        if self.config.reduce_on_sentiment_conflict:
            if signal_direction == "long" and sentiment.level in (SentimentLevel.VERY_BEARISH, SentimentLevel.BEARISH):
                adjusted_size *= 0.5
                reason = f"Long signal reduced: {sentiment.level.value} sentiment"
            elif signal_direction == "short" and sentiment.level in (SentimentLevel.VERY_BULLISH, SentimentLevel.BULLISH):
                adjusted_size *= 0.5
                reason = f"Short signal reduced: {sentiment.level.value} sentiment"
            else:
                reason = f"Sentiment: {sentiment.level.value} ({sentiment.score:.2f})"
        else:
            reason = f"Sentiment: {sentiment.level.value} ({sentiment.score:.2f})"

        return adjusted_size, base_probability, True, reason


def format_sentiment(result: SentimentResult) -> str:
    """Format sentiment result."""
    emoji = {
        SentimentLevel.VERY_BEARISH: "🔴",
        SentimentLevel.BEARISH: "🟠",
        SentimentLevel.NEUTRAL: "⚪",
        SentimentLevel.BULLISH: "🟢",
        SentimentLevel.VERY_BULLISH: "💚",
    }

    lines = [
        f"{emoji.get(result.level, '?')} {result.symbol} - {result.level.value.upper()}",
        f"   Score: {result.score:.2f} | News: {result.news_count} | Modifier: {result.signal_modifier:.1f}x",
    ]

    if result.recent_headlines:
        lines.append("   Headlines:")
        for h in result.recent_headlines[:3]:
            lines.append(f"     • {h[:55]}...")

    return "\n".join(lines)
