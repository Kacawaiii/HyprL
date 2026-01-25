"""
Multi-Source Sentiment Analyzer for HyprL
Scrapes sentiment from multiple sources and aggregates with reliability scores.
"""

import requests
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class SentimentSignal:
    """Single sentiment signal from a source."""
    source: str
    symbol: str
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    volume: int  # number of mentions
    timestamp: datetime
    reliability: float  # historical accuracy of this source
    raw_data: Optional[dict] = None


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from all sources."""
    symbol: str
    final_score: float  # weighted average
    final_confidence: float
    total_mentions: int
    sources: list[SentimentSignal]
    consensus: str  # "bullish", "bearish", "mixed", "neutral"
    strength: str  # "strong", "moderate", "weak"
    timestamp: datetime


# Source reliability based on historical accuracy (can be updated over time)
SOURCE_RELIABILITY = {
    "reddit_wsb": 0.45,  # WSB is noisy but can move markets
    "reddit_stocks": 0.55,  # More rational discussion
    "reddit_investing": 0.60,  # Long-term focused
    "stocktwits": 0.50,  # Mixed quality
    "finviz_news": 0.65,  # News headlines
    "yahoo_finance": 0.60,  # General news
    "fear_greed_index": 0.70,  # CNN Fear & Greed - good for overall market
}


class RedditScraper:
    """Scrape sentiment from Reddit (no auth needed for public data)."""

    SUBREDDITS = {
        "wallstreetbets": {"weight": 1.2, "reliability": 0.45},
        "stocks": {"weight": 1.0, "reliability": 0.55},
        "investing": {"weight": 0.8, "reliability": 0.60},
        "stockmarket": {"weight": 0.9, "reliability": 0.55},
    }

    # Keywords for sentiment
    BULLISH_WORDS = [
        "buy", "long", "calls", "moon", "rocket", "bullish", "undervalued",
        "breaking out", "squeeze", "yolo", "diamond hands", "hold", "buying",
        "green", "pump", "rally", "soar", "surge", "upgrade", "beat"
    ]

    BEARISH_WORDS = [
        "sell", "short", "puts", "crash", "dump", "bearish", "overvalued",
        "breaking down", "paper hands", "selling", "red", "tank", "drop",
        "fall", "plunge", "downgrade", "miss", "recession", "bubble"
    ]

    def __init__(self):
        self.headers = {
            "User-Agent": "HyprL-Sentiment/1.0 (research bot)"
        }

    def scrape_subreddit(self, subreddit: str, symbol: str, limit: int = 100) -> SentimentSignal:
        """Scrape a subreddit for mentions of a symbol."""
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": symbol,
            "sort": "new",
            "limit": limit,
            "t": "day",  # Last 24 hours
            "restrict_sr": True
        }

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            posts = data.get("data", {}).get("children", [])

            if not posts:
                return None

            # Analyze sentiment
            bullish_count = 0
            bearish_count = 0
            total_score = 0

            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "").lower()
                selftext = post_data.get("selftext", "").lower()
                text = title + " " + selftext
                score = post_data.get("score", 0)  # upvotes

                # Count sentiment words
                post_bullish = sum(1 for w in self.BULLISH_WORDS if w in text)
                post_bearish = sum(1 for w in self.BEARISH_WORDS if w in text)

                # Weight by upvotes
                weight = max(1, score / 10)

                if post_bullish > post_bearish:
                    bullish_count += weight
                elif post_bearish > post_bullish:
                    bearish_count += weight

            # Calculate score
            total = bullish_count + bearish_count
            if total == 0:
                score = 0
                confidence = 0
            else:
                score = (bullish_count - bearish_count) / total
                confidence = min(1.0, total / 50)  # More mentions = more confidence

            sub_config = self.SUBREDDITS.get(subreddit, {"reliability": 0.5})

            return SentimentSignal(
                source=f"reddit_{subreddit}",
                symbol=symbol,
                score=score,
                confidence=confidence,
                volume=len(posts),
                timestamp=datetime.utcnow(),
                reliability=sub_config["reliability"],
                raw_data={"posts_analyzed": len(posts), "bullish": bullish_count, "bearish": bearish_count}
            )

        except Exception as e:
            print(f"Reddit scrape error ({subreddit}): {e}")
            return None

    def get_sentiment(self, symbol: str) -> list[SentimentSignal]:
        """Get sentiment from all subreddits."""
        signals = []
        for subreddit in self.SUBREDDITS:
            signal = self.scrape_subreddit(subreddit, symbol)
            if signal:
                signals.append(signal)
        return signals


class StockTwitsScraper:
    """Scrape sentiment from StockTwits (free API)."""

    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"

    def get_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Get sentiment for a symbol from StockTwits."""
        url = f"{self.base_url}/streams/symbol/{symbol}.json"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            messages = data.get("messages", [])

            if not messages:
                return None

            # Count sentiment from tagged messages
            bullish = 0
            bearish = 0
            neutral = 0

            for msg in messages:
                sentiment = msg.get("entities", {}).get("sentiment", {})
                if sentiment:
                    if sentiment.get("basic") == "Bullish":
                        bullish += 1
                    elif sentiment.get("basic") == "Bearish":
                        bearish += 1
                    else:
                        neutral += 1

            total = bullish + bearish
            if total == 0:
                return None

            score = (bullish - bearish) / total
            confidence = min(1.0, total / 30)

            return SentimentSignal(
                source="stocktwits",
                symbol=symbol,
                score=score,
                confidence=confidence,
                volume=len(messages),
                timestamp=datetime.utcnow(),
                reliability=SOURCE_RELIABILITY["stocktwits"],
                raw_data={"bullish": bullish, "bearish": bearish, "neutral": neutral}
            )

        except Exception as e:
            print(f"StockTwits error: {e}")
            return None


class FearGreedIndex:
    """Get CNN Fear & Greed Index for overall market sentiment."""

    def get_sentiment(self) -> Optional[SentimentSignal]:
        """Fetch current Fear & Greed Index."""
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code != 200:
                return None

            data = resp.json()
            score_data = data.get("fear_and_greed", {})
            score = score_data.get("score", 50)

            # Convert 0-100 to -1 to +1
            normalized_score = (score - 50) / 50

            # Determine confidence based on extremes
            if score < 20 or score > 80:
                confidence = 0.8
            elif score < 30 or score > 70:
                confidence = 0.6
            else:
                confidence = 0.4

            return SentimentSignal(
                source="fear_greed_index",
                symbol="MARKET",
                score=normalized_score,
                confidence=confidence,
                volume=1,
                timestamp=datetime.utcnow(),
                reliability=SOURCE_RELIABILITY["fear_greed_index"],
                raw_data={"raw_score": score, "rating": score_data.get("rating", "neutral")}
            )

        except Exception as e:
            print(f"Fear & Greed error: {e}")
            return None


class FinvizNewsScraper:
    """Scrape news headlines from Finviz."""

    def get_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Get news sentiment for a symbol."""
        url = f"https://finviz.com/quote.ashx?t={symbol}"

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code != 200:
                return None

            # Simple headline analysis
            text = resp.text.lower()

            positive_words = ["upgrade", "beat", "surge", "rally", "buy", "outperform", "bullish", "growth"]
            negative_words = ["downgrade", "miss", "crash", "sell", "underperform", "bearish", "decline", "cut"]

            pos_count = sum(text.count(w) for w in positive_words)
            neg_count = sum(text.count(w) for w in negative_words)

            total = pos_count + neg_count
            if total == 0:
                return None

            score = (pos_count - neg_count) / total
            confidence = min(1.0, total / 10)

            return SentimentSignal(
                source="finviz_news",
                symbol=symbol,
                score=score,
                confidence=confidence,
                volume=total,
                timestamp=datetime.utcnow(),
                reliability=SOURCE_RELIABILITY["finviz_news"],
                raw_data={"positive": pos_count, "negative": neg_count}
            )

        except Exception as e:
            print(f"Finviz error: {e}")
            return None


class MultiSourceSentiment:
    """Aggregate sentiment from multiple sources."""

    def __init__(self):
        self.reddit = RedditScraper()
        self.stocktwits = StockTwitsScraper()
        self.fear_greed = FearGreedIndex()
        self.finviz = FinvizNewsScraper()

    def get_sentiment(self, symbol: str) -> AggregatedSentiment:
        """Get aggregated sentiment for a symbol."""
        signals = []

        # Collect from all sources
        reddit_signals = self.reddit.get_sentiment(symbol)
        signals.extend(reddit_signals)

        st_signal = self.stocktwits.get_sentiment(symbol)
        if st_signal:
            signals.append(st_signal)

        fg_signal = self.fear_greed.get_sentiment()
        if fg_signal:
            signals.append(fg_signal)

        finviz_signal = self.finviz.get_sentiment(symbol)
        if finviz_signal:
            signals.append(finviz_signal)

        # Aggregate
        if not signals:
            return AggregatedSentiment(
                symbol=symbol,
                final_score=0,
                final_confidence=0,
                total_mentions=0,
                sources=[],
                consensus="neutral",
                strength="weak",
                timestamp=datetime.utcnow()
            )

        # Weighted average by reliability and confidence
        total_weight = 0
        weighted_score = 0
        total_mentions = 0

        for s in signals:
            weight = s.reliability * s.confidence
            weighted_score += s.score * weight
            total_weight += weight
            total_mentions += s.volume

        final_score = weighted_score / total_weight if total_weight > 0 else 0
        final_confidence = min(1.0, total_weight / len(signals))

        # Determine consensus
        if final_score > 0.3:
            consensus = "bullish"
        elif final_score < -0.3:
            consensus = "bearish"
        elif abs(final_score) < 0.1:
            consensus = "neutral"
        else:
            consensus = "mixed"

        # Determine strength
        if abs(final_score) > 0.6 and final_confidence > 0.6:
            strength = "strong"
        elif abs(final_score) > 0.3 or final_confidence > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return AggregatedSentiment(
            symbol=symbol,
            final_score=final_score,
            final_confidence=final_confidence,
            total_mentions=total_mentions,
            sources=signals,
            consensus=consensus,
            strength=strength,
            timestamp=datetime.utcnow()
        )

    def get_report(self, symbol: str) -> str:
        """Get a human-readable sentiment report."""
        agg = self.get_sentiment(symbol)

        lines = [
            f"=== Sentiment Report: {symbol} ===",
            f"Timestamp: {agg.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"Final Score: {agg.final_score:+.2f} ({agg.consensus.upper()})",
            f"Confidence: {agg.final_confidence:.0%}",
            f"Strength: {agg.strength}",
            f"Total Mentions: {agg.total_mentions}",
            f"",
            f"--- Sources ---"
        ]

        for s in agg.sources:
            lines.append(
                f"  {s.source}: {s.score:+.2f} "
                f"(confidence: {s.confidence:.0%}, "
                f"reliability: {s.reliability:.0%}, "
                f"mentions: {s.volume})"
            )

        return "\n".join(lines)


# CLI interface
if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

    analyzer = MultiSourceSentiment()
    report = analyzer.get_report(symbol)
    print(report)
