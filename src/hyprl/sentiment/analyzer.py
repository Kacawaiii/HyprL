from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from hyprl.data.news import NewsArticle


@dataclass(slots=True)
class SentimentScorer:
    analyzer: SentimentIntensityAnalyzer

    @classmethod
    def default(cls) -> "SentimentScorer":
        return cls(analyzer=SentimentIntensityAnalyzer())

    def score_headlines(self, headlines: Iterable[str]) -> List[float]:
        scores: List[float] = []
        for headline in headlines:
            if not headline:
                continue
            polarity = self.analyzer.polarity_scores(headline)["compound"]
            scores.append(polarity)
        return scores

    def annotate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        for article in articles:
            article.sentiment = self.analyzer.polarity_scores(article.title)["compound"]
        return articles
