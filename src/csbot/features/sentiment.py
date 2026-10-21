"""Task 5 — sentiment-aware replies (VADER: fast, offline, tuned for short messages)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@dataclass(frozen=True)
class Sentiment:
    label: str  # positive | negative | neutral
    compound: float

    @property
    def emoji(self) -> str:
        return {"positive": "🙂", "negative": "😟", "neutral": "😐"}[self.label]


def detect_sentiment(text: str) -> Sentiment:
    score = _analyzer().polarity_scores(text)["compound"]
    label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
    return Sentiment(label, score)


def tone_instruction(sentiment: Sentiment | str) -> str:
    label = sentiment.label if isinstance(sentiment, Sentiment) else sentiment
    if label == "negative":
        return ("The customer sounds upset or frustrated. Start by briefly acknowledging their frustration and "
                "apologising for the inconvenience, stay calm and patient, then give a concrete next step. "
                "If the issue cannot be solved from the context, offer to escalate to a human agent.")
    if label == "positive":
        return "The customer sounds happy. Reply in a warm, upbeat tone while staying concise and helpful."
    return "Reply in a clear, polite and professional tone."
