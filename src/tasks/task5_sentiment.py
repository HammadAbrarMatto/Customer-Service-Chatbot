"""
task5_sentiment.py
------------------
INTERNSHIP TASK 5
"Integrate sentiment analysis into the chatbot to detect and respond
appropriately to customer emotions. Recognise positive, negative, or neutral
sentiments and adjust responses."

What this does, in plain words:
- Before the chatbot answers, we check the FEELING of the user's message:
  positive, negative, or neutral.
- We then tell the chatbot HOW to reply based on that feeling. For example, if
  the user is upset (negative), the bot is asked to be extra calm and caring.

How sentiment is detected:
- We use VADER, a small rule-based tool that is fast, free, and needs no API.
  It works well for short customer messages. (You can swap in a Gemini-based
  classifier later if you want, but VADER keeps this offline and cheap.)
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create the analyzer once and reuse it.
_analyzer = SentimentIntensityAnalyzer()


def detect_sentiment(text: str) -> str:
    """
    Return "positive", "negative", or "neutral" for the given text.

    VADER gives a "compound" score between -1 (very negative) and +1 (very
    positive). We use common thresholds to bucket it.
    """
    score = _analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def tone_instruction(sentiment: str) -> str:
    """
    Give the chatbot a short instruction on how to set its tone, based on the
    detected sentiment. This text gets added to the prompt before answering.
    """
    if sentiment == "negative":
        return (
            "The user sounds upset or frustrated. Reply in a calm, patient, and "
            "empathetic way. Acknowledge their feelings first, then help."
        )
    if sentiment == "positive":
        return (
            "The user sounds happy. Reply in a warm, friendly, and upbeat tone "
            "while staying helpful."
        )
    return "The user sounds neutral. Reply in a clear, polite, and helpful tone."


# Manual test:  python tasks/task5_sentiment.py
if __name__ == "__main__":
    samples = [
        "This is the worst service ever, I'm so angry!",
        "Thank you so much, you've been wonderful!",
        "I want to know the refund policy.",
    ]
    for s in samples:
        mood = detect_sentiment(s)
        print(f"[{mood}] {s}")
