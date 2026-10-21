"""The unified support chat: retrieval + sentiment + language + conversation memory."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from . import llm
from .config import settings
from .features.language import detect_language, language_instruction
from .features.sentiment import Sentiment, detect_sentiment, tone_instruction
from .knowledge import KnowledgeBase
from .vectorstore import Hit

SYSTEM = """You are a friendly customer-support assistant for an online AI/data-science course provider.
Rules:
- Answer ONLY from the CONTEXT (FAQ entries). Prefer the wording of the matching answer.
- If the context does not contain the answer, say you don't have that information and offer to connect
  the customer with a human agent. Never invent prices, dates, policies or links.
- Keep answers short (2–5 sentences) unless the customer asks for detail.
{tone}
{language}"""


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Analysis:
    language: str
    sentiment: Sentiment
    hits: list[Hit] = field(default_factory=list)


def analyze(message: str, kb: KnowledgeBase | None) -> Analysis:
    return Analysis(
        language=detect_language(message),
        sentiment=detect_sentiment(message),
        hits=kb.search(message) if kb is not None and kb.ready else [],
    )


def build_prompt(message: str, analysis: Analysis, history: list[Turn]) -> tuple[str, str]:
    system = SYSTEM.format(tone=tone_instruction(analysis.sentiment), language=language_instruction(analysis.language))
    context = "\n\n".join(f"[{i}] {h.document.text}" for i, h in enumerate(analysis.hits, 1)) or "(no relevant FAQ)"
    recent = history[-2 * settings.max_history_turns:]
    convo = "\n".join(f"{t.role.upper()}: {t.content}" for t in recent)
    prompt = f"CONTEXT:\n{context}\n\nCONVERSATION SO FAR:\n{convo or '(new conversation)'}\n\nCUSTOMER: {message}"
    return system, prompt


def respond(message: str, history: list[Turn], kb: KnowledgeBase | None) -> tuple[Analysis, Iterator[str]]:
    """Returns the analysis immediately and a token stream for the answer."""
    analysis = analyze(message, kb)
    system, prompt = build_prompt(message, analysis, history)
    return analysis, llm.stream(prompt, system=system)
