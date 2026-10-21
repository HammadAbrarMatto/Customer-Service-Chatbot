"""Task 4 — arXiv domain expert: semantic paper search, summaries, explanations."""

from __future__ import annotations

from collections import Counter

from .. import llm
from ..knowledge import KnowledgeBase


def search(kb: KnowledgeBase, query: str, k: int = 8) -> list[dict]:
    if not kb.ready:
        raise FileNotFoundError("arXiv index not built yet — run: python -m csbot.cli build-arxiv <json>")
    return [{**h.document.metadata, "score": round(h.score, 3)} for h in kb.search(query, k=k, min_score=0.0)]


def summarise(title: str, abstract: str) -> str:
    return llm.generate(
        f"Paper: {title}\n\nAbstract: {abstract}\n\n"
        "Give: (1) a 2-sentence plain-English summary, (2) the key contribution, (3) the method in one line, "
        "(4) who would find it useful. Use Markdown bullets."
    )


def explain(concept: str, papers: list[dict] | None = None) -> str:
    ctx = "\n".join(f"- {p['title']}: {p.get('abstract', '')[:400]}" for p in (papers or [])[:3])
    extra = f"\n\nRelated papers you may reference:\n{ctx}" if ctx else ""
    return llm.generate(
        f'Explain "{concept}" for a beginner in under 180 words, with one everyday analogy and one short example. '
        f"End with 2 suggested follow-up topics.{extra}"
    )


def compare(papers: list[dict]) -> str:
    listing = "\n\n".join(f"[{i}] {p['title']}\n{p.get('abstract', '')[:900]}" for i, p in enumerate(papers, 1))
    return llm.generate(f"Compare these papers in a Markdown table (problem, method, data, key result), "
                        f"then one paragraph on how they relate.\n\n{listing}")


def category_counts(papers: list[dict]) -> Counter:
    return Counter(c for p in papers for c in p.get("categories", "").split())
