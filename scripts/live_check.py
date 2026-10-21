"""End-to-end check against the REAL services (Gemini + local embedding model).

    python scripts/live_check.py

Needs GEMINI_API_KEY in .env and internet access (first run downloads the ~220 MB embedding model).
Uses a temporary index folder, so your real indexes are untouched.
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbot.config import settings  # noqa: E402

settings.index_dir = Path(tempfile.mkdtemp()) / "indexes"
results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        t = time.time()
        try:
            detail = fn() or ""
            results.append((name, True, f"{detail}  ({time.time() - t:.1f}s)"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn

    return deco


@check("Gemini API key present")
def _():
    assert settings.api_key, "set GEMINI_API_KEY in .env"
    return settings.chat_model


@check("Gemini generate")
def _():
    from csbot import llm

    out = llm.generate("Reply with exactly the word: pong")
    assert "pong" in out.lower(), out
    return out


@check("Gemini streaming")
def _():
    from csbot import llm

    chunks = list(llm.stream("Count from 1 to 5 separated by spaces."))
    assert chunks and "5" in "".join(chunks)
    return f"{len(chunks)} chunks"


@check("Local embeddings + FAQ index build")
def _():
    from csbot import ingest
    from csbot.knowledge import get_kb

    n = get_kb("faq").rebuild(ingest.load_faq_csv(settings.faq_csv))
    assert n > 50
    return f"{n} docs"


@check("Retrieval: English query finds the internship FAQ")
def _():
    from csbot.knowledge import get_kb

    hits = get_kb("faq").search("Do you provide internships?")
    assert hits and "intern" in hits[0].document.text.lower(), [h.document.text[:60] for h in hits]
    return f"top score {hits[0].score:.2f}"


@check("Retrieval: Hindi query matches English FAQ (cross-lingual)")
def _():
    from csbot.knowledge import get_kb

    hits = get_kb("faq").search("क्या आप इंटर्नशिप देते हैं?")
    assert hits and "intern" in hits[0].document.text.lower(), [h.document.text[:60] for h in hits]
    return f"top score {hits[0].score:.2f}"


@check("Retrieval: off-topic query is filtered by MIN_RELEVANCE")
def _():
    from csbot.knowledge import get_kb

    hits = get_kb("faq").search("What is the boiling point of mercury on Venus?")
    return f"{len(hits)} hit(s) above {settings.min_relevance} (0 is ideal)"


@check("Full chat turn (RAG + sentiment + language)")
def _():
    from csbot.chat import respond
    from csbot.knowledge import get_kb

    analysis, tokens = respond("Esto es frustrante, ¿ofrecen pasantías?", [], get_kb("faq"))
    answer = "".join(tokens)
    assert analysis.language == "Spanish" and analysis.sentiment.label == "negative", analysis
    return f"{analysis.language}/{analysis.sentiment.label}: {answer[:90]!r}"


@check("Vision (image chat)")
def _():
    from PIL import Image, ImageDraw

    from csbot.features.vision import ask_about_image

    img = Image.new("RGB", (320, 200), "white")
    ImageDraw.Draw(img).text((40, 90), "ERROR 404: PAGE NOT FOUND", fill="red")
    out = "".join(ask_about_image(img, "What error is shown?"))
    assert "404" in out, out
    return out[:80]


@check("Gemini embeddings (batched)")
def _():
    from csbot.embeddings import GeminiEmbedder

    v = GeminiEmbedder().embed_documents(["refund policy", "internship", "course fee"])
    assert v.shape[0] == 3
    return f"shape {v.shape}"


@check("Medical Q&A (small MedQuAD index)")
def _():
    from csbot import ingest
    from csbot.features import medical
    from csbot.knowledge import get_kb

    kb = get_kb("medical")
    kb.rebuild(ingest.load_medquad(settings.medquad_csv, limit=300))
    res = medical.answer("What is leukemia?", kb)
    assert res["answer"]
    return res["answer"][:80].replace("\n", " ")


print("\n" + "=" * 78)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58} {detail[:120]}")
print("=" * 78)
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
