from __future__ import annotations

import pytest

from csbot import ingest
from csbot.chat import Analysis, Turn, analyze, build_prompt
from csbot.config import settings
from csbot.features.language import detect_language, language_instruction
from csbot.features.medical import recognise_entities
from csbot.features.sentiment import detect_sentiment, tone_instruction
from csbot.knowledge import KnowledgeBase, get_kb
from csbot.vectorstore import Document, VectorStore


# ── chunking & ingestion ────────────────────────────────────────────────────
def test_chunk_text_sizes_and_coverage():
    text = " ".join(f"Sentence {i} talks about refunds and policies." for i in range(200))
    chunks = ingest.chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 5 and all(len(c) <= 300 for c in chunks)
    assert "Sentence 199" in chunks[-1]


def test_faq_csv_loads_all_rows():
    docs = ingest.load_faq_csv(settings.faq_csv)
    assert len(docs) > 50
    assert all(d.metadata["question"] and d.metadata["answer"] for d in docs)


def test_load_faq_bytes_accepts_question_answer_columns():
    docs = ingest.load_faq_bytes(b"\xef\xbb\xbfquestion,answer\nHi?,Hello\n,skip\n", "x.csv")
    assert len(docs) == 1 and docs[0].metadata["answer"] == "Hello"


def test_medquad_loader_limit():
    docs = ingest.load_medquad(settings.medquad_csv, limit=25)
    assert len(docs) == 25 and docs[0].metadata["source"] == "MedQuAD"


# ── vector store ────────────────────────────────────────────────────────────
def test_vectorstore_search_and_roundtrip(tmp_path, embedder):
    store = VectorStore(embedder)
    store.add([Document("refund policy money back"), Document("internship placement support"),
               Document("python course duration weeks")])
    assert store.search("do you give a refund", k=1)[0].document.text.startswith("refund")
    store.save(tmp_path / "s")
    loaded = VectorStore.load(tmp_path / "s", embedder)
    assert len(loaded) == 3
    assert loaded.search("internship", k=1)[0].document.text.startswith("internship")


def test_vectorstore_rejects_mismatched_embedder(tmp_path, embedder):
    store = VectorStore(embedder)
    store.add([Document("a b c")])
    store.save(tmp_path / "s")

    class Other(type(embedder)):
        name = "other"

    with pytest.raises(ValueError):
        VectorStore.load(tmp_path / "s", Other())


def test_index_files_are_not_pickles(tmp_path, embedder):
    store = VectorStore(embedder)
    store.add([Document("hello", {"k": 1})])
    store.save(tmp_path / "s")
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["docs.jsonl", "index.faiss", "meta.json"]


# ── knowledge base ──────────────────────────────────────────────────────────
def test_kb_rebuild_add_and_log(embedder):
    kb = KnowledgeBase("faq", embedder)
    assert kb.rebuild(ingest.load_faq_csv(settings.faq_csv)) > 50
    before = len(kb)
    kb.add([Document("Q: Do you offer EMI?\nA: Yes, 6-month EMI is available.", {"question": "EMI?"})], "test")
    assert len(KnowledgeBase("faq", embedder)) == before + 1  # persisted to disk
    assert any("added 1" in line for line in kb.history())
    hits = kb.search("Do you offer EMI payment?", min_score=0.0)
    assert any("EMI" in h.document.text for h in hits)


def test_kb_threshold_filters_irrelevant(embedder):
    kb = KnowledgeBase("faq", embedder)
    kb.rebuild([Document("refund policy")])
    assert kb.search("zebra giraffe safari", min_score=0.5) == []


# ── sentiment / language / entities ─────────────────────────────────────────
@pytest.mark.parametrize("text,label", [("This is the worst service ever, I'm furious!", "negative"),
                                        ("Thank you so much, you've been wonderful!", "positive"),
                                        ("What is the refund policy?", "neutral")])
def test_sentiment(text, label):
    s = detect_sentiment(text)
    assert s.label == label
    assert tone_instruction(s)


@pytest.mark.parametrize("text,lang", [("Hola, necesito ayuda con mi pedido por favor", "Spanish"),
                                       ("Bonjour, je voudrais une information sur ma commande", "French"),
                                       ("नमस्ते, मुझे मेरे ऑर्डर के बारे में मदद चाहिए", "Hindi"),
                                       ("Hello, can you help me with my order?", "English"),
                                       ("ok", "English")])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_language_instruction():
    assert "Hindi" in language_instruction("Hindi") and language_instruction("English") == "Reply in English."


def test_medical_entities():
    e = recognise_entities("I have a headache and fever, could it be migraine? Do I need an MRI?")
    assert e["symptoms"] == ["fever", "headache"] and e["diseases"] == ["migraine"] and e["tests"] == ["mri"]


# ── chat prompt ─────────────────────────────────────────────────────────────
def test_build_prompt_includes_context_tone_language_and_history(embedder):
    kb = get_kb("faq")
    kb.rebuild(ingest.load_faq_csv(settings.faq_csv))
    a = analyze("Esto es terrible, ¿ofrecen pasantías?", kb)
    assert a.language == "Spanish"
    system, prompt = build_prompt("¿ofrecen pasantías?", a, [Turn("user", "hi"), Turn("assistant", "hello!")])
    assert "Spanish" in system and "CONTEXT" in prompt and "USER: hi" in prompt


def test_build_prompt_without_hits():
    a = Analysis(language="English", sentiment=detect_sentiment("hi"), hits=[])
    _, prompt = build_prompt("hi", a, [])
    assert "(no relevant FAQ)" in prompt


def test_gemini_embedder_batches(monkeypatch):
    from types import SimpleNamespace

    from csbot import llm
    from csbot.embeddings import GeminiEmbedder

    calls = []

    def embed_content(model, contents, config):
        calls.append(len(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 2.0]) for _ in contents])

    fake = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    monkeypatch.setattr(llm, "client", lambda: fake)
    vecs = GeminiEmbedder(dims=3).embed_documents([f"t{i}" for i in range(250)])
    assert calls == [100, 100, 50] and vecs.shape == (250, 3)
    assert abs(float((vecs[0] ** 2).sum()) - 1) < 1e-6  # normalised
