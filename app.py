"""Customer Service Chatbot — Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from csbot import ingest  # noqa: E402
from csbot.chat import Turn, respond  # noqa: E402
from csbot.config import settings  # noqa: E402
from csbot.knowledge import get_kb  # noqa: E402
from csbot.llm import MissingAPIKey  # noqa: E402

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖", layout="wide")

PAGES = ["💬 Support chat", "📚 Knowledge base", "🖼️ Image chat", "🩺 Medical Q&A", "📄 arXiv expert"]

with st.sidebar:
    st.title("🤖 Support Bot")
    page = st.radio("Feature", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"LLM: `{settings.chat_model}`")
    st.caption(f"Embeddings: `{settings.embedding_backend}`")
    if not settings.api_key:
        st.error("Set `GEMINI_API_KEY` in `.env` to enable answers.", icon="🔑")


@st.cache_resource(show_spinner="Loading knowledge base…")
def kb(name: str):
    return get_kb(name)


def ensure_faq_index():
    base = kb("faq")
    if not base.ready:
        with st.spinner("First run: indexing the FAQ dataset (local embeddings, ~20 s)…"):
            base.rebuild(ingest.load_faq_csv(settings.faq_csv))
    return base


def show_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"📎 {len(hits)} source(s)"):
        for h in hits:
            q = h.document.metadata.get("question") or h.document.metadata.get("title") or h.document.metadata.get(
                "source", "")
            st.markdown(f"**{q}** · relevance `{h.score:.2f}`")
            st.caption(h.document.metadata.get("answer", h.document.text)[:400])


def handle_llm_errors(fn):
    try:
        return fn()
    except MissingAPIKey as e:
        st.error(str(e))
    except FileNotFoundError as e:
        st.warning(str(e))
    except Exception as e:  # noqa: BLE001 – surface API errors (quota, model name, network)
        st.error(f"LLM error: {e}")
    return None


# ── 💬 Unified support chat (base RAG + Task 5 sentiment + Task 6 language) ────
if page == PAGES[0]:
    st.header("💬 Ask about our courses & services")
    st.caption("Answers come from the FAQ knowledge base. Mood and language are detected automatically, so "
               "write in Hindi, Spanish, French… and the bot replies in kind.")
    faq = ensure_faq_index()
    st.session_state.setdefault("chat", [])
    history: list[Turn] = st.session_state.chat

    c1, c2 = st.columns([6, 1])
    if c2.button("🧹 New chat"):
        st.session_state.chat = []
        st.rerun()
    if not history:
        st.markdown("Try: *Do you provide internships?* · *¿Ofrecen certificados?* · "
                    "*This is the third time my payment failed, I'm really annoyed!*")

    for t in history:
        with st.chat_message(t.role):
            st.markdown(t.content)

    if msg := st.chat_input("Type your message…"):
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            result = handle_llm_errors(lambda: respond(msg, history, faq))
            if result:
                analysis, tokens = result
                s = analysis.sentiment
                st.caption(f"{s.emoji} {s.label} ({s.compound:+.2f}) · 🌐 {analysis.language} · "
                           f"{len(analysis.hits)} FAQ match(es)")
                answer = handle_llm_errors(lambda: st.write_stream(tokens))
                show_sources(analysis.hits)
                if answer:
                    history += [Turn("user", msg), Turn("assistant", answer)]

# ── 📚 Task 1: dynamic knowledge base ──────────────────────────────────────────
elif page == PAGES[1]:
    st.header("📚 Expand the knowledge base")
    faq = ensure_faq_index()
    st.metric("Documents in FAQ index", len(faq))
    t_csv, t_url, t_log = st.tabs(["CSV upload", "Web page", "Update log"])
    with t_csv:
        st.caption("Columns: `prompt,response` (or `question,answer`).")
        up = st.file_uploader("FAQ CSV", type=["csv"])
        if up and st.button("➕ Add to knowledge base", type="primary"):
            docs = ingest.load_faq_bytes(up.getvalue(), up.name)
            with st.spinner(f"Embedding {len(docs)} rows…"):
                n = faq.add(docs, source=up.name)
            st.success(f"Added {n} Q&A pairs.")
    with t_url:
        url = st.text_input("Page URL", placeholder="https://example.com/help/refunds")
        if url and st.button("➕ Fetch & add", type="primary"):
            try:
                docs = ingest.load_url(url)
                with st.spinner(f"Embedding {len(docs)} chunks…"):
                    n = faq.add(docs, source=url)
                st.success(f"Added {n} chunks.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not ingest page: {e}")
    with t_log:
        log = faq.history()
        st.code("\n".join(log[-50:]) or "No updates yet.")
        if st.button("♻️ Rebuild from data/faq.csv"):
            with st.spinner("Rebuilding…"):
                n = faq.rebuild(ingest.load_faq_csv(settings.faq_csv))
            st.success(f"Rebuilt with {n} documents.")

# ── 🖼️ Task 2: multimodal ─────────────────────────────────────────────────────
elif page == PAGES[2]:
    from PIL import Image

    from csbot.features.vision import ask_about_image

    st.header("🖼️ Ask about an image")
    st.caption("e.g. a damaged parcel, an error message screenshot, a receipt")
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    q = st.text_input("Your question (optional)")
    if up:
        img = Image.open(up)
        st.image(img, width=420)
        if st.button("🔍 Ask Gemini", type="primary"):
            handle_llm_errors(lambda: st.write_stream(ask_about_image(img, q)))

# ── 🩺 Task 3: medical Q&A ───────────────────────────────────────────────────
elif page == PAGES[3]:
    from csbot.features import medical

    st.header("🩺 Medical Q&A (MedQuAD)")
    med = kb("medical")
    if not med.ready:
        st.info("Build the medical index first:\n\n`python -m csbot.cli build-medical --limit 5000`")
    q = st.text_input("Medical question", placeholder="What are the symptoms of glaucoma?")
    if q and med.ready:
        ents = medical.recognise_entities(q)
        if any(ents.values()):
            st.caption(" · ".join(f"**{k}**: {', '.join(v)}" for k, v in ents.items() if v))
        with st.spinner("Searching MedQuAD…"):
            res = handle_llm_errors(lambda: medical.answer(q, med))
        if res:
            st.markdown(res["answer"])
            show_sources(res["sources"])

# ── 📄 Task 4: arXiv expert ──────────────────────────────────────────────────
else:
    import pandas as pd

    from csbot.features import arxiv

    st.header("📄 arXiv research expert")
    ax = kb("arxiv")
    if not ax.ready:
        st.info("Build the arXiv index first (Kaggle `arxiv-metadata-oai-snapshot.json`):\n\n"
                "`python -m csbot.cli build-arxiv path/to/arxiv.json --category cs. --limit 5000`")
    t_search, t_explain = st.tabs(["🔎 Search papers", "💡 Explain a concept"])
    with t_search:
        q = st.text_input("Search by meaning", placeholder="efficient attention for long sequences")
        if q and ax.ready:
            papers = arxiv.search(ax, q)
            st.session_state.papers = papers
            counts = arxiv.category_counts(papers)
            if counts:
                st.bar_chart(pd.Series(counts).sort_values(ascending=False).head(10))
            for i, p in enumerate(papers):
                with st.expander(f"{p['title']}  ·  `{p['score']}`"):
                    st.caption(f"{p.get('authors', '')} · {p.get('categories', '')} · [{p['id']}]({p.get('url', '')})")
                    st.write(p.get("abstract", ""))
                    if st.button("📝 Summarise", key=f"sum{i}"):
                        handle_llm_errors(lambda p=p: st.markdown(arxiv.summarise(p["title"], p["abstract"])))
            if len(papers) >= 2 and st.button("⚖️ Compare top 3"):
                handle_llm_errors(lambda: st.markdown(arxiv.compare(papers[:3])))
    with t_explain:
        concept = st.text_input("Concept", placeholder="attention mechanism")
        if concept:
            related = arxiv.search(ax, concept, k=3) if ax.ready else []
            handle_llm_errors(lambda: st.markdown(arxiv.explain(concept, related)))
