# Customer Service Chatbot 🤖

[![CI](https://github.com/HammadAbrarMatto/Customer-Service-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/HammadAbrarMatto/Customer-Service-Chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini-8e44ad)
![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20FastEmbed-green)

A retrieval-augmented customer-support assistant built on **Google Gemini**, **FAISS** and **Streamlit**. A single chat answers from the company FAQ, reads the customer's mood, and replies in the customer's own language. Separate tools cover image questions, medical Q&A (MedQuAD) and an arXiv research expert.

---

## Features

| | Feature | Details |
|---|---|---|
| 💬 | **Support chat (RAG)** | Multi-turn chat with streaming answers grounded in the FAQ, source citations, a relevance threshold, and an "I don't know → human agent" fallback |
| 😟 | **Sentiment-aware tone** (Task 5) | VADER detects frustration or happiness; the reply acknowledges it and offers escalation when needed |
| 🌐 | **Multilingual** (Task 6) | Lingua detects 20+ languages. Multilingual embeddings match a Hindi or Spanish question to an English FAQ, and the answer comes back in the customer's language |
| 📚 | **Live knowledge-base expansion** (Task 1) | Upload a CSV or add a web page from the UI or CLI. Documents are merged incrementally and every update is logged |
| 🖼️ | **Image chat** (Task 2) | Ask about a photo of a damaged item, an error screenshot or a receipt (Gemini vision, streamed) |
| 🩺 | **Medical Q&A** (Task 3) | RAG over MedQuAD (16k Q&A pairs → 28k passages), entity recognition (symptoms, diseases, treatments, tests), safety prompt and disclaimer |
| 📄 | **arXiv expert** (Task 4) | Semantic paper search, category chart, per-paper summaries, side-by-side comparison, concept explanations |

---

## What changed in v2

| Problem in v1 | v2 |
|---|---|
| `gemini-2.5-flash` was shut down on 16 Oct 2026, so the app no longer answered | Default model is **`gemini-3.6-flash`**, configurable through `.env` |
| `google-generativeai` and LangChain 0.2 are deprecated, and `langchain-community` is being sunset | Official **`google-genai`** SDK plus a ~80-line FAISS store, with no LangChain dependency |
| Embeddings went one text per request with a 15 s sleep, so indexes were capped at **50 documents** and took 15–30 min to build | **Local FastEmbed** embeddings by default: free, no rate limits, all FAQ rows in seconds and thousands of MedQuAD/arXiv docs. Gemini embeddings remain an option and are now **batched 100 per call** |
| English-only embeddings | Multilingual embedding model, so cross-lingual retrieval works |
| Indexes saved with pickle (`allow_dangerous_deserialization=True`) | FAISS file + JSON, so loading an index can never run code |
| `score_threshold=0.7` was silently ignored, so irrelevant FAQs were always stuffed into the prompt | A real cosine-similarity threshold (`MIN_RELEVANCE`) |
| Six separate single-shot pages; sentiment and language were never combined with the FAQ | One **multi-turn chat** that combines RAG, sentiment and language, with memory and streaming |
| Medical/arXiv paths were relative to the working directory, and the FAQ CSV was duplicated | Absolute paths from one config, a `data/` folder and one copy of each dataset |
| `langdetect` no longer installs on Python 3.13 | `lingua-language-detector` |
| No tests | 27 offline tests (fake LLM + embedder), ruff and CI |

---

## Project structure

```
Customer-Service-Chatbot/
├── app.py                     # Streamlit UI (all features)
├── src/csbot/
│   ├── config.py              # settings (.env)
│   ├── llm.py                 # Gemini generate / stream
│   ├── embeddings.py          # FastEmbed (local) or Gemini (batched)
│   ├── vectorstore.py         # FAISS + JSON persistence
│   ├── knowledge.py           # named KBs: faq / medical / arxiv, update log
│   ├── ingest.py              # CSV, web, MedQuAD, arXiv loaders + chunking
│   ├── chat.py                # RAG + sentiment + language + memory
│   ├── cli.py                 # build / update indexes, ask from terminal
│   └── features/
│       ├── sentiment.py  language.py  vision.py  medical.py  arxiv.py
├── data/
│   ├── faq.csv                # base FAQ (prompt,response)
│   └── medquad_prepared.csv   # MedQuAD flattened
├── notebooks/analysis.ipynb   # dataset analysis & sentiment evaluation
├── docs/Project_Report.docx
├── tests/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/HammadAbrarMatto/Customer-Service-Chatbot.git
cd Customer-Service-Chatbot
python -m venv venv
venv\Scripts\activate            # Windows   (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env           # add GEMINI_API_KEY (free: https://aistudio.google.com/app/apikey)
streamlit run app.py
```

On first launch the FAQ index builds automatically, which takes about 20 s. The local embedding model (~220 MB) downloads once.

### Optional indexes

```bash
# Medical Q&A (MedQuAD CSV is already in data/)
python -m csbot.cli build-medical --limit 5000     # 0 = all ~28k chunks

# arXiv expert (download arxiv-metadata-oai-snapshot.json from Kaggle)
python -m csbot.cli build-arxiv path/to/arxiv-metadata-oai-snapshot.json --category cs. --limit 5000

# Grow the support KB from the terminal (also possible in the UI)
python -m csbot.cli add-csv more_faqs.csv
python -m csbot.cli add-url https://example.com/help/refunds

# Quick test without the UI
python -m csbot.cli ask "Do you provide internships?"
```

Run the CLI with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`). To refresh the knowledge base periodically, schedule `add-url` / `add-csv` with cron or Windows Task Scheduler.

### Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | – | required for answers |
| `CHAT_MODEL` / `VISION_MODEL` | `gemini-3.6-flash` | any current Gemini model |
| `EMBEDDING_BACKEND` | `local` | `local` (FastEmbed) or `gemini` |
| `LOCAL_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | any FastEmbed model |
| `MIN_RELEVANCE` | `0.35` | FAQ matches below this cosine score are ignored |

Changing the embedding backend requires rebuilding the indexes (the app detects a mismatch and says so).

---

## ✅ Live check

`python scripts/live_check.py` runs against the real Gemini API and the local embedding model: generate, streaming, FAQ index build, English and **Hindi→English cross-lingual** retrieval, the off-topic filter, a full Spanish/negative chat turn, vision, batched Gemini embeddings and medical Q&A. It uses a temporary index folder, so your real indexes are untouched.

## Tests

```bash
pip install pytest ruff
pytest -q && ruff check .
```

The suite uses a hashing embedder and a fake LLM, so it runs offline without an API key.

## License

MIT
