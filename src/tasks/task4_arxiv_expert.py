"""
task4_arxiv_expert.py
---------------------
INTERNSHIP TASK 4
"Develop a chatbot that serves as an expert in a specific domain. Use the arXiv
dataset for scientific papers. Train on a subset (e.g. computer science).
Implement information extraction, summarization, and explanation. Streamlit with
paper searching and concept visualization."

What this does, in plain words:
- Loads a subset of the arXiv metadata (title + abstract + category).
- Builds a FAISS database so users can SEARCH papers by meaning, not just words.
- For a chosen paper, Gemini can SUMMARISE the abstract in plain language.
- Gemini can EXPLAIN a concept the user asks about, in simple terms.

Where to get the data:
- arXiv dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
- It is a large JSON-lines file. The helper below reads it, keeps only the
  category you choose (default: computer science "cs."), and keeps only the
  first N papers so it stays small and fast.
"""

import os
import json

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_helper import embeddings, llm

ARXIV_VECTORDB_PATH = "faiss_index_arxiv"


def build_arxiv_db(
    json_path: str,
    category_prefix: str = "cs.",
    max_papers: int = 50,
) -> int:
    """
    Build a FAISS database from a subset of the arXiv JSON file.

    json_path:       path to arxiv-metadata-oai-snapshot.json (from Kaggle).
    category_prefix: keep only papers whose category starts with this
                     (default "cs." = computer science).
    max_papers:      stop after this many, to keep things small.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"arXiv JSON not found: {json_path}")

    docs = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            if len(docs) >= max_papers:
                break
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            categories = paper.get("categories", "")
            if not categories.startswith(category_prefix):
                continue

            title = paper.get("title", "").strip().replace("\n", " ")
            abstract = paper.get("abstract", "").strip().replace("\n", " ")
            if not title or not abstract:
                continue

            # We store the abstract as the searchable text, title as metadata.
            docs.append(
                Document(
                    page_content=f"{title}. {abstract}",
                    metadata={"title": title, "id": paper.get("id", "")},
                )
            )

    if not docs:
        raise ValueError("No papers matched. Check the category prefix.")

    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(ARXIV_VECTORDB_PATH)
    return len(docs)


def _load_db() -> FAISS:
    if not os.path.exists(ARXIV_VECTORDB_PATH):
        raise FileNotFoundError("arXiv database missing. Run build_arxiv_db() first.")
    return FAISS.load_local(
        ARXIV_VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def search_papers(query: str, k: int = 5) -> list:
    """Return the top k papers most relevant to the query."""
    vectordb = _load_db()
    results = vectordb.similarity_search(query, k=k)
    return [
        {"title": d.metadata.get("title", ""), "snippet": d.page_content[:300]}
        for d in results
    ]


def summarise(text: str) -> str:
    """Summarise a paper abstract (or any text) in plain language."""
    prompt = f"""Summarise the following scientific text in 4-5 simple sentences
that a non-expert can understand. Avoid heavy jargon.

TEXT:
{text}
"""
    return llm.invoke(prompt)


def explain_concept(concept: str) -> str:
    """Explain a scientific concept simply, with a short example."""
    prompt = f"""Explain the concept "{concept}" in simple language for a
beginner. Give one short everyday example to make it clear. Keep it under 150
words.
"""
    return llm.invoke(prompt)


# Manual test:  python tasks/task4_arxiv_expert.py
if __name__ == "__main__":
    print(explain_concept("attention mechanism in transformers"))
