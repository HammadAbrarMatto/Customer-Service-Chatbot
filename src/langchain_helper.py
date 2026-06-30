"""
langchain_helper.py
-------------------
Uses direct REST calls to Google v1 API for embeddings.
Both google-generativeai and langchain-google-genai are hardcoded to v1beta
which does not support gemini-embedding-001. This bypasses them entirely.
"""

import time
import requests
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from typing import List

from config import (
    GOOGLE_API_KEY,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    VECTORDB_PATH,
    FAQ_DATASET_PATH,
)

llm = GoogleGenerativeAI(
    model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1
)

_MODEL_ID = EMBEDDING_MODEL.replace("models/", "")
_EMBED_URL_SINGLE = (
    f"https://generativelanguage.googleapis.com/v1/models/"
    f"{_MODEL_ID}:embedContent?key={GOOGLE_API_KEY}"
)


def _embed_one(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Call the REST API for a single text with automatic retry."""
    body = {
        "model": f"models/{_MODEL_ID}",
        "content": {"parts": [{"text": text}]},
    }
    for attempt in range(5):
        resp = requests.post(_EMBED_URL_SINGLE, json=body)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        elif resp.status_code in (429, 503):
            wait = 60 * (attempt + 1)
            print(f"Rate limited ({resp.status_code}), waiting {wait}s...")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    resp.raise_for_status()


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, text in enumerate(texts):
            results.append(_embed_one(text, "RETRIEVAL_DOCUMENT"))
        time.sleep(15)  # free tier allows ~5 req/min
        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(texts)} documents...")
        return results

    def embed_query(self, text: str) -> List[float]:
        return _embed_one(text, "RETRIEVAL_QUERY")


embeddings = GeminiEmbeddings()


def create_vector_db(csv_path: str = FAQ_DATASET_PATH) -> None:
    loader = CSVLoader(file_path=csv_path, source_column="prompt", encoding="utf-8")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(VECTORDB_PATH)


def load_vector_db() -> FAISS:
    return FAISS.load_local(
        VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def get_qa_chain() -> RetrievalQA:
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an
answer based on this context only. In the answer try to provide as much text as
possible from the "response" section in the source document context without
making many changes. If the answer is not found in the context, kindly state
"I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you provide internship?"))
