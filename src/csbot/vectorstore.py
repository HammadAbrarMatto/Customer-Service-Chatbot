"""Minimal FAISS vector store with safe, pickle-free persistence.

v1 used LangChain's FAISS wrapper, which saves documents with pickle and needs
``allow_dangerous_deserialization=True`` to load. Here the index is a plain FAISS
file and documents are JSON, so loading an index can never execute code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .embeddings import Embedder


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    document: Document
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, embedder: Embedder, dim: int | None = None):
        self.embedder = embedder
        self.docs: list[Document] = []
        self.index: faiss.Index | None = faiss.IndexFlatIP(dim) if dim else None

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, docs: list[Document]) -> int:
        docs = [d for d in docs if d.text.strip()]
        if not docs:
            return 0
        vecs = self.embedder.embed_documents([d.text for d in docs])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.ascontiguousarray(vecs, dtype="float32"))
        self.docs.extend(docs)
        return len(docs)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.docs or self.index is None:
            return []
        q = np.ascontiguousarray(self.embedder.embed_query(query)[None, :], dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self.docs)))
        return [Hit(self.docs[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        with (path / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        (path / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "count": len(self.docs)}))

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> VectorStore:
        meta = json.loads((path / "meta.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"Index at {path} was built with '{meta['embedder']}' but the current embedder is "
                f"'{embedder.name}'. Rebuild the index or change EMBEDDING_BACKEND."
            )
        store = cls(embedder)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with (path / "docs.jsonl").open(encoding="utf-8") as f:
            store.docs = [Document(**json.loads(line)) for line in f if line.strip()]
        return store

    @staticmethod
    def exists(path: Path) -> bool:
        return (path / "index.faiss").exists() and (path / "docs.jsonl").exists()
