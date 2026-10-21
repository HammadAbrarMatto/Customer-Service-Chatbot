"""Named knowledge bases (faq, medical, arxiv) stored under `indexes/<name>/`."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from .config import settings
from .embeddings import Embedder, get_embedder
from .vectorstore import Document, Hit, VectorStore


class KnowledgeBase:
    def __init__(self, name: str, embedder: Embedder | None = None):
        self.name = name
        self.path = settings.index_path(name)
        self.embedder = embedder or get_embedder()
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            if VectorStore.exists(self.path):
                self._store = VectorStore.load(self.path, self.embedder)
            else:
                self._store = VectorStore(self.embedder)
        return self._store

    @property
    def ready(self) -> bool:
        return len(self.store) > 0

    def __len__(self) -> int:
        return len(self.store)

    def rebuild(self, docs: list[Document]) -> int:
        self._store = VectorStore(self.embedder)
        n = self._store.add(docs)
        self._store.save(self.path)
        self._log(f"rebuilt with {n} documents")
        return n

    def add(self, docs: list[Document], source: str = "") -> int:
        """Incrementally merge new documents (Task 1: dynamic knowledge-base expansion)."""
        n = self.store.add(docs)
        if n:
            self.store.save(self.path)
            self._log(f"added {n} documents from {source or 'unknown source'}")
        return n

    def search(self, query: str, k: int | None = None, min_score: float | None = None) -> list[Hit]:
        hits = self.store.search(query, k or settings.top_k)
        threshold = settings.min_relevance if min_score is None else min_score
        return [h for h in hits if h.score >= threshold]

    def history(self) -> list[str]:
        log = self.path / "updates.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def _log(self, msg: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "updates.log").open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


@lru_cache(maxsize=8)
def get_kb(name: str) -> KnowledgeBase:
    return KnowledgeBase(name)
