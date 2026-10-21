from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest


class HashEmbedder:
    """Deterministic bag-of-words embedder for offline tests."""

    name = "test:hash"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, embedder):
    from csbot import knowledge
    from csbot.config import settings

    monkeypatch.setattr(settings, "index_dir", tmp_path / "indexes")
    monkeypatch.setattr(knowledge, "get_embedder", lambda *a, **k: embedder)
    knowledge.get_kb.cache_clear()
    yield
    knowledge.get_kb.cache_clear()
