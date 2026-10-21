"""Embedding back-ends.

* ``LocalEmbedder`` — FastEmbed (ONNX, CPU). Free, private, no rate limits; the default
  multilingual model lets a Hindi or Spanish question match an English FAQ.
* ``GeminiEmbedder`` — ``gemini-embedding-001`` with **batched** requests (up to 100 texts
  per call) and exponential back-off, instead of v1's one-request-per-text loop.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import settings


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


class LocalEmbedder:
    def __init__(self, model: str | None = None, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.name = f"local:{model or settings.local_embedding_model}"
        self._model = TextEmbedding(model or settings.local_embedding_model)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self._model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    BATCH = 100

    def __init__(self, model: str | None = None, dims: int = 768):
        self.model = model or settings.gemini_embedding_model
        self.dims = dims
        self.name = f"gemini:{self.model}:{dims}"

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import errors, types

        from .llm import client

        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dims)
        for attempt in range(6):
            try:
                resp = client().models.embed_content(model=self.model, contents=texts, config=cfg)
                return _normalize(np.array([e.values for e in resp.embeddings]))
            except errors.APIError as e:
                if getattr(e, "code", None) in (429, 500, 503) and attempt < 5:
                    time.sleep(min(60, 2 ** attempt * 5))
                    continue
                raise
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        parts = [self._call(texts[i:i + self.BATCH], "RETRIEVAL_DOCUMENT") for i in range(0, len(texts), self.BATCH)]
        return np.vstack(parts) if parts else np.zeros((0, self.dims), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=2)
def get_embedder(backend: str | None = None) -> Embedder:
    backend = (backend or settings.embedding_backend).lower()
    if backend == "gemini":
        return GeminiEmbedder()
    return LocalEmbedder()
