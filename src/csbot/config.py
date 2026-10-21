"""All settings in one place. Override any of them in `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gemini-3.6-flash"))
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "gemini-3.6-flash"))

    # "local" = FastEmbed on CPU (free, no rate limits) · "gemini" = gemini-embedding-001 API
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "local"))
    local_embedding_model: str = field(default_factory=lambda: _env("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL))
    gemini_embedding_model: str = field(default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))

    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(PROJECT_ROOT / "indexes"))))

    top_k: int = 4
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.35")))
    max_history_turns: int = 6

    @property
    def faq_csv(self) -> Path:
        return self.data_dir / "faq.csv"

    @property
    def medquad_csv(self) -> Path:
        return self.data_dir / "medquad_prepared.csv"

    def index_path(self, name: str) -> Path:
        return self.index_dir / name


settings = Settings()
