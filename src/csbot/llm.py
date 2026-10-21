"""Gemini access through the official `google-genai` SDK."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from .config import settings


class MissingAPIKey(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "No Gemini API key found. Add GEMINI_API_KEY=... to a .env file in the project root "
            "(free key: https://aistudio.google.com/app/apikey)."
        )


@lru_cache(maxsize=1)
def client():
    if not settings.api_key:
        raise MissingAPIKey()
    from google import genai

    return genai.Client(api_key=settings.api_key)


def _config(system: str | None):
    from google.genai import types

    return types.GenerateContentConfig(system_instruction=system) if system else None


def generate(prompt: Any, system: str | None = None, model: str | None = None) -> str:
    resp = client().models.generate_content(model=model or settings.chat_model, contents=prompt,
                                            config=_config(system))
    return (resp.text or "").strip()


def stream(prompt: Any, system: str | None = None, model: str | None = None) -> Iterator[str]:
    for chunk in client().models.generate_content_stream(model=model or settings.chat_model, contents=prompt,
                                                         config=_config(system)):
        if chunk.text:
            yield chunk.text
