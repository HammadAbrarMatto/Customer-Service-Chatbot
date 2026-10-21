"""Task 2 — multimodal chat about an uploaded image (Gemini vision)."""

from __future__ import annotations

from collections.abc import Iterator

from .. import llm
from ..config import settings

SYSTEM = ("You are a customer-support assistant looking at a photo a customer uploaded (e.g. a damaged product, "
          "an error screen, a receipt). Describe what is relevant, then answer their question or suggest next steps.")


def ask_about_image(image, question: str = "") -> Iterator[str]:
    """`image` is a PIL.Image; the google-genai SDK accepts it directly in `contents`."""
    prompt = question.strip() or "Describe this image in detail and point out anything a support agent should notice."
    return llm.stream([image, prompt], system=SYSTEM, model=settings.vision_model)
