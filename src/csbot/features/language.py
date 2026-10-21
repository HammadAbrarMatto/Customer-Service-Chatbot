"""Task 6 — automatic language detection (Lingua: offline and accurate on short text).

Replaces `langdetect`, which is unmaintained, nondeterministic by default, and no longer
builds on recent Python versions.
"""

from __future__ import annotations

import re
from functools import lru_cache

SUPPORTED = ["ENGLISH", "HINDI", "URDU", "BENGALI", "TAMIL", "TELUGU", "MARATHI", "GUJARATI", "PUNJABI",
             "SPANISH", "FRENCH", "GERMAN", "PORTUGUESE", "ITALIAN", "ARABIC", "CHINESE", "JAPANESE",
             "KOREAN", "RUSSIAN", "TURKISH", "INDONESIAN"]


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    langs = [getattr(Language, n) for n in SUPPORTED if hasattr(Language, n)]
    return LanguageDetectorBuilder.from_languages(*langs).with_minimum_relative_distance(0.1).build()


def detect_language(text: str, default: str = "English") -> str:
    """Return a friendly language name such as 'Hindi'. Very short / ambiguous text → default."""
    if len(re.sub(r"[\W\d_]+", "", text)) < 3:
        return default
    lang = _detector().detect_language_of(text)
    return lang.name.title() if lang else default


def language_instruction(language: str) -> str:
    if language == "English":
        return "Reply in English."
    return (f"The customer wrote in {language}. Reply ONLY in {language}, using natural, polite and culturally "
            f"appropriate wording for a {language} speaker. Keep product names and URLs unchanged.")
