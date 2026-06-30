"""
task6_multilanguage.py
----------------------
INTERNSHIP TASK 6
"Extend the chatbot to support at least three additional languages. Detect the
user's language automatically, switch between languages seamlessly, and provide
culturally appropriate responses."

What this does, in plain words:
- Automatically DETECTS what language the user typed in.
- Lets the chatbot REPLY in that same language (or a language the user picks).
- We support detection for many languages out of the box; the README lists the
  three+ we test (Hindi, Spanish, French) plus English.

How detection works:
- We use langdetect, a small offline library, to guess the language code.
- For the actual translation/answering, Gemini already understands and writes
  many languages, so we just instruct it which language to reply in.
"""

from langdetect import detect, DetectorFactory

# Make detection results repeatable (langdetect is random by default).
DetectorFactory.seed = 0

# Map language codes to friendly names. We focus on these but Gemini handles more.
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "ja": "Japanese",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the text and return a friendly name.

    Falls back to "English" if detection fails (for example on very short text).
    """
    try:
        code = detect(text)
    except Exception:
        return "English"
    return LANGUAGE_NAMES.get(code, "English")


def language_instruction(language_name: str) -> str:
    """
    Build a short instruction telling Gemini which language to reply in, and to
    keep the reply natural and culturally appropriate.
    """
    return (
        f"Reply ONLY in {language_name}. Use natural, polite, and culturally "
        f"appropriate wording for a {language_name} speaker. Do not mix languages."
    )


# Manual test:  python tasks/task6_multilanguage.py
if __name__ == "__main__":
    samples = [
        "Hello, how are you?",
        "Hola, necesito ayuda con mi pedido.",
        "Bonjour, je voudrais une information sur ma commande.",
        "नमस्ते, मुझे मेरे ऑर्डर के बारे में मदद चाहिए।",
    ]
    for s in samples:
        print(f"{detect_language(s):10s} <- {s}")
