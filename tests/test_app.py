"""Headless UI test with the LLM and embeddings faked."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    from csbot import llm
    from csbot.config import settings

    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(llm, "stream", lambda prompt, system=None, model=None: iter(["Yes, ", "we do!"]))
    monkeypatch.setattr(llm, "generate", lambda prompt, system=None, model=None: "generated")


def test_chat_page_answers_and_records_history():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.chat_input[0].set_value("Do you provide internships?").run()
    assert not at.exception
    assert [t.role for t in at.session_state.chat] == ["user", "assistant"]
    assert at.session_state.chat[1].content == "Yes, we do!"


@pytest.mark.parametrize("page", ["📚 Knowledge base", "🖼️ Image chat", "🩺 Medical Q&A", "📄 arXiv expert"])
def test_other_pages_render(page):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
