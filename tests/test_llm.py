"""Tests for conversa.generated.llm.

These tests monkeypatch a fake `openai` module to avoid network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from conversa.features import llm_api


@patch("conversa.features.llm_api.openai")
def test_answer_with_fake_openai(mock_openai, monkeypatch):
    def create(model, input, temperature):
        # simple assertion to ensure the wrapper passed expected model
        assert "gpt" in model
        # ensure our user message is present
        assert any(m.get("role") == "user" and m.get("content") == "hi" for m in input)
        res = MagicMock()
        res.output_text = "Hello back"
        return res

    # Attach ChatCompletion with a create method
    mock_openai.OpenAI().responses.create.side_effect = create

    res = llm_api.call_llm(
        "hi",
        history=[{"role": "user", "content": "previous"}],
        sys_prompt="You are a helpful assistant.",
    )
    assert isinstance(res, str)
    assert "Hello back" in res


@pytest.mark.slow
def test_answer_with_real_openai():
    """Test the llm_api.answer function with a real OpenAI call."""
    res = llm_api.call_llm(
        "Hi! What is the value of 2+2?", sys_prompt="You are a helpful assistant."
    )
    assert isinstance(res, str)
    assert len(res) > 0
    assert "4" in res
