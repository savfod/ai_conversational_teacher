"""Tests for aiteacher.generated.llm.

These tests monkeypatch a fake `openai` module to avoid network calls.
"""
import sys
import types


def test_answer_with_fake_openai(monkeypatch):
    dummy = types.ModuleType("openai")

    def create(model, messages, temperature, max_tokens):
        # simple assertion to ensure the wrapper passed expected model
        assert model == "gpt-3.5-turbo"
        # ensure our user message is present
        assert any(m.get("role") == "user" and m.get("content") == "hi" for m in messages)
        return {"choices": [{"message": {"content": "Hello back"}}]}

    # Attach ChatCompletion with a create method
    dummy.ChatCompletion = types.SimpleNamespace(create=create)
    dummy.api_key = None

    monkeypatch.setitem(sys.modules, "openai", dummy)

    from aiteacher.generated import llm

    res = llm.answer("hi", history=[("user", "previous")])
    assert isinstance(res, str)
    assert "Hello back" in res
