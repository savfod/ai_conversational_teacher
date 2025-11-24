"""Text LLM calling helper.

This module provides a minimal wrapper to call OpenAI Chat Completions.
"""

from typing import Any, Iterable, TypeVar

import dotenv
import openai

from conversa.util.logs import log_function_duration

# todo: move to CLI, when it is ready
dotenv.load_dotenv()

# models
DEFAULT_STRUCTURED_MODEL = "gpt-4o-2024-08-06"
DEFAULT_TEXT_MODEL = "gpt-4o-mini-2024-07-18"

T = TypeVar("T")


def _check_history_format(history: Iterable[Any]) -> None:
    """Check that history items are in supported format.

    Raises:
        ValueError: If a history item has an unsupported shape.
    """
    for item in history:
        assert isinstance(item, dict)
        assert "role" in item and "content" in item
        assert item["role"] in ("user", "assistant")


@log_function_duration()
def call_llm_structured(
    query: str,
    sys_prompt: str,
    answer_format: type[T],
    model=DEFAULT_STRUCTURED_MODEL,
    history: Iterable[Any] = tuple(),
) -> T:
    """Return an LLM answer for `query`, parsed into a structured format.

    Args:
        query: The user's prompt/question.
        sys_prompt: System prompt.
        history: Optional list of prior messages. Each item must be a dict
            with keys 'role' and 'content'.
        model: The LLM model to use.
        answer_format: A type (e.g., dataclass or pydantic model) to parse the
            LLM response into.

    Returns:
        The parsed LLM response as an instance of `answer_format`.
    """

    client = openai.OpenAI()

    _check_history_format(history)
    messages = [
        {"role": "system", "content": sys_prompt},
        *history,
        {"role": "user", "content": query},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=answer_format,
    )

    answer = response.output_parsed
    assert isinstance(answer, answer_format)
    return answer


@log_function_duration()
def call_llm(
    query: str,
    sys_prompt: str,
    history: Iterable[Any] = tuple(),
    model: str = DEFAULT_TEXT_MODEL,
) -> str:
    """Return an LLM answer for `query`.

    Args:
        query: The user's prompt/question.
        sys_prompt: System prompt.
        history: Optional list of prior messages. Each item must be a dict
            with keys 'role' and 'content'.
        model: The LLM model to use.
    """
    _check_history_format(history)
    messages = [
        {"role": "system", "content": sys_prompt},
        *history,
        {"role": "user", "content": query},
    ]

    client = openai.OpenAI()
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=0,
    )

    content = resp.output_text
    assert isinstance(content, str)
    return content
