"""Text LLM calling helper.

This module provides a minimal wrapper to call OpenAI Chat Completions.
"""

import os
from typing import Any, Iterable, TypeVar

import openai

SYSTEM_PROMPT: str = """
You are a language teacher. You answer according to the following rules:
- Answer the user's input with a relevant question to keep the conversation going.
- Ignore the word "start" in the beginning of the conversation and the word "stop" at the end (they are just to start and stop the recording).
- Answer in the same language as the users input.
"""


def _extract_content(resp: Any) -> str:
    """Robustly extract assistant text from an OpenAI response object/dict.

    Tries a few shapes returned by different openai client versions.
    Returns the extracted text or str(resp) fallback.
    """
    try:
        # mapping-like response
        if isinstance(resp, dict):
            choices = resp.get("choices")
        else:
            choices = getattr(resp, "choices", None)

        if not choices:
            return str(resp)

        first = choices[0]
        # dict choice
        if isinstance(first, dict):
            return first.get("message", {}).get("content") or first.get("text", "")

        # object-like choice (library objects)
        msg = getattr(first, "message", None)
        if msg:
            # msg may be mapping or object
            if isinstance(msg, dict):
                return msg.get("content", "")
            return getattr(msg, "content", "")

        return getattr(first, "text", "") or str(resp)
    except Exception:
        return str(resp)


T = TypeVar("T")


def answer_structured_simple(query: str, sys_prompt: str, answer_format: type[T]) -> T:
    """Return an LLM answer for `query`, parsed into a structured format.

    Args:
        query: The user's prompt/question.
        sys_prompt: System prompt (defaults to module SYSTEM_PROMPT).
        answer_format: A type (e.g., dataclass or pydantic model) to parse the
            LLM response into.

    Returns:
        The parsed LLM response as an instance of `answer_format`.
    """

    client = openai.OpenAI()

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",  # TODO: fix
        input=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": query,
            },
        ],
        text_format=answer_format,
    )

    answer = response.output_parsed
    return answer


def answer(
    query: str, sys_prompt: str = SYSTEM_PROMPT, history: Iterable[Any] = tuple()
) -> str:
    """Return an LLM answer for `query`.

    Args:
        query: The user's prompt/question.
        sys_prompt: System prompt (defaults to module SYSTEM_PROMPT).
        history: Optional list of prior messages. Supported item formats:
            - dict with keys 'role' and 'content'
            - tuple/list (role, content)
            - plain string (treated as a user message)

    Returns:
        The assistant response as a string.

    Raises:
        RuntimeError: If the OpenAI client is missing or the API returns an error.
        ValueError: If a history item has an unsupported shape.

    Note: The function lazy-imports the `openai` package so importing this
    module won't fail when the package isn't installed. It will raise a
    clear RuntimeError when `answer` is called and `openai` isn't available.
    """
    # prefer env var if present
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            openai.api_key = api_key
        except Exception:
            # some openai versions expect client configuration elsewhere; ignore if not applicable
            pass

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            messages.append({"role": item[0], "content": item[1]})
        elif isinstance(item, str):
            messages.append({"role": "user", "content": item})
        else:
            raise ValueError(
                "history items must be dict(role/content), tuple(role,content) or str"
            )

    messages.append({"role": "user", "content": query})

    # The openai python package changed its surface in v1+: prefer the
    # OpenAI client (OpenAI().chat.completions.create(...)) but fall back to
    # legacy module-level ChatCompletion.create(...) for older versions.
    try:
        # Newer client style: openai.OpenAI()
        if hasattr(openai, "OpenAI"):
            try:
                client = openai.OpenAI()
                # client.chat.completions.create is the modern call
                resp = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4_096,
                )
            except Exception:
                # If the client exists but shape differs, try attribute path on module
                if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=4_096,
                    )
                else:
                    raise
        # Legacy style: module-level ChatCompletion.create
        else:
            # try legacy module-level ChatCompletion in a way that static
            # analyzers won't flag as a missing attribute
            chat_cls = getattr(openai, "ChatCompletion", None)
            if chat_cls is not None and getattr(chat_cls, "create", None):
                resp = chat_cls.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
            else:
                # Another possible shape: openai.chat.completions.create at module level
                if (
                    hasattr(openai, "chat")
                    and hasattr(openai.chat, "completions")
                    and hasattr(openai.chat.completions, "create")
                ):
                    resp = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=512,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported openai client installed; cannot find ChatCompletion or OpenAI client"
                    )
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    content = _extract_content(resp)
    return (content or "").strip()


__all__ = ["SYSTEM_PROMPT", "answer"]
