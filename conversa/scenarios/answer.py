from conversa.features.llm_api import call_llm

SYSTEM_PROMPT: str = """
You are a language teacher and a helpful assistant. You answer according to the following rules:
- Answer the user's input with a relevant question to keep the conversation going.
- Ignore the word "start" in the beginning of the conversation and the word "stop" at the end (they are just to start and stop the recording).
- Answer in the same language as the users input.
"""


def teacher_answer(
    query: str,
) -> str:
    """Return a language teacher LLM answer for `query`.

    Args:
        query: The user's prompt/question.

    Returns:
        A string containing the language teacher's answer.
    """
    return call_llm(
        query=query,
        sys_prompt=SYSTEM_PROMPT,
    )
