"""Stuffed baseline - single-turn prompt."""

import time
from typing import Callable

from cast_agent.llm import LLMClient
from cast_agent.types import AgentResult


def run_stuffed(
    question: str,
    llm: LLMClient,
    system_prompt: str,
    user_content: str | None = None,
    extract_answer: Callable[[str], str] | None = None,
) -> AgentResult:
    """Single-turn baseline: send system + user message, return answer.

    Parameters
    ----------
    question : str
        The question text.
    llm : LLMClient
        LLM client instance.
    system_prompt : str
        System prompt (benchmark-specific).
    user_content : str, optional
        Full user message. Defaults to ``f"Question: {question}"``.
    extract_answer : callable, optional
        Post-processor for the raw LLM output (e.g., SQL fence extractor).
        If None, the raw content is used as the answer.
    """
    t0 = time.time()

    if user_content is None:
        user_content = f"Question: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    content, usage = llm.chat(messages)
    answer = extract_answer(content) if extract_answer else content.strip()

    return AgentResult(
        final_answer=answer,
        messages=messages + [{"role": "assistant", "content": content}],
        total_tokens=llm.total_tokens,
        total_llm_calls=llm.total_calls,
        root_turns=1,
        wall_time=time.time() - t0,
        turn_token_log=[{"turn": 1, "agent": "root", **usage}],
    )
