"""Shared agent loop used by both ReAct and CAST modes."""

import time
from typing import Callable

from cast_agent.history import compact_history
from cast_agent.llm import LLMClient
from cast_agent.parsing import parse_final_answer
from cast_agent.types import AgentResult


def run_loop(
    question: str,
    llm: LLMClient,
    system_prompt: str,
    max_turns: int,
    *,
    try_dispatch: Callable[[str, int, int], str | None],
    format_hint: str,
    on_turn: Callable[[dict], None] | None = None,
    auto_submit: bool = False,
    no_compact: bool = False,
    keep_last: int = 6,
    answer_noun: str = "answer",
) -> AgentResult:
    """Core agent loop shared by run_react and run_cast.

    Parameters
    ----------
    try_dispatch : callable(content, turn, max_turns) -> str | None
        Attempt to extract and execute tool calls from the assistant
        response.  Return a user-message string if tools were dispatched,
        or ``None`` if the response contained no recognisable tool use.
    format_hint : str
        Message injected when the response contains neither a final
        answer nor a tool dispatch (the "not recognised" nudge).
    """
    t0 = time.time()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    turn_token_log: list[dict] = []
    trace_log: list[dict] = []

    for turn in range(max_turns):
        resp = llm.chat_full(
            messages if no_compact else compact_history(messages, keep_last=keep_last)
        )
        turn_token_log.append({"turn": turn + 1, "agent": "root", **resp.usage})
        messages.append(LLMClient.make_assistant_message(resp))

        trace_entry: dict = {"turn": turn + 1, "content": resp.content}
        if resp.reasoning:
            trace_entry["reasoning"] = resp.reasoning
        trace_log.append(trace_entry)

        if on_turn:
            on_turn({"turn": turn + 1, "agent": "root", "max_turns": max_turns})

        answer = parse_final_answer(resp.content)
        if answer is not None:
            return AgentResult(
                final_answer=answer,
                messages=messages,
                total_tokens=llm.total_tokens,
                total_llm_calls=llm.total_calls,
                root_turns=turn + 1,
                wall_time=time.time() - t0,
                turn_token_log=turn_token_log,
                trace_log=trace_log,
            )

        user_message = try_dispatch(resp.content, turn, max_turns)
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            continue

        # No tool dispatch and no final answer — nudge the model.
        remaining = max_turns - turn - 1
        if remaining <= 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "WARNING: This is your LAST turn. You MUST output "
                        f"<final_answer>YOUR {answer_noun.upper()}</final_answer> NOW."
                    ),
                }
            )
        elif remaining <= 3:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have only {remaining} turns left. Submit your {answer_noun} with "
                        f"<final_answer>YOUR {answer_noun.upper()}</final_answer>."
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": format_hint})

    # Turn budget exhausted.
    return AgentResult(
        final_answer="SUBMIT" if auto_submit else None,
        messages=messages,
        total_tokens=llm.total_tokens,
        total_llm_calls=llm.total_calls,
        root_turns=max_turns,
        wall_time=time.time() - t0,
        turn_token_log=turn_token_log,
        trace_log=trace_log,
    )
