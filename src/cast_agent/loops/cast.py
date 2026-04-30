"""CAST - root agent with structured tool access."""

from typing import Callable

from cast_agent.tools import ToolLayer
from cast_agent.llm import LLMClient
from cast_agent.parsing import parse_request_context
from cast_agent.types import AgentResult
from cast_agent.loops._loop import run_loop


def run_cast(
    question: str,
    tool_layer: ToolLayer,
    llm: LLMClient,
    system_prompt: str,
    max_turns: int = 20,
    on_turn: Callable[[dict], None] | None = None,
    auto_submit: bool = False,
    no_compact: bool = False,
    keep_last: int = 6,
    answer_noun: str = "SQL",
) -> AgentResult:
    context_log: list[dict] = []

    def try_dispatch(content: str, turn: int, max_turns: int) -> str | None:
        context_raw = parse_request_context(content)
        if context_raw is None:
            return None

        result_text = tool_layer.execute_batch(context_raw)

        remaining_turns = max_turns - turn - 1
        turn_threshold = max(int(max_turns * 0.2), 1)
        if remaining_turns == turn_threshold and remaining_turns > 0:
            result_text += (
                f"\n\n\u26a0 TURN WARNING: You have {remaining_turns} turns "
                f"remaining. Finalize your {answer_noun} and submit with "
                f"<final_answer>YOUR {answer_noun.upper()}</final_answer>."
            )

        context_log.append(
            {
                "batch": tool_layer.batch_count,
                "raw_request": context_raw,
                "result": result_text,
            }
        )
        return result_text

    format_hint = (
        "Your response was not recognized. Use one of:\n\n"
        "To request context:\n"
        '  <request_context>[{"type": "...", ...}]</request_context>\n\n'
        f"To submit your final {answer_noun}:\n"
        f"  <final_answer>YOUR {answer_noun.upper()}</final_answer>"
    )

    result = run_loop(
        question,
        llm,
        system_prompt,
        max_turns,
        try_dispatch=try_dispatch,
        format_hint=format_hint,
        on_turn=on_turn,
        auto_submit=auto_submit,
        no_compact=no_compact,
        keep_last=keep_last,
        answer_noun=answer_noun,
    )
    result.context_packets = context_log
    return result
