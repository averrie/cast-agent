"""ReAct agent loop - direct tool access."""

from typing import Callable

from cast_agent.llm import LLMClient
from cast_agent.parsing import parse_tool_calls
from cast_agent.tools import ToolSet
from cast_agent.types import AgentResult
from cast_agent.loops._loop import run_loop


def run_react(
    question: str,
    tools: ToolSet,
    llm: LLMClient,
    system_prompt: str,
    max_turns: int = 20,
    on_turn: Callable[[dict], None] | None = None,
    no_compact: bool = False,
    keep_last: int = 6,
    answer_noun: str = "answer",
) -> AgentResult:
    all_tool_calls: list[dict] = []

    def try_dispatch(content: str, turn: int, max_turns: int) -> str | None:
        calls = parse_tool_calls(content)
        if not calls:
            return None
        results: list[str] = []
        for tc in calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            result = tools.execute(name, args if isinstance(args, dict) else {})
            all_tool_calls.append({"name": name, "args": args, "result": result})
            results.append(f"[{name}]\n{result}")
        return "\n\n".join(results)

    format_hint = (
        "Your response was not recognized. Use one of:\n\n"
        "To call a tool:\n"
        '  <tool_call>{"name": "...", "args": {...}}</tool_call>\n\n'
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
        no_compact=no_compact,
        keep_last=keep_last,
        answer_noun=answer_noun,
    )
    result.tool_calls = all_tool_calls
    return result
