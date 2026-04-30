"""Shared data structures for the CAST agent."""

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """Result of running the root agent on a single question."""

    final_answer: str | None = None
    context_packets: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_llm_calls: int = 0
    root_turns: int = 0
    wall_time: float = 0.0
    turn_token_log: list[dict] = field(default_factory=list)
    trace_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "context_packets": self.context_packets,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls,
            "root_turns": self.root_turns,
            "wall_time": self.wall_time,
            "turn_token_log": self.turn_token_log,
            "trace_log": self.trace_log,
        }
