"""Tool abstractions for the CAST agent framework.

ToolSet - imperative tool dispatch interface (for ReAct loop)
ToolLayer - structured batch-JSON dispatch interface (for CAST loop)
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cast_agent.tools.json_repair import repair_json


# ------------------------------------------------------------------
# ToolSet - imperative interface (ReAct)
# ------------------------------------------------------------------


class ToolSet(ABC):
    """Interface for direct tool calling (used by run_react)."""

    @abstractmethod
    def execute(self, name: str, args: dict) -> str:
        """Dispatch a named tool call and return the string result."""

    @abstractmethod
    def tool_descriptions(self) -> str:
        """Return a text block listing available tools for inclusion in prompts."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""


# ------------------------------------------------------------------
# ToolLayer - structured batch interface (CAST)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSchema:
    """Runtime schema for a request type."""

    required: dict[str, tuple[type, ...]] = field(default_factory=dict)
    optional: dict[str, tuple[type, ...]] = field(default_factory=dict)
    allow_extra: bool = False


@dataclass(frozen=True)
class ToolBudgetConfig:
    """Per-category batch budgets for CAST tool requests."""

    read_batches: int | None = 5
    write_batches: int | None = None
    test_batches: int | None = None
    exec_batches: int | None = None

    def limit_for(self, category: str) -> int | None:
        if category == "read":
            return self.read_batches
        if category == "write":
            return self.write_batches
        if category == "test":
            return self.test_batches
        if category == "exec":
            return self.exec_batches
        return None


class ToolLayer(ABC):
    """Interface for structured JSON-batch context access (used by run_cast)."""

    KNOWN_TYPES: frozenset[str] = frozenset()
    WRITE_TYPES: frozenset[str] = frozenset()
    TEST_TYPES: frozenset[str] = frozenset()
    EXEC_TYPES: frozenset[str] = frozenset()

    REQUEST_SCHEMAS: dict[str, RequestSchema] = {}
    ALLOW_JSON_REPAIR: bool = False
    MAX_BATCH_RESULT_CHARS: int | None = None

    def __init__(
        self,
        max_request_batches: int = 5,
        *,
        budget_config: ToolBudgetConfig | None = None,
        allow_json_repair: bool | None = None,
    ):
        self.budget_config = (
            budget_config
            if budget_config is not None
            else ToolBudgetConfig(read_batches=max_request_batches)
        )
        self.max_request_batches = (
            self.budget_config.read_batches
            if self.budget_config.read_batches is not None
            else max_request_batches
        )
        self.allow_json_repair = (
            self.ALLOW_JSON_REPAIR if allow_json_repair is None else allow_json_repair
        )
        self._budget_usage = {
            "read": 0,
            "write": 0,
            "test": 0,
            "exec": 0,
        }
        self.total_requests = 0

    @property
    def batch_count(self) -> int:
        """Backward-compatible alias for read-batch usage."""
        return self._budget_usage["read"]

    def execute_batch(self, raw_json: str) -> str:
        requests_or_error = self._parse_requests(raw_json)
        if isinstance(requests_or_error, str):
            return requests_or_error
        requests = requests_or_error

        categories = self._batch_categories(requests)
        budget_error = self._check_budget(categories)
        if budget_error is not None:
            return budget_error
        self._consume_budget(categories)

        results: list[str] = []
        for i, req in enumerate(requests):
            if not isinstance(req, dict):
                results.append(f"[request {i + 1}] ERROR: Not a JSON object.")
                continue

            req_type = req.get("type", "")
            if not isinstance(req_type, str) or not req_type:
                results.append(f"[request {i + 1}] ERROR: Missing 'type' field.")
                continue

            validation_error = self._validate_request(req_type, req)
            if validation_error is not None:
                results.append(f"[{req_type}] ERROR: {validation_error}")
                continue

            self.total_requests += 1
            try:
                result = self._dispatch(req_type, req)
                results.append(f"[{req_type}]\n{result}")
            except Exception as e:
                results.append(f"[{req_type}] ERROR: {e}")

        header = f"--- {self._batch_label(categories)} ({len(requests)} request(s)) ---"
        footer = self._build_footer()

        body = header + "\n\n" + "\n\n".join(results) + "\n\n" + footer
        if self.MAX_BATCH_RESULT_CHARS and len(body) > self.MAX_BATCH_RESULT_CHARS:
            body = (
                body[: self.MAX_BATCH_RESULT_CHARS]
                + f"\n\n... [BATCH TRUNCATED at {self.MAX_BATCH_RESULT_CHARS:,} chars - "
                "request fewer items per batch or use narrower line ranges]"
            )
        return body

    def _parse_requests(self, raw_json: str) -> list | str:
        try:
            requests = json.loads(raw_json)
        except json.JSONDecodeError as e:
            if not self.allow_json_repair:
                return (
                    f"ERROR: Could not parse context request as JSON (strict mode): {e}\n\n"
                    "Your request must be a JSON array of objects inside "
                    "<request_context>...</request_context>.\n"
                    "Example:\n"
                    '<request_context>[{"type": "list_tables"}]</request_context>'
                )

            repaired = repair_json(raw_json)
            if repaired is None:
                return (
                    f"ERROR: Could not parse context request as JSON: {e}\n\n"
                    "JSON repair fallback was enabled but could not recover your request.\n"
                    "Your request must be a JSON array of objects inside "
                    "<request_context>...</request_context>."
                )
            requests = repaired

        if isinstance(requests, dict):
            requests = [requests]
        if not isinstance(requests, list):
            return "ERROR: Context request must be a JSON array of objects."
        return requests

    def _batch_categories(self, requests: list) -> set[str]:
        categories: set[str] = set()
        for req in requests:
            if not isinstance(req, dict):
                categories.add("read")
                continue
            req_type = req.get("type")
            if not isinstance(req_type, str) or not req_type:
                categories.add("read")
                continue
            categories.add(self._request_category(req_type))
        return categories or {"read"}

    def _request_category(self, req_type: str) -> str:
        if req_type in self.TEST_TYPES:
            return "test"
        if req_type in self.EXEC_TYPES:
            return "exec"
        if req_type in self.WRITE_TYPES:
            return "write"
        return "read"

    def _check_budget(self, categories: set[str]) -> str | None:
        for category in sorted(categories):
            limit = self.budget_config.limit_for(category)
            if limit is None:
                continue
            used = self._budget_usage[category]
            if used < limit:
                continue

            if category == "read":
                return (
                    f"Context request limit reached ({limit}/{limit}). You MUST now "
                    "submit your final answer based on the context you have. Submit with "
                    "<final_answer>YOUR ANSWER</final_answer>."
                )
            return (
                f"{category.title()} request limit reached ({limit}/{limit}). "
                f"Submit fewer {category} requests and continue."
            )
        return None

    def _consume_budget(self, categories: set[str]) -> None:
        for category in categories:
            self._budget_usage[category] += 1

    def _batch_label(self, categories: set[str]) -> str:
        if categories == {"read"}:
            return f"Context Batch #{self.batch_count}"
        labels = ", ".join(sorted(categories))
        return f"Context Batch ({labels})"

    def _build_footer(self) -> str:
        remaining = self._remaining_batches("read")
        if remaining is None:
            return "--- read context batches: unlimited ---"

        footer = f"--- {remaining} context batch(es) remaining ---"
        limit = self.budget_config.read_batches
        if limit is None:
            return footer

        threshold = max(int(limit * 0.2), 1)
        if remaining == threshold and remaining > 0:
            footer += (
                f"\nWARNING: You have {remaining} exploration batch(es) left. "
                "Start finalizing your answer now."
            )
        return footer

    def _remaining_batches(self, category: str) -> int | None:
        limit = self.budget_config.limit_for(category)
        if limit is None:
            return None
        return max(limit - self._budget_usage[category], 0)

    def _validate_request(self, req_type: str, req: dict) -> str | None:
        if self.KNOWN_TYPES and req_type not in self.KNOWN_TYPES:
            known = ", ".join(sorted(self.KNOWN_TYPES))
            return f"Unknown context type: '{req_type}'. Available: {known}"

        schema = self.REQUEST_SCHEMAS.get(req_type)
        if schema is None:
            return None

        missing = [name for name in schema.required if name not in req]
        if missing:
            return f"Missing required field(s): {', '.join(sorted(missing))}"

        expected = {**schema.required, **schema.optional}
        for field_name, allowed_types in expected.items():
            if field_name not in req:
                continue
            value = req[field_name]
            if not isinstance(value, allowed_types):
                expected_names = ", ".join(t.__name__ for t in allowed_types)
                got_name = type(value).__name__
                return (
                    f"Invalid type for '{field_name}': expected {expected_names}, "
                    f"got {got_name}"
                )

        if not schema.allow_extra:
            allowed = set(expected) | {"type"}
            extras = sorted(k for k in req if k not in allowed)
            if extras:
                return (
                    f"Unexpected field(s): {', '.join(extras)}. "
                    "Arguments must be top-level fields defined by this request type."
                )
        return None

    @abstractmethod
    def _dispatch(self, req_type: str, req: dict) -> str:
        """Execute a single context request."""

    @abstractmethod
    def tool_descriptions(self) -> str:
        """Return a text block describing available context request types."""

    def close(self) -> None:
        pass
