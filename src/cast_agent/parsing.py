"""Parse structured XML-like tags from model output."""

import json
import re


def _tag_re(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)


_REQUEST_CONTEXT_RE = _tag_re("request_context")
_FINAL_ANSWER_RE = _tag_re("final_answer")
_TOOL_CALL_RE = _tag_re("tool_call")


def parse_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        return content if content else None
    return None


def parse_request_context(text: str) -> str | None:
    m = _REQUEST_CONTEXT_RE.search(text)
    if m:
        return m.group(1).strip() or None
    return None


def parse_final_answer(text: str) -> str | None:
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
        return ans if ans else None
    return None


def parse_tool_calls(text: str) -> list[dict]:
    results = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "name" in parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            pass
    return results
