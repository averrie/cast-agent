"""JSON repair helpers for malformed LLM output."""

import json
import re


def try_parse(s: str) -> list[dict] | None:
    try:
        result = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return result
    return None


def repair_json(raw: str) -> list[dict] | None:
    s = raw.strip()

    if s.startswith("{"):
        result = try_parse(s)
        if result is not None:
            return result

    fixed = s.replace('\\\\"', '"')
    if fixed != s:
        result = try_parse(fixed)
        if result is not None:
            return result

    fixed = re.sub(
        r'(?<=": ")(.*?)(?=")',
        lambda m: m.group(0).replace("\n", "\\n"),
        s,
        flags=re.DOTALL,
    )
    if fixed != s:
        result = try_parse(fixed)
        if result is not None:
            return result

    fixed = re.sub(r",\s*([}\]])", r"\1", s)
    if fixed != s:
        result = try_parse(fixed)
        if result is not None:
            return result

    fixed = s
    for _ in range(5):
        try:
            json.loads(fixed)
            break
        except json.JSONDecodeError as e:
            if "delimiter" not in e.msg:
                break
            if e.pos < len(fixed) and fixed[e.pos] == ":":
                fixed = fixed[: e.pos] + "," + fixed[e.pos + 1 :]
            else:
                break
    if fixed != s:
        result = try_parse(fixed)
        if result is not None:
            return result

    result = extract_string_value_requests(s)
    if result is not None:
        return result

    return None


def extract_json_string_field(raw: str, key: str) -> str | None:
    m = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw,
        flags=re.DOTALL,
    )
    if not m:
        return None
    val = m.group(1)
    try:
        return json.loads(f'"{val}"')
    except (json.JSONDecodeError, ValueError):
        return val.replace("\\n", "\n").replace('\\"', '"')


def extract_string_value_requests(raw: str) -> list[dict] | None:
    results = []
    depth = 0
    obj_start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_str = raw[obj_start : i + 1]
                parsed = parse_single_request(obj_str)
                if parsed is not None:
                    results.append(parsed)
                obj_start = None
    return results if results else None


def parse_single_request(obj_str: str) -> dict | None:
    try:
        obj = json.loads(obj_str)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', obj_str)
    if not type_match:
        return None
    req_type = type_match.group(1)
    result = {"type": req_type}

    if req_type in ("run_command", "run_tests"):
        command = extract_json_string_field(obj_str, "command")
        if command is not None:
            result["command"] = command
            return result

    elif req_type == "write_file":
        path = extract_json_string_field(obj_str, "path")
        if path is not None:
            result["path"] = path
        content = extract_json_string_field(obj_str, "content")
        if content is not None:
            result["content"] = content
            return result

    elif req_type == "edit_file":
        path = extract_json_string_field(obj_str, "path")
        if path is not None:
            result["path"] = path
        for key in ("old_str", "new_str"):
            val = extract_json_string_field(obj_str, key)
            if val is not None:
                result[key] = val
        if "old_str" in result:
            return result

    return None
