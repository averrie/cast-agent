"""Conversation history compaction for token efficiency."""


def compact_history(
    messages: list[dict],
    keep_last: int = 6,
) -> list[dict]:
    if len(messages) <= keep_last + 2:
        return messages

    compact_cutoff = len(messages) - keep_last
    result: list[dict] = []

    for i, msg in enumerate(messages):
        if i < 2:
            result.append(msg)
        elif i >= compact_cutoff:
            result.append(msg)
        elif msg["role"] == "assistant":
            if "reasoning_details" in msg:
                result.append(
                    {k: v for k, v in msg.items() if k != "reasoning_details"}
                )
            else:
                result.append(msg)
        elif msg["role"] == "user":
            compacted = _compact_message(msg["content"])
            if compacted != msg["content"]:
                result.append({"role": "user", "content": compacted})
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def _compact_message(content: str) -> str:
    if content.startswith("--- Context Batch"):
        return _compact_context_batch(content)
    if content.startswith("["):
        return _compact_tool_message(content)
    return content


def _compact_context_batch(content: str) -> str:
    lines = content.split("\n")
    header = lines[0] if lines else ""
    footer = ""
    footer_idx = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith("---") and "remaining" in lines[idx]:
            footer_idx = idx
            footer = "\n".join(lines[idx:])
            break

    blocks: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    for line in lines[1:footer_idx]:
        if line.startswith("[") and line.endswith("]"):
            if current_type is not None:
                blocks.append((current_type, current_lines))
            current_type = line
            current_lines = []
        elif current_type is not None:
            if line.strip():
                current_lines.append(line)

    if current_type is not None:
        blocks.append((current_type, current_lines))

    tag = header.replace(" ---", " [COMPACTED] ---", 1)
    parts = [tag]
    for type_header, body_lines in blocks:
        if not body_lines:
            parts.append(type_header)
        elif len(body_lines) <= 3:
            parts.append(type_header + "\n" + "\n".join(body_lines))
        else:
            preview = body_lines[0]
            parts.append(f"{type_header} {preview} (+ {len(body_lines) - 1} lines)")
    parts.append(footer)
    return "\n".join(parts)


def _compact_tool_message(content: str, min_lines_to_truncate: int = 10) -> str:
    if not content.startswith("["):
        return content

    blocks = content.split("\n\n")
    compacted: list[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if lines[0].startswith("[") and lines[0].endswith("]"):
            header = lines[0]
            body_lines = lines[1:]
            if len(body_lines) <= min_lines_to_truncate:
                compacted.append(block)
            else:
                preview = "\n".join(body_lines[:2])
                compacted.append(
                    f"{header}\n{preview}\n... ({len(body_lines)} lines, truncated)"
                )
        else:
            compacted.append(block)

    return "\n\n".join(compacted)
