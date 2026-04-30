"""SWE-bench benchmark tools - Docker-based repo exploration and editing."""

import difflib
import shlex

from cast_agent.tools import (
    RequestSchema,
    ToolBudgetConfig,
    ToolLayer,
    ToolSet,
)
from examples.swebench.docker_env import SWEBenchEnv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _truncate_lines(text: str, max_lines: int, *, head_ratio: float = 0.7) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    head_n = int(max_lines * head_ratio)
    tail_n = max_lines - head_n
    elided = len(lines) - max_lines
    return (
        "".join(lines[:head_n])
        + f"\n[... {elided} lines truncated ...]\n"
        + "".join(lines[-tail_n:])
    )


def _find_closest_match(
    content: str, target: str, *, threshold: float = 0.5
) -> dict | None:
    target_lines = target.splitlines()
    content_lines = content.splitlines()
    n = len(target_lines)
    if n == 0 or not content_lines:
        return None

    best_ratio = 0.0
    best_start = 0
    for start in range(max(1, len(content_lines) - n + 1)):
        window = "\n".join(content_lines[start : start + n])
        ratio = difflib.SequenceMatcher(None, target, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start

    if best_ratio < threshold:
        return None

    matched_text = "\n".join(content_lines[best_start : best_start + n])
    return {"line": best_start + 1, "text": matched_text}


def _find_match_locations(content: str, target: str) -> list[int]:
    locations = []
    start = 0
    while True:
        idx = content.find(target, start)
        if idx == -1:
            break
        line_num = content[:idx].count("\n") + 1
        locations.append(line_num)
        start = idx + 1
    return locations


# ------------------------------------------------------------------
# RepoTools - ReAct (direct access)
# ------------------------------------------------------------------


class RepoTools(ToolSet):
    """SWE-bench tools for the ReAct loop."""

    KNOWN_TOOLS = frozenset(
        [
            "bash",
            "read_file",
            "search",
            "list_files",
            "write_file",
            "run_tests",
        ]
    )

    def __init__(self, env: SWEBenchEnv, max_output_chars: int = 16_000):
        self.env = env
        self.max_output = max_output_chars

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output:
            return text
        half = self.max_output // 2
        elided = len(text) - self.max_output
        return (
            text[:half]
            + f"\n\n[... {elided:,} characters elided ...]\n\n"
            + text[-half:]
        )

    def execute(self, name: str, args: dict) -> str:
        if name == "bash":
            command = args.get("command", "")
            if not command:
                return "ERROR: 'bash' requires a 'command' argument."
            out, rc = self.env.run(command)
            result = self._truncate(out)
            if rc != 0:
                result += f"\n[return code: {rc}]"
            return result

        if name == "read_file":
            path = args.get("path", "")
            if not path:
                return "ERROR: 'read_file' requires a 'path' argument."
            start = args.get("start")
            end = args.get("end")
            if start is not None and end is not None:
                start_i = int(start)
                end_i = int(end)
                if end_i < start_i:
                    return "ERROR: 'end' must be greater than or equal to 'start'."
                cmd = (
                    f"awk 'NR>={start_i} && NR<={end_i} "
                    f'{{printf "%4d| %s\\n", NR, $0}}\' {shlex.quote(path)}'
                )
            else:
                cmd = f"awk '{{printf \"%4d| %s\\n\", NR, $0}}' {shlex.quote(path)}"
            out, rc = self.env.run(cmd)
            if rc != 0:
                return f"ERROR reading {path}: {out}"
            return self._truncate(out)

        if name == "search":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            if not pattern:
                return "ERROR: 'search' requires a 'pattern' argument."
            cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null"
            out, _ = self.env.run(cmd)
            return (
                self._truncate(out) if out else f"No matches for '{pattern}' in {path}."
            )

        if name == "list_files":
            directory = args.get("directory", ".")
            cmd = f"ls -la {shlex.quote(directory)}"
            out, rc = self.env.run(cmd)
            if rc != 0:
                return f"ERROR listing {directory}: {out}"
            return self._truncate(out)

        if name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path:
                return "ERROR: 'write_file' requires a 'path' argument."
            out, rc = self.env.write_file(path, content)
            if rc != 0:
                return f"ERROR writing {path}: {out}"
            return f"Successfully wrote {path}."

        if name == "run_tests":
            command = args.get("command", "")
            if not command:
                return "ERROR: 'run_tests' requires a 'command' argument."
            out, rc = self.env.run(command, timeout=120)
            result = self._truncate(out)
            if rc == 0:
                result = f"[TESTS PASSED (rc=0)]\n{result}"
            else:
                result = f"[TESTS FAILED (rc={rc})]\n{result}"
            return result

        known = ", ".join(sorted(self.KNOWN_TOOLS))
        return f"Unknown tool: '{name}'. Available: {known}"

    def tool_descriptions(self) -> str:
        return (
            '1. bash - {"command": "..."} Run any bash command.\n'
            '2. read_file - {"path": "...", "start": N, "end": N} '
            "Read file with line numbers (start/end optional).\n"
            '3. search - {"pattern": "...", "path": "."} '
            "Grep for pattern in path (default: current dir).\n"
            '4. list_files - {"directory": "."} '
            "List directory contents.\n"
            '5. write_file - {"path": "...", "content": "..."} '
            "Create or overwrite a file.\n"
            '6. run_tests - {"command": "..."} '
            "Run a test command (120s timeout)."
        )

    def close(self) -> None:
        self.env.cleanup()


# ------------------------------------------------------------------
# RepoToolLayer - CAST (structured context access)
# ------------------------------------------------------------------


class RepoToolLayer(ToolLayer):
    """SWE-bench tool layer for the CAST loop."""

    KNOWN_TYPES = frozenset(
        [
            "file_tree",
            "read_file",
            "search",
            "find_files",
            "run_command",
            "write_file",
            "edit_file",
            "run_tests",
        ]
    )
    WRITE_TYPES = frozenset(["edit_file", "write_file"])
    TEST_TYPES = frozenset(["run_tests"])
    EXEC_TYPES = frozenset(["run_command"])

    REQUEST_SCHEMAS = {
        "file_tree": RequestSchema(
            optional={"directory": (str,), "max_depth": (int, float, str)}
        ),
        "read_file": RequestSchema(
            required={"path": (str,)},
            optional={
                "start_line": (int, float, str),
                "end_line": (int, float, str),
            },
        ),
        "search": RequestSchema(
            required={"pattern": (str,)},
            optional={"path": (str,), "context_lines": (int, float, str)},
        ),
        "find_files": RequestSchema(
            required={"pattern": (str,)},
            optional={"directory": (str,)},
        ),
        "run_command": RequestSchema(required={"command": (str,)}),
        "write_file": RequestSchema(required={"path": (str,), "content": (str,)}),
        "edit_file": RequestSchema(
            required={"path": (str,), "old_str": (str,), "new_str": (str,)}
        ),
        "run_tests": RequestSchema(required={"command": (str,)}),
    }

    def __init__(
        self,
        env: SWEBenchEnv,
        max_request_batches: int = 15,
        *,
        budget_config: ToolBudgetConfig | None = None,
        allow_json_repair: bool | None = None,
    ):
        super().__init__(
            max_request_batches=max_request_batches,
            budget_config=budget_config,
            allow_json_repair=allow_json_repair,
        )
        self.env = env

    def _dispatch(self, req_type: str, req: dict) -> str:
        if req_type == "file_tree":
            return self._file_tree(req)
        if req_type == "read_file":
            return self._read_file(req)
        if req_type == "search":
            return self._search(req)
        if req_type == "find_files":
            return self._find_files(req)
        if req_type == "run_command":
            return self._run_command(req)
        if req_type == "write_file":
            return self._write_file(req)
        if req_type == "edit_file":
            return self._edit_file(req)
        if req_type == "run_tests":
            return self._run_tests(req)

    # --- Read types ---

    def _file_tree(self, req: dict) -> str:
        directory = req.get("directory", ".")
        max_depth = min(int(req.get("max_depth", 3)), 5)
        cmd = (
            f"find {shlex.quote(directory)} -mindepth 1 -maxdepth {max_depth} "
            f"-not -path '*/.*' "
            f"2>/dev/null | sort | head -100"
        )
        out, _ = self.env.run(cmd)
        if not out:
            return f"No files found in {directory}."
        lines = out.splitlines()
        result = "\n".join(lines)
        if len(lines) >= 100:
            result += "\n[... truncated at 100 entries]"
        return result

    def _read_file(self, req: dict) -> str:
        path = req.get("path", "")
        if not path:
            return "ERROR: 'read_file' requires a 'path' argument."
        start_line = req.get("start_line")
        end_line = req.get("end_line")
        if start_line is not None and end_line is not None:
            start_line, end_line = int(start_line), int(end_line)
            if end_line - start_line + 1 > 200:
                end_line = start_line + 199
            cmd = (
                f"sed -n '{start_line},{end_line}p' {shlex.quote(path)} "
                f"| awk '{{printf \"%4d| %s\\n\", NR + {start_line - 1}, $0}}'"
            )
        else:
            cmd = f"awk '{{printf \"%4d| %s\\n\", NR, $0}}' {shlex.quote(path)}"
        out, rc = self.env.run(cmd)
        if rc != 0:
            return f"ERROR reading {path}: {out}"
        if not out:
            return f"File {path} is empty or does not exist."
        return _truncate_lines(out, 200)

    def _search(self, req: dict) -> str:
        pattern = req.get("pattern", "")
        path = req.get("path", ".")
        context = min(int(req.get("context_lines", 2)), 5)
        if not pattern:
            return "ERROR: 'search' requires a 'pattern' argument."
        cmd = (
            f"grep -rn -C {context} {shlex.quote(pattern)} {shlex.quote(path)} "
            f"2>/dev/null | head -300"
        )
        out, _ = self.env.run(cmd)
        if not out:
            return f"No matches for '{pattern}' in {path}."
        return _truncate_lines(out, 150)

    def _find_files(self, req: dict) -> str:
        pattern = req.get("pattern", "")
        directory = req.get("directory", ".")
        if not pattern:
            return "ERROR: 'find_files' requires a 'pattern' argument."
        cmd = (
            f"find {shlex.quote(directory)} "
            f"-name {shlex.quote(pattern)} "
            f"-not -path '*/\\.*' "
            f"2>/dev/null | head -50"
        )
        out, _ = self.env.run(cmd)
        if not out:
            return f"No files matching '{pattern}' in {directory}."
        lines = out.splitlines()
        result = "\n".join(lines)
        if len(lines) >= 50:
            result += "\n[... truncated at 50 paths]"
        return result

    def _run_command(self, req: dict) -> str:
        command = req.get("command", "")
        if not command:
            return "ERROR: 'run_command' requires a 'command' argument."
        out, rc = self.env.run(command)
        result = _truncate_lines(out, 200)
        if rc != 0:
            result = f"[return code: {rc}]\n{result}"
        return result

    # --- Write types ---

    def _write_file(self, req: dict) -> str:
        path = req.get("path", "")
        content = req.get("content", "")
        if not path:
            return "ERROR: 'write_file' requires a 'path' argument."
        mkdir_out, mkdir_rc = self.env.run(
            f'mkdir -p -- "$(dirname -- {shlex.quote(path)})"'
        )
        if mkdir_rc != 0:
            return f"ERROR creating parent directory for {path}: {mkdir_out}"
        out, rc = self.env.write_file(path, content)
        if rc != 0:
            return f"ERROR writing {path}: {out}"
        out2, _ = self.env.run(f"wc -l < {shlex.quote(path)}")
        return f"Wrote {path} ({out2.strip()} lines)."

    def _edit_file(self, req: dict) -> str:
        path = req.get("path", "")
        old_str = req.get("old_str", "")
        new_str = req.get("new_str", "")
        if not path:
            return "ERROR: 'edit_file' requires a 'path' argument."
        if not old_str:
            return "ERROR: 'edit_file' requires a non-empty 'old_str' argument."

        content_out, rc = self.env.run(f"cat {shlex.quote(path)}")
        if rc != 0:
            return f"ERROR reading {path}: {content_out}"
        if content_out and not content_out.endswith("\n"):
            content_out += "\n"

        count = content_out.count(old_str)
        if count == 0:
            best = _find_closest_match(content_out, old_str)
            if best:
                return (
                    f"ERROR: old_str not found in {path}.\n\n"
                    f"Closest match (line {best['line']}):\n{best['text']}\n\n"
                    "Check for whitespace, indentation, or content differences."
                )
            return f"ERROR: old_str not found in {path}."
        if count > 1:
            locations = _find_match_locations(content_out, old_str)
            loc_str = "\n".join(f"  - line {loc}" for loc in locations[:5])
            return (
                f"ERROR: old_str found {count} times in {path}:\n{loc_str}\n\n"
                "Include more surrounding lines in old_str to match exactly once."
            )

        new_content = content_out.replace(old_str, new_str, 1)
        out, rc = self.env.write_file(path, new_content)
        if rc != 0:
            return f"ERROR writing {path}: {out}"

        lines_before = content_out[: content_out.index(old_str)].count("\n") + 1
        new_lines = new_str.count("\n") + 1
        start = max(1, lines_before - 2)
        end = lines_before + new_lines + 2
        ctx_cmd = (
            f"awk '{{printf \"%4d| %s\\n\", NR, $0}}' {shlex.quote(path)} "
            f"| sed -n '{start},{end}p'"
        )
        ctx_out, _ = self.env.run(ctx_cmd)
        return f"Edited {path} (replaced 1 occurrence).\n\nContext:\n{ctx_out}"

    def _run_tests(self, req: dict) -> str:
        command = req.get("command", "")
        if not command:
            return "ERROR: 'run_tests' requires a 'command' argument."
        out, rc = self.env.run(command, timeout=120)
        result = _truncate_lines(out, 150)
        if rc == 0:
            header = "[TESTS PASSED (rc=0)]"
        else:
            header = f"[TESTS FAILED (rc={rc})]"
        return f"{header}\n{result}"

    # --- Descriptions ---

    def tool_descriptions(self) -> str:
        return (
            '1. file_tree - "directory": "." (optional), "max_depth": 3 (optional, max 5). '
            "Returns directory tree (max 100 entries).\n"
            '2. read_file - "path": "..." (required), "start_line": N, "end_line": N (optional). '
            "Returns file content with line numbers (max 200 lines).\n"
            '3. search - "pattern": "..." (required), "path": "." (optional), '
            '"context_lines": 2 (optional, max 5). '
            "Grep for pattern, returns matches with context (max 30 matches).\n"
            '4. find_files - "pattern": "..." (required), "directory": "." (optional). '
            "Find files by glob pattern (max 50 results).\n"
            '5. run_command - "command": "..." (required). '
            "Execute a bash command (max 200 lines output).\n"
            '6. write_file - "path": "..." (required), "content": "..." (required). '
            "Create or overwrite a file.\n"
            '7. edit_file - "path": "..." (required), "old_str": "..." (required), '
            '"new_str": "..." (required). '
            "Search-and-replace edit (old_str must match exactly once).\n"
            '8. run_tests - "command": "..." (required). '
            "Run a test command (120s timeout, max 150 lines output)."
        )

    def close(self) -> None:
        self.env.cleanup()
