"""Process-isolated Python REPL for the compute tool type."""

import ast
import contextlib
import io
import multiprocessing as mp
import queue
import threading
import traceback
from typing import Any


_MAX_CODE_CHARS = 20_000
_FORBIDDEN_NAME_EXACT = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "open",
        "compile",
        "globals",
        "locals",
        "vars",
        "input",
        "getattr",
        "setattr",
        "delattr",
    }
)


def _safe_builtins() -> dict[str, Any]:
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _build_namespace() -> dict[str, Any]:
    import collections
    import decimal
    import fractions
    import math
    import re
    import statistics
    from collections import Counter, OrderedDict, defaultdict

    namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
    namespace.update(
        {
            "math": math,
            "statistics": statistics,
            "re": re,
            "decimal": decimal,
            "fractions": fractions,
            "collections": collections,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "OrderedDict": OrderedDict,
        }
    )
    try:
        import numpy as np  # type: ignore

        namespace["np"] = np
    except Exception:
        pass
    try:
        import scipy  # type: ignore
        import scipy.stats  # type: ignore  # noqa: F401

        namespace["scipy"] = scipy
    except Exception:
        pass
    return namespace


class _SafetyVisitor(ast.NodeVisitor):
    def _check_name(self, name: str) -> None:
        if name.startswith("__") or name in _FORBIDDEN_NAME_EXACT:
            raise ValueError(f"Forbidden name in compute code: {name}")

    def visit_Name(self, node: ast.Name) -> None:
        self._check_name(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError(f"Forbidden attribute in compute code: {node.attr}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("Import statements are not allowed in compute code.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("Import statements are not allowed in compute code.")


def _validate_user_code(code: str) -> None:
    if len(code) > _MAX_CODE_CHARS:
        raise ValueError(
            f"Compute code too long ({len(code)} chars). Max is {_MAX_CODE_CHARS}."
        )
    tree = ast.parse(code, mode="exec")
    _SafetyVisitor().visit(tree)


def _execute_in_namespace(code: str, namespace: dict[str, Any]) -> str:
    _validate_user_code(code)
    stdout_buf = io.StringIO()

    try:
        compiled_eval = compile(code, "<compute>", "eval")
    except SyntaxError:
        compiled_eval = None

    with contextlib.redirect_stdout(stdout_buf):
        if compiled_eval is not None:
            value = eval(compiled_eval, namespace)  # noqa: S307
            out = stdout_buf.getvalue()
            if value is not None:
                out += repr(value) if not out else f"\n{repr(value)}"
            return out or "(no output)"

        compiled_exec = compile(code, "<compute>", "exec")
        exec(compiled_exec, namespace)  # noqa: S102
        out = stdout_buf.getvalue()
        return out or "(no output)"


def _worker_main(request_q: Any, response_q: Any) -> None:
    namespace = _build_namespace()
    while True:
        msg = request_q.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")
        if op == "shutdown":
            return
        if op != "execute":
            continue

        code = msg.get("code", "")
        try:
            result = _execute_in_namespace(code, namespace)
            response_q.put({"ok": True, "result": result})
        except Exception:
            response_q.put({"ok": False, "error": traceback.format_exc()})


class SandboxedREPL:
    """Execute numerical Python snippets in a process-isolated namespace.

    The worker process persists namespace state across calls.
    On timeout, the worker is terminated and restarted.
    """

    TIMEOUT = 30  # seconds

    def __init__(self):
        self._lock = threading.Lock()
        start_method = mp.get_start_method(allow_none=True)
        if start_method is None:
            available = mp.get_all_start_methods()
            start_method = "fork" if "fork" in available else "spawn"
        self._ctx = mp.get_context(start_method)
        self._request_q: Any | None = None
        self._response_q: Any | None = None
        self._proc: Any | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        self._request_q = self._ctx.Queue()
        self._response_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._request_q, self._response_q),
            daemon=True,
        )
        self._proc.start()

    def _stop_worker(self) -> None:
        if self._proc is None:
            return

        try:
            if self._request_q is not None:
                self._request_q.put_nowait({"op": "shutdown"})
        except Exception:
            pass

        try:
            self._proc.join(timeout=0.5)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)

        for q in (self._request_q, self._response_q):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass

        self._proc = None
        self._request_q = None
        self._response_q = None

    def _restart_worker(self) -> None:
        self._stop_worker()
        self._start_worker()

    def execute(self, code: str) -> str:
        """Execute code and return stdout + last expression value."""
        if not isinstance(code, str):
            return "ERROR: compute code must be a string."

        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._restart_worker()

            try:
                self._request_q.put({"op": "execute", "code": code})
            except Exception as e:
                self._restart_worker()
                return f"ERROR: Failed to send code to compute worker: {e}"

            try:
                response = self._response_q.get(timeout=self.TIMEOUT)
            except queue.Empty:
                self._restart_worker()
                return f"ERROR: Code execution timed out after {self.TIMEOUT}s."
            except Exception as e:
                self._restart_worker()
                return f"ERROR: Compute worker failure: {e}"

            if not isinstance(response, dict):
                self._restart_worker()
                return "ERROR: Invalid response from compute worker."

            if response.get("ok"):
                return response.get("result", "(no output)") or "(no output)"

            return f"ERROR:\n{response.get('error', 'Unknown compute error')}"

    def reset(self) -> None:
        """Reset namespace between questions."""
        with self._lock:
            self._restart_worker()

    def close(self) -> None:
        """Terminate worker process."""
        with self._lock:
            self._stop_worker()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
