"""Execution accuracy (EX) evaluation for text-to-SQL."""

import hashlib
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DANGEROUS_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|PRAGMA)\b",
    re.IGNORECASE,
)


def strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    sql = re.sub(r"--.*?$", "", sql, flags=re.M)
    return sql.strip()


def pick_single_statement(sql: str) -> str:
    sql = strip_sql_comments(sql).strip().strip(";")
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    for p in reversed(parts):
        if re.match(r"^(WITH|SELECT)\b", p, flags=re.IGNORECASE):
            return p
    return parts[-1]


def row_to_key(row: Tuple[Any, ...]) -> Tuple[Tuple[str, str], ...]:
    out = []
    for v in row:
        if v is None:
            out.append(("N", ""))
        elif isinstance(v, bytes):
            out.append(("B", v.hex()))
        elif isinstance(v, bool):
            out.append(("I", "1" if v else "0"))
        elif isinstance(v, int):
            out.append(("I", str(v)))
        elif isinstance(v, float):
            out.append(("F", f"{round(v, 6):.6f}"))
        else:
            out.append(("S", str(v)))
    return tuple(out)


def _cell_key(v: Any) -> bytes:
    if v is None:
        return b"N:"
    if isinstance(v, bytes):
        return b"B:" + v
    if isinstance(v, bool):
        return b"I:" + (b"1" if v else b"0")
    if isinstance(v, int):
        return b"I:" + str(v).encode("utf-8")
    if isinstance(v, float):
        return b"F:" + f"{round(v, 6):.6f}".encode("utf-8")
    return b"S:" + str(v).encode("utf-8", errors="replace")


def _row_hash64(row: Tuple[Any, ...]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for v in row:
        h.update(_cell_key(v))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little", signed=False)


@dataclass
class ExecResult:
    ok: bool
    rows: Optional[List[Tuple[Any, ...]]] = None
    sig: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def execute_sqlite_readonly(
    db_path: str,
    sql: str,
    timeout_s: float = 30.0,
    max_rows: int = 500_000,
    materialize_up_to: int = 50_000,
) -> ExecResult:
    sql = pick_single_statement(sql)
    if not sql:
        return ExecResult(False, error="Empty SQL after parsing")
    if DANGEROUS_SQL_RE.search(sql):
        return ExecResult(False, error="Dangerous/unsupported SQL blocked")

    con = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()
        try:
            cur.execute("PRAGMA query_only=1;")
            cur.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            pass

        deadline = time.time() + timeout_s

        def progress_handler():
            return 1 if time.time() > deadline else 0

        con.set_progress_handler(progress_handler, 10000)
        cur.execute(sql)

        rows: List[Tuple[Any, ...]] = []
        rowcount = 0
        xor64 = 0
        sum64 = 0
        sumsq64 = 0

        while True:
            batch = cur.fetchmany(2000)
            if not batch:
                break
            for r in batch:
                row = tuple(r)
                rowcount += 1
                if max_rows > 0 and rowcount > max_rows:
                    con.close()
                    return ExecResult(False, error=f"Too many rows (> {max_rows})")

                if rowcount <= materialize_up_to:
                    rows.append(row)
                else:
                    if rows:
                        rows = []

                h = _row_hash64(row)
                xor64 ^= h
                sum64 = (sum64 + h) & ((1 << 64) - 1)
                sumsq64 = (sumsq64 + (h * h & ((1 << 64) - 1))) & ((1 << 64) - 1)

        sig = {
            "rowcount": rowcount,
            "xor64": xor64,
            "sum64": sum64,
            "sumsq64": sumsq64,
            "materialized": rowcount <= materialize_up_to,
        }
        con.close()
        return ExecResult(True, rows=rows if sig["materialized"] else None, sig=sig)

    except Exception as e:
        if con:
            try:
                con.close()
            except Exception:
                pass
        return ExecResult(False, error=str(e))


def execution_accuracy(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout_s: float = 30.0,
    max_rows: int = 500_000,
    materialize_up_to: int = 50_000,
) -> tuple[bool, dict]:
    pred = execute_sqlite_readonly(
        db_path, pred_sql, timeout_s, max_rows, materialize_up_to
    )
    gold = execute_sqlite_readonly(
        db_path, gold_sql, timeout_s, max_rows, materialize_up_to
    )

    info = {
        "pred_ok": pred.ok,
        "gold_ok": gold.ok,
        "pred_error": pred.error,
        "gold_error": gold.error,
    }
    if not pred.ok or not gold.ok:
        return False, info

    if pred.rows is not None and gold.rows is not None:
        pred_ms = Counter(row_to_key(r) for r in pred.rows)
        gold_ms = Counter(row_to_key(r) for r in gold.rows)
        return (pred_ms == gold_ms), info

    return (pred.sig == gold.sig), info
