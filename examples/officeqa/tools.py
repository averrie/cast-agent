"""OfficeQA benchmark tools - Treasury Bulletin corpus and structured context layer."""

import re
import zipfile
from pathlib import Path

from cast_agent.tools import RequestSchema, ToolBudgetConfig, ToolLayer
from cast_agent.tools.repl import SandboxedREPL  # generic tool from core
from cast_agent.tools.web import WebSearchClient  # generic tool from core

# ------------------------------------------------------------------
# Truncation helper
# ------------------------------------------------------------------

OUTPUT_TRUNCATION = 25_000

_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt")


def _truncate(text: str, limit: int = OUTPUT_TRUNCATION) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [TRUNCATED at {limit:,} chars]"


# ------------------------------------------------------------------
# CorpusIndex - in-memory corpus for U.S. Treasury Bulletin text files
# ------------------------------------------------------------------


class CorpusIndex:
    """Load transformed Treasury Bulletin .txt files and provide search/read."""

    def __init__(self, source: str):
        """Load corpus from a zip file or a directory of .txt files."""
        self.docs: dict[str, str] = {}
        self.metadata: dict[str, dict] = {}

        src = Path(source)
        if src.is_file() and src.suffix == ".zip":
            self._load_from_zip(str(src))
        elif src.is_dir():
            self._load_from_dir(src)
        else:
            raise FileNotFoundError(f"Corpus source not found: {source}")

    def _load_from_zip(self, zip_path: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".txt"):
                    continue
                content = z.read(name).decode("utf-8", errors="replace")
                fname = Path(name).name
                self.docs[fname] = content
                self.metadata[fname] = self._parse_metadata(fname, content)

    def _load_from_dir(self, dir_path: Path) -> None:
        for p in sorted(dir_path.glob("*.txt")):
            content = p.read_text(encoding="utf-8", errors="replace")
            self.docs[p.name] = content
            self.metadata[p.name] = self._parse_metadata(p.name, content)

    @staticmethod
    def _parse_metadata(filename: str, content: str) -> dict:
        m = _FILENAME_RE.match(filename)
        year = int(m.group(1)) if m else 0
        month = int(m.group(2)) if m else 0
        lines = content.split("\n")
        return {
            "year": year,
            "month": month,
            "size": len(content),
            "line_count": len(lines),
        }

    def list_documents(
        self, year_start: int | None = None, year_end: int | None = None
    ) -> str:
        results = []
        for fname in sorted(self.docs):
            meta = self.metadata[fname]
            if year_start and meta["year"] < year_start:
                continue
            if year_end and meta["year"] > year_end:
                continue
            results.append(
                f"{fname}  (year={meta['year']}, month={meta['month']:02d}, "
                f"lines={meta['line_count']}, size={meta['size']:,} chars)"
            )
        header = f"{len(results)} document(s) found"
        if year_start or year_end:
            header += f" (filter: {year_start or '...'}-{year_end or '...'})"
        return _truncate(header + "\n" + "\n".join(results))

    def search(
        self,
        query: str,
        max_results: int = 30,
        case_sensitive: bool = False,
        files: list[str] | None = None,
    ) -> str:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

        hits: list[str] = []
        target_files = files if files else sorted(self.docs)
        for fname in target_files:
            if fname not in self.docs:
                continue
            lines = self.docs[fname].split("\n")
            for i, line in enumerate(lines):
                if pattern.search(line):
                    ctx_start = max(0, i - 1)
                    ctx_end = min(len(lines), i + 2)
                    context = "\n".join(
                        f"  {'>' if j == i else ' '} {j+1}: {lines[j]}"
                        for j in range(ctx_start, ctx_end)
                    )
                    hits.append(f"{fname}:{i+1}\n{context}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break

        if not hits:
            return f"No matches found for: {query}"
        header = f"{len(hits)} match(es) for: {query}"
        if len(hits) >= max_results:
            header += f" (capped at {max_results})"
        return _truncate(header + "\n\n" + "\n\n".join(hits))

    def read_document(
        self, filename: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        if filename not in self.docs:
            avail = [f for f in self.docs if filename.lower() in f.lower()]
            if avail:
                return (
                    f"File not found: {filename}. Did you mean: {', '.join(avail[:5])}?"
                )
            return f"File not found: {filename}"

        content = self.docs[filename]
        if start_line is not None or end_line is not None:
            lines = content.split("\n")
            s = (start_line or 1) - 1
            e = end_line or len(lines)
            selected = lines[s:e]
            header = f"{filename} (lines {s+1}-{min(e, len(lines))} of {len(lines)})"
            return _truncate(header + "\n" + "\n".join(selected))

        header = f"{filename} ({self.metadata[filename]['line_count']} lines)"
        return _truncate(header + "\n" + content)

    def document_info(self, filename: str, preview_lines: int = 20) -> str:
        if filename not in self.docs:
            return f"File not found: {filename}"
        meta = self.metadata[filename]
        lines = self.docs[filename].split("\n")
        preview = "\n".join(lines[:preview_lines])
        return (
            f"File: {filename}\n"
            f"Year: {meta['year']}, Month: {meta['month']:02d}\n"
            f"Size: {meta['size']:,} chars, {meta['line_count']} lines\n"
            f"\n--- First {min(preview_lines, len(lines))} lines ---\n"
            f"{preview}"
        )


# ------------------------------------------------------------------
# OfficeQAToolLayer - structured context layer for the CAST loop
# ------------------------------------------------------------------


class OfficeQAToolLayer(ToolLayer):
    """Tool layer for exploring a Treasury Bulletin text corpus."""

    MAX_BATCH_RESULT_CHARS = 25_000  # match paper's 25K char output truncation
    EXEC_TYPES = frozenset(["compute"])

    KNOWN_TYPES = frozenset(
        [
            "list_documents",
            "search",
            "read_document",
            "document_info",
            "compute",
            "web_search",
        ]
    )
    REQUEST_SCHEMAS = {
        "list_documents": RequestSchema(
            optional={"year_start": (int, float, str), "year_end": (int, float, str)}
        ),
        "search": RequestSchema(
            required={"query": (str,)},
            optional={
                "max_results": (int, float, str),
                "case_sensitive": (bool,),
                "files": (list, str),
            },
        ),
        "read_document": RequestSchema(
            required={"filename": (str,)},
            optional={
                "start_line": (int, float, str),
                "end_line": (int, float, str),
            },
        ),
        "document_info": RequestSchema(
            required={"filename": (str,)},
            optional={"preview_lines": (int, float, str)},
        ),
        "compute": RequestSchema(required={"code": (str,)}),
        "web_search": RequestSchema(
            required={"query": (str,)},
            optional={"max_results": (int, float, str)},
        ),
    }

    def __init__(
        self,
        corpus: CorpusIndex,
        max_request_batches: int = 200,
        *,
        budget_config: ToolBudgetConfig | None = None,
        allow_json_repair: bool | None = None,
    ):
        super().__init__(
            max_request_batches=max_request_batches,
            budget_config=budget_config,
            allow_json_repair=allow_json_repair,
        )
        self.corpus = corpus
        self.repl = SandboxedREPL()
        self.web = WebSearchClient()

    def _dispatch(self, req_type: str, req: dict) -> str:
        if req_type == "list_documents":
            year_start = req.get("year_start")
            year_end = req.get("year_end")
            if year_start is not None:
                year_start = int(year_start)
            if year_end is not None:
                year_end = int(year_end)
            return self.corpus.list_documents(year_start, year_end)

        if req_type == "search":
            query = req.get("query", "")
            if not query:
                return "ERROR: 'search' requires a 'query' argument."
            max_results = int(req.get("max_results", 30))
            case_sensitive = bool(req.get("case_sensitive", False))
            files = req.get("files")
            if isinstance(files, str):
                files = [files]
            return self.corpus.search(query, max_results, case_sensitive, files)

        if req_type == "read_document":
            filename = req.get("filename", "")
            if not filename:
                return "ERROR: 'read_document' requires a 'filename' argument."
            start_line = req.get("start_line")
            end_line = req.get("end_line")
            if start_line is not None:
                start_line = int(start_line)
            if end_line is not None:
                end_line = int(end_line)
            return self.corpus.read_document(filename, start_line, end_line)

        if req_type == "document_info":
            filename = req.get("filename", "")
            if not filename:
                return "ERROR: 'document_info' requires a 'filename' argument."
            preview_lines = int(req.get("preview_lines", 20))
            return self.corpus.document_info(filename, preview_lines)

        if req_type == "compute":
            code = req.get("code", "")
            if not code:
                return "ERROR: 'compute' requires a 'code' argument."
            result = self.repl.execute(code)
            return _truncate(result)

        if req_type == "web_search":
            query = req.get("query", "")
            if not query:
                return "ERROR: 'web_search' requires a 'query' argument."
            max_results = int(req.get("max_results", 5))
            return _truncate(self.web.search(query, max_results))

    def tool_descriptions(self) -> str:
        return (
            '1. list_documents - "year_start": 1940 (optional), "year_end": 1950 (optional). '
            "Lists all document files, optionally filtered by year range.\n"
            '2. search - "query": "national defense" (required), "max_results": 30 (optional, default 30), '
            '"case_sensitive": false (optional), "files": ["file1.txt"] (optional, search specific files). '
            "Searches for a keyword or regex pattern across documents. Returns matching lines with context.\n"
            '3. read_document - "filename": "treasury_bulletin_1941_01.txt" (required), '
            '"start_line": 100 (optional), "end_line": 200 (optional). '
            "Reads a document or a line range from it.\n"
            '4. document_info - "filename": "treasury_bulletin_1941_01.txt" (required), '
            '"preview_lines": 20 (optional). '
            "Returns metadata (year, month, size, line count) and a preview.\n"
            '5. compute - "code": "math.sqrt(2)" (required). '
            "Executes Python code in a persistent sandbox with math, statistics, numpy, scipy "
            "pre-imported. Import statements are disabled; call the preloaded modules directly. "
            "Use for calculations, data processing, regressions, etc. "
            "The namespace persists across calls so you can build up intermediate results.\n"
            '6. web_search - "query": "US CPI 1940 annual average" (required), '
            '"max_results": 5 (optional). '
            "Searches the web for external information not in the corpus "
            "(e.g., CPI data, exchange rates, BLS statistics)."
        )

    def reset_repl(self) -> None:
        """Reset the Python REPL between questions."""
        self.repl.reset()

    def close(self) -> None:
        try:
            self.repl.close()
        except Exception:
            pass

        ddgs = getattr(self.web, "_ddgs", None)
        close_fn = getattr(ddgs, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
