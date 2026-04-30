"""BIRD benchmark tools -- SQLite access and structured context layer."""

import sqlite3

from cast_agent.tools import RequestSchema, ToolBudgetConfig, ToolLayer, ToolSet


# ------------------------------------------------------------------
# DBTools - read-only SQLite exploration (ReAct)
# ------------------------------------------------------------------


class DBTools(ToolSet):
    def __init__(self, db_path: str, max_rows: int = 5, max_queries: int = 50):
        self.db_path = db_path
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.execute("PRAGMA query_only = 1;")
        self.max_rows = max_rows
        self.max_queries = max_queries
        self.query_count = 0

    def list_tables(self) -> str:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cur.fetchall()]
        return "\n".join(tables) if tables else "(no tables)"

    def get_schema(self, table: str) -> str:
        cur = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        row = cur.fetchone()
        return row[0] if row else f"Table '{table}' not found"

    def sample_rows(self, table: str, n: int = 5) -> str:
        n = max(1, min(n, self.max_rows))
        try:
            cur = self.conn.execute(f"SELECT * FROM [{table}] LIMIT ?", (n,))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            header = " | ".join(cols)
            lines = [header, "-" * len(header)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)
        except Exception as e:
            return f"Error sampling '{table}': {e}"

    def run_sql(self, query: str) -> str:
        self.query_count += 1
        if self.query_count > self.max_queries:
            return "ERROR: Query budget exceeded. Finalize your answer."
        try:
            cur = self.conn.execute(query)
            if cur.description is None:
                return "(query executed, no result set)"
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(self.max_rows)
            header = " | ".join(cols)
            lines = [header, "-" * len(header)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            note = ""
            if len(rows) == self.max_rows:
                note = f"\n(output truncated at {self.max_rows} rows)"
            return "\n".join(lines) + note
        except Exception as e:
            return f"SQL ERROR: {e}"

    def get_foreign_keys(self) -> str:
        fks: list[str] = []
        for table in self._table_names():
            try:
                cur = self.conn.execute(f"PRAGMA foreign_key_list([{table}])")
                for row in cur.fetchall():
                    fks.append(f"{table}.{row[3]} -> {row[2]}.{row[4]}")
            except Exception:
                pass
        return "\n".join(fks) if fks else "(no foreign keys found)"

    def execute(self, name: str, args: dict) -> str:
        if name == "list_tables":
            return self.list_tables()
        if name == "get_schema":
            return self.get_schema(args.get("table", ""))
        if name == "sample_rows":
            return self.sample_rows(args.get("table", ""), args.get("n", 5))
        if name == "run_sql":
            return self.run_sql(args.get("query", ""))
        if name == "get_foreign_keys":
            return self.get_foreign_keys()
        return f"Unknown tool: {name}"

    def tool_descriptions(self) -> str:
        return (
            "- list_tables: {} - returns all table names in the database\n"
            '- get_schema: {"table": "name"} - returns the CREATE TABLE statement\n'
            '- sample_rows: {"table": "name", "n": 5} - returns sample rows\n'
            '- run_sql: {"query": "SELECT ..."} - executes SQL and returns results '
            f"(capped at {self.max_rows} rows)\n"
            "- get_foreign_keys: {} - returns all foreign key relationships"
        )

    def _table_names(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


# ------------------------------------------------------------------
# BIRDToolLayer - structured context layer (CAST)
# ------------------------------------------------------------------


class BIRDToolLayer(ToolLayer):

    KNOWN_TYPES = frozenset(
        [
            "list_tables",
            "schema",
            "sample",
            "distinct",
            "count",
            "join_path",
            "describe",
        ]
    )
    REQUEST_SCHEMAS = {
        "list_tables": RequestSchema(),
        "schema": RequestSchema(required={"tables": (list, str)}),
        "sample": RequestSchema(
            required={"table": (str,)},
            optional={
                "n": (int, float, str),
                "where": (str,),
                "order": (str,),
            },
        ),
        "distinct": RequestSchema(
            required={"table": (str,), "column": (str,)},
            optional={"limit": (int, float, str)},
        ),
        "count": RequestSchema(
            required={"table": (str,)},
            optional={"where": (str,)},
        ),
        "join_path": RequestSchema(
            required={"from_table": (str,), "to_table": (str,)},
        ),
        "describe": RequestSchema(
            required={"table": (str,), "column": (str,)},
        ),
    }

    def __init__(
        self,
        db_tools: DBTools,
        max_request_batches: int = 5,
        *,
        budget_config: ToolBudgetConfig | None = None,
        allow_json_repair: bool | None = None,
    ):
        super().__init__(
            max_request_batches=max_request_batches,
            budget_config=budget_config,
            allow_json_repair=allow_json_repair,
        )
        self.db = db_tools

    def _dispatch(self, req_type: str, req: dict) -> str:
        if req_type == "list_tables":
            return self.db.list_tables()

        if req_type == "schema":
            tables = req.get("tables", [])
            if isinstance(tables, str):
                tables = [tables]
            if not tables:
                return "ERROR: 'schema' requires a 'tables' list."
            parts = [self.db.get_schema(t) for t in tables]
            return "\n\n".join(parts)

        if req_type == "sample":
            table = req.get("table", "")
            n = req.get("n", 5)
            where = req.get("where", "")
            order = req.get("order", "")
            if not table:
                return "ERROR: 'sample' requires a 'table' argument."
            if where or order:
                sql = f"SELECT * FROM [{table}]"
                if where:
                    sql += f" WHERE {where}"
                if order:
                    sql += f" ORDER BY {order}"
                sql += f" LIMIT {int(n)}"
                return self.db.run_sql(sql)
            return self.db.sample_rows(table, int(n))

        if req_type == "distinct":
            table = req.get("table", "")
            column = req.get("column", "")
            limit = min(int(req.get("limit", 20)), 50)
            if not table or not column:
                return "ERROR: 'distinct' requires 'table' and 'column'."
            sql = f"SELECT DISTINCT [{column}] FROM [{table}] LIMIT {limit}"
            return self.db.run_sql(sql)

        if req_type == "count":
            table = req.get("table", "")
            where = req.get("where", "")
            if not table:
                return "ERROR: 'count' requires a 'table' argument."
            sql = f"SELECT COUNT(*) FROM [{table}]"
            if where:
                sql += f" WHERE {where}"
            return self.db.run_sql(sql)

        if req_type == "join_path":
            from_table = req.get("from_table", "")
            to_table = req.get("to_table", "")
            if not from_table or not to_table:
                return "ERROR: 'join_path' requires 'from_table' and 'to_table'."
            fks = self.db.get_foreign_keys()
            parts = [
                f"All foreign key relationships:\n{fks}",
                f"\nSchema for {from_table}:\n{self.db.get_schema(from_table)}",
                f"\nSchema for {to_table}:\n{self.db.get_schema(to_table)}",
            ]
            return "\n".join(parts)

        if req_type == "describe":
            table = req.get("table", "")
            column = req.get("column", "")
            if not table or not column:
                return "ERROR: 'describe' requires 'table' and 'column'."
            parts = [
                f"Schema:\n{self.db.get_schema(table)}",
                f"Null analysis:\n"
                + self.db.run_sql(
                    f"SELECT COUNT(*) AS total, "
                    f"SUM(CASE WHEN [{column}] IS NULL THEN 1 ELSE 0 END) AS nulls "
                    f"FROM [{table}]"
                ),
                f"Distinct count:\n"
                + self.db.run_sql(
                    f"SELECT COUNT(DISTINCT [{column}]) AS distinct_count "
                    f"FROM [{table}]"
                ),
                f"Min/Max:\n"
                + self.db.run_sql(
                    f"SELECT MIN([{column}]) AS min_val, "
                    f"MAX([{column}]) AS max_val FROM [{table}]"
                ),
            ]
            return "\n\n".join(parts)

    def tool_descriptions(self) -> str:
        return (
            "1. list_tables - No args. Returns all table names.\n"
            '2. schema - "tables": ["t1","t2"] (required). Returns CREATE TABLE DDL.\n'
            '3. sample - "table": "t" (required), "n": 5 (optional, default 5), '
            '"where": "col = \'x\'" (optional), "order": "col ASC" (optional). '
            "Returns sample rows, optionally filtered and/or sorted.\n"
            '4. distinct - "table": "t" (required), "column": "c" (required), '
            '"limit": 20 (optional, default 20, max 50). Returns unique values.\n'
            '5. count - "table": "t" (required), "where": "col > 5" (optional). '
            "Returns row count.\n"
            '6. join_path - "from_table": "t1" (required), "to_table": "t2" (required). '
            "Returns FK relationships and schemas for both tables.\n"
            '7. describe - "table": "t" (required), "column": "c" (required). '
            "Returns type, null%, cardinality, min/max."
        )

    def close(self) -> None:
        pass
