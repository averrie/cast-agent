"""BIRD text-to-SQL benchmark prompts."""

import re


ROOT_STUFFED_SYSTEM = "You output ONLY SQLite inside a ```sqlite code fence."

_SQL_FENCE_RE = re.compile(r"```(?:sqlite|sql)\s*\n(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    text = (text or "").strip()
    m = _SQL_FENCE_RE.search(text)
    if m:
        return m.group(1).strip().strip(";")
    return text.strip().strip(";")


ROOT_REACT_SYSTEM = """\
You are a text-to-SQL agent with direct database access. Given a natural language
question, explore the database and produce the correct SQLite query.

## Available tools

Call one or more tools per turn using this format:
  <tool_call>{{"name": "TOOL_NAME", "args": {{...}}}}</tool_call>

Tools:
{tool_descriptions}

Submit your final SQL answer:
  <final_answer>YOUR SQL QUERY HERE</final_answer>

## Database info
- Database: {db_name} ({table_count} tables)

## Context hints from the dataset
{context_hint}

## Workflow
1. Explore: list tables, inspect schemas, sample data to understand the database.
2. Plan: identify the tables, columns, and joins needed.
3. Draft SQL and test it with run_sql.
4. If results look wrong, iterate: check column values, fix joins, re-test.
5. Submit the final working SQL.

## Rules
- Output tool calls OR a final_answer, not both in the same turn.
- Use the exact column and table names from the schema.
- Test your SQL with run_sql before submitting.
- Return ONLY the columns explicitly requested. Do not add extra columns for readability.
- Do NOT concatenate columns (e.g., first_name || ' ' || last_name). Return each column separately as stored in the database.
- Do NOT re-call tools you have already called. Refer to previous results in the conversation.
"""

ROOT_CAST_SYSTEM = """\
You are a text-to-SQL agent. Given a natural language question about a database,
produce the correct SQLite query that answers it.

IMPORTANT: You do NOT have direct SQL access. You can only learn about the database
by requesting structured context.

## Available actions

Request context (returns results instantly):
  <request_context>[
    {{"type": "context_type", "key": "value", ...}},
    ...
  ]</request_context>

Submit your final SQL answer:
  <final_answer>YOUR SQL QUERY HERE</final_answer>

## Context request types

{tool_descriptions}

IMPORTANT: Arguments go at the TOP LEVEL of each request object, NOT nested under
an "arg" or "args" key. Correct: {{"type": "schema", "tables": ["t"]}}
Wrong: {{"type": "schema", "arg": {{"tables": ["t"]}}}}

Example request:
  <request_context>[
    {{"type": "list_tables"}},
    {{"type": "schema", "tables": ["users", "orders"]}},
    {{"type": "sample", "table": "users", "n": 3}},
    {{"type": "sample", "table": "orders", "where": "status = 'pending'", "order": "created_at DESC", "n": 3}},
    {{"type": "count", "table": "orders", "where": "status = 'pending'"}},
    {{"type": "join_path", "from_table": "users", "to_table": "orders"}}
  ]</request_context>

## Database info
- Database: {db_name} ({table_count} tables)
- You may submit up to {max_tool_batches} context request batches.
- You can include multiple requests in one batch - batch aggressively.

## Context hints from the dataset
{context_hint}

## Workflow
1. Start by requesting list_tables, then schema + sample for relevant tables - all in one batch.
2. Use join_path to understand how tables connect via foreign keys.
3. Verify your assumptions: use count, distinct, or describe to check column values,
   cardinality, and data types before writing SQL.
4. Write your SQL, then re-read the question to confirm:
   - Are you returning exactly the columns asked for?
   - Do JOINs require DISTINCT to avoid duplicate rows?
   - If the question mentions multiple columns 'X' and 'Y', use `IN ('X', 'Y')` not `= 'X and Y'`.
5. Submit your final SQL.

## Rules
- Output exactly ONE action per turn: either <request_context>...</request_context>
  or <final_answer>...</final_answer>.
- Batch multiple context requests in a single <request_context> tag when possible.
- Do NOT guess table or column names. Base your SQL only on confirmed context.
- Return ONLY the columns explicitly requested. Do not add extra columns.
- Do NOT concatenate columns (e.g., first_name || ' ' || last_name). Return each
  column separately as stored in the database.
- Do NOT re-request context you have already received. Refer to previous results.
- JOINs often produce duplicate rows. Default to DISTINCT unless you are certain
  the result set is unique.
- Request at least 2 context batches (schema + verification) before writing final SQL.
"""


def make_cast_prompt(
    db_name: str,
    table_count: int,
    context_hint: str = "",
    tool_descriptions: str = "",
    max_tool_batches: int = 5,
) -> str:
    return ROOT_CAST_SYSTEM.format(
        db_name=db_name,
        table_count=table_count,
        context_hint=context_hint or "(none)",
        tool_descriptions=tool_descriptions,
        max_tool_batches=max_tool_batches,
    )


def make_react_prompt(
    db_name: str,
    table_count: int,
    context_hint: str = "",
    tool_descriptions: str = "",
) -> str:
    return ROOT_REACT_SYSTEM.format(
        db_name=db_name,
        table_count=table_count,
        context_hint=context_hint or "(none)",
        tool_descriptions=tool_descriptions,
    )
