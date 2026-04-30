#!/usr/bin/env python3
"""CAST-BIRD evaluation harness - run stuffed/react/cast on BIRD dev subsets."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from examples.bird.tools import DBTools, BIRDToolLayer
from examples.bird.evaluation import (
    execution_accuracy,
    execute_sqlite_readonly,
    pick_single_statement,
)
from cast_agent.harness_utils import (
    add_llm_cli_args,
    append_jsonl,
    call_with_retry,
    collect_sampling_kwargs,
    load_checkpoint_ids,
    load_dotenv_if_available,
    resolve_thinking_flag,
)
from cast_agent.llm import LLMClient
from examples.bird.prompts import (
    ROOT_STUFFED_SYSTEM,
    extract_sql,
    make_cast_prompt,
    make_react_prompt,
)
from examples.bird.schema import (
    DevTablesIndex,
    column_meanings_from_dev_tables,
    load_column_meanings_from_db_desc,
    schema_from_dev_tables,
)
from cast_agent.loops import run_stuffed, run_react, run_cast
from cast_agent.tools import ToolBudgetConfig


load_dotenv_if_available()


# ----------------------------
# Database path resolution
# ----------------------------


def find_db_path(db_root: str, db_id: str) -> str:
    db_dir = os.path.join(db_root, db_id)
    for ext in (".sqlite", ".sqlite3", ".db"):
        candidate = os.path.join(db_dir, f"{db_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    if os.path.isdir(db_dir):
        for f in os.listdir(db_dir):
            if f.endswith((".sqlite", ".sqlite3", ".db")):
                return os.path.join(db_dir, f)
    raise FileNotFoundError(f"No SQLite database found for {db_id} in {db_dir}")


def count_tables(db_path: str) -> int:
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    n = cur.fetchone()[0]
    con.close()
    return n


# ----------------------------
# Per-question worker
# ----------------------------


def eval_one(
    example: dict,
    mode: str,
    db_root: str,
    dev_tables: DevTablesIndex,
    llm_kwargs: dict,
    max_turns: int,
    max_tool_batches: int,
    save_trace: bool,
    allow_json_repair: bool = False,
    no_compact: bool = False,
) -> dict:
    qid = str(example["question_id"])
    db_id = example["db_id"]
    question = example["question"]
    gold_sql = example["SQL"]
    context_hint = example.get("evidence", "")
    difficulty = example.get("difficulty", "")

    db_path = find_db_path(db_root, db_id)
    table_count = count_tables(db_path)

    def _new_llm() -> LLMClient:
        return LLMClient(**llm_kwargs)

    result = None
    try:
        if mode == "stuffed":
            rec = dev_tables.get(db_id)

            def _run_stuffed_once():
                llm = _new_llm()
                if rec:
                    col_meanings = column_meanings_from_dev_tables(rec)
                    csv_meanings = load_column_meanings_from_db_desc(db_root, db_id)
                    col_meanings.update(csv_meanings)
                    schema_text = schema_from_dev_tables(rec, col_meanings)
                else:
                    tools = DBTools(db_path)
                    try:
                        tables = tools.list_tables().split("\n")
                        schema_text = "\n\n".join(
                            tools.get_schema(t) for t in tables if t.strip()
                        )
                    finally:
                        tools.close()

                prompt_parts = [
                    f"Database: {db_id}\n",
                    f"Schema:\n{schema_text}\n",
                ]
                if context_hint:
                    prompt_parts.append(f"Hint: {context_hint}\n")
                prompt_parts.append(f"Question: {question}")
                user_content = "\n".join(prompt_parts)

                return run_stuffed(
                    question,
                    llm,
                    ROOT_STUFFED_SYSTEM,
                    user_content=user_content,
                    extract_answer=extract_sql,
                )

            result = call_with_retry(_run_stuffed_once)

        elif mode == "react":

            def _run_react_once():
                llm = _new_llm()
                tools = DBTools(db_path)
                system_prompt = make_react_prompt(
                    db_name=db_id,
                    table_count=table_count,
                    context_hint=context_hint,
                    tool_descriptions=tools.tool_descriptions(),
                )
                try:
                    return run_react(
                        question,
                        tools,
                        llm,
                        system_prompt,
                        max_turns=max_turns,
                        no_compact=no_compact,
                        answer_noun="SQL",
                    )
                finally:
                    tools.close()

            result = call_with_retry(_run_react_once)

        elif mode == "cast":

            def _run_cast_once():
                llm = _new_llm()
                tools = DBTools(db_path)
                tool_layer = BIRDToolLayer(
                    tools,
                    max_request_batches=max_tool_batches,
                    budget_config=ToolBudgetConfig(read_batches=max_tool_batches),
                    allow_json_repair=allow_json_repair,
                )
                system_prompt = make_cast_prompt(
                    db_name=db_id,
                    table_count=table_count,
                    context_hint=context_hint,
                    tool_descriptions=tool_layer.tool_descriptions(),
                    max_tool_batches=max_tool_batches,
                )
                try:
                    return run_cast(
                        question,
                        tool_layer,
                        llm,
                        system_prompt,
                        max_turns=max_turns,
                        auto_submit=True,
                        no_compact=no_compact,
                    )
                finally:
                    tools.close()

            result = call_with_retry(_run_cast_once)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        return {
            "question_id": qid,
            "db_id": db_id,
            "error": str(e),
            "pred_sql": "",
            "gold_sql": gold_sql,
            "ex_correct": False,
            "difficulty": difficulty,
        }

    pred_sql = result.final_answer or ""
    pred_sql = pick_single_statement(pred_sql)

    try:
        correct, ex_info = execution_accuracy(db_path, pred_sql, gold_sql)
    except Exception as e:
        correct = False
        ex_info = {"error": str(e)}

    record = {
        "question_id": qid,
        "db_id": db_id,
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "ex_correct": correct,
        "difficulty": difficulty,
        "total_tokens": result.total_tokens,
        "total_llm_calls": result.total_llm_calls,
        "root_turns": result.root_turns,
        "wall_time": round(result.wall_time, 2),
    }

    if save_trace:
        record["trace_log"] = result.trace_log
        record["context_packets"] = result.context_packets
        record["tool_calls"] = result.tool_calls
        record["ex_info"] = ex_info
        pred_exec = execute_sqlite_readonly(db_path, pred_sql)
        gold_exec = execute_sqlite_readonly(db_path, gold_sql)
        if pred_exec.rows is not None:
            record["pred_output"] = [list(r) for r in pred_exec.rows[:20]]
        if gold_exec.rows is not None:
            record["gold_output"] = [list(r) for r in gold_exec.rows[:20]]

    return record


# ----------------------------
# Main harness
# ----------------------------


def main():
    parser = argparse.ArgumentParser(description="CAST-BIRD evaluation harness")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["stuffed", "react", "cast"],
        help="Agent mode: stuffed (baseline), react (tool-calling), cast (structured context)",
    )
    parser.add_argument(
        "--gold", type=str, required=True, help="Path to dev.json (BIRD format)"
    )
    parser.add_argument(
        "--db_root", type=str, required=True, help="Path to dev_databases/ directory"
    )
    parser.add_argument(
        "--tables_json",
        type=str,
        default="",
        help="Path to dev_tables.json (for stuffed baseline schema)",
    )
    add_llm_cli_args(
        parser,
        include_provider=True,
        include_sampling=True,
        include_max_tokens=False,
    )
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_tool_batches", type=int, default=5)
    parser.add_argument(
        "--allow_json_repair",
        action="store_true",
        help="Enable JSON repair fallback for malformed CAST request payloads.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to N questions (0 = all)"
    )
    parser.add_argument(
        "--question_ids",
        type=str,
        default="",
        help="Comma-separated question IDs to run",
    )
    parser.add_argument(
        "--save_trace", action="store_true", help="Include full traces in JSONL output"
    )
    parser.add_argument(
        "--no_compact", action="store_true", help="Disable history compaction"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL filename (default: bird_{mode}.jsonl)",
    )
    args = parser.parse_args()

    resolve_thinking_flag(args)
    llm_kwargs = {
        "model": args.model,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "provider": args.provider,
        "base_url": args.base_url or None,
        "api_key": args.api_key or None,
        **collect_sampling_kwargs(args),
    }

    with open(args.gold, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions from {args.gold}")

    if args.question_ids:
        target_ids = set(args.question_ids.split(","))
        dataset = [ex for ex in dataset if str(ex["question_id"]) in target_ids]
        print(f"Filtered to {len(dataset)} questions")

    if args.limit > 0:
        dataset = dataset[: args.limit]

    dev_tables = DevTablesIndex(args.tables_json)

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_name = args.output if args.output else f"bird_{args.mode}.jsonl"
    jsonl_path = os.path.join(args.output_dir, jsonl_name)

    completed_ids = load_checkpoint_ids(jsonl_path, id_field="question_id")
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    remaining = [ex for ex in dataset if str(ex["question_id"]) not in completed_ids]
    print(f"Running {len(remaining)} questions ({args.mode})")

    if not remaining:
        print("Nothing to do.")
    else:
        with open(jsonl_path, "a", encoding="utf-8") as jsonl_f:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        eval_one,
                        ex,
                        args.mode,
                        args.db_root,
                        dev_tables,
                        llm_kwargs,
                        args.max_turns,
                        args.max_tool_batches,
                        args.save_trace,
                        args.allow_json_repair,
                        args.no_compact,
                    ): ex
                    for ex in remaining
                }

                correct = 0
                total = 0
                pbar = tqdm(
                    as_completed(futures), total=len(futures), desc="Evaluating"
                )
                for future in pbar:
                    try:
                        record = future.result()
                    except Exception as e:
                        ex = futures[future]
                        record = {
                            "question_id": str(ex["question_id"]),
                            "db_id": ex["db_id"],
                            "error": str(e),
                            "pred_sql": "",
                            "gold_sql": ex["SQL"],
                            "ex_correct": False,
                        }

                    total += 1
                    if record.get("ex_correct"):
                        correct += 1

                    append_jsonl(jsonl_f, record)
                    pbar.set_postfix(
                        {"EX": f"{correct}/{total} ({100*correct/total:.1f}%)"}
                    )

    all_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    by_qid = {}
    for rec in all_records:
        by_qid[str(rec["question_id"])] = rec

    predictions = {}
    for qid, rec in sorted(by_qid.items(), key=lambda x: int(x[0])):
        sql = rec.get("pred_sql", "")
        db_id = rec.get("db_id", "")
        predictions[qid] = f"{sql}\t----- bird -----\t{db_id}"

    pred_path = os.path.join(args.output_dir, "dev.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    total = len(by_qid)
    correct = sum(1 for r in by_qid.values() if r.get("ex_correct"))
    errors = sum(1 for r in by_qid.values() if r.get("error"))
    avg_tokens = (
        sum(r.get("total_tokens", 0) for r in by_qid.values()) / total if total else 0
    )
    avg_turns = (
        sum(r.get("root_turns", 0) for r in by_qid.values()) / total if total else 0
    )

    print(f"\n{'='*50}")
    print(f"{args.mode} | {args.model}")
    print(f"{'='*50}")
    print(f"Total: {total}")
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}% EX)")
    print(f"Errors: {errors}")
    print(f"Avg tokens: {avg_tokens:.0f}")
    print(f"Avg turns: {avg_turns:.1f}")
    print(f"\nPredictions: {pred_path}")
    print(f"Detail: {jsonl_path}")


if __name__ == "__main__":
    main()
