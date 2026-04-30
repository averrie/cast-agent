#!/usr/bin/env python3
"""CAST-OfficeQA evaluation harness."""

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from cast_agent.harness_utils import (
    add_llm_cli_args,
    append_jsonl,
    call_with_retry,
    load_checkpoint_ids,
    load_dotenv_if_available,
    resolve_thinking_flag,
)
from cast_agent.llm import LLMClient
from cast_agent.loops import run_cast
from cast_agent.tools import ToolBudgetConfig
from examples.officeqa.tools import CorpusIndex, OfficeQAToolLayer
from examples.officeqa.prompts import make_officeqa_cast_prompt
from examples.officeqa.evaluation import score_answer


# ----------------------------
# Live progress tracking
# ----------------------------

load_dotenv_if_available()

class ProgressTracker:
    """Thread-safe tracker for per-question turn progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: dict[str, dict] = {}
        self._pbar: tqdm | None = None
        self._total_turns = 0

    def set_bar(self, bar: tqdm) -> None:
        self._pbar = bar

    def on_turn(self, uid: str, info: dict) -> None:
        with self._lock:
            self._active[uid] = {
                "turn": info["turn"],
                "max_turns": info["max_turns"],
                "started": self._active.get(uid, {}).get("started", time.time()),
            }
            self._total_turns += 1
            self._refresh()

    def on_complete(self, uid: str) -> None:
        with self._lock:
            self._active.pop(uid, None)

    def _refresh(self) -> None:
        if not self._pbar:
            return
        n = len(self._active)
        if not self._active:
            info_str = f"turns={self._total_turns}"
        else:
            turns = [info["turn"] for info in self._active.values()]
            turn_range = f"{min(turns)}-{max(turns)}" if min(turns) != max(turns) else str(turns[0])
            info_str = f"active={n} turns={self._total_turns} t={turn_range}"
        self._pbar.set_postfix_str(info_str)

    def format_active(self) -> str:
        with self._lock:
            n = len(self._active)
            if not self._active:
                return f"turns={self._total_turns}"
            turns = [info["turn"] for info in self._active.values()]
            turn_range = f"{min(turns)}-{max(turns)}" if min(turns) != max(turns) else str(turns[0])
            return f"active={n} turns={self._total_turns} t={turn_range}"


# ----------------------------
# Per-question worker
# ----------------------------


def eval_one(
    example: dict,
    corpus: CorpusIndex,
    llm_kwargs: dict,
    max_turns: int,
    max_tool_batches: int,
    keep_last: int,
    save_trace: bool,
    allow_json_repair: bool = False,
    max_exec_batches: int | None = None,
    tracker: ProgressTracker | None = None,
) -> dict:
    uid = example["uid"]
    question = example["question"]
    ground_truth = example["answer"]
    source_files = example.get("source_files", "")
    difficulty = example.get("difficulty", "")

    years = sorted(set(m["year"] for m in corpus.metadata.values() if m["year"]))
    year_range = f"{min(years)}-{max(years)}" if years else "unknown"

    def _on_turn(info: dict) -> None:
        if tracker:
            tracker.on_turn(uid, info)

    def _run_once():
        llm = LLMClient(**llm_kwargs)
        tool_layer = OfficeQAToolLayer(
            corpus=corpus,
            max_request_batches=max_tool_batches,
            budget_config=ToolBudgetConfig(
                read_batches=max_tool_batches,
                exec_batches=max_exec_batches,
            ),
            allow_json_repair=allow_json_repair,
        )
        system_prompt = make_officeqa_cast_prompt(
            corpus_size=len(corpus.docs),
            year_range=year_range,
            tool_descriptions=tool_layer.tool_descriptions(),
            max_tool_batches=max_tool_batches,
        )
        try:
            result = run_cast(
                question,
                tool_layer,
                llm,
                system_prompt,
                max_turns=max_turns,
                on_turn=_on_turn,
                auto_submit=True,
                keep_last=keep_last,
                answer_noun="answer",
            )
            return result, tool_layer.batch_count
        finally:
            tool_layer.close()

    try:
        result, tool_batches = call_with_retry(_run_once)
    except Exception as e:
        return {
            "uid": uid,
            "error": str(e),
            "predicted": "",
            "ground_truth": ground_truth,
            "correct_0pct": False,
            "correct_1pct": False,
            "correct_5pct": False,
            "difficulty": difficulty,
        }

    predicted = result.final_answer or ""

    scores = {}
    for tol_name, tol_val in [("0pct", 0.00), ("1pct", 0.01), ("5pct", 0.05)]:
        try:
            s = score_answer(ground_truth, predicted, tolerance=tol_val)
            scores[f"correct_{tol_name}"] = s == 1.0
        except Exception:
            scores[f"correct_{tol_name}"] = False

    record = {
        "uid": uid,
        "predicted": predicted,
        "ground_truth": ground_truth,
        **scores,
        "difficulty": difficulty,
        "source_files": source_files,
        "total_tokens": result.total_tokens,
        "total_llm_calls": result.total_llm_calls,
        "root_turns": result.root_turns,
        "wall_time": round(result.wall_time, 2),
        "tool_batches": tool_batches,
    }

    if save_trace:
        record["trace_log"] = result.trace_log
        record["context_packets"] = result.context_packets

    return record


# ----------------------------
# Main harness
# ----------------------------


def main():
    parser = argparse.ArgumentParser(description="CAST-OfficeQA evaluation harness")
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to officeqa_pro.csv or officeqa_full.csv")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to corpus zip or directory of .txt files")
    add_llm_cli_args(
        parser,
        include_provider=True,
        include_sampling=False,
        include_max_tokens=False,
    )
    parser.add_argument("--max_turns", type=int, default=200)
    parser.add_argument("--max_tool_batches", type=int, default=200)
    parser.add_argument(
        "--max_exec_batches",
        type=int,
        default=0,
        help="Max compute/exec batches (0 = unlimited).",
    )
    parser.add_argument(
        "--allow_json_repair",
        action="store_true",
        help="Enable JSON repair fallback for malformed CAST request payloads.",
    )
    parser.add_argument("--keep_last", type=int, default=30,
                        help="Sliding window: keep N most recent messages")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0 = all)")
    parser.add_argument("--question_ids", type=str, default="", help="Comma-separated UIDs to run")
    parser.add_argument("--save_trace", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    resolve_thinking_flag(args)
    llm_kwargs = {
        "model": args.model,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "provider": args.provider,
        "base_url": args.base_url or None,
        "api_key": args.api_key or None,
    }
    max_exec_batches = args.max_exec_batches if args.max_exec_batches > 0 else None

    with open(args.questions, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    print(f"Loaded {len(dataset)} questions from {args.questions}")

    if args.question_ids:
        target_ids = set(args.question_ids.split(","))
        dataset = [ex for ex in dataset if ex["uid"] in target_ids]
        print(f"Filtered to {len(dataset)} questions")

    if args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Loading corpus from {args.corpus} ...")
    t0 = time.time()
    corpus = CorpusIndex(args.corpus)
    print(f"Loaded {len(corpus.docs)} documents in {time.time() - t0:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "officeqa_cast.jsonl")

    completed_ids = load_checkpoint_ids(jsonl_path, id_field="uid")
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    remaining = [ex for ex in dataset if ex["uid"] not in completed_ids]
    print(f"Running {len(remaining)} questions (CAST, {args.model})")

    if not remaining:
        print("Nothing to do.")
    else:
        tracker = ProgressTracker()
        with open(jsonl_path, "a", encoding="utf-8") as jsonl_f:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        eval_one,
                        ex,
                        corpus,
                        llm_kwargs,
                        args.max_turns,
                        args.max_tool_batches,
                        args.keep_last,
                        args.save_trace,
                        args.allow_json_repair,
                        max_exec_batches,
                        tracker,
                    ): ex
                    for ex in remaining
                }

                correct = 0
                total = 0
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Questions")
                tracker.set_bar(pbar)

                for future in pbar:
                    try:
                        record = future.result()
                    except Exception as e:
                        ex = futures[future]
                        record = {
                            "uid": ex["uid"],
                            "error": str(e),
                            "predicted": "",
                            "ground_truth": ex["answer"],
                            "correct_0pct": False,
                            "correct_1pct": False,
                            "correct_5pct": False,
                        }

                    total += 1
                    uid = record.get("uid", "?")
                    if record.get("correct_0pct"):
                        correct += 1
                        status = "CORRECT"
                    elif record.get("error"):
                        status = "ERROR"
                    else:
                        status = "WRONG"

                    tracker.on_complete(uid)

                    turns = record.get("root_turns", 0)
                    tokens = record.get("total_tokens", 0)
                    wall = record.get("wall_time", 0)
                    tqdm.write(
                        f"  {uid} {status:7s} | "
                        f"turns={turns:3d} tokens={tokens:>8,} "
                        f"time={wall:>6.1f}s | "
                        f"pred={str(record.get('predicted',''))[:50]!r}"
                    )

                    append_jsonl(jsonl_f, record)

                    pbar.set_postfix_str(
                        f"Acc={correct}/{total}({100*correct/total:.1f}%) {tracker.format_active()}"
                    )

    all_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    by_uid = {}
    for rec in all_records:
        by_uid[rec["uid"]] = rec

    total = len(by_uid)
    if total == 0:
        print("No results to summarize.")
        return

    correct_0 = sum(1 for r in by_uid.values() if r.get("correct_0pct"))
    correct_1 = sum(1 for r in by_uid.values() if r.get("correct_1pct"))
    correct_5 = sum(1 for r in by_uid.values() if r.get("correct_5pct"))
    errors = sum(1 for r in by_uid.values() if r.get("error"))
    avg_tokens = sum(r.get("total_tokens", 0) for r in by_uid.values()) / total
    avg_turns = sum(r.get("root_turns", 0) for r in by_uid.values()) / total
    avg_time = sum(r.get("wall_time", 0) for r in by_uid.values()) / total

    print(f"\n{'='*60}")
    print(f"CAST | {args.model}")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Correct (0% tol): {correct_0}/{total} ({100*correct_0/total:.1f}%)")
    print(f"Correct (1% tol): {correct_1}/{total} ({100*correct_1/total:.1f}%)")
    print(f"Correct (5% tol): {correct_5}/{total} ({100*correct_5/total:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Avg tokens: {avg_tokens:,.0f}")
    print(f"Avg turns: {avg_turns:.1f}")
    print(f"Avg wall time: {avg_time:.1f}s")
    print(f"\nResults: {jsonl_path}")


if __name__ == "__main__":
    main()
