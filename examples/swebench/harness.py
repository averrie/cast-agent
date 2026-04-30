"""CAST eval harness for SWE-bench.

Modes:
  react - root agent has direct tools (bash, read_file, etc.).
  cast  - root has only structured context layer.

Uses mini-SWE-agent's DockerEnvironment for container management.
Outputs predictions compatible with SWE-bench evaluation.
"""

import argparse
import contextlib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

from cast_agent.harness_utils import (
    add_llm_cli_args,
    append_jsonl,
    build_llm_from_args,
    call_with_retry,
    load_checkpoint_ids,
    load_dotenv_if_available,
    resolve_thinking_flag,
)
from examples.swebench.prompts import make_react_prompt, make_cast_prompt
from cast_agent.loops import run_react, run_cast
from cast_agent.tools import ToolBudgetConfig
from examples.swebench.tools import RepoTools, RepoToolLayer
from examples.swebench.docker_env import SWEBenchEnv


# ----------------------------
# Dataset helpers
# ----------------------------

load_dotenv_if_available()

DATASET_MAPPING = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def extract_repo_name(instance_id: str) -> str:
    """Extract 'org/repo' from an instance_id like 'django__django-16527'."""
    parts = instance_id.split("-", 1)
    if parts:
        repo_part = parts[0]
        return repo_part.replace("__", "/")
    return instance_id


# ----------------------------
# Per-instance worker
# ----------------------------


def _eval_one_instance(
    instance: dict,
    args,
) -> tuple[dict, dict]:
    """Evaluate one SWE-bench instance. Returns (meta, record)."""
    instance_id = instance["instance_id"]
    repo_name = extract_repo_name(instance_id)
    problem_statement = instance["problem_statement"]

    def _on_turn(info: dict) -> None:
        turn = info.get("turn", "?")
        max_t = info.get("max_turns", "?")
        tqdm.write(f"  {instance_id} {info.get('agent', 'root')} turn {turn}/{max_t}")

    def _run_once() -> tuple[dict, dict]:
        env = None
        tool_layer = None
        tools = None
        try:
            try:
                env = SWEBenchEnv(instance, timeout=args.docker_timeout)
            except Exception as e:
                raise RuntimeError(f"docker_setup: {e}") from e

            llm = build_llm_from_args(args)

            t0 = time.time()
            if args.mode == "cast":
                tool_layer = RepoToolLayer(
                    env=env,
                    max_request_batches=args.max_tool_rounds,
                    budget_config=ToolBudgetConfig(
                        read_batches=args.max_tool_rounds,
                        write_batches=(
                            args.max_write_batches
                            if args.max_write_batches > 0
                            else None
                        ),
                        test_batches=(
                            args.max_test_batches if args.max_test_batches > 0 else None
                        ),
                        exec_batches=(
                            args.max_exec_batches if args.max_exec_batches > 0 else None
                        ),
                    ),
                    allow_json_repair=args.allow_json_repair,
                )
                system_prompt = make_cast_prompt(
                    repo_name=repo_name,
                    tool_descriptions=tool_layer.tool_descriptions(),
                    max_tool_batches=args.max_tool_rounds,
                )
                agent_result = run_cast(
                    question=problem_statement,
                    tool_layer=tool_layer,
                    llm=llm,
                    system_prompt=system_prompt,
                    max_turns=args.max_root_turns,
                    on_turn=_on_turn,
                    auto_submit=args.auto_submit,
                )
                tool_batch_count = tool_layer.batch_count
                tool_request_count = tool_layer.total_requests
            else:
                tools = RepoTools(env=env)
                system_prompt = make_react_prompt(
                    repo_name=repo_name,
                    tool_descriptions=tools.tool_descriptions(),
                )
                agent_result = run_react(
                    question=problem_statement,
                    tools=tools,
                    llm=llm,
                    system_prompt=system_prompt,
                    max_turns=args.max_root_turns,
                    on_turn=_on_turn,
                )
                tool_batch_count = 0
                tool_request_count = 0

            wall_time = time.time() - t0
            patch = env.get_patch()

            record = {
                "instance_id": instance_id,
                "repo": repo_name,
                "mode": args.mode,
                "patch": patch,
                "patch_length": len(patch),
                "agent_submitted": agent_result.final_answer is not None,
                "total_tokens": agent_result.total_tokens,
                "total_llm_calls": agent_result.total_llm_calls,
                "root_turns": agent_result.root_turns,
                "tool_batch_count": tool_batch_count,
                "tool_request_count": tool_request_count,
                "tool_calls_count": len(agent_result.tool_calls),
                "wall_time": round(wall_time, 2),
            }

            if args.save_trace:
                record["tool_calls"] = agent_result.tool_calls
                record["context_packets"] = agent_result.context_packets
                record["turn_token_log"] = agent_result.turn_token_log
                record["trace_log"] = agent_result.trace_log

            return {"evaluated": True, "ok": bool(patch)}, record
        finally:
            if tools is not None:
                tools.close()
            elif tool_layer is not None:
                tool_layer.close()
            elif env is not None:
                env.cleanup()

    try:
        return call_with_retry(_run_once)
    except Exception as e:
        tqdm.write(f"[WARN] {instance_id}: agent failed: {e}")
        tqdm.write(traceback.format_exc())
        err = str(e)
        if not err.startswith("docker_setup:"):
            err = f"agent_error: {err}"
        return (
            {"evaluated": False, "ok": False},
            {
                "instance_id": instance_id,
                "repo": repo_name,
                "error": err,
            },
        )


# ----------------------------
# Predictions file (SWE-bench compatible)
# ----------------------------


def save_predictions(predictions: dict, path: str) -> None:
    """Save predictions in SWE-bench evaluation format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser(description="CAST-SWE-bench evaluation harness.")

    # Dataset
    ap.add_argument(
        "--dataset",
        default="verified",
        help=(
            "SWE-bench subset: 'verified', 'lite', 'full', or a HuggingFace dataset ID. "
            "Default: verified."
        ),
    )
    ap.add_argument("--split", default="test", help="Dataset split (default: test).")

    # Mode
    ap.add_argument(
        "--mode",
        required=True,
        choices=["react", "cast"],
        help="Agent mode: react (tool-calling) or cast (structured context).",
    )

    # Model
    add_llm_cli_args(
        ap,
        include_provider=True,
        include_sampling=False,
        include_max_tokens=True,
    )

    # Agent budget
    ap.add_argument(
        "--max_root_turns",
        type=int,
        default=75,
        help="Max root agent turns (default 75).",
    )
    ap.add_argument(
        "--max_tool_rounds",
        type=int,
        default=75,
        help="Max context request batches (CAST mode, default 75).",
    )
    ap.add_argument(
        "--max_write_batches",
        type=int,
        default=0,
        help="Max write batches in CAST mode (0 = unlimited).",
    )
    ap.add_argument(
        "--max_test_batches",
        type=int,
        default=0,
        help="Max test batches in CAST mode (0 = unlimited).",
    )
    ap.add_argument(
        "--max_exec_batches",
        type=int,
        default=0,
        help="Max exec-command batches in CAST mode (0 = unlimited).",
    )
    ap.add_argument(
        "--allow_json_repair",
        action="store_true",
        help="Enable JSON repair fallback for malformed CAST request payloads.",
    )

    ap.add_argument(
        "--auto_submit",
        action="store_true",
        default=True,
        help="Auto-submit when turn limit is reached if a patch exists (default: on).",
    )
    ap.add_argument(
        "--no_auto_submit",
        action="store_true",
        help="Disable auto-submit on turn limit.",
    )

    # Docker
    ap.add_argument(
        "--docker_timeout",
        type=int,
        default=60,
        help="Docker command timeout in seconds (default 60).",
    )

    # Filtering
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--instance_ids",
        default="",
        help="Comma-separated instance_ids to run (e.g. django__django-16527).",
    )

    # Output
    ap.add_argument(
        "--save_predictions",
        default="",
        help="Path to save SWE-bench compatible predictions JSON.",
    )
    ap.add_argument("--save_jsonl", default="", help="Path to save per-instance JSONL.")
    ap.add_argument(
        "--save_trace",
        action="store_true",
        help="Include full tool_calls, context_packets, and traces in JSONL.",
    )

    # Concurrency
    ap.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent instances (default 3, Docker-limited).",
    )

    args = ap.parse_args()

    # Handle --no_thinking / --no_auto_submit
    resolve_thinking_flag(args)
    if args.no_auto_submit:
        args.auto_submit = False

    # Load dataset
    dataset_path = DATASET_MAPPING.get(args.dataset, args.dataset)
    print(f"Loading dataset: {dataset_path} (split: {args.split})")
    ds = load_dataset(dataset_path, split=args.split)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    if args.instance_ids:
        id_filter = {x.strip() for x in args.instance_ids.split(",") if x.strip()}
        ds = ds.filter(lambda ex: ex["instance_id"] in id_filter)

    # Auto-resume from existing JSONL
    file_mode = "w"
    if args.save_jsonl and os.path.isfile(args.save_jsonl):
        done_ids = load_checkpoint_ids(args.save_jsonl, id_field="instance_id")
        if done_ids:
            ds = ds.filter(lambda ex: ex["instance_id"] not in done_ids)
            tqdm.write(f"[resume] {len(done_ids)} already done, {len(ds)} remaining.")
        file_mode = "a"

    print(f"\nMode: {args.mode}")
    print(f"Model: {args.model}  thinking={args.thinking}")
    print(f"Root turns: {args.max_root_turns}", end="")
    if args.mode == "cast":
        print(f"  Evidence batches: {args.max_tool_rounds}")
    else:
        print()
    print(f"Docker timeout: {args.docker_timeout}s")
    print(f"Instances: {len(ds)}")
    print()

    total = 0
    evaluated = 0
    patches_produced = 0

    # SWE-bench predictions dict
    predictions: dict = {}

    # Load existing predictions if resuming
    if args.save_predictions and os.path.isfile(args.save_predictions):
        with open(args.save_predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    out_ctx = (
        open(args.save_jsonl, file_mode, encoding="utf-8")
        if args.save_jsonl
        else contextlib.nullcontext()
    )

    with out_ctx as out_f:
        with tqdm(total=len(ds), desc=f"Eval {args.mode}") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_eval_one_instance, ex, args): ex for ex in ds
                }
                for future in as_completed(futures):
                    try:
                        meta, record = future.result()
                    except Exception as e:
                        ex = futures[future]
                        tqdm.write(f"[ERROR] {ex.get('instance_id', '?')}: {e}")
                        pbar.update(1)
                        total += 1
                        continue

                    total += 1
                    if meta["evaluated"]:
                        evaluated += 1
                        if meta["ok"]:
                            patches_produced += 1

                    # Update predictions dict
                    instance_id = record.get("instance_id", "")
                    patch = record.get("patch", "")
                    if instance_id and patch:
                        predictions[instance_id] = {
                            "model_name_or_path": f"cast-{args.mode}",
                            "instance_id": instance_id,
                            "model_patch": patch,
                        }

                    if out_f:
                        append_jsonl(out_f, record)

                    # Incrementally save predictions
                    if args.save_predictions:
                        save_predictions(predictions, args.save_predictions)

                    pbar.update(1)

    print(f"\n=== Summary ({args.mode}) ===")
    print(f"Total: {total}")
    print(f"Evaluated: {evaluated}")
    print(f"Patches produced: {patches_produced}")
    if args.save_jsonl:
        print(f"JSONL: {args.save_jsonl}")
    if args.save_predictions:
        print(f"Predictions: {args.save_predictions}")
        print(
            f"\nTo evaluate:\n"
            f"  python -m swebench.harness.run_evaluation \\\n"
            f"    --predictions_path {args.save_predictions} \\\n"
            f"    --dataset_name {dataset_path} \\\n"
            f"    --run_id cast_{args.mode}"
        )


if __name__ == "__main__":
    main()
