"""Shared helpers for benchmark harness CLI/bootstrap wiring."""

import argparse
import json
import os
import random
import time

from cast_agent.llm import LLMClient


PROVIDER_CHOICES = ("openrouter", "azure", "local")


def load_dotenv_if_available() -> None:
    """Load .env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def add_llm_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str = "deepseek/deepseek-v3.2",
    include_provider: bool = True,
    include_sampling: bool = False,
    include_max_tokens: bool = False,
) -> None:
    """Register a consistent set of model/provider flags."""
    parser.add_argument("--model", type=str, default=default_model)
    if include_provider:
        parser.add_argument(
            "--provider",
            type=str,
            default="openrouter",
            choices=list(PROVIDER_CHOICES),
            help="API provider (default: openrouter). 'local' uses localhost:1234.",
        )
        parser.add_argument(
            "--base_url", type=str, default="", help="API base URL override"
        )
        parser.add_argument("--api_key", type=str, default="", help="API key override")

    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)

    if include_sampling:
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--top_k", type=int, default=None)

    if include_max_tokens:
        parser.add_argument("--max_tokens", type=int, default=8192)


def resolve_thinking_flag(args: argparse.Namespace) -> None:
    """Apply --no_thinking override in-place."""
    if getattr(args, "no_thinking", False):
        args.thinking = False


def collect_sampling_kwargs(args: argparse.Namespace) -> dict:
    """Extract sampling kwargs supported by LLMClient."""
    sampling_kwargs: dict = {}
    top_p = getattr(args, "top_p", None)
    top_k = getattr(args, "top_k", None)
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    return sampling_kwargs


def build_llm_from_args(
    args: argparse.Namespace,
    **overrides,
) -> LLMClient:
    """Create LLMClient from shared CLI args + optional explicit overrides."""
    kwargs = {
        "model": getattr(args, "model", "deepseek/deepseek-v3.2"),
        "thinking": getattr(args, "thinking", True),
        "temperature": getattr(args, "temperature", 1.0),
    }

    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    provider = getattr(args, "provider", None)
    if provider:
        kwargs["provider"] = provider
    base_url = getattr(args, "base_url", None)
    if base_url:
        kwargs["base_url"] = base_url
    api_key = getattr(args, "api_key", None)
    if api_key:
        kwargs["api_key"] = api_key

    kwargs.update(collect_sampling_kwargs(args))
    kwargs.update(overrides)
    return LLMClient(**kwargs)


# ------------------------------------------------------------------
# Retry helper
# ------------------------------------------------------------------


def call_with_retry(
    fn,
    max_retries: int = 3,
    base_delay: float = 2.0,
):
    """Call *fn* with exponential backoff + jitter on failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) * (0.5 + random.random())
                time.sleep(delay)
    raise last_err


# ------------------------------------------------------------------
# JSONL checkpoint helpers
# ------------------------------------------------------------------


def load_checkpoint_ids(
    jsonl_path: str,
    id_field: str = "question_id",
) -> set[str]:
    """Read completed record IDs from an existing JSONL file."""
    completed: set[str] = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if id_field in rec:
                    completed.add(str(rec[id_field]))
            except (json.JSONDecodeError, ValueError):
                pass
    return completed


def append_jsonl(f, record: dict) -> None:
    """Append a single JSON record to an open file handle and flush."""
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()
