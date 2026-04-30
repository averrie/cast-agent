"""LLM client wrapper for OpenAI-compatible APIs (OpenRouter, Azure, etc.)."""

import os
import time as _time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, int]
    reasoning: list[dict] | None = None


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str | None = None
    api_key_envs: tuple[str, ...] = ()
    base_url_env: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(api_key_envs=("OPENAI_API_KEY",)),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_envs=("OPENROUTER_API_KEY",),
    ),
    # Azure works with the OpenAI client when targeting the Azure v1 endpoint.
    "azure": ProviderConfig(
        api_key_envs=("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
        base_url_env="AZURE_OPENAI_BASE_URL",
    ),
    "local": ProviderConfig(base_url="http://localhost:1234/v1"),  # LM Studio / Ollama
}


def _first_env(keys: Iterable[str]) -> str | None:
    for key in keys:
        if not key:
            continue
        val = os.environ.get(key)
        if val:
            return val
    return None


def register_provider(
    name: str,
    base_url: str | None = None,
    env_key: str | None = None,
    *,
    api_key_envs: Iterable[str] | None = None,
    base_url_env: str | None = None,
) -> None:
    """Register a custom provider using the standard OpenAI client shape.

    This keeps all vendors behind the same `base_url` + `api_key` interface.
    """
    if api_key_envs is not None:
        key_envs = tuple(k for k in api_key_envs if k)
    elif env_key:
        key_envs = (env_key,)
    else:
        key_envs = ()
    PROVIDERS[name] = ProviderConfig(
        base_url=base_url,
        api_key_envs=key_envs,
        base_url_env=base_url_env,
    )


class LLMClient:
    """Stateful multi-turn conversation client via OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str = "deepseek/deepseek-v3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openrouter",
        thinking: bool = True,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        **sampling_kwargs,
    ):
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.sampling_kwargs = sampling_kwargs  # e.g. top_p, top_k

        provider_cfg = PROVIDERS.get(provider, ProviderConfig())

        resolved_base_url = base_url
        if resolved_base_url is None:
            resolved_base_url = provider_cfg.base_url or (
                os.environ.get(provider_cfg.base_url_env)
                if provider_cfg.base_url_env
                else None
            )

        resolved_api_key = (
            api_key if api_key is not None else _first_env(provider_cfg.api_key_envs)
        )

        if provider == "azure" and not resolved_base_url:
            raise ValueError(
                "provider='azure' requires base_url or AZURE_OPENAI_BASE_URL "
                "(expected format: https://<resource>.openai.azure.com/openai/v1/)."
            )

        client_kwargs: dict[str, Any] = {}
        if resolved_api_key is not None:
            client_kwargs["api_key"] = resolved_api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = OpenAI(**client_kwargs)
        self.total_tokens = 0
        self.total_calls = 0

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        resp = self._raw_chat(messages)
        return resp.content, resp.usage

    def chat_full(self, messages: list[dict]) -> LLMResponse:
        return self._raw_chat(messages)

    def _raw_chat(self, messages: list[dict]) -> LLMResponse:
        extra = {"reasoning": {"enabled": True}} if self.thinking else {}
        extra.update(self.sampling_kwargs)
        _t0 = _time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body=extra if extra else None,
        )
        _elapsed = _time.perf_counter() - _t0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if resp.usage:
            usage["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = (
                getattr(resp.usage, "completion_tokens", 0) or 0
            )
            usage["total_tokens"] = getattr(resp.usage, "total_tokens", 0) or 0
        usage["elapsed_s"] = round(_elapsed, 3)
        if _elapsed > 0 and usage["completion_tokens"]:
            usage["tokens_per_s"] = round(usage["completion_tokens"] / _elapsed, 1)
        self.total_tokens += usage["total_tokens"]
        self.total_calls += 1

        if not resp.choices:
            return LLMResponse(content="", usage=usage, reasoning=None)

        content = (resp.choices[0].message.content or "").strip()

        msg = resp.choices[0].message
        reasoning = None
        raw_reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning_details", None)
            or getattr(msg, "reasoning", None)
        )
        if raw_reasoning:
            if isinstance(raw_reasoning, list):
                reasoning = raw_reasoning
            elif isinstance(raw_reasoning, str):
                reasoning = [{"type": "thinking", "thinking": raw_reasoning}]
            else:
                reasoning = [{"type": "raw", "content": str(raw_reasoning)}]

        return LLMResponse(content=content, usage=usage, reasoning=reasoning)

    def reset_counters(self) -> None:
        self.total_tokens = 0
        self.total_calls = 0

    @staticmethod
    def make_assistant_message(response: LLMResponse) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.reasoning is not None:
            msg["reasoning_details"] = response.reasoning
        return msg
