# CAST - Context Access through Structured Tools

A benchmark-agnostic agent framework for evaluating how LLMs interact with external context through structured tool interfaces.

## Architecture

```
src/
└── cast_agent/                 # the library (pip installable)
    ├── llm.py                  # LLMClient - OpenAI-compatible, pluggable providers
    ├── types.py                # AgentResult
    ├── parsing.py              # XML tag parsers (request_context, final_answer, tool_call)
    ├── history.py              # Sliding-window conversation compaction
    ├── loops/                  # Three agent modes
    │   ├── stuffed.py          # Single-turn baseline
    │   ├── react.py            # ReAct tool-calling loop
    │   └── cast.py             # Structured context access (CAST)
    └── tools/                  # Tool abstractions + generic tools
        ├── __init__.py         # ToolSet ABC (ReAct), ToolLayer ABC (CAST)
        ├── repl.py             # SandboxedREPL - Python code execution
        └── web.py              # WebSearchClient - DuckDuckGo search

tests/                          # unit tests for cast_agent

examples/                       # benchmark implementations (not part of the package)
├── bird/                       # BIRD text-to-SQL
├── officeqa/                   # OfficeQA document QA
└── swebench/                   # SWE-bench code repair
```

## Quick Start

```bash
pip install -e .
```

To run example harnesses (BIRD / OfficeQA / SWE-bench):

```bash
pip install -e ".[examples]"
```

To run tests:

```bash
pip install -e ".[dev]"
```

```python
from cast_agent.llm import LLMClient
from cast_agent.loops import run_cast
from cast_agent.tools import RequestSchema, ToolLayer

# 1. Implement your tool layer
class MyToolLayer(ToolLayer):
    KNOWN_TYPES = frozenset(["search", "read"])
    REQUEST_SCHEMAS = {
        "search": RequestSchema(required={"query": (str,)}),
        "read": RequestSchema(required={"path": (str,)}),
    }

    def _dispatch(self, req_type, req):
        if req_type == "search":
            return do_search(req["query"])
        if req_type == "read":
            return do_read(req["path"])
        return f"Unknown type: {req_type}"

    def tool_descriptions(self):
        return '1. search - "query": "..." (required)\n2. read - "path": "..." (required)'

# 2. Run the agent
llm = LLMClient(model="deepseek/deepseek-v3.2", provider="openrouter")
tool_layer = MyToolLayer(max_request_batches=10)
system_prompt = f"You are a QA agent.\n\n{tool_layer.tool_descriptions()}\n..."

result = run_cast("What is X?", tool_layer, llm, system_prompt, max_turns=20)
print(result.final_answer)
```

## Adding a New Benchmark

See `examples/` for complete implementations. The pattern is:

### 1. Implement tools

For **ReAct** mode, subclass `ToolSet`:

```python
from cast_agent.tools import ToolSet

class MyTools(ToolSet):
    def execute(self, name: str, args: dict) -> str: ...
    def tool_descriptions(self) -> str: ...
    def close(self) -> None: ...
```

For **CAST** mode, subclass `ToolLayer`:

```python
from cast_agent.tools import ToolLayer

class MyToolLayer(ToolLayer):
    KNOWN_TYPES = frozenset(["search", "read", "compute"])
    MAX_BATCH_RESULT_CHARS = 25_000  # optional output cap
    REQUEST_SCHEMAS = {
        "search": RequestSchema(required={"query": (str,)}),
        "read": RequestSchema(required={"path": (str,)}),
        "compute": RequestSchema(required={"code": (str,)}),
    }

    def _dispatch(self, req_type: str, req: dict) -> str: ...
    def tool_descriptions(self) -> str: ...
```

### 2. Write prompts, evaluation, and a harness

```
examples/mybench/
├── __init__.py
├── tools.py          # MyTools + MyToolLayer
├── prompts.py        # system prompt templates
├── evaluation.py     # scoring function
├── harness.py        # eval runner (argparse + ThreadPoolExecutor)
└── __main__.py       # python -m examples.mybench
```

### 3. Run

```bash
python -m examples.mybench --help
```

## LLM Providers

Built-in: `openai`, `openrouter`, `azure`, `local` (LM Studio / Ollama at localhost:1234).

- `openrouter`: defaults to `https://openrouter.ai/api/v1` + `OPENROUTER_API_KEY`.
- `azure`: uses the standard OpenAI client but requires an Azure v1 endpoint via
  `base_url` or `AZURE_OPENAI_BASE_URL`
  (e.g. `https://<resource>.openai.azure.com/openai/v1/`) and
  `AZURE_OPENAI_API_KEY` (or legacy `AZURE_API_KEY`).
- `local`: defaults to `http://localhost:1234/v1`.

You can also enforce a pure OpenAI-compatible setup for any vendor by providing
just `base_url` and `api_key` directly when creating `LLMClient`.

Register custom providers at runtime:

```python
from cast_agent.llm import register_provider
register_provider("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY")
```

## Running Examples

```bash
# BIRD text-to-SQL
python -m examples.bird --mode cast --gold data/dev.json --db_root data/dev_databases

# OfficeQA document QA
python -m examples.officeqa --questions officeqa_pro.csv --corpus corpus.zip

# SWE-bench code repair
python -m examples.swebench --mode cast --dataset verified --limit 5
```
