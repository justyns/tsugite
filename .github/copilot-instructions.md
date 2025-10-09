## Tsugite: AI Contributor Guide (Concise)

Tsugite is a micro-agent CLI: agents are Markdown + YAML frontmatter rendered via Jinja2 and executed by `TsugiteAgent` with a lightweight tool registry.

### Core modules shipping today
- `tsugite/tsugite.py` – Typer CLI entrypoint
- `md_agents.py` – frontmatter parsing + `AgentConfig`
- `renderer.py` – Jinja2 renderer + helpers (`now()`, `today()`, `slugify()`, `env`)
- `agent_runner.py` – prefetch, tool wiring, execution loop
- `core/agent.py` – LiteLLM-backed agent loop
- `models.py` – model string parsing + provider dispatch
- `tools/` – filesystem / shell / http tool registry
- `benchmarks/` – template + model evaluation harness

### Design gap callouts
Design docs mention permissions, history/audit JSONL, orchestration directives (`tsu:spawn`, `tsu:foreach`, `tsu:cond`), and memory/git/network tools. None of these exist yet—do not fabricate calls. When implementing future systems, keep them isolated with focused tests before wiring them into the core loop.

### Agent structure basics
Frontmatter keys: `name`, `model`; optional: `max_steps`, `tools`, `prefetch`, `context_budget`. Example:
```markdown
---
name: hello_world
model: ollama:qwen2.5-coder:7b
max_steps: 2
tools: []
---
Output exactly what is requested.
```
`prefetch` entries `{tool, args, assign}` run before rendering; failures return `None`, so templates must guard accordingly.

### Model string convention
Use `provider:model[:variant]` (e.g. `ollama:qwen2.5-coder:7b`, `openai:gpt-4o-mini`). `parse_model_string` handles the split; Ollama defaults to the OpenAI-compatible base `http://localhost:11434/v1` unless overridden.

### Adding / updating tools
```python
from tsugite.tools import tool
@tool
def line_count(path: str) -> int: ...
```
Ensure import side-effects register the tool (edit `tools/__init__.py` if needed). The adapter preserves signatures/docstrings; raise `ValueError` for bad inputs and `RuntimeError` for execution failures. Exceptions are surfaced as strings to the agent.

### Safety
`tools/shell.py` blocks dangerous substrings (`rm -rf /`, `sudo rm`, `dd if=` etc.). Never relax the guard—only extend it. Avoid new mutation/network tools until permission + audit layers exist.

### Rendering & context
Strict Jinja: undefined variables raise immediately. Helpers: `now()`, `today()`, `slugify()`, `file_exists`, `read_text`, `env`. Use `prefetch` for deterministic data to reduce token use and reasoning steps.

### Benchmarks & tests
Bench specs live under `benchmarks/` (fields: `expected_output`, `evaluation_criteria`, `{{ model }}` placeholder). Usual loop:

```bash
uv sync --dev
uv run pytest
```

Work test-first wherever possible: add or update the minimal failing test, run that target, then widen to `uv run pytest` before merging. Also run `uv run black .` and `uv run ruff check .` (88-char). Public APIs should ship with type hints and concise docstrings.

### Error handling norms
- `ValueError` → bad inputs/config
- `RuntimeError` → execution failure (name the failing component/tool/path)
- Return raw strings only through the tool adapter fallback when nothing better is available

### Current capability snapshot
Parsing ✅ • Prefetch ✅ • Rendering ✅ • Tool exec ✅ • Model abstraction ✅ • Benchmarks ✅ • Permissions ❌ • History ❌ • Advanced orchestration ❌

### Extension guidance
Introduce missing systems (permissions/history/orchestration) as separate modules with minimal public interfaces, then integrate. Maintain backward compatibility for model string + agent frontmatter keys. Keep PRs focused and add targeted tests in `tests/` for each new behaviour.

Questions or ambiguities: raise them so this guide can iterate.