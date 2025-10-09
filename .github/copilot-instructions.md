## Tsugite: AI Contributor Guide (Concise)

Purpose: Micro‑agent CLI. Agents = Markdown + YAML frontmatter rendered via Jinja2, executed via TsugiteAgent with a lightweight tool registry.

### Core Implemented Modules
`tsugite/tsugite.py` (CLI) • `md_agents.py` (frontmatter parsing + `AgentConfig`) • `renderer.py` (Jinja2 + filters: `now()`, `today()`, `slugify()`, `env`) • `agent_runner.py` (prefetch + execution) • `core/agent.py` (TsugiteAgent with LiteLLM) • `models.py` (model string parsing + provider dispatch) • `tools/` (fs / shell / http) • `benchmarks/` (model + template evaluation).

### Design vs Current State
Design docs mention permissions, history/audit JSONL, orchestration directives (`tsu:spawn`, `tsu:foreach`, `tsu:cond`), memory/git/network tools. These are NOT fully present—do not fabricate calls. Add future systems as isolated modules with tests before integration.

### Agent Structure
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
`prefetch` entries `{tool, args, assign}` run before rendering; failures -> `None` (templates must guard).

### Model String Convention
`provider:model[:variant]` (e.g. `ollama:qwen2.5-coder:7b`, `openai:gpt-4o-mini`). Parsed by `parse_model_string`; Ollama defaults to OpenAI-compatible base `http://localhost:11434/v1` unless overridden.

### Adding / Updating Tools
```python
from tsugite.tools import tool
@tool
def line_count(path: str) -> int: ...
```
Ensure import side-effect registers it (edit `tools/__init__.py` if needed). Adapter preserves signature/docstring; raise `ValueError` / `RuntimeError` with contextual message. Adapter converts exceptions to string for agent surface.

### Safety
`tools/shell.py` blocks dangerous substrings (`rm -rf /`, `sudo rm`, `dd if=` etc.). Never relax; only extend list. Avoid new mutation/network tools until permission + audit layers exist.

### Rendering & Context
Strict Jinja: undefined vars raise early. Helpers: `now()`, `today()`, `slugify()`. Env vars via `env`. Use `prefetch` for deterministic data to reduce token use / steps.

### Benchmarks & Tests
Bench specs under `benchmarks/` (fields: `expected_output`, `evaluation_criteria`, `{{ model }}` placeholder). Run:
```bash
uv sync --dev
uv run pytest
```
Style: `uv run black .` + `uv run ruff check .` (88-char). Supply type hints + concise docstrings for public APIs.

### Error Handling Norms
`ValueError` = bad inputs/config; `RuntimeError` = execution failure. Messages must name failing component/tool/path. Return raw strings only through tool adapter fallback.

### Current Capability Snapshot
Parsing ✅ • Prefetch ✅ • Rendering ✅ • Tool exec ✅ • Model abstraction ✅ • Benchmarks ✅ • Permissions ❌ • History ❌ • Advanced orchestration ❌.

### Extension Guidance
Introduce missing systems (permissions/history/orchestration) as separate modules with minimal public interfaces, then integrate. Maintain backward compatibility for model string + agent frontmatter keys. Keep PRs focused; add targeted tests in `tests/` for each new behavior.

Questions or ambiguities: raise them so this guide can iterate.