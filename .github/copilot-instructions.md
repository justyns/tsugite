## Tsugite: AI Contributor Guide

Tsugite is a micro-agent CLI: agents are Markdown + YAML frontmatter, rendered via Jinja2, and executed by `TsugiteAgent` with a lightweight tool registry.

### Architecture

**Flow:** `Agent.md` (YAML + Jinja2) → `renderer.py` → `TsugiteAgent` → LiteLLM → Tools

**Key Modules:**
- `cli/__init__.py` - Typer CLI (run, chat, render, config, mcp, tools, agents, attachments, cache, benchmark)
- `md_agents.py` - Parse frontmatter + agent resolution + directives
- `builtin_agents.py` - Built-in agent definitions (builtin-default, builtin-chat-assistant)
- `agent_inheritance.py` - Agent resolution pipeline + inheritance chain
- `chat.py` - Chat session management with history
- `ui/textual_chat.py` - Textual TUI for interactive chat
- `agent_runner.py` - Execution orchestration, prefetch, tool wiring, multi-step
- `core/agent.py` - LiteLLM agent loop + streaming
- `renderer.py` - Jinja2 rendering + helpers (now, today, slugify, env, file_exists, read_text, is_file, is_dir)
- `tools/` - Tool registry + implementations
- `shell_tool_config.py` + `tools/shell_tools.py` - Custom shell command wrappers
- `models.py` - Model string parsing + provider dispatch
- `config.py` - Configuration (default_model, model_aliases, default_base_agent, chat_theme)

**Development:**
```bash
uv sync --dev
uv run pytest tests/test_specific.py -v
uv run black . && uv run ruff check .
```

**Agent Resolution Order:**
1. Built-in agents (`builtin-default`, `builtin-chat-assistant`) - embedded in code
2. `.tsugite/{name}.md` - project-local shared
3. `agents/{name}.md` - project convention
4. `./{name}.md` - current directory
5. `~/.config/tsugite/agents/{name}.md` - global user agents

**Model String:** `provider:model[:variant]` (e.g., `ollama:qwen2.5-coder:7b`, `openai:gpt-4o`)

### Built-in Agents System

Embedded agents accessible without file distribution. Special paths like `<builtin-default>` bypass filesystem checks.

**Current Built-ins:**
1. `builtin-default` - Base agent with task tracking
2. `builtin-chat-assistant` - Chat agent with tools (read_file, write_file, list_files, web_search, run), text_mode enabled

**Implementation:**
```python
# In builtin_agents.py
BUILTIN_FOO_CONTENT = """---
name: foo
model: ollama:qwen2.5-coder:7b
tools: []
---
{{ user_prompt }}
"""

def get_builtin_foo():
    from .md_agents import parse_agent
    return parse_agent(BUILTIN_FOO_CONTENT, Path("<builtin-foo>"))

def is_builtin_agent(name: str) -> bool:
    return name in ("builtin-default", "builtin-chat-assistant", "builtin-foo")
```

**Integration:**
- `agent_inheritance.py:find_agent_file()` - Check `is_builtin_agent()`, return `Path("<builtin-name>")`
- `agent_inheritance.py:load_extended_agent()` - Handle `<builtin-*>` paths, call getter
- `md_agents.py:parse_agent_file()` - Handle `<builtin-*>` paths before `.exists()` check

### Agent Inheritance

**Chain:** Default base → Extended → Current (highest precedence)

**Merge rules:**
- Scalars (model, max_steps): Child overwrites
- Lists (tools): Merge + deduplicate
- Dicts (mcp_servers): Merge, child keys override
- Strings (instructions): Concatenate with `\n\n`

**Opt-out:** `extends: none` skips all inheritance.

### Chat Mode

**Components:**
- `chat.py:ChatManager` - Session management, history (max_history), turn execution
- `ui/textual_chat.py:ChatApp` - Textual TUI
- `ui/widgets/` - MessageList, ThoughtLog

**Default Agent:** `builtin-chat-assistant`. Override with `.tsugite/chat_assistant.md`.

**Flow:**
1. User input → `ChatManager.run_turn()`
2. Inject `chat_history` context (list of ChatTurn objects)
3. Call `agent_runner.run_agent()` with `force_text_mode=True`
4. Store result in ChatTurn (timestamp, messages, token_count, cost)
5. Prune history if `> max_history`

**Text Mode:** `text_mode: true` allows LLM responses without code blocks.

### Directives

**Types:**
1. `<!-- tsu:ignore -->...<!-- /tsu:ignore -->` - Strip developer docs before LLM
2. `<!-- tsu:tool name="..." args={...} assign="..." -->` - Execute tool during render
3. `<!-- tsu:step name="..." assign="..." -->` - Multi-step boundaries

**Parsing:** `md_agents.py:extract_step_directives()`, `extract_tool_directives()`

**Execution:**
- Ignore blocks: `renderer.py` strips before Jinja2
- Tool directives: `agent_runner.py:execute_tool_directives()` runs before rendering
- Step directives: `agent_runner.py:run_multistep_agent()` executes sequentially

**Scoping:**
- Preamble directives: Execute once, available to all steps
- Step directives: Execute per step, scoped to that step
- Variables from previous steps: Always available in later steps

### Development Workflows

**Adding Tools:**
```python
# In tools/{category}.py
from tsugite.tools import tool

@tool
def my_tool(param: str, optional: int = 10) -> dict:
    """Brief description.

    Args:
        param: What it does
        optional: Optional parameter

    Returns:
        Result dictionary
    """
    if not param:
        raise ValueError("param required")
    return {"output": do_work(param, optional)}

# Add to tools/__init__.py:
# from .category import my_tool  # noqa: F401
```

**Error conventions:**
- `ValueError` → bad inputs
- `RuntimeError` → execution failures
- Return dict/str; exceptions surfaced as strings to agent

**Safety:** `tools/shell.py` blocks dangerous patterns (`rm -rf /`, `sudo rm`, `dd if=`). Never relax.

**Creating Agents:**

Basic:
```markdown
---
name: my_agent
model: ollama:qwen2.5-coder:7b
max_steps: 5
tools: [read_file, write_file]
---
Task: {{ user_prompt }}
```

Chat:
```markdown
---
name: my_chat
model: openai:gpt-4o
text_mode: true
tools: [read_file, web_search]
---
{% if chat_history %}
{% for turn in chat_history %}
**User:** {{ turn.user_message }}
**Assistant:** {{ turn.agent_response }}
{% endfor %}
{% endif %}
{{ user_prompt }}
```

Multi-step:
```markdown
---
name: researcher
max_steps: 10
tools: [web_search, write_file]
---
Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="data" -->
Research {{ user_prompt }}.

<!-- tsu:step name="summarize" assign="summary" -->
Summarize: {{ data }}

<!-- tsu:step name="save" -->
Write {{ summary }} to file.
```

### Testing

**Commands:**
```bash
uv run pytest                              # All tests
uv run pytest tests/test_file.py::test_x  # Specific test
uv run pytest --cov=tsugite                # Coverage
```

**Patterns:**
```python
from tsugite.md_agents import parse_agent_file
from tsugite.builtin_agents import get_builtin_default_agent, is_builtin_agent
from tsugite.agent_inheritance import find_agent_file
from tsugite.chat import ChatManager
from tsugite.tools import get_tool

# Agent parsing
agent = parse_agent_file(Path("agent.md"))
assert agent.config.name == "expected"

# Built-in agents
assert is_builtin_agent("builtin-default")
agent = get_builtin_default_agent()

# Resolution
path = find_agent_file("builtin-default", Path.cwd())
assert str(path) == "<builtin-default>"

# Chat
manager = ChatManager(Path("agent.md"), max_history=10)
response = manager.run_turn("Hello")
assert len(manager.conversation_history) == 1

# Tools
tool = get_tool("read_file")
assert tool is not None
```

### Code Standards

**Python 3.12+**, type hints on public APIs

**Style:**
- Imports: stdlib → third-party → local (blank lines between)
- Line length: 88 characters (Black)
- Docstrings: `Args:` / `Returns:` for public functions

**Errors:**
- `ValueError` → invalid input/config
- `RuntimeError` → execution failure (include component name)

**Safety:**
- Shell tool guard: Never relax dangerous patterns blocklist
- MCP: Requires `--trust-mcp-code` flag
- File ops: Use `Path.resolve()` to prevent traversal

**Testing:** Test-first workflow. Add minimal failing test, run target, widen to full suite before merge.

### Rendering & Context

**Jinja2:** Undefined variables raise immediately. Use prefetch or tool directives to ensure variables exist.

**Helpers:** `now()`, `today()`, `slugify()`, `file_exists()`, `read_text()`, `is_file()`, `is_dir()`, `env.*`

**Prefetch:** Reduces token use by fetching data before LLM sees prompt. Failures assign `None`—guard with `{% if var %}`.

**Tool Directives:** More flexible than prefetch for multi-step. Scoped per-step, can access step variables.

**Best practice:** Prefetch for initial setup, tool directives for dynamic per-step data.
