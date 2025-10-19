# Agent Development Guide

Quick reference for building, composing, and running Tsugite agents.

## Commands

| Task | Command |
|------|---------|
| Run agent | `tsugite run agent.md "task"` or `tsugite run +name "task"` |
| Chat mode | `tsugite chat` or `tsugite chat +agent` |
| Debug prompt | `tsugite run agent.md "task" --debug` |
| Plain output | `tsugite run +agent "task" --plain` |
| Headless (CI) | `tsugite run +agent "task" --headless` |
| Render only | `tsugite render agent.md "task"` |
| Multi-agent | `tsugite run +coordinator +helper1 +helper2 "task"` |
| MCP register | `tsugite mcp add name --url http://host/mcp` |
| MCP test | `tsugite mcp test name --trust-code` |
| Custom tool | `tsugite tools add name -c "cmd {arg}" -p arg:required` |
| Config | `tsugite config set model provider:model` |
| Model alias | `tsugite config set model-alias fast openai:gpt-4o-mini` |

## Agent Structure

Agents are Markdown + YAML frontmatter:

```markdown
---
name: my_agent
model: ollama:qwen2.5-coder:7b
max_steps: 5
tools: [read_file, write_file]
---

Task: {{ user_prompt }}
```

### Frontmatter Fields

| Key | Default | Description |
|-----|---------|-------------|
| `name` | — (required) | Agent identifier |
| `model` | Config default | `provider:model[:variant]` |
| `max_steps` | `5` | Reasoning iterations |
| `tools` | `[]` | Tool names, globs (`*_search`), categories (`@fs`), exclusions (`-delete_file`) |
| `custom_tools` | `[]` | Per-agent shell command wrappers |
| `prefetch` | `[]` | Tools to run before rendering |
| `attachments` | `[]` | Context to auto-load |
| `instructions` | — | Extra system guidance |
| `extends` | Config default | Parent agent to inherit from |
| `mcp_servers` | — | MCP servers + tool safelists |
| `context_budget` | Unlimited | Prompt length cap |
| `reasoning_effort` | — | For o1/o3 models: `low`, `medium`, `high` |
| `text_mode` | `false` | Allow text responses without code blocks |

### Template Helpers

| Helper | Example |
|--------|---------|
| `{{ user_prompt }}` | CLI prompt argument |
| `{{ now() }}` | ISO 8601 timestamp |
| `{{ today() }}` | `YYYY-MM-DD` date |
| `{{ slugify(text) }}` | Filesystem-safe slug |
| `{{ file_exists(path) }}` | Boolean path check |
| `{{ is_file(path) }}` / `{{ is_dir(path) }}` | Type checks |
| `{{ read_text(path, default="") }}` | Safe file read |
| `{{ env.HOME }}` | Environment variables |

## Built-in Agents

### builtin-default

Minimal base agent with task tracking. Default base for inheritance.

```bash
tsugite run builtin-default "task"
```

### builtin-chat-assistant

Conversational agent for chat mode. Tools: `read_file`, `write_file`, `list_files`, `web_search`, `run`. Text mode enabled.

```bash
tsugite chat  # Uses this by default
```

### Overriding Built-ins

Create `.tsugite/chat_assistant.md` or `agents/chat_assistant.md` to override for your project.

## Agent Resolution Order

When referencing by name (e.g., `+myagent`):

1. Built-in agents (if name matches exactly)
2. `.tsugite/{name}.md`
3. `agents/{name}.md`
4. `./{name}.md`
5. `~/.config/tsugite/agents/{name}.md`

Explicit paths skip resolution: `tsugite run ./path/to/agent.md`

## Agent Inheritance

```yaml
---
name: specialized
extends: builtin-default  # Inherit from built-in
model: openai:gpt-4o       # Override model
tools: [read_file, run]    # Add tools
---
```

**Inheritance chain:** Default base → Extended → Current

**Merge rules:**
- Scalars (model, max_steps): Child overwrites
- Lists (tools): Merge + deduplicate
- Dicts (mcp_servers): Merge, child keys override
- Strings (instructions): Concatenate with `\n\n`

**Opt out:** `extends: none`

**Configure default base:**

```bash
tsugite config set default_base_agent builtin-default
tsugite config set default_base_agent none  # Disable
```

## Context Injection

### File References

```bash
tsugite run +agent "Review @src/main.py"
tsugite run +agent "Fix @app.py based on @tests/test.py"
```

### Attachments

```bash
# Create reusable context
tsugite attachments add standards project/STYLE.md
tsugite attachments add api-docs https://api.example.com/docs

# Use in prompts
tsugite run +agent "task" -f standards -f api-docs

# Agent-defined attachments
---
attachments:
  - standards
  - api-docs
---
```

**Resolution order:** Agent attachments → CLI `-f` → File refs `@` → Prompt

### Custom Tools

**Global:** `~/.config/tsugite/custom_tools.yaml`

```yaml
tools:
  - name: file_search
    description: Search with ripgrep
    command: "rg {pattern} {path}"
    parameters:
      pattern: {required: true}
      path: "."
```

**Per-agent:**

```yaml
---
custom_tools:
  - name: local_grep
    command: "grep -r {pattern} ."
    parameters:
      pattern: {required: true}
---
```

See `CUSTOM_TOOLS.md` for full guide.

### Prefetch

Execute tools before rendering:

```yaml
---
prefetch:
  - tool: read_file
    args: {path: "config.json"}
    assign: config
---

Config: {{ config }}
```

Failures assign `None`. Guard with `{% if config %}`.

## Interactive User Input

Tsugite provides two tools for collecting user input during agent execution, available only in interactive mode:

### ask_user - Single Question

Ask one question at a time:

```python
# Text input
name = ask_user(question="What is your name?", question_type="text")

# Yes/No question
save = ask_user(question="Save to file?", question_type="yes_no")
# Returns: "yes" or "no"

# Multiple choice (arrow key navigation)
format = ask_user(
    question="Choose format:",
    question_type="choice",
    options=["json", "txt", "md"]
)
```

**Question types:**
- `text`: Freeform text input
- `yes_no`: Binary yes/no question
- `choice`: Multiple choice with arrow key navigation (requires `options` list)

### ask_user_batch - Multiple Questions

For better UX when collecting multiple related inputs, use `ask_user_batch` to ask all questions at once:

```python
responses = ask_user_batch(questions=[
    {"id": "name", "question": "What is your name?", "type": "text"},
    {"id": "email", "question": "Email address?", "type": "text"},
    {"id": "save", "question": "Save to file?", "type": "yes_no"},
    {
        "id": "format",
        "question": "Choose format:",
        "type": "choice",
        "options": ["json", "txt", "md"]
    }
])

# Access responses by ID
user_name = responses["name"]
user_email = responses["email"]
should_save = responses["save"]  # "yes" or "no"
file_format = responses["format"]
```

**Question structure:**
- `id` (required): Unique identifier used as key in response dict
- `question` (required): The question text
- `type` (required): Question type - "text", "yes_no", or "choice"
- `options` (optional): List of options for "choice" type (minimum 2)

**Features:**
- Arrow key navigation (↑/↓) or vim-style (j/k) for choice questions
- Colored interface with highlighting
- Protection against accidental double-Enter
- Returns dict mapping question IDs to responses
- Validates question structure and enforces unique IDs

**Agent configuration:**

```yaml
---
name: user_registration
tools: [ask_user_batch, write_file]
---
```

Interactive tools are automatically filtered out in non-interactive mode (e.g., CI/headless). Check `{{ is_interactive }}` variable in templates.

## Multi-Step Agents

```markdown
---
name: researcher
max_steps: 10
tools: [web_search, write_file]
---

Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="findings" -->
Research {{ user_prompt }}.

<!-- tsu:step name="summarize" assign="summary" -->
Summarize: {{ findings }}

<!-- tsu:step name="save" -->
Write {{ summary }} to file.
```

**Key concepts:**
- Content before first directive = preamble (shared across steps)
- Step context: `{{ step_number }}`, `{{ total_steps }}`, `{{ step_name }}`
- Variables persist across steps
- Execution stops on first failure

**Step parameters:**

```markdown
<!-- tsu:step name="..." assign="..." temperature="0.7" reasoning_effort="high" -->
```

## Directives

### Documentation Blocks

Strip content before LLM sees it:

```markdown
<!-- tsu:ignore -->
## Developer Notes
This agent does X. Usage: `tsugite run agent.md "task"`
Can contain {{ undeclared_vars }} without errors.
<!-- /tsu:ignore -->
```

Multiple blocks supported. Useful for inline documentation.

### Tool Directives

Execute tools during rendering:

```markdown
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->

Config: {{ config }}
```

**Multi-step support:**

```markdown
<!-- tsu:tool name="read_file" args={"path": "sources.txt"} assign="sources" -->

<!-- tsu:step name="research" -->
<!-- tsu:tool name="read_file" args={"path": "context.txt"} assign="context" -->
Research using {{ context }} and {{ sources }}
```

**Scoping:**
- Preamble directives: Execute once, available to all steps
- Step directives: Execute per step, scoped to that step
- Previous step variables always available

**vs. Prefetch:** Tool directives are inline and support per-step execution. Prefetch is global in frontmatter.

## Model Providers

Format: `provider:model[:variant]`

**Examples:**
- `ollama:qwen2.5-coder:7b`
- `openai:gpt-4o`
- `anthropic:claude-3-5-sonnet`
- `google:gemini-pro`

**API keys:** Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

**Aliases:**

```bash
tsugite config set model-alias fast openai:gpt-4o-mini
tsugite run +agent "task" --model fast
```

## Reasoning Models

**Full reasoning visible:**
- Claude (`claude-3-7-sonnet`)
- Deepseek

**Token counts only:**
- OpenAI o1/o3 (shows "🧠 Used N reasoning tokens")

**Control effort** (o1, o1-preview, o3, o3-mini only):

```yaml
---
model: openai:o1
reasoning_effort: high  # low, medium, high
---
```

Per-step:

```markdown
<!-- tsu:step name="quick" reasoning_effort="low" -->
<!-- tsu:step name="deep" reasoning_effort="high" -->
```

**Limitations:** o1/o3 models don't support `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`.

## MCP Integration

```yaml
---
mcp_servers:
  basic-memory: null              # All tools
  pubmed: [search, fetch_article] # Specific tools only
---
```

**Register:**

```bash
tsugite mcp add basic-memory --url http://localhost:3000/mcp
tsugite mcp add pubmed --command uvx --args "pubmedmcp@0.1.3"
```

**Run:** Requires `--trust-mcp-code` for remote code execution.

## Multi-Agent Composition

```bash
tsugite run +coordinator +helper1 +helper2 "task"
```

First agent = coordinator. Gets `spawn_{name}()` tools for each helper.

## Development

### Setup

```bash
uv sync --dev
uv run pytest                              # All tests
uv run pytest tests/test_file.py::test_x  # Specific test
uv run black . && uv run ruff check .
```

### Project Structure

- `cli/__init__.py` - Typer CLI (run, chat, render, config, mcp, tools, etc.)
- `md_agents.py` - Agent parsing, directives
- `builtin_agents.py` - Built-in agent definitions
- `agent_inheritance.py` - Resolution + inheritance
- `agent_runner.py` - Execution orchestration
- `core/agent.py` - LiteLLM agent loop
- `renderer.py` - Jinja2 rendering
- `chat.py` - Chat session management
- `ui/textual_chat.py` - Textual TUI
- `tools/` - Tool registry
- `models.py` - Model string parsing
- `config.py` - Configuration

### Style

- Python 3.12+, type hints on public APIs
- Imports: stdlib → third-party → local (blank lines between)
- Line length: 88 characters (Black)
- Errors: `ValueError` for input, `RuntimeError` for execution failures
- Docstrings: `Args:` / `Returns:` for public functions

### Adding Built-in Agents

1. Add to `builtin_agents.py`:

```python
BUILTIN_NEW_CONTENT = """---
name: new_agent
tools: []
---
{{ user_prompt }}
"""

def get_builtin_new():
    from .md_agents import parse_agent
    return parse_agent(BUILTIN_NEW_CONTENT, Path("<builtin-new>"))
```

2. Update `is_builtin_agent()` to include `"builtin-new"`
3. Handle in `agent_inheritance.py:load_extended_agent()`
4. Handle in `md_agents.py:parse_agent_file()`
5. Add tests

### Testing

```python
from tsugite.md_agents import parse_agent_file
from tsugite.chat import ChatManager
from tsugite.tools import get_tool

def test_agent():
    agent = parse_agent_file(Path("agent.md"))
    assert agent.config.name == "expected"

def test_chat():
    manager = ChatManager(Path("agent.md"), max_history=10)
    response = manager.run_turn("Hello")
    assert len(manager.conversation_history) == 1

def test_tool():
    tool = get_tool("read_file")
    assert tool is not None
```

## Additional Resources

- **Custom tools:** `CUSTOM_TOOLS.md`
- **Docker:** `bin/README.md`
- **Examples:** `agents/examples/`
