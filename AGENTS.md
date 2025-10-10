# Agent Development Guide

Quick reference for building, composing, and running Tsugite agents.

## Table of Contents

- [Quick Commands](#quick-commands)
- [Context Injection](#context-injection) - File references, attachments, custom tools
- [Agent Configuration](#agent-configuration) - Frontmatter, tools, models
- [Multi-Step Agents](#multi-step-agents) - Step directives, variables
- [Directives](#directives) - Ignore blocks, tool directives
- [Development](#development) - Testing, linting, project structure
- [Advanced Features](#advanced-features) - MCP, Docker, reasoning models

## Quick Commands

### Basic Usage

| Task | Example |
| --- | --- |
| Run agent | `tsugite run agent.md "task"` or `tsugite run +assistant "task"` |
| Debug prompt | `tsugite run agent.md "task" --debug` |
| Plain output | `tsugite run +assistant "task" --plain` (or pipe: `\| grep result`) |
| Headless (CI) | `tsugite run +assistant "task" --headless` |

### Multi-Agent Composition

```bash
# Inline agents
tsugite run +assistant +jira +coder "fix bug #123"

# Explicit helpers
tsugite run +assistant --with-agents "jira coder" "task"

# Mixed (files + shortcuts)
tsugite run agents/custom.md +jira ./helpers/coder.md "task"
```

### MCP Servers

```bash
# Register servers
tsugite mcp add basic-memory --url http://localhost:3000/mcp
tsugite mcp add pubmed --command uvx --args "pubmedmcp@0.1.3"

# Manage
tsugite mcp list
tsugite mcp show basic-memory
tsugite mcp test basic-memory --trust-code
```

### Custom Tools

```bash
# Add shell command wrapper
tsugite tools add file_search \
  -c "rg {pattern} {path}" \
  -d "Search files" \
  -p pattern:required \
  -p path:.

# Manage
tsugite tools list
tsugite tools validate
tsugite tools check agent.md
```

See `CUSTOM_TOOLS.md` for full guide on creating shell tool wrappers.

## Context Injection

### File References (`@filename`)

Inject file contents directly into prompts:

```bash
tsugite run +assistant "Review @src/main.py"
tsugite run +coder "Fix @app.py based on @tests/test_app.py"
tsugite run +writer "Summarize @'docs/user guide.md'"  # Quote paths with spaces
```

Files are read and formatted automatically. **Note**: `@word` patterns trigger file lookup.

### Attachments (Reusable Context)

Store and reuse context from files, URLs, or YouTube:

```bash
# Create attachments
tsugite attachments add coding-standards docs/style-guide.md
tsugite attachments add api-docs https://api.example.com/docs
tsugite attachments add tutorial https://youtube.com/watch?v=abc123
cat notes.txt | tsugite attachments add notes -

# Use in prompts
tsugite run +assistant "task" -f coding-standards -f api-docs
tsugite run +coder "review @app.py" -f coding-standards

# One-time use (without saving)
tsugite run +assistant "task" -f ./config.json -f https://example.com/doc.md

# Manage
tsugite attachments list
tsugite cache clear
```

**Agent-defined attachments:**

```markdown
---
name: code_reviewer
tools: [read_file, write_file]
attachments:
  - coding-standards
  - security-checklist
---

Task: {{ user_prompt }}
```

**Resolution order:** Agent attachments → CLI attachments (`-f`) → File refs (`@`) → Prompt

See [Attachment Resolution](#attachment-resolution-order) for details.

### Custom Tools

Define shell command wrappers without writing Python:

**Global** (`~/.config/tsugite/custom_tools.yaml`):

```yaml
tools:
  - name: file_search
    description: Search files with ripgrep
    command: "rg {pattern} {path}"
    parameters:
      pattern: {required: true}  # Type inferred as str
      path: "."                   # Type inferred from default

  - name: count_lines
    command: "wc -l {file}"
    parameters:
      file: {required: true}
```

**Per-agent** (frontmatter):

```markdown
---
name: researcher
tools: [file_search]
custom_tools:
  - name: local_grep
    command: "grep -r {pattern} ."
    parameters:
      pattern: {required: true}
---
```

**CLI**:

```bash
tsugite tools add find_files \
  -c "find {path} -name {pattern}" \
  -p pattern:required \
  -p path:.
```

See `CUSTOM_TOOLS.md` for parameter types, validation, troubleshooting.

## Agent Configuration

### Frontmatter Reference

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `name` | ✅ | — | Agent identifier |
| `model` | ❌ | `ollama:qwen2.5-coder:7b` | `provider:model[:variant]` |
| `max_steps` | ❌ | `5` | Reasoning iterations |
| `tools` | ❌ | `[]` | Tool names (supports `@category`, globs, `-exclusions`) |
| `custom_tools` | ❌ | `[]` | Per-agent shell tool definitions |
| `prefetch` | ❌ | `[]` | Tools to run before rendering |
| `attachments` | ❌ | `[]` | Context to auto-load |
| `instructions` | ❌ | — | Extra system guidance |
| `mcp_servers` | ❌ | — | MCP servers + tool safelists |
| `context_budget` | ❌ | Unlimited | Prompt length cap |

### Tool Selection

```markdown
---
tools:
  - read_file              # Exact name
  - write_file
  - @fs                    # All filesystem tools
  - "*_search"             # Glob pattern
  - -delete_file           # Exclude specific tool
  - -@dangerous            # Exclude category
---
```

### Template Helpers

| Helper | Purpose |
| --- | --- |
| `{{ user_prompt }}` | CLI prompt argument |
| `{{ now() }}` | ISO 8601 timestamp |
| `{{ today() }}` | `YYYY-MM-DD` date |
| `{{ slugify(text) }}` | Filesystem-safe slug |
| `{{ file_exists(path) }}` | Path existence check |
| `{{ is_file(path) }}` / `{{ is_dir(path) }}` | Type checks |
| `{{ read_text(path, default="") }}` | Safe file read |
| `{{ env.HOME }}` | Environment variables |

### Prefetch

Execute tools before rendering; results available in templates:

```markdown
---
prefetch:
  - tool: read_file
    args: {path: "config.json"}
    assign: config
  - tool: list_files
    args: {directory: "src"}
    assign: source_files
---

Config: {{ config }}
Files: {{ source_files }}
```

Failures assign `None`—guard with `{% if variable %}`. For complex workflows, prefer [tool directives](#tool-directives-tsu tool).

## Multi-Step Agents

Chain steps with variable passing:

```markdown
---
name: research_writer
max_steps: 10
tools: [write_file, web_search]
---

Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="findings" -->
Research {{ user_prompt }} and capture 3-5 key insights.

<!-- tsu:step name="outline" assign="structure" -->
Create article outline from:
{{ findings }}

<!-- tsu:step name="write" assign="article" -->
Draft article following {{ structure }}.

<!-- tsu:step name="save" -->
Write article to `output.md` and confirm success.
```

**Key points:**
- Content before first directive = preamble (shared across steps)
- Step context: `{{ step_number }}`, `{{ total_steps }}`, `{{ step_name }}`
- Variables persist: `findings` and `structure` available in later steps
- Execution stops on first failure

## Directives

### Documentation Blocks: `<!-- tsu:ignore -->`

Strip content before sending to LLM (for developer notes):

```markdown
---
name: data_processor
tools: [read_file, write_file]
---

<!-- tsu:ignore -->
## Documentation

Processes CSV files and generates reports.

**Usage:** `tsugite run data_processor.md "Analyze sales"`

**Variables:**
- `user_prompt`: Task description
- `data`: CSV content (from tool directive)

Template patterns: {{ undeclared_var }}  # Won't cause errors!
<!-- /tsu:ignore -->

# Data Processing Agent

Task: {{ user_prompt }}
```

**Features:**
- Multiple blocks supported
- Can contain Jinja2 syntax without errors
- Stripped before rendering

### Tool Directives: `<!-- tsu:tool -->`

Execute tools during rendering (before LLM sees prompt):

```markdown
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->
<!-- tsu:tool name="read_file" args={"path": "schema.json"} assign="schema" -->

Current config:
```json
{{ config }}
```

Schema:
```json
{{ schema }}
```
```

**Multi-step support:**

```markdown
<!-- tsu:tool name="read_file" args={"path": "sources.txt"} assign="sources" -->

Available sources: {{ sources }}

<!-- tsu:step name="research" assign="findings" -->

<!-- tsu:tool name="read_file" args={"path": "context.txt"} assign="context" -->

Research using context: {{ context }}
And sources: {{ sources }}
```

**Scoping:**
- Preamble directives: Execute once, available to all steps
- Step directives: Execute per step, scoped to that step
- Variables from previous steps always available

**vs. Prefetch:**

| Feature | Prefetch (YAML) | Tool Directives |
|---------|-----------------|-----------------|
| Location | Frontmatter | Inline content |
| Visibility | Global | Context-specific |
| Multi-step | Once | Per-step capable |

**Error handling:** Failed tools assign `None`. Check with `{% if variable %}`.

Examples: `examples/tool_directives_demo.md`, `examples/multistep_with_directives.md`

## Development

### Setup & Testing

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest                              # All tests
uv run pytest tests/test_file.py::test_x  # Specific test

# Lint & format
uv run ruff check .
uv run black .
```

### Project Structure

- CLI entrypoint: `tsugite/tsugite.py`
- Agent parsing: `tsugite/md_agents.py`
- Template rendering: `tsugite/renderer.py`
- Execution: `tsugite/agent_runner.py`
- Tools: `tsugite/tools/` (ensure side-effect imports)
- Custom tools: `tsugite/tools/shell_tools.py`, `tsugite/shell_tool_config.py`

### Style Guidelines

- Python 3.12+; type hints on public APIs
- Imports: stdlib → third-party → local (blank line between groups)
- Line length: 88 characters (Black)
- Errors: `ValueError` for invalid input, `RuntimeError` for execution failures
- Docstrings: concise with `Args` / `Returns` for public functions

## Advanced Features

### Model Providers

```markdown
---
model: provider:model[:variant]
---
```

**Supported providers** (via LiteLLM): `ollama`, `openai`, `anthropic`, `google`, `github_copilot`, etc.

**API keys required:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### Reasoning Models (o1, o3, Claude)

**Full reasoning visible:**
- Claude (`claude-3-7-sonnet`): Extended thinking shown
- Deepseek: Reasoning exposed

**Token counts only:**
- OpenAI o1/o3: Content hidden, shows "🧠 Used N reasoning tokens"

**Control reasoning effort** (o1, o1-preview, o3, o3-mini only; NOT o1-mini):

```markdown
---
model: openai:o1
reasoning_effort: high  # Options: low, medium, high
---
```

**Per-step control:**

```markdown
<!-- tsu:step name="quick_analysis" reasoning_effort="low" -->
<!-- tsu:step name="deep_think" reasoning_effort="high" -->
```

**Parameter limitations (o1/o3):**
- Not supported: `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`
- Supported: `max_completion_tokens`, `messages`, `stream`

Tsugite auto-filters unsupported parameters.

### MCP Integration

```markdown
---
mcp_servers:
  basic-memory: null              # All tools
  pubmed: [search, fetch_article] # Specific tools only
---
```

Run with `--trust-mcp-code` to execute remote code.

See [MCP Commands](#mcp-servers) for registration.

### Docker Execution (Optional)

Run agents in isolated containers. **Completely optional**—no Docker dependencies in core.

```bash
# Build runtime (one-time)
docker build -f Dockerfile.runtime -t tsugite/runtime .

# Run in container
tsugite run --docker agent.md "task"
tsugite run --docker --network none agent.md "task"

# Or use wrapper directly
tsugite-docker run agent.md "task"

# Persistent sessions
tsugite-docker-session start my-work
tsugite-docker --container my-work run agent.md "task 1"
tsugite-docker --container my-work run agent.md "task 2"
tsugite-docker-session stop my-work
```

**Use cases:** Untrusted agents, reproducible environments, debugging

See `bin/README.md` for complete Docker documentation.

### Multi-Agent Composition

```bash
tsugite run +assistant +coder "task"
```

- First agent = coordinator
- Receives `spawn_{agent}()` tools for each helper
- Agent resolution: `.tsugite/{name}.md` → `agents/{name}.md` → `./{name}.md` → `~/.config/tsugite/agents/{name}.md`

## Additional Resources

- Examples: `agents/examples/`, `docs/test_agents/`
- Custom tools: `CUSTOM_TOOLS.md`
- Docker: `bin/README.md`
- Design docs: `docs/` (permissions, history, task tracking)
