# AGENTS.md

This file provides guidance to AI agents (tsugite, Claude Code, etc.) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_file.py::test_function

# Run tests with coverage
uv run pytest --cov=tsugite --cov-report=html

# Lint and format
uv run black .
uv run ruff check .
uv run pylint tsugite
```

### Testing Agents
```bash
# Run a basic agent (both `tsugite` and `tsu` aliases work)
uv run tsugite run examples/simple_variable_injection.md "test task"

# Run with specific model
uv run tsu run +agent "task" --model openai:gpt-4o-mini

# Preview agent rendering (see what LLM receives)
uv run tsu render agent.md "task"

# Validate agent schema
uv run tsu validate agents/*.md
```

### Schema Management
```bash
# Regenerate JSON schema after modifying AgentConfig
uv run python scripts/regenerate_schema.py
```

## Architecture

Tsugite is an agentic CLI that executes AI agents defined as markdown files with YAML frontmatter. The architecture follows a pipeline pattern: parse → prepare → render → execute.

### Core Execution Flow

1. **CLI Entry** (`tsugite/cli/__init__.py`)
   - Typer-based CLI with subcommands
   - Main commands: `run` (single-shot), `chat` (interactive), `render` (preview)
   - Additional: `daemon`, `workspace`, `init`, `validate`, `benchmark`, `mcp`, `serve`, `agents`, `config`, `attachments`, `cache`, `tools`, `history`

2. **Agent Resolution & Inheritance** (`tsugite/agent_inheritance.py`)
   - Resolves agent names to file paths using search order:
     - Workspace agents directory (if workspace provided)
     - `.tsugite/{name}.md` (project-local)
     - `agents/{name}.md` (project convention)
     - `./{name}.md` (current directory)
     - `builtin_agents/{name}.md` (package-provided)
     - Global directories (`~/.config/tsugite/agents/`)
   - Merges agent inheritance chains (scalars override, lists merge, dicts deep merge)

3. **Agent Parsing** (`tsugite/md_agents.py`)
   - Parses YAML frontmatter → `AgentConfig` (Pydantic model)
   - Extracts markdown content
   - Parses directives:
     - `<!-- tsu:step -->` - Multi-step workflow stages
     - `<!-- tsu:ignore -->` - Documentation blocks (stripped before LLM)
     - `<!-- tsu:tool -->` - Execute tools during rendering

4. **Agent Preparation** (`tsugite/agent_preparation.py`)
   - **Unified preparation pipeline** (used by both `run` and `render` commands)
   - Executes prefetch tools (frontmatter `prefetch` field)
   - Executes tool directives (inline `<!-- tsu:tool -->`)
   - Builds template context (tasks, variables, helpers)
   - Renders Jinja2 templates
   - Expands tool globs and categories
   - Loads auto_load_skills
   - Builds system prompt with tools and instructions

5. **Template Rendering** (`tsugite/renderer.py`)
   - Jinja2 environment with custom helpers:
     - `{{ user_prompt }}`, `{{ tasks }}`, `{{ task_summary }}`
     - `{{ now() }}`, `{{ today() }}`, `{{ yesterday() }}`, `{{ tomorrow() }}`
     - `{{ slugify(text) }}`, `{{ cwd() }}`
     - `{{ file_exists(path) }}`, `{{ is_file(path) }}`, `{{ is_dir(path) }}`, `{{ read_text(path) }}`
   - Context also includes: `env` (os.environ), `datetime`, `timedelta`
   - Supports step-scoped variables for multi-step agents
   - Strips ignored sections before LLM execution

6. **Agent Execution** (`tsugite/agent_runner/runner.py`)
   - Orchestrates multi-step or single-step execution
   - Manages step metrics (duration, success/failure)
   - Handles looping steps (repeat_while/repeat_until)
   - Supports continue_on_error and per-step timeouts
   - Integrates with history system for conversation continuity

7. **LLM Agent Loop** (`tsugite/core/agent.py`)
   - `TsugiteAgent` - Custom agent using LiteLLM directly
   - Think-Code-Observation loop (max_turns iterations)
   - Supports reasoning models (o1, o3, Claude extended thinking)
   - Direct control over model parameters (temperature, reasoning_effort)
   - Code execution via `LocalExecutor`

8. **Tool System** (`tsugite/tools/`)
   - Tool registry with built-in tools (fs, http, shell, tasks, agents, memory, skills, history, interactive)
   - Category system: `@fs`, `@http`, `@shell`, `@tasks`, `@agents`, `@memory`, `@skills`, `@history`, `@interactive`
   - Custom shell tools (config-based command wrappers)
   - MCP integration (`tsugite/mcp_client.py`)
   - Tool expansion supports globs (`*_search`) and exclusions (`-delete_file`)

9. **Event System** (`tsugite/events/`)
   - Event-driven architecture for UI decoupling
   - 24 event types: execution, LLM, meta, progress, skills
   - `EventBus` dispatches to multiple handlers
   - Handlers: Rich console, plain text, JSONL, Textual TUI, chat, REPL

10. **History System** (`tsugite/history/`)
    - JSONL-based conversation history (one `.jsonl` file per conversation)
    - Stores turns (user prompts, assistant responses)
    - Supports continuation (`--continue` flag, optional `--conversation-id`)
    - Indexed by conversation_id, agent_name, machine_name

11. **Configuration** (`tsugite/config.py`)
    - XDG-compliant paths: `~/.config/tsugite/` or `$XDG_CONFIG_HOME/tsugite/`
    - JSON config with model aliases, default settings
    - Attachment storage (reusable context)
    - MCP server registration
    - Auto-context discovery (CLAUDE.md, AGENTS.md, CONTEXT.md)

### Key Data Structures

**AgentConfig** (`md_agents.py:43`)
- Pydantic model for agent frontmatter
- Validates all fields (name, model, tools, max_turns, etc.)
- Schema exported to `tsugite/schemas/agent.schema.json`

**PreparedAgent** (`agent_preparation.py:17`)
- Dataclass containing everything for execution/display
- Ensures `render` shows exactly what `run` executes
- Contains: agent, config, system_message, user_message, tools, context

**Tool** (`core/tools.py`)
- Unified tool interface for LLM function calling
- Supports async/sync execution
- Parameters with JSON schema validation

**Agent** (`md_agents.py`)
- Parsed agent with content and step directives
- Steps contain: name, assign variable, temperature, timeout, continue_on_error, repeat conditions

### Multi-Step Agents

Multi-step agents use `<!-- tsu:step -->` directives to create workflows:

```markdown
<!-- tsu:step name="research" assign="findings" timeout="60" -->
Research topic

<!-- tsu:step name="summarize" assign="summary" repeat_while="has_pending_tasks" -->
Summarize findings
```

**Step Execution:**
- Preamble (content before first step) rendered once, shared across steps
- Each step has isolated execution context
- Variables persist across steps (findings → summary)
- Supports timeouts, continue_on_error, repeat conditions
- Metrics tracked per step (duration, status, errors)

**Looping Steps:**
- `repeat_while` / `repeat_until` with Jinja2 conditions
- Helper conditions: `has_pending_tasks`, `all_tasks_complete`, etc.
- Safety: `max_iterations` (default 10) prevents infinite loops
- Loop context: `{{ iteration }}`, `{{ max_iterations }}`, `{{ is_looping_step }}`

### Event-Driven UI

All UI output goes through the event system:

**Event Types** (events/base.py:9):
- Execution: TASK_START, STEP_START, CODE_EXECUTION, OBSERVATION, FINAL_ANSWER, ERROR
- LLM: LLM_MESSAGE, REASONING_CONTENT, REASONING_TOKENS
- Meta: COST_SUMMARY, STREAM_CHUNK, STREAM_COMPLETE, INFO, WARNING, DEBUG_MESSAGE
- Skills: SKILL_LOADED, SKILL_UNLOADED, SKILL_LOAD_FAILED
- Progress: STEP_PROGRESS, FILE_READ

**Handlers** (ui/):
- `base.py` - UIHandler interface
- `plain.py` - Plain text (no colors, copy-paste friendly)
- `jsonl.py` - JSONL protocol (for subprocess subagents)
- `textual_chat.py` - Textual TUI for chat mode
- `chat.py`, `repl_handler.py` - Interactive chat/REPL modes

### Agent Spawning (Multi-Agent)

Agents can spawn other agents using the `spawn_agent` tool:

```bash
# Multi-agent mode: primary +helper1 +helper2
tsugite run +coordinator +researcher +writer "task"
```

**Visibility control:**
- `visibility: public` (default) - Can be spawned by any agent
- `visibility: private` - Requires explicit allow list (multi-agent mode)
- `visibility: internal` - Helper agents, not meant to be public
- `spawnable: false` - Hard block, cannot be spawned (safety)

**Subagent context:**
- `{{ is_subagent }}` - Boolean
- `{{ parent_agent }}` - Parent agent name

### Prompt Caching

Automatic caching for supported providers (OpenAI, Anthropic, Bedrock, Deepseek):

- Attachments sent as separate system content blocks with `cache_control` markers
- Skills also receive cache markers for reuse across turns
- Cache markers added automatically by `tsugite/core/agent.py`

### Multi-Modal Attachments

Tsugite supports vision (images), audio, and document understanding through the attachment system.

**Supported Content Types:**
- **Images**: JPEG, PNG, GIF, WebP, SVG, BMP, TIFF
- **Audio**: MP3, WAV, OGG, M4A, FLAC (base64 encoded)
- **Documents**: PDF, Word (DOCX/DOC), Excel (XLSX/XLS), PowerPoint (PPTX/PPT)
- **Text**: Plain text, HTML (converted to markdown), JSON, XML, etc.

**How It Works:**

1. **URL Attachments** (Images/Documents):
   - LiteLLM fetches the URL directly (no download overhead)
   - Example: `tsu run -f https://example.com/chart.png "Describe this chart"`
   - Automatically detected via HTTP HEAD request

2. **Local File Attachments**:
   - Images: Read as bytes, base64 encoded
   - Documents: Read as bytes, base64 encoded
   - Text: Read as UTF-8
   - Example: `tsu run -f image.jpg "What's in this image?"`

3. **LLM Integration**:
   - Images: Sent as `image_url` content blocks (URL or data URI)
   - Audio: Sent as `input_audio` content blocks (base64)
   - Documents: Sent as `file` content blocks (URL or data URI)
   - Text: Sent as `text` content blocks (wrapped in XML tags)

**Examples:**

```bash
# Analyze an image from URL
tsu run -f https://example.com/chart.png "Describe this chart"

# Analyze a local image
tsu run -f screenshot.png "What error is shown?"

# Analyze a PDF
tsu run -f report.pdf "Summarize this report"

# Multiple attachments
tsu run -f image1.jpg -f image2.jpg "Compare these images"
```

**Model Support:**
- Vision: Claude 3.5 Sonnet, GPT-4o, GPT-4 Turbo, Gemini Pro Vision, etc.
- Audio: GPT-4o Audio Preview
- Documents: Claude (Bedrock/API), GPT-4o, Gemini, Bedrock Anthropic models

**Implementation Details:**
- `Attachment` dataclass: `tsugite/attachments/base.py`
- URL handler: `tsugite/attachments/url.py` (detects content type via HEAD request)
- File handler: `tsugite/attachments/file.py` (detects type by extension)
- Auto-context handler: `tsugite/attachments/auto_context.py` (discovers CONTEXT.md, AGENTS.md, CLAUDE.md)
- YouTube handler: `tsugite/attachments/youtube.py` (fetches transcripts)
- LLM formatting: `tsugite/core/agent.py:_format_attachment()`

## Development Patterns

### Adding a New Built-in Agent

1. Create `tsugite/builtin_agents/my_agent.md`
2. Add frontmatter with `name`, `tools`, `description`
3. Write agent template
4. No code changes needed - auto-discovered
5. Add tests in `tests/test_builtin_agents.py`

### Adding a New Tool

1. Create tool function in `tsugite/tools/`
2. Register in tool registry (`tools/__init__.py`)
3. Define parameters with JSON schema
4. Add to category if applicable (`@fs`, `@http`, etc.)
5. Add tests in `tests/test_*_tools.py`

### Adding a New Event Type

1. Add to `EventType` enum (`events/base.py`)
2. Create event class in `events/events.py`
3. Update UI handlers to handle new event
4. Add tests in `tests/events/`

### Modifying AgentConfig

1. Update `AgentConfig` Pydantic model (`md_agents.py`)
2. Regenerate schema: `uv run python scripts/regenerate_schema.py`
3. Update documentation in CLAUDE.md (user guide section)
4. Add validation tests in `tests/test_agent_parser.py`

## Testing Strategy

- **TDD preferred**: Write tests first when building new features. Tests don't need to be elaborate — simple tests that verify the expected behavior are enough. This creates a fast feedback loop and catches integration issues early (e.g., contextvar propagation, tool registration) before manual testing.
- **Unit tests**: Individual functions and classes
- **Pipeline tests**: Mock only `TsugiteAgent`/`litellm.acompletion` but exercise the full pipeline (parsing → rendering → preparation → tool expansion → execution). Most tests in the suite are this style.
- **Smoke tests**: `tests/smoke_test.sh` hits a real LLM API (requires `OPENAI_API_KEY`, not run in CI)
- **Fixtures**: `conftest.py` provides shared test data
- **Mocking**: Use `@pytest.fixture` for LLM responses
- **Async tests**: Mark with `@pytest.mark.asyncio`
- **Coverage target**: 80%+ (check with `--cov-report=html`)

## Code Style

- Python 3.11+ (type hints on public APIs)
- Line length: 120 characters (Black)
- Import order: stdlib → third-party → local (blank lines between groups)
- Errors: `ValueError` for input validation, `RuntimeError` for execution failures
- Docstrings: Google style with `Args:` and `Returns:` sections
- Pydantic models: Use `extra="forbid"` to catch typos in YAML frontmatter

## Common Pitfalls

1. **Don't break render/run parity**: Changes to rendering must update BOTH `renderer.py` and `agent_preparation.py`
2. **Don't emit print/console directly**: Use event system (`EventBus.emit()`)
3. **Don't forget schema regen**: After modifying `AgentConfig`, regenerate schema
4. **Don't hardcode paths**: Use XDG utilities (`get_xdg_config_path()`, etc.)
5. **Don't use blocking IO in async**: Use `asyncio.to_thread()` for sync tools
6. **Test both sync and async paths**: Many tools support both execution modes
7. **Don't embed prompts in adapters/code**: Use context variables + conditional blocks in `default.md` instead. Add new context vars in `_build_agent_context()` (base adapter) and default them in `agent_preparation.py`, then use `{% if var %}` in the agent template. This keeps all prompt content in one place and leverages the existing rendering pipeline.

## Code Review Policy

- **Automatically run the `code-simplifier:code-simplifier` agent** (via the Task tool) after implementing any changes, without waiting for the user to ask. This catches duplication, unnecessary complexity, and keeps the codebase DRY.
- When making implementation plans, include a final "simplify/review" step that uses the code-simplifier agent on all modified files.
