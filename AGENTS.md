# Agent Development Guide

Use this guide as a grab-and-go reference for building, composing, and running Tsugite agents.

## Quick command crib sheet

| Task | Example |
| --- | --- |
| Run an agent | `tsugite run research_writer.md "AI in healthcare"`
| Inspect rendered prompt | `tsugite run research_writer.md "AI in healthcare" --debug`
| Plain output (copy-paste friendly) | `tsugite run +assistant "task" --plain`
| Piped output (auto-plain) | `tsugite run +assistant "task" \| grep result`
| Combine inline agents | `tsugite run +assistant +jira +coder "fix bug #123"`
| Provide helper agents explicitly | `tsugite run +assistant --with-agents "jira coder" "fix bug #123"`
| Trust MCP tool code | `tsugite run agent.md "task" --trust-mcp-code`
| Compose from files | `tsugite run agents/custom.md +jira ./helpers/coder.md "task"`

### MCP server management

| Action | Example |
| --- | --- |
| Register HTTP server | `tsugite mcp add basic-memory --url http://localhost:3000/mcp`
| Register stdio server | `tsugite mcp add pubmed --command uvx --args "--quiet" --args "pubmedmcp@0.1.3" --env "UV_PYTHON=3.12"`
| Force update existing entry | `tsugite mcp add basic-memory --url http://new-url.com/mcp --force`
| List / show servers | `tsugite mcp list`, `tsugite mcp show basic-memory`
| Smoke-test tools | `tsugite mcp test basic-memory --trust-code`

### Multi-agent shortcuts

* Coordinator with inline roster: `tsugite run +coordinator +researcher +writer +reviewer "Research AI trends and create a blog post"`
* Coordinator plus explicit roster flag: `tsugite run +coordinator --with-agents "researcher writer" "task"`
* Project manager pipeline: `tsugite run +project_manager +github +tester "Create feature branch, implement login, and run tests"`

### Output Modes

Tsugite supports different output modes for various use cases:

**Plain Mode (`--plain`):** Copy-paste friendly output without box-drawing characters

```bash
tsugite run +assistant "task" --plain
```

**Auto-Detection:** Plain mode automatically activates when:
- Output is piped or redirected (`tsugite run +assistant "task" | grep result`)
- `NO_COLOR` environment variable is set
- stdout is not a TTY

**Headless Mode (`--headless`):** For CI/scripts, result to stdout, progress to stderr

```bash
tsugite run +assistant "task" --headless
```

**Benefits:**
- **Plain mode**: Easy to copy/paste terminal output without box characters
- **Auto-detection**: Zero configuration for scripts and pipelines
- **Standards compliance**: Respects `NO_COLOR` environment variable
- **Flexibility**: Explicit `--plain` flag when auto-detection isn't desired

### File references

Use `@filename` syntax to automatically inject file contents into prompts:

tsugite run +assistant "Review @src/main.py and suggest improvements"
tsugite run +coder "Fix bugs in @app.py based on @tests/test_app.py"
tsugite run +assistant @README.md  # Single file as entire context
tsugite run +writer "Summarize @"docs/user guide.md""  # Quoted paths for spaces
tsugite run +assistant "Compare @file1.py and @file2.py"  # Multiple files

The files are automatically read and injected as formatted context. The CLI will show which files were expanded in the info panel.

**Note**: `@word` patterns that start with alphanumeric characters will be treated as file references. Use `@"path"` for explicit file references or avoid `@` prefix for non-file content (e.g., write "email user123" instead of "@user123").

### Attachments (reusable context)

Attachments are reusable content references that can point to files, URLs, YouTube videos, or inline text. They're stored as lightweight references with automatic caching, similar to `llm`'s attachments but with enhanced handler support.

**Managing Attachments:**

# Add attachments from files, URLs, or stdin
tsugite attachments add coding-standards docs/style-guide.md  # File reference
tsugite attachments add api-docs https://api.example.com/reference  # URL reference
tsugite attachments add tutorial https://youtube.com/watch?v=abc123  # YouTube transcript
cat context.txt | tsugite attachments add mycontext -  # Inline text (stdin)

# List all attachments
tsugite attachments list

# Show attachment details (displays type, cache status)
tsugite attachments show coding-standards
tsugite attachments show coding-standards --content  # Fetch and show full content

# Search attachments
tsugite attachments search "api"

# Remove attachments
tsugite attachments remove old-docs

# Cache management
tsugite cache list  # Show all cached attachments
tsugite cache info coding-standards  # Show cache details for specific attachment
tsugite cache clear  # Clear entire cache
tsugite cache clear coding-standards  # Clear cache for specific attachment

**Using attachments in prompts**

# Use saved attachments with -f flag
tsugite run +assistant "implement login" -f coding-standards -f api-docs

# Mix attachments with file references
tsugite run +coder "review @app.py" -f coding-standards

# Use direct files/URLs without saving (one-time use)
tsugite run +assistant "check" -f ./config.json
tsugite run +assistant "summarize" -f https://example.com/doc.md

# Multiple attachments
tsugite run +reviewer "check PR" -f style -f security-guide -f api-docs

# Force refresh cached content
tsugite run +assistant "update" -f api-docs --refresh-cache

**How it works**

- **Storage**: Attachments registry stored in `~/.tsugite/attachments.json` (or `$XDG_CONFIG_HOME/tsugite/attachments.json`)
- **Caching**: Downloaded content cached in `~/.cache/tsugite/attachments/` (or `$XDG_CACHE_HOME/tsugite/attachments/`)
- **Handler System**: Auto-detects content type from source:
  - **Inline**: stdin text stored directly
  - **File**: local file paths (absolute or relative)
  - **YouTube**: `youtube.com` or `youtu.be` URLs → transcript with timestamps
  - **URL**: HTTP(S) URLs (HTML auto-converted to markdown)
- **Cache behavior**: Permanent cache with manual refresh via `--refresh-cache` flag
- **Sync**: Sync `~/.tsugite/` (registry only) with Nextcloud/Dropbox for cross-machine access

**Output Format:**

<Attachment: coding-standards>
[style guide content]
</Attachment: coding-standards>

<Attachment: api-docs>
[api documentation]
</Attachment: api-docs>

<File: app.py>
[file content from @app.py]
</File: app.py>

Task: review app.py following coding-standards and api-docs

**Agent-Defined Attachments:**

Agents can specify attachments in their frontmatter, so they always load required context automatically:

---
name: code_reviewer
model: openai:gpt-4o-mini
tools: [read_file, write_file]
attachments:
  - coding-standards
  - security-checklist
---

You are a code reviewer. Follow our standards when reviewing code.

Task: {{ user_prompt }}

**Attachment Resolution Order:**

1. **Agent attachments** (from agent definition) - prepended first
2. **CLI attachments** (from `-f` flag) - prepended after agent attachments
3. **File references** (from `@filename`) - prepended after attachments
4. **User prompt** - comes last

Example:
tsugite run code_reviewer.md "review @app.py" -f extra-context

Results in:
<Attachment: coding-standards>      # From agent definition
...
</Attachment: coding-standards>

<Attachment: security-checklist>     # From agent definition
...
</Attachment: security-checklist>

<Attachment: extra-context>          # From CLI -f flag
...
</Attachment: extra-context>

<File: app.py>                     # From @filename
...
</File: app.py>

Task: review app.py               # User prompt

### Docker Container Execution (Optional)

Run agents in isolated Docker containers for safety and reproducibility using **optional wrapper scripts**.

These scripts are **completely separate** from tsugite core—no Docker dependencies in the main codebase. The wrapper scripts (`tsugite-docker` and `tsugite-docker-session`) are automatically installed as console scripts when you install tsugite.

**Setup:**

# Build runtime image (one-time setup)
docker build -f Dockerfile.runtime -t tsugite/runtime .

# Wrapper scripts are automatically available after installation:
# - tsugite-docker
# - tsugite-docker-session

**Two Usage Patterns:**

You can use either the integrated `--docker` flag or call the wrapper directly:

# Pattern 1: Integrated flag (convenient)
tsugite run --docker agent.md "task"
tsugite run --docker --network none agent.md "task"
tsugite run --docker --keep agent.md "task"
tsugite run --container my-session agent.md "task"

# Pattern 2: Direct wrapper call (explicit)
tsugite-docker run agent.md "task"
tsugite-docker --network none run agent.md "task"
tsugite-docker --keep run agent.md "task"
tsugite-docker --container my-session run agent.md "task"

Both patterns work identically—the first delegates to the second. Use whichever feels more natural.

**All tsugite flags work transparently:**

tsugite run --docker agent.md "task" --debug --verbose
tsugite-docker run +assistant "query" -f context.md
tsugite run --docker agent.md "task" --headless

**Session Management:**

# Start persistent session
tsugite-docker-session start my-work

# Run multiple agents in same session
tsugite-docker --container my-work run agent.md "task 1"
tsugite-docker --container my-work run agent.md "task 2"

# List all sessions
tsugite-docker-session list

# Execute arbitrary commands in session
tsugite-docker-session exec my-work bash
tsugite-docker-session exec my-work python script.py

# Stop session (keeps container)
tsugite-docker-session stop my-work

# Stop and remove session
tsugite-docker-session stop my-work --remove

**How It Works:**

- Wrapper scripts parse Docker flags and build `docker run` commands
- All other arguments forwarded to tsugite unchanged
- Agents run in `tsugite/runtime` Docker image (Python 3.12)
- Default network mode: `host` (works with Podman)
- Alternative modes: `bridge`, `none`, or custom

**Volume Mounts:**

Automatically mounted for full functionality:
- `/workspace` - Current directory (read-only for security)
- `~/.config/tsugite` - Config and MCP server settings (read-only)
- `~/.cache/tsugite` - Attachment cache (read-write)
- Environment variables - API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

**Use cases**

- **Untrusted agents** - Run agents from the internet safely
- **Reproducible environments** - Consistent execution across machines
- **Multi-agent workflows** - Multiple agents sharing container state
- **Debugging** - Keep container alive to inspect state after errors
- **Development** - Fast iteration with persistent containers

**Design Philosophy:**

Instead of integrating Docker into tsugite core, we provide simple wrapper scripts (~100 lines total) that follow the Unix philosophy: do one thing well and work with other tools. This approach eliminates 550+ lines of integrated code while providing the same functionality.

See `bin/README.md` for complete documentation and examples.

## Quickstart Checklist

- Install dev dependencies with `uv sync --dev`; use `uv add` for new packages.
- Prefer TDD: add or update tests first, run the focused pytest target you touched, then `uv run pytest` before merging.
- Keep formatting and linting clean: `uv run black .` and `uv run ruff check .`.
- Prefer type hints on public functions and raise precise errors (`ValueError` vs `RuntimeError`).
- Skim existing agents in `agents/` and `.tsugite/` to reuse proven patterns.

## Project Map

- CLI entrypoint: `tsugite/tsugite.py`.
- Agent config + frontmatter parsing: `tsugite/md_agents.py`.
- Template rendering helpers (Jinja2 + filters): `tsugite/renderer.py`.
- Runtime execution + prefetch handling: `tsugite/agent_runner.py`.
- Tool registration + adapters: `tsugite/tools/` (ensure side-effect imports when adding tools).
- Benchmarks + regression specs: `benchmarks/` (see `benchmarks/README.md`).

## Build & Test Commands

| Task | Command | Notes |
| --- | --- | --- |
| Install dev deps | `uv sync --dev` | Idempotent; required before tests. |
| Run all tests | `uv run pytest` | Narrow scope with `tests/test_file.py::test_name` while iterating. |
| Lint | `uv run ruff check .` | Prefer fixing warnings over ignoring. |
| Format | `uv run black .` | Black's 88-character width is enforced. |
| Benchmark smoke | `uv run pytest tests/test_benchmark_core.py` | Run when touching benchmark logic. |

## Style Guidelines

- Target Python 3.12+; avoid legacy compatibility shims.
- Imports: stdlib, third-party, then local—each group separated by a blank line.
- Line length: 88 characters (Black default).
- Provide type hints on public APIs and reusable helpers.
- Docstrings: concise triple-quoted summaries with `Args` / `Returns` for public functions or classes.
- Raise `ValueError` for invalid inputs and `RuntimeError` for execution failures; name the failing component/tool/path.
- Favor f-strings for interpolation and keep comments only when clarifying intent.

## Agent Frontmatter Cheatsheet

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `name` | ✅ | — | Identifier surfaced in CLI output. |
| `model` | ❌ | `ollama:qwen2.5-coder:7b` | `provider:model[:variant]`; parsed by `parse_model_string`. |
| `max_steps` | ❌ | `5` | Reasoning iterations per run. |
| `tools` | ❌ | `[]` | Registered tool names; ensure module import registers them. |
| `prefetch` | ❌ | `[]` | Tool calls executed before rendering; assign results for template reuse. |
| `attachments` | ❌ | `[]` | Attachment aliases to auto-load as context (prepended to all prompts). |
| `context_budget` | ❌ | Unlimited | Hard cap for rendered prompt length. |
| `permissions_profile` | ❌ | — | Placeholder until the permissions engine ships. |
| `instructions` | ❌ | — | Extra system guidance appended at runtime. |
| `mcp_servers` | ❌ | — | MCP servers and optional tool safelists. |

### Prefetch Tips

- Use prefetch for expensive operations or structured data reused in multiple template locations.
- Prefetch failures resolve to `None`; guard with `{% if variable %}` before use.
- Prefer template helpers (below) for simple existence checks.

## Template Helper Reference

| Helper | Purpose |
| --- | --- |
| `now()` | Current timestamp in ISO 8601 format. |
| `today()` | Current date as `YYYY-MM-DD`. |
| `slugify(text)` | Converts text into a filesystem-friendly slug. |
| `file_exists(path)` | Returns `True` if a path exists. |
| `is_file(path)` / `is_dir(path)` | Distinguish file vs directory. |
| `read_text(path, default="")` | Safe file read with fallback value. |
| `env` | Mapping of environment variables (e.g., `env.HOME`). |

## Multi-Step Agents

- Mark steps with `<!-- tsu:step name="step_id" assign="variable" -->`.
- Content before the first directive becomes a shared preamble for every step.
- Execution halts on the first failure; give steps descriptive names to aid debugging.
- Per-step context variables: `{{ step_number }}`, `{{ total_steps }}`, `{{ step_name }}`, plus any previously assigned variables.
- Keep steps focused (analyze → plan → execute) to control token usage and simplify reasoning.

### Example

````markdown
---
name: research_writer
max_steps: 10
tools: [write_file, web_search]
---

Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="findings" -->
Research the topic with `web_search` and capture 3–5 insights.

<!-- tsu:step name="outline" assign="structure" -->
Outline the article using:
{{ findings }}

<!-- tsu:step name="write" assign="article" -->
Draft the article following {{ structure }}.

<!-- tsu:step name="save" -->
Write `article.md`, confirm it exists, and surface the final text.
````

## Model Providers

- Model strings follow `provider:model[:variant]`; parsed by `tsugite.models.parse_model_string`.
- Built-in providers via LiteLLM: `ollama` (local, default URL `http://localhost:11434/v1`), `openai`, `anthropic`, `google`, `github_copilot`, plus any other LiteLLM-supported providers.
- Ensure requisite API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are present before running agents.
- Ollama requires the local server; override base URL via environment variables when needed.

## Reasoning Models (o1, o3, Claude Extended Thinking)

Tsugite supports reasoning models, with varying levels of reasoning visibility:

### Models and Reasoning Visibility

**Full Reasoning Content (Exposed):**
- **Anthropic Claude** (`claude-3-7-sonnet`) - Extended thinking shown in full
- **Deepseek** - Reasoning content exposed via API
- Displayed in magenta panels with full step-by-step thinking

**Reasoning Token Counts Only (Hidden Content):**
- **OpenAI o1 series** (`o1`, `o1-mini`, `o1-preview`) - Content hidden by OpenAI
- **OpenAI o3 series** (`o3`, `o3-mini`) - Content hidden by OpenAI
- Shows "🧠 Used N reasoning tokens" instead of actual reasoning
- Reasoning happens internally but isn't exposed via Chat Completions API

### Why O1/O3 Don't Show Reasoning Content

OpenAI's o1/o3 models use reasoning tokens to think through problems, but **the actual reasoning content is not exposed** via the standard Chat Completions API. You only see:
- The final answer
- A count of how many reasoning tokens were used
- Total tokens consumed (including hidden reasoning)

This is an intentional design decision by OpenAI. To get reasoning summaries from o1/o3, you would need OpenAI's newer Responses API (not yet supported by tsugite).

### Controlling Reasoning Effort

**Supported Models:** `o1`, `o1-preview`, `o3`, `o3-mini` (NOT `o1-mini`)

```markdown
---
name: deep_thinker
model: openai:o1  # or o1-preview, o3, o3-mini
reasoning_effort: high  # Options: low, medium, high
---
Task: {{ user_prompt }}
```

**Important:** `o1-mini` does NOT support the `reasoning_effort` parameter.

### Per-Step Reasoning Control

```markdown
<!-- tsu:step name="analyze" reasoning_effort="low" -->
Quick analysis (fewer reasoning tokens)

<!-- tsu:step name="deep_think" reasoning_effort="high" -->
Thorough reasoning (more reasoning tokens)
```

### Example Usage

```bash
# O1-mini - shows reasoning token count
tsugite run agents/examples/reasoning_model_test.md "Explain quantum computing"
# Output: "🧠 Used 128 reasoning tokens"

# Claude - shows full reasoning content
tsugite run --model anthropic:claude-3-7-sonnet agents/examples/reasoning_model_test.md "Explain quantum computing"
# Output: "[Panel with full thinking process]"
```

### Parameter Limitations

OpenAI o1/o3 models have a more restricted parameter set than GPT-4:
- **Not supported:** `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`
- **o1-mini only:** Also doesn't support `reasoning_effort`
- **Supported:** `max_completion_tokens`, `messages`, `stream`

Tsugite automatically filters out unsupported parameters to prevent API errors.

## MCP Server Integration

1. Add servers with `tsugite mcp add …`. Config lives under XDG paths (`~/.tsugite/mcp.json`, `$XDG_CONFIG_HOME/tsugite/mcp.json`, or `~/.config/tsugite/mcp.json`).
2. `--url` registers HTTP servers; `--command` + `--args` launches stdio servers (e.g., `npx`, `uvx`).
3. Reference servers in frontmatter using `mcp_servers`. Provide a list of tool names to safelist or set the value to `null`/omit to expose all tools.
4. Run agents with MCP tools using `--trust-mcp-code` when remote execution is acceptable.
5. Inspect or validate servers with `tsugite mcp list`, `tsugite mcp show`, and `tsugite mcp test`.

## Multi-Agent Composition

- Invoke multiple agents positionally: `tsugite run +assistant +coder "task"`; or supply extra agents via `--with-agents "coder reviewer"`.
- The first agent acts as coordinator and receives helper tools named `spawn_{agent}` plus `spawn_agent(path, prompt)`.
- Delegate specialized work (research → write → review, manager → coder → tester) and merge results deterministically.
- Agent resolution order for `+name`: `.tsugite/{name}.md`, `agents/{name}.md`, `./{name}.md`, `~/.tsugite/agents/{name}.md`, `~/.config/tsugite/agents/{name}.md`.
- Keep delegated agents idempotent and explicit about outputs so orchestration stays predictable.

## Additional Resources

- Explore `agents/examples/` and `docs/test_agents/` for ready-to-run patterns.
- Design docs in `docs/` (permissions, history, task tracking, structured output) outline in-progress systems—keep new features modular to integrate cleanly later.
