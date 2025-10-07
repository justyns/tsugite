# Agent Development Guide

## Quick cheat sheet of example commands

_Quick CLI cheat sheet for common `tsugite` workflows:_

tsugite run research_writer.md "AI in healthcare"
tsugite run research_writer.md "AI in healthcare" --debug
tsugite mcp add basic-memory --url http://localhost:3000/mcp
tsugite mcp add pubmed --command uvx --args "--quiet" --args "pubmedmcp@0.1.3" --env "UV_PYTHON=3.12"
tsugite mcp add ai-tools \
tsugite mcp add basic-memory --url http://new-url.com/mcp --force
tsugite mcp list
tsugite mcp show basic-memory
tsugite mcp test basic-memory --trust-code
tsugite run agent.md "task" --trust-mcp-code
tsugite run agents/assistant.md "query"
tsugite run +assistant "query"
tsugite run +assistant +jira +coder "create ticket and fix bug #123"
tsugite run +assistant +jira create a ticket for bug 123
tsugite run +assistant --with-agents "+jira +coder" "create ticket and fix bug #123"
tsugite run +coordinator +researcher +writer +reviewer "Research AI trends and create a blog post"
tsugite run +coordinator --with-agents "researcher,writer,reviewer" \
tsugite run +project_manager +github +tester create feature branch implement login and run tests
tsugite run +project_manager +github +tester "Create feature branch, implement login, and run tests"
tsugite run +assistant +jira +coder "fix bug #123"
tsugite run +assistant --with-agents "jira coder" "fix bug #123"
tsugite run +assistant --with-agents jira,coder "fix bug #123"
tsugite run +assistant +jira create a ticket for bug 123
tsugite run agents/custom.md +jira ./helpers/coder.md "task"
tsugite run +coordinator +researcher +writer "task"
tsugite run +coordinator --with-agents "researcher writer" "task"

### File References

Use `@filename` syntax to automatically inject file contents into prompts:

tsugite run +assistant "Review @src/main.py and suggest improvements"
tsugite run +coder "Fix bugs in @app.py based on @tests/test_app.py"
tsugite run +assistant @README.md  # Single file as entire context
tsugite run +writer "Summarize @"docs/user guide.md""  # Quoted paths for spaces
tsugite run +assistant "Compare @file1.py and @file2.py"  # Multiple files

The files are automatically read and injected as formatted context. The CLI will show which files were expanded in the info panel.

**Note**: `@word` patterns that start with alphanumeric characters will be treated as file references. Use `@"path"` for explicit file references or avoid `@` prefix for non-file content (e.g., write "email user123" instead of "@user123").

### Attachments (Reusable Context)

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

**Using Attachments in Prompts:**

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

**How It Works:**

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

## Quickstart Checklist

- Install dev dependencies with `uv sync --dev`; use `uv add` for new packages.
- Run the focused pytest target you touched, then `uv run pytest` before merging.
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
