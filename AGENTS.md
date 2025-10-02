# Agent Development Guide

## Build/Test Commands

- **Install dev dependencies**: `uv sync --dev`
- Use `uv add` and related commands to manage dependencies
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_function_name`
- **Run specific test file**: `uv run pytest tests/test_cli.py`
- **Lint code**: `uv run ruff check .`
- **Format code**: `uv run black .`
- **Type check**: No mypy configured; use ruff for basic checks

## Code Style Guidelines

- **Line length**: 120 characters (black + ruff configured)
- **Target Python**: 3.12+
- **Imports**: Use absolute imports, group stdlib/third-party/local with blank lines
- **Type hints**: Use for function signatures, especially public APIs
- **Error handling**: Use specific exceptions (ValueError, RuntimeError), include context
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Use triple quotes with Args/Returns sections for public functions
- **String formatting**: Use f-strings for simple interpolation

## Project Structure

- Main CLI: `tsugite/tsugite.py` (typer-based)
- Models: `tsugite/models.py` (smolagents integration)
- Tools: `tsugite/tools/` directory
- Tests: `tests/` with pytest fixtures
- No comments unless they add meaningful context
- Write new unit tests for all new features

## Frontmatter Reference

- `name` *(required)* – Agent identifier shown in CLI.
- `model` *(optional)* – Defaults to `ollama:qwen2.5-coder:7b`.
- `max_steps` *(optional)* – Defaults to `5`.
- `tools` *(optional list)* – Tool names registered via `@tool`.
- `prefetch` *(optional list)* – Tool calls to run before rendering.
- `permissions_profile` *(optional)* – Placeholder for future permissions engine.
- `instructions` *(optional string)* – Additional system guidance appended to Tsugite's default runtime instructions before hitting the LLM.
- `mcp_servers` *(optional dict)* – MCP servers to load tools from. Keys are server names from MCP config file (see XDG paths below), values are optional tool lists.

## Template Helpers

Agent templates use Jinja2 syntax and have access to built-in helper functions for common operations.

### Available Helper Functions

**Date/Time:**
- `now()` - Current timestamp (ISO format)
- `today()` - Today's date (YYYY-MM-DD)

**Text Processing:**
- `slugify(text)` - Convert text to slug format

**Filesystem:**
- `file_exists(path)` - Check if file or directory exists (returns bool)
- `is_file(path)` - Check if path is a file (returns bool)
- `is_dir(path)` - Check if path is a directory (returns bool)
- `read_text(path, default="")` - Safely read file content, return default on error

**Environment:**
- `env` - Dictionary of environment variables (e.g., `env.HOME`)

### Examples

**Conditional based on file existence:**

```yaml
---
name: config_checker
model: ollama:qwen2.5-coder:7b
tools: [read_file, write_file]
---

{% if file_exists("config.json") %}
## Configuration Found
Config exists at config.json
Content preview: {{ read_text("config.json")[:100] }}
{% else %}
## No Configuration
Please create a config.json file first.
{% endif %}

Task: {{ user_prompt }}
```

**Check multiple paths:**

```jinja2
{% if is_file("data.csv") and is_dir("output") %}
Ready to process data.csv into output/
{% elif not is_file("data.csv") %}
Error: data.csv not found
{% elif not is_dir("output") %}
Error: output directory missing
{% endif %}
```

**Using prefetch vs template helpers:**

Template helpers are best for simple checks in conditionals. Use `prefetch` when you need to:
- Pass results to tools as arguments
- Reuse expensive operations multiple times
- Run operations that modify state

```yaml
# Good: Simple existence check
{% if file_exists("config.json") %}...{% endif %}

# Better with prefetch: Complex operation used multiple times
prefetch:
  - name: read_file
    args:
      path: "config.json"
    assign: config_content
# Then use {{ config_content }} throughout template
```

## Multi-Step Agents

Multi-step agents enable forced sequential execution where each step is a full agent run. Results from earlier steps can be assigned to variables and used in later steps.

### Syntax

Use HTML comment directives to mark steps:

```markdown
<!-- tsu:step name="step_name" assign="variable_name" -->
```

- `name` *(required)* - Unique identifier for the step
- `assign` *(optional)* - Variable name to store this step's result

### How It Works

1. **Sequential Execution** - Each step runs as a complete agent session
2. **Variable Assignment** - Step results are captured and made available to subsequent steps
3. **Context Accumulation** - Each step has access to all previous results via Jinja2 variables
4. **Abort on Failure** - If any step fails, execution stops immediately
5. **Shared Preamble** - Content before the first step directive is prepended to all steps as shared context

### Preamble (Shared Context)

Content before the first `<!-- tsu:step -->` directive is automatically included in **every step** as shared context. This is useful for:

- Headers and introductions that provide framing
- Common task descriptions (e.g., `Task: {{ user_prompt }}`)
- Shared instructions or constraints
- Template variables that all steps should see

**Example:**
```markdown
---
name: analyzer
---

# Task Analysis Framework

Task: {{ user_prompt }}

Guidelines:
- Be thorough and systematic
- Consider edge cases
- Provide clear reasoning

<!-- tsu:step name="analyze" -->
Analyze the task...

<!-- tsu:step name="execute" -->
Execute based on analysis...
```

In this example, the header, task description, and guidelines appear in both the "analyze" and "execute" steps.

### Example

```markdown
---
name: research_writer
max_steps: 10
tools: [write_file, web_search]
---

# Research & Write Pipeline

Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="findings" -->
Research the topic using web_search. Gather 3-5 key insights.

<!-- tsu:step name="outline" assign="structure" -->
Create an outline for an article using these findings:

{{ findings }}

<!-- tsu:step name="write" assign="article" -->
Write a full article following this structure:

{{ structure }}

Using these facts:
{{ findings }}

<!-- tsu:step name="save" -->
Save the article to article.md:

{{ article }}

Confirm the file was created successfully.
```

### Usage

```bash
# Multi-step agents are detected automatically
tsugite run research_writer.md "AI in healthcare"

# Shows progress like:
# [Step 1/4: research] Starting...
# [Step 1/4: research] Complete
# [Step 2/4: outline] Starting...
# ...

# Debug mode shows each step's rendered prompt
tsugite run research_writer.md "AI in healthcare" --debug
```

### Best Practices

**Step Design:**
- Keep steps focused on a single responsibility
- Use descriptive step names
- Assign variables when results will be reused

**Preamble Usage:**
- Use preamble for shared context needed by all steps
- Include task description and common instructions in preamble
- Keep preamble concise - it's prepended to every step
- Template variables in preamble are rendered for each step

**Context Passing:**
- Reference previous results using `{{ variable_name }}`
- All standard template helpers work in steps
- Each step sees `{{ user_prompt }}` and all previous assignments
- Task list is shared across all steps (use `{{ task_summary }}` to see tasks)

**Step Context Variables:**
Multi-step execution provides automatic context variables:
- `{{ step_number }}` - Current step number (1-indexed)
- `{{ step_name }}` - Current step's name attribute
- `{{ total_steps }}` - Total number of steps in the agent

Example usage:
```jinja2
<!-- tsu:step name="analyze" -->
## Step {{ step_number }} of {{ total_steps }}: {{ step_name }}
Analyze the user's request...

<!-- tsu:step name="execute" -->
{% if step_number == total_steps %}
This is the final step - provide the complete result.
{% else %}
Continue with step {{ step_number + 1 }}...
{% endif %}
```

**Task Management:**
- Tasks created in step 1 are visible in steps 2, 3, etc.
- Use task tracking to coordinate work across steps
- Task list persists throughout the entire multi-step execution
- Reference `{{ task_summary }}` in templates to see current tasks

**Error Handling:**
- Steps fail fast - execution aborts on first error
- Use clear step names for better error messages
- Test steps individually during development

**Performance:**
- Each step is a full agent run with its own token usage
- Consider combining simple operations into one step
- Use `max_steps` to control reasoning iterations within each step

### Limitations

Current limitations (may be expanded in future):
- No conditional steps (all steps always execute)
- No parallel execution (strict sequential order)
- No loops or DAG workflows
- Step results are strings only

## Model Providers

Tsugite supports multiple model providers through LiteLLM integration. Specify models using the format `provider:model-name`.

### Supported Providers

**Ollama** (local models):
```yaml
model: ollama:qwen2.5-coder:7b
model: ollama:llama3.2:latest
```
- Connects to local Ollama instance at `http://localhost:11434/v1`
- No API key required

**OpenAI**:
```yaml
model: openai:gpt-4
model: openai:gpt-4o-mini
```
- Requires `OPENAI_API_KEY` environment variable
- Billed per token via OpenAI API

**Anthropic** (Claude):
```yaml
model: anthropic:claude-3-5-sonnet-20241022
model: anthropic:claude-3-5-haiku-20241022
```
- Requires `ANTHROPIC_API_KEY` environment variable
- Billed per token via Anthropic API

**Google** (Gemini):
```yaml
model: google:gemini-1.5-pro
model: google:gemini-2.0-flash-exp
```
- Requires Google API key
- Billed per token via Google AI API

**GitHub Copilot**:
```yaml
model: github_copilot:gpt-4
```
- Requires paid GitHub Copilot subscription
- First run will prompt OAuth authentication via device flow
- LiteLLM stores auth tokens automatically
- **Billed via subscription, not per-token**

Optional environment variables:
- `GITHUB_COPILOT_TOKEN_DIR` - Custom token storage directory
- `GITHUB_COPILOT_ACCESS_TOKEN_FILE` - Custom access token file path

**Other Providers**:
Tsugite supports 100+ providers through LiteLLM. Use format `provider:model-name` and it will attempt to connect via LiteLLM. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for full list.

## MCP Server Integration

Tsugite supports loading tools from [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers, enabling access to external tools and data sources.

### Configuration File

Tsugite follows the [XDG Base Directory specification](https://wiki.archlinux.org/title/XDG_Base_Directory) for config file locations.

**Config file locations (checked in order):**
1. `~/.tsugite/mcp.json`
2. `$XDG_CONFIG_HOME/tsugite/mcp.json` (if `XDG_CONFIG_HOME` is set)
3. `~/.config/tsugite/mcp.json` (XDG default)

If a config file already exists, Tsugite will continue using that location. New configs use the XDG location.

**Example config file:**

```json
{
  "mcpServers": {
    "basic-memory": {
      "type": "http",
      "url": "http://localhost:3000/mcp"
    },
    "ai-tools": {
      "command": "npx",
      "args": ["-y", "@ivotoby/openapi-mcp-server"],
      "env": {
        "API_BASE_URL": "https://api.example.com",
        "OPENAPI_SPEC_PATH": "https://api.example.com/openapi.json",
        "API_HEADERS": "Authorization: Bearer your-token-here"
      }
    }
  }
}
```

### Server Types

**HTTP Servers** (streamable-http):
```json
{
  "server-name": {
    "type": "http",
    "url": "https://example.com/mcp"
  }
}
```

**Stdio Servers** (command-based):
```json
{
  "server-name": {
    "command": "npx",
    "args": ["-y", "package-name"],
    "env": {
      "API_KEY": "your-key"
    }
  }
}
```

### Agent Configuration

Reference MCP servers in agent frontmatter:

```yaml
---
name: research_agent
model: ollama:qwen2.5-coder:7b
tools: [read_file, write_file]
mcp_servers:
  basic-memory: [search_notes, write_note]  # Only these tools
  ai-tools:  # Empty/null = all tools from this server
---
```

**Tool Filtering:**
- Specify a list of tool names to allow only those tools
- Use `null` or omit the value to load all tools from the server

### CLI Commands

**Add a new server:**

HTTP server:
```bash
tsugite mcp add basic-memory --url http://localhost:3000/mcp
```

Stdio server with arguments:
```bash
tsugite mcp add pubmed --command uvx --args "--quiet" --args "pubmedmcp@0.1.3" --env "UV_PYTHON=3.12"
```

Stdio server with multiple env vars:
```bash
tsugite mcp add ai-tools \
  --command npx \
  --args "-y" \
  --args "@ivotoby/openapi-mcp-server" \
  --env "API_BASE_URL=https://api.example.com" \
  --env "OPENAPI_SPEC_PATH=https://api.example.com/openapi.json" \
  --env "API_HEADERS=Authorization: Bearer your-token"
```

Overwrite existing server:
```bash
tsugite mcp add basic-memory --url http://new-url.com/mcp --force
```

**List configured servers:**
```bash
tsugite mcp list
```

**Show server details:**
```bash
tsugite mcp show basic-memory
```

**Test server connection:**
```bash
tsugite mcp test basic-memory --trust-code
```

**Run agent with MCP tools:**
```bash
tsugite run agent.md "task" --trust-mcp-code
```

### Security Considerations

- MCP servers can execute arbitrary code - only use trusted servers
- Use `--trust-mcp-code` flag to enable remote code execution
- Sensitive environment variables (tokens, keys) are redacted in `mcp show` output
- Connection failures are handled gracefully - agent continues with available tools

## Multi-Agent Composition

Tsugite supports running multiple agents together, where a primary agent can delegate work to other agents.

### Agent Shorthand Syntax

Use `+name` to reference agents by name instead of file path:

```bash
# Instead of:
tsugite run agents/assistant.md "query"

# Use shorthand:
tsugite run +assistant "query"
```

**Agent search locations** (in order):
1. `.tsugite/{name}.md` - Project-local shared agents
2. `agents/{name}.md` - Project convention directory
3. `./{name}.md` - Current directory
4. `~/.tsugite/agents/{name}.md` - User global agents
5. `~/.config/tsugite/agents/{name}.md` - XDG global agents

### Delegating to Multiple Agents

Tsugite supports two syntaxes for multi-agent delegation:

**1. Positional syntax (simple and intuitive):**
```bash
# Multiple agents positionally
tsugite run +assistant +jira +coder "create ticket and fix bug #123"

# Unquoted multi-word prompts
tsugite run +assistant +jira create a ticket for bug 123
```

**2. Using --with-agents option:**
```bash
tsugite run +assistant --with-agents "+jira +coder" "create ticket and fix bug #123"
```

Both syntaxes work identically. Use positional for quick commands, use `--with-agents` when you have complex prompts with special characters.

**How it works:**
1. The primary agent (first one: `+assistant`) receives the user prompt
2. Additional agents (`jira`, `coder`) are made available via delegation tools
3. The primary agent gets `spawn_jira(prompt)` and `spawn_coder(prompt)` tools
4. The primary agent can delegate subtasks:
   ```python
   # In the agent's execution:
   ticket_result = spawn_jira("Create ticket for bug #123")
   fix_result = spawn_coder("Fix the authentication bug")
   ```

### Examples

**Example 1: Coordinator with specialists (positional syntax)**

```bash
# Positional - clean and simple
tsugite run +coordinator +researcher +writer +reviewer "Research AI trends and create a blog post"

# Or with --with-agents
tsugite run +coordinator --with-agents "researcher,writer,reviewer" \
  "Research AI trends and create a blog post"
```

The coordinator agent can:
- Use `spawn_researcher("Find latest AI trends")` to gather information
- Use `spawn_writer("Write blog post about: {research}")` to create content
- Use `spawn_reviewer("Review this blog post: {content}")` to get feedback

**Example 2: Development workflow with unquoted prompts**

```bash
# Unquoted multi-word prompt
tsugite run +project_manager +github +tester create feature branch implement login and run tests

# Or quoted
tsugite run +project_manager +github +tester "Create feature branch, implement login, and run tests"
```

The project manager can:
- Use `spawn_github("Create feature branch for login")` to manage git
- Implement the feature itself or delegate to another agent
- Use `spawn_tester("Run unit tests for login module")` to verify

**Example 3: All supported syntaxes**

```bash
# Positional agents
tsugite run +assistant +jira +coder "fix bug #123"

# Using --with-agents (space-separated)
tsugite run +assistant --with-agents "jira coder" "fix bug #123"

# Using --with-agents (comma-separated)
tsugite run +assistant --with-agents jira,coder "fix bug #123"

# Unquoted prompts (when no special characters)
tsugite run +assistant +jira create a ticket for bug 123

# Mix of paths and names
tsugite run agents/custom.md +jira ./helpers/coder.md "task"
```

### Creating Delegation-Aware Agents

Agents can use the `spawn_agent` tool directly or use auto-generated wrappers:

**Using spawn_agent (always available):**
```yaml
---
name: coordinator
model: ollama:qwen2.5-coder:7b
tools: [spawn_agent]
---

Task: {{ user_prompt }}

Delegate subtasks using spawn_agent(agent_path, prompt):
- spawn_agent("agents/researcher.md", "Find information about X")
- spawn_agent("+writer", "Write content based on: ...")
```

**Using auto-generated delegation tools:**
```bash
# When you run (either syntax):
tsugite run +coordinator +researcher +writer "task"
# OR
tsugite run +coordinator --with-agents "researcher writer" "task"

# The coordinator automatically gets these tools:
# - spawn_researcher(prompt)
# - spawn_writer(prompt)
# - spawn_agent(agent_path, prompt)  # Still available for ad-hoc delegation
```

The auto-generated tools are simpler to use since the agent path is pre-filled.

**When to use each syntax:**
- **Positional** (`+a +b +c "task"`): Quick commands, simple prompts, fewer agents
- **--with-agents**: Complex prompts with quotes/special chars, many agents, scripting
- **Unquoted prompts**: Very quick ad-hoc commands without special characters

### Best Practices

1. **Clear responsibilities**: Each agent should have a focused purpose
2. **Descriptive names**: Use names that indicate the agent's role (`+github`, `+tester`)
3. **Reusable agents**: Store commonly-used agents in `.tsugite/` or `~/.tsugite/agents/`
4. **Document delegation**: Include delegation instructions in agent templates
5. **Error handling**: Agents should handle delegation failures gracefully
