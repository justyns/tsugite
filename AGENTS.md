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
