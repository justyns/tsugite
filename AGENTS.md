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
