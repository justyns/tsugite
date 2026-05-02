# Plugin Tools

Plugins can register Python functions as tools the LLM can call. Tools added by plugins live in the same registry as built-in tools and are invoked through the same dispatch path.

## Entry Point

Declare in your plugin's `pyproject.toml`. The unified `tsugite.plugins` group is the recommended form - importing the module triggers `@tool`-decorated functions to register themselves alongside any `@hook` and `@subscribe` decorators in the same module.

```toml
[project.entry-points."tsugite.plugins"]
my_plugin = "tsugite_my_plugin"
```

No `register_tools()` function is needed. The dedicated `tsugite.tools` group is still supported for the function form (see "Advanced" below).

## Defining tools

Decorate any function with `@tool`:

```python
# tsugite_my_plugin/__init__.py
from tsugite.tools import tool


@tool(category="my_plugin")
def my_plugin_search(query: str, limit: int = 10) -> list[dict]:
    """Search the my_plugin index.

    Args:
        query: Search string.
        limit: Maximum results to return.

    Returns:
        List of matching records.
    """
    return [...]
```

The function name becomes the tool name (`my_plugin_search`). The first line of the docstring is the short description. Parameter types and defaults come from the signature.

## Decorator options

```python
@tool(category="my_plugin", parent_only=False, interactive_only=False, require_daemon=False)
```

- `category` — overrides the default category (the function's module basename). Plugins almost always set this so `@<category>` references in agent tool specs (e.g. `tools: ["@my_plugin"]`) resolve to the plugin's tools.
- `parent_only` — tool runs in the parent process, not the sandbox. Use for tools that need direct user interaction or spawn subprocesses.
- `interactive_only` — tool is hidden in scheduled tasks (where there's no UI to display output).
- `require_daemon` — tool only registers when running in daemon mode.

## Plugin config

Module-only plugins read config at tool-call time via their own mechanism (environment variables, on-disk config, etc.) since the import has no config parameter. For most cases this is fine.

If you need config to drive *which* tools register (e.g. only register a tool when an API key is present), use the function form below.

## Advanced: function-form entry point

```toml
[project.entry-points."tsugite.tools"]
my_plugin = "tsugite_my_plugin:register_tools"
```

```python
def register_tools(config: dict) -> list:
    tools = [my_plugin_search]
    if config.get("api_key"):
        tools.append(my_plugin_premium_fetch)
    return tools
```

The callable receives the per-plugin config dict from the daemon's `~/.tsugite/daemon.yaml`:

```yaml
plugins:
  my_plugin:
    enabled: true
    api_key: "..."
```

`enabled: false` skips loading entirely. The returned tool functions are registered into the same global registry as `@tool`-decorated tools.

## Calling other tools

A plugin tool can call any registered tool, including built-ins, via the unified dispatch:

```python
from tsugite.tools import call_tool

@tool(category="my_plugin")
def my_plugin_summarize(path: str) -> str:
    contents = call_tool("read_file", path=path)
    return contents[:500]
```

## Behavior

- Plugin tools register after built-in tools at first tool access (lazy load via `_ensure_tools_loaded`).
- Re-registration is idempotent (registry is keyed by tool name).
- A failing entry point is logged at `WARNING` and recorded in `PluginInfo.error`; other plugins still load.
- Tools surface in `uv run tsu tools list` and inspect via `uv run tsu tools show <name>`.
- `uv run tsu plugin list` shows discovered plugins and their entry points.

## Workspace vs PyPI install

First-party plugins live in `plugins/` and install via `uv sync --all-extras`. Third-party (or first-party for end users) install via `pip install tsugite-<name>`. Both go through the same entry-point discovery.

See [plugins.md](plugins.md) for the workspace setup, naming convention, and lockstep release model.
