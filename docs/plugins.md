# Plugins

Most plugins need only one entry point, `tsugite.plugins`. Importing the module triggers `@tool`, `@hook`, and `@subscribe` decorators.

```toml
[project.entry-points."tsugite.plugins"]
my_plugin = "tsugite_my_plugin"
```

## Creating a plugin

The fastest path is the scaffolder:

```bash
uv run tsu plugin create discord
```

This creates `plugins/tsugite-discord/` with a working pyproject, an example tool, and a passing test. Then:

```bash
uv sync --all-extras
uv run pytest plugins/tsugite-discord/tests/
```

`plugins/tsugite-discord/tsugite_discord/__init__.py` contains the example tools.

## Naming convention

| Layer | Pattern | Example |
|---|---|---|
| PyPI distribution | `tsugite-<name>` | `tsugite-discord` |
| Python module | `tsugite_<name>` | `tsugite_discord` |
| Workspace directory | `plugins/tsugite-<name>/` | `plugins/tsugite-discord/` |
| `@tool(category=...)` | bare `<name>` | `@tool(category="discord")` |

The `category=` value is what `@<name>` references in agent tool specs (e.g. `tools: ["@discord"]`).

## Workspace layout

First-party plugins live under `plugins/` as uv workspace members:

```
tsugite/
├── pyproject.toml            # root: declares [tool.uv.workspace] members = ["plugins/*"]
├── tsugite/                  # the core package
└── plugins/
    ├── tsugite-tmux/
    │   ├── pyproject.toml
    │   ├── tsugite_tmux/__init__.py
    │   └── tests/test_tmux_tools.py
    └── tsugite-<your-plugin>/
        └── ...
```

`uv sync` installs the core + all plugins listed in `[dependency-groups] dev`. Each plugin can also be published to PyPI independently and installed by end users via `pip install tsugite-<name>`.

## Lifecycle

1. **Discovery**: at first tool access, `tsugite.tools._ensure_tools_loaded()` calls `load_tool_plugins()`, `load_hook_plugins()`, `load_event_subscriber_plugins()` (see `tsugite/plugins.py`).
2. **Loading**: each entry point is loaded; if it resolves to a callable, it's invoked with the per-plugin config dict; if it resolves to a module, the import alone is treated as registration (decorators register tools).
3. **Registration**: tools land in the global registry, hooks in `_plugin_hooks`, event subscriptions in `_plugin_subscriptions`. New `EventBus` instances auto-attach plugin subscriptions in their `__init__`.
4. **Errors**: any plugin that fails to load is recorded in `PluginInfo.error` and logged at `WARNING`; other plugins are unaffected.

## Plugin config

Each plugin can receive a config dict from the daemon's `~/.tsugite/daemon.yaml` under `plugins:`:

```yaml
plugins:
  discord:
    enabled: true
    bot_token: "..."
```

The `register_*` callable receives this dict as its sole argument. `enabled: false` skips loading entirely.

## Inspecting plugins

```bash
uv run tsu plugin list           # all discovered plugins
uv run tsu plugin info <name>    # detail
uv run tsu tools list            # all tools (built-in + plugin)
uv run tsu tools show <tool>     # tool signature + module
```

## Event subscriber plugins

Use the `@subscribe` decorator and the unified entry point:

```python
# tsugite_my_plugin/__init__.py
from tsugite.events.bus import subscribe
from tsugite.events.events import CustomEvent


@subscribe(event_name="tool_call")
def on_tool_call(event):
    if event.tool_name == "run":
        print(f"[my-plugin] saw run({event.arguments})")


@subscribe(event_name="tool_call", predicate=lambda e: e.tool_name == "http_request")
def on_http(event):
    print(f"[my-plugin] http {event.arguments.get('url')}")
```

```toml
[project.entry-points."tsugite.plugins"]
my_plugin = "tsugite_my_plugin"
```

`event_name` matches against `event.event_name` (e.g. `"tool_call"`, `"task_start"`); `None` (the default) receives all events. `predicate` is an optional `(event) -> bool` gate. Plugins can also emit cross-plugin signals via `CustomEvent(custom_name="my_plugin.something_happened", payload={...})` and other plugins filter on the same string.

For config-driven registration, use the function form via `tsugite.event_subscribers`:

```python
def register_event_subscribers(config):
    from tsugite.events.bus import Subscription
    subs = [Subscription(handler=on_tool_call, event_name="tool_call")]
    if config.get("debug"):
        subs.append(Subscription(handler=debug_logger))
    return subs
```

## Adapter plugins

Tools, hooks, and subscribers use the `tsugite.plugins` entry point above. A second kind of
plugin - the **adapter** - registers under `tsugite.adapters` and extends the running daemon:
it can front a chat platform, mount its own HTTP routes under `/api/plugins/<name>` (authed or
public), and register job executors. See [plugin-adapters.md](plugin-adapters.md).

## Versioning and release (lockstep)

Core + every workspace plugin ship together with the same version number, driven by a single git tag. To bump:

```bash
uv run python scripts/bump_version.py 0.14.0
git diff
git commit -am "chore: bump version to 0.14.0"
git tag v0.14.0
git push origin master --tags
```

The push triggers `.github/workflows/pypi-publish.yml` which builds wheels for every workspace member (`uv build --all-packages`), publishes to PyPI, and creates a GitHub release with auto-generated notes from `cliff.toml` (conventional-commits parsing).

Each plugin must be registered as a separate trusted publisher on PyPI (one-time setup per package).
