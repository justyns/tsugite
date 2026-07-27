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

## Command plugins

A plugin can add daemon slash commands (the `/name` commands in the web composer and Discord) via the `tsugite.commands` group. Like the unified group it is module-only: importing the module runs the `@adapter_command` decorator, which registers the command into the daemon's shared registry. Command handlers are daemon-coupled by nature - they receive a daemon `BaseAdapter` as their first argument.

```python
# tsugite_my_plugin/commands.py
from tsugite_daemon.commands import CommandParam, adapter_command


@adapter_command(
    name="terminals",
    description="List the daemon's terminal sessions and their state",
    params=[
        CommandParam("state", str, "Filter by state", required=False, choices=["running", "succeeded"]),
    ],
)
async def cmd_terminals(adapter, state: str | None = None) -> str:
    terminals = adapter.terminal_store.list_all()
    return "\n".join(f"[{t.state}] {t.id}: {t.cmd}" for t in terminals) or "No terminals found."
```

```toml
[project.entry-points."tsugite.commands"]
my_plugin = "tsugite_my_plugin.commands"
```

Each `CommandParam` carries `required`, an optional `choices` list (a fixed enum the UI offers), and an optional `widget` hint (`"model"`, `"effort"`, ...) naming a rich input the web UI's autocomplete renders for that argument; omit `widget` for a plain text field. The daemon loads command plugins the first time `get_commands()` runs, so they are registered before the first `/api/commands` list or command run.

## Context providers

A plugin can contribute **context items** - structured `{key, label, value}` records that get folded into the agent's context (as a `<client_context>` block) and rendered in the web UI's context gutter, the same path the browser's own providers (e.g. location) use. There are two kinds, and one provider may be either or both:

- **Menu provider** - appears in the composer's "add context" menu. On pick the daemon runs `capture` server-side. Add `choices` to first offer a submenu; the picked value arrives as `capture`'s `arg` (it is `None` for a direct capture on pick).
- **Detector** - `detect` scans the outgoing message server-side at send time and attaches an item for anything it recognizes (a URL, a ticket id). Detectors run best-effort: a raising detector is logged and skipped, never breaking the send.

Register providers at import time via the module-only `tsugite.context_providers` group:

```python
# tsugite_my_plugin/context.py
from tsugite.context import ContextChoice, ContextItem, ContextProvider, register_context_provider


# Menu provider (with a submenu). `context` carries {session_id, user_id, agent, workspace_dir}.
def open_files(context: dict) -> list[ContextChoice]:
    return [ContextChoice(value="README.md", label="README.md")]


def capture_file(arg: str | None, context: dict) -> list[ContextItem]:
    if not arg:
        return []
    return [ContextItem(key=f"file:{arg}", label=arg, value=open(arg).read())]


register_context_provider(
    ContextProvider(key="file", label="Project file", icon="doc", choices=open_files, capture=capture_file)
)


# Detector - attach an item for each ticket id mentioned in the message.
import re

_TICKET = re.compile(r"\b([A-Z]+-\d+)\b")


def detect_tickets(message: str, context: dict) -> list[ContextItem]:
    return [ContextItem(key=f"ticket:{t}", label=t, value=f"Ticket {t}") for t in dict.fromkeys(_TICKET.findall(message))]


register_context_provider(ContextProvider(key="ticket", label="Ticket", icon="tag", detect=detect_tickets))
```

```toml
[project.entry-points."tsugite.context_providers"]
my_plugin = "tsugite_my_plugin.context"
```

`ContextProvider(key, label, icon="sparkle", capture=None, choices=None, detect=None)`: `key` is the stable id, `capture(arg, context)` runs a menu pick, `choices(context)` builds its optional submenu, and `detect(message, context)` scans a message. A provider shows in the menu iff it has a `capture`. The daemon exposes menu providers over `/api/context-providers`; detectors run automatically as each message is sent. `examples/tsugite-example-plugin/tsugite_example_plugin/context.py` is a heavily commented, copy-paste reference of both kinds.
