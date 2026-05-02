# Plugin Event Subscribers

Plugins can subscribe to the EventBus to react to anything emitted during agent runs - tool calls, LLM messages, cost summaries, secret access, file reads/writes, custom events from other plugins, and so on.

## Entry Point

Declare in your plugin's `pyproject.toml`. The unified `tsugite.plugins` group is the recommended form - importing the module triggers `@subscribe`-decorated functions to register themselves alongside any `@tool` and `@hook` decorators in the same module.

```toml
[project.entry-points."tsugite.plugins"]
my_plugin = "tsugite_my_plugin"
```

No `register_event_subscribers()` function is needed. The dedicated `tsugite.event_subscribers` group is still supported for the function form (see "Advanced" below).

## Defining subscribers

Decorate any function with `@subscribe(...)`:

```python
# tsugite_my_plugin/__init__.py
from tsugite.events.bus import subscribe


@subscribe(event_name="tool_call")
def on_tool_call(event):
    """Print every tool invocation."""
    print(f"[my-plugin] {event.tool_name}({event.arguments})")


@subscribe(event_name="tool_call", predicate=lambda e: e.tool_name == "run")
def on_run_only(event):
    """Only fire on the `run` tool."""
    audit_log(event)


@subscribe()  # no filter - receives every event
def log_everything(event):
    debug_log.write(f"{event.event_name}: {event}")
```

Subscribers run synchronously in the order they were registered. Errors are logged to stderr but do not affect other subscribers or the agent run.

## `@subscribe` decorator options

```python
@subscribe(event_name=None, *, predicate=None)
```

- `event_name` - match against `event.event_name` (e.g. `"tool_call"`, `"task_start"`, `"llm_message"`). Defaults to `None`, which receives every event.
- `predicate` - optional `(event) -> bool` gate evaluated only after `event_name` matches. Use this for finer filtering than the name alone provides.

## Event names

`event.event_name` is derived from `EventType` (lowercase enum name). The current set:

- Execution: `task_start`, `step_start`, `code_execution`, `observation`, `final_answer`, `error`
- LLM: `llm_message`, `reasoning_content`, `reasoning_tokens`, `llm_wait_progress`
- Streaming: `stream_chunk`, `stream_complete`
- Tool audit: `tool_call`, `tool_result`
- Files: `file_read`, `file_write`
- Skills: `skill_loaded`, `skill_unloaded`, `skill_load_failed`
- Misc: `info`, `warning`, `debug_message`, `step_progress`, `cost_summary`, `prompt_snapshot`, `secret_access`, `content_block`, `reaction`
- Plugin-defined: `custom` (matched by `custom_name`, see below)

`uv run python -c "from tsugite.events.base import EventType; print([t.name.lower() for t in EventType])"` prints the live list.

## Cross-plugin signaling: `CustomEvent`

Plugins can emit their own events for other plugins to subscribe to. Use `CustomEvent` with a freeform `custom_name`:

```python
from tsugite.events.events import CustomEvent
from tsugite.ui_context import get_ui_context


def emit_task_created(task_id: int):
    bus = get_ui_context().event_bus
    if bus is not None:
        bus.emit(CustomEvent(custom_name="vikunja.task.created", payload={"id": task_id}))
```

Receivers filter by the custom name via `event_name` (the `event_name` property on `CustomEvent` returns `custom_name`):

```python
@subscribe(event_name="vikunja.task.created")
def react_to_task_creation(event):
    notify_user(event.payload["id"])
```

Convention: namespace your custom names with the plugin name (`vikunja.task.created`, not `task.created`) to avoid collisions.

## Behavior

- Subscribers are auto-attached to every `EventBus` instance constructed after plugin load (see `EventBus.__init__`).
- Subscribers loaded after a bus is constructed are NOT retroactively attached to that bus.
- Errors in a subscriber's handler or predicate are caught, logged to stderr with a traceback, and isolated - they don't break other subscribers or the agent run.
- Subscribers run synchronously; if you need to do slow work, spawn a background asyncio task or thread inside your handler.
- Subscriber order matches registration order (deterministic per import order).

## Inspecting subscribers

There's no dedicated CLI today. To list registered subscriptions in a Python REPL:

```python
from tsugite.plugins import get_plugin_subscriptions
for sub in get_plugin_subscriptions():
    print(f"{sub.event_name or '*':30s} -> {sub.handler.__module__}.{sub.handler.__name__}")
```

## Advanced: function-form entry point

For config-driven registration (e.g. only subscribe when a feature flag is set), use the function form:

```toml
[project.entry-points."tsugite.event_subscribers"]
my_plugin = "tsugite_my_plugin:register_event_subscribers"
```

```python
from tsugite.events.bus import Subscription


def register_event_subscribers(config: dict) -> list[Subscription]:
    subs = [Subscription(handler=on_tool_call, event_name="tool_call")]
    if config.get("debug"):
        subs.append(Subscription(handler=debug_logger))  # event_name=None - all events
    return subs
```

The callable receives the per-plugin config dict from the daemon's `~/.tsugite/daemon.yaml`. `enabled: false` skips loading entirely. Most plugins should prefer `@subscribe` above; reach for this form only when subscriptions depend on runtime config.
