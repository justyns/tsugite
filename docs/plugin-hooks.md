# Plugin Hooks

Plugins can register Python callables that run at points in the agent lifecycle - before/after tool calls, around the LLM context build, on session end, and so on.

## Entry Point

Declare in your plugin's `pyproject.toml`. The unified `tsugite.plugins` group is the recommended form - importing the module triggers `@hook`-decorated functions to register themselves alongside any `@tool` and `@subscribe` decorators in the same module.

```toml
[project.entry-points."tsugite.plugins"]
my_plugin = "tsugite_my_plugin"
```

No `register_hooks()` function is needed. The dedicated `tsugite.hooks` group is still supported for the function form (see "Advanced" below).

## Defining hooks

Decorate any function with `@hook(phase, ...)`:

```python
# tsugite_my_plugin/__init__.py
from tsugite.hooks import hook


@hook("pre_message", capture_as="user_role")
async def lookup_user_role(ctx: dict) -> str:
    """Inject the calling user's role into the agent context as `user_role`."""
    return await fetch_role_for(ctx.get("user_id"))


@hook("post_tool", tools=["http_request"])
def log_http_call(ctx: dict) -> None:
    """Log every http_request invocation."""
    logger.info("http_request hit %s", ctx.get("url"))
```

The function name becomes the hook's display label unless `name=` is passed. Async and sync callables both work.

## `@hook` decorator options

```python
@hook(phase, *, name=None, capture_as=None, match=None, only_interactive=False, tools=None)
```

- `phase` - the lifecycle phase to attach to (see list below).
- `name` - display label used in logs and status output. Defaults to the function name.
- `capture_as` - if set, the return value is injected into the agent's context under this key. Only meaningful for capturing phases.
- `match` - Jinja2 expression evaluated against the context; the hook is skipped if it evaluates falsy.
- `only_interactive` - if `True`, the hook skips in non-interactive (scheduled) runs.
- `tools` - list of tool names to gate on. Only meaningful for `pre_tool_call` / `post_tool` phases.

## Hook Phases

Phases that support `capture_as` (return value injected into context):

- `pre_message` - before each user message is processed
- `pre_context_build` - after pre_message, before LLM context is assembled
- `post_context_build` - after context built, before LLM call (can override `system_message`, `rendered_prompt`)

Phases that ignore return values:

- `pre_tool_call` - before a tool executes
- `post_tool` - after a tool executes successfully
- `pre_response` - before final answer is emitted
- `post_response` - after final answer is emitted
- `session_end` - after session is saved to history
- `pre_compact` - before context compaction
- `post_compact` - after context compaction

## Context keys

Each phase receives a `context: dict` with relevant keys:

- `pre_context_build` - `message`, `agent_name`, plus any captured vars from `pre_message`
- `session_end` - `session_id`, `agent_name`, `result`, `model`, `tokens`, `cost`, `status`
- `pre_tool_call` / `post_tool` - `tool`, plus the tool's arguments

## Behavior

- Plugin hooks run after workspace YAML hooks for the same phase.
- Errors are logged but never abort the agent run.
- Async callables are awaited; sync callables are run in a thread.
- Non-capturing phases run as background tasks.
- Hooks are loaded lazily on first tool access.

## Advanced: function-form entry point

For config-driven registration (e.g. only register a hook when an API key is set), use the function form:

```toml
[project.entry-points."tsugite.hooks"]
my_plugin = "tsugite_my_plugin:register_hooks"
```

The callable receives the per-plugin config dict from the daemon's `~/.tsugite/daemon.yaml` and returns `dict[str, list[HookRule]]`:

```python
from tsugite.hooks import HookRule


def register_hooks(config: dict) -> dict:
    rules = {"pre_message": [HookRule(type="python", hook_callable=base_hook, name="base")]}
    if config.get("premium"):
        rules.setdefault("post_response", []).append(
            HookRule(type="python", hook_callable=premium_hook, name="premium")
        )
    return rules
```

Most plugins should prefer the `@hook` decorator above; reach for this form only when registration depends on runtime config.
