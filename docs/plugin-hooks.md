# Plugin Hooks

Plugins can register Python hooks that run at various points in the agent lifecycle.

## Entry Point

Declare in your plugin's `pyproject.toml`:

```toml
[project.entry-points."tsugite.hooks"]
my_plugin = "my_plugin:register_hooks"
```

## Register Function

The entry point must resolve to a callable that accepts a `config: dict` and returns `dict[str, list[HookRule]]` - a mapping of phase names to lists of `HookRule` objects.

The `config` dict comes from the daemon config's `plugins:` section for your plugin name.

Each `HookRule` with `type="python"` takes a `hook_callable` - an async or sync function that receives `context: dict`. If `capture_as` is set, the return value is injected into the agent context under that key.

## Hook Phases

Phases that support `capture_as` (return value injected into context):

- `pre_message` - before each user message is processed
- `pre_context_build` - after pre_message, before LLM context is assembled
- `post_context_build` - after context built, before LLM call (can override `system_message`, `rendered_prompt`)

Hooks that ignore return values:

- `pre_tool_call` - before a tool executes
- `post_tool` - after a tool executes successfully
- `pre_response` - before final answer is emitted
- `post_response` - after final answer is emitted
- `session_end` - after session is saved to history
- `pre_compact` - before context compaction
- `post_compact` - after context compaction

## HookRule Fields

- `type` - `"python"` for callable hooks (also `"shell"`, `"agent"`)
- `hook_callable` - async or sync Python callable, receives `context: dict`
- `name` - display label (used in logging/status)
- `capture_as` - if set, the return value is captured and injected into context
- `match` - Jinja2 expression; hook skipped if it evaluates falsy
- `only_interactive` - if `True`, skips in non-interactive (scheduled) runs
- `tools` - list of tool names to match (for `pre_tool_call`/`post_tool` phases)

## Context Keys

Each phase receives a `context: dict` with relevant keys:

- `pre_context_build` - `message`, `agent_name`, plus any captured vars from `pre_message`
- `session_end` - `session_id`, `agent_name`, `result`, `model`, `tokens`, `cost`, `status`
- `pre_tool_call` / `post_tool` - `tool`, plus the tool's arguments

## Behavior

- Plugin hooks run after workspace YAML hooks for the same phase
- Errors are logged but never abort the agent run
- Async callables are awaited; sync callables are run in a thread
- Non-capturing phases run as background tasks
- Hooks are loaded lazily on first tool access
