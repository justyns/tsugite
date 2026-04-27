# Agents

An agent is a markdown file with YAML frontmatter.  The frontmatter configures behavior, the body is the prompt.  Jinja2 templating makes it more dynamic.

```markdown
---
name: researcher
model: anthropic:claude-sonnet-4-20250514
max_turns: 10
tools: [web_search, fetch_text, read_file, write_file, final_answer]
---

You are a research assistant.

Current date: {{ now() }}
Task: {{ user_prompt }}
```

## Paths

When you run `tsu run +name`, tsugite searches for the agent file in this order:

1. Workspace agents dir (if using a workspace)
2. `.tsugite/{name}.md`
3. `agents/{name}.md`
4. `./{name}.md`
5. Built-in agents (`tsugite/builtin_agents/`)
6. `~/.config/tsugite/agents/`

You can also pass a direct path: `tsu run ./my-agent.md "task"`.

## Frontmatter

All fields are optional except `name`.

| Field              | Type | Default          | Description                                                       |
|--------------------|------|------------------|-------------------------------------------------------------------|
| `name`             | str  | required         | Agent identifier                                                  |
| `description`      | str  | `""`             | Short description                                                 |
| `model`            | str  | config default   | Provider and model (`provider:model-name`)                        |
| `max_turns`        | int  | `5`              | Max LLM round-trips before stopping                               |
| `tools`            | list | `[]`             | Tools the agent can use                                           |
| `extends`          | str  | built-in default | Parent agent to inherit from (`"none"` to opt out)                |
| `instructions`     | str  | `""`             | Behavioral guidelines (combined with defaults)                    |
| `attachments`      | list | `[]`             | Files or URLs to include as context                               |
| `auto_load_skills` | list | `[]`             | Skills to load automatically at startup                           |
| `skill_paths`      | list | `[]`             | Extra directories to search for skills                            |
| `prefetch`         | list | `[]`             | Tools to run before execution, results available in templates     |
| `custom_tools`     | list | `[]`             | Inline tool definitions                                           |
| `reasoning_effort` | str  | none             | For reasoning models: `low`, `medium`, `high`                     |
| `disable_history`  | bool | `false`          | Don't save this agent's sessions                                  |
| `auto_context`     | bool | none             | Override auto-context file loading                                |
| `spawnable`        | bool | `true`           | Whether other agents can spawn this one                           |
| `network`          | dict | none             | Allowed domains for sandbox mode                                  |
| `allowed_secrets`  | list | `[]`             | Restrict which secrets this agent can access (empty = all)        |
| `run_if`           | str  | none             | Jinja2 condition, skip agent if false, useful for scheduled tasks |

## Jinja templating

The entire agent file (frontmatter `instructions` + markdown body) is rendered with Jinja2 before execution.

### Variables

These are always available in templates:

```
{{ user_prompt }}         # The user's task/message
{{ agent_name }}          # This agent's name
{{ CWD }}                 # Current working directory
{{ INVOKED_FROM }}        # Directory where the command was run
{{ WORKSPACE_DIR }}       # Workspace path (if applicable)
{{ is_interactive }}      # True if a user is present, e.g. tty or web ui
{{ is_daemon }}           # True if running in daemon mode
{{ is_scheduled }}        # True if triggered by scheduler
{{ is_subagent }}         # True if spawned by another agent
{{ parent_agent }}        # Parent agent name (if subagent)
{{ tools }}               # List of configured tool names
{{ chat_history }}        # Previous messages (chat mode)
```

Variables from `prefetch` and `<!-- tsu:tool -->` directives are also available by their `assign` name.

### Functions

```
{{ now() }}               # Current datetime (ISO format)
{{ today() }}             # Today's date (YYYY-MM-DD)
{{ yesterday() }}         # Yesterday's date
{{ tomorrow() }}          # Tomorrow's date
{{ days_ago(3) }}         # Datetime N days ago
{{ weeks_ago(2) }}        # Monday of N weeks ago
{{ date_format(dt, fmt) }}# strftime formatting
{{ slugify("Some Text") }}# "some-text"
{{ read_text("path") }}   # Read file contents
{{ file_exists("path") }} # Check if file exists
{{ cwd() }}               # Current working directory
{{ tmux_sessions() }}     # Active tmux sessions
```

`env` is available as a dict of environment variables: `{{ env.HOME }}` or `{{ env.get("MY_VAR", "fallback") }}`.

### Control flow

Standard Jinja2:

```markdown
{% if is_daemon %}
You're running as a daemon agent.
{% endif %}

{% for skill in available_skills %}
- {{ skill }}
{% endfor %}
```

## Tools

Tools are specified as a list.  A few formats work:

```yaml
# Individual tools
tools: [read_file, write_file, run, final_answer]

# Categories (@ prefix)
tools: ["@fs", "@http", "@secrets"]

# Glob patterns
tools: ["*_file", "web_*"]

# Exclusions (- prefix, applied after expansion)
tools: ["@fs", "-delete_file", "-remove_directory"]
```

Built-in categories include `@fs`, `@http`, `@shell`, `@secrets`, `@schedule`, `@notify`, `@sessions`, `@tmux`.

## Inheritance

Agents can inherit from other agents with `extends`:

```yaml
---
name: code-reviewer
extends: default
tools: [read_file, list_files, final_answer]
---
```

- Scalars (model, max_turns) are overwritten by the child
- Lists (tools, attachments, auto_load_skills) are merged and deduplicated
- Instructions are concatenated
- `extends: "none"` opts out of all inheritance

The default base agent is `default.md` (the built-in one).  Resolution uses the same search paths as agent discovery.

## Prefetch

Run tools before execution and make the results available in templates:

```yaml
---
name: agent
prefetch:
  - tool: list_agents
    args: {}
    assign: available_agents
  - tool: list_files
    args: { path: "." }
    assign: project_files
---

Available agents: {{ available_agents }}
Files in project: {{ project_files }}
```

Failures are silent (the variable gets `None`).

## Attachments

Include files or URLs as additional context:

```yaml
attachments:
  - /path/to/file.txt
  - https://example.com/data.json
  - "{{ WORKSPACE_DIR }}/notes.md"
```

Supports text files, PDFs, images, and URLs.  Jinja2 works in attachment paths.  All attachments are injected as context. Prefix with `-` to remove a workspace default (e.g. `-MEMORY.md`).

### Attachment specs (dict form)

Use the dict form to bind attachment content to a Jinja variable, render an on-demand index instead of full content, or expand globs:

```yaml
attachments:
  # Bind file content as a variable usable in body/instructions templates
  - path: MEMORY.md
    assign: memory_content

  # Bind without injecting (the LLM doesn't see it; the template does)
  - path: USER.md
    assign: user_prefs
    attach: false

  # Glob - one attachment per matched file; assign binds list[dict] of {path, content}
  - path: "notes/*.md"
    assign: notes

  # Index mode - emits a single <attachment mode="index"> with path+heading bullets,
  # no full file content. Agents read individual entries via read_file().
  - path: "memory/topics/*.md"
    mode: index
    name: topic_index             # optional override; default derives from glob
    assign: topics                # list[dict] of {path, heading, size_bytes, mtime}
    index_format: first_heading   # path_only | first_line | first_heading | frontmatter
    max_entries: 50
```

Variable shapes when `assign:` is set:

| Spec | Variable shape |
|---|---|
| `mode: full`, single concrete file | `str` (file content) |
| `mode: full`, glob | `list[dict]` of `{path, content}` (binaries skipped) |
| `mode: index` | `list[dict]` of `{path, heading, size_bytes, mtime}` |
| Single file, missing | `None` |
| Empty glob | `[]` |

Validation rules (enforced at parse time):

- `assign:` must be a valid Python identifier.
- Two specs cannot share the same `assign:` value.
- `attach: false` requires `assign:`.
- `path:` cannot start with `-` (use string form for removal).

Collisions between an `assign:` name and a built-in/prefetch variable resolve in favor of the attachment binding, with a warning logged.

See `examples/attachment_assign_demo.md` and `examples/attachment_index_demo.md` for full working examples.

## Multi-step agents

Use `<!-- tsu:step -->` directives to chain multiple LLM calls.  Content before the first step is the preamble, shared across all steps.

```markdown
---
name: research-and-write
model: openai:gpt-4o
tools: [web_search, fetch_text, write_file, final_answer]
---

You are a research assistant.

<!-- tsu:step name="research" assign="findings" -->
Research this topic: {{ user_prompt }}
Save your findings using final_answer().

<!-- tsu:step name="write" -->
Using the research findings, write a report.

The variable `findings` is available in Python.
```

### Step attributes

| Attribute           | Type  | Description                              |
|---------------------|-------|------------------------------------------|
| `name`              | str   | Required.  Step identifier               |
| `assign`            | str   | Variable name to store the step's result |
| `max_tokens`        | int   | Token limit for this step                |
| `reasoning_effort`  | str   | `low`, `medium`, `high`                  |
| `json`              | bool  | Enable JSON response mode                |
| `max_retries`       | int   | Retry count on failure (default 0)       |
| `retry_delay`       | float | Seconds between retries                  |
| `continue_on_error` | bool  | Skip step on error instead of failing    |
| `timeout`           | int   | Execution timeout in seconds             |
| `repeat_while`      | str   | Jinja2 expression, loop while truthy     |
| `repeat_until`      | str   | Jinja2 expression, loop until truthy     |
| `max_iterations`    | int   | Loop safety limit (default 10)           |

### Variable passing

When a step has `assign="var_name"`, the result is available in later steps:

- In Jinja2 templates: `{{ var_name }}`
- In Python code execution: just use `var_name` directly

## Other directives

### tsu:tool

Execute a tool during template rendering (before the agent runs):

```html
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->
```

The result is available as `{{ config }}` in the template.

An alternative is using the `prefetch` frontmatter.

### tsu:ignore

Remove content before rendering:

```html
<!-- tsu:ignore -->
This won't be sent to the LLM.  Useful for comments.
<!-- /tsu:ignore -->
```

## Network (sandbox)

When running with `--sandbox`, declare which domains the agent needs:

```yaml
network:
  domains:
    - api.github.com
    - "*.openai.com"
```

These are added to the sandbox proxy's allowlist alongside any `--allow-domain` flags from the CLI.
