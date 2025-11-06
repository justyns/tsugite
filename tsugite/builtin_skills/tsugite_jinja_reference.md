---
name: tsugite_jinja_reference
description: Reference for Jinja templating in Tsugite agents, including helpers and context variables; load when editing templates or debugging rendering
---

# Tsugite Jinja Reference

## Rendering Model

- Tsugite uses Jinja2 with `StrictUndefined`, `trim_blocks=True`, and `lstrip_blocks=True`. Any missing variable raises immediately.
- Rendering occurs after `<!-- tsu:ignore -->` blocks are stripped and after prefetch/tool directives populate the context.
- Instructions in frontmatter also pass through the same renderer, so keep logic consistent across body and `instructions`.

## Default Context Keys

Available in both the agent body and `instructions` during preparation:

### Always Available
- `user_prompt`: Raw user task text (string)
- `task_summary`: Markdown summary of tracked tasks (string)
- `tasks`: List of task dicts with keys: `title`, `status`, `optional` (list)
- `tools`: Expanded tool list after glob/category expansion (list of strings)
- `text_mode`: Boolean matching the agent config flag
- `is_interactive`: True when running in interactive CLI/TUI (boolean)

### Chat Mode Context
- `chat_history`: List of prior conversation turns (list of dicts with `role` and `content`)

### Subagent Context
- `is_subagent`: True if agent spawned by another agent (boolean)
- `parent_agent`: Parent agent name or None (string or None)

### Multi-Step Context (only in step content)
- `iteration`: Current loop iteration number (1-indexed, int)
- `max_iterations`: Maximum allowed iterations (int)
- `is_looping_step`: True if current step has repeat conditions (boolean)
- Variables from previous steps via `assign` attribute

### Dynamic Context
- Prefetch outputs: Each entry in `prefetch` injects its result by `assign` name
- Tool directive outputs: Each `assign` variable from `<!-- tsu:tool ... -->` is available

## Helper Functions

Injected via `AgentRenderer.env.globals`:

### Date/Time Helpers
- `now()` → ISO timestamp string (`YYYY-MM-DDTHH:MM:SS`)
  ```jinja2
  Generated at {{ now() }}
  ```
- `today()` → Date string (`YYYY-MM-DD`)
  ```jinja2
  Today's date: {{ today() }}
  ```

### String Helpers
- `slugify(text)` → Lowercase ASCII slug (replaces special chars with dashes)
  ```jinja2
  Filename: {{ user_prompt|slugify }}.txt
  URL: /tasks/{{ task_name|slugify }}
  ```

### Filesystem Helpers
- `file_exists(path)` → Boolean, checks if path exists
- `is_file(path)` → Boolean, checks if path is a file
- `is_dir(path)` → Boolean, checks if path is a directory
  ```jinja2
  {% if file_exists("config.json") %}
  Config found, loading settings...
  {% else %}
  No config, using defaults...
  {% endif %}
  ```

- `read_text(path, default="")` → Read file content, return default on error
  ```jinja2
  Instructions: {{ read_text(".instructions.md", "No instructions found") }}
  ```

### Environment Access
- `env` → Dictionary of environment variables (`os.environ`)
  ```jinja2
  Debug mode: {{ env.get("DEBUG", "false") }}
  API endpoint: {{ env["API_URL"] }}
  ```

## Task Management Helpers

### Task Loop Conditions

These helpers are used in multi-step `repeat_while` / `repeat_until` conditions:

- `has_pending_tasks()` → True if any tasks are not completed
- `all_tasks_complete()` → True if all tasks are completed
- `has_task_status(status)` → True if any task has given status
- `task_count_by_status(status)` → Count of tasks with given status

**Example usage in step directive:**
```markdown
<!-- tsu:step name="work_on_tasks" repeat_while="has_pending_tasks()" max_iterations="10" -->
Work on the next pending task until all tasks are complete.
```

## Control Patterns

### Conditional Blocks

**Text mode toggling:**
```jinja2
{% if text_mode %}
Provide explanations without code blocks.
{% else %}
You can use code blocks freely:
```python
# Example code
```
{% endif %}
```

**Environment-based configuration:**
```jinja2
{% if env.get("TSUGITE_DEBUG") == "true" %}
Debug mode enabled - provide verbose output.
{% endif %}

{% if env.get("SKIP_TESTS") %}
Skip running tests.
{% else %}
Run full test suite after implementation.
{% endif %}
```

**Interactive mode detection:**
```jinja2
{% if is_interactive %}
Ask clarifying questions if needed.
{% else %}
Make reasonable assumptions and proceed autonomously.
{% endif %}
```

**Subagent detection:**
```jinja2
{% if is_subagent %}
You are running as a subagent spawned by {{ parent_agent }}.
Report results concisely.
{% else %}
You are the primary agent. Provide detailed explanations.
{% endif %}
```

### Task Iteration

**Filter tasks by status:**
```jinja2
{% if tasks %}
Pending tasks:
{% for task in tasks if task.status == "pending" %}
- [ ] {{ task.title }}
{% endfor %}

In progress:
{% for task in tasks if task.status == "in_progress" %}
- [~] {{ task.title }}
{% endfor %}

Completed:
{% for task in tasks if task.status == "completed" %}
- [x] {{ task.title }}
{% endfor %}
{% else %}
No tasks defined.
{% endif %}
```

**Count and conditionals:**
```jinja2
{% set pending = tasks|selectattr("status", "equalto", "pending")|list %}
{% if pending|length > 0 %}
You have {{ pending|length }} pending task(s):
{% for task in pending %}
{{ loop.index }}. {{ task.title }}
{% endfor %}
{% else %}
All tasks complete!
{% endif %}
```

**Task summary with loop variables:**
```jinja2
{% for task in tasks %}
Task {{ loop.index }} of {{ loop.length }}: {{ task.title }}
Status: {{ task.status }}
{% if task.optional %}(optional){% endif %}
{% if not loop.last %}---{% endif %}
{% endfor %}
```

### Multi-Step Loops

**Iteration context:**
```jinja2
{% if is_looping_step %}
Iteration {{ iteration }} of {{ max_iterations }} maximum.
{% if iteration == 1 %}
This is the first iteration - start fresh.
{% elif iteration >= max_iterations - 1 %}
This is the last iteration - wrap up.
{% endif %}
{% endif %}
```

**Using previous step results:**
```jinja2
<!-- tsu:step name="research" assign="findings" -->
Research the topic and gather information.

<!-- tsu:step name="analyze" assign="analysis" -->
{% if findings %}
Previous research findings:
{{ findings }}

Now analyze these findings...
{% else %}
No findings from previous step.
{% endif %}
```

### Filter Chains

**String manipulation:**
```jinja2
{{ user_prompt|lower|replace(" ", "_") }}
{{ task_name|title|truncate(50) }}
{{ content|trim|wordcount }}
```

**List operations:**
```jinja2
{% set tool_names = tools|map("lower")|list %}
{% set unique_tools = tools|unique|sort|list %}
First tool: {{ tools|first }}
Last tool: {{ tools|last }}
Random tool: {{ tools|random }}
```

**Conditional filters:**
```jinja2
{% set important_tasks = tasks|selectattr("optional", "equalto", False)|list %}
{% set task_titles = tasks|map(attribute="title")|list %}
{% set completed = tasks|rejectattr("status", "equalto", "completed")|list %}
```

## Safety Notes

- Because of `StrictUndefined`, initialize optional values in `prefetch` or add defaults (e.g., `{{ metadata or "" }}`).
- Keep heavy file reads outside templates; prefer prefetch or tool directives to populate large blobs.
- When referencing attachments, remember they are added separately to the prompt—use instructions to mention their usage rather than embedding full content.

## Common Gotchas

### Undefined Variables

**Problem:** Template fails with "undefined variable" error
```jinja2
{{ optional_config }}  <!-- Fails if not defined -->
```

**Solutions:**
```jinja2
{# Option 1: Default value #}
{{ optional_config | default("default value") }}
{{ optional_config or "default value" }}

{# Option 2: Guard with if #}
{% if optional_config is defined %}
Config: {{ optional_config }}
{% endif %}

{# Option 3: Define in prefetch or tool directive #}
<!-- Use prefetch or tool directive to populate optional_config -->
```

### Loop Variable Scope

**Problem:** Loop variables not accessible outside loop
```jinja2
{% for task in tasks %}
  {% set task_name = task.title %}
{% endfor %}
{{ task_name }}  <!-- Not available here -->
```

**Solution:** Use namespace for cross-scope variables
```jinja2
{% set ns = namespace(last_task='') %}
{% for task in tasks %}
  {% set ns.last_task = task.title %}
{% endfor %}
Last task was: {{ ns.last_task }}
```

### Whitespace Control

**Problem:** Extra blank lines in output
```jinja2
{% for item in items %}
{{ item }}
{% endfor %}
<!-- Creates blank lines between items -->
```

**Solution:** Use whitespace control
```jinja2
{% for item in items -%}
{{ item }}
{% endfor -%}
{# Or use trim_blocks=True (already enabled in Tsugite) #}
```

### Escaping Special Characters

**Problem:** Need literal Jinja2 syntax in output
```jinja2
{{ user_input }}  <!-- This gets rendered -->
```

**Solution:** Use raw blocks or escape
```jinja2
{% raw %}
To use variables, write {{ variable_name }}
{% endraw %}

{# Or escape individual parts #}
Use {{ "{{" }} and {{ "}}" }} for Jinja syntax
```

### Filter vs Function

**Problem:** Confusion between filters and functions
```jinja2
{{ slugify user_prompt }}  <!-- Wrong -->
{{ tasks.length }}  <!-- Wrong -->
```

**Solution:** Know the difference
```jinja2
{# Filters use pipe #}
{{ user_prompt | slugify }}
{{ tasks | length }}

{# Functions use parentheses #}
{{ now() }}
{{ file_exists("config.json") }}
```

## Debugging Steps

### 1. Preview Rendered Template

```bash
# See exactly what the LLM receives
tsugite render +agent "task description" --debug
```

This shows:
- Full rendered agent content
- Available context variables
- Expanded tools list
- System prompt structure

### 2. Check Variable Availability

Add temporary diagnostic blocks:
```jinja2
<!-- Debug: Check what variables are available -->
{% if tasks is defined %}
Tasks: {{ tasks | length }} items
{% else %}
Tasks not defined
{% endif %}

{% if user_prompt %}
Prompt: "{{ user_prompt }}"
{% else %}
No user prompt
{% endif %}

<!-- Debug: Dump task structure -->
{% for task in tasks %}
Task {{ loop.index }}:
  title: {{ task.title }}
  status: {{ task.status }}
  optional: {{ task.optional }}
{% endfor %}
```

### 3. Guard Optional Variables

Always guard variables that may not exist:
```jinja2
{% if var is defined and var %}
  Use {{ var }}
{% endif %}

{{ var | default("fallback") }}
```

### 4. Validate Step Dependencies

For multi-step agents, ensure variables assigned:
```markdown
<!-- tsu:step name="step1" assign="result1" -->
Generate result...

<!-- tsu:step name="step2" -->
{% if result1 is defined %}
Previous result: {{ result1 }}
{% else %}
ERROR: step1 did not assign result1!
{% endif %}
```

### 5. Test Incrementally

Build templates incrementally:
1. Start with static content
2. Add one variable at a time
3. Test with `render` command after each addition
4. Add conditionals and loops last

### 6. Check Tool Directive Execution

Verify tool directives ran:
```markdown
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->

{% if config is defined %}
Config loaded successfully
{% else %}
Tool directive may have failed - check logs
{% endif %}
```

## Advanced Patterns

### Dynamic Tool Lists

```jinja2
{% set fs_tools = tools | select("match", ".*file.*") | list %}
You have access to these file tools:
{% for tool in fs_tools %}
- {{ tool }}
{% endfor %}
```

### Conditional Instructions

```jinja2
{% if "code_execution" in tools %}
You can execute Python code to solve this task.
{% endif %}

{% if "spawn_agent" in tools and not is_subagent %}
You can spawn subagents to parallelize work.
{% endif %}
```

### Custom Task Filtering

```jinja2
{% set urgent_tasks = [] %}
{% for task in tasks %}
  {% if "urgent" in task.title | lower or "asap" in task.title | lower %}
    {% set _ = urgent_tasks.append(task) %}
  {% endif %}
{% endfor %}

{% if urgent_tasks %}
🚨 URGENT TASKS (prioritize these):
{% for task in urgent_tasks %}
- {{ task.title }}
{% endfor %}
{% endif %}
```

### Template Macros

```jinja2
{% macro render_task(task) %}
**{{ task.title }}**
Status: {{ task.status }}
{% if task.optional %}(optional){% endif %}
{% endmacro %}

Current tasks:
{% for task in tasks %}
{{ render_task(task) }}
{% endfor %}
```

## Quick Reference Card

| Pattern | Example |
|---------|---------|
| **Variable** | `{{ user_prompt }}` |
| **Default** | `{{ var \| default("fallback") }}` |
| **Conditional** | `{% if condition %}...{% endif %}` |
| **Loop** | `{% for item in list %}...{% endfor %}` |
| **Filter** | `{{ text \| lower \| trim }}` |
| **Function** | `{{ now() }}` |
| **Comment** | `{# This is a comment #}` |
| **Raw block** | `{% raw %}{{ literal }}{% endraw %}` |
| **Set variable** | `{% set name = value %}` |
| **Check defined** | `{% if var is defined %}` |
| **Loop index** | `{{ loop.index }}` (1-indexed) |
| **Loop first/last** | `{% if loop.first %}` |
