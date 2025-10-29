# Agent Development Guide

Quick reference for building, composing, and running Tsugite agents.

## Commands

| Task | Command |
|------|---------|
| Run agent | `tsugite run agent.md "task"` or `tsugite run +name "task"` |
| Chat mode | `tsugite chat` or `tsugite chat +agent` |
| Continue latest | `tsugite run --continue "prompt"` or `tsugite chat --continue` |
| Continue specific | `tsugite run --continue --conversation-id CONV_ID "prompt"` or `tsugite chat --continue CONV_ID` |
| List history | `tsugite history list` |
| Show conversation | `tsugite history show CONV_ID` |
| Debug prompt | `tsugite run agent.md "task" --debug` |
| Plain output | `tsugite run +agent "task" --plain` |
| Headless (CI) | `tsugite run +agent "task" --headless` |
| Final only | `tsugite run +agent "task" --final-only` (or `--quiet`) |
| Render prompt | `tsugite render agent.md "task"` |
| Render raw | `tsugite render agent.md "task" --raw` |
| Multi-agent | `tsugite run +coordinator +helper1 +helper2 "task"` |
| MCP register | `tsugite mcp add name --url http://host/mcp` |
| MCP test | `tsugite mcp test name --trust-code` |
| Custom tool | `tsugite tools add name -c "cmd {arg}" -p arg:required` |
| Config | `tsugite config set model provider:model` |
| Model alias | `tsugite config set model-alias fast openai:gpt-4o-mini` |

## Conversation Continuity

Both `run` and `chat` modes support continuing previous conversations using the `--continue` flag.

### Chat Mode Resume

```bash
# Resume latest conversation
tsugite chat --continue

# Resume specific conversation
tsugite chat --continue CONV_ID

# Resume latest conversation with different agent
tsugite chat +different-agent --continue
```

**Behavior:**
- Loads conversation history into chat UI
- Displays all previous turns
- Continues appending to same conversation ID
- Resumes from last turn timestamp

### Run Mode Continuation

Run mode becomes multi-turn when using `--continue`:

```bash
# Continue latest conversation (auto-detects agent)
tsugite run --continue "next prompt"

# Continue latest conversation with specific agent
tsugite run +agent --continue "next prompt"

# Continue specific conversation by ID (auto-detects agent)
tsugite run --continue --conversation-id CONV_ID "next prompt"

# Continue with different agent than original (agent switching)
tsugite run +different-agent --continue --conversation-id CONV_ID "prompt"
```

**Behavior:**
- Loads previous conversation as `chat_history` context
- Agent sees full conversation history
- Executes next turn with new prompt
- Appends to same conversation ID

### Finding Conversations

```bash
# List all conversations
tsugite history list

# List by machine
tsugite history list --machine laptop

# List by agent
tsugite history list --agent researcher

# Show conversation details
tsugite history show CONV_ID

# Show as JSON
tsugite history show CONV_ID --format json
```

### Disabling History

**Global:**
```yaml
# ~/.config/tsugite/config.yaml
history_enabled: false
```

**Per-agent:**
```yaml
---
name: my_agent
disable_history: true
---
```

**Per-run:**
```bash
tsugite run +agent "task" --no-history
tsugite chat --no-history
```

## Agent Structure

Agents are Markdown + YAML frontmatter:

```markdown
---
name: my_agent
model: ollama:qwen2.5-coder:7b
max_turns: 5
tools: [read_file, write_file]
---

Task: {{ user_prompt }}
```

### Frontmatter Fields

| Key | Default | Description |
|-----|---------|-------------|
| `name` | — (required) | Agent identifier |
| `model` | Config default | `provider:model[:variant]` |
| `max_turns` | `5` | Reasoning turns (think-act cycles) per workflow step |
| `tools` | `[]` | Tool names, globs (`*_search`), categories (`@fs`), exclusions (`-delete_file`) |
| `custom_tools` | `[]` | Per-agent shell command wrappers |
| `prefetch` | `[]` | Tools to run before rendering |
| `initial_tasks` | `[]` | Tasks to pre-populate (strings or dicts with title/status/optional) |
| `attachments` | `[]` | Context to auto-load |
| `instructions` | — | Extra system guidance |
| `extends` | Config default | Parent agent to inherit from |
| `mcp_servers` | — | MCP servers + tool safelists |
| `context_budget` | Unlimited | Prompt length cap |
| `reasoning_effort` | — | For o1/o3 models: `low`, `medium`, `high` |
| `text_mode` | `false` | Allow text responses without code blocks |

### Template Helpers

| Helper | Example |
|--------|---------|
| `{{ user_prompt }}` | CLI prompt argument |
| `{{ tasks }}` | List of all tasks with id, title, status, optional fields (for iteration) |
| `{{ task_summary }}` | Formatted string summary of all tasks (for display) |
| `{{ is_subagent }}` | `True` if spawned by another agent, `False` otherwise |
| `{{ parent_agent }}` | Name of parent agent (e.g., `"coordinator"`) or `None` if not a subagent |
| `{{ now() }}` | ISO 8601 timestamp |
| `{{ today() }}` | `YYYY-MM-DD` date |
| `{{ slugify(text) }}` | Filesystem-safe slug |
| `{{ file_exists(path) }}` | Boolean path check |
| `{{ is_file(path) }}` / `{{ is_dir(path) }}` | Type checks |
| `{{ read_text(path, default="") }}` | Safe file read |
| `{{ env.HOME }}` | Environment variables |

## Built-in Agents

### default

Minimal base agent with task tracking. Default base for inheritance.

```bash
tsugite run default "task"
```

### chat-assistant

Conversational agent for chat mode. Tools: `read_file`, `write_file`, `list_files`, `web_search`, `run`. Text mode enabled.

```bash
tsugite chat  # Uses this by default
```

### Overriding Package Agents

Package-provided agents are stored as `.md` files in the package's `builtin_agents/` directory. You can override them by creating agents with the same name in your project directories. Project agents take precedence over package-provided agents.

Create `.tsugite/default.md` or `agents/default.md` to override for your project.

## Agent Resolution Order

When referencing by name (e.g., `+myagent`):

1. `.tsugite/{name}.md` (project-local shared)
2. `agents/{name}.md` (project convention)
3. `./{name}.md` (current directory)
4. Package agents directory (`tsugite/builtin_agents/`)
5. Global agent directories (`~/.config/tsugite/agents/`, etc.)

Explicit paths skip resolution: `tsugite run ./path/to/agent.md`

**Note:** Package-provided agents are checked after project agents, allowing you to override them locally.

## Agent Inheritance

```yaml
---
name: specialized
extends: default  # Inherit from package-provided default
model: openai:gpt-4o       # Override model
tools: [read_file, run]    # Add tools
---
```

**Inheritance chain:** Default base → Extended → Current

**Merge rules:**
- Scalars (model, max_turns): Child overwrites
- Lists (tools): Merge + deduplicate
- Dicts (mcp_servers): Merge, child keys override
- Strings (instructions): Concatenate with `\n\n`

**Opt out:** `extends: none`

**Configure default base:**

```bash
tsugite config set default_base_agent default
tsugite config set default_base_agent none  # Disable
```

## Pre-populating Tasks

Agents can start with predefined tasks using `initial_tasks` in frontmatter. Tasks are automatically populated when the agent starts and persist across multi-step execution.

**Simple format** (strings default to pending/required):
```yaml
---
name: code_reviewer
initial_tasks:
  - "Read and analyze the code"
  - "Check for security issues"
  - "Check for performance problems"
---
```

**Detailed format** (specify status and optional flag):
```yaml
---
name: project_manager
initial_tasks:
  - title: "Core feature implementation"
    status: pending
    optional: false
  - title: "Add nice formatting"
    status: pending
    optional: true
  - title: "Write documentation"
    optional: true  # status defaults to pending
---
```

**Mixed format**:
```yaml
---
name: developer
initial_tasks:
  - "Must implement authentication"  # Required
  - title: "Performance optimization"
    optional: true  # Nice-to-have
  - "Write tests"  # Required
---
```

**Inheritance**: Parent tasks are inherited first, then child tasks:
```yaml
---
name: specialized_reviewer
extends: code_reviewer  # Gets parent's tasks
initial_tasks:
  - "Check code style"  # Added to parent's tasks
---
```

Optional tasks are marked with ✨ in the task summary. Agents are instructed to complete all required tasks, while optional tasks are nice-to-have.

## Looping Through Tasks in Templates

Tasks are available as the `tasks` variable for Jinja2 iteration in agent templates:

```yaml
---
name: task_processor
initial_tasks:
  - "Analyze data"
  - title: "Generate report"
    optional: true
  - "Write summary"
---

# Task List

{% for task in tasks %}
{{ loop.index }}. {{ task.title }} ({{ task.status }})
   {% if task.optional %}✨ Optional{% endif %}
{% endfor %}

# Instructions

Process each task systematically...
```

**Available task fields:**
- `task.id` - Task ID (integer)
- `task.title` - Task description
- `task.status` - Current status (pending/in_progress/completed/blocked/cancelled)
- `task.optional` - Boolean (true/false)
- `task.parent_id` - Parent task ID or null
- `task.created_at`, `task.updated_at`, `task.completed_at` - ISO 8601 timestamps

**Common patterns:**

Filter by status:
```jinja2
{% for task in tasks if task.status == "pending" %}
- TODO: {{ task.title }}
{% endfor %}
```

Show only required tasks:
```jinja2
{% for task in tasks if not task.optional %}
- {{ task.title }} (REQUIRED)
{% endfor %}
```

Count tasks by type:
```jinja2
Required: {{ tasks | selectattr('optional', 'equalto', false) | list | length }}
Optional: {{ tasks | selectattr('optional', 'equalto', true) | list | length }}
```

## Context Injection

### File References

```bash
tsugite run +agent "Review @src/main.py"
tsugite run +agent "Fix @app.py based on @tests/test.py"
```

### Attachments

```bash
# Create reusable context
tsugite attachments add standards project/STYLE.md
tsugite attachments add api-docs https://api.example.com/docs

# Use in prompts
tsugite run +agent "task" -f standards -f api-docs

# Agent-defined attachments
---
attachments:
  - standards
  - api-docs
---
```

**Resolution order:** Agent attachments → CLI `-f` → File refs `@` → Prompt

### Custom Tools

**Global:** `~/.config/tsugite/custom_tools.yaml`

```yaml
tools:
  - name: file_search
    description: Search with ripgrep
    command: "rg {pattern} {path}"
    parameters:
      pattern: {required: true}
      path: "."
```

**Per-agent:**

```yaml
---
custom_tools:
  - name: local_grep
    command: "grep -r {pattern} ."
    parameters:
      pattern: {required: true}
---
```

See `CUSTOM_TOOLS.md` for full guide.

### Prefetch

Execute tools before rendering:

```yaml
---
prefetch:
  - tool: read_file
    args: {path: "config.json"}
    assign: config
---

Config: {{ config }}
```

Failures assign `None`. Guard with `{% if config %}`.

## Interactive User Input

Tsugite provides two tools for collecting user input during agent execution, available only in interactive mode:

### ask_user - Single Question

Ask one question at a time:

```python
# Text input
name = ask_user(question="What is your name?", question_type="text")

# Yes/No question
save = ask_user(question="Save to file?", question_type="yes_no")
# Returns: "yes" or "no"

# Multiple choice (arrow key navigation)
format = ask_user(
    question="Choose format:",
    question_type="choice",
    options=["json", "txt", "md"]
)
```

**Question types:**
- `text`: Freeform text input
- `yes_no`: Binary yes/no question
- `choice`: Multiple choice with arrow key navigation (requires `options` list)

### ask_user_batch - Multiple Questions

For better UX when collecting multiple related inputs, use `ask_user_batch` to ask all questions at once:

```python
responses = ask_user_batch(questions=[
    {"id": "name", "question": "What is your name?", "type": "text"},
    {"id": "email", "question": "Email address?", "type": "text"},
    {"id": "save", "question": "Save to file?", "type": "yes_no"},
    {
        "id": "format",
        "question": "Choose format:",
        "type": "choice",
        "options": ["json", "txt", "md"]
    }
])

# Access responses by ID
user_name = responses["name"]
user_email = responses["email"]
should_save = responses["save"]  # "yes" or "no"
file_format = responses["format"]
```

**Question structure:**
- `id` (required): Unique identifier used as key in response dict
- `question` (required): The question text
- `type` (required): Question type - "text", "yes_no", or "choice"
- `options` (optional): List of options for "choice" type (minimum 2)

**Features:**
- Arrow key navigation (↑/↓) or vim-style (j/k) for choice questions
- Colored interface with highlighting
- Protection against accidental double-Enter
- Returns dict mapping question IDs to responses
- Validates question structure and enforces unique IDs

**Agent configuration:**

```yaml
---
name: user_registration
tools: [ask_user_batch, write_file]
---
```

Interactive tools are automatically filtered out in non-interactive mode (e.g., CI/headless). Check `{{ is_interactive }}` variable in templates.

## File Editing Tools

Tsugite provides intelligent file editing tools that allow LLMs to make precise edits without reading/writing entire files.

### read_file - Read Files with Optional Line Range

Read entire files or specific line ranges:

```python
read_file(path, start_line=None, end_line=None)
```

**Parameters:**
- `path`: File path (required)
- `start_line`: Starting line number, 1-indexed (0 also accepted, treated as 1) (optional)
- `end_line`: Ending line number, 1-indexed, inclusive (optional)

**Returns:**
- If `start_line` is None: Full file content as plain text
- If `start_line` is provided: Numbered lines in format "LINE_NUM: content"

**Examples:**
```yaml
---
tools: [read_file]
---

# Read entire file
{{ read_file("config.json") }}

# Read specific lines
{{ read_file("src/main.py", start_line=10, end_line=20) }}

# Read from line 50 to end
{{ read_file("data.txt", start_line=50) }}
```

**Use cases:**
- Reading entire files (backward compatible)
- Reading specific functions or sections
- Verifying content before editing
- Reducing context usage for large files

### get_file_info - File Metadata

Get file information without reading full content:

```python
get_file_info(path)
```

**Returns:**
- `line_count`: Total lines in file
- `size_bytes`: File size
- `last_modified`: ISO timestamp
- `exists`: Whether file exists
- `is_directory`: Whether path is a directory

**Example:**
```yaml
---
tools: [get_file_info, read_file_lines]
---

{% set info = get_file_info("config.json") %}
File has {{ info.line_count }} lines
```

### edit_file - Smart File Editing (Single or Batch)

Edit files with intelligent multi-strategy matching. Supports both single edits and batch edits in one tool.

```python
# Single edit mode
edit_file(path, old_string, new_string, expected_replacements=1)

# Batch edit mode
edit_file(path, edits=[...])
```

**Parameters:**
- `path`: File path (required)
- **Single edit mode:**
  - `old_string`: Text to find (include 3+ lines of context)
  - `new_string`: Replacement text (must differ from old_string)
  - `expected_replacements`: Expected number of matches (default: 1)
- **Batch edit mode:**
  - `edits`: List of edit operations, each with:
    - `old_string`: Text to find (required)
    - `new_string`: Replacement text (required)
    - `expected_replacements`: Match count (optional, default: 1)

**Replacement Strategies** (tried in order):
1. **Exact match** - Direct string matching
2. **Line-trimmed** - Ignores leading/trailing whitespace per line
3. **Block-anchor** - Matches using first/last lines as anchors with fuzzy middle content
4. **Whitespace-normalized** - Normalizes all whitespace to single spaces
5. **Indentation-flexible** - Strips minimum indentation before matching

**Examples:**

Single edit:
```yaml
---
tools: [read_file, edit_file]
---

<!-- First, read to verify content -->
Current content:
{{ read_file("app.py", start_line=15, end_line=20) }}

<!-- Then edit with context -->
{{ edit_file(
    path="app.py",
    old_string="def process_data():\n    return raw_data\n    # TODO: add validation",
    new_string="def process_data():\n    validate(raw_data)\n    return raw_data"
) }}
```

Batch edits (atomic operation):
```yaml
---
tools: [edit_file]
---

{{ edit_file(
    path="config.py",
    edits=[
        {"old_string": "DEBUG = True", "new_string": "DEBUG = False"},
        {"old_string": "TIMEOUT = 30", "new_string": "TIMEOUT = 60"},
        {"old_string": "LOG_LEVEL = 'INFO'", "new_string": "LOG_LEVEL = 'ERROR'"}
    ]
) }}
```

**Best practices:**
- Include 3+ lines of context in `old_string` for single edits
- Use `read_file` with line range first to verify content
- For multiple occurrences, specify `expected_replacements`
- Batch mode is atomic: if any edit fails, none are applied
- Each batch edit operates on the result of the previous edit

**Use cases:**
- Single edits: Precise code changes with context validation
- Batch edits: Multiple related changes in one atomic operation
- Configuration updates: Change multiple settings together
- Refactoring: Rename variables consistently throughout a file

### Common Workflow

```markdown
---
name: code_updater
tools: [get_file_info, read_file, edit_file]
---

1. Check file info:
{% set info = get_file_info("{{ user_prompt }}") %}
File exists: {{ info.exists }}, Lines: {{ info.line_count }}

2. Read relevant section:
{% if info.exists and info.line_count < 100 %}
{{ read_file("{{ user_prompt }}") }}
{% else %}
{{ read_file("{{ user_prompt }}", start_line=1, end_line=50) }}
{% endif %}

3. Make targeted edit (single or batch):
{{ edit_file(
    path="{{ user_prompt }}",
    old_string="[exact text with context]",
    new_string="[modified text]"
) }}
```

### Error Handling

All file editing tools provide clear, actionable error messages:

- **No matches found**: "No matches found. Ensure old_string matches file content exactly. Use read_file to verify."
- **Multiple matches**: "Found 3 matches but expected 1. Either add more context or use expected_replacements=3."
- **Identical strings**: "Search and replace strings must be different."
- **File not found**: "File not found: /path/to/file"
- **Conflicting parameters**: "Provide either old_string/new_string OR edits, not both"
- **Missing parameters**: "Must provide either old_string/new_string OR edits"

### Tool Summary

Tsugite's file editing tools are now consolidated for simplicity:

| Tool | Purpose | Modes |
|------|---------|-------|
| `read_file` | Read files | Full file OR line range |
| `edit_file` | Edit files | Single edit OR batch edits |
| `get_file_info` | File metadata | Info without reading content |

**Total: 3 tools** (down from 5, simpler for LLMs to use)

## Multi-Step Agents

```markdown
---
name: researcher
max_turns: 10
tools: [web_search, write_file]
---

Topic: {{ user_prompt }}

<!-- tsu:step name="research" assign="findings" -->
Research {{ user_prompt }}.

<!-- tsu:step name="summarize" assign="summary" -->
Summarize: {{ findings }}

<!-- tsu:step name="save" -->
Write {{ summary }} to file.
```

**Key concepts:**
- Content before first directive = preamble (shared across steps)
- Step context: `{{ step_number }}`, `{{ total_steps }}`, `{{ step_name }}`
- Variables persist across steps
- Execution stops on first failure

**Step parameters:**

```markdown
<!-- tsu:step name="..." assign="..." temperature="0.7" reasoning_effort="high" continue_on_error="true" timeout="30" -->
```

### Multi-Step Agent Reliability Features

Phase 1 improvements for robust long-running agents:

#### Better Error Messages

When a step fails, errors now include:
- Step number and name
- Previous step context
- Available variables
- All retry attempts
- Debugging suggestions

Example error:
```
Step Execution Failed
────────────────────────────────────────────────────────────
Step: fetch_data (2/5)
Previous Step: initialize
Attempts: 3

Available Variables: config, user_id

Errors:
  Attempt 1: Connection timeout after 30s
  Attempt 2: Rate limit exceeded
  Attempt 3: Network unreachable

💡 Tips:
  - Check if required variables are set
  - Verify previous step completed successfully
  - Review step dependencies
```

#### Continue on Error

Allow steps to fail gracefully without aborting the entire agent. Failed step assigns `None` to its variable.

```markdown
<!-- tsu:step name="optional_fetch" assign="extra_data" continue_on_error="true" -->
Fetch supplementary data that's nice to have but not critical.

<!-- tsu:step name="process" -->
Process results. Extra data: {{ extra_data or "Not available" }}
```

**Use cases:**
- Optional data enrichment
- Non-critical API calls
- Best-effort operations
- Partial results acceptable

**Behavior:**
- Step executes normally
- On failure, assigns `None` to variable
- Logs warning with error details
- Continues to next step

#### Per-Step Timeouts

Prevent individual steps from hanging indefinitely. Timeout in seconds.

```markdown
<!-- tsu:step name="quick_check" timeout="10" -->
Fast operation that should complete quickly.

<!-- tsu:step name="web_scrape" timeout="60" -->
Fetch data from potentially slow endpoints.

<!-- tsu:step name="analysis" timeout="300" -->
Deep analysis that needs more time.
```

**Behavior:**
- Step canceled after timeout
- Raises `asyncio.TimeoutError`
- Respects `continue_on_error` if set
- No timeout = unlimited (default)

**Combine features:**
```markdown
<!-- tsu:step name="external_api" assign="api_data" timeout="30" continue_on_error="true" -->
Fetch from external API with 30s timeout. Continue if it fails.
```

#### Dry-Run Mode

Preview agent execution without running. Shows execution plan, dependencies, and cost estimates.

```bash
tsugite run +my_agent "task" --dry-run
```

**Output:**
```
Agent: my_agent
Model: anthropic:claude-3-5-sonnet
Total Steps: 5

Step Execution Plan:
──────────────────────────────────────────────────────────────
Step 1: initialize
  Variables: config
  Dependencies: None
  Timeout: None
  Continue on error: No

Step 2: fetch_data
  Variables: raw_data
  Dependencies: config
  Timeout: 60s
  Continue on error: No

Step 3: process
  Variables: processed
  Dependencies: raw_data
  Timeout: None
  Continue on error: No

[...]

Estimated Cost: ~$0.15 (based on ~60K tokens @ $3/MTok)
Estimated Duration: ~3-5 minutes
```

**Use for:**
- Validating step dependencies
- Estimating costs before long runs
- Debugging agent structure
- Planning resource allocation

#### Step Metrics

Automatic performance tracking displayed after execution:

```
Step Execution Metrics
──────────────────────────────────────────────────────────────
Step              Duration    Status    Error
──────────────────────────────────────────────────────────────
initialize           2.3s   success
fetch_data          45.1s   success
optional_api         0.0s   skipped   Connection refused
process             12.8s   success
generate_report     34.2s   success
──────────────────────────────────────────────────────────────
Total              94.4s
──────────────────────────────────────────────────────────────
```

**Metrics tracked:**
- Step name and number
- Duration in seconds
- Status (success, failed, skipped)
- Error message if failed/skipped
- Total execution time

**Use for:**
- Identifying bottlenecks
- Optimizing slow steps
- Tracking agent performance
- Debugging execution flow

### Looping Steps

Steps can repeat based on conditions using `repeat_while` or `repeat_until`:

**Repeat while condition is true:**
```markdown
<!-- tsu:step name="task_worker" repeat_while="has_pending_required_tasks" max_iterations="20" -->
Process one task. This step repeats while there are pending required tasks.
```

**Repeat until condition is true:**
```markdown
<!-- tsu:step name="process" repeat_until="all_tasks_complete" max_iterations="15" -->
Work on tasks. Repeats until all tasks are marked complete.
```

**Custom Jinja2 expressions:**
```markdown
<!-- tsu:step name="worker" repeat_while="{{ tasks | selectattr('status', 'equalto', 'pending') | list | length > 0 }}" -->
Process pending tasks. Uses Jinja2 to check if any pending tasks remain.
```

**Available helper conditions:**
- `has_pending_tasks` - Any tasks with status=pending
- `has_pending_required_tasks` - Any non-optional pending tasks
- `all_tasks_complete` - All tasks are completed
- `has_incomplete_tasks` - Any tasks not completed
- `has_in_progress_tasks` - Any tasks currently in progress
- `has_blocked_tasks` - Any tasks marked as blocked

**Loop context variables:**
- `{{ iteration }}` - Current iteration number (1-indexed)
- `{{ max_iterations }}` - Maximum allowed iterations
- `{{ is_looping_step }}` - Boolean indicating if step can loop

**Parameters:**
- `repeat_while` - Condition to continue repeating (Jinja2 expression or helper name)
- `repeat_until` - Condition to stop repeating (Jinja2 expression or helper name)
- `max_iterations` - Maximum iterations (default: 10)

**Example: Process all tasks one at a time**
```yaml
---
name: task_processor
initial_tasks:
  - "Task 1"
  - "Task 2"
  - "Task 3"
max_turns: 50
---

<!-- tsu:step name="process_one_task" repeat_while="has_pending_tasks" max_iterations="20" -->

## Iteration {{ iteration }} / {{ max_iterations }}

Find the first pending task, work on it, and mark it complete.
This step will repeat until all tasks are done.
```

**Safety:**
- Steps stop at `max_iterations` to prevent infinite loops
- Default `max_iterations` is 10 (can be overridden)
- Warning displayed when limit is reached

## Directives

### Documentation Blocks

Strip content before LLM sees it:

```markdown
<!-- tsu:ignore -->
## Developer Notes
This agent does X. Usage: `tsugite run agent.md "task"`
Can contain {{ undeclared_vars }} without errors.
<!-- /tsu:ignore -->
```

Multiple blocks supported. Useful for inline documentation.

### Tool Directives

Execute tools during rendering:

```markdown
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->

Config: {{ config }}
```

**Multi-step support:**

```markdown
<!-- tsu:tool name="read_file" args={"path": "sources.txt"} assign="sources" -->

<!-- tsu:step name="research" -->
<!-- tsu:tool name="read_file" args={"path": "context.txt"} assign="context" -->
Research using {{ context }} and {{ sources }}
```

**Scoping:**
- Preamble directives: Execute once, available to all steps
- Step directives: Execute per step, scoped to that step
- Previous step variables always available

**vs. Prefetch:** Tool directives are inline and support per-step execution. Prefetch is global in frontmatter.

## Model Providers

Format: `provider:model[:variant]`

**Examples:**
- `ollama:qwen2.5-coder:7b`
- `openai:gpt-4o`
- `anthropic:claude-3-5-sonnet`
- `google:gemini-pro`

**API keys:** Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

**Aliases:**

```bash
tsugite config set model-alias fast openai:gpt-4o-mini
tsugite run +agent "task" --model fast
```

## Reasoning Models

**Full reasoning visible:**
- Claude (`claude-3-7-sonnet`)
- Deepseek

**Token counts only:**
- OpenAI o1/o3 (shows "🧠 Used N reasoning tokens")

**Control effort** (o1, o1-preview, o3, o3-mini only):

```yaml
---
model: openai:o1
reasoning_effort: high  # low, medium, high
---
```

Per-step:

```markdown
<!-- tsu:step name="quick" reasoning_effort="low" -->
<!-- tsu:step name="deep" reasoning_effort="high" -->
```

**Limitations:** o1/o3 models don't support `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`.

## MCP Integration

```yaml
---
mcp_servers:
  basic-memory: null              # All tools
  pubmed: [search, fetch_article] # Specific tools only
---
```

**Register:**

```bash
tsugite mcp add basic-memory --url http://localhost:3000/mcp
tsugite mcp add pubmed --command uvx --args "pubmedmcp@0.1.3"
```

**Run:** Requires `--trust-mcp-code` for remote code execution.

## Multi-Agent Composition

```bash
tsugite run +coordinator +helper1 +helper2 "task"
```

First agent = coordinator. Gets `spawn_{name}()` tools for each helper.

### Subagent Context Awareness

When an agent spawns another agent using `spawn_agent()`, the spawned agent receives context variables:

```yaml
---
name: helper
---
{% if is_subagent %}
I was spawned by {{ parent_agent }}.
{% else %}
I'm running as a top-level agent.
{% endif %}
```

**Context variables:**
- `{{ is_subagent }}` - `True` when spawned by another agent, `False` otherwise
- `{{ parent_agent }}` - Name of the parent agent (e.g., `"coordinator"`), or `None` if top-level

**Example use cases:**
- Adjust verbosity (subagents might be less verbose)
- Skip user interaction in subagents (they can't prompt the user)
- Return structured data instead of formatted output
- Track delegation chains for debugging

## Development

### Setup

```bash
uv sync --dev
uv run pytest                              # All tests
uv run pytest tests/test_file.py::test_x  # Specific test
uv run black . && uv run ruff check .
```

### Project Structure

- `cli/__init__.py` - Typer CLI (run, chat, render, config, mcp, tools, etc.)
- `md_agents.py` - Agent parsing, directives
- `builtin_agents.py` - Package agent utilities
- `agent_inheritance.py` - Resolution + inheritance
- `agent_runner.py` - Execution orchestration
- `core/agent.py` - LiteLLM agent loop
- `renderer.py` - Jinja2 rendering
- `chat.py` - Chat session management
- `ui/textual_chat.py` - Textual TUI
- `tools/` - Tool registry
- `models.py` - Model string parsing
- `config.py` - Configuration

### Style

- Python 3.12+, type hints on public APIs
- Imports: stdlib → third-party → local (blank lines between)
- Line length: 88 characters (Black)
- Errors: `ValueError` for input, `RuntimeError` for execution failures
- Docstrings: `Args:` / `Returns:` for public functions

### Adding Package Agents

Package-provided agents are now file-based for consistency:

1. Create `tsugite/builtin_agents/new_agent.md`:

```markdown
---
name: new_agent
description: Description of new agent
tools: []
---
{{ user_prompt }}
```

2. The agent will be automatically discovered - no code changes needed
3. Users can reference it with `+new_agent` or `new_agent`
4. Add tests to verify the agent works
5. Ensure `pyproject.toml` includes the builtin_agents directory in package data

### Testing

```python
from tsugite.md_agents import parse_agent_file
from tsugite.chat import ChatManager
from tsugite.tools import get_tool

def test_agent():
    agent = parse_agent_file(Path("agent.md"))
    assert agent.config.name == "expected"

def test_chat():
    manager = ChatManager(Path("agent.md"), max_history=10)
    response = manager.run_turn("Hello")
    assert len(manager.conversation_history) == 1

def test_tool():
    tool = get_tool("read_file")
    assert tool is not None
```

## Event System

Tsugite uses an event-driven architecture for UI and progress tracking. Events are emitted during agent execution and handled by UI modules (rich console, plain text, JSONL, Textual TUI).

### Event Structure

**Location:** `tsugite/events/`
- `base.py` - EventType enum, BaseEvent class
- `events.py` - All 19 event classes (consolidated)
- `bus.py` - EventBus for dispatching events

**19 Event Types:**
- Execution: `TASK_START`, `STEP_START`, `CODE_EXECUTION`, `TOOL_CALL`, `OBSERVATION`, `FINAL_ANSWER`
- LLM: `LLM_MESSAGE`, `EXECUTION_RESULT`, `EXECUTION_LOGS`, `REASONING_CONTENT`, `REASONING_TOKENS`
- Meta: `COST_SUMMARY`, `STREAM_CHUNK`, `STREAM_COMPLETE`, `INFO`, `ERROR`
- Progress: `DEBUG_MESSAGE`, `WARNING`, `STEP_PROGRESS`

### Error Handling Patterns

1. **Tool Results** (`ObservationEvent`):
   - Success: `ObservationEvent(success=True, observation="result", tool="tool_name")`
   - Failure: `ObservationEvent(success=False, error="error msg", tool="tool_name")`

2. **Code Execution** (`ExecutionResultEvent`):
   - Success: `ExecutionResultEvent(success=True, logs=[...], output="result")`
   - Failure: `ExecutionResultEvent(success=False, error="error msg")`

3. **General/Fatal Errors** (`ErrorEvent`):
   - `ErrorEvent(error="error msg", error_type="Error Type", step=N)`
   - Used for: Format errors, max turns exceeded, critical failures

### JSONL Protocol

For subprocess-based subagents, events are serialized to JSONL:
- Tool results: `{"type": "tool_result", "tool": "name", "success": bool, "output"?: str, "error"?: str}`
- Errors: `{"type": "error", "error": str, "step": int}`
- Full schema documented in `tsugite/ui/jsonl.py`

## Additional Resources

- **Custom tools:** `CUSTOM_TOOLS.md`
- **Docker:** `bin/README.md`
- **Examples:** `agents/examples/`
