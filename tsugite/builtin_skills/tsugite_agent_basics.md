---
name: tsugite_agent_basics
description: Overview of how Tsugite agents are defined, resolved, and executed; load when inspecting, authoring, or troubleshooting agents
---

# Tsugite Agent Basics

## Agent Files

Agents are Markdown files with YAML frontmatter followed by templated content.

### Required Fields
- `name`: Unique identifier for the agent

### Recommended Fields
- `description`: Brief explanation of agent purpose
- `model`: LLM model to use (falls back to config default)
- `tools`: List of tools agent can use (supports globs and categories)
- `max_turns`: Maximum reasoning iterations (default: 5)

### Optional Fields
- `prefetch`: List of tool calls to execute before rendering
- `auto_load_skills`: Skills to load before execution
- `attachments`: Files to include in prompt (with caching)
- `instructions`: Additional Jinja2-templated instructions
- `reasoning_effort`: For reasoning models (low/medium/high)
- `visibility`: public/private/internal (default: public)
- `spawnable`: Allow other agents to spawn this one (default: true)
- `extends`: Parent agent to inherit from (default: auto-detect base)
- `custom_tools`: Define agent-specific shell command tools
- `mcp_servers`: MCP server configuration
- `disable_history`: Opt out of conversation persistence
- `auto_context`: Auto-load CLAUDE.md, CONTEXT.md files

## Resolution Order

When an agent name is requested, Tsugite resolves it in this order (first match wins):
1. `.tsugite/{name}.md`
2. `agents/{name}.md`
3. `./{name}.md`
4. `tsugite/builtin_agents/{name}.md`
5. `$XDG_CONFIG_HOME/tsugite/agents/{name}.md`

Built-in agents can be overridden by placing a same-named file earlier in this chain.

## Inheritance System

### How Inheritance Works

Agents automatically inherit from a base agent unless you opt out. This enables:
- Shared tools and settings across projects
- Consistent instructions and guidelines
- Easy overrides for specific agents

**Inheritance chain:**
1. **Default base agent** (auto-detected):
   - `.tsugite/default.md` (project-local base)
   - Config's `default_base_agent` setting
   - Built-in "default" agent (fallback)

2. **Explicit parent** (via `extends` field):
   - `extends: my_base` - Inherit from my_base agent
   - `extends: none` - Opt out of inheritance

3. **Current agent** (highest precedence)

### Merge Strategy

**Scalars (single values):** Child overwrites parent
- `model`, `max_turns`, `reasoning_effort`
- Child value completely replaces parent value

**Lists (arrays):** Merge and deduplicate
- `tools`, `attachments`, `auto_load_skills`
- Both parent and child values included
- Duplicates removed (first occurrence wins)

**Lists (no deduplication):** Concatenate
- `prefetch`
- Parent items first, then child items
- Order preserved

**Lists of dicts:** Deduplicate by name
- `custom_tools`
- Merged based on "name" field
- Child entry overrides parent if same name

**Dicts (objects):** Deep merge
- `mcp_servers`
- Child keys override parent keys
- New keys added

**Strings (special):** Concatenate
- `instructions`
- Parent instructions + "\n\n" + child instructions

### Inheritance Examples

**Example 1: Project Base Agent**

`.tsugite/default.md`:
```yaml
---
name: default
tools:
  - read_file
  - write_file
  - list_directory
auto_load_skills:
  - python_best_practices
max_turns: 10
instructions: |
  Follow project coding standards.
  Write tests for new functionality.
---

You are a helpful coding assistant.
```

`agents/api_developer.md`:
```yaml
---
name: api_developer
extends: default  # Explicit (optional, would auto-detect anyway)
tools:
  - http_request  # Added to inherited tools
  - search_web
auto_load_skills:
  - api_design_basics  # Added to inherited skills
max_turns: 15  # Overrides parent's 10
instructions: |
  Design RESTful APIs following OpenAPI 3.0 spec.
---

You specialize in API development.
```

**Result after merging:**
```yaml
name: api_developer
tools: [read_file, write_file, list_directory, http_request, search_web]
auto_load_skills: [python_best_practices, api_design_basics]
max_turns: 15
instructions: |
  Follow project coding standards.
  Write tests for new functionality.

  Design RESTful APIs following OpenAPI 3.0 spec.
```

**Example 2: Opt Out of Inheritance**

```yaml
---
name: standalone_agent
extends: none  # No inheritance
tools:
  - code_execution
---

Minimal agent with only specified tools.
```

**Example 3: Custom Tool Inheritance**

Parent:
```yaml
custom_tools:
  - name: deploy
    command: ./scripts/deploy.sh
```

Child:
```yaml
custom_tools:
  - name: deploy  # Same name - overrides parent
    command: ./scripts/deploy.sh --env=staging
  - name: test  # New tool - added
    command: pytest
```

Result: Child's deploy command replaces parent's, test tool added.

## Directives & Instructions

Tsugite augments Markdown with three types of directives:

### Ignore Blocks

Strip documentation/comments before rendering:

```markdown
<!-- tsu:ignore -->
This content is for developers, not shown to LLM.
Can include multiple lines, examples, notes, etc.
<!-- /tsu:ignore -->

This content IS shown to the LLM.
```

### Tool Directives

Execute tools during agent preparation (before rendering):

```markdown
<!-- tsu:tool name="read_file" args={"path": "config.json"} assign="config" -->

The config file contains: {{ config }}
```

**When to use:**
- Load data needed during template rendering
- Alternative to `prefetch` frontmatter field
- Results available as template variables

### Step Directives

Define multi-step workflows (see Multi-Step Agents section below).

```markdown
<!-- tsu:step name="research" assign="findings" -->
Research the topic...

<!-- tsu:step name="summarize" assign="summary" -->
Summarize findings from previous step...
```

### Instructions Field

The `instructions` field in frontmatter supports Jinja2 templates:

```yaml
instructions: |
  {% if env.get("PRODUCTION") == "true" %}
  IMPORTANT: This is production - be extra careful!
  {% endif %}

  Follow these guidelines:
  {% for guideline in project_guidelines %}
  - {{ guideline }}
  {% endfor %}
```

Instructions are concatenated: parent + "\n\n" + child (via inheritance).

## Preparation Pipeline

`AgentPreparer` follows a fixed pipeline:
1. Parse frontmatter into `AgentConfig` (unknown keys rejected).
2. Execute `prefetch` tool calls.
3. Execute tool directives (or stub them during `render`).
4. Build the template context (`user_prompt`, tasks, tool outputs, etc.).
5. Render Markdown via Jinja2 with strict undefined handling.
6. Combine default + agent instructions.
7. Expand tool specs (globs, categories) and create tool objects.
8. Build the final system prompt + user prompt payload.

Understanding this flow helps target failures (e.g., prefetch errors, missing template vars, tool expansion issues).

## Skill and Attachment Hooks

- `auto_load_skills` lists skills to load before the first LLM turn.
- `attachments` lists attachments to load before the first LLM turn.

## Multi-Step Agents

Multi-step agents use `<!-- tsu:step -->` directives to create workflows with multiple sequential phases.

### Basic Multi-Step Structure

```markdown
---
name: research_and_write
tools:
  - search_web
  - write_file
---

<!-- Preamble: Rendered once, shared across all steps -->
You are a research and writing assistant.

<!-- tsu:step name="research" assign="findings" -->
Research the topic: {{ user_prompt }}
Gather comprehensive information.

<!-- tsu:step name="outline" assign="outline" -->
Previous research:
{{ findings }}

Create an outline based on the research.

<!-- tsu:step name="write" assign="article" -->
Research findings:
{{ findings }}

Outline:
{{ outline }}

Write the final article.
```

**How it works:**
1. Preamble rendered once (content before first step)
2. Each step executes independently with isolated context
3. Variables persist across steps via `assign` attribute
4. Previous step results available in next step

### Step Attributes

**Basic:**
- `name`: Required, unique step identifier
- `assign`: Optional, variable name to store step result

**Execution Control:**
- `timeout`: Seconds before timeout (default: no limit)
- `max_retries`: Retry attempts on failure (default: 0)
- `retry_delay`: Seconds between retries (default: 1)
- `continue_on_error`: Continue to next step if this fails (default: false)

**Model Parameters:**
- `temperature`: Override agent's temperature for this step
- `max_tokens`: Override token limit for this step
- `top_p`: Override top_p for this step

**Loop Control:**
- `repeat_while`: Jinja2 condition, repeat while true
- `repeat_until`: Jinja2 condition, repeat until true
- `max_iterations`: Maximum loop iterations (default: 10)

**JSON Output:**
- `json`: true - Request JSON output
- `response_format`: JSON schema for structured output

### Looping Steps Examples

**Example 1: Item Processing Loop**

```markdown
---
name: item_processor
---

<!-- tsu:step name="process_items" repeat_while="items | length > 0" max_iterations="10" -->
{% if iteration == 1 %}
Starting item processing...
{% endif %}

Current iteration: {{ iteration }} of {{ max_iterations }}

Process the next item from the list. Remove it when complete.
```

**Example 2: Convergence Loop**

```markdown
---
name: iterative_improver
---

<!-- tsu:step name="improve" assign="result" repeat_until="result.contains('DONE')" max_iterations="5" -->
{% if iteration == 1 %}
Initial attempt:
{% else %}
Previous result:
{{ result }}

Iteration {{ iteration }}: Improve based on feedback.
{% endif %}

When satisfied, end your response with "DONE".
```

**Example 3: Error Retry Loop**

```markdown
---
name: resilient_agent
---

<!-- tsu:step name="risky_operation" max_retries="3" retry_delay="2" continue_on_error="true" -->
Attempt risky operation.
If it fails, will retry up to 3 times with 2 second delay.

<!-- tsu:step name="fallback" -->
This step runs even if previous step failed (continue_on_error=true).
Implement fallback logic here.
```

### Loop Context Variables

Available in `repeat_while` / `repeat_until` conditions:

- `iteration` - Current iteration number (1-indexed)
- `max_iterations` - Maximum iterations allowed
- Any Jinja2 expression using step variables

**Example conditions:**
```markdown
repeat_while="not done"
repeat_until="result == 'success'"
repeat_while="items | length > 0"
```

### Step Variables in Templates

**Loop context:**
```jinja2
{% if is_looping_step %}
This is a looping step.
Current iteration: {{ iteration }}
Maximum iterations: {{ max_iterations }}
{% endif %}
```

**Previous step results:**
```jinja2
{% if research_findings is defined %}
Research from previous step: {{ research_findings }}
{% else %}
No research findings available yet.
{% endif %}
```

**Conditional based on iteration:**
```jinja2
{% if iteration == 1 %}
First iteration - gather data.
{% elif iteration == max_iterations %}
Last iteration - finalize results.
{% else %}
Iteration {{ iteration }} - continue processing.
{% endif %}
```

### Multi-Step Best Practices

1. **Use preamble for shared context**
   - Agent identity, general instructions
   - Shared once across all steps

2. **Keep steps focused**
   - Each step should have single clear goal
   - Avoid mixing research, analysis, and output in one step

3. **Pass data explicitly**
   - Use `assign` to capture step results
   - Reference previous results in next step's template

4. **Set reasonable timeouts**
   - Prevent steps from hanging indefinitely
   - Longer timeouts for complex steps (research, analysis)

5. **Use continue_on_error sparingly**
   - Only when next step can handle failure
   - Implement fallback logic in subsequent steps

6. **Test loop conditions carefully**
   - Ensure loops can terminate
   - Always set `max_iterations` as safety net

7. **Monitor iteration count**
   - Provide feedback at key iterations
   - Adjust strategy if approaching max_iterations

## Debugging Tips

- Use `tsugite render +agent "prompt" --debug` to inspect the fully rendered prompt and executed directives.
- If a template variable is undefined, Jinja raises immediately because Tsugite uses `StrictUndefined`.
- Inspect inheritance via `tsugite agents show --chain {name}` (CLI helper) to confirm which base files contribute settings.
