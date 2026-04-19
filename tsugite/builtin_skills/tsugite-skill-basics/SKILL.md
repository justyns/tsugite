---
name: tsugite-skill-basics
description: How Tsugite skills are structured, discovered, and loaded at runtime; load when creating, auditing, or debugging skills
---

# Tsugite Skill Basics

## What Are Skills?

Skills are reusable knowledge modules that agents can load on-demand. They provide domain-specific guidance, reference material, or workflow patterns without bloating the base prompt.

**Key benefits:**
- Token efficiency: Load only what you need
- Modularity: Share skills across agents
- Prompt caching: Skills use ephemeral cache for better performance
- Maintainability: Update skills independently of agents

## Skill Layout

Tsugite follows the [agentskills.io](https://agentskills.io) specification. A skill is a **directory containing `SKILL.md`**, optionally with bundled resources:

```
skill-name/
    SKILL.md            # required: frontmatter + instructions
    scripts/            # optional: executable scripts the agent can run
    references/         # optional: longer-form docs the agent can read on demand
    assets/             # optional: templates, data, or other static resources
```

`SKILL.md` itself is a Jinja2-rendered Markdown file:

```yaml
---
name: skill-name
description: Brief description of what this skill provides
---

# Skill Content

Your instructional content here.

Available in templates:
- {{ today() }} - Current date (YYYY-MM-DD)
- {{ now() }} - Current timestamp (ISO format)
- {{ env.get("VAR_NAME") }} - Environment variables
```

**Required frontmatter:**
- `name`: lowercase alphanumerics + hyphens, must match the directory name, ≤64 characters.
- `description`: concise one-liner (shown in the skill index so the agent knows when to load it).

**Optional frontmatter the spec defines** (tsugite parses but does not yet enforce): `license`, `compatibility`, `metadata`, `allowed-tools`.

**Tsugite extension:** `triggers: [keywords]` - when any listed keyword appears in the user prompt (word-boundary match, case-insensitive), the skill is auto-loaded. Not part of the spec; safe to omit for cross-client portability.

## Discovery Order

Skills are discovered by scanning these roots in priority order (first match per unique name wins):

1. Workspace: `<workspace>/.agents/skills/`, `<workspace>/skills/`
2. User-configured `skill_paths` from agent frontmatter
3. Project: `.agents/skills/`, `.tsugite/skills/`, `skills/`
4. Built-in: `builtin_skills/` (shipped with tsugite)
5. User: `~/.agents/skills/`, `~/.config/tsugite/skills/`

`.agents/skills/` is the cross-client standard, so any skill authored for another agentskills.io client can be dropped in.

**Override behavior:** Place a directory with the same `name` earlier in the search order to shadow a built-in. For example, `.tsugite/skills/python-math/SKILL.md` replaces the built-in `python-math`.

## Loading Skills

### Auto-load in Agent Frontmatter

```yaml
---
name: my-agent
auto_load_skills:
  - python-best-practices
  - tsugite-jinja-reference
  - api-design-basics
tools:
  - read_file
---

Agent content here.
```

Skills listed here load during agent preparation, before the first LLM turn.

### Trigger-based Auto-load (tsugite extension)

Give a skill `triggers` in its frontmatter and tsugite will auto-load it whenever a trigger word appears in the user prompt:

```yaml
---
name: docker-compose-patterns
description: Docker Compose patterns and best practices
triggers:
  - docker
  - compose
  - docker-compose
---
```

### Dynamic Loading

Agents can load a skill mid-execution:

```python
load_skill("python-math")
```

### List Available Skills

```python
list_available_skills()
# - python-math: Reference for performing common math operations in Python
# - tsugite-jinja-reference: Reference for Jinja templating in Tsugite agents
```

## Bundled Resources

When a skill ships with `scripts/`, `references/`, or `assets/`, tsugite appends a `<skill_resources>` block to the loaded content:

```
<skill_resources dir="/abs/path/to/skill-name">
- scripts/validate.sh
- references/api.md
- assets/template.json
</skill_resources>
```

The agent uses the `dir` attribute to execute scripts (`bash /abs/path/to/skill-name/scripts/validate.sh`) and to read references on demand. Referenced paths in your `SKILL.md` should stay relative to the skill directory - see the [spec guidance on scripts](https://agentskills.io/skill-creation/using-scripts).

## How Skills Are Injected

1. **Preparation:** `auto_load_skills` + trigger-matched skills are loaded; each `SKILL.md` is read, rendered, and its bundled resources are listed.
2. **Caching:** Rendered content is stored in `SkillManager._loaded_skills`.
3. **Injection:** Each skill becomes a tagged block in the cached context turn, using the tag recommended by the agentskills.io client-implementation doc:

   ```
   <skill_content name="python-math">
   ...rendered SKILL.md...
   </skill_content>
   ```

   The context turn is cached ephemerally so repeated calls reuse the same tokens.

## Skill Template Context

Skills are rendered with a **minimal context**:

- `{{ today() }}` - Current date (YYYY-MM-DD)
- `{{ now() }}` - Current timestamp (ISO)
- `{{ env }}` - Dict of environment variables
- `{{ user_prompt }}` - Empty string (reserved)

**Not available:** `tools`, `is_interactive`, prefetch results, tool-directive outputs. Skills should provide general knowledge, not task-specific context.

## Best Practices

### 1. Keep Skills Focused

One skill = one domain or workflow.
- Good: `python-math`, `api-design`, `git-workflows`
- Bad: `everything-python`, `misc-helpers`

### 2. Spec-compliant Names

Lowercase alphanumerics + hyphens, matching the directory name. Tsugite warns (but still loads) when this is violated, and some agentskills.io tooling will refuse to validate a non-compliant skill.

### 3. Provide Code Examples

Concrete examples beat abstract prose. Show real invocations, not pseudocode.

### 4. Reference, Don't Embed

For long API references, use `references/` and point to the file from `SKILL.md`:

```markdown
For the full API, read `references/api.md`.
```

The agent will only read it when the task actually requires it (progressive disclosure).

### 5. Respect Turn-based Execution in Code Examples

Tsugite agents execute in a **turn-based model** - each code block runs independently and you don't see results until the next turn.

Good (one call per block):

```python
# Step 1: Search
file_search(pattern="class.*Base", path="src")
```

Then:

```python
# Step 2: Read a result
read_file("src/base.py")
```

Bad (chains results inside one block - won't work):

```python
results = file_search(pattern="class", path="src")
read_file(results[0])
```

## Common Issues

### Skill Not Found

- Directory not in any search path, or missing `SKILL.md`.
- `name` in frontmatter doesn't match the target you're loading.
- Invalid YAML frontmatter - check with `python -c "import yaml; print(yaml.safe_load(open('skill-name/SKILL.md').read().split('---')[1]))"`.

### Undefined Variable in Template

Skills only have `today()`, `now()`, `env`, and `user_prompt`. Remove references to other variables.

### Name/Directory Mismatch Warning

The frontmatter `name` must match the parent directory name. Tsugite logs a warning but still loads the skill.

## Creating Your First Skill

```bash
mkdir -p .tsugite/skills/my-domain-skill
cat > .tsugite/skills/my-domain-skill/SKILL.md <<'EOF'
---
name: my-domain-skill
description: Brief description that helps agents know when to use this
---

# My Domain Skill

## Common Patterns

When working with [domain], follow these conventions:

### Pattern 1

```python
# concrete example
```

## Anti-patterns

- Don't do X because Y
- Instead, do Z
EOF
```

Test it by rendering an agent that loads it:

```bash
cat > test.md <<'EOF'
---
name: test
auto_load_skills: [my-domain-skill]
tools: []
---
Test task
EOF

tsugite render test.md "test task" --debug
```

## Summary

- Skills are **directories with `SKILL.md`**, optionally with `scripts/`, `references/`, `assets/`.
- Discovered from workspace, project, built-in, and user roots; `.agents/skills/` is the cross-client path.
- Loaded via `auto_load_skills`, `triggers` (tsugite extension), or `load_skill()`.
- Rendered with minimal Jinja2 context; bundled resources are enumerated in the loaded content.
- First match wins, so project skills shadow built-ins.

**Next Steps:**
- Load `tsugite-jinja-reference` for template syntax.
- Browse `builtin_skills/` for directory-based examples.
- Create project skills in `.tsugite/skills/` or `.agents/skills/`.
