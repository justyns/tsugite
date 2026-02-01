---
name: tsugite_skill_basics
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

## Skill File Structure

Skills are **single Markdown files** with YAML frontmatter (not directories):

```yaml
---
name: skill_name
description: Brief description of what this skill provides
---

# Skill Content

Your instructional content here. This is a Jinja2 template with limited context.

Available in templates:
- {{ today() }} - Current date (YYYY-MM-DD)
- {{ now() }} - Current timestamp (ISO format)
- {{ env.get("VAR_NAME") }} - Environment variables

Example: Today is {{ today() }}, use this for date-based examples.
```

**Required frontmatter fields:**
- `name`: Unique identifier (lowercase, use hyphens for spaces)
- `description`: Concise explanation (appears in skill index, helps agents decide when to load)

**Optional fields:** You can add custom metadata, but only `name` and `description` are used by tsugite.

## Discovery Order

Skills are discovered by scanning these directories in order (first match wins for each unique name):

1. `.tsugite/skills/` - Project-local skills (shared across project)
2. `skills/` - Project convention directory
3. `builtin_skills/` - Package-provided skills (from tsugite installation)
4. `~/.config/tsugite/skills/` - Global user skills

**Override behavior:** Place a skill with the same name earlier in the search path to override built-ins. For example, create `.tsugite/skills/python_math.md` to replace the built-in version.

## Loading Skills

### Auto-load in Agent Frontmatter

The most common way to load skills:

```yaml
---
name: my_agent
auto_load_skills:
  - python_best_practices
  - tsugite_jinja_reference
  - api_design_basics
tools:
  - read_file
---

Your agent content here.
```

Skills are loaded during agent preparation, before the first LLM turn.

### Dynamic Loading with Functions

Agents can also load skills during execution using the `load_skill` function:

```python
# Agent can decide to load skills based on the task
load_skill("python_math")
```

### List Available Skills

```python
# See all discoverable skills
list_available_skills()

# Output format:
# - python_math: Reference for performing common math operations in Python
# - tsugite_jinja_reference: Reference for Jinja templating in Tsugite agents
# - ...
```

## How Skills Are Injected Into Prompts

Understanding this helps debug issues and optimize skill usage:

1. **Agent preparation phase:**
   - `auto_load_skills` list processed
   - Each skill file loaded and validated
   - Skill content rendered as Jinja2 template (with minimal context)
   - Rendered content cached in `SkillManager._loaded_skills`

2. **System prompt construction:**
   - Skills become separate system message blocks
   - Each skill wrapped: `<Skill: name>\n{content}\n</Skill: name>`
   - Skills appear after main system prompt but before agent content
   - Each skill block tagged with `cache_control: {"type": "ephemeral"}`

3. **LLM receives:**
   ```
   System: [Base instructions]
   System: <Skill: python_math>
   [Python math content]
   </Skill: python_math>
   System: <Skill: tsugite_jinja_reference>
   [Jinja reference content]
   </Skill: tsugite_jinja_reference>
   System: [Agent-specific content]
   User: [User prompt]
   ```

This structure enables prompt caching - skills are cached separately and reused across conversations.

## Skill Template Context

Skills are rendered as Jinja2 templates, but with **minimal context** (unlike agents):

### Available Variables

- `{{ today() }}` - Current date string (YYYY-MM-DD)
- `{{ now() }}` - Current timestamp (ISO format: YYYY-MM-DDTHH:MM:SS)
- `{{ env }}` - Dictionary of environment variables
- `{{ env.get("VAR_NAME", "default") }}` - Safe environment variable access

### NOT Available in Skills

Skills cannot access agent-specific context:
- `user_prompt` - Set to empty string
- `tools` - Not available
- `is_interactive` - Not available
- Prefetch results - Not available
- Tool directive outputs - Not available

**Design implication:** Keep skills relatively static or use only date/time/env helpers. Skills should provide general knowledge, not task-specific context.

## Best Practices

### 1. Keep Skills Focused

One skill = one domain or workflow:
- ✅ Good: `python_math`, `api_design`, `git_workflows`
- ❌ Bad: `everything_python`, `misc_helpers`

### 2. Use Clear Structure

LLMs scan markdown structure efficiently:

```markdown
# Main Topic

## Subtopic 1
Brief explanation

**Key pattern:**
```code example```

## Subtopic 2
...
```

### 3. Provide Code Examples

Examples > abstract descriptions:

```python
# Good - Concrete example
from pathlib import Path
config = Path("~/.config/app/config.json").expanduser()

# Less helpful - Abstract explanation
# "Use pathlib for cross-platform paths"
```

### 4. Reference, Don't Embed

Link to official docs rather than copying entire API references:

```markdown
For advanced options, see [requests documentation](https://requests.readthedocs.io).

Common patterns:
- `requests.get(url, params={"key": "value"})` - Query parameters
- `requests.post(url, json=data)` - Send JSON payload
```

### 5. Test Rendering

Validate Jinja2 syntax before deploying:

```bash
# Create minimal test agent
cat > test_skill.md <<'EOF'
---
name: test
auto_load_skills: [your_skill_name]
tools: []
---
Test task
EOF

# Preview rendered output
tsugite render test_skill.md "test task" --debug
```

### 6. Use Descriptive Names

- Use lowercase with hyphens
- Make purpose obvious from name
- ✅ Good: `python_async_patterns`, `rest_api_design`
- ❌ Bad: `patterns`, `api_stuff`, `helper_1`

### 7. Respect Turn-Based Execution in Code Examples

**Critical:** Tsugite agents execute in a **turn-based model** - each code block runs independently and you don't see results until the next turn.

When writing code examples in skills, show realistic single-purpose code blocks:

✅ **Good - One function call per block:**
```python
# Step 1: Search for files
file_search(pattern="class.*Base", path="src")
```

Then in next turn:
```python
# Step 2: Read the file found in previous search
read_file("src/base.py")
```

❌ **Bad - Multiple operations assuming intermediate results:**
```python
# Don't show this - won't work!
results = file_search(pattern="class", path="src")
read_file(results[0])  # Can't use results - not visible until next turn
```

**How to structure multi-step workflows in skills:**
- Use numbered lists with separate code blocks
- Add "Then:" or "Next turn:" between steps
- Show text descriptions of what to do with results
- Emphasize that each code block is independent

**Example pattern:**
```markdown
1. First, search for authentication code:
```python
file_search(pattern="(?i)auth|login", path="src")
```

2. Based on search results, read the main auth file:
```python
read_file("src/auth/authenticate.py")
```

3. Then trace dependencies found in that file:
```python
file_search(pattern="verify_token", path="src")
```
```

This teaches agents the correct execution model and prevents unrealistic code examples.

## Common Issues & Debugging

### Skill Not Found

**Symptom:** "Skill 'foo' not found" message

**Causes:**
- Skill file doesn't exist in any search path
- Frontmatter missing `name` field
- Name mismatch (case-sensitive)
- Invalid YAML syntax

**Fix:**
```bash
# Check what skills are discovered
tsugite run +agent "list available skills"

# Verify file exists
ls skills/your_skill.md
ls .tsugite/skills/your_skill.md
```

### Skill Not Loaded (Silent Failure)

**Symptom:** Agent runs but skill content not visible, no error message

**Causes:**
- Skill frontmatter has invalid YAML (currently fails silently)
- Missing `name` field in frontmatter
- Jinja2 rendering error

**Fix:**
```bash
# Try loading the skill with an agent
tsugite render test_agent.md "test" --debug

# Check for parse errors manually
python -c "import yaml; print(yaml.safe_load(open('skills/your_skill.md').read().split('---')[1]))"
```

### Undefined Variable in Skill Template

**Symptom:** Template rendering fails with "undefined variable" error

**Cause:** Skill tries to use agent-specific variables (`user_prompt`, `tasks`, etc.)

**Fix:** Skills only have access to `today()`, `now()`, and `env`. Remove references to other variables or make them conditional:

```jinja2
{# Bad - will fail #}
Current task: {{ user_prompt }}

{# Good - use only available context #}
Today's date: {{ today() }}
Debug mode: {{ env.get("DEBUG", "false") }}
```

### Duplicate Skill Names

**Symptom:** Expected skill content doesn't appear

**Cause:** Another skill with same name found earlier in search path

**Fix:** Check all search locations and rename or move conflicting file:

```bash
find . -name "*.md" -path "*/skills/*" -o -path "*/.tsugite/skills/*"
```

## Creating Your First Skill

### Step 1: Choose Location

```bash
# Project-specific skill
mkdir -p skills
touch skills/my_skill.md

# Or project-local (preferred for shared skills)
mkdir -p .tsugite/skills
touch .tsugite/skills/my_skill.md
```

### Step 2: Add Frontmatter

```yaml
---
name: my_domain_skill
description: Brief description that helps agents know when to use this
---
```

### Step 3: Write Instructional Content

Focus on patterns, examples, and workflows:

```markdown
# My Domain Skill

## Common Patterns

When working with [domain], follow these conventions:

### Pattern 1: Description

```[language]
# Code example
```

### Pattern 2: Description

Key points:
- Point 1
- Point 2

## Anti-patterns to Avoid

- ❌ Don't do X because Y
- ✅ Instead, do Z
```

### Step 4: Test It

```bash
# Create test agent
cat > test.md <<'EOF'
---
name: test
auto_load_skills: [my_domain_skill]
tools: []
---
Test the skill
EOF

# Verify skill loads and renders
tsugite render test.md "test task" --debug | grep "Skill: my_domain_skill"
```

### Step 5: Use in Agents

```yaml
---
name: my_agent
auto_load_skills:
  - my_domain_skill
---

Agent content here
```

## Complete Example Skill

Here's a complete, production-ready skill:

```markdown
---
name: docker_compose_patterns
description: Docker Compose configuration patterns and best practices
---

# Docker Compose Patterns

## Basic Service Definition

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://db:5432/app
    depends_on:
      - db
    volumes:
      - ./app:/app
    restart: unless-stopped
```

## Development vs Production

**Development:** Use volume mounts for hot reload
```yaml
volumes:
  - ./src:/app/src  # Source code
  - /app/node_modules  # Don't override dependencies
```

**Production:** Use built images, no mounts
```yaml
image: myapp:{{ env.get("VERSION", "latest") }}
restart: always
```

## Common Service Patterns

**PostgreSQL:**
```yaml
db:
  image: postgres:16
  environment:
    POSTGRES_PASSWORD: ${DB_PASSWORD}
  volumes:
    - pgdata:/var/lib/postgresql/data
```

**Redis Cache:**
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

**Nginx Reverse Proxy:**
```yaml
nginx:
  image: nginx:alpine
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
  ports:
    - "80:80"
```

## Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Networks

Explicit networks for service isolation:
```yaml
services:
  app:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend

networks:
  frontend:
  backend:
```

Last updated: {{ today() }}
```

## Advanced Topics

### Skill Lifecycle

1. **Discovery:** Frontmatter scanned from all search paths
2. **Registration:** Unique names added to skill index
3. **Loading:** Full file read when requested
4. **Rendering:** Jinja2 template executed with minimal context
5. **Caching:** Rendered content stored in `_loaded_skills` dict
6. **Injection:** Added as separate system message blocks with cache markers

### Performance Considerations

- **Token cost:** Skills add tokens to every LLM call
- **Load only what you need:** Don't auto-load large skills unnecessarily
- **Leverage caching:** Skills use ephemeral caching, so repeated calls are cheaper
- **Keep focused:** Smaller, focused skills > large comprehensive ones

### Multi-Agent Scenarios

- Skills are **per-session:** Each agent execution has its own `SkillManager` instance
- No sharing between concurrent agents
- Subagents inherit parent's skill manager (skills are available to subagents)

## Summary

**Key Takeaways:**
- Skills are single `.md` files with frontmatter (not directories)
- Discovered from 4 locations (project-local, convention, built-in, global)
- Auto-loaded via `auto_load_skills` or dynamically via `load_skill` tool
- Injected as separate cached system message blocks
- Limited template context (date/time/env only)
- First match wins in discovery order (enables overrides)

**Next Steps:**
- Load `tsugite_jinja_reference` skill to learn about template syntax
- Browse `builtin_skills/` directory for examples
- Create project-specific skills in `skills/` or `.tsugite/skills/`
- Experiment with dynamic loading using `load_skill` tool
