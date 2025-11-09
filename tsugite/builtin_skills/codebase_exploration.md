---
name: codebase_exploration
description: Strategies and patterns for exploring unfamiliar codebases efficiently, with adaptive approaches based on size and scope
---

# Codebase Exploration Strategies

This skill provides systematic approaches for understanding unfamiliar codebases. Use these patterns to build comprehensive mental models while managing token budgets efficiently.

## Important: Turn-Based Execution Model

**Each code block executes independently** - you won't see results until the next turn. This means:

✅ **Good:** One tool call per code block
```python
file_search(pattern="class.*Base", path="src")
```

❌ **Bad:** Multiple operations in one block (you won't see intermediate results)
```python
# Don't do this - you can't use results from first call in second
results = file_search(pattern="class", path="src")
read_file(results[0])  # Won't work - results not visible until next turn
```

**Best practice:** Execute one tool, observe results, then decide next action based on what you learned.

## Multi-Phase Exploration Pattern

Explore codebases in progressive phases, each building on the previous:

### Phase 1: Structure Discovery (Quick)
**Goal:** Understand project organization and technology stack

**Actions:**
- Use `glob` to discover file patterns: `**/*.py`, `**/*.ts`, `src/**/*`
- Identify key directories (src, tests, docs, config)
- Count files to estimate size (informs next phases)
- Look for configuration files (package.json, pyproject.toml, Cargo.toml)
- Find documentation (README, CLAUDE.md, docs/)

**Example Pattern:**
1. List Python files to count them
```python
list_files(path=".", pattern="*.py")
```
2. Find documentation
```python
list_files(path=".", pattern="README.md")
```
3. Understand testing structure
```python
list_files(path="tests", pattern="test_*.py")
```
4. Find configuration files
```python
list_files(path=".", pattern="*.json")
```

### Phase 2: Entry Points & Architecture (Standard)
**Goal:** Find where execution begins and identify key abstractions

**Actions:**
- Locate entry points: main.py, cli.py, __main__.py, index.ts, app.py
- Read core configuration and setup files
- Identify base classes, interfaces, key abstractions
- Map directory structure to architectural layers
- Find routing/dispatch mechanisms (if applicable)

**Example Pattern:**
1. Read main entry point
```python
read_file("src/main.py")
```
2. Find base abstractions
```python
file_search(pattern="class.*Base|class.*Abstract|Protocol|Interface", path="src")
```
3. Read key base class
```python
read_file("src/base.py")
```
4. Extract symbols from main module
```python
file_search(pattern="^def |^class ", path="src/main.py")
```

### Phase 3: Symbol Mapping (Deep)
**Goal:** Build a map of key classes, functions, and their relationships

**Actions:**
- Extract class/function signatures from core modules
- Identify imports and dependencies
- Map module relationships
- Find cross-cutting concerns (auth, logging, error handling)

**Example Pattern:**
1. Find core symbols
```python
file_search(pattern="^class |^def |^async def", path="src")
```
2. Map dependencies
```python
file_search(pattern="^from |^import", path="src")
```
3. Find special patterns (decorators, routes, commands)
```python
file_search(pattern="@decorator|@.*router|@.*command", path="src")
```
4. Read 3-5 most central files identified from previous searches
```python
read_file("src/core/router.py")
```

### Phase 4: Deep Dive & Execution Tracing (Thorough)
**Goal:** Understand specific features or trace execution flows end-to-end

**Actions:**
- Trace one feature from entry point to data layer
- Follow function call chains
- Understand data transformations
- Identify error handling patterns

**Example Pattern:**
1. Find authentication-related code
```python
file_search(pattern="login|auth|authenticate", path="src")
```
2. Read authentication entry point
```python
read_file("src/auth/login.py")
```
3. Trace service layer (based on imports seen in login.py)
```python
read_file("src/auth/service.py")
```
4. Follow to repository/database layer
```python
file_search(pattern="class.*Repository|def.*query", path="src/auth")
```

## Adaptive Thoroughness Strategies

Automatically adjust exploration depth based on codebase characteristics:

### Small Codebase (<100 files)
- **Approach:** Phase 1 + Phase 2 (structure + entry points)
- **Depth:** Can read most key files
- **Time:** 5-10 tool calls
- **Output:** Concise overview with file listings

**Strategy:**
- List all files to see structure, then read README and main entry point
- Read 5-10 core files based on initial discovery
- Generate simple architectural summary

**Example turn sequence:**
```python
list_files(path=".", pattern="*")
```
Then:
```python
read_file("README.md")
```
Then:
```python
read_file("src/main.py")
```

### Medium Codebase (100-1000 files)
- **Approach:** All 4 phases, selective deep dives
- **Depth:** Symbol maps + targeted file reads
- **Time:** 15-25 tool calls
- **Output:** Structured report with symbol map and key patterns

**Strategy:**
- Phase 1: Count files and discover structure
- Phase 2: Read entry points and configuration
- Phase 3: Extract symbols from core modules
- Phase 4: Deep dive into specific areas based on findings

**Example turn sequence:**
```python
run("find . -type f | wc -l")
```
Then:
```python
list_files(path=".", pattern="*")
```
Then:
```python
read_file("README.md")
```
Then:
```python
file_search(pattern="^class |^def ", path="src")
```

### Large Codebase (>1000 files)
- **Approach:** Focused exploration only (avoid full scans)
- **Depth:** Narrow scope first, then expand
- **Time:** 20-30+ tool calls
- **Output:** Scope-limited report with references to unexplored areas

**Strategy:**
- Start with focused scope (see Focused Exploration below)
- Use file_search extensively (faster than reading files)
- Build symbol map for specific subdirectories only
- Read only most critical files (3-5 max initially)

**Example turn sequence:**
```python
file_search(pattern="class|def", path="src/specific_module")
```
Then:
```python
read_file("src/main.py")
```
Then continue exploring based on findings.

## Focused Exploration

When scope is too large or user requests specific area investigation:

### By Path/Directory
**Use case:** "Explore the API layer" or "Understand src/auth"

**Turn-by-turn pattern:**
1. List files in scope
```python
list_files(path="src/api", pattern="*")
```
2. Find API symbols
```python
file_search(pattern="^class |^def ", path="src/api")
```
3. Read main file
```python
read_file("src/api/__init__.py")
```
4. Continue exploring routes/endpoints and data models based on findings

### By Technology/Language
**Use case:** "Show me all TypeScript components" or "Python utilities"

**Turn-by-turn pattern:**
1. Find component files
```python
list_files(path=".", pattern="*.tsx")
```
2. Search for exports
```python
file_search(pattern="export (default |const |function)", path=".")
```
3. Read representative files
```python
read_file("src/components/Button.tsx")
```

### By Concern/Feature
**Use case:** "How is authentication handled?" or "Database access patterns"

**Turn-by-turn pattern:**
1. Find authentication code
```python
file_search(pattern="(?i)auth|login|session", path=".")
```
2. Read main authentication module (identified from search results)
```python
read_file("src/auth/main.py")
```
3. Trace auth flow from entry to verification
```python
file_search(pattern="verify|validate|check", path="src/auth")
```

### By Pattern/Convention
**Use case:** "Find all API endpoints" or "Locate CLI commands"

**Turn-by-turn pattern:**
1. Search for pattern markers
```python
file_search(pattern="@app.route|@router|@command|@click", path=".")
```
2. Read routing configuration
```python
read_file("src/routes.py")
```
3. Map pattern usage across codebase based on findings

## Tool Usage Patterns

### File Discovery (list_files)
**Best for:** Finding files by name pattern, understanding structure

```python
# List all Python files
list_files(path=".", pattern="*.py")

# Technology detection
list_files(path="src", pattern="*.ts")

# Find test files
list_files(path="tests", pattern="test_*.py")

# Configuration files
list_files(path=".", pattern="*.{json,yaml,toml}")
```

**Note:** Use `file_exists(path)` to check if specific files exist before reading.

### Content Search (file_search)
**Best for:** Finding symbols, patterns, keywords across many files

The `file_search` tool uses **ripgrep** (rg) for fast, powerful code search:

```python
# Function/class definitions
file_search(pattern="^class |^def |^async def", path=".")

# Imports and dependencies
file_search(pattern="^from |^import", path="src")

# Specific patterns
file_search(pattern="@router|@app.route", path=".")  # Web routes
file_search(pattern="TODO|FIXME|XXX", path=".")      # Code annotations
file_search(pattern="raise |throw ", path=".")        # Error patterns

# Case-insensitive feature search (use (?i) for case-insensitive in ripgrep)
file_search(pattern="(?i)authentication|login", path=".")
```

**Ripgrep pattern syntax:**
- `^` - Start of line
- `$` - End of line
- `|` - OR operator
- `(?i)` - Case-insensitive flag
- `\b` - Word boundary
- `.` - Any character
- `*` - Zero or more
- `+` - One or more

**Performance tips:**
- Ripgrep is **extremely fast** - don't hesitate to search entire codebases
- Automatically ignores .gitignore'd files and binary files
- Use specific patterns to reduce noise

### Advanced Code Search (Optional)

If available via custom tools configuration, consider using:

**ast-grep (structural search):**
- Search code by AST patterns, not text
- Example: Find all function calls with specific signatures
- More precise than regex for code structure

**fd (fast file finder):**
- Modern alternative to `find`
- Respects .gitignore automatically
- Faster for large codebases

**Configure custom tools** in `~/.config/tsugite/custom_tools.yaml`:
```yaml
tools:
  - name: ast_search
    description: Search code using AST patterns
    command: "ast-grep -p '{pattern}' {path}"
    parameters:
      pattern: {required: true}
      path: "."
```

### File Reading (read_file)
**Best for:** Deep understanding of specific files

```python
# Read full file
read_file(path="src/main.py")

# Read specific line range (for large files)
read_file(path="src/main.py", start_line=1, end_line=50)
```

**Priority order for reading:**
1. Entry points (main.py, index.ts, cli.py)
2. Configuration (setup.py, package.json, Cargo.toml)
3. Documentation (README.md, ARCHITECTURE.md)
4. Base classes and core abstractions
5. Representative examples (one controller, one model, one test)

**Avoid reading everything - be selective!**

### Shell Commands (run/run_safe)
**Best for:** Quick directory listings, file counts, specialized commands

```python
# Directory structure
run("ls -la")
run("ls src/")

# File counts by type
run("find . -name '*.py' | wc -l")

# Quick tree view (if installed)
run("tree -L 2 src/")

# Git information
run("git log --oneline -10")
run("git branch --show-current")
```

**Use `run_safe` for more restrictive execution** (blocks potentially dangerous commands).

## Structured Report Format

Generate exploration reports in this format:

```markdown
# Codebase Exploration Report

**Project:** [Name from README or config]
**Size:** [Small/Medium/Large] ([N] files)
**Primary Languages:** [Python, TypeScript, etc.]
**Explored:** [YYYY-MM-DD]

## Overview

[2-3 sentence description of what the codebase does]

## Architecture

**Structure:**
- `src/` - [Description]
- `tests/` - [Description]
- `docs/` - [Description]

**Key Patterns:**
- [Pattern 1: e.g., "MVC architecture"]
- [Pattern 2: e.g., "Dependency injection via constructor"]
- [Pattern 3: e.g., "Async/await for all I/O"]

## Entry Points

**Main Execution:**
- `src/main.py:10` - Application entry point
- `src/cli.py:5` - Command-line interface

**Key Abstractions:**
- `src/core/base.py:20` - `BaseAgent` - Foundation for all agents
- `src/tools/tool.py:15` - `Tool` - Tool interface

## Core Modules

### [Module 1 Name]
**Location:** `src/module1/`
**Purpose:** [What it does]
**Key Files:**
- `file1.py` - [Description]
- `file2.py` - [Description]

### [Module 2 Name]
**Location:** `src/module2/`
**Purpose:** [What it does]
**Key Files:**
- `file1.py` - [Description]

## Cross-Cutting Concerns

**Authentication/Authorization:** [How it's handled]
**Error Handling:** [Patterns used]
**Logging:** [Logging framework and usage]
**Configuration:** [Config system]

## Technology Stack

**Runtime:** [Python 3.11, Node.js 20, etc.]
**Key Dependencies:**
- [Library 1] - [Purpose]
- [Library 2] - [Purpose]

**Build/Dev Tools:**
- [Tool 1] - [Purpose]

## Testing Strategy

**Framework:** [pytest, jest, etc.]
**Coverage:** [If visible in config]
**Test Location:** `tests/`
**Key Patterns:** [How tests are organized]

## Next Steps for Deep Exploration

Areas that warrant further investigation:
1. [Area 1] - [Why interesting]
2. [Area 2] - [Why interesting]

---
*Report generated via codebase_exploration skill on {{ today() }}*
```

## Common Exploration Scenarios

### Scenario 1: "Give me an overview of this codebase"
**Approach:** Phase 1 + Phase 2 (structure + entry points)

**Turn-by-turn:**
1. Determine codebase size
```python
run("find . -type f | wc -l")
```
2. Read project description
```python
read_file("README.md")
```
3. Count primary language files
```python
list_files(path=".", pattern="*.py")
```
4. Read main entry point
```python
read_file("src/main.py")
```
5. Explore structure and generate overview report

### Scenario 2: "How is [feature] implemented?"
**Approach:** Focused exploration by concern

**Turn-by-turn:**
1. Find files related to feature
```python
file_search(pattern="(?i)authentication", path=".")
```
2. Read most relevant file from search results
```python
read_file("src/auth/authenticate.py")
```
3. Trace function calls by searching for imported functions
```python
file_search(pattern="def verify_token", path="src")
```
4. Summarize implementation approach

### Scenario 3: "Explain the architecture"
**Approach:** Phase 1 + Phase 2 + Phase 3

**Turn-by-turn:**
1. Analyze directory structure
```python
list_files(path=".", pattern="*")
```
2. Read architectural documentation if exists
```python
read_file("ARCHITECTURE.md")
```
3. Find base classes and interfaces
```python
file_search(pattern="class.*Base|ABC", path=".")
```
4. Read key abstractions and document patterns

### Scenario 4: "Find where [thing] is used"
**Approach:** Search-based exploration

**Turn-by-turn:**
1. Search for all occurrences
```python
file_search(pattern="UserModel", path=".")
```
2. Read representative usage sites
```python
read_file("src/controllers/user_controller.py")
```
3. Summarize usage patterns

### Scenario 5: "What tests exist for [component]?"
**Approach:** Test-focused exploration

**Turn-by-turn:**
1. Find test files
```python
list_files(path="tests", pattern="*auth*")
```
2. Search for test cases
```python
file_search(pattern="def test_|it\\(|describe\\(", path="tests")
```
3. Read representative test
```python
read_file("tests/test_auth.py")
```
4. Note testing patterns and coverage

## Best Practices

### Progressive Context Building
- **Start broad, narrow down:** Overview → specific areas → individual files
- **Build mental map first:** Structure before details
- **Use file_search before read_file:** Find relevant files before reading deeply
- **Read selectively:** 5-10 files max in initial exploration

### Token Efficiency
- **Count files first:** Informs thoroughness strategy
- **Use list_files for discovery:** Fast, structured output
- **Use file_search for filtering:** Narrow down before reading
- **Read only key files:** Entry points, base classes, representative examples
- **Avoid reading test files initially:** Unless specifically investigating tests
- **Leverage ripgrep's speed:** file_search is extremely fast, use liberally

### When to Stop Exploring
- **Sufficient context gathered:** Can answer user's question
- **Diminishing returns:** Reading more files not adding new patterns
- **Token budget concerns:** Already used 20+ tool calls
- **User has specific task:** Switch from exploration to implementation

### Anti-Patterns to Avoid
- ❌ **Reading every file:** Wastes tokens, unnecessary for understanding
- ❌ **No structure analysis:** Jumping straight to code without understanding organization
- ❌ **Ignoring documentation:** README and CLAUDE.md often have critical context
- ❌ **Not using file_search:** Reading files to find simple patterns when search would suffice
- ❌ **Over-focusing on one area:** Missing architectural patterns visible in structure

## Integration with Task Tracking

When exploring complex codebases, use task tracking to organize findings:

**First, add all exploration phases as tasks:**
```python
task_add(title="Phase 1: Analyze directory structure and file counts")
task_add(title="Phase 2: Read README and entry points")
task_add(title="Phase 3: Map core symbols and abstractions")
task_add(title="Phase 4: Generate exploration report")
```

**Then update as you progress (one per turn):**
```python
task_update(task_id=1, status="in_progress")
```

**After completing a phase:**
```python
task_complete(task_id=1)
```

## Quick Reference

**For small codebases (<100 files):**
→ Read README + main files + generate simple overview

**For medium codebases (100-1000 files):**
→ Structure analysis + symbol mapping + selective deep dives + structured report

**For large codebases (>1000 files):**
→ Focused exploration only + file_search-heavy approach + scope-limited report

**When user asks "how is X implemented":**
→ file_search for keywords + read_file relevant files + trace execution

**When user asks "give me an overview":**
→ list_files structure + read_file entry points + architecture patterns + report

**When stuck or unclear:**
→ Ask user to narrow scope or specify area of interest

**Core Tools:**
- `list_files(path, pattern)` - Find files by name pattern
- `file_search(pattern, path)` - Search code with ripgrep (fast!)
- `read_file(path)` - Read file contents
- `run(command)` - Execute shell commands
- `file_exists(path)` - Check file existence

---

*This skill combines proven approaches from Claude Code's exploration workflow, Aider's repository mapping, and academic research on code navigation. Apply adaptively based on codebase characteristics and user needs.*
