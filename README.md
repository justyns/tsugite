# tsugite

Tsugite (Japanese: 継ぎ手, the art of joinery in woodworking) is a developer-facing agentic CLI.

Define AI agents as markdown files with YAML frontmatter. Chain multiple steps together, pass data between them, and use any LLM (OpenAI, Anthropic, Ollama, etc).

## Installation

```bash
# Recommended: Install with uv
uv tool install tsugite-cli

# Alternative: Install with pipx
pipx install tsugite-cli

# Or with pip
pip install tsugite-cli
```

**Note:** The package name is `tsugite-cli`, but the command is `tsugite` (or `tsu` for short).

## Quick Start

```bash
# Run an agent
tsugite run examples/simple_variable_injection.md "test it"

# Create your own agent
cat > my_agent.md << 'EOF'
---
name: hello
model: openai:gpt-4o-mini
---

Task: {{ user_prompt }}

Just say hello and use final_answer() to return your greeting.
EOF

tsugite run my_agent.md "greet the user"
```

## Features

- **Multi-step workflows** - Chain steps with `<!-- tsu:step -->`, pass data between them
- **Variable injection** - Step outputs automatically available as Python variables
- **Multiple LLM providers** - OpenAI, Anthropic, Ollama, Google, GitHub Copilot
- **MCP integration** - Connect to Model Context Protocol servers
- **Temperature control** - Set per-step model parameters
- **Copy-paste friendly output** - `--plain` flag or auto-detection for pipe/redirect

## CLI Options

```bash
# Plain output (no box-drawing characters, copy-paste friendly)
tsugite run +assistant "task" --plain

# Auto-detection: plain mode activates when piped or NO_COLOR is set
tsugite run +assistant "task" | grep result

# Headless mode for scripts (result to stdout, progress to stderr)
tsugite run +assistant "task" --headless

# Continue latest conversation (auto-detects agent)
tsugite run --continue "follow-up prompt"

# Continue specific conversation
tsugite run --continue --conversation-id CONV_ID "follow-up prompt"
tsugite chat --continue CONV_ID

# View conversation history
tsugite history list
tsugite history show CONV_ID
```

## Development

```bash
# Clone and install for development
git clone https://github.com/justyns/tsugite.git
cd tsugite
uv sync --dev
```

See `examples/` for working agents and `CLAUDE.md` for AI-generated documentation.