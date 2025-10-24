# tsugite

Tsugite (Japanese: 継手, meaning "joint" in woodworking) is a developer-facing agentic CLI inspired by Claude Code, built on smolagents.

Define AI agents as markdown files with YAML frontmatter. Chain multiple steps together, pass data between them, and use any LLM (OpenAI, Anthropic, Ollama, etc).

## Quick Start

```bash
# Install
uv sync --dev

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
- **ACP integration** - Connect to Agent Client Protocol services (Claude Code, etc.)
- **MCP integration** - Connect to Model Context Protocol servers
- **Temperature control** - Set per-step model parameters

See `examples/` for working agents and `CLAUDE.md` for AI-generated documentation.