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

## Sandbox

Agent code can run inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox with filesystem and network isolation.

```bash
# Sandbox with specific domains allowed
tsu run +default "task" --sandbox --allow-domain "github.com" --allow-domain "*.openai.com"

# Full network isolation
tsu run +default "task" --sandbox --no-network
```

**How it works:**

```
CLI --sandbox --allow-domain "*.github.com"
 └─ SubprocessExecutor writes harness script
     └─ bwrap --unshare-pid --unshare-net --die-with-parent ...
         └─ TCP↔UDS bridge (HTTP_PROXY → Unix socket)
             └─ ConnectProxy filters CONNECT requests by domain:port
```

**Default mounts:**

- Read-only: `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, `/etc/ssl`, `/etc/resolv.conf`, CA certs, Python venv + `sys.path`
- Read-write: workspace directory, internal state directory
- Tmpfs: `/tmp` (fresh each run)

Extra bind mounts can be added programmatically via `SandboxConfig.extra_ro_binds` and `extra_rw_binds`, but these are not yet exposed through CLI flags or agent frontmatter.

**Network:**
- Network is namespace-isolated; outbound HTTP/HTTPS goes through a filtering CONNECT proxy
- Direct connections to bare IP addresses are always blocked
- `--no-network` skips the proxy entirely — no connectivity at all

**`--allow-domain` syntax:**

| Pattern | Allows |
|---|---|
| `github.com` | ports 80, 443 |
| `github.com:22` | port 22 only |
| `*.github.com:8080` | port 8080 on subdomains |
| `*:*` | all domains, all ports |

## Development

```bash
# Clone and install for development
git clone https://github.com/justyns/tsugite.git
cd tsugite
uv sync --dev
```

See `examples/` for working agents and `CLAUDE.md` for AI-generated documentation.