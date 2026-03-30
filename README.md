# tsugite

Tsugite (継ぎ手) is an agent framework where you define AI agents as markdown files and run them from the CLI, a web UI, or through scheduled tasks.

I built it because none of the existing agent frameworks did what I wanted.  I needed something self-hosted, model-agnostic, and simple enough that an agent is just a text file I can edit and version control.

Originally it was meant to be a framework for micro-agents inspired by [ESA](https://github.com/meain/esa), but has grown a lot since that goal.

## What an agent looks like

A simple "hello world" agent looks like:

```markdown
---
name: morning-brief
model: anthropic:claude-sonnet-4-20250514
tools: [web_search, fetch_text, write_file, final_answer]
---

You are a morning briefing assistant.

Current date: {{ now() }}
User location: {{ env("LOCATION", "unknown") }}

Check the weather, scan top news, and write a short briefing.
Use final_answer() to return the result.
```

YAML frontmatter for config, markdown body for instructions, Jinja for dynamic context.  Run it with:

```bash
tsu run +morning-brief "what's happening today"
```

## Key ideas

- **Agents are markdown.** A basic agent is just markdown with yaml frontmatter.  Advanced agents are still just markdown but can use jinja templating and a special `<!--tsu -->` syntax.
- **Code execution over tool-calling.** Inspired by [smolagents](https://github.com/huggingface/smolagents).  Instead of using native tool calling, LLMs write python code.  Tools are exposed as python functions.
- **Any LLM.** Plugin interface to add support for additional LLM providers.  Built-in we have openai-compatible apis, ollama, anthropic, and claude code.
- **Workspaces.** Each workspace is a persistent directory with agents, skills, memory files, and config.  The agent runs inside its workspace and can read/write files, spawn sub-agents, manage schedules, and persist state across conversations.  Workspaces are entirely optional.
- **CLI and/or Daemon with a Web UI** Use `tsu run` commands for cli-only, or run `tsu daemon` for a daemon that supports scheduled tasks, a web ui, and some other neat things.

## Install

```bash
uv tool install tsugite-cli    # recommended
pipx install tsugite-cli       # alternative
pip install tsugite-cli        # or plain pip
```

The package is `tsugite-cli`, the command is `tsugite` (or `tsu` for short).

## Quick start

```bash
# Initialize a workspace
tsu init my-workspace
cd my-workspace

# Run the built-in default agent
tsu run +default "summarize the files in this directory"

# Run an agent file directly
tsu run my-agent.md "do the thing"

# Start the web UI
tsu serve
```

## Features

- **Multi-step workflows** with `<!-- tsu:step -->` to chain steps and pass data between them
- **Scheduling** built-in cron for recurring agent tasks (daily summaries, monitoring, etc.)
- **Web UI** for conversations, with Discord as an alternative interface
- **Sub-agents** that can spawn other agents for specific subtasks
- **Skills** reusable knowledge files agents can load on demand, mostly compatible with [agentskills.io](https://agentskills.io/)
- **Hooks** that fire shell commands on lifecycle events (post-tool, pre-message, pre/post-compact)
- **Sandbox** (linux only) via bubblewrap with filesystem and network isolation
- **KV store** for persistent agent state
- **MCP** integration for connecting to MCP servers

## Agents in more detail

Agents support YAML frontmatter for configuration:

```yaml
---
name: code-reviewer
model: anthropic:claude-sonnet-4-20250514
max_turns: 15
tools: [read_file, list_files, web_search, final_answer]
auto_load_skills: [coding-standards]
---
```

You can restrict which tools an agent has access to, set turn limits, auto-load skills, attach context files, and extend other agents.  TODO: See `docs/` for the full spec.

Multi-step agents use `<!--tsu -->` comments as directives:

```markdown
<!-- tsu:step name="research" model="openai:gpt-4o" -->
Research the topic and save findings to a variable.

<!-- tsu:step name="write" -->
Using the research from the previous step, write a summary.
The variable `research` is available as a Python variable.
```

For a complete example, check the built-in [default agent](tsugite/builtin_agents/default.md).

## Sandbox

On Linux only (for now), agent code runs inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox when you pass `--sandbox`:

```bash
tsu run +default "task" --sandbox --allow-domain "github.com"
tsu run +default "task" --sandbox --no-network
```

Filesystem access is limited to the workspace.  Network goes through a filtering proxy that only allows domains you specify.

## Config and Data Directories

All paths follow [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) conventions and can be overridden with the standard environment variables.

| Path                                   | Default                              | Contents                                     |
|----------------------------------------|--------------------------------------|----------------------------------------------|
| `$XDG_CONFIG_HOME/tsugite/`            | `~/.config/tsugite/`                 | `config.json`, `mcp.json`, `daemon.yaml`     |
| `$XDG_DATA_HOME/tsugite/history/`      | `~/.local/share/tsugite/history/`    | Session history (JSONL per session)          |
| `$XDG_DATA_HOME/tsugite/daemon/`       | `~/.local/share/tsugite/daemon/`     | Daemon state                                 |
| `$XDG_DATA_HOME/tsugite/secrets/`      | `~/.local/share/tsugite/secrets/`    | Encrypted secrets (`secrets.db`)             |
| `$XDG_DATA_HOME/tsugite/kvstore/`      | `~/.local/share/tsugite/kvstore/`    | Key-value store (`kv.db`)                    |
| `$XDG_DATA_HOME/tsugite/usage/`        | `~/.local/share/tsugite/usage/`      | Usage (cost and token) tracking (`usage.db`) |
| `$XDG_DATA_HOME/tsugite/workspaces/`   | `~/.local/share/tsugite/workspaces/` | Workspace directories                        |
| `$XDG_CACHE_HOME/tsugite/attachments/` | `~/.cache/tsugite/attachments/`      | Attachment cache                             |


## Development

```bash
git clone https://github.com/justyns/tsugite.git
cd tsugite
uv sync --dev
```

## Status

This is a personal project I use daily.  It works for my use cases but isn't polished for general consumption yet.  Issues and PRs welcome, but set expectations accordingly.  Documentation is very sparse because I keep changing things.