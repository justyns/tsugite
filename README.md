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

# Start the daemon (web UI, Discord/Telegram bots)
tsu daemon
```

### Output modes

`tsu run` keeps its terminal output plain by default so logs are copy-pasteable and behave
in nested tmux / non-Rich-friendly shells. Pick a richer or quieter mode when you need it:

| Mode | When to use |
| --- | --- |
| **Default (plain)** | Everyday interactive runs and piped output (`tsu run ... \| less`). |
| `--ui live` | Long-running interactive runs where a persistent footer showing turn / current tool / tokens / cost / elapsed time is useful. Falls back to plain if stdout is not a TTY or `NO_COLOR` is set. |
| `--headless` | CI/scripts: result on stdout, no progress chrome. Combine with `--verbose` for stderr trace. |
| `--plain` | Force plain explicitly (same as the default; useful when overriding configs/aliases). |

## Features

- **Multi-step workflows** with `<!-- tsu:step -->` to chain steps and pass data between them
- **Scheduling** built-in cron for recurring agent tasks (daily summaries, monitoring, etc.)
- **Web UI** for conversations, with Discord as an alternative interface
- **Sub-agents** that can spawn other agents for specific subtasks
- **Skills** directory-based knowledge modules (mostly) following the [agentskills.io](https://agentskills.io/) SKILL.md format
- **Hooks** that fire shell commands on lifecycle events (post-tool, pre-message, pre/post-compact)
- **Sandbox** (linux only) via bubblewrap with filesystem and network isolation

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

### Daemon sandbox

The daemon does not sandbox by default. Opt in per `daemon.yaml`, globally and/or
per agent (the per-agent block overrides the global field by field):

```yaml
# Global default applied to every agent unless it overrides
sandbox:
  enabled: true
  allow_domains: ["github.com", "pypi.org"]   # via the filtering proxy
  # no_network: true                          # or cut network entirely
  # extra_ro_binds: ["~/.config/some-tool"]   # extra read-only mounts
  # extra_rw_binds: []                         # extra read-write mounts

agents:
  researcher:
    workspace_dir: ~/work/research
    agent_file: researcher
    # inherits the global sandbox above
  trusted-ops:
    workspace_dir: ~/work/ops
    agent_file: ops
    sandbox:
      enabled: false   # opt this one agent back out
```

When an agent runs sandboxed there are **no escape paths**:

- Its per-turn code and shell `run()` execute inside bubblewrap.
- `spawn_agent`, `start_session`, and `spawn_job` propagate the sandbox so child
  agents / sessions / job workers + verifiers stay isolated.
- Jobs honor the agent's sandbox however they're created (the `spawn_job` tool or
  the `/job` command); predicate acceptance criteria (`cmd:`/`exit_code:`) run inside
  bubblewrap (workspace-only, no network), not on the host.
- Terminal commands run inside bubblewrap (workspace-only filesystem, no network),
  whether opened by the agent (`pty_create`) or for the agent's session via `/run`
  or the web UI - the terminal inherits its session's agent sandbox config.

The scheduling tools (`schedule_create`, `background_task`,
`schedule_run`/`update`/`enable`) are **refused** while sandboxed, since they arrange
persistent or detached host execution that outlives the sandboxed turn; run those
agents unsandboxed if they need to schedule.

Notes:
- Linux-only and requires `bwrap` on the host (user-namespace support). If an agent
  has `sandbox.enabled` but `bwrap` is missing, the daemon **refuses to start** (fail
  closed) rather than running its code unsandboxed.
- The shipped Docker image does not include bubblewrap, and unprivileged containers
  usually can't use user namespaces - daemon sandboxing targets a bare-metal /
  privileged host.
- Agent frontmatter `network: {domains: [...]}` is merged into the agent's
  `allow_domains` when sandboxed.

## Config and Data Directories

All paths follow [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) conventions and can be overridden with the standard environment variables.

| Path                                   | Default                              | Contents                                     |
|----------------------------------------|--------------------------------------|----------------------------------------------|
| `$XDG_CONFIG_HOME/tsugite/`            | `~/.config/tsugite/`                 | `config.json`, `daemon.yaml`                 |
| `$XDG_DATA_HOME/tsugite/history/`      | `~/.local/share/tsugite/history/`    | Session history (JSONL per session)          |
| `$XDG_DATA_HOME/tsugite/daemon/`       | `~/.local/share/tsugite/daemon/`     | Daemon state                                 |
| `$XDG_DATA_HOME/tsugite/secrets/`      | `~/.local/share/tsugite/secrets/`    | Encrypted secrets (`secrets.db`)             |
| `$XDG_DATA_HOME/tsugite/usage/`        | `~/.local/share/tsugite/usage/`      | Usage (cost and token) tracking (`usage.db`) |
| `$XDG_DATA_HOME/tsugite/workspaces/`   | `~/.local/share/tsugite/workspaces/` | Workspace directories                        |
| `$XDG_CACHE_HOME/tsugite/attachments/` | `~/.cache/tsugite/attachments/`      | Attachment cache                             |


## Development

```bash
git clone https://github.com/justyns/tsugite.git
cd tsugite
uv sync --all-extras
```

## Status

This is a personal project I use daily.  It works for my use cases but isn't polished for general consumption yet.  Issues and PRs welcome, but set expectations accordingly.  Documentation is very sparse because I keep changing things.