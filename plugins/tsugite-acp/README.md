# tsugite-acp

Tsugite provider plugin that routes completions through an ACP-compatible agent
subprocess. Defaults to [claude-agent-acp](https://github.com/agentclientprotocol/claude-agent-acp),
the official Claude Agent SDK adapter for the Agent Client Protocol.

## Install

The plugin is a UV workspace member of the tsugite repo, so a top-level

```bash
uv sync --all-extras
```

picks it up. Standalone install (e.g. into another tsugite venv) works too:

```bash
uv add tsugite-acp
```

The plugin requires Node.js + `npx` on PATH at runtime so that the default
agent (`@agentclientprotocol/claude-agent-acp`) can be fetched on first use.

## Usage

Pick the provider via the model string:

```bash
uv run tsu run examples/simple_variable_injection.md "say hi" --model acp:sonnet
```

Aliases: `acp:opus`, `acp:sonnet`, `acp:haiku`. Full IDs (`acp:claude-sonnet-4-6`,
etc.) work too.

## Authentication

Auth is delegated to the Claude Agent SDK that powers `claude-agent-acp`:

- Set `ANTHROPIC_API_KEY` for direct API access.
- Or run `claude` once to log in with a Claude Max subscription; the SDK
  will pick up that token from `~/.claude/`.

## Configuration

Override the agent command by setting `TSUGITE_ACP_COMMAND` (shlex-split):

```bash
TSUGITE_ACP_COMMAND="node /opt/my-acp/dist/index.js" \
  uv run tsu run agent.md "task" --model acp:sonnet
```

If unset, the default is `npx -y @agentclientprotocol/claude-agent-acp`.

### Permission policy

The agent emits `session/request_permission` when it wants to run a tool. By
default the plugin auto-allows every request (parity with how the existing
`claude_code` provider runs). The `PermissionPolicy` class in
`tsugite_acp/policy.py` supports allow/deny rules with fnmatch globs (e.g.
`Bash(git *)`, deny wins over allow), but it is not yet wired up to a
user-facing config file.

## Capabilities

For v1 the client advertises `fs.read_text_file`, `fs.write_text_file`, and
`terminal` as **unsupported**. The agent runs its built-in Read/Write/Edit/Bash
tools directly against the working directory rather than asking tsugite to
proxy them. This matches how `claude_code` already operates.

Bridging tsugite tools through ACP's `mcpServers` slot, edit-review surfacing,
and image/audio prompt support are tracked as out-of-scope for v1.

## Tests

Unit tests:

```bash
uv run pytest plugins/tsugite-acp/tests/
```

Live integration smoke (requires `npx`, an Anthropic credential, and explicit
opt-in):

```bash
TSUGITE_ACP_INTEGRATION=1 uv run pytest plugins/tsugite-acp/tests/integration/
```
