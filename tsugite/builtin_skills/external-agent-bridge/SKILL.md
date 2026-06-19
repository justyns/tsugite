---
name: external-agent-bridge
description: How an external coding agent (Claude Code, Cursor, Aider, etc.) reuses tsugite agents and skills. Read before using any tsugite skill; their code runs in tsugite's tool namespace (read_file, http_request, get_secret) and must be run via `tsu exec`, not directly in a shell.
---

# Reusing tsugite from an external coding agent

tsugite agents and skills run Python in tsugite's tool namespace: functions like
`read_file`, `http_request`, and `get_secret` are already in scope, no imports. To run that
code, use `tsu exec` (not a plain shell or REPL).

## Using a tsugite skill

A skill is a directory with a `SKILL.md` (frontmatter + instructions) and optional
`scripts/`. To use one:

1. Read its `SKILL.md` to see what it does and which tsugite tools its code calls. The
   frontmatter may list `allowed-tools`.
2. Look up the exact signatures before writing code, so you pass the right keyword args:
   ```bash
   tsu tools list                # all tools, grouped by category
   tsu tools show read_file      # params for one tool, e.g. read_file(path=...)
   ```
3. Run the code with the tools it needs:
   ```bash
   tsu exec snippet.py --tools @fs,@http
   echo 'print(read_file(path="README.md")[:200])' | tsu exec -
   ```
   If the skill ships a script, run it directly:
   `tsu exec .claude/skills/<name>/scripts/foo.py --tools @fs`.

When writing the code:
- Call tools with keyword args: `read_file(path="x")`, not `read_file("x")`. Get the
  param names from `tsu tools show <name>`.
- `open()` is blocked; use `read_file` / `write_file`.
- A trailing expression is printed, like `python -c`.

## Tools

- `--tools` selects what is available: categories (`@fs`, `@http`, `@secrets`, `@shell`,
  `@time`, ...) or bare names. Pass several by repeating the flag or comma-separating
  (`--tools @fs,@http`). Default exposes `@fs`, `@http`, `@secrets`.
- `--agent +name` inherits that agent's exact tools and secret allowlist.
- `tsu tools list` / `tsu tools show <name>` are the source of truth for names and params.

## Secrets

`get_secret(name="...")` returns the value, masked as `***` in output. Allowlist it:

```bash
tsu exec snippet.py --allow-secret gh-token
tsu exec snippet.py --agent +deploy           # use the agent's allowed_secrets
```

No allowlist means all secrets are allowed (still masked).

## Sandbox

```bash
tsu exec snippet.py --no-network                  # no network (implies --sandbox)
tsu exec snippet.py --sandbox --allow-domain api.github.com
```

## Whole agents

If a task matches an existing agent, run it instead of rebuilding it:

```bash
tsu agents list
tsu run +<agent> "task"
```

See `docs/external-agent-integration.md` in the tsugite repo for more.
