# Reusing tsugite from external coding agents

tsugite agents and skills can be reused from another coding agent (Claude Code, Cursor, Aider, anything that runs shell commands).  `tsu exec` runs a Python snippet with tsugite's tools (`read_file`, `http_request`, `get_secret`, ...) in scope, so a skill's code runs the same as it would inside a tsugite agent.

## tsu exec

```bash
echo 'read_file(path="README.md")' | tsu exec -        # from stdin
tsu exec snippet.py --tools @fs,@http                  # from a file
```

- `--tools` picks which tools are available (`@fs`, `@http`, `@secrets`, or names); repeat the flag or comma-separate for several.  Default: `@fs`, `@http`, `@secrets`.
- `--agent +name` inherits that agent's `tools` and `allowed_secrets`.
- Call tools with keyword args: `read_file(path="x")`, `get_secret(name="gh-token")`.  Run `tsu tools show <name>` for a tool's exact parameters.
- `open()` is blocked; use `read_file` / `write_file`.  A trailing expression prints, like `python -c`.

### Secrets

Values are masked (`***`) in output and gated by an allowlist:

```bash
tsu exec snippet.py --allow-secret gh-token   # only gh-token
tsu exec snippet.py --agent +deploy           # use the agent's allowed_secrets
```

No allowlist means all secrets are allowed (still masked).  See [secrets.md](secrets.md).

### Sandbox

```bash
tsu exec snippet.py --no-network                       # no network (implies --sandbox)
tsu exec snippet.py --sandbox --allow-domain api.github.com
```

Same flags as `tsu run`; see [sandbox.md](sandbox.md).  `tsu exec` also turns on the sandbox when you pass `--no-network` or `--allow-domain`.

## Running a whole agent

```bash
tsu agents list
tsu run +<agent> "task"
```

## Surfacing tsugite skills to your agent

Each agent finds skills its own way.  Claude Code reads them from `.claude/skills/`.  Link tsugite's builtin skills in there (works installed or from a checkout):

```bash
SKILLS=$(python -c 'from tsugite.skill_discovery import get_builtin_skills_path as p; print(p())')
for d in "$SKILLS"/*/; do n=$(basename "$d"); ln -sfn "$d" ".claude/skills/$n"; done
```

tsugite also discovers skills from project (`.agents/skills/`, `skills/`) and global roots.  `.claude/` is usually gitignored, so the links stay in your working copy.

## The bridge skill

tsugite ships an `external-agent-bridge` skill that tells the agent to run skill Python through `tsu exec`.  The loop above links it in with the rest; to grab just that one:

```bash
cp -r "$SKILLS/external-agent-bridge" .claude/skills/
```

Point your agent's instructions file (`AGENTS.md` / `CLAUDE.md`) at it so it stays loaded.
