# tsugite-cc-driver

Run an interactive **Claude Code** (`claude`) session as a verified tsugite **job**.

A job created with `executor="cc"` is driven by a real `claude` process in a PTY.
The plugin registers Claude Code HTTP `Stop` hooks; on each stop it either lets the
attempt finish (a completion marker is present, or the continue budget is spent),
pauses for input (see below), or injects a natural-language nudge to keep going.
When an attempt finishes, completion routes into tsugite's existing job machinery:
predicate + LLM acceptance-criteria verification, the retry loop (the failed-AC
followup is typed into the **same live session**), stuck/mark-done/cancel, and the
web job tile with the embedded terminal.

## Worker-initiated needs-input

If the driven claude is blocked on information it cannot obtain itself, it ends its
reply with `CCDRIVER_NEED_INPUT: <question>` (the protocol is baked into its initial
prompt). The job then parks in the non-terminal `awaiting_input` state - the PTY
stays alive, no verification attempt is consumed, and the phase timer keeps running
so an unanswered question still times out to `stuck`. The spawning session's agent
is woken with the question and answers via `respond_to_job(job_id, answer)`, which
resumes the attempt (typed into the live session, or respawned with `--resume` if
the PTY died while paused). A human can instead type the answer directly into the
TUI - the next Stop resumes and grades the turn normally.

## When NOT to use it

For headless, non-interactive automation use the `tsugite-claude-code` provider
instead - it drives Claude Code through the SDK without a PTY. cc-driver is for
jobs you want to watch: the TUI is embedded in the job tile and you can take over
by typing at any time.

## Config (`daemon.yaml` -> `plugins.cc_driver`)

| key | default | meaning |
|-----|---------|---------|
| `enabled` | `false` | load the plugin |
| `claude_binary` | `claude` | path/name of the CLI |
| `model` | `sonnet` | default `--model` (alias or full name); a per-job `model` overrides it, `null` uses claude's own default |
| `permission_mode` | `bypassPermissions` | passed to `--permission-mode` |
| `sandbox` | `true` | run claude under bubblewrap (see below) |
| `provision_trust` | `true` | auto-write Claude Code trust for a job's workspace (see below); `false` = require pre-trusted workspaces |
| `max_consecutive_continues` | `5` | Stop nudges before an attempt is handed to the verifier anyway |
| `completion_marker` | `CCDRIVER_GOAL_COMPLETE` | token claude ends its reply with when done |
| `needs_input_marker` | `CCDRIVER_NEED_INPUT` | line claude emits to pause the job for supervisor input |
| `ax_screen_reader` | `false` | opt in to `--ax-screen-reader` (flat text, cleaner PTY capture; off keeps the normal TUI) |
| `effort` | `null` | default `--effort` (low/medium/high/xhigh/max); per-job `effort` overrides, null uses claude's own default |
| `base_url` | `http://127.0.0.1:8374` | daemon URL the hook receiver is reached at |
| `state_dir` | XDG state | where per-job `settings.json` files are written |

`max_attempts`, `timeout_minutes`, `model`, `effort`, and acceptance criteria are
ordinary per-job fields (`/job ... --executor cc --effort max`, or `spawn_job(...,
executor="cc", effort="max")`).

## Workspace trust

Claude Code shows a blocking "Is this a project you trust?" dialog the first time
it runs in a directory, and **no flag skips it** (`bypassPermissions`,
`--dangerously-skip-permissions`, `--add-dir`, `IS_SANDBOX=1` were all verified not
to). A daemon-spawned session in an untrusted cwd would hang there forever, so
cc-driver checks trust **before** spawning.

By default (`provision_trust: true`) cc-driver provisions trust itself: it writes
`projects["<abs path>"].hasTrustDialogAccepted = true` into the config Claude reads
(`<CLAUDE_CONFIG_DIR>/.claude.json`, else `~/.claude.json`) via an atomic
read-modify-write that preserves every other entry, so no manual trust step is
needed. Trust is checked against the cwd and its ancestors (Claude Code treats a
subdir of a trusted project as trusted); for a `--repo` job cc-driver provisions
the **repo root** rather than the ephemeral worktree, so one entry covers every
future worktree instead of a dead entry per job.

Set `provision_trust: false` to keep cc-driver read-only: a job in an untrusted
workspace fails with instructions, and you trust each workspace once yourself
(run `claude` there and accept the prompt).

## Sandbox caveats

`sandbox: true` runs claude filesystem-isolated to the job workspace with `~/.claude`
bound read-write (credential refresh) and **network ON** - the driven claude needs the
API. This is filesystem isolation only, not egress filtering.

## Restart behaviour

DriveState is in-memory. If the daemon restarts, the PTY dies with it and the existing
orphan-recovery marks the job `errored` (retryable). The terminal log and the CC
transcript survive on disk for forensics.
