# Sandbox

On Linux, agent code can run inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox: the filesystem is limited to the workspace, and network goes through a filtering proxy that only allows the domains you list.

It needs `bwrap` on the host (it uses user namespaces) and does nothing on other platforms.  It isn't bulletproof - it's bubblewrap plus a domain-filtering proxy, not a VM.

## CLI

Pass `--sandbox` to `tsu run`:

```bash
tsu run +default "task" --sandbox --allow-domain "github.com"
tsu run +default "task" --sandbox --no-network
```

## Daemon

The daemon doesn't sandbox anything by default.  Turn it on in `daemon.yaml`, globally and/or per agent.  A per-agent `sandbox:` block overrides the global one field by field.

```yaml
sandbox:                              # global default for every agent
  enabled: true
  allow_domains: ["github.com", "pypi.org"]
  # no_network: true                  # cut network entirely instead
  # extra_ro_binds: ["~/.config/foo"] # extra read-only mounts
  # extra_rw_binds: []                # extra read-write mounts

agents:
  researcher:
    workspace_dir: ~/work/research
    agent_file: researcher            # uses the global sandbox
  ops:
    workspace_dir: ~/work/ops
    agent_file: ops
    sandbox:
      enabled: false                  # this one runs unsandboxed
```

When an agent is sandboxed, the things it can run stay inside the sandbox:

- Its per-turn Python and the `run()` shell tool.
- Anything it spawns - `spawn_agent`, `start_session`, `spawn_job` (worker and verifier) inherit the sandbox.
- Jobs made with `/job` too, including `cmd:` / `exit_code:` acceptance criteria, which run in the sandbox instead of on the host.
- Terminals, whether opened by the agent (`pty_create`) or for its session via `/run` or the web UI - scoped to the workspace with no network.

The scheduling tools (`schedule_create`, `background_task`, `schedule_run`, and friends) are refused while sandboxed, because they set up work that runs later, outside the sandboxed turn.  Run those agents unsandboxed if they need to schedule.

## The daemon config is the ceiling

Whether an agent is sandboxed is the operator's call in `daemon.yaml`.  An agent can't turn its own sandbox off or widen it: there's no tool for it, its code runs inside the sandbox, and the inherited policy is stored where the agent can't edit it.

An agent can make itself *more* restricted from its frontmatter:

```yaml
sandbox:
  enabled: true                  # opt in even if the daemon didn't
  no_network: true               # drop network
  allow_domains: ["github.com"]  # narrow to fewer domains
```

These only tighten.  `enabled` and `no_network` can flip on but not off, and the domains (here and in `network:`) are capped by the daemon's allowlist - an agent can never reach a domain the daemon didn't allow.

## Notes

- Linux only, and `bwrap` must be installed.  If an agent sets `sandbox.enabled` but `bwrap` is missing, the daemon refuses to start instead of running unsandboxed.
- The Docker image doesn't include bubblewrap, and unprivileged containers usually can't use user namespaces, so this is meant for a bare-metal or privileged host.
