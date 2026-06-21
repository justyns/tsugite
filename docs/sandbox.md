# Sandbox

On Linux, agent code can run inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox: the filesystem is limited to the workspace, network goes through a filtering proxy that only allows the domains you list, and the environment is cleared down to a safe set so agent code can't read the host's API keys or tokens from `os.environ` (use `pass_env` to expose specific non-secret vars).

It needs `bwrap` on the host and does nothing on other platforms.

**Note**: Consider sandboxing experimental for now.  It may change in the future, but you could also achieve something similar with containers, vms, or [srt](https://github.com/anthropic-experimental/sandbox-runtime).

## CLI

Pass `--sandbox` to `tsu run`:

```bash
tsu run +default "task" --sandbox --allow-domain "github.com"
tsu run +default "task" --sandbox --no-network
tsu run +default "task" --sandbox --pass-env MY_TOOL_CONFIG   # expose a specific env var
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
  # pass_env: ["MY_TOOL_CONFIG"]      # extra env var NAMES to expose (values read from the
                                      # environment). The sandbox otherwise clears the env so
                                      # agent code can't read host secrets; reach secrets via
                                      # the get_secret tool, not pass_env.

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

- per-turn python code exec
- tools (except scheduling tools, see below)
- spawned agents and jobs
- pty sessions

The scheduling tools (`schedule_create`, `background_task`, `schedule_run`, and friends) are refused while sandboxed, because they set up work that runs later, outside the sandboxed turn.  Run those agents unsandboxed if they need to schedule.
