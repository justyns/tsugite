# Adapter plugins

Adapter plugins extend the **daemon** (not the light CLI core). Where a `tsugite.plugins`
plugin adds tools/hooks/subscribers (see [plugins.md](plugins.md)), an adapter plugin plugs
into the running daemon: it can front a chat platform, mount its own HTTP routes, and
register job executors. They load only when the daemon starts.

## Entry point

Adapters register under a separate group, `tsugite.adapters`, resolving to a **factory**:

```toml
[project.entry-points."tsugite.adapters"]
cc_driver = "tsugite_cc_driver.adapter:create_adapter"
```

```python
def create_adapter(*, config, agents_config, session_store, identity_map):
    cfg = CCDriverConfig(**(config or {}))
    return CCDriverAdapter(cfg, session_store=session_store, identity_map=identity_map)
```

`config` is this plugin's `daemon.yaml` block (`plugins.<name>`); the other three kwargs are
daemon runtime handles. Return a `BaseAdapter` instance, or `None` to opt out of loading.

## Config and the enable gate

```yaml
plugins:
  cc_driver:
    enabled: true
    permission_mode: bypassPermissions
```

The `plugins.<name>` dict is passed to the factory as `config`. `enabled` gates loading and
**defaults to `True`**: an installed adapter loads unless you set `enabled: false`. A plugin
that wants to be off by default can also return `None` from its factory when its own config
flag is unset.

## Lifecycle

At daemon start (`gateway.py`), after the HTTP server and jobs orchestrator are up:

1. `load_adapter_plugins()` discovers every `tsugite.adapters` entry point, skips the disabled
   ones, and calls each factory. A factory that raises is logged at `WARNING` and skipped -
   never aborting startup; other adapters are unaffected.
2. For each returned adapter the gateway calls `attach_plugin_http(...)`, then
   `attach_plugin_executors(...)`, then `await adapter.start()`.
3. `adapter.stop()` runs at daemon shutdown.

## HTTP routes

An adapter contributes Starlette routes by overriding either method; both are duck-typed and
mounted under `/api/plugins/<plugin_name>`.

| Method | Auth | Use for |
|---|---|---|
| `get_http_routes()` | daemon bearer token (each route wrapped) | web-UI-style authenticated consumers |
| `get_public_http_routes()` | none - the plugin owns access control | receivers that can't send a bearer token (inbound webhooks, CLI hooks) |

Both default to `[]`. An authed handler runs only after the bearer-token check has passed, so it
never checks auth itself. A public route is expected to gate itself (token-in-path, API-key
header, or nothing).

```python
from starlette.responses import JSONResponse
from starlette.routing import Route


class MyAdapter(BaseAdapter):
    def get_http_routes(self):
        # Mounted at GET /api/plugins/my_plugin/status; wrapped with the daemon token.
        return [Route("/status", self._status, methods=["GET"])]

    async def _status(self, request):
        return JSONResponse({"ok": True})

    def get_public_http_routes(self):
        # Mounted at POST /api/plugins/my_plugin/hook/{token}; the plugin gates it itself.
        return [Route("/hook/{token}", self._hook, methods=["POST"])]
```

When HTTP is disabled but an adapter declares routes, the gateway logs a `WARNING` and skips
them (not a crash). Route lists that raise while being collected are logged and skipped.

## Job executors

An adapter supplies non-agent job executors by returning `{name: executor}` from
`get_job_executors()` (default `{}`). Each name is matched against `Job.executor`; a job created
with `executor="<name>"` runs through that executor instead of starting an agent session.

The executor is duck-typed - no base class:

```python
async def start(self, job, followup: str | None) -> None:
    # Kick off the work. followup is None on the first attempt; on a retry it is the
    # failed-AC / hint guidance the agent path would have re-spawned with - feed it into
    # the live session. Report the outcome via the orchestrator's complete_worker / fail_worker.

async def cancel(self, job) -> None:
    # Tear down the child (e.g. kill a PTY). Called on a clean finalize (done/cancelled)
    # BEFORE the worktree is pruned, since the child holds the cwd open. Best-effort.
```

Completion routes back through the existing jobs machinery (predicate + LLM acceptance-criteria
verification, retry loop, stuck/mark-done/cancel, web tile). To reach it, the executor calls
`orchestrator.complete_worker(job_id, summary)` or `orchestrator.fail_worker(job_id, error)`. The
gateway hands the orchestrator to the adapter via `set_jobs_orchestrator(orchestrator)` (if the
adapter defines it) at registration time, so the executor and any hook route can report outcomes.

When the jobs orchestrator is disabled but an adapter registers executors, the gateway logs a
`WARNING` and skips them.

## Example

`tsugite-cc-driver` is the first real adapter plugin: one public hook route plus one `"cc"` job
executor that drives an interactive Claude Code session in a PTY. See
[../plugins/tsugite-cc-driver/README.md](../plugins/tsugite-cc-driver/README.md).
