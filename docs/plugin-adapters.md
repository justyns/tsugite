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

## UI surfaces

A page comes from a plugin's `tsugite.plugins` module, not from its adapter, so a plugin needs an
adapter only when it also wants routes, job executors, or a lifecycle. Each surface becomes a tab
kind in the multiplexer, openable from the command palette; `nav=True` also adds a nav-rail entry,
and `mode` decides what clicking it does. The plugin ships HTML and the host frames it.

```python
from tsugite.ui_surfaces import register_ui_surface

register_ui_surface(
    kind="panel",           # namespaced to plugin/<plugin_name>/panel
    label="Example panel",  # tab + nav-rail label
    icon="plug",            # a host icon name; an unknown one falls back to plug
    entry="ui/panel.html",  # resolved under /api/plugins/<plugin_name>/
    assets=Path(__file__).parent / "ui",  # served at /api/plugins/<plugin_name>/ui/
    nav=True,
    mode="workspace",       # what its nav-rail row does; "full" is the default
    params=["path"],        # the only tab params forwarded into the iframe URL
    events=["my_thing_update"],  # the only daemon events forwarded into the frame
)
```

A page that is generated rather than static needs no `assets` directory. Decorate the
function that returns it, and the daemon serves it as the surface's `entry`:

```python
from tsugite.ui_surfaces import ui_surface

@ui_surface(kind="dash", label="Homelab", nav=True)
def dashboard_page() -> str:
    return """<!doctype html><meta charset=utf-8><title>Homelab</title><h1>ok</h1>
<script>
  addEventListener('message', (e) => {
    if (e.data?.type === 'tsugite:init') parent.postMessage({ type: 'tsugite:ready' }, location.origin);
  });
</script>"""
```

A page function must complete the `tsugite:ready` handshake below, the same as a static `entry`.

A page is served **unauthenticated**, because the browser loads a surface as a navigation and a
navigation carries no bearer header. Return an HTML shell and fetch anything private from the
adapter's `get_http_routes()`.

`kind` is required, and so is one of `entry` or `page`; a descriptor missing either is logged and
skipped. A plugin whose import fails registers nothing. Surfaces are skipped entirely when HTTP is
disabled.

One `assets` directory per plugin; a second, different one is logged and ignored, and a path that
isn't a directory is reported at startup.

### The bridge

The host posts `tsugite:init` once the frame loads and waits for `tsugite:ready`; a surface that
never answers gets an error state with a Reload button after 10 seconds. Both `init` and
`tsugite:theme` (fired on a theme switch) carry the resolved values of every design token, so a
page can skin itself across all five themes without importing anything from the host.

| Direction | Message | Payload |
|---|---|---|
| host → plugin | `tsugite:init` | `{version, surface: {kind, params}, theme: {name, tokens}, token, user}` |
| host → plugin | `tsugite:theme` | `{theme: {name, tokens}}` |
| host → plugin | `tsugite:event` | `{event: {type, data}}` - one daemon broadcast the surface declared |
| plugin → host | `tsugite:ready` | completes the handshake |
| plugin → host | `tsugite:title` | `{title}` - renames the docked tab |
| plugin → host | `tsugite:focus` | makes the surface's pane the focused one |

```html
<script>
  function applyTheme(theme) {
    for (const [name, value] of Object.entries(theme.tokens)) {
      document.documentElement.style.setProperty(name, value);
    }
  }

  addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'tsugite:init') {
      applyTheme(msg.theme);
      parent.postMessage({ type: 'tsugite:ready' }, location.origin);
      parent.postMessage({ type: 'tsugite:title', title: 'My panel' }, location.origin);
    } else if (msg.type === 'tsugite:theme') {
      applyTheme(msg.theme);
    }
  });
</script>
```

Reply to `location.origin` rather than `'*'`: the frame is same-origin with the host, and a wildcard
reply goes to whatever page framed the surface.

`init.token` is the daemon bearer token, for calling your own `get_http_routes()` endpoints:

```js
fetch('/api/plugins/my_plugin/status', { headers: { Authorization: 'Bearer ' + msg.token } });
```

Read it from `init` rather than from browser storage; the host owns where the token comes from.

`init.user` is the id of the human viewing the surface, so a page can tell their actions apart from
an agent's.

`tsugite:event` carries the daemon broadcasts named in the descriptor's `events` list, and only those:

```js
if (msg.type === 'tsugite:event' && msg.event.type === 'my_thing_update') {
  refresh(msg.event.data);
}
```

Read the shell's stream over the bridge rather than opening your own `/api/events`, which behind a
reverse proxy leaves a second long-lived request to the origin pending forever. A surface that
declares no `events` is sent none.

`mode` decides what a nav-rail click does: `full` (the default) hands the surface the whole workspace
region, `workspace` docks it as a tab beside the surfaces already open there. An unknown value warns
and falls back to `full`.

Send `tsugite:focus` on `pointerdown` and `focusin`, so a click inside the surface moves workspace
pane focus with it. A browser delivers neither event outside the frame, so the host cannot see the
click on its own:

```js
const claimFocus = () => parent.postMessage({ type: 'tsugite:focus' }, location.origin);
addEventListener('pointerdown', claimFocus);
addEventListener('focusin', claimFocus);
```

A surface framing something it does not own needs nothing extra: the host reads focus arriving at the
surface's own iframe as the same claim.

`examples/tsugite-example-plugin/` ships a working surface: the adapter declaration and the page
that answers this handshake.

Style the page with those tokens (`--bg0..4`, `--tx0..3`, `--bd0..2`, `--acc`, `--st-*`, `--r-*`,
`--sp-*`, `--fs-*`, `--font-ui`/`--font-mono`, `--t-1..3`, `--ease`) rather than hardcoded colors,
which would not follow a theme switch.

### Threat model

A surface's assets are served without auth, the same as the daemon's own web bundle (`/`,
`/sw.js`, `/static/*`, none of which require a token either). Serve only the UI shell from
`assets`, and read anything worth protecting from your authenticated routes.

The frame is same-origin under `/api/plugins/<plugin_name>/`, sandboxed to
`allow-scripts allow-forms allow-same-origin`. Sharing the origin is what lets a page reach its own
authenticated routes with the token `init` hands it, so the sandbox attributes are hardening rather
than the trust boundary. A plugin surface is code the operator installed, gated by the same
`plugins.<name>.enabled` flag as the rest of the adapter; a hostile adapter already has
Python-level access to the daemon. Anything a surface embeds from a third-party origin should be
loaded from inside the plugin's own page, so the daemon only ever frames plugin-owned HTML.

A tab whose plugin is later disabled or uninstalled survives a reload and renders a "plugin isn't
installed" placeholder, so an arranged layout is never silently dropped.

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
