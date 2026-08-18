"""Wiring a loaded plugin into the running daemon: its HTTP routes, its web UI
surfaces, and its job executors."""

import asyncio
import inspect
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _collect_plugin_ui(plugin_name: str, declared: list) -> tuple[list[dict], Path | None]:
    """Split a plugin's declared UI surfaces into the web-UI payload and the
    assets directory to mount, deciding once what a valid descriptor is.

    `kind` becomes `plugin/<plugin_name>/<kind>`, the one identifier the web UI
    uses as the mux tab kind, the nav view id, and the hash route, and `entry`
    resolves under the plugin's own `/api/plugins/<plugin_name>/` mount. A
    descriptor missing either is dropped with a warning rather than reaching the
    UI as an unopenable tab, and its `assets` is dropped with it.

    `mode` decides what a nav-rail click does: `full` (the default) hands the
    surface the whole workspace region, `workspace` docks it as a tab beside the
    surfaces already open there.

    `events` names the broadcast types the web UI forwards into the surface's
    frame over its bridge. The browser holds one `/api/events` stream for the
    whole origin, so a surface names what it wants rather than opening a second
    one, and a surface that names nothing is never shown the daemon's traffic.

    `assets` is served at `/api/plugins/<plugin_name>/ui/`, so an entry of
    `ui/panel.html` resolves to `panel.html` inside it. One directory per
    plugin; a second, different one is a mistake worth naming. A path that is
    not a directory is reported here instead of at StaticFiles construction,
    which would abort daemon startup over one plugin.
    """
    surfaces = []
    dirs = []
    for item in declared:
        kind = item.get("kind")
        entry = item.get("entry") or (f"page/{kind}" if item.get("page") else None)
        if not kind or not entry:
            logger.warning(
                "Plugin '%s': UI surface %r needs 'kind' and one of 'entry' or 'page'; skipping it", plugin_name, item
            )
            continue
        mode = item.get("mode", "full")
        if mode not in ("full", "workspace"):
            logger.warning(
                "Plugin '%s': UI surface '%s' declares unknown mode %r; using 'full'", plugin_name, kind, mode
            )
            mode = "full"
        surfaces.append(
            {
                "plugin": plugin_name,
                "kind": f"plugin/{plugin_name}/{kind}",
                "label": item.get("label") or kind,
                # Only the web UI knows which icon names it ships, so it owns the fallback.
                "icon": item.get("icon", ""),
                "entry": f"/api/plugins/{plugin_name}/{entry.lstrip('/')}",
                "nav": bool(item.get("nav")),
                "params": list(item.get("params") or []),
                "events": list(item.get("events") or []),
                "mode": mode,
            }
        )
        if (assets := item.get("assets")) and Path(assets) not in dirs:
            dirs.append(Path(assets))
    if not dirs:
        return surfaces, None
    if len(dirs) > 1:
        logger.warning("Plugin '%s' declares %d UI asset dirs; serving only %s", plugin_name, len(dirs), dirs[0])
    if not dirs[0].is_dir():
        logger.warning("Plugin '%s': UI assets path %s is not a directory; its surfaces will 404", plugin_name, dirs[0])
        return surfaces, None
    return surfaces, dirs[0]


def _page_endpoint(plugin_name: str, kind: str, page):
    from starlette.responses import HTMLResponse

    is_async = inspect.iscoroutinefunction(page)

    async def endpoint(request):
        try:
            html = await page() if is_async else await asyncio.to_thread(page)
            # Generated per request, so nothing here is safe to cache.
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})
        except Exception:
            logger.warning("Plugin '%s': page for UI surface '%s' raised", plugin_name, kind, exc_info=True)
            return HTMLResponse("<!doctype html><title>Plugin page failed</title>", status_code=500)

    return endpoint


def _page_routes(plugin_name: str, declared: list) -> list:
    """Serve each `page` callable as a public route, which `_collect_plugin_ui`
    points the surface's `entry` at.

    Public because the browser loads a surface as a navigation, which carries no
    bearer header. The `/page/` prefix keeps these clear of the `/ui/` assets mount,
    which matches on prefix.
    """
    from starlette.routing import Route

    return [
        Route(f"/page/{item['kind']}", _page_endpoint(plugin_name, item["kind"], item["page"]), methods=["GET"])
        for item in declared
        if item.get("page") and item.get("kind")
    ]


def attach_plugin_http(http_server, plugin_name: str, adapter, declared_surfaces: list | None = None) -> None:
    """Wire a plugin's HTTP surface into the daemon's Starlette app.

    Sets the shared SSE bus on the adapter (so it can broadcast events), then
    mounts its `get_http_routes()` (auth-wrapped) and `get_public_http_routes()`
    (no auth) under `/api/plugins/<plugin_name>`, and registers `declared_surfaces`
    for the web UI. A plugin that lacks the route methods is skipped, and one that
    raises while producing them is logged and skipped.

    Surfaces arrive as an argument, so a plugin that only registered a page needs no
    adapter: pass `adapter=None`. Routes and surfaces mount together because they share
    one Starlette Mount, which matches on prefix.
    """
    if http_server is not None and adapter is not None:
        try:
            adapter.event_bus = http_server.event_bus
        except Exception:  # noqa: BLE001 -- a read-only/exotic adapter shouldn't abort startup
            logger.debug("Could not set event_bus on plugin '%s'", plugin_name)

    def _collect(method_name: str) -> list:
        method = getattr(adapter, method_name, None)
        if method is None:
            return []
        try:
            return list(method() or [])
        except Exception:
            logger.warning("Plugin '%s' %s() raised; skipping those entries", plugin_name, method_name, exc_info=True)
            return []

    declared = declared_surfaces or []
    authed = _collect("get_http_routes")
    public = [*_collect("get_public_http_routes"), *_page_routes(plugin_name, declared)]
    surfaces, assets = _collect_plugin_ui(plugin_name, declared)
    if not authed and not public and not surfaces:
        return
    if http_server is None:
        logger.warning("Plugin '%s' registers HTTP routes or UI surfaces but HTTP is disabled; skipping", plugin_name)
        return
    if assets:
        from starlette.routing import Mount

        from tsugite_daemon.adapters.http.helpers import _NoCacheStaticFiles

        public = [*public, Mount("/ui", app=_NoCacheStaticFiles(directory=str(assets)))]
    http_server.mount_plugin_routes(plugin_name, authed, public)
    http_server.plugin_ui_surfaces.extend(surfaces)


def attach_plugin_executors(jobs_orchestrator, plugin_name: str, adapter) -> None:
    """Register a loaded adapter plugin's job executors on the orchestrator.

    Reads `get_job_executors() -> dict[str, executor]` so a plugin (e.g.
    cc-driver) can supply a non-agent executor. No-op when the plugin exposes no
    executors or the orchestrator is disabled; a plugin that raises while
    producing its executors is logged and skipped.
    """
    method = getattr(adapter, "get_job_executors", None)
    if method is None:
        return
    try:
        executors = dict(method() or {})
    except Exception:
        logger.warning("Plugin '%s' get_job_executors() raised; skipping", plugin_name, exc_info=True)
        return
    if not executors:
        return
    if jobs_orchestrator is None:
        logger.warning(
            "Plugin '%s' registers job executors but the jobs orchestrator is disabled; skipping", plugin_name
        )
        return
    # Hand the orchestrator back to the adapter so its executors can report
    # completion/failure via complete_worker/fail_worker.
    setter = getattr(adapter, "set_jobs_orchestrator", None)
    if setter is not None:
        try:
            setter(jobs_orchestrator)
        except Exception:
            logger.warning("Plugin '%s' set_jobs_orchestrator() raised", plugin_name, exc_info=True)
    for name, executor in executors.items():
        jobs_orchestrator.register_executor(name, executor)
