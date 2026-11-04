"""IntrospectionMixin: read-only registry endpoints for the daemon HTTP API.

Thin, additive wrappers that surface facts the core registries already expose:
GET /api/plugins over tsugite.plugins.discover_plugins() and GET /api/tools over
the tsugite.tools registry (the same surface `tsu plugins list` / `tsu tools
list` render). Neither mutates state.
"""

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.auth import SCOPE_PLUGIN_PREFIX

SURFACE_TOKEN_TTL_SECONDS = 24 * 3600


class IntrospectionMixin:
    def _introspection_routes(self) -> list:
        return [
            Route("/api/plugins", self._api_list_plugins, methods=["GET"]),
            Route("/api/tools", self._api_list_tools, methods=["GET"]),
            Route("/api/plugin-surface-token", self._api_plugin_surface_token, methods=["POST"]),
        ]

    async def _api_plugin_surface_token(self, request: Request) -> JSONResponse:
        """Mint the token a plugin surface frame gets in place of the viewer's own:
        valid only under /api/plugins/<plugin>/."""
        if err := self._check_auth(request):
            return err
        try:
            body = await request.json()
        except ValueError:
            body = None
        plugin = body.get("plugin") if isinstance(body, dict) else None
        if not isinstance(plugin, str) or plugin not in {s["plugin"] for s in self.plugin_ui_surfaces}:
            return JSONResponse({"error": "plugin is not a loaded plugin with a UI surface"}, status_code=400)
        token = self._token_store.issue(
            ttl=SURFACE_TOKEN_TTL_SECONDS,
            scope=f"{SCOPE_PLUGIN_PREFIX}{plugin}",
            identity=f"plugin-surface:{plugin}",
        )
        return JSONResponse({"token": token})

    async def _api_list_plugins(self, request: Request) -> JSONResponse:
        """Discovered plugins across every entry-point group, plus the UI surfaces
        the loaded adapter plugins declare. Read-only wrapper over
        discover_plugins(); no enable/disable mutation."""
        if err := self._check_auth(request):
            return err
        from tsugite.link_previews import link_resolver_patterns
        from tsugite.plugins import discover_plugins

        plugin_config = None
        gateway = self.gateway
        if gateway is not None:
            plugin_config = getattr(getattr(gateway, "config", None), "plugins", None)
        # discover_plugins() walks every installed distribution's entry points,
        # which is blocking IO the web UI now pays on every boot for its ui_surfaces.
        discovered = await asyncio.to_thread(discover_plugins, plugin_config)
        return JSONResponse(
            {
                "plugins": [
                    {
                        "name": p.name,
                        "group": p.group,
                        "enabled": p.enabled,
                        "loaded": p.loaded,
                        "error": p.error,
                    }
                    for p in discovered
                ],
                "ui_surfaces": self.plugin_ui_surfaces,
                "link_resolvers": link_resolver_patterns(),
            }
        )

    async def _api_list_tools(self, request: Request) -> JSONResponse:
        """Registered tools with category and origin. Category is the tool's
        explicit @tool(category=...) or its module basename (same rule
        get_tools_by_category uses); source is builtin for core tsugite tools,
        plugin for anything registered from another package."""
        if err := self._check_auth(request):
            return err
        from tsugite.tools import iter_tool_infos

        tools = []
        for tool_info in iter_tool_infos():
            module = tool_info.func.__module__
            tools.append(
                {
                    "name": tool_info.name,
                    "category": tool_info.category or module.split(".")[-1],
                    "description": tool_info.description,
                    "source": "builtin" if module == "tsugite" or module.startswith("tsugite.") else "plugin",
                }
            )
        tools.sort(key=lambda t: (t["category"], t["name"]))
        return JSONResponse({"tools": tools})
