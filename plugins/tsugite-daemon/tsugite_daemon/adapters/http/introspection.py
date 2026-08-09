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


class IntrospectionMixin:
    def _introspection_routes(self) -> list:
        return [
            Route("/api/plugins", self._api_list_plugins, methods=["GET"]),
            Route("/api/tools", self._api_list_tools, methods=["GET"]),
        ]

    async def _api_list_plugins(self, request: Request) -> JSONResponse:
        """Discovered plugins across every entry-point group, plus the UI surfaces
        the loaded adapter plugins declare. Read-only wrapper over
        discover_plugins(); no enable/disable mutation."""
        if err := self._check_auth(request):
            return err
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
