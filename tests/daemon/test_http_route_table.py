"""Lock the HTTP route table shape and its ordering invariants.

Starlette matches routes in declaration order, so a few literal routes must
precede their `{param}` siblings and the catch-alls must stay last. This test
pins the table as two golden snapshots -- a top-level view (clean-prefix
domains collapse to a single Mount row) and a fully expanded view (every leaf
route, absolute path) -- plus the load-bearing ordering rules. The refactor
that grouped, split, and Mounted these routes could only move them
deliberately: each intentional change updates the goldens in the same commit.
"""

import pytest
from starlette.routing import Mount, Route
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore


@pytest.fixture
def server(tmp_path):
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
    )


def _row(path: str, route) -> tuple[str, tuple[str, ...], str]:
    methods = tuple(sorted((route.methods or set()) - {"HEAD", "OPTIONS"}))
    return (path, methods, route.endpoint.__name__)


def _route_table(app) -> list[tuple[str, tuple[str, ...], str]]:
    """Top-level view: Mounts collapse to one ("MOUNT",) row carrying the name.

    Nested sub-apps (StaticFiles, the per-domain route groups) don't leak their
    internals here -- see `_full_route_table` for the expanded lock.
    """
    rows: list[tuple[str, tuple[str, ...], str]] = []
    for r in app.routes:
        if isinstance(r, Mount):
            rows.append((r.path, ("MOUNT",), r.name))
        elif isinstance(r, Route):
            rows.append(_row(r.path, r))
    return rows


def _full_route_table(app) -> list[tuple[str, tuple[str, ...], str]]:
    """Expanded view: recurse one level into route-group Mounts, absolute paths.

    A Mount whose app is not a router (StaticFiles) has no sub-Routes to expand,
    so it stays a single ("MOUNT",) row.
    """
    rows: list[tuple[str, tuple[str, ...], str]] = []
    for r in app.routes:
        if isinstance(r, Mount):
            sub = [sr for sr in (getattr(r, "routes", None) or []) if isinstance(sr, Route)]
            if sub:
                for sr in sub:
                    rows.append(_row(r.path + sr.path, sr))
            else:
                rows.append((r.path, ("MOUNT",), r.name))
        elif isinstance(r, Route):
            rows.append(_row(r.path, r))
    return rows


GOLDEN_ROUTE_TABLE = [
    ("/api/health", ("GET",), "_health"),
    ("/api/agents", ("GET",), "_list_agents"),
    ("/api/models", ("GET",), "_list_models"),
    ("/api/events", ("GET",), "_events"),
    ("/api/commands", ("GET",), "_list_commands"),
    ("/api/agents/{agent}/sessions", ("GET",), "_list_sessions"),
    ("/api/agents/{agent}/sessions/new", ("POST",), "_new_interactive_session"),
    ("/api/agents/{agent}/sessions/{session_id}/branch", ("POST",), "_branch"),
    ("/api/agents/{agent}/chat", ("POST",), "_chat"),
    ("/api/agents/{agent}/chat/cancel", ("POST",), "_cancel_chat"),
    ("/api/agents/{agent}/upload", ("POST",), "_upload"),
    ("/api/agents/{agent}/status", ("GET",), "_status"),
    ("/api/agents/{agent}/attachments", ("GET",), "_attachments"),
    ("/api/agents/{agent}/history", ("GET",), "_history"),
    ("/api/agents/{agent}/prompt-snapshot", ("GET",), "_prompt_snapshot"),
    ("/api/agents/{agent}/config", ("PATCH",), "_update_agent_config"),
    ("/api/agents/{agent}/compact", ("POST",), "_compact"),
    ("/api/agents/{agent}/respond", ("POST",), "_respond"),
    ("/api/agents/{agent}/unload-skill", ("POST",), "_unload_skill"),
    ("/api/agents/{agent}/effort-levels", ("GET",), "_effort_levels"),
    ("/api/agents/{agent}/workspace", ("GET",), "_list_workspace_files"),
    ("/api/agents/{agent}/workspace/content", ("GET",), "_read_workspace_file"),
    ("/api/agents/{agent}/workspace/content", ("PUT",), "_save_workspace_file"),
    ("/api/agents/{agent}/workspace/attach", ("POST",), "_attach_workspace_file"),
    ("/api/agents/{agent}/commands/{command_name}", ("POST",), "_run_command"),
    ("/api/sessions", ("MOUNT",), "sessions"),
    ("/api/sessions", ("GET",), "_api_list_sessions"),
    ("/api/sessions", ("POST",), "_api_start_session"),
    ("/api/schedules", ("MOUNT",), "schedules"),
    ("/api/schedules", ("GET",), "_list_schedules"),
    ("/api/schedules", ("POST",), "_create_schedule"),
    ("/api/jobs", ("MOUNT",), "jobs"),
    ("/api/jobs", ("GET",), "_api_list_jobs"),
    ("/api/executors", ("GET",), "_api_list_executors"),
    ("/api/terminals", ("MOUNT",), "terminals"),
    ("/api/terminals", ("GET",), "_api_list_terminals"),
    ("/api/terminals", ("POST",), "_api_create_terminal"),
    ("/api/webhooks", ("MOUNT",), "webhooks"),
    ("/api/webhooks", ("GET",), "_list_webhooks"),
    ("/api/webhooks", ("POST",), "_create_webhook"),
    ("/webhook/{token}", ("POST",), "_webhook"),
    ("/api/agent-files", ("GET",), "_list_agent_files"),
    ("/api/agent-files/content", ("GET",), "_read_agent_file"),
    ("/api/agent-files/content", ("PUT",), "_save_agent_file"),
    ("/api/skill-files", ("GET",), "_list_skill_files"),
    ("/api/skill-files/content", ("GET",), "_read_skill_file"),
    ("/api/skill-files/content", ("PUT",), "_save_skill_file"),
    ("/api/skills/issues", ("GET",), "_list_skill_issues"),
    ("/api/push", ("MOUNT",), "push"),
    ("/api/secrets", ("MOUNT",), "secrets"),
    ("/api/secrets", ("GET",), "_secrets_list"),
    ("/api/usage", ("MOUNT",), "usage"),
    ("/static", ("MOUNT",), "static"),
    ("/sw.js", ("GET",), "_serve_sw"),
    ("/", ("GET",), "_serve_ui"),
]


FULL_ROUTE_TABLE = [
    ("/api/health", ("GET",), "_health"),
    ("/api/agents", ("GET",), "_list_agents"),
    ("/api/models", ("GET",), "_list_models"),
    ("/api/events", ("GET",), "_events"),
    ("/api/commands", ("GET",), "_list_commands"),
    ("/api/agents/{agent}/sessions", ("GET",), "_list_sessions"),
    ("/api/agents/{agent}/sessions/new", ("POST",), "_new_interactive_session"),
    ("/api/agents/{agent}/sessions/{session_id}/branch", ("POST",), "_branch"),
    ("/api/agents/{agent}/chat", ("POST",), "_chat"),
    ("/api/agents/{agent}/chat/cancel", ("POST",), "_cancel_chat"),
    ("/api/agents/{agent}/upload", ("POST",), "_upload"),
    ("/api/agents/{agent}/status", ("GET",), "_status"),
    ("/api/agents/{agent}/attachments", ("GET",), "_attachments"),
    ("/api/agents/{agent}/history", ("GET",), "_history"),
    ("/api/agents/{agent}/prompt-snapshot", ("GET",), "_prompt_snapshot"),
    ("/api/agents/{agent}/config", ("PATCH",), "_update_agent_config"),
    ("/api/agents/{agent}/compact", ("POST",), "_compact"),
    ("/api/agents/{agent}/respond", ("POST",), "_respond"),
    ("/api/agents/{agent}/unload-skill", ("POST",), "_unload_skill"),
    ("/api/agents/{agent}/effort-levels", ("GET",), "_effort_levels"),
    ("/api/agents/{agent}/workspace", ("GET",), "_list_workspace_files"),
    ("/api/agents/{agent}/workspace/content", ("GET",), "_read_workspace_file"),
    ("/api/agents/{agent}/workspace/content", ("PUT",), "_save_workspace_file"),
    ("/api/agents/{agent}/workspace/attach", ("POST",), "_attach_workspace_file"),
    ("/api/agents/{agent}/commands/{command_name}", ("POST",), "_run_command"),
    ("/api/sessions/{session_id}/settings", ("GET",), "_session_settings_get"),
    ("/api/sessions/{session_id}/settings", ("PATCH",), "_session_settings_patch"),
    ("/api/sessions/", ("GET",), "_api_list_sessions"),
    ("/api/sessions/", ("POST",), "_api_start_session"),
    ("/api/sessions/{session_id}/metadata", ("GET",), "_api_get_metadata"),
    ("/api/sessions/{session_id}/metadata", ("PATCH",), "_api_update_metadata"),
    ("/api/sessions/{session_id}/metadata/{key}", ("DELETE",), "_api_delete_metadata"),
    ("/api/sessions/{session_id}/scratchpad", ("GET",), "_api_get_scratchpad"),
    ("/api/sessions/{session_id}/scratchpad", ("PUT",), "_api_update_scratchpad"),
    ("/api/sessions/{session_id}", ("GET",), "_api_get_session"),
    ("/api/sessions/{session_id}", ("PATCH",), "_api_update_session"),
    ("/api/sessions/{session_id}/cancel", ("POST",), "_api_cancel_session"),
    ("/api/sessions/{session_id}/restart", ("POST",), "_api_restart_session"),
    ("/api/sessions/{session_id}/events", ("GET",), "_api_session_events"),
    ("/api/sessions/{session_id}/pin", ("POST",), "_api_pin_session"),
    ("/api/sessions/{session_id}/unpin", ("POST",), "_api_unpin_session"),
    ("/api/sessions/pinned/reorder", ("POST",), "_api_reorder_pins"),
    ("/api/sessions/clear-primary", ("POST",), "_api_clear_primary"),
    ("/api/sessions/{session_id}/set-primary", ("POST",), "_api_set_primary"),
    ("/api/sessions/{session_id}/mark-viewed", ("POST",), "_api_mark_viewed"),
    ("/api/sessions", ("GET",), "_api_list_sessions"),
    ("/api/sessions", ("POST",), "_api_start_session"),
    ("/api/schedules/", ("GET",), "_list_schedules"),
    ("/api/schedules/", ("POST",), "_create_schedule"),
    ("/api/schedules/cleanup", ("POST",), "_cleanup_schedules"),
    ("/api/schedules/{schedule_id}", ("GET",), "_get_schedule"),
    ("/api/schedules/{schedule_id}", ("PATCH",), "_update_schedule"),
    ("/api/schedules/{schedule_id}", ("DELETE",), "_delete_schedule"),
    ("/api/schedules/{schedule_id}/enable", ("POST",), "_enable_schedule"),
    ("/api/schedules/{schedule_id}/disable", ("POST",), "_disable_schedule"),
    ("/api/schedules/{schedule_id}/run", ("POST",), "_run_schedule"),
    ("/api/schedules/{schedule_id}/sessions", ("GET",), "_schedule_sessions"),
    ("/api/schedules", ("GET",), "_list_schedules"),
    ("/api/schedules", ("POST",), "_create_schedule"),
    ("/api/jobs/", ("GET",), "_api_list_jobs"),
    ("/api/jobs/{job_id}/cancel", ("POST",), "_api_cancel_job"),
    ("/api/jobs/{job_id}/mark-done", ("POST",), "_api_mark_job_done"),
    ("/api/jobs/{job_id}/retry", ("POST",), "_api_retry_job"),
    ("/api/jobs", ("GET",), "_api_list_jobs"),
    ("/api/executors", ("GET",), "_api_list_executors"),
    ("/api/terminals/", ("GET",), "_api_list_terminals"),
    ("/api/terminals/", ("POST",), "_api_create_terminal"),
    ("/api/terminals/{terminal_id}", ("GET",), "_api_get_terminal"),
    ("/api/terminals/{terminal_id}/kill", ("POST",), "_api_kill_terminal"),
    ("/api/terminals/{terminal_id}/stdin", ("POST",), "_api_terminal_stdin"),
    ("/api/terminals/{terminal_id}/restart", ("POST",), "_api_restart_terminal"),
    ("/api/terminals/{terminal_id}/stream", ("GET",), "_api_terminal_stream"),
    ("/api/terminals", ("GET",), "_api_list_terminals"),
    ("/api/terminals", ("POST",), "_api_create_terminal"),
    ("/api/webhooks/", ("GET",), "_list_webhooks"),
    ("/api/webhooks/", ("POST",), "_create_webhook"),
    ("/api/webhooks/{token}", ("DELETE",), "_delete_webhook"),
    ("/api/webhooks", ("GET",), "_list_webhooks"),
    ("/api/webhooks", ("POST",), "_create_webhook"),
    ("/webhook/{token}", ("POST",), "_webhook"),
    ("/api/agent-files", ("GET",), "_list_agent_files"),
    ("/api/agent-files/content", ("GET",), "_read_agent_file"),
    ("/api/agent-files/content", ("PUT",), "_save_agent_file"),
    ("/api/skill-files", ("GET",), "_list_skill_files"),
    ("/api/skill-files/content", ("GET",), "_read_skill_file"),
    ("/api/skill-files/content", ("PUT",), "_save_skill_file"),
    ("/api/skills/issues", ("GET",), "_list_skill_issues"),
    ("/api/push/vapid-key", ("GET",), "_push_vapid_key"),
    ("/api/push/subscribe", ("POST",), "_push_subscribe"),
    ("/api/push/unsubscribe", ("POST",), "_push_unsubscribe"),
    ("/api/secrets/", ("GET",), "_secrets_list"),
    ("/api/secrets/{name:path}", ("POST",), "_secrets_set"),
    ("/api/secrets/{name:path}", ("DELETE",), "_secrets_delete"),
    ("/api/secrets", ("GET",), "_secrets_list"),
    ("/api/usage/summary", ("GET",), "_usage_summary"),
    ("/api/usage/agents", ("GET",), "_usage_agents"),
    ("/api/usage/models", ("GET",), "_usage_models"),
    ("/api/usage/total", ("GET",), "_usage_total"),
    ("/static", ("MOUNT",), "static"),
    ("/sw.js", ("GET",), "_serve_sw"),
    ("/", ("GET",), "_serve_ui"),
]


def test_route_table_snapshot(server):
    assert _route_table(server.app) == GOLDEN_ROUTE_TABLE


def test_full_route_table_snapshot(server):
    assert _full_route_table(server.app) == FULL_ROUTE_TABLE


def _mount_subpaths(app, prefix: str) -> list[str]:
    for r in app.routes:
        if isinstance(r, Mount) and r.path == prefix:
            return [sr.path for sr in r.routes if isinstance(sr, Route)]
    raise AssertionError(f"no mount at {prefix}")


def _first_index(items, predicate) -> int:
    for i, item in enumerate(items):
        if predicate(item):
            return i
    raise AssertionError("no match found")


def test_clear_primary_precedes_set_primary(server):
    subpaths = _mount_subpaths(server.app, "/api/sessions")
    clear = subpaths.index("/clear-primary")
    set_primary = subpaths.index("/{session_id}/set-primary")
    assert clear < set_primary


def test_schedules_cleanup_precedes_param_route(server):
    subpaths = _mount_subpaths(server.app, "/api/schedules")
    cleanup = subpaths.index("/cleanup")
    param = _first_index(subpaths, lambda p: p == "/{schedule_id}")
    assert cleanup < param


def test_catchalls_last(server):
    table = _route_table(server.app)
    assert [row[0] for row in table[-3:]] == ["/static", "/sw.js", "/"]


def _mounts_with_root_handler(app) -> list[str]:
    """Every top-level Mount that has a GET root ('/') sub-route - i.e. a
    collection whose bare prefix would 307-redirect without mounted_api_routes.
    Auto-discovered so a future collection (or plugin) that forgets the helper is
    caught here instead of silently reintroducing the redirect."""
    prefixes = []
    for r in app.routes:
        if not isinstance(r, Mount):
            continue
        sub = [sr for sr in (getattr(r, "routes", None) or []) if isinstance(sr, Route)]
        if any(sr.path == "/" and "GET" in (sr.methods or set()) for sr in sub):
            prefixes.append(r.path)
    return prefixes


# A Mount("/api/jobs", ...) only matches sub-paths starting with "/", so the
# bare collection root 307-redirects to the trailing-slash form - and Starlette
# builds that Location as an absolute internal-host URL a proxied browser can't
# follow. EVERY collection with a root handler must resolve its bare prefix
# directly (auth check runs -> 401), never a redirect.
def test_no_collection_root_redirects(server):
    client = TestClient(server.app)
    prefixes = _mounts_with_root_handler(server.app)
    assert prefixes, "expected to discover mounted collections with a root handler"
    for prefix in prefixes:
        resp = client.get(prefix, follow_redirects=False)
        assert resp.status_code != 307, f"GET {prefix} must not 307-redirect (use mounted_api_routes)"
        assert resp.status_code == 401, f"GET {prefix} must reach the auth check, got {resp.status_code}"
