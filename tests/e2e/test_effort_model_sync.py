"""E2E: the reasoning-effort selector must follow the session's selected model.

Regression for the bug where switching a session's model left the effort
dropdown showing the previous model's levels (e.g. `xhigh` lingering for a model
that doesn't support it), and a now-invalid effort was left selected.
"""

from .helpers import (
    CONV_VIEW,
    open_conversations,
    reload_conversations_view,
    select_session_in_view,
    wait_for_session_in_list,
)


def _conv(page):
    return f"Alpine.$data(document.querySelector({CONV_VIEW!r}))"


def test_effort_dropdown_syncs_to_session_model(authenticated_page, e2e_session_store, e2e_adapter):
    page = authenticated_page

    original_model = e2e_adapter.agent_config.model
    # Pin the agent default to an xhigh-capable model so reload() seeds the effort
    # dropdown deterministically regardless of the global default model.
    e2e_adapter.agent_config.model = "claude_code:opus"
    try:
        user_id = page.evaluate("Alpine.store('app').userId")
        session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
        # Start on a model that supports xhigh, with xhigh selected.
        e2e_session_store.set_model_override(session.id, "claude_code:opus")
        e2e_session_store.set_reasoning_effort(session.id, "xhigh")

        open_conversations(page)
        reload_conversations_view(page)
        wait_for_session_in_list(page, session.id)
        select_session_in_view(page, session.id)

        conv = _conv(page)
        page.wait_for_function(f"{conv}.effortLevels.includes('xhigh')", timeout=3000)
        page.wait_for_function(f"{conv}.sessionEffort === 'xhigh'", timeout=3000)

        # Switch to a model that does NOT support xhigh. The function form makes
        # Playwright await the full setSessionModel -> reload -> clamp chain.
        page.evaluate(
            "(sel) => Alpine.$data(document.querySelector(sel)).setSessionModel('openai:o3')",
            CONV_VIEW,
        )
        page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

        effort_levels = page.evaluate(f"{conv}.effortLevels")
        session_effort = page.evaluate(f"{conv}.sessionEffort")

        # Dropdown refreshed to the new model's levels (no stale xhigh)...
        assert effort_levels == ["low", "medium", "high"]
        # ...and the now-unsupported xhigh was clamped down to a valid level.
        assert session_effort == "high"
    finally:
        e2e_adapter.agent_config.model = original_model


def test_effort_dropdown_follows_session_switch(authenticated_page, e2e_session_store, e2e_adapter):
    page = authenticated_page

    original_model = e2e_adapter.agent_config.model
    e2e_adapter.agent_config.model = "claude_code:opus"
    try:
        user_id = page.evaluate("Alpine.store('app').userId")
        # Two sessions with different model overrides: A supports xhigh, B does not.
        session_a = e2e_session_store.get_or_create_interactive(user_id, "test-agent")
        e2e_session_store.set_model_override(session_a.id, "claude_code:opus")
        session_b = e2e_session_store.create_default_session(user_id, "test-agent")
        e2e_session_store.set_model_override(session_b.id, "openai:o3")

        open_conversations(page)
        reload_conversations_view(page)
        wait_for_session_in_list(page, session_a.id)
        wait_for_session_in_list(page, session_b.id)

        conv = _conv(page)
        select_session_in_view(page, session_a.id)
        page.wait_for_function(f"{conv}.effortLevels.includes('xhigh')", timeout=3000)

        # Switching sessions must re-derive the dropdown from B's model.
        select_session_in_view(page, session_b.id)
        page.wait_for_function(
            f'JSON.stringify({conv}.effortLevels) === \'["low","medium","high"]\'',
            timeout=3000,
        )
        assert page.evaluate(f"{conv}.effortLevels") == ["low", "medium", "high"]
    finally:
        e2e_adapter.agent_config.model = original_model
