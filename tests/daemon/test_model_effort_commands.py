"""/model and /effort slash commands: show or change the open chat's per-session
model override and reasoning effort, mirroring what the web UI picker does through
the settings PATCH (same validation, same storage)."""

from unittest.mock import MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter
from tsugite_daemon.commands import CommandError, cmd_effort, cmd_model
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


def _adapter(tmp_path):
    store = SessionStore(tmp_path / "session_store.json")
    adapter = MagicMock()
    adapter.session_store = store
    adapter.agent_name = "default"
    adapter.resolve_model.return_value = "claude_code:haiku"
    # Exercise the real per-session resolvers (they honor the override and read the
    # live model registry) instead of the mock's auto-stubs.
    adapter.resolve_session_model.side_effect = lambda sid: BaseAdapter.resolve_session_model(adapter, sid)
    adapter.session_effort_levels.side_effect = lambda sid: BaseAdapter.session_effort_levels(adapter, sid)
    return adapter, store


def _session(store, sid="sess-open-chat"):
    s = Session(
        id=sid,
        source=SessionSource.INTERACTIVE.value,
        user_id="user-1",
        message_count=1,
    )
    store.create_session(s)
    return s


# ── /model ──


@pytest.mark.asyncio
async def test_cmd_model_no_arg_marks_session_override(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    store.set_model_override("sess-open-chat", "codex_cli:gpt-5.5")
    out = await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    assert "codex_cli:gpt-5.5" in out
    assert "override" in out.lower()
    assert "claude_code:haiku" in out  # agent default noted alongside


@pytest.mark.asyncio
async def test_cmd_model_no_arg_shows_agent_default(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    assert "claude_code:haiku" in out
    assert "default" in out.lower()
    assert "override" not in out.lower()


@pytest.mark.asyncio
async def test_cmd_model_set_valid_persists(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="codex_cli:gpt-5.5")
    assert store.get_model_override("sess-open-chat") == "codex_cli:gpt-5.5"
    assert "codex_cli:gpt-5.5" in out


@pytest.mark.asyncio
async def test_cmd_model_malformed_rejected_with_hint(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    with pytest.raises(CommandError) as ei:
        await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="notamodel")
    assert "notamodel" in str(ei.value)
    assert store.get_model_override("sess-open-chat") is None  # rejected, not applied


@pytest.mark.asyncio
async def test_cmd_model_set_alias_stored_raw_like_picker(tmp_path):
    """A provider:alias form (opus -> claude-opus-4-8) is accepted and stored
    VERBATIM, exactly as the picker's PATCH stores it, so downstream alias
    resolution stays identical (the command must not pre-resolve before storing)."""
    from tsugite.models import get_provider_and_model

    adapter, store = _adapter(tmp_path)
    _session(store)
    await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="claude_code:opus")
    stored = store.get_model_override("sess-open-chat")
    assert stored == "claude_code:opus"  # raw alias kept, not the resolved id
    assert get_provider_and_model(stored)[2] == "claude-opus-4-8"  # yet still resolvable


@pytest.mark.asyncio
async def test_cmd_model_unknown_provider_rejected(tmp_path):
    """A typo'd provider (get_provider_and_model only parses the string shape) must
    be rejected before it poisons the next turn. Live repro: notreal:fake-9."""
    adapter, store = _adapter(tmp_path)
    _session(store)
    with pytest.raises(CommandError) as ei:
        await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="notreal:fake-9")
    assert "notreal" in str(ei.value)
    assert store.get_model_override("sess-open-chat") is None  # rejected, not applied


@pytest.mark.asyncio
async def test_cmd_model_unlisted_id_on_known_provider_accepted_with_caution(tmp_path):
    """A KNOWN provider with an unlisted model id is accepted (API providers take
    arbitrary ids), NOT rejected -- but the success message carries a caution."""
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_model(
        adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="claude_code:totally-fake"
    )
    assert store.get_model_override("sess-open-chat") == "claude_code:totally-fake"  # accepted + stored raw
    assert "recognized model" in out.lower()  # caution present
    assert "claude_code" in out


@pytest.mark.asyncio
async def test_cmd_model_unlisted_id_on_api_provider_no_caution(tmp_path):
    """An API provider (openai/anthropic/openrouter/ollama) legitimately accepts
    arbitrary model ids, so an unrecognized id must NOT trigger the caution. Only
    providers that declare a finite, definitive model set (claude_code, codex_cli)
    should nag — otherwise every valid brand-new API model id looks 'wrong'."""
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_model(
        adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="openai:brand-new-model-9"
    )
    assert store.get_model_override("sess-open-chat") == "openai:brand-new-model-9"  # accepted + stored raw
    assert "recognized model" not in out.lower()  # NO caution: openai model set isn't definitive


@pytest.mark.asyncio
async def test_cmd_model_default_clears_override(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    store.set_model_override("sess-open-chat", "codex_cli:gpt-5.5")
    out = await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="default")
    assert store.get_model_override("sess-open-chat") is None
    assert "claude_code:haiku" in out


@pytest.mark.asyncio
async def test_cmd_model_clear_alias_also_clears(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    store.set_model_override("sess-open-chat", "codex_cli:gpt-5.5")
    await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="clear")
    assert store.get_model_override("sess-open-chat") is None


# ── /effort ──


@pytest.mark.asyncio
async def test_cmd_effort_no_arg_lists_supported_levels(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    for lvl in ("low", "high", "max"):  # claude_code:haiku exposes low..max
        assert lvl in out


@pytest.mark.asyncio
async def test_cmd_effort_set_valid_persists(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    out = await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="high")
    assert store.get_reasoning_effort("sess-open-chat") == "high"
    assert "high" in out


@pytest.mark.asyncio
async def test_cmd_effort_unsupported_rejected_with_levels(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    with pytest.raises(CommandError) as ei:
        await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="bogus")
    msg = str(ei.value)
    assert "bogus" in msg
    assert "low" in msg  # error lists the supported levels
    assert store.get_reasoning_effort("sess-open-chat") is None


@pytest.mark.asyncio
async def test_cmd_effort_default_clears(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    store.set_reasoning_effort("sess-open-chat", "high")
    await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="default")
    assert store.get_reasoning_effort("sess-open-chat") is None


@pytest.mark.asyncio
async def test_cmd_effort_levels_are_model_dependent(tmp_path):
    """The override's model (gpt-5.5, no 'max') drives both the shown levels and
    validation, not the agent default (haiku, which does support 'max')."""
    adapter, store = _adapter(tmp_path)
    _session(store)
    store.set_model_override("sess-open-chat", "codex_cli:gpt-5.5")
    out = await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    assert "xhigh" in out
    assert "max" not in out
    with pytest.raises(CommandError):
        await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="max")


# ── settings broadcast: /model and /effort push a session_update so open chats
#    in other tabs refresh their model/effort chips live ──


@pytest.mark.asyncio
async def test_cmd_model_broadcasts_settings_change(tmp_path):
    adapter, store = _adapter(tmp_path)
    adapter.event_bus = MagicMock()
    _session(store)
    await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="codex_cli:gpt-5.5")
    updates = [c for c in adapter.event_bus.emit.call_args_list if c.args and c.args[0] == "session_update"]
    assert updates, "expected a session_update broadcast"
    payload = updates[-1].args[1]
    assert payload["action"] == "settings"
    assert payload["id"] == "sess-open-chat"
    assert payload["model"] == "codex_cli:gpt-5.5"


@pytest.mark.asyncio
async def test_cmd_model_default_broadcasts_the_reset(tmp_path):
    adapter, store = _adapter(tmp_path)
    adapter.event_bus = MagicMock()
    _session(store)
    store.set_model_override("sess-open-chat", "codex_cli:gpt-5.5")
    await cmd_model(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="default")
    updates = [c for c in adapter.event_bus.emit.call_args_list if c.args and c.args[0] == "session_update"]
    assert updates and updates[-1].args[1]["action"] == "settings"
    assert updates[-1].args[1]["model"] is None


@pytest.mark.asyncio
async def test_cmd_effort_broadcasts_settings_change(tmp_path):
    adapter, store = _adapter(tmp_path)
    adapter.event_bus = MagicMock()
    _session(store)
    await cmd_effort(adapter=adapter, user_id="user-1", session_id="sess-open-chat", message="high")
    updates = [c for c in adapter.event_bus.emit.call_args_list if c.args and c.args[0] == "session_update"]
    assert updates, "expected a session_update broadcast"
    payload = updates[-1].args[1]
    assert payload["action"] == "settings"
    assert payload["id"] == "sess-open-chat"
    assert payload["reasoning_effort"] == "high"
