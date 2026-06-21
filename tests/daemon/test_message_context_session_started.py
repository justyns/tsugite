"""<session_started> rendering inside <message_context>.

The element gives the agent a signal of *how long the conversation has been
running* (separate from "now"), which matters for long-lived sessions where
"yesterday" is ambiguous between yesterday-relative-to-now and
yesterday-relative-to-conversation-start.
"""

from datetime import datetime, timedelta, timezone

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext  # noqa: F401
from tsugite_daemon.config import AgentConfig
from tsugite_daemon.session_store import SessionStore


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: test-agent\n---\n\nHi.\n")

    store = SessionStore(tmp_path / "store.json")
    config = AgentConfig(workspace_dir=ws, agent_file=str(ws / "agent.md"))
    return _StubAdapter("test-agent", config, store)


@pytest.fixture
def channel_context():
    return ChannelContext(source="http", channel_id=None, user_id="alice", reply_to="http:alice")


def _seed_session(adapter, age: timedelta | None = None) -> str:
    session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
    if age is not None:
        session.created_at = (datetime.now(timezone.utc) - age).isoformat()
    return session.id


class TestSessionStartedRendering:
    def test_aged_session_renders_relative_age(self, adapter, channel_context):
        _seed_session(adapter, age=timedelta(days=6))
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_started>" in rendered
        assert "6 days ago" in rendered
        assert "</session_started>" in rendered

    def test_week_old_session_collapses_to_weeks(self, adapter, channel_context):
        _seed_session(adapter, age=timedelta(days=7))
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "1 week ago" in rendered

    def test_fresh_session_renders_just_now(self, adapter, channel_context):
        _seed_session(adapter)
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_started>" in rendered
        assert "just now" in rendered

    def test_session_started_inside_message_context(self, adapter, channel_context):
        _seed_session(adapter, age=timedelta(hours=3))
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        mc_start = rendered.index("<message_context>")
        mc_end = rendered.index("</message_context>")
        ss_pos = rendered.index("<session_started>")
        assert mc_start < ss_pos < mc_end

    def test_session_started_after_datetime(self, adapter, channel_context):
        _seed_session(adapter, age=timedelta(hours=3))
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        dt_pos = rendered.index("<datetime>")
        ss_pos = rendered.index("<session_started>")
        assert dt_pos < ss_pos

    def test_missing_created_at_omits_element(self, adapter, channel_context):
        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        session.created_at = ""
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_started>" not in rendered

    def test_unparseable_created_at_omits_element(self, adapter, channel_context):
        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        session.created_at = "not-a-real-datetime"
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<session_started>" not in rendered


class TestLastActiveRendering:
    """`<last_active>` lets the agent distinguish a stale resumed session
    ("we last spoke 3 days ago") from session age. Different signal — a
    multi-day session can still have just-now activity, and vice versa.
    """

    def test_aged_last_active_renders_relative(self, adapter, channel_context):
        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        session.last_active = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<last_active>" in rendered
        assert "3 days ago" in rendered

    def test_last_active_after_session_started(self, adapter, channel_context):
        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        session.last_active = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        ss_pos = rendered.index("<session_started>")
        la_pos = rendered.index("<last_active>")
        assert ss_pos < la_pos

    def test_missing_last_active_omits_element(self, adapter, channel_context):
        session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
        session.last_active = ""
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<last_active>" not in rendered


class TestSchedulerTimingRendering:
    """When source=scheduler, expose when this fire was planned for and when
    it actually fired. Lets agents notice misfires / queue delays.
    """

    def test_renders_scheduled_for_and_actual_fire_time(self, adapter):
        scheduled = "2026-05-04T09:00:00+00:00"
        actual = "2026-05-04T09:04:30+00:00"
        ctx = ChannelContext(
            source="scheduler",
            channel_id=None,
            user_id="scheduler:bot",
            reply_to="scheduler:bot",
            metadata={"scheduled_for": scheduled, "actual_fire_time": actual},
        )
        rendered = adapter._build_message_context("hello", ctx, "scheduler:bot")
        assert "<scheduled_for>" in rendered
        assert "<actual_fire_time>" in rendered
        # Both should render the date — exact tz formatting depends on agent_config
        assert "2026-05-04" in rendered

    def test_omits_elements_for_non_scheduler_sources(self, adapter, channel_context):
        # channel_context.source is "http" by default fixture
        rendered = adapter._build_message_context("hello", channel_context, "alice")
        assert "<scheduled_for>" not in rendered
        assert "<actual_fire_time>" not in rendered
