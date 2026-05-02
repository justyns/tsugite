"""Tests for the plugin-facing filtered subscription API."""

from tsugite.events import (
    BaseEvent,
    EventBus,
    LLMMessageEvent,
    TaskStartEvent,
    ToolCallEvent,
)
from tsugite.events.bus import Subscription
from tsugite.events.events import CustomEvent


class TestEventNameProperty:
    def test_task_start_event_name(self):
        event = TaskStartEvent(task="x", model="y")
        assert event.event_name == "task_start"

    def test_tool_call_event_name(self):
        event = ToolCallEvent(tool_name="run", arguments={})
        assert event.event_name == "tool_call"

    def test_llm_message_event_name(self):
        event = LLMMessageEvent(content="hi")
        assert event.event_name == "llm_message"


class TestCustomEvent:
    def test_custom_event_name_override(self):
        event = CustomEvent(custom_name="vikunja.task.created", payload={"id": 1})
        assert event.event_name == "vikunja.task.created"
        assert event.payload == {"id": 1}

    def test_custom_event_default_payload(self):
        event = CustomEvent(custom_name="x.y")
        assert event.payload == {}

    def test_custom_event_is_base_event(self):
        event = CustomEvent(custom_name="foo")
        assert isinstance(event, BaseEvent)


class TestSubscribeFiltered:
    def test_filter_by_name_matches_only_named_events(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(received.append, event_name="tool_call")

        bus.emit(ToolCallEvent(tool_name="run", arguments={}))
        bus.emit(LLMMessageEvent(content="hello"))

        assert len(received) == 1
        assert received[0].event_name == "tool_call"

    def test_filter_none_receives_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(received.append, event_name=None)

        bus.emit(ToolCallEvent(tool_name="run", arguments={}))
        bus.emit(LLMMessageEvent(content="hello"))

        assert len(received) == 2

    def test_filter_matches_custom_event_name(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(received.append, event_name="vikunja.task.created")

        bus.emit(CustomEvent(custom_name="vikunja.task.created", payload={"id": 1}))
        bus.emit(CustomEvent(custom_name="other.event", payload={}))

        assert len(received) == 1
        assert received[0].payload == {"id": 1}

    def test_predicate_gates_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(
            received.append,
            event_name="tool_call",
            predicate=lambda e: e.tool_name == "run",
        )

        bus.emit(ToolCallEvent(tool_name="run", arguments={}))
        bus.emit(ToolCallEvent(tool_name="read_file", arguments={}))

        assert len(received) == 1
        assert received[0].tool_name == "run"

    def test_predicate_without_event_name(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(received.append, predicate=lambda e: e.event_name == "task_start")

        bus.emit(TaskStartEvent(task="x", model="y"))
        bus.emit(LLMMessageEvent(content="hi"))

        assert len(received) == 1


class TestErrorIsolation:
    def test_filtered_handler_error_does_not_break_others(self):
        bus = EventBus()
        received = []

        def boom(event):
            raise ValueError("intentional")

        bus.subscribe_filtered(boom, event_name="task_start")
        bus.subscribe_filtered(received.append, event_name="task_start")

        bus.emit(TaskStartEvent(task="x", model="y"))
        assert len(received) == 1

    def test_unfiltered_handler_unaffected_by_filtered_failure(self):
        bus = EventBus()
        received = []

        def boom(event):
            raise RuntimeError("boom")

        bus.subscribe(received.append)
        bus.subscribe_filtered(boom, event_name=None)

        bus.emit(TaskStartEvent(task="x", model="y"))
        assert len(received) == 1


class TestExistingSubscribeUnchanged:
    def test_subscribe_still_receives_every_event(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)

        bus.emit(TaskStartEvent(task="x", model="y"))
        bus.emit(LLMMessageEvent(content="hi"))
        bus.emit(ToolCallEvent(tool_name="run", arguments={}))

        assert len(received) == 3


class TestUnsubscribeFiltered:
    def test_unsubscribe_filtered_removes_handler(self):
        bus = EventBus()
        received = []
        bus.subscribe_filtered(received.append, event_name="tool_call")

        bus.emit(ToolCallEvent(tool_name="run", arguments={}))
        bus.unsubscribe_filtered(received.append)
        bus.emit(ToolCallEvent(tool_name="run", arguments={}))

        assert len(received) == 1

    def test_unsubscribe_filtered_unknown_handler_is_noop(self):
        bus = EventBus()
        bus.unsubscribe_filtered(lambda e: None)


class TestPluginSubscriptionAutoAttach:
    def setup_method(self):
        import tsugite.plugins

        tsugite.plugins._plugin_subscriptions = []

    def teardown_method(self):
        import tsugite.plugins

        tsugite.plugins._plugin_subscriptions = []

    def test_event_bus_auto_attaches_plugin_subscriptions(self):
        import tsugite.plugins

        received = []
        tsugite.plugins._plugin_subscriptions.append(
            Subscription(event_name="task_start", handler=received.append, predicate=None)
        )

        bus = EventBus()
        bus.emit(TaskStartEvent(task="x", model="y"))
        bus.emit(LLMMessageEvent(content="hi"))

        assert len(received) == 1
        assert received[0].event_name == "task_start"

    def test_buses_created_before_subscription_do_not_pick_up_later_plugins(self):
        import tsugite.plugins

        bus = EventBus()
        received = []
        tsugite.plugins._plugin_subscriptions.append(
            Subscription(event_name="task_start", handler=received.append, predicate=None)
        )

        bus.emit(TaskStartEvent(task="x", model="y"))
        assert received == []


class TestSubscribeDecorator:
    def setup_method(self):
        import tsugite.plugins

        tsugite.plugins._plugin_subscriptions = []

    def teardown_method(self):
        import tsugite.plugins

        tsugite.plugins._plugin_subscriptions = []

    def test_decorator_registers_subscription(self):
        from tsugite.events.bus import subscribe
        from tsugite.plugins import get_plugin_subscriptions

        @subscribe(event_name="tool_call")
        def on_tool_call(event):
            pass

        subs = get_plugin_subscriptions()
        assert len(subs) == 1
        assert subs[0].event_name == "tool_call"
        assert subs[0].handler is on_tool_call
        assert subs[0].predicate is None

    def test_decorator_with_predicate(self):
        from tsugite.events.bus import subscribe
        from tsugite.plugins import get_plugin_subscriptions

        @subscribe(event_name="tool_call", predicate=lambda e: e.tool_name == "run")
        def on_run(event):
            pass

        sub = get_plugin_subscriptions()[0]
        assert sub.predicate is not None

    def test_decorator_no_filter_receives_all(self):
        from tsugite.events.bus import subscribe
        from tsugite.plugins import get_plugin_subscriptions

        @subscribe()
        def on_anything(event):
            pass

        sub = get_plugin_subscriptions()[0]
        assert sub.event_name is None
        assert sub.predicate is None

    def test_multiple_decorators_accumulate(self):
        from tsugite.events.bus import subscribe
        from tsugite.plugins import get_plugin_subscriptions

        @subscribe(event_name="task_start")
        def first(event):
            pass

        @subscribe(event_name="tool_call")
        def second(event):
            pass

        subs = get_plugin_subscriptions()
        assert len(subs) == 2
        assert {s.event_name for s in subs} == {"task_start", "tool_call"}
