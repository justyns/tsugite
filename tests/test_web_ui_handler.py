"""Tests for SSE UI handler."""

import asyncio
import queue

import pytest

from tsugite.custom_ui import UIEvent
from tsugite.web.ui_handler import SSEUIHandler


class TestSSEUIHandler:
    """Test SSE UI handler."""

    def test_init(self):
        """Test handler initialization."""
        handler = SSEUIHandler()

        assert handler.event_queue is not None
        assert isinstance(handler.event_queue, queue.Queue)
        assert handler._final_result is None
        assert handler._done is False

    def test_handle_event_queues_event(self):
        """Test that events are queued."""
        handler = SSEUIHandler()

        handler.handle_event(UIEvent.TASK_START, {"task": "Test task", "model": "test-model"})

        assert not handler.event_queue.empty()
        event = handler.event_queue.get_nowait()
        assert event["event"] == "task_start"
        assert event["data"]["task"] == "Test task"
        assert event["data"]["model"] == "test-model"

    def test_handle_multiple_events(self):
        """Test handling multiple events."""
        handler = SSEUIHandler()

        handler.handle_event(UIEvent.TASK_START, {"task": "Test task"})
        handler.handle_event(UIEvent.STEP_START, {"step": 1})
        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": "print('hello')"})

        assert handler.event_queue.qsize() == 3

        event1 = handler.event_queue.get_nowait()
        assert event1["event"] == "task_start"

        event2 = handler.event_queue.get_nowait()
        assert event2["event"] == "step_start"

        event3 = handler.event_queue.get_nowait()
        assert event3["event"] == "code_execution"

    def test_final_answer_sets_done(self):
        """Test that FINAL_ANSWER sets done flag."""
        handler = SSEUIHandler()

        assert not handler.is_done

        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "Task complete"})

        assert handler.is_done
        assert handler.final_result == "Task complete"

    def test_error_sets_done(self):
        """Test that ERROR does not set done flag."""
        handler = SSEUIHandler()

        handler.handle_event(UIEvent.ERROR, {"error": "Test error", "error_type": "RuntimeError"})

        # Error doesn't set done, only FINAL_ANSWER does
        assert not handler.is_done
        assert handler.final_result == ""

    @pytest.mark.asyncio
    async def test_get_event_async(self):
        """Test async event retrieval."""
        handler = SSEUIHandler()

        # Queue an event
        handler.handle_event(UIEvent.TASK_START, {"task": "Async test"})

        # Get event asynchronously
        event = await handler.get_event()

        assert event["event"] == "task_start"
        assert event["data"]["task"] == "Async test"

    @pytest.mark.asyncio
    async def test_get_event_blocks_until_available(self):
        """Test that get_event waits for events."""
        handler = SSEUIHandler()

        # Start a task that queues an event after a delay
        async def queue_delayed():
            await asyncio.sleep(0.1)
            handler.handle_event(UIEvent.TASK_START, {"task": "Delayed"})

        # Start both tasks
        queue_task = asyncio.create_task(queue_delayed())
        event = await handler.get_event()

        assert event["event"] == "task_start"
        assert event["data"]["task"] == "Delayed"
        await queue_task

    def test_has_events(self):
        """Test has_events method."""
        handler = SSEUIHandler()

        assert not handler.has_events()

        handler.handle_event(UIEvent.TASK_START, {"task": "Test"})

        assert handler.has_events()

        handler.event_queue.get_nowait()

        assert not handler.has_events()

    def test_event_name_conversion(self):
        """Test that event enum is converted to lowercase name."""
        handler = SSEUIHandler()

        test_cases = [
            (UIEvent.TASK_START, "task_start"),
            (UIEvent.STEP_START, "step_start"),
            (UIEvent.CODE_EXECUTION, "code_execution"),
            (UIEvent.TOOL_CALL, "tool_call"),
            (UIEvent.OBSERVATION, "observation"),
            (UIEvent.FINAL_ANSWER, "final_answer"),
            (UIEvent.ERROR, "error"),
        ]

        for event_enum, expected_name in test_cases:
            handler.handle_event(event_enum, {"test": "data"})
            queued_event = handler.event_queue.get_nowait()
            assert queued_event["event"] == expected_name

    def test_thread_safety(self):
        """Test that queue is thread-safe."""
        import threading

        handler = SSEUIHandler()

        def queue_events():
            for i in range(100):
                handler.handle_event(UIEvent.STEP_START, {"step": i})

        # Create multiple threads
        threads = [threading.Thread(target=queue_events) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have 500 events total
        assert handler.event_queue.qsize() == 500

    def test_console_suppression(self):
        """Test that console output is suppressed."""
        handler = SSEUIHandler()

        # Handler should have a console writing to /dev/null
        assert handler.console is not None
        assert handler.show_panels is False
