"""Tests for task tracking functionality."""

from datetime import datetime

import pytest

from tsugite.tools.tasks import (
    TaskManager,
    TaskStatus,
    get_task_manager,
    reset_task_manager,
    task_add,
    task_complete,
    task_get,
    task_list,
    task_update,
)


class TestTaskManager:
    """Test the TaskManager class."""

    def setup_method(self):
        """Set up fresh task manager for each test."""
        self.manager = TaskManager()

    def test_add_task(self):
        """Test adding a new task."""
        task_id = self.manager.add_task("Test task")
        assert task_id == 1

        task = self.manager.get_task(task_id)
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.parent_id is None

    def test_add_subtask(self):
        """Test adding a subtask with a parent."""
        parent_id = self.manager.add_task("Parent task")
        child_id = self.manager.add_task("Child task", parent_id=parent_id)

        child_task = self.manager.get_task(child_id)
        assert child_task.parent_id == parent_id

    def test_add_subtask_invalid_parent(self):
        """Test adding subtask with non-existent parent fails."""
        with pytest.raises(ValueError, match="Parent task 999 does not exist"):
            self.manager.add_task("Child task", parent_id=999)

    def test_update_task(self):
        """Test updating task status."""
        task_id = self.manager.add_task("Test task")

        self.manager.update_task(task_id, TaskStatus.IN_PROGRESS)
        task = self.manager.get_task(task_id)
        assert task.status == TaskStatus.IN_PROGRESS

    def test_update_nonexistent_task(self):
        """Test updating non-existent task fails."""
        with pytest.raises(ValueError, match="Task 999 does not exist"):
            self.manager.update_task(999, TaskStatus.COMPLETED)

    def test_complete_task_sets_completed_at(self):
        """Test that completing a task sets completed_at timestamp."""
        task_id = self.manager.add_task("Test task")

        before_complete = datetime.now()
        self.manager.update_task(task_id, TaskStatus.COMPLETED)
        after_complete = datetime.now()

        task = self.manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert before_complete <= task.completed_at <= after_complete

    def test_list_tasks_no_filter(self):
        """Test listing all tasks."""
        self.manager.add_task("Task 1")
        self.manager.add_task("Task 2", TaskStatus.IN_PROGRESS)

        tasks = self.manager.list_tasks()
        assert len(tasks) == 2
        assert [t.title for t in tasks] == ["Task 1", "Task 2"]

    def test_list_tasks_with_status_filter(self):
        """Test listing tasks filtered by status."""
        self.manager.add_task("Pending task")
        self.manager.add_task("In progress task", TaskStatus.IN_PROGRESS)

        pending_tasks = self.manager.list_tasks(TaskStatus.PENDING)
        assert len(pending_tasks) == 1
        assert pending_tasks[0].title == "Pending task"

    def test_get_task_summary_empty(self):
        """Test task summary with no tasks."""
        summary = self.manager.get_task_summary()
        assert "No tasks yet" in summary

    def test_get_task_summary_with_tasks(self):
        """Test task summary with various tasks."""
        parent_id = self.manager.add_task("Parent task")
        self.manager.add_task("Child task", parent_id=parent_id)
        self.manager.add_task("Completed task", TaskStatus.COMPLETED)

        summary = self.manager.get_task_summary()
        assert "Current Tasks (1 completed / 3 total)" in summary
        assert "⏸️" in summary  # pending icon
        assert "✅" in summary  # completed icon


class TestTaskTools:
    """Test the task tool functions."""

    def setup_method(self):
        """Reset task manager for each test."""
        reset_task_manager()

    def test_task_add_tool(self):
        """Test task_add tool function."""
        task_id = task_add("Test task")
        assert task_id == 1

        # Verify the task was created
        manager = get_task_manager()
        task = manager.get_task(task_id)
        assert task.title == "Test task"

    def test_task_add_with_parent_id_zero(self):
        """Test that parent_id=0 is treated as None (root task)."""
        task_id = task_add("Root task", parent_id=0)
        assert task_id == 1

        # Verify the task was created as a root task (parent_id=None)
        manager = get_task_manager()
        task = manager.get_task(task_id)
        assert task.title == "Root task"
        assert task.parent_id is None

    def test_task_add_with_invalid_status(self):
        """Test task_add with invalid status."""
        with pytest.raises(ValueError, match="Invalid status 'invalid'"):
            task_add("Test task", status="invalid")

    def test_task_update_tool(self):
        """Test task_update tool function."""
        task_id = task_add("Test task")
        task_update(task_id, "in_progress")

        manager = get_task_manager()
        task = manager.get_task(task_id)
        assert task.status == TaskStatus.IN_PROGRESS

    def test_task_complete_tool(self):
        """Test task_complete tool function."""
        task_id = task_add("Test task")
        task_complete(task_id)

        manager = get_task_manager()
        task = manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED

    def test_task_list_tool(self):
        """Test task_list tool function."""
        task_add("Task 1")
        task_add("Task 2", status="completed")

        all_tasks = task_list()
        assert len(all_tasks) == 2

        completed_tasks = task_list(status="completed")
        assert len(completed_tasks) == 1
        assert completed_tasks[0]["title"] == "Task 2"

    def test_task_get_tool(self):
        """Test task_get tool function."""
        task_id = task_add("Test task")
        task_data = task_get(task_id)

        assert task_data["id"] == task_id
        assert task_data["title"] == "Test task"
        assert task_data["status"] == "pending"

    def test_subtask_creation(self):
        """Test creating subtasks through tools."""
        parent_id = task_add("Parent task")
        child_id = task_add("Child task", parent_id=parent_id)

        child_data = task_get(child_id)
        assert child_data["parent_id"] == parent_id


class TestTaskStatusEnum:
    """Test the TaskStatus enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        expected_values = {"pending", "in_progress", "completed", "blocked", "cancelled"}
        actual_values = {status.value for status in TaskStatus}
        assert actual_values == expected_values

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("in_progress") == TaskStatus.IN_PROGRESS
        assert TaskStatus("completed") == TaskStatus.COMPLETED
        assert TaskStatus("blocked") == TaskStatus.BLOCKED
        assert TaskStatus("cancelled") == TaskStatus.CANCELLED

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            TaskStatus("invalid_status")
