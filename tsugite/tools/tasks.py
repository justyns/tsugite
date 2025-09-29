"""Task tracking tools for agents to manage work across execution steps."""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from . import tool


class TaskStatus(Enum):
    """Valid status values for tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Individual task with metadata."""

    id: int
    title: str
    status: TaskStatus
    parent_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class TaskManager:
    """Manages tasks during a single agent execution session."""

    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self._next_id = 1

    def add_task(self, title: str, status: TaskStatus = TaskStatus.PENDING, parent_id: Optional[int] = None) -> int:
        """Add a new task and return its ID."""
        task_id = self._next_id
        self._next_id += 1

        if parent_id is not None and parent_id not in self.tasks:
            raise ValueError(f"Parent task {parent_id} does not exist")

        task = Task(id=task_id, title=title, status=status, parent_id=parent_id)

        self.tasks[task_id] = task
        return task_id

    def update_task(self, task_id: int, status: TaskStatus) -> None:
        """Update a task's status."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")

        task = self.tasks[task_id]
        task.status = status
        task.updated_at = datetime.now()

        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()

    def get_task(self, task_id: int) -> Task:
        """Get a specific task by ID."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")
        return self.tasks[task_id]

    def list_tasks(self, status: Optional[TaskStatus] = None, parent_id: Optional[int] = None) -> List[Task]:
        """List tasks with optional filtering."""
        tasks = list(self.tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if parent_id is not None:
            tasks = [t for t in tasks if t.parent_id == parent_id]

        # Sort by creation time
        return sorted(tasks, key=lambda t: t.created_at)

    def get_task_summary(self) -> str:
        """Generate a formatted summary of all tasks for agent context."""
        if not self.tasks:
            return "## Current Tasks\nNo tasks yet."

        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])

        summary = [f"## Current Tasks ({completed_tasks} completed / {total_tasks} total)\n"]

        # Group tasks by status
        status_groups = {
            "Active Tasks": [TaskStatus.IN_PROGRESS, TaskStatus.PENDING],
            "Blocked Tasks": [TaskStatus.BLOCKED],
            "Completed Tasks": [TaskStatus.COMPLETED],
            "Cancelled Tasks": [TaskStatus.CANCELLED],
        }

        status_icons = {
            TaskStatus.PENDING: "⏸️",
            TaskStatus.IN_PROGRESS: "⏳",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.CANCELLED: "❌",
        }

        for group_name, statuses in status_groups.items():
            group_tasks = [t for t in self.tasks.values() if t.status in statuses]
            if not group_tasks:
                continue

            summary.append(f"### {group_name}")

            # Show parent tasks first, then their subtasks indented
            parent_tasks = [t for t in group_tasks if t.parent_id is None]
            parent_tasks.sort(key=lambda t: t.id)

            for parent in parent_tasks:
                icon = status_icons[parent.status]
                summary.append(f"[{parent.id}] {icon} {parent.title}")

                # Show subtasks indented
                subtasks = [t for t in group_tasks if t.parent_id == parent.id]
                subtasks.sort(key=lambda t: t.id)
                for subtask in subtasks:
                    sub_icon = status_icons[subtask.status]
                    summary.append(f"  └─ [{subtask.id}] {sub_icon} {subtask.title}")

            # Show orphaned subtasks (parent not in this group)
            orphaned = [
                t for t in group_tasks if t.parent_id is not None and t.parent_id not in [p.id for p in parent_tasks]
            ]
            for orphan in orphaned:
                icon = status_icons[orphan.status]
                summary.append(f"[{orphan.id}] {icon} {orphan.title} (subtask of #{orphan.parent_id})")

            summary.append("")

        summary.append("Status: ⏸️ pending | ⏳ in_progress | ✅ completed | 🚫 blocked | ❌ cancelled")

        return "\n".join(summary)


# Global task manager instance for the current agent session
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the current task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def reset_task_manager() -> None:
    """Reset the task manager (used at start of new agent session)."""
    global _task_manager
    _task_manager = TaskManager()


@tool
def task_add(title: str, status: str = "pending", parent_id: Optional[int] = None) -> int:
    """Add a new task or subtask.

    Args:
        title: Description of the task
        status: Task status (pending/in_progress/completed/blocked/cancelled)
        parent_id: ID of parent task if this is a subtask

    Returns:
        ID of the created task
    """
    try:
        task_status = TaskStatus(status)
    except ValueError:
        valid_statuses = [s.value for s in TaskStatus]
        raise ValueError(f"Invalid status '{status}'. Valid options: {valid_statuses}")

    manager = get_task_manager()
    return manager.add_task(title, task_status, parent_id)


@tool
def task_update(task_id: int, status: str) -> None:
    """Update a task's status.

    Args:
        task_id: ID of the task to update
        status: New status (pending/in_progress/completed/blocked/cancelled)
    """
    try:
        task_status = TaskStatus(status)
    except ValueError:
        valid_statuses = [s.value for s in TaskStatus]
        raise ValueError(f"Invalid status '{status}'. Valid options: {valid_statuses}")

    manager = get_task_manager()
    manager.update_task(task_id, task_status)


@tool
def task_complete(task_id: int) -> None:
    """Mark a task as completed (shortcut for task_update).

    Args:
        task_id: ID of the task to mark as completed
    """
    manager = get_task_manager()
    manager.update_task(task_id, TaskStatus.COMPLETED)


@tool
def task_list(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List tasks with optional status filtering.

    Args:
        status: Optional status filter (pending/in_progress/completed/blocked/cancelled)

    Returns:
        List of task dictionaries with id, title, status, parent_id
    """
    status_filter = None
    if status is not None:
        try:
            status_filter = TaskStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in TaskStatus]
            raise ValueError(f"Invalid status '{status}'. Valid options: {valid_statuses}")

    manager = get_task_manager()
    tasks = manager.list_tasks(status_filter)

    return [
        {
            "id": task.id,
            "title": task.title,
            "status": task.status.value,
            "parent_id": task.parent_id,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
        for task in tasks
    ]


@tool
def task_get(task_id: int) -> Dict[str, Any]:
    """Get details for a specific task.

    Args:
        task_id: ID of the task to retrieve

    Returns:
        Task dictionary with all details
    """
    manager = get_task_manager()
    task = manager.get_task(task_id)

    return {
        "id": task.id,
        "title": task.title,
        "status": task.status.value,
        "parent_id": task.parent_id,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
