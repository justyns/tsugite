"""Helper functions for event emission."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def emit_file_read_event(path: str, content: str, operation: str) -> None:
    """Emit FileReadEvent if event bus is available.

    Args:
        path: Path to the file that was read
        content: Content that was read from the file
        operation: Type of operation ("attachment", "auto_context", "tool_call", "prefetch")
    """
    from tsugite.ui_context import get_event_bus

    event_bus = get_event_bus()
    if event_bus:
        from tsugite.events import FileReadEvent

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        byte_count = len(content.encode("utf-8"))
        event_bus.emit(FileReadEvent(path=path, line_count=line_count, byte_count=byte_count, operation=operation))


def emit_info_event(message: str) -> None:
    """Emit InfoEvent if event bus is available.

    Args:
        message: Info message to emit
    """
    from tsugite.ui_context import get_event_bus

    event_bus = get_event_bus()
    if event_bus:
        from tsugite.events import InfoEvent

        event_bus.emit(InfoEvent(message=message))


def emit_skill_loaded_event(skill_name: str, description: str) -> None:
    """Emit SkillLoadedEvent if event bus is available.

    Args:
        skill_name: Name of the skill that was loaded
        description: Description of the skill
    """
    from tsugite.ui_context import get_event_bus

    event_bus = get_event_bus()
    if event_bus:
        from tsugite.events import SkillLoadedEvent

        event_bus.emit(SkillLoadedEvent(skill_name=skill_name, description=description))


def emit_skill_unloaded_event(skill_name: str) -> None:
    """Emit SkillUnloadedEvent if event bus is available.

    Args:
        skill_name: Name of the skill that was unloaded
    """
    from tsugite.ui_context import get_event_bus

    event_bus = get_event_bus()
    if event_bus:
        from tsugite.events import SkillUnloadedEvent

        event_bus.emit(SkillUnloadedEvent(skill_name=skill_name))
