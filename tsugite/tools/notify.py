"""Notification tools for scheduled agent tasks."""

import asyncio
import logging
import threading
from contextlib import contextmanager

from . import tool

logger = logging.getLogger(__name__)

_notifier = None
_loop = None


def set_notifier(callback, loop=None):
    """Called by the daemon gateway to set/clear the notification callback.

    Args:
        callback: Async function(message, channel_configs) -> dict, or None to clear
        loop: Event loop the callback runs on
    """
    global _notifier, _loop
    _notifier = callback
    _loop = loop


_local = threading.local()


@contextmanager
def notify_context(channel_configs):
    """Set notification channels for the current thread's agent run.

    Args:
        channel_configs: List of (name, NotificationChannelConfig) tuples
    """
    _local.channels = channel_configs
    try:
        yield
    finally:
        _local.channels = None


def send_notification(message: str, channel_configs: list) -> dict:
    """Send a notification to channels (thread-safe, callable from any thread)."""
    if not _notifier or not _loop:
        return {"error": "Notifier not configured"}

    future = asyncio.run_coroutine_threadsafe(_notifier(message, channel_configs), _loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        logger.error("Notification dispatch failed: %s", e)
        return {"error": str(e)}


@tool(require_daemon=True)
def notify_user(message: str) -> dict:
    """Send a notification message to the user via configured channels.

    Use this to proactively notify the user about important findings, progress updates,
    or alerts during scheduled task execution. Only available when the schedule has
    notify_tool enabled.

    Args:
        message: The notification message to send

    Returns:
        Dict with delivery status per channel
    """
    channels = getattr(_local, "channels", None)
    if not channels:
        return {"error": "No notification channels configured for this run"}

    return send_notification(message, channels)
