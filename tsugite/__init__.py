"""Tsugite: Micro-agent runner for task automation."""

import logging
import os
import sys
import warnings

__version__ = "0.1.0"

# Configure logging to stderr (keep stdout clean for piping)
_log_level = os.environ.get("TSUGITE_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stderr,
)

# Suppress LiteLLM's harmless async cleanup warning
warnings.filterwarnings("ignore", message=".*close_litellm_async_clients.*", category=RuntimeWarning)
