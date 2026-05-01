"""Tsugite: Micro-agent runner for task automation."""

import logging
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("tsugite-cli")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Configure logging to stderr (keep stdout clean for piping)
_log_level = os.environ.get("TSUGITE_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stderr,
)
