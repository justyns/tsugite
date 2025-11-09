"""Tsugite: Micro-agent runner for task automation."""

import warnings

__version__ = "0.1.0"

# Suppress LiteLLM's harmless async cleanup warning
warnings.filterwarnings("ignore", message=".*close_litellm_async_clients.*", category=RuntimeWarning)
