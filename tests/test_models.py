"""Tests for models module."""

from pathlib import Path

import pytest

from tsugite.config import add_model_alias, save_config, Config
from tsugite.models import parse_model_string, resolve_model_alias


def test_parse_model_string():
    """Test parsing model strings."""
    provider, model, variant = parse_model_string("ollama:qwen2.5-coder:7b")
    assert provider == "ollama"
    assert model == "qwen2.5-coder"
    assert variant == "7b"

    provider, model, variant = parse_model_string("openai:gpt-4")
    assert provider == "openai"
    assert model == "gpt-4"
    assert variant is None


def test_parse_model_string_invalid():
    """Test parsing invalid model strings."""
    with pytest.raises(ValueError):
        parse_model_string("invalid")


def test_resolve_model_alias_with_full_string(tmp_path):
    """Test that full model strings pass through unchanged."""
    config_path = tmp_path / "config.json"
    add_model_alias("cheap", "openai:gpt-4o-mini", config_path)

    full_model = "ollama:qwen2.5-coder:7b"
    resolved = resolve_model_alias(full_model)
    assert resolved == full_model


def test_resolve_model_alias_with_alias(tmp_path, monkeypatch):
    """Test resolving an alias to its model string."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    config_path = tmp_path / "tsugite" / "config.json"
    add_model_alias("cheap", "openai:gpt-4o-mini", config_path)

    resolved = resolve_model_alias("cheap")
    assert resolved == "openai:gpt-4o-mini"


def test_resolve_model_alias_nonexistent(tmp_path, monkeypatch):
    """Test resolving nonexistent alias returns original string."""
    config_path = tmp_path / "config.json"
    Config().save_config = lambda path: None

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    resolved = resolve_model_alias("nonexistent")
    assert resolved == "nonexistent"
