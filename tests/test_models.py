"""Tests for models module."""

import pytest

from tsugite.acp_model import ACPModel
from tsugite.config import Config, add_model_alias
from tsugite.models import get_model, parse_model_string, resolve_model_alias


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
    Config().save_config = lambda path: None

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    resolved = resolve_model_alias("nonexistent")
    assert resolved == "nonexistent"


def test_parse_model_string_acp():
    """Test parsing ACP model strings."""
    # Simple ACP model
    provider, model, variant = parse_model_string("acp:claude-code")
    assert provider == "acp"
    assert model == "claude-code"
    assert variant is None

    # ACP model with URL
    provider, model, variant = parse_model_string("acp:claude-code:http://localhost:8080")
    assert provider == "acp"
    assert model == "claude-code"
    assert variant == "http://localhost:8080"

    # ACP model with custom server
    provider, model, variant = parse_model_string("acp:claude-3-5-sonnet-20241022:http://custom-server:9000")
    assert provider == "acp"
    assert model == "claude-3-5-sonnet-20241022"
    assert variant == "http://custom-server:9000"


def test_get_model_acp():
    """Test creating ACP model."""
    # Test default ACP model
    model = get_model("acp:claude-code")
    assert isinstance(model, ACPModel)
    assert model.server_url == "http://localhost:8080"
    assert model.model_id == "claude-code"

    # Test ACP model with custom URL
    model = get_model("acp:claude-code:http://localhost:9000")
    assert isinstance(model, ACPModel)
    assert model.server_url == "http://localhost:9000"
    assert model.model_id == "claude-code"

    # Test ACP model with explicit server_url parameter
    model = get_model("acp:custom-model", server_url="http://example.com:8080")
    assert isinstance(model, ACPModel)
    assert model.server_url == "http://example.com:8080"
    assert model.model_id == "custom-model"


def test_acp_model_initialization():
    """Test ACPModel initialization."""
    model = ACPModel(server_url="http://localhost:8080", model_id="test-model")
    assert model.server_url == "http://localhost:8080"
    assert model.model_id == "test-model"
    assert model.timeout == 300.0

    # Test with custom timeout
    model = ACPModel(server_url="http://localhost:8080", timeout=60.0)
    assert model.timeout == 60.0

    # Test URL normalization (trailing slash removal)
    model = ACPModel(server_url="http://localhost:8080/")
    assert model.server_url == "http://localhost:8080"
