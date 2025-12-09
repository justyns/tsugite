"""Tests for config module."""

import json

from tsugite.config import (
    Config,
    get_model_alias,
    load_config,
    remove_model_alias,
    save_config,
    update_config,
)


def test_load_config_nonexistent(tmp_path):
    """Test loading config from nonexistent file returns default config."""
    config_path = tmp_path / "config.json"
    config = load_config(config_path)

    assert config.default_model is None
    assert config.model_aliases == {}


def test_save_and_load_config(tmp_path):
    """Test saving and loading config."""
    config_path = tmp_path / "config.json"

    config = Config(default_model="ollama:qwen2.5-coder:7b", model_aliases={"cheap": "openai:gpt-4o-mini"})

    save_config(config, config_path)

    assert config_path.exists()

    loaded_config = load_config(config_path)
    assert loaded_config.default_model == "ollama:qwen2.5-coder:7b"
    assert loaded_config.model_aliases == {"cheap": "openai:gpt-4o-mini"}


def test_set_default_model(tmp_path):
    """Test setting default model."""
    config_path = tmp_path / "config.json"

    update_config(config_path, lambda cfg: setattr(cfg, "default_model", "ollama:llama3:8b"))

    config = load_config(config_path)
    assert config.default_model == "ollama:llama3:8b"


def test_add_model_alias(tmp_path):
    """Test adding model alias."""
    config_path = tmp_path / "config.json"

    update_config(config_path, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o-mini"}))
    update_config(config_path, lambda cfg: cfg.model_aliases.update({"expensive": "openai:o1"}))

    config = load_config(config_path)
    assert config.model_aliases == {"cheap": "openai:gpt-4o-mini", "expensive": "openai:o1"}


def test_get_model_alias(tmp_path):
    """Test getting model alias."""
    config_path = tmp_path / "config.json"

    update_config(config_path, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o-mini"}))

    alias_value = get_model_alias("cheap", config_path)
    assert alias_value == "openai:gpt-4o-mini"

    nonexistent = get_model_alias("nonexistent", config_path)
    assert nonexistent is None


def test_remove_model_alias(tmp_path):
    """Test removing model alias."""
    config_path = tmp_path / "config.json"

    update_config(config_path, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o-mini"}))
    update_config(config_path, lambda cfg: cfg.model_aliases.update({"expensive": "openai:o1"}))

    removed = remove_model_alias("cheap", config_path)
    assert removed is True

    config = load_config(config_path)
    assert "cheap" not in config.model_aliases
    assert "expensive" in config.model_aliases

    removed_again = remove_model_alias("cheap", config_path)
    assert removed_again is False


def test_config_with_empty_values(tmp_path):
    """Test config with no default model and no aliases."""
    config_path = tmp_path / "config.json"

    config = Config()
    save_config(config, config_path)

    with open(config_path) as f:
        data = json.load(f)

    # Fields with default values will always be saved
    assert data == {
        "chat_theme": "gruvbox",
        "history_enabled": True,
        "auto_context_enabled": True,
        "auto_context_files": [".tsugite/CONTEXT.md", "AGENTS.md", "CLAUDE.md"],
        "auto_context_include_global": True,
        "memory_enabled": False,
        "memory_embedding_model": "BAAI/bge-small-en-v1.5",
        "memory_embedding_dimension": 384,
    }


def test_update_existing_alias(tmp_path):
    """Test updating an existing alias."""
    config_path = tmp_path / "config.json"

    update_config(config_path, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o-mini"}))
    update_config(config_path, lambda cfg: cfg.model_aliases.update({"cheap": "openai:gpt-4o"}))

    alias_value = get_model_alias("cheap", config_path)
    assert alias_value == "openai:gpt-4o"


def test_config_directory_creation(tmp_path):
    """Test that parent directories are created."""
    config_path = tmp_path / "subdir" / "nested" / "config.json"

    update_config(config_path, lambda cfg: setattr(cfg, "default_model", "ollama:qwen2.5-coder:7b"))

    assert config_path.exists()
    assert config_path.parent.exists()


def test_load_invalid_json(tmp_path):
    """Test loading config with invalid JSON."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json {")

    config = load_config(config_path)
    assert config.default_model is None
    assert config.model_aliases == {}
