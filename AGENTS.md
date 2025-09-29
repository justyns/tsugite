# Agent Development Guide

## Build/Test Commands

- **Install dev dependencies**: `uv sync --dev`
- Use `uv add` and related commands to manage dependencies
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_function_name`
- **Run specific test file**: `uv run pytest tests/test_cli.py`
- **Lint code**: `uv run ruff check .`
- **Format code**: `uv run black .`
- **Type check**: No mypy configured; use ruff for basic checks

## Code Style Guidelines

- **Line length**: 120 characters (black + ruff configured)
- **Target Python**: 3.12+
- **Imports**: Use absolute imports, group stdlib/third-party/local with blank lines
- **Type hints**: Use for function signatures, especially public APIs
- **Error handling**: Use specific exceptions (ValueError, RuntimeError), include context
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Use triple quotes with Args/Returns sections for public functions
- **String formatting**: Use f-strings for simple interpolation

## Project Structure

- Main CLI: `tsugite/tsugite.py` (typer-based)
- Models: `tsugite/models.py` (smolagents integration)
- Tools: `tsugite/tools/` directory
- Tests: `tests/` with pytest fixtures
- No comments unless they add meaningful context
- Write new unit tests for all new features

## Frontmatter Reference

- `name` *(required)* – Agent identifier shown in CLI.
- `model` *(optional)* – Defaults to `ollama:qwen2.5-coder:7b`.
- `max_steps` *(optional)* – Defaults to `5`.
- `tools` *(optional list)* – Tool names registered via `@tool`.
- `prefetch` *(optional list)* – Tool calls to run before rendering.
- `permissions_profile` *(optional)* – Placeholder for future permissions engine.
- `instructions` *(optional string)* – Additional system guidance appended to Tsugite’s default runtime instructions before hitting the LLM.
