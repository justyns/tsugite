# Tsugite Tests

This directory contains the test suite for Tsugite.

## Running Tests

### Unit Tests

Run the full test suite:

```bash
uv run pytest
```

Run specific test file:

```bash
uv run pytest tests/test_agent.py
```

Run with coverage:

```bash
uv run pytest --cov=tsugite --cov-report=html
```

### Smoke Tests

**Smoke tests are NOT run automatically** to avoid API costs. They test real LiteLLM integration with actual API calls.

#### Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
```

#### Running Smoke Tests

```bash
bash tests/smoke_test.sh
```

#### What Smoke Tests Verify

1. **Real LiteLLM integration** - Tests actual API calls, not mocks
2. **Async context detection** - Catches issues like missing asyncio imports
3. **Tool integration** - Verifies tools work in real execution
4. **New agents** - Tests recently added agents end-to-end

#### When to Run Smoke Tests

- Before releases
- After upgrading LiteLLM
- After major refactors to agent execution
- When debugging integration issues
- After adding new agents

## Test Structure

### Unit Tests (`tests/`)

- **`core/`** - Tests for core agent implementation (TsugiteAgent, executor, tools)
- **`test_*.py`** - Feature-specific tests (CLI, rendering, parsing, etc.)
- All unit tests mock LiteLLM to avoid API calls

### Integration Tests

Currently implemented as smoke tests (see above).

## Writing Tests

### Mocking LiteLLM

When writing unit tests that use `TsugiteAgent`:

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_example(mock_litellm_response):
    with patch("tsugite.core.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=mock_litellm_response("test response")
        )

        agent = TsugiteAgent(...)
        result = await agent.run("test task")

        assert result == expected
```

### Testing Agents

For testing agent behavior without API calls:

```python
# Test agent parsing
from tsugite.md_agents import parse_agent_file
agent_config, content = parse_agent_file(agent_path)

# Test rendering
from tsugite.renderer import render_prompt
rendered = render_prompt(content, context={"user_prompt": "test"})
```

## Coverage

Aim for:
- 80%+ overall coverage
- 90%+ for critical paths (agent execution, tool handling)
- 100% for security-critical code (file operations, shell execution)

View coverage report:

```bash
uv run pytest --cov=tsugite --cov-report=html
open htmlcov/index.html
```

## CI/CD

Tests run automatically on:
- Push to any branch
- Pull requests
- Before releases

Note: Smoke tests do NOT run in CI to avoid API costs.
