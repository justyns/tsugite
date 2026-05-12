# tests/e2e_smoke/

Real-LLM end-to-end smoke tests. This directory is **separate** from
`tests/e2e/` on purpose: tests here hit real provider APIs and cost money.

## Running

Auto-skips unless **both** of these are set:

- `TSUGITE_E2E_REAL_LLM=1` (explicit opt-in)
- One of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`

```bash
TSUGITE_E2E_REAL_LLM=1 OPENAI_API_KEY=$OPENAI_API_KEY \
  uv run pytest tests/e2e_smoke/ -v
```

## What's covered

Just the provider-real wiring: that the daemon, agent runtime, and frontend
can complete one full turn end-to-end against a real model. Plumbing only;
behaviour and correctness are covered by `tests/e2e/` (mocked LLM) and the
unit/integration tiers.

## Do not import these fixtures elsewhere

The conftest here intentionally **does not** install the `mock_chat` tripwire
that the main `tests/e2e/conftest.py` uses. Reusing this conftest from
another test file would silently re-enable real LLM calls.
