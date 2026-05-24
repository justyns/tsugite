# tsugite-codex-cli

Tsugite provider plugin that consumes a ChatGPT Plus/Pro subscription quota by
reusing the OAuth tokens written by OpenAI's [`codex`
CLI](https://developers.openai.com/codex). Requests go to the Codex Responses
API at `https://chatgpt.com/backend-api/codex/responses` with a refreshed bearer
token sourced from `~/.codex/auth.json`.

This avoids the "double-pay" problem: ChatGPT Plus/Pro already includes Codex
quota, but tsugite's stock OpenAI provider authenticates against the separately
billed Platform API.

## Install

UV workspace member of the tsugite repo, so a top-level

```bash
uv sync --all-extras
```

picks it up. Standalone install (e.g. into another tsugite venv) works too:

```bash
uv add tsugite-codex-cli
```

## Prerequisite: `codex login`

Initial auth always flows through the Codex CLI. Install it from the
[Codex docs](https://developers.openai.com/codex) and run

```bash
codex login
```

once. Choose the ChatGPT sign-in (not API key). The CLI writes
`~/.codex/auth.json` with `"auth_mode": "chatgpt"`. The plugin reads from that
same file and refreshes the access token in place whenever it nears expiry.

Override the location with `CODEX_HOME=/some/other/dir`.

## Usage

Pick the provider via the model string:

```bash
uv run tsu run examples/simple_variable_injection.md "say hi" --model codex_cli:gpt-5.4
```

Available models: `codex_cli:gpt-5.4`, `codex_cli:gpt-5.4-mini`,
`codex_cli:gpt-5.4-nano`. Use `provider.list_models()` to discover what your
ChatGPT plan currently exposes; the plugin queries the Codex `/models` endpoint
and falls back to the three above if that call fails.

## Token-sink caveat

OAuth refresh tokens rotate. If `codex` CLI and tsugite are both refreshing the
same `~/.codex/auth.json` concurrently, an old refresh token can get invalidated
upstream and the next refresh attempt will return `invalid_grant`. When that
happens, the plugin raises `CodexAuthError` with a "Run `codex login` again"
message - just re-run the CLI login.

The plugin uses a sidecar file lock (`auth.json.lock`) to serialise refreshes
between tsugite processes. It does **not** coordinate with the `codex` CLI's
own lock convention (not audited), so cross-tool refresh races are still
possible.

## Cost reporting

Calls are subscription-billed, so `Usage.cost` is always `0.0`. Token counts
populate normally so the per-turn budget banner still works.
