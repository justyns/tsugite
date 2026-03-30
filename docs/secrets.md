# Secrets

Agents sometimes need API keys, tokens, or credentials.  The secrets system offers a more secure way to use those secrets and masks them from LLM-visible output.  i.e. your api key shouldn't be sent to the llm.

This isn't bullet-proof, but it's better than "hey, please don't print my secrets".

## Quick start

```bash
# Set a secret (prompts for value)
tsu secrets set github-token

# List stored secrets
tsu secrets list

# Delete
tsu secrets delete github-token
```

Allow the `get_secret` tool in your agent, and then it can use it like:

```python
token = get_secret("github-token")
result = http_request(
    "https://api.github.com/user",
    headers={"Authorization": f"Bearer {token}"}
)
```

The token is used in the request but shows up as `***` in logs and agent output.

## Backends

Four built-in backends.  Set the backend in `~/.config/tsugite/config.json` under `secrets.provider`.

### env (default)

Reads from environment variables.  Secret name `my-token` looks for `MY_TOKEN` (uppercased, hyphens become underscores).  Read-only.

```json
{
  "secrets": {
    "provider": "env",
    "prefix": "TSUGITE_"
  }
}
```

With `prefix`, `my-token` looks for `TSUGITE_MY_TOKEN` first.

### sqlite

sqlite-backed storage where each secret is encrypted.  NOTE: Using a key_file or env variable for the passphrase makes this not much better than plain-text secrets in files.

```json
{
  "secrets": {
    "provider": "sqlite",
    "key_file": "~/.tsugite_key"
  }
}
```

The passphrase can also come from the `TSUGITE_SECRETS_KEY` env var, or it'll prompt interactively if neither is set.  The database defaults to `~/.local/share/tsugite/secrets/secrets.db`.

### file

Stores each secret as a plaintext file in a directory.

```json
{
  "secrets": {
    "provider": "file",
    "path": "~/.local/share/tsugite/secrets"
  }
}
```

### exec

Runs a command to fetch secrets.  For integrating with external tools like pass, 1Password, or Vault.  Read-only.

```json
{
  "secrets": {
    "provider": "exec",
    "command": ["pass", "show", "{{ name }}"],
    "list_command": "pass ls"
  }
}
```

The `{{ name }}` placeholder is Jinja-templated.  Only alphanumeric names plus `-` and `_` are allowed.

A few more examples:

```json
// 1Password
{ "command": ["op", "read", "op://vault/item/{{ name }}"] }

// Vault
{ "command": "vault kv get -field=value secret/{{ name }}" }
```

## Masking

When `get_secret()` is called, the value is registered for masking.  That value is then masked in logs and agent output with `***`.  Masking happens on:

- Log records in daemon mode
- Tool output before the LLM sees it

Secret names are not masked, only values.  This isn't bullet-proof, but it's better than "hey, please don't print my secrets".

## Per-agent access control

By default, any agent with the `get_secret` tool can access any secret.  You can restrict this with `allowed_secrets` in frontmatter:

```yaml
---
name: safe-agent
tools: [get_secret, http_request, final_answer]
allowed_secrets: [public-api-key]
---
```

If `allowed_secrets` is set, the agent gets a `PermissionError` when trying to access anything not on the list.  Empty list means no restrictions.

## Backend selection priority

1. `TSUGITE_SECRETS_BACKEND` env var (overrides everything)
2. `secrets.provider` in config.json
3. Falls back to `env`

## Custom secrets backend plugins

Custom backends can be registered via the `tsugite.secrets` entry point group.  TODO: Examples