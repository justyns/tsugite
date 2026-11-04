# tsugite-cursor

Tsugite provider plugin that runs Cursor models (`cursor:<model>`) through the
[Cursor Python SDK](https://cursor.com/docs/sdk/python) with Cursor's built-in tools turned off.

## Setup

Create a user API key in the Cursor dashboard and set `CURSOR_API_KEY`, or store it in tsugite's secrets backend as
`cursor-api-key`. The SDK needs a paid Cursor plan.

## Usage

```bash
uv run tsu run my-agent.md "say hi" --model cursor:composer-2.5
```

Model ids come from your account (`tsu models list --provider cursor`). `auto` maps to Cursor's `default`, which
some plans do not include. A model without parameters runs its catalog default variant. Parameters use Cursor's
bracket form: `cursor:grok-4.7[effort=high]`. A `context=` value such as `1m` or `400k`
also sets tsugite's context window for that model. Without one, tsugite assumes 200k.

The SDK has no system-prompt option, so tsugite's system prompt goes in a `<tsugite_system_prompt>` block at the start
of the first message. Resumed sessions do not repeat it.
