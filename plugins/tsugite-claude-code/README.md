# tsugite-claude-code

Tsugite plugin that adds the `claude_code` provider, which routes LLM calls through the
`claude --print` CLI subprocess instead of a direct HTTP provider. This enables Claude Max
subscription auth and lets Claude Code manage its own session (resume, auto-compaction).

It registers under the `tsugite.providers` entry-point group as `claude_code`, so any model
string like `claude_code:opus` resolves to this provider once the plugin is installed.

Requires the [Claude Code CLI](https://github.com/anthropics/claude-code):

```
npm install -g @anthropic-ai/claude-code
```
