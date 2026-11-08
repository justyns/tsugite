# tsugite-mcp

Connect remote MCP servers (Streamable HTTP) to the tsugite daemon, with OAuth sign-in.

## Install

`uv sync` installs it with the workspace. It stays inactive until you turn it on in `daemon.yaml`:

```yaml
plugins:
  mcp:
    enabled: true
```

## Use

Open **Connectors** in the web UI, enter a name (lowercase letters and digits) and the server's URL,
and press Add. A server that needs OAuth shows **Sign in**, which opens its authorization page. The
callback is `<daemon origin>/api/plugins/mcp/callback`, so the daemon must be reachable at the origin
your browser uses. A server that only accepts loopback redirects shows a field instead, where you paste the address
the browser lands on after you approve (the desktop app does this itself).

Each tool of a connected server is a tsugite tool named `mcp_<name>_<tool>` in the category
`mcp_<name>`. Enable one server for an agent with `tools: ["@mcp_notion"]`, or pick tools by name or glob.
