# ACP Integration Guide

This guide explains how to use tsugite with ACP (Agent Client Protocol) services like Claude Code.

## What is ACP?

Agent Client Protocol (ACP) is a standard protocol for communicating with AI agents over HTTP. It enables tsugite to connect to external agent services, with the primary use case being Claude Code.

**Key Resources:**
- [ACP Specification](https://agentclientprotocol.com/)
- [ACP GitHub](https://github.com/agentclientprotocol/agent-client-protocol)
- [Claude Code ACP Server](https://github.com/zed-industries/claude-code-acp)
- [Zed Documentation](https://zed.dev/docs/ai/external-agents)

## Quick Start

### 1. Set up Claude Code ACP Server

```bash
# Clone and run the claude-code-acp server
git clone https://github.com/zed-industries/claude-code-acp
cd claude-code-acp
npm install
npm start
```

The server will start on `http://localhost:8080` by default.

### 2. Create an Agent

Create a markdown file with ACP model configuration:

```markdown
---
name: my_acp_agent
model: acp:claude-code
max_steps: 10
tools: [write_file, read_file, shell_command]
---

Task: {{ user_prompt }}

Complete the above task using available tools.
```

### 3. Run the Agent

```bash
tsugite run my_agent.md "Create a hello world Python script"
```

## Model String Format

ACP models use the format: `acp:model_name[:url]`

### Examples

**Default localhost:**
```yaml
model: acp:claude-code
# Uses http://localhost:8080
```

**Explicit URL:**
```yaml
model: acp:claude-code:http://localhost:8080
```

**Custom server:**
```yaml
model: acp:claude-3-5-sonnet-20241022:http://my-server:9000
```

**Runtime override:**
```bash
tsugite run agent.md "task" --model acp:claude-code:http://custom-host:8080
```

## Configuration Options

### Server URL

You can specify the ACP server URL in multiple ways:

1. **In model string** (recommended):
   ```yaml
   model: acp:claude-code:http://localhost:8080
   ```

2. **Via command line**:
   ```bash
   tsugite run agent.md "task" --model acp:claude-code:http://localhost:9000
   ```

3. **Default**: If no URL is specified, defaults to `http://localhost:8080`

### Model Selection

The model name (middle part of the string) can be:
- A service identifier (e.g., `claude-code`)
- A specific model name (e.g., `claude-3-5-sonnet-20241022`)
- `default` to use the server's default model

The ACP server determines how to interpret the model name.

### Timeout

Default timeout is 300 seconds (5 minutes). This is suitable for long-running agent tasks.

## Using with Claude Code

Claude Code is Anthropic's official CLI for Claude. The claude-code-acp project wraps it with an ACP-compatible HTTP interface.

### Setup

1. **Install Claude Code** (if not already installed)

2. **Install and run claude-code-acp**:
   ```bash
   git clone https://github.com/zed-industries/claude-code-acp
   cd claude-code-acp
   npm install
   npm start
   ```

3. **Configure your agent**:
   ```yaml
   model: acp:claude-code
   ```

### Example Workflow

```bash
# Create a simple agent
cat > claude_helper.md << 'EOF'
---
name: claude_helper
model: acp:claude-code
max_steps: 15
tools: [write_file, read_file, shell_command]
---

Task: {{ user_prompt }}

Use available tools to complete the task.
EOF

# Run it
tsugite run claude_helper.md "Analyze the current directory and create a README"
```

## Advanced Usage

### Using Multiple Models

You can create agents that use different providers for different steps:

```markdown
---
name: multi_model_workflow
model: acp:claude-code
max_steps: 10
tools: [write_file, read_file]
---

Initial analysis...

<!-- tsu:step name="fast_analysis" assign="analysis" -->
<!-- model: ollama:qwen2.5-coder:7b -->
Quick analysis of the codebase using local model.

<!-- tsu:step name="detailed_review" assign="review" -->
<!-- model: acp:claude-code -->
Detailed review using Claude Code:
{{ analysis }}
```

### Custom ACP Servers

If you're running your own ACP-compatible service:

```yaml
model: acp:my-custom-agent:https://my-acp-server.com:443
```

### Error Handling

If the ACP server is unavailable, you'll see an error like:
```
RuntimeError: ACP request failed: Connection refused
```

Make sure:
1. The ACP server is running
2. The URL is correct
3. No firewall is blocking the connection

## Troubleshooting

### Connection Refused

**Problem**: `RuntimeError: ACP request failed: Connection refused`

**Solutions**:
- Verify the ACP server is running: `curl http://localhost:8080/health` (if health endpoint exists)
- Check the port number matches
- Ensure no firewall blocking

### Timeout Errors

**Problem**: Requests timing out after 5 minutes

**Solutions**:
- Complex tasks may need longer timeouts (modify `acp_model.py` or add timeout parameter support)
- Check if the ACP server is processing the request

### Response Format Issues

**Problem**: Unexpected response format from ACP server

**Solution**:
- The `ACPModel` class in `tsugite/acp_model.py` handles multiple response formats
- If you encounter issues, check the ACP server's response format and update `_extract_response_text()`

## API Reference

### ACPModel Class

```python
from tsugite.acp_model import ACPModel

model = ACPModel(
    server_url="http://localhost:8080",
    model_id="claude-code",
    timeout=300.0,
)
```

**Parameters**:
- `server_url` (str): Base URL of the ACP server
- `model_id` (str, optional): Model identifier to pass to server
- `timeout` (float): Request timeout in seconds (default: 300.0)

## Contributing

If you find issues with ACP integration or have suggestions:
1. Test with the latest claude-code-acp version
2. Check the ACP specification for protocol updates
3. Submit issues with example reproduction cases

## See Also

- [Agent Development Guide](CLAUDE.md)
- [Example ACP Agent](docs/examples/agents/acp_claude_code.md)
- [Model Providers Documentation](CLAUDE.md#model-providers)
