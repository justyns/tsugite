---
name: acp_claude_code_example
model: acp:claude-code
max_steps: 10
tools: [write_file, read_file]
---

# Claude Code via ACP Example

This agent demonstrates using Claude Code through the Agent Client Protocol (ACP).

## Prerequisites

Before running this agent, you need to:

1. Install and run the claude-code-acp server:
   ```bash
   # Clone the repository
   git clone https://github.com/zed-industries/claude-code-acp
   cd claude-code-acp

   # Install dependencies and run
   npm install
   npm start
   ```

2. The server will run on http://localhost:8080 by default

## Usage

Run this agent with:
```bash
tsugite run docs/examples/agents/acp_claude_code.md "your task here"
```

Or with a custom server URL:
```bash
tsugite run docs/examples/agents/acp_claude_code.md "your task" \
  --model acp:claude-code:http://localhost:9000
```

## Task

{{ user_prompt }}

Please complete the task above using the available tools.
