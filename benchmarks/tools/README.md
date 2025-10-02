# Tool Calling Benchmarks

These benchmarks test a model's ability to correctly use CodeAgent Python blocks to call tools on the first try.

## Benchmarks

### file_read_tool
Tests basic file reading capability:
- Can the model call `read_file(path)` correctly?
- Can it pass the right file path?
- Does it complete in max_steps: 2?

### file_write_tool
Tests basic file writing capability:
- Can the model call `write_file(path, content)` correctly?
- Can it pass both path and content arguments?
- Does it report success?

### file_write_read_tool
Tests chaining multiple tool calls:
- Can the model write a file then read it back?
- Does it correctly sequence the operations?
- Does it complete in max_steps: 3?

### shell_command_tool
Tests shell command execution:
- Can the model call `run_shell_command(command)` correctly?
- Can it handle different command types?
- Does it capture and return output?

## Test Fixtures

The `fixtures/` directory contains test files used by the file_read_tool benchmark:
- `sample.txt` - Simple text file
- `data.txt` - Multi-line data file
- `config.json` - JSON configuration file

## Running Tool Benchmarks

Test all tool benchmarks:
```bash
uv run tsugite benchmark run --models "ollama:qwen2.5-coder:7b" --categories "tools"
```

Test a specific tool benchmark:
```bash
uv run tsugite benchmark run --models "ollama:qwen2.5-coder:7b" --filter "file_read_tool"
```

## What This Tests

These benchmarks specifically verify:
1. **Python code generation** - Can the model generate valid Python code?
2. **Tool syntax** - Does it use the correct function names and signatures?
3. **Argument passing** - Can it pass the right arguments in the right order?
4. **First-try capability** - Does it succeed in minimal steps (max_steps: 2-3)?

This reveals which models understand the CodeAgent format vs which struggle with the Python-based tool calling approach.
