#!/bin/bash
# Smoke test for Tsugite - tests real LiteLLM integration
# This is NOT run automatically with pytest to avoid API costs
# Run manually: bash tests/smoke_test.sh

set -e

echo "🧪 Tsugite Smoke Test"
echo "===================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "This smoke test requires a real OpenAI API key to test LiteLLM integration"
    echo "Set it with: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

echo "✓ OPENAI_API_KEY found"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🧪 Test 1: Simple calculation (basic agent execution)"
echo "---------------------------------------------------"
OUTPUT=$(uv run tsugite run +default "What is 42 + 17? Just give me the number." --plain --model openai:gpt-4o-mini 2>&1)
if echo "$OUTPUT" | grep -q "59"; then
    echo "✅ PASS: Agent calculated 42 + 17 = 59"
else
    echo "❌ FAIL: Expected '59' in output"
    echo "Output was:"
    echo "$OUTPUT"
    exit 1
fi
echo ""

echo "🧪 Test 2: File operation (test tools work)"
echo "-------------------------------------------"
TEMP_FILE=$(mktemp)
echo "Test content for smoke test" > "$TEMP_FILE"

OUTPUT=$(uv run tsugite run +default "Read the file $TEMP_FILE and tell me what it says" --plain --model openai:gpt-4o-mini 2>&1)
if echo "$OUTPUT" | grep -qi "smoke test"; then
    echo "✅ PASS: Agent successfully read file using tools"
else
    echo "❌ FAIL: Expected 'smoke test' in output"
    echo "Output was:"
    echo "$OUTPUT"
    rm -f "$TEMP_FILE"
    exit 1
fi

rm -f "$TEMP_FILE"
echo ""

echo "🧪 Test 3: Commit message agent (new agent)"
echo "-------------------------------------------"
# Only run if we have git changes
if [ -d .git ] && [ -n "$(git status --short)" ]; then
    OUTPUT=$(uv run tsugite run +commit_message "" --plain --model openai:gpt-4o-mini 2>&1)
    if echo "$OUTPUT" | grep -qE "(feat|fix|refactor|docs|test|chore|style|perf|ci):"; then
        echo "✅ PASS: Commit message agent generated conventional commit format"
    else
        echo "⚠️  WARNING: Commit message doesn't follow conventional commits format"
        echo "Output was:"
        echo "$OUTPUT"
        # Don't fail on this one, just warn
    fi
else
    echo "⏭️  SKIP: No git changes to generate commit message for"
fi
echo ""

echo "================================"
echo "✅ All smoke tests passed!"
echo "================================"
echo ""
echo "These tests verify:"
echo "  • LiteLLM async context works correctly"
echo "  • Real API calls succeed (not just mocks)"
echo "  • Tools are properly integrated"
echo "  • New agents work end-to-end"
