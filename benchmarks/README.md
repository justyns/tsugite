# Tsugite Benchmark Framework

A comprehensive benchmarking system for evaluating language models against a standardized suite of agent templates and tasks.

## Overview

The benchmark framework provides:
- **Standardized test suites** across multiple categories
- **Multi-model comparison** with detailed metrics
- **Comprehensive evaluation** including correctness, performance, quality, and cost
- **Rich reporting** in multiple formats (JSON, Markdown, HTML, CSV)
- **Extensible architecture** for adding new tests and evaluators

## Quick Start

### Running Basic Benchmarks

```bash
# Test a single model on basic tasks
tsugite benchmark run --models "ollama:qwen2.5-coder:7b" --categories "basic"

# Compare multiple models
tsugite benchmark run --models "ollama:qwen2.5-coder:7b,openai:gpt-4o-mini" --categories "basic,tools"

# Full benchmark suite
tsugite benchmark run --models "ollama:qwen2.5-coder:7b,openai:gpt-4o-mini" --format "html"
```

### Example Output

```
Running benchmarks...
Models: ollama:qwen2.5-coder:7b, openai:gpt-4o-mini
Categories: basic, tools, scenarios, performance

Evaluating model: ollama:qwen2.5-coder:7b
  Running test: basic_001
  Running test: basic_002
  Running test: tools_001
  ...

==================================================
Benchmark Complete
==================================================
Duration: 45.2s
Models: 2
Tests: 8
Average Accuracy: 87.5%
Best Model: openai:gpt-4o-mini
```

## Test Categories

### Basic Tests (`benchmarks/basic/`)
Foundation tests for core functionality:

- **hello_world.md**: Simple output test
- **math_operations.md**: Arithmetic calculations
- **string_manipulation.md**: Text processing
- **conditional_logic.md**: Logic evaluation

### Tool Tests (`benchmarks/tools/`)
Tests for tool usage and orchestration:

- **file_operations.md**: File read/write operations
- **command_execution.md**: System command execution

### Scenario Tests (`benchmarks/scenarios/`)
Complex real-world scenarios:

- **data_analysis.md**: CSV data analysis with insights
- **code_generation.md**: Python code generation and testing

### Performance Tests (`benchmarks/performance/`)
Efficiency and speed tests:

- **token_efficiency.md**: Minimal token usage
- **speed_test.md**: Response time optimization

## Creating Custom Tests

### Test Agent Format

```markdown
---
name: my_test
test_id: custom_001
description: "Custom test description"
model: "{{ model }}"  # Will be replaced by framework
timeout: 30
expected_output: "Expected Result"
expected_type: "string"  # string, number, json, code
weight: 1.0
evaluation_criteria:
  accuracy:
    type: "keyword"
    keywords: ["Expected", "Result"]
    weight: 1.0
---

# Task
Your test instructions here.

Output should be: Expected Result
```

### Evaluation Criteria Types

#### Keyword Matching
```yaml
evaluation_criteria:
  completeness:
    type: "keyword"
    keywords: ["complete", "done", "finished"]
    weight: 0.5
```

#### Length Requirements
```yaml
evaluation_criteria:
  appropriate_length:
    type: "length"
    min_length: 50
    max_length: 200
    optimal_length: 100
    weight: 0.3
```

#### Format Validation
```yaml
evaluation_criteria:
  json_format:
    type: "format"
    format: "json"  # json, code, markdown
    weight: 0.4
```

#### Sentiment Analysis
```yaml
evaluation_criteria:
  positive_tone:
    type: "sentiment"
    sentiment: "positive"  # positive, negative, neutral
    weight: 0.2
```

## CLI Commands

### Run Benchmarks

```bash
tsugite benchmark run [OPTIONS]
```

**Options:**
- `--models`: Comma-separated list of models to test
- `--categories`: Test categories to run (default: all)
- `--format`: Report format (json, markdown, html, csv)
- `--output`: Custom output file path
- `--filter`: Filter tests by name/ID
- `--parallel/--sequential`: Execution mode
- `--repeat`: Number of test repetitions

**Examples:**
```bash
# Basic usage
tsugite benchmark run --models "ollama:llama2"

# Advanced usage
tsugite benchmark run \
  --models "ollama:qwen2.5-coder:7b,openai:gpt-4o-mini" \
  --categories "basic,tools" \
  --format "html" \
  --output "my_benchmark.html" \
  --repeat 3
```

### Compare Models (Coming Soon)

```bash
tsugite benchmark compare --baseline "openai:gpt-4o" --models "ollama:*"
```

### Generate Reports (Coming Soon)

```bash
tsugite benchmark report --format html --output results.html
```

## Configuration

### Global Config (`benchmarks/config.yaml`)

```yaml
categories:
  basic:
    weight: 1.0
    timeout: 30
    required: true

models:
  defaults:
    temperature: 0.1
    max_tokens: 2000
    timeout: 60

scoring:
  weights:
    correctness: 0.4
    performance: 0.2
    efficiency: 0.2
    quality: 0.2
```

## Metrics and Scoring

### Core Metrics

- **Correctness**: Output matches expected results
- **Performance**: Execution speed and efficiency
- **Quality**: Output quality based on criteria
- **Cost**: Estimated API/compute costs
- **Reliability**: Error rates and consistency

### Scoring System

- **Excellent**: 90-100% accuracy
- **Good**: 75-89% accuracy
- **Fair**: 60-74% accuracy
- **Poor**: 40-59% accuracy
- **Very Poor**: <40% accuracy

### Cost Analysis

Automatic cost estimation for different model providers:
- OpenAI models (GPT-4, GPT-3.5)
- Anthropic models (Claude)
- Local models (Ollama) - marked as free

## Report Formats

### JSON Report
Complete data export for programmatic analysis:
```json
{
  "benchmark_info": {...},
  "model_performances": {...},
  "detailed_results": {...},
  "summary": {...}
}
```

### Markdown Report
Human-readable summary with tables and charts:
- Executive summary
- Model rankings
- Category breakdown
- Detailed test results

### HTML Report
Interactive web report with:
- Rich visualizations
- Progress bars
- Color-coded performance tiers
- Responsive design

### CSV Export
Tabular data for spreadsheet analysis:
```csv
Model,Test_ID,Category,Passed,Score,Duration,Token_Usage,Cost,Error
model1,basic_001,basic,True,0.95,1.2,150,0.001,
```

## Extending the Framework

### Adding New Evaluators

```python
from tsugite.benchmark.evaluators import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, **kwargs):
        # Your evaluation logic
        return {
            "passed": True,
            "score": 0.9,
            "metrics": {...}
        }
```

### Adding New Test Categories

1. Create directory: `benchmarks/my_category/`
2. Add test agents: `benchmarks/my_category/test_name.md`
3. Update config: `benchmarks/config.yaml`

### Custom Metrics

```python
from tsugite.benchmark.metrics import BenchmarkMetrics

# Add to BenchmarkMetrics class
@staticmethod
def calculate_custom_metric(data):
    # Your metric calculation
    return result
```

## Best Practices

### Test Design
- Make tests deterministic when possible
- Use clear, specific expected outputs
- Include edge cases and error conditions
- Test both success and failure scenarios

### Model Selection
- Include diverse model types (local vs API)
- Test different model sizes
- Consider cost vs performance tradeoffs

### Performance Optimization
- Use parallel execution for speed
- Set appropriate timeouts
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model is available and spelled correctly
2. **Timeout errors**: Increase timeout in test configuration
3. **Tool registration**: Verify required tools are available
4. **Permission errors**: Check file system permissions for output directory

### Debug Mode

Enable debug output for troubleshooting:
```bash
tsugite benchmark run --models "test-model" --debug
```

## Contributing

To contribute new tests or improvements:

1. Create test agents following the format above
2. Add comprehensive evaluation criteria
3. Include both positive and negative test cases
4. Update documentation and examples
5. Add unit tests for new functionality

## License

This benchmark framework is part of the Tsugite project and follows the same licensing terms.