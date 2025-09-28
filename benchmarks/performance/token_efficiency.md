---
name: token_efficiency
test_id: performance_001
description: "Test minimal token usage for simple task"
model: "{{ model }}"
timeout: 15
expected_output: "Sum: 15"
expected_type: "string"
weight: 1.5
evaluation_criteria:
  correctness:
    type: "keyword"
    keywords: ["15"]
    weight: 0.7
  conciseness:
    type: "length"
    min_length: 5
    max_length: 20
    weight: 0.3
---

# Task
Calculate 7 + 8 and respond with: "Sum: [result]"

Be as concise as possible.