---
name: conditional_logic
test_id: basic_004
description: "Test conditional logic and decision making"
model: "{{ model }}"
timeout: 25
expected_output: "PASS"
expected_type: "string"
weight: 1.0
evaluation_criteria:
  logic:
    type: "keyword"
    keywords: ["PASS"]
    weight: 1.0
---

# Task
Evaluate the following conditions and output "PASS" if all are true, "FAIL" otherwise:

1. 10 > 5
2. "hello" contains the letter "e"
3. 15 is divisible by 3
4. The length of "test" is 4

Return only: PASS or FAIL