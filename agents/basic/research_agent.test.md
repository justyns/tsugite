---
agent_under_test: research_agent.md
test_id: research_agent_planning
timeout: 180
category: basic
description: Test research agent's planning and execution capabilities
---

# Test Cases for Research Agent

## Test Case 1: Simple Research with Planning
**Prompt:** "Research the benefits and drawbacks of renewable energy sources"

**Requires Plan:** true

**Expected Plan Elements:**
- Identify key renewable energy types
- Benefits analysis
- Drawbacks analysis
- Structure findings

**Plan Evaluation:**
- min_plan_steps: 3
- structured_sections: true

**Expected Behaviors:**
- Should create a structured analysis
- Should cover multiple renewable energy types

**Evaluation:**
- contains: ["solar", "wind", "benefits", "drawbacks"]
- min_length: 200

## Test Case 2: Complex Research Task
**Prompt:** "Analyze the impact of artificial intelligence on job markets, including both positive and negative effects, and provide recommendations"

**Requires Plan:** true

**Expected Plan Elements:**
- Research AI impact on employment
- Analyze positive effects
- Analyze negative effects
- Develop recommendations

**Plan Evaluation:**
- min_plan_steps: 4
- structured_sections: true

**Evaluation:**
- contains: ["artificial intelligence", "job market", "positive", "negative", "recommendations"]
- content_pattern: "recommendation.*:"
- min_length: 300

## Test Case 3: No Planning Required (Control)
**Prompt:** "List three types of renewable energy"

**Expected Output:** should mention solar, wind, and hydro energy

**Evaluation:**
- contains: ["solar", "wind", "hydro"]
- min_length: 50