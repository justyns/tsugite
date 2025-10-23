"""Test execution logic for benchmark framework."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from ..agent_runner import run_agent
from ..exceptions import AgentExecutionError
from .config import (
    EVALUATION_WEIGHTS,
    SIMILARITY_THRESHOLDS,
)
from .discovery import BenchmarkTest, TestCase
from .evaluators import CorrectnessEvaluator, LLMEvaluator
from .metrics import BenchmarkTestResult


class TestExecutor:
    """Executes benchmark tests against models."""

    def __init__(self, output_dir: Path, llm_evaluator_model: str = "openai:gpt-4o-mini"):
        """Initialize test executor.

        Args:
            output_dir: Directory for temporary files
            llm_evaluator_model: Model to use for LLM evaluation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.correctness_evaluator = CorrectnessEvaluator()
        self.llm_evaluator = LLMEvaluator(evaluator_model=llm_evaluator_model)

    async def run_test(self, model_name: str, test: BenchmarkTest) -> BenchmarkTestResult:
        """Run all test cases for a single agent against a model.

        Args:
            model_name: Name of model to test
            test: Benchmark test to run

        Returns:
            Aggregated test result
        """
        start_time = time.time()

        if not test.test_cases:
            raise ValueError(f"No test cases found for {test.test_id}")

        try:
            # Create temporary agent file with target model
            temp_agent_path = self._create_temp_agent(test, model_name)

            try:
                # Run all test cases
                case_results, aggregated_output, total_steps, total_tokens, total_cost = await self._run_all_test_cases(
                    test, temp_agent_path, model_name
                )

                # Calculate aggregate results
                total_passed = sum(1 for r in case_results if r["passed"])
                overall_passed = total_passed == len(test.test_cases)

                # Calculate weighted score
                total_score = sum(r["score"] * test.test_cases[i].weight for i, r in enumerate(case_results))
                total_weight = sum(tc.weight for tc in test.test_cases)
                overall_score = total_score / total_weight if total_weight > 0 else 0.0

                # Aggregate metrics
                aggregate_metrics = {
                    "case_results": case_results,
                    "cases_passed": total_passed,
                    "total_cases": len(test.test_cases),
                    "pass_rate": total_passed / len(test.test_cases),
                }

                duration = time.time() - start_time

                # Use first case's expected output for legacy compatibility
                expected_output = test.test_cases[0].expected_output if test.test_cases else ""

                return BenchmarkTestResult(
                    test_id=test.test_id,
                    model=model_name,
                    passed=overall_passed,
                    score=overall_score,
                    duration=duration,
                    output=aggregated_output,
                    expected_output=expected_output or "",
                    category=test.category,
                    error=None,
                    token_usage={"total": total_tokens},
                    cost=total_cost,
                    steps_taken=total_steps,
                    metrics=aggregate_metrics,
                )

            finally:
                # Clean up temp file
                if temp_agent_path.exists():
                    temp_agent_path.unlink()

        except AgentExecutionError as e:
            # Agent execution failed with execution details
            duration = time.time() - start_time

            execution_trace = None
            if e.execution_steps:
                # Build execution trace from steps in exception
                execution_trace = [
                    {
                        "step": step.step_number,
                        "thought": step.thought,
                        "code": step.code,
                        "output": step.output,
                        "tools_called": step.tools_called,
                        "error": step.error,
                    }
                    for step in e.execution_steps
                ]

            metrics = {}
            if execution_trace:
                metrics["execution_trace"] = execution_trace

            return BenchmarkTestResult(
                test_id=test.test_id,
                model=model_name,
                passed=False,
                score=0.0,
                duration=duration,
                output="",
                expected_output="",
                category=test.category,
                error=str(e),
                token_usage={"total": e.token_usage} if e.token_usage else {},
                cost=e.cost or 0.0,
                steps_taken=e.step_count,
                metrics=metrics,
            )

        except Exception as e:
            # Other unexpected errors without execution details
            duration = time.time() - start_time

            return BenchmarkTestResult(
                test_id=test.test_id,
                model=model_name,
                passed=False,
                score=0.0,
                duration=duration,
                output="",
                expected_output="",
                category=test.category,
                error=str(e),
                token_usage={},
                cost=0.0,
                steps_taken=0,
                metrics={},
            )

    def _create_temp_agent(self, test: BenchmarkTest, model_name: str) -> Path:
        """Create temporary agent file with specified model.

        Args:
            test: Benchmark test
            model_name: Model name to inject

        Returns:
            Path to temporary agent file
        """
        original_content = test.agent_path.read_text()

        # Find YAML frontmatter boundaries
        lines = original_content.split("\n")
        yaml_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                yaml_end = i
                break

        if yaml_end == -1:
            raise ValueError("Invalid YAML frontmatter format")

        # Parse original YAML
        original_yaml = "\n".join(lines[1:yaml_end])
        yaml_data = yaml.safe_load(original_yaml) or {}
        markdown_content = "\n".join(lines[yaml_end + 1 :])

        # Create clean YAML with only agent fields
        clean_yaml_data = {
            "name": yaml_data.get("name", test.test_id),
            "description": yaml_data.get("description", ""),
            "model": model_name,  # Override model
            "max_turns": yaml_data.get("max_turns", 5),
            "tools": yaml_data.get("tools", []),
            "text_mode": yaml_data.get("text_mode"),
            "prefetch": yaml_data.get("prefetch", []),
            "permissions_profile": yaml_data.get("permissions_profile", "default"),
            "context_budget": yaml_data.get("context_budget"),
        }

        # Remove None values and empty lists
        clean_yaml_data = {k: v for k, v in clean_yaml_data.items() if v is not None and v != []}

        # Create clean agent content
        clean_yaml = yaml.dump(clean_yaml_data, default_flow_style=False)
        clean_agent_content = f"---\n{clean_yaml}---\n\n{markdown_content}"

        # Write temporary agent file
        temp_agent_path = self.output_dir / f"temp_{test.test_id}_{model_name.replace(':', '_')}.md"
        temp_agent_path.write_text(clean_agent_content)

        return temp_agent_path

    async def _run_all_test_cases(
        self, test: BenchmarkTest, temp_agent_path: Path, model_name: str
    ) -> tuple[list[Dict[str, Any]], str, int, int, float]:
        """Run all test cases for a test.

        Args:
            test: Benchmark test
            temp_agent_path: Path to temporary agent file
            model_name: Model name

        Returns:
            Tuple of (case_results, aggregated_output, total_steps, total_tokens, total_cost)
        """
        case_results = []
        raw_outputs = []
        aggregated_output_parts = []
        total_steps = 0
        total_tokens = 0
        total_cost = 0.0

        for test_case in test.test_cases:
            case_start = time.time()

            # Prepare prompt with planning instruction if needed
            final_prompt = self._prepare_prompt(test_case)

            # Run the agent with token usage tracking to get step count
            result_tuple = run_agent(
                agent_path=temp_agent_path,
                prompt=final_prompt,
                model_override=model_name,
                debug=False,
                return_token_usage=True,
            )

            # Unpack result: (output, token_count, cost, step_count, execution_steps)
            result, token_count, cost, steps, execution_steps = result_tuple
            total_steps += steps
            total_tokens += token_count or 0
            total_cost += cost or 0.0

            case_duration = time.time() - case_start

            # Evaluate this test case
            case_evaluation = await self._evaluate_test_case(test_case, result, case_duration, execution_steps, test)

            case_results.append(case_evaluation)
            raw_outputs.append(str(result))
            aggregated_output_parts.append(
                f"Test Case: {test_case.name}\nPrompt: {test_case.prompt}\nOutput: {result}\n---\n"
            )

        # Create aggregated output
        if len(raw_outputs) == 1:
            aggregated_output = raw_outputs[0]
        else:
            aggregated_output = "\n".join(aggregated_output_parts)

        return case_results, aggregated_output, total_steps, total_tokens, total_cost

    def _prepare_prompt(self, test_case: TestCase) -> str:
        """Prepare prompt with optional planning instruction.

        Args:
            test_case: Test case

        Returns:
            Final prompt
        """
        if not test_case.requires_plan:
            return test_case.prompt

        planning_instruction = """

PLANNING REQUIREMENT: Before executing this task, you must first create a detailed plan. Your response should be structured as follows:

1. **PLAN SECTION**: Start with a clear, step-by-step plan that outlines how you will approach this task
2. **EXECUTION SECTION**: Then proceed with executing the plan

Make sure your plan includes the key steps and reasoning before you start executing."""

        return test_case.prompt + planning_instruction

    async def _evaluate_test_case(
        self,
        test_case: TestCase,
        result: str,
        duration: float,
        execution_steps: list = None,
        test: BenchmarkTest = None,
    ) -> Dict[str, Any]:
        """Evaluate a single test case result.

        Args:
            test_case: Test case definition
            result: Agent output
            duration: Execution duration
            execution_steps: List of StepResult objects from execution
            test: BenchmarkTest object with agent configuration

        Returns:
            Evaluation dictionary with passed, score, and metrics
        """
        evaluation = {
            "test_case": test_case.name,
            "passed": False,
            "score": 0.0,
            "metrics": {},
        }

        try:
            # 1. Basic correctness evaluation
            if test_case.expected_output:
                evaluation = self._evaluate_correctness(test_case, result, evaluation)

            # 2. Custom evaluation criteria (includes tool checking)
            if test_case.evaluation:
                evaluation = self._evaluate_custom_criteria(test_case, result, evaluation, execution_steps, test)

            # 3. LLM evaluation (if enabled)
            if test_case.use_llm_evaluation and test_case.llm_evaluation_criteria:
                evaluation = await self._evaluate_with_llm(test_case, result, evaluation)

            # 4. Plan evaluation (if required)
            if test_case.requires_plan:
                evaluation = self._evaluate_planning(test_case, result, evaluation)

        except Exception as e:
            evaluation["error"] = str(e)

        # Add execution trace for failed tests (for debugging)
        # Only save for failures to reduce JSON size
        if execution_steps and not evaluation["passed"]:
            evaluation["execution_trace"] = [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "code": step.code,
                    "output": step.output,
                    "tools_called": step.tools_called,
                    "error": step.error,
                }
                for step in execution_steps
            ]

        return evaluation

    def _evaluate_correctness(self, test_case: TestCase, result: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate output correctness."""
        correctness = self.correctness_evaluator.evaluate(
            output=result,
            expected=test_case.expected_output,
            output_type="string",
        )
        evaluation["passed"] = correctness["passed"]
        evaluation["score"] = correctness["score"]
        evaluation["metrics"]["correctness"] = correctness
        return evaluation

    def _evaluate_custom_criteria(
        self,
        test_case: TestCase,
        result: str,
        evaluation: Dict[str, Any],
        execution_steps: list = None,
        test: BenchmarkTest = None,
    ) -> Dict[str, Any]:
        """Evaluate custom criteria."""
        custom_checks = {}

        for criterion, expected_value in test_case.evaluation.items():
            check_passed = self._check_criterion(criterion, expected_value, result, execution_steps, test)
            custom_checks[criterion] = check_passed

        evaluation["metrics"]["custom"] = custom_checks

        # ALL custom criteria must pass for score of 1.0, otherwise 0.0
        if not test_case.expected_output:
            all_custom_pass = all(custom_checks.values()) if custom_checks else True
            evaluation["score"] = 1.0 if all_custom_pass else 0.0
            evaluation["passed"] = all_custom_pass

        return evaluation

    def _check_criterion(
        self,
        criterion: str,
        expected_value: Any,
        result: str,
        execution_steps: list = None,
        test: BenchmarkTest = None,
    ) -> bool:
        """Check a single custom criterion.

        Args:
            criterion: Criterion name
            expected_value: Expected value
            result: Agent output
            execution_steps: List of StepResult objects from execution
            test: BenchmarkTest object with agent configuration

        Returns:
            True if criterion passes
        """
        if criterion == "tool_called":
            # Check if specific tools were called during execution
            if not execution_steps:
                return False

            # Collect all tools called across all steps
            tools_called = set()
            for step in execution_steps:
                if hasattr(step, "tools_called") and step.tools_called:
                    tools_called.update(step.tools_called)

            # expected_value can be a single tool name or a list of required tools
            if isinstance(expected_value, list):
                # ALL listed tools must have been called
                return all(tool in tools_called for tool in expected_value)
            else:
                # Single tool must have been called
                return expected_value in tools_called

        elif criterion == "exact_match":
            return result.strip() == expected_value

        elif criterion == "contains":
            result_lower = result.lower()
            if isinstance(expected_value, list):
                return all(str(item).lower() in result_lower for item in expected_value)
            else:
                return str(expected_value).lower() in result_lower

        elif criterion == "valid_json":
            try:
                json.loads(result.strip())
                return expected_value  # true means should be valid
            except (json.JSONDecodeError, ValueError):
                return not expected_value  # false if expecting invalid

        elif criterion == "contains_keys":
            try:
                parsed = json.loads(result.strip())
                if isinstance(parsed, dict):
                    if isinstance(expected_value, list):
                        return all(key in parsed for key in expected_value)
                    else:
                        return expected_value in parsed
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            return False

        elif criterion == "content_pattern":
            return bool(re.search(expected_value, result))

        elif criterion == "matches":
            # Regex pattern matching on agent output
            # expected_value can be a single pattern or list of patterns
            if isinstance(expected_value, list):
                # ALL patterns must match
                return all(bool(re.search(pattern, result)) for pattern in expected_value)
            else:
                # Single pattern must match
                return bool(re.search(expected_value, result))

        elif criterion == "file_check":
            # Verify file contents directly
            # expected_value is a dict with: path, exists, contains, matches, exact
            if not isinstance(expected_value, dict):
                return False

            file_path = expected_value.get("path")
            if not file_path:
                return False

            from pathlib import Path

            path = Path(file_path)

            # Check existence
            if expected_value.get("exists"):
                if not path.exists():
                    return False

            # If file doesn't exist and we're not checking existence, fail
            if not path.exists():
                return False

            # Read file content for verification
            try:
                file_content = path.read_text(encoding="utf-8")
            except Exception:
                # File read failed
                return False

            # Check exact match
            if "exact" in expected_value:
                if file_content.strip() != expected_value["exact"].strip():
                    return False

            # Check contains (list of strings that must all appear)
            if "contains" in expected_value:
                contains_list = expected_value["contains"]
                if isinstance(contains_list, list):
                    if not all(str(item) in file_content for item in contains_list):
                        return False
                else:
                    if str(contains_list) not in file_content:
                        return False

            # Check regex matches
            if "matches" in expected_value:
                pattern = expected_value["matches"]
                if isinstance(pattern, list):
                    # All patterns must match
                    if not all(bool(re.search(p, file_content)) for p in pattern):
                        return False
                else:
                    # Single pattern must match
                    if not bool(re.search(pattern, file_content)):
                        return False

            # All checks passed
            return True

        elif criterion == "min_length":
            return len(result) >= expected_value

        else:
            # Default to simple substring check
            return str(expected_value).lower() in result.lower()

    async def _evaluate_with_llm(self, test_case: TestCase, result: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate using LLM judge."""
        llm_eval = await self.llm_evaluator.evaluate(
            output=result,
            task_description=test_case.prompt,
            evaluation_criteria=test_case.llm_evaluation_criteria,
            rubric=test_case.llm_evaluation_rubric or {},
        )
        evaluation["metrics"]["llm_evaluation"] = llm_eval

        # Blend with existing score or use standalone
        if evaluation["score"] == 0.0:
            evaluation["score"] = llm_eval["llm_score"]
            evaluation["passed"] = llm_eval["llm_score"] >= SIMILARITY_THRESHOLDS.llm_evaluation_threshold
        else:
            # Blend LLM score with existing evaluation
            blended_score = (
                evaluation["score"] * EVALUATION_WEIGHTS.base_weight
                + llm_eval["llm_score"] * EVALUATION_WEIGHTS.llm_weight
            )
            evaluation["score"] = blended_score

        return evaluation

    def _evaluate_planning(self, test_case: TestCase, result: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate planning quality."""
        plan_score = 0.0
        plan_checks = {}

        # Check for planning indicators
        plan_indicators = ["plan", "step", "approach", "strategy", "outline", "first", "then", "next", "finally"]
        has_plan_structure = any(indicator in result.lower() for indicator in plan_indicators)
        plan_checks["has_planning_structure"] = has_plan_structure
        if has_plan_structure:
            plan_score += 0.3

        # Check for specific plan elements
        if test_case.expected_plan_elements:
            elements_found = sum(1 for element in test_case.expected_plan_elements if element.lower() in result.lower())
            element_score = elements_found / len(test_case.expected_plan_elements)
            plan_score += element_score * 0.4

            for element in test_case.expected_plan_elements:
                plan_checks[f"element_{element}"] = element.lower() in result.lower()

        # Check custom plan evaluation criteria
        if test_case.plan_evaluation:
            custom_plan_score = self._evaluate_plan_criteria(test_case, result, plan_checks)
            plan_score += custom_plan_score * 0.3

        evaluation["metrics"]["planning"] = {
            "plan_score": plan_score,
            "plan_checks": plan_checks,
            "required": True,
        }

        # Blend planning score with existing evaluation
        if evaluation["score"] == 0.0:
            evaluation["score"] = plan_score
            evaluation["passed"] = plan_score >= SIMILARITY_THRESHOLDS.behavior_pass_threshold
        else:
            blended_score = (
                evaluation["score"] * (1 - EVALUATION_WEIGHTS.planning_weight)
                + plan_score * EVALUATION_WEIGHTS.planning_weight
            )
            evaluation["score"] = blended_score
            # Plan must meet minimum threshold
            evaluation["passed"] = evaluation["passed"] and plan_score >= EVALUATION_WEIGHTS.planning_minimum

        return evaluation

    def _evaluate_plan_criteria(self, test_case: TestCase, result: str, plan_checks: Dict[str, Any]) -> float:
        """Evaluate custom planning criteria.

        Args:
            test_case: Test case
            result: Agent output
            plan_checks: Dictionary to store check results

        Returns:
            Custom plan score (0.0 to 1.0)
        """
        custom_plan_score = 0.0
        criteria_count = len(test_case.plan_evaluation)

        for criterion, expected_value in test_case.plan_evaluation.items():
            if criterion == "min_plan_steps":
                # Count numbered or bulleted steps
                steps = len(re.findall(r"(\d+\.|•|-|\*)\s", result))
                plan_checks[criterion] = steps >= expected_value
                if steps >= expected_value:
                    custom_plan_score += 1.0 / criteria_count

            elif criterion == "structured_sections":
                # Look for section headers
                sections = any(header in result.upper() for header in ["PLAN", "EXECUTION", "APPROACH", "STRATEGY"])
                plan_checks[criterion] = sections == expected_value
                if sections == expected_value:
                    custom_plan_score += 1.0 / criteria_count

            else:
                # Default criterion check
                criterion_met = str(expected_value).lower() in result.lower()
                plan_checks[criterion] = criterion_met
                if criterion_met:
                    custom_plan_score += 1.0 / criteria_count

        return custom_plan_score
