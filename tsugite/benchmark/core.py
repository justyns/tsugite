"""Core benchmark framework for evaluating Tsugite agents."""

import time
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import json
import re

from ..agent_runner import run_agent
from ..utils import parse_yaml_frontmatter
from .metrics import BenchmarkMetrics, BenchmarkTestResult, ModelPerformance
from .evaluators import CorrectnessEvaluator, PerformanceEvaluator, QualityEvaluator, LLMEvaluator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    models: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=lambda: ["basic"])
    timeout: int = 120  # seconds
    parallel: bool = True
    temperature: float = 0.1  # Low for reproducibility
    max_tokens: int = 2000
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    include_cost_analysis: bool = True
    repeat_count: int = 1  # Number of times to run each test for averaging
    llm_evaluator_model: str = "openai:gpt-4o-mini"  # Model to use for LLM evaluation


@dataclass
class TestCase:
    """Individual test case within a benchmark test."""

    name: str
    prompt: str
    expected_output: Optional[str] = None
    expected_behaviors: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    requires_plan: bool = False
    expected_plan_elements: List[str] = field(default_factory=list)
    plan_evaluation: Dict[str, Any] = field(default_factory=dict)
    # LLM evaluation fields
    use_llm_evaluation: bool = False
    llm_evaluation_criteria: str = ""
    llm_evaluation_rubric: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTest:
    """Test specification for an agent benchmark."""

    name: str
    agent_path: Path
    test_id: str
    category: str
    description: str = ""
    expected_output: Optional[str] = None
    expected_type: str = "string"
    timeout: int = 60
    weight: float = 1.0
    requires_tools: List[str] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    test_path: Optional[Path] = None
    # LLM evaluation fields
    use_llm_evaluation: bool = False
    llm_evaluation_criteria: str = ""
    llm_evaluation_rubric: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run."""

    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    total_duration: float
    model_performances: Dict[str, ModelPerformance]
    test_results: Dict[str, Dict[str, BenchmarkTestResult]]  # model -> test_id -> result
    summary: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


def _extract_common_test_fields(
    metadata: Dict[str, Any],
    *,
    fallback_id: str,
    fallback_description: str,
    default_timeout: int,
) -> Dict[str, Any]:
    """Normalize shared metadata fields for benchmark tests."""

    weight_value = metadata.get("weight", 1.0)
    try:
        weight = float(weight_value)
    except (TypeError, ValueError):
        weight = 1.0

    requires_tools = metadata.get("requires_tools", metadata.get("tools", []))

    return {
        "test_id": metadata.get("test_id", fallback_id),
        "description": metadata.get("description", fallback_description),
        "timeout": metadata.get("timeout", default_timeout),
        "requires_tools": requires_tools,
        "weight": weight,
        "expected_output": metadata.get("expected_output"),
        "expected_type": metadata.get("expected_type", "string"),
        "evaluation_criteria": metadata.get("evaluation_criteria", {}),
    }


class BenchmarkRunner:
    """Main benchmark runner for evaluating models across test suites."""

    benchmark_dir: Path = Path("benchmarks")

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmark_dir = Path(type(self).benchmark_dir)
        self.test_cache: Dict[str, List[BenchmarkTest]] = {}

        # Initialize evaluators
        self.correctness_evaluator = CorrectnessEvaluator()
        self.performance_evaluator = PerformanceEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.llm_evaluator = LLMEvaluator(evaluator_model=config.llm_evaluator_model)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_tests(
        self, categories: Optional[List[str]] = None, agent_path: Optional[Path] = None
    ) -> List[BenchmarkTest]:
        """Discover agent + test.md pairs in specified categories or for a specific agent."""

        if agent_path:
            # Test a specific agent
            agent_path = Path(agent_path)
            category = agent_path.parent.name if agent_path.parent.parent == self.benchmark_dir else "custom"
            test_file = agent_path.with_suffix(".test.md")
            if test_file.exists():
                return [self._parse_agent_test_pair(agent_path, test_file, category)]
            return [self._parse_benchmark_test(agent_path, category)]

        if categories is None:
            categories = self.config.categories

        tests = []
        for category in categories:
            if category in self.test_cache:
                tests.extend(self.test_cache[category])
                continue

            category_dir = self.benchmark_dir / category
            if not category_dir.exists():
                print(f"Warning: Category directory not found: {category_dir}")
                continue

            category_tests = []
            for agent_file in category_dir.glob("*.md"):
                if agent_file.name.endswith(".test.md"):
                    continue

                test_file = agent_file.with_suffix(".test.md")

                try:
                    if test_file.exists():
                        test = self._parse_agent_test_pair(agent_file, test_file, category)
                    else:
                        test = self._parse_benchmark_test(agent_file, category)
                    category_tests.append(test)
                except Exception as e:
                    print(f"Error parsing benchmark test {agent_file}: {e}")

            self.test_cache[category] = category_tests
            tests.extend(category_tests)

        return tests

    def _discover_single_agent_test(self, agent_path: Path) -> BenchmarkTest:
        """Discover test for a single agent file."""
        agent_path = Path(agent_path)
        test_path = agent_path.with_suffix(".test.md")

        if not test_path.exists():
            return self._parse_benchmark_test(agent_path, "custom")

        return self._parse_agent_test_pair(agent_path, test_path, "custom")

    def _parse_agent_test_pair(self, agent_path: Path, test_path: Path, category: str) -> BenchmarkTest:
        """Parse an agent + test.md pair into a BenchmarkTest."""
        try:
            test_metadata, markdown_content = parse_yaml_frontmatter(test_path.read_text(), "Test file")

            common = _extract_common_test_fields(
                test_metadata,
                fallback_id=agent_path.stem,
                fallback_description=f"Test for {agent_path.stem}",
                default_timeout=self.config.timeout,
            )

            test_cases = self._parse_test_cases(markdown_content)

            # Validate agent file exists and is readable
            if not agent_path.exists():
                raise ValueError(f"Agent file not found: {agent_path}")

            expected_output = common["expected_output"]
            if expected_output is None and test_cases:
                expected_output = test_cases[0].expected_output

            return BenchmarkTest(
                name=agent_path.stem,
                agent_path=agent_path,
                test_id=common["test_id"],
                category=category,
                description=common["description"],
                expected_output=expected_output,
                expected_type=common["expected_type"],
                timeout=common["timeout"],
                weight=common["weight"],
                requires_tools=common["requires_tools"],
                test_cases=test_cases,
                evaluation_criteria=common["evaluation_criteria"],
                test_path=test_path,
                use_llm_evaluation=test_metadata.get("use_llm_evaluation", False),
                llm_evaluation_criteria=test_metadata.get("llm_evaluation_criteria", ""),
                llm_evaluation_rubric=test_metadata.get("llm_evaluation_rubric", {}),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse agent test pair {agent_path}/{test_path}: {e}")

    def _parse_benchmark_test(self, agent_path: Path, category: str) -> BenchmarkTest:
        """Parse a single benchmark markdown file with embedded expectations."""
        try:
            metadata, markdown_content = parse_yaml_frontmatter(agent_path.read_text(), "Benchmark file")

            common = _extract_common_test_fields(
                metadata,
                fallback_id=agent_path.stem,
                fallback_description=metadata.get("name", ""),
                default_timeout=self.config.timeout,
            )

            prompt = metadata.get("prompt") or self._extract_prompt_from_markdown(markdown_content)

            test_case = TestCase(
                name=f"{common['test_id']}_default",
                prompt=prompt,
                expected_output=common["expected_output"],
                expected_behaviors=metadata.get("expected_behaviors", {}),
                evaluation=metadata.get("evaluation", {}),
                weight=common["weight"],
                requires_plan=metadata.get("requires_plan", False),
                expected_plan_elements=metadata.get("expected_plan_elements", []),
                plan_evaluation=metadata.get("plan_evaluation", {}),
                use_llm_evaluation=metadata.get("use_llm_evaluation", False),
                llm_evaluation_criteria=metadata.get("llm_evaluation_criteria", ""),
                llm_evaluation_rubric=metadata.get("llm_evaluation_rubric", {}),
            )

            return BenchmarkTest(
                name=metadata.get("name", agent_path.stem),
                agent_path=agent_path,
                test_id=common["test_id"],
                category=category,
                description=common["description"],
                expected_output=common["expected_output"],
                expected_type=common["expected_type"],
                timeout=common["timeout"],
                weight=common["weight"],
                requires_tools=common["requires_tools"],
                test_cases=[test_case],
                evaluation_criteria=common["evaluation_criteria"],
                test_path=None,
                use_llm_evaluation=metadata.get("use_llm_evaluation", False),
                llm_evaluation_criteria=metadata.get("llm_evaluation_criteria", ""),
                llm_evaluation_rubric=metadata.get("llm_evaluation_rubric", {}),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse benchmark test {agent_path}: {e}")

    @staticmethod
    def _extract_prompt_from_markdown(markdown_content: str) -> str:
        """Derive a prompt from markdown content when none is provided explicitly."""
        for line in markdown_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped
        return "Describe the required task in detail."

    @staticmethod
    def _extract_inline_field(content: str, label: str) -> Optional[str]:
        """Extract single-line field values like Prompt or Expected Output."""

        pattern_label = re.escape(label)
        quoted = re.search(rf"\*\*{pattern_label}:\*\*\s*\"([^\"]+)\"", content)
        if quoted:
            return quoted.group(1).strip()

        block = re.search(rf"\*\*{pattern_label}:\*\*\s*(.+?)(?=\n\*\*|$)", content, re.DOTALL)
        if block:
            return block.group(1).strip()

        return None

    @staticmethod
    def _extract_block(content: str, label: str) -> Optional[str]:
        """Extract multi-line blocks introduced by a bold label."""

        pattern_label = re.escape(label)
        match = re.search(rf"\*\*{pattern_label}:\*\*\n(.*?)(?=\n\*\*|\Z)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_bullet_list(block: Optional[str]) -> List[str]:
        if not block:
            return []
        items = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
        return items

    @staticmethod
    def _coerce_value(value: str) -> Any:
        raw = value.strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "null":
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw.strip("'\"")

    @classmethod
    def _parse_key_value_block(cls, block: Optional[str]) -> Dict[str, Any]:
        if not block:
            return {}
        result: Dict[str, Any] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("- ") or ":" not in line:
                continue
            key, raw_value = line[2:].split(":", 1)
            result[key.strip()] = cls._coerce_value(raw_value)
        return result

    @staticmethod
    def _derive_behavior_expectations(items: List[str]) -> Dict[str, Any]:
        expectations: Dict[str, Any] = {}
        for item in items:
            lowered = item.lower()
            if "use" in lowered and "tool" in lowered:
                expectations["tool_usage"] = True
            elif "file" in lowered and "create" in lowered:
                expectations["file_created"] = True
        return expectations

    @staticmethod
    def _add_model_error(errors: List[str], test: BenchmarkTest, model_name: str, reason: str) -> None:
        errors.append(f"Test {test.test_id} for model {model_name}: {reason}")

    def _parse_test_cases(self, markdown_content: str) -> List[TestCase]:
        """Parse test cases from markdown content."""
        test_cases = []

        # Split content by ## headers (test cases)
        sections = re.split(r"\n## (.+?)\n", markdown_content)

        # Skip the first section (usually intro text)
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            case_name = sections[i].strip()
            case_content = sections[i + 1].strip()

            # Parse the test case content
            test_case = self._parse_single_test_case(case_name, case_content)
            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _parse_single_test_case(self, name: str, content: str) -> Optional[TestCase]:
        """Parse a single test case from markdown content."""
        prompt = self._extract_inline_field(content, "Prompt")
        if not prompt:
            return None

        expected_output = self._extract_inline_field(content, "Expected Output")

        behaviors_block = self._extract_block(content, "Expected Behaviors")
        expected_behaviors = self._derive_behavior_expectations(self._parse_bullet_list(behaviors_block))

        evaluation = self._parse_key_value_block(self._extract_block(content, "Evaluation"))

        requires_plan_text = self._extract_inline_field(content, "Requires Plan")
        requires_plan = bool(requires_plan_text and requires_plan_text.strip().lower() in {"true", "yes", "1"})

        expected_plan_elements = self._parse_bullet_list(self._extract_block(content, "Expected Plan Elements"))
        plan_evaluation = self._parse_key_value_block(self._extract_block(content, "Plan Evaluation"))

        # Parse LLM evaluation fields
        use_llm_evaluation_text = self._extract_inline_field(content, "Use LLM Evaluation")
        use_llm_evaluation = bool(
            use_llm_evaluation_text and use_llm_evaluation_text.strip().lower() in {"true", "yes", "1"}
        )

        llm_evaluation_criteria = self._extract_inline_field(content, "LLM Evaluation Criteria") or ""

        llm_rubric_block = self._extract_block(content, "LLM Evaluation Rubric")
        llm_evaluation_rubric = self._parse_key_value_block(llm_rubric_block)

        return TestCase(
            name=name,
            prompt=prompt,
            expected_output=expected_output,
            expected_behaviors=expected_behaviors,
            evaluation=evaluation,
            weight=1.0,
            requires_plan=requires_plan,
            expected_plan_elements=expected_plan_elements,
            plan_evaluation=plan_evaluation,
            use_llm_evaluation=use_llm_evaluation,
            llm_evaluation_criteria=llm_evaluation_criteria,
            llm_evaluation_rubric=llm_evaluation_rubric,
        )

    async def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        test_filter: Optional[str] = None,
        agent_path: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Run benchmark suite against specified models."""
        start_time = datetime.now()

        if models is None:
            models = self.config.models
        if not models:
            raise ValueError("No models specified for benchmarking")

        # Discover tests
        if agent_path:
            tests = self.discover_tests(agent_path=agent_path)
        else:
            tests = self.discover_tests(categories)
            if test_filter:
                tests = [
                    t
                    for t in tests
                    if test_filter.lower() in t.name.lower() or test_filter.lower() in t.test_id.lower()
                ]

        if not tests:
            raise ValueError("No tests found matching criteria")

        print(f"Running {len(tests)} tests across {len(models)} models...")

        # Initialize results tracking
        model_performances = {}
        test_results = {model: {} for model in models}
        errors = []

        # Run tests for each model
        for model_name in models:
            print(f"\nEvaluating model: {model_name}")
            try:
                model_performance, model_test_results, model_errors = await self._run_model_tests(model_name, tests)
                model_performances[model_name] = model_performance
                test_results[model_name] = model_test_results
                errors.extend(model_errors)
            except Exception as e:
                error_msg = f"Failed to evaluate model {model_name}: {e}"
                errors.append(error_msg)
                print(f"Error: {error_msg}")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Generate summary
        summary = self._generate_summary(model_performances, test_results)

        return BenchmarkResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            model_performances=model_performances,
            test_results=test_results,
            summary=summary,
            errors=errors,
        )

    async def _run_model_tests(self, model_name: str, tests: List[BenchmarkTest]) -> tuple:
        """Run all tests for a single model."""
        model_test_results = {}
        model_errors = []

        # Aggregate metrics
        total_tests = len(tests)
        passed_tests = 0
        total_duration = 0.0
        total_tokens = 0
        total_cost = 0.0

        for test in tests:
            print(f"  Running test: {test.test_id}")

            try:
                test_result = await self._run_single_test(model_name, test)
                model_test_results[test.test_id] = test_result

                if test_result.error:
                    self._add_model_error(model_errors, test, model_name, f"reported error: {test_result.error}")

                if test_result.passed:
                    passed_tests += 1
                total_duration += test_result.duration
                total_tokens += test_result.token_usage.get("total", 0)
                total_cost += test_result.cost

            except Exception as e:
                self._add_model_error(model_errors, test, model_name, f"failed with error: {e}")
                print(f"    Error: {model_errors[-1]}")

                # Create failed test result
                test_result = BenchmarkTestResult(
                    test_id=test.test_id,
                    model=model_name,
                    passed=False,
                    score=0.0,
                    duration=0.0,
                    output="",
                    expected_output=test.expected_output or "",
                    error=str(e),
                    token_usage={},
                    cost=0.0,
                    metrics={},
                )
                model_test_results[test.test_id] = test_result

        # Calculate overall metrics
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        avg_duration = total_duration / total_tests if total_tests > 0 else 0.0

        model_performance = ModelPerformance(
            model=model_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            accuracy=accuracy,
            average_duration=avg_duration,
            total_duration=total_duration,
            total_tokens=total_tokens,
            total_cost=total_cost,
            scores_by_category={},  # Will be calculated in summary
        )

        return model_performance, model_test_results, model_errors

    async def _run_single_test(self, model_name: str, test: BenchmarkTest) -> BenchmarkTestResult:
        """Run all test cases for a single agent against a model."""
        start_time = time.time()

        if not test.test_cases:
            raise ValueError(f"No test cases found for {test.test_id}")

        try:
            # Parse the original agent file to extract clean YAML and content
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

            # Extract original YAML and parse it
            import yaml

            original_yaml = "\n".join(lines[1:yaml_end])
            yaml_data = yaml.safe_load(original_yaml) or {}
            markdown_content = "\n".join(lines[yaml_end + 1 :])

            # Create clean YAML with only agent fields
            clean_yaml_data = {
                "name": yaml_data.get("name", test.test_id),
                "description": yaml_data.get("description", ""),
                "model": model_name,  # Use the target model
                "max_steps": yaml_data.get("max_steps", 5),
                "tools": yaml_data.get("tools", []),
                "prefetch": yaml_data.get("prefetch", []),
                "permissions_profile": yaml_data.get("permissions_profile", "default"),
                "context_budget": yaml_data.get("context_budget"),
            }

            # Remove None values
            clean_yaml_data = {k: v for k, v in clean_yaml_data.items() if v is not None}

            # Create clean agent content
            clean_yaml = yaml.dump(clean_yaml_data, default_flow_style=False)
            clean_agent_content = f"---\n{clean_yaml}---\n\n{markdown_content}"

            # Write temporary agent file
            temp_agent_path = self.config.output_dir / f"temp_{test.test_id}_{model_name.replace(':', '_')}.md"
            temp_agent_path.write_text(clean_agent_content)

            # Run all test cases and aggregate results
            case_results = []
            total_passed = 0
            total_score = 0.0
            total_weight = 0.0
            aggregated_output = []
            raw_outputs = []

            try:
                for test_case in test.test_cases:
                    case_start = time.time()

                    # Prepare the prompt (add planning requirement if needed)
                    final_prompt = test_case.prompt
                    if test_case.requires_plan:
                        planning_instruction = """

PLANNING REQUIREMENT: Before executing this task, you must first create a detailed plan. Your response should be structured as follows:

1. **PLAN SECTION**: Start with a clear, step-by-step plan that outlines how you will approach this task
2. **EXECUTION SECTION**: Then proceed with executing the plan

Make sure your plan includes the key steps and reasoning before you start executing."""
                        final_prompt = test_case.prompt + planning_instruction

                    # Run the agent with the (possibly modified) prompt
                    result = run_agent(
                        agent_path=temp_agent_path,
                        prompt=final_prompt,
                        model_override=model_name,
                        debug=False,
                    )

                    case_duration = time.time() - case_start

                    # Evaluate this test case
                    case_evaluation = await self._evaluate_test_case(test_case, result, case_duration)

                    case_results.append(case_evaluation)
                    raw_outputs.append(str(result))
                    aggregated_output.append(
                        f"Test Case: {test_case.name}\nPrompt: {test_case.prompt}\nOutput: {result}\n---\n"
                    )

                    # Aggregate scoring
                    if case_evaluation["passed"]:
                        total_passed += 1
                    total_score += case_evaluation["score"] * test_case.weight
                    total_weight += test_case.weight

                end_time = time.time()
                duration = end_time - start_time

                # Calculate aggregate results
                overall_passed = total_passed == len(test.test_cases)  # All must pass
                overall_score = total_score / total_weight if total_weight > 0 else 0.0

                # Aggregate metrics
                aggregate_metrics = {
                    "case_results": case_results,
                    "cases_passed": total_passed,
                    "total_cases": len(test.test_cases),
                    "pass_rate": total_passed / len(test.test_cases),
                }

                # Use first case's expected output for legacy compatibility
                expected_output = test.test_cases[0].expected_output if test.test_cases else ""

                output_content = "\n".join(aggregated_output)
                if len(raw_outputs) == 1:
                    output_content = raw_outputs[0]

                return BenchmarkTestResult(
                    test_id=test.test_id,
                    model=model_name,
                    passed=overall_passed,
                    score=overall_score,
                    duration=duration,
                    output=output_content,
                    expected_output=expected_output or "",
                    error=None,
                    token_usage={},  # TODO: Aggregate from individual cases
                    cost=0.0,  # TODO: Aggregate from individual cases
                    metrics=aggregate_metrics,
                )

            finally:
                # Clean up temp file
                if temp_agent_path.exists():
                    temp_agent_path.unlink()

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            return BenchmarkTestResult(
                test_id=test.test_id,
                model=model_name,
                passed=False,
                score=0.0,
                duration=duration,
                output="",
                expected_output="",
                error=str(e),
                token_usage={},
                cost=0.0,
                metrics={},
            )

    async def _evaluate_test_case(self, test_case: TestCase, result: str, duration: float) -> Dict[str, Any]:
        """Evaluate a single test case result."""
        evaluation = {
            "test_case": test_case.name,
            "passed": False,
            "score": 0.0,
            "metrics": {},
        }

        try:
            # Basic correctness evaluation
            if test_case.expected_output:
                correctness = self.correctness_evaluator.evaluate(
                    output=result,
                    expected=test_case.expected_output,
                    output_type="string",
                )
                evaluation["passed"] = correctness["passed"]
                evaluation["score"] = correctness["score"]
                evaluation["metrics"]["correctness"] = correctness

            # Expected behaviors evaluation
            if test_case.expected_behaviors:
                behavior_score = 0.0
                behavior_checks = {}

                for behavior, expected_value in test_case.expected_behaviors.items():
                    if behavior == "tool_usage":
                        # Simple check for tool usage indication in output
                        tool_used = any(keyword in result.lower() for keyword in ["using tool", "calling", "executed"])
                        behavior_checks[behavior] = tool_used == expected_value
                        if tool_used == expected_value:
                            behavior_score += 0.3

                    elif behavior == "file_created":
                        # Simple check for file creation indication
                        file_created = any(keyword in result.lower() for keyword in ["created", "saved", "written"])
                        behavior_checks[behavior] = file_created == expected_value
                        if file_created == expected_value:
                            behavior_score += 0.3

                evaluation["metrics"]["behaviors"] = behavior_checks
                if not test_case.expected_output:  # Use behavior score if no exact output expected
                    evaluation["score"] = behavior_score
                    evaluation["passed"] = behavior_score >= 0.6

            # Custom evaluation criteria
            if test_case.evaluation:
                custom_score = 0.0
                custom_checks = {}

                for criterion, expected_value in test_case.evaluation.items():
                    if criterion == "exact_match":
                        custom_checks[criterion] = result.strip() == expected_value
                    elif criterion == "contains":
                        if isinstance(expected_value, list):
                            custom_checks[criterion] = all(item in result for item in expected_value)
                        else:
                            custom_checks[criterion] = expected_value in result
                    elif criterion == "content_pattern":
                        import re

                        custom_checks[criterion] = bool(re.search(expected_value, result))
                    elif criterion == "min_length":
                        custom_checks[criterion] = len(result) >= expected_value
                    else:
                        # Default to simple equality check
                        custom_checks[criterion] = str(expected_value).lower() in result.lower()

                    if custom_checks[criterion]:
                        custom_score += 1.0 / len(test_case.evaluation)

                evaluation["metrics"]["custom"] = custom_checks
                if not test_case.expected_output and not test_case.expected_behaviors:
                    evaluation["score"] = custom_score
                    evaluation["passed"] = custom_score >= 0.7

            # LLM evaluation for test case (if enabled)
            if test_case.use_llm_evaluation and test_case.llm_evaluation_criteria:
                llm_eval = await self.llm_evaluator.evaluate(
                    output=result,
                    task_description=test_case.prompt,
                    evaluation_criteria=test_case.llm_evaluation_criteria,
                    rubric=test_case.llm_evaluation_rubric or {},
                )
                evaluation["metrics"]["llm_evaluation"] = llm_eval

                # Use LLM evaluation if no other scoring method was successful
                if evaluation["score"] == 0.0:
                    evaluation["score"] = llm_eval["llm_score"]
                    evaluation["passed"] = llm_eval["llm_score"] >= 0.7
                else:
                    # Blend LLM score with existing evaluation (LLM weighted at 40%)
                    blended_score = (evaluation["score"] * 0.6) + (llm_eval["llm_score"] * 0.4)
                    evaluation["score"] = blended_score

            # Plan evaluation (if planning was required)
            if test_case.requires_plan:
                plan_score = 0.0
                plan_checks = {}

                # Check if there's evidence of planning in the output
                plan_indicators = [
                    "plan",
                    "step",
                    "approach",
                    "strategy",
                    "outline",
                    "first",
                    "then",
                    "next",
                    "finally",
                    "before",
                    "after",
                ]

                has_plan_structure = any(indicator in result.lower() for indicator in plan_indicators)
                plan_checks["has_planning_structure"] = has_plan_structure
                if has_plan_structure:
                    plan_score += 0.3

                # Check for specific plan elements if specified
                if test_case.expected_plan_elements:
                    elements_found = 0
                    for element in test_case.expected_plan_elements:
                        element_present = element.lower() in result.lower()
                        plan_checks[f"element_{element}"] = element_present
                        if element_present:
                            elements_found += 1

                    element_score = elements_found / len(test_case.expected_plan_elements)
                    plan_score += element_score * 0.4

                # Check custom plan evaluation criteria
                if test_case.plan_evaluation:
                    custom_plan_score = 0.0
                    for criterion, expected_value in test_case.plan_evaluation.items():
                        if criterion == "min_plan_steps":
                            # Count numbered or bulleted steps
                            import re

                            steps = len(re.findall(r"(\d+\.|•|-|\*)\s", result))
                            plan_checks[criterion] = steps >= expected_value
                            if steps >= expected_value:
                                custom_plan_score += 1.0 / len(test_case.plan_evaluation)
                        elif criterion == "structured_sections":
                            # Look for section headers
                            sections = any(
                                header in result.upper()
                                for header in [
                                    "PLAN",
                                    "EXECUTION",
                                    "APPROACH",
                                    "STRATEGY",
                                ]
                            )
                            plan_checks[criterion] = sections == expected_value
                            if sections == expected_value:
                                custom_plan_score += 1.0 / len(test_case.plan_evaluation)
                        else:
                            # Default criterion check
                            criterion_met = str(expected_value).lower() in result.lower()
                            plan_checks[criterion] = criterion_met
                            if criterion_met:
                                custom_plan_score += 1.0 / len(test_case.plan_evaluation)

                    plan_score += custom_plan_score * 0.3

                evaluation["metrics"]["planning"] = {
                    "plan_score": plan_score,
                    "plan_checks": plan_checks,
                    "required": True,
                }

                # Adjust overall score to include planning
                if evaluation["score"] == 0.0:  # If no other scoring was done
                    evaluation["score"] = plan_score
                    evaluation["passed"] = plan_score >= 0.6
                else:
                    # Blend planning score with existing score (planning weighted at 30%)
                    blended_score = (evaluation["score"] * 0.7) + (plan_score * 0.3)
                    evaluation["score"] = blended_score
                    evaluation["passed"] = (
                        evaluation["passed"] and plan_score >= 0.4
                    )  # Plan must meet minimum threshold

        except Exception as e:
            evaluation["error"] = str(e)

        return evaluation

    async def _evaluate_test_result(self, test: BenchmarkTest, result: str, duration: float) -> Dict[str, Any]:
        """Evaluate test result using configured evaluators."""
        evaluation = {
            "passed": False,
            "score": 0.0,
            "metrics": {},
            "token_usage": {},
            "cost": 0.0,
        }

        try:
            # Correctness evaluation
            if test.expected_output:
                correctness = self.correctness_evaluator.evaluate(
                    output=result,
                    expected=test.expected_output,
                    output_type=test.expected_type,
                )
                evaluation["passed"] = correctness["passed"]
                evaluation["score"] = correctness["score"]
                evaluation["metrics"]["correctness"] = correctness

            # Performance evaluation
            performance = self.performance_evaluator.evaluate(
                duration=duration,
                timeout=test.timeout,
            )
            evaluation["metrics"]["performance"] = performance

            # Quality evaluation (if no expected output for exact matching)
            if not test.expected_output and test.evaluation_criteria:
                quality = await self.quality_evaluator.evaluate(
                    output=result,
                    criteria=test.evaluation_criteria,
                )
                evaluation["metrics"]["quality"] = quality
                if not evaluation["passed"]:  # Use quality score if no correctness check
                    evaluation["passed"] = quality["score"] >= 0.7
                    evaluation["score"] = quality["score"]

            # LLM evaluation (for complex tasks without predetermined outputs)
            if test.use_llm_evaluation and test.llm_evaluation_criteria:
                expected_format = "" if not test.expected_type or test.expected_type == "string" else test.expected_type

                llm_eval = await self.llm_evaluator.evaluate(
                    output=result,
                    task_description=test.description,
                    evaluation_criteria=test.llm_evaluation_criteria,
                    expected_format=expected_format,
                    rubric=test.llm_evaluation_rubric or {},
                )
                evaluation["metrics"]["llm_evaluation"] = llm_eval

                # Use LLM evaluation as primary scoring if no expected output
                if not test.expected_output:
                    evaluation["passed"] = llm_eval["llm_score"] >= 0.7
                    evaluation["score"] = llm_eval["llm_score"]
                else:
                    # Blend LLM score with correctness score if both available
                    evaluation["score"] = (evaluation["score"] + llm_eval["llm_score"]) / 2

            # Estimate token usage and cost (simplified)
            evaluation["token_usage"] = {
                "input": len(result.split()) * 1.3,  # Rough estimate
                "output": len(result.split()),
                "total": len(result.split()) * 2.3,
            }
            evaluation["cost"] = evaluation["token_usage"]["total"] * 0.00001  # Rough estimate

        except Exception as e:
            evaluation["error"] = str(e)

        return evaluation

    def _generate_summary(
        self,
        model_performances: Dict[str, ModelPerformance],
        test_results: Dict[str, Dict[str, BenchmarkTestResult]],
    ) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            "total_models": len(model_performances),
            "model_rankings": [],
            "category_performance": {},
            "best_model": None,
            "worst_model": None,
            "average_accuracy": 0.0,
            "total_tests": 0,
        }

        if not model_performances:
            return summary

        # Calculate model rankings by accuracy
        ranked_models = sorted(model_performances.items(), key=lambda x: x[1].accuracy, reverse=True)

        summary["model_rankings"] = [
            {
                "model": model,
                "accuracy": perf.accuracy,
                "avg_duration": perf.average_duration,
                "total_cost": perf.total_cost,
            }
            for model, perf in ranked_models
        ]

        summary["best_model"] = ranked_models[0][0] if ranked_models else None
        summary["worst_model"] = ranked_models[-1][0] if ranked_models else None

        # Calculate average accuracy
        total_accuracy = sum(perf.accuracy for perf in model_performances.values())
        summary["average_accuracy"] = total_accuracy / len(model_performances)

        # Get total tests from first model
        if model_performances:
            first_model = next(iter(model_performances.values()))
            summary["total_tests"] = first_model.total_tests

        return summary
