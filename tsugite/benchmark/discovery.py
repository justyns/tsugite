"""Test discovery and parsing for benchmark framework."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import parse_yaml_frontmatter
from .utils import (
    extract_block,
    extract_inline_field,
    extract_prompt_from_markdown,
    parse_bullet_list,
    parse_key_value_block,
)


@dataclass
class TestCase:
    """Individual test case within a benchmark test."""

    name: str
    prompt: str
    expected_output: Optional[str] = None
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


class TestDiscovery:
    """Handles discovery and parsing of benchmark tests."""

    def __init__(self, benchmark_dir: Path, default_timeout: int = 120):
        """Initialize test discovery.

        Args:
            benchmark_dir: Root directory containing benchmark tests
            default_timeout: Default timeout for tests in seconds
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.default_timeout = default_timeout
        self.test_cache: Dict[str, List[BenchmarkTest]] = {}

    def discover_tests(
        self, categories: Optional[List[str]] = None, agent_path: Optional[Path] = None
    ) -> List[BenchmarkTest]:
        """Discover agent + test.md pairs in specified categories or for a specific agent.

        Args:
            categories: List of category names to search (e.g., ["basic", "tools"])
            agent_path: Specific agent file to test (bypasses category search)

        Returns:
            List of discovered benchmark tests
        """
        # Test a specific agent
        if agent_path:
            agent_path = Path(agent_path)
            category = agent_path.parent.name if agent_path.parent.parent == self.benchmark_dir else "custom"
            test_file = agent_path.with_suffix(".test.md")

            if test_file.exists():
                return [self._parse_agent_test_pair(agent_path, test_file, category)]
            return [self._parse_benchmark_test(agent_path, category)]

        # Discover tests from categories
        if categories is None:
            categories = ["basic"]

        tests = []
        for category in categories:
            # Check cache first
            if category in self.test_cache:
                tests.extend(self.test_cache[category])
                continue

            # Discover from filesystem
            category_dir = self.benchmark_dir / category
            if not category_dir.exists():
                print(f"Warning: Category directory not found: {category_dir}")
                continue

            category_tests = self._discover_category_tests(category)
            self.test_cache[category] = category_tests
            tests.extend(category_tests)

        return tests

    def _discover_category_tests(self, category: str) -> List[BenchmarkTest]:
        """Discover all tests in a category directory.

        Args:
            category: Category name

        Returns:
            List of tests in this category
        """
        category_dir = self.benchmark_dir / category
        category_tests = []

        for agent_file in category_dir.glob("*.md"):
            # Skip test definition files and documentation
            if agent_file.name.endswith(".test.md") or agent_file.name == "README.md":
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

        return category_tests

    def _parse_agent_test_pair(self, agent_path: Path, test_path: Path, category: str) -> BenchmarkTest:
        """Parse an agent + test.md pair into a BenchmarkTest.

        Args:
            agent_path: Path to agent definition file
            test_path: Path to test definition file
            category: Test category

        Returns:
            Parsed benchmark test

        Raises:
            ValueError: If parsing fails
        """
        try:
            test_metadata, markdown_content = parse_yaml_frontmatter(test_path.read_text(), "Test file")

            common = self._extract_common_test_fields(
                test_metadata,
                fallback_id=agent_path.stem,
                fallback_description=f"Test for {agent_path.stem}",
            )

            test_cases = self._parse_test_cases(markdown_content)

            # Validate agent file exists
            if not agent_path.exists():
                raise ValueError(f"Agent file not found: {agent_path}")

            # Use first test case's expected output if no global output specified
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
        """Parse a single benchmark markdown file with embedded expectations.

        Args:
            agent_path: Path to agent/benchmark file
            category: Test category

        Returns:
            Parsed benchmark test

        Raises:
            ValueError: If parsing fails
        """
        try:
            metadata, markdown_content = parse_yaml_frontmatter(agent_path.read_text(), "Benchmark file")

            common = self._extract_common_test_fields(
                metadata,
                fallback_id=agent_path.stem,
                fallback_description=metadata.get("name", ""),
            )

            prompt = metadata.get("prompt") or extract_prompt_from_markdown(markdown_content)

            test_case = TestCase(
                name=f"{common['test_id']}_default",
                prompt=prompt,
                expected_output=common["expected_output"],
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

    def _extract_common_test_fields(
        self,
        metadata: Dict[str, Any],
        fallback_id: str,
        fallback_description: str,
    ) -> Dict[str, Any]:
        """Extract common metadata fields from test definition.

        Args:
            metadata: Parsed YAML frontmatter
            fallback_id: ID to use if not specified
            fallback_description: Description to use if not specified

        Returns:
            Dictionary of common fields
        """
        weight_value = metadata.get("weight", 1.0)
        try:
            weight = float(weight_value)
        except (TypeError, ValueError):
            weight = 1.0

        requires_tools = metadata.get("requires_tools", metadata.get("tools", []))

        return {
            "test_id": metadata.get("test_id", fallback_id),
            "description": metadata.get("description", fallback_description),
            "timeout": metadata.get("timeout", self.default_timeout),
            "requires_tools": requires_tools,
            "weight": weight,
            "expected_output": metadata.get("expected_output"),
            "expected_type": metadata.get("expected_type", "string"),
            "evaluation_criteria": metadata.get("evaluation_criteria", {}),
        }

    def _parse_test_cases(self, markdown_content: str) -> List[TestCase]:
        """Parse test cases from markdown content.

        Args:
            markdown_content: Markdown content containing test cases

        Returns:
            List of parsed test cases
        """
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
        """Parse a single test case from markdown content.

        Args:
            name: Test case name
            content: Test case markdown content

        Returns:
            Parsed test case or None if parsing fails
        """
        # Extract prompt - required
        prompt = extract_inline_field(content, "Prompt")
        if not prompt:
            return None

        # Extract expected output
        expected_output = extract_inline_field(content, "Expected Output")

        # Parse evaluation criteria
        evaluation = parse_key_value_block(extract_block(content, "Evaluation"))

        # Parse planning requirements
        requires_plan_text = extract_inline_field(content, "Requires Plan")
        requires_plan = bool(requires_plan_text and requires_plan_text.strip().lower() in {"true", "yes", "1"})

        expected_plan_elements = parse_bullet_list(extract_block(content, "Expected Plan Elements"))
        plan_evaluation = parse_key_value_block(extract_block(content, "Plan Evaluation"))

        # Parse LLM evaluation fields
        use_llm_evaluation_text = extract_inline_field(content, "Use LLM Evaluation")
        use_llm_evaluation = bool(
            use_llm_evaluation_text and use_llm_evaluation_text.strip().lower() in {"true", "yes", "1"}
        )

        llm_evaluation_criteria = extract_inline_field(content, "LLM Evaluation Criteria") or ""

        llm_rubric_block = extract_block(content, "LLM Evaluation Rubric")
        llm_evaluation_rubric = parse_key_value_block(llm_rubric_block)

        return TestCase(
            name=name,
            prompt=prompt,
            expected_output=expected_output,
            evaluation=evaluation,
            weight=1.0,
            requires_plan=requires_plan,
            expected_plan_elements=expected_plan_elements,
            plan_evaluation=plan_evaluation,
            use_llm_evaluation=use_llm_evaluation,
            llm_evaluation_criteria=llm_evaluation_criteria,
            llm_evaluation_rubric=llm_evaluation_rubric,
        )
