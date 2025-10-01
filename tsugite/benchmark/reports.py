"""Report generation for benchmark results."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .core import BenchmarkResult
from .metrics import BenchmarkTestResult, ModelPerformance


class ReportGenerator:
    """Generate various formats of benchmark reports."""

    def __init__(self, result: BenchmarkResult):
        self.result = result

    def generate_json_report(self, output_path: Path) -> None:
        """Generate a comprehensive JSON report."""
        report_data = {
            "benchmark_info": {
                "start_time": self.result.start_time.isoformat(),
                "end_time": self.result.end_time.isoformat(),
                "total_duration": self.result.total_duration,
                "models_tested": list(self.result.model_performances.keys()),
                "total_tests": len(self._get_all_test_ids()),
            },
            "config": asdict(self.result.config),
            "summary": self.result.summary,
            "model_performances": {model: asdict(perf) for model, perf in self.result.model_performances.items()},
            "detailed_results": {
                model: {test_id: asdict(result) for test_id, result in tests.items()}
                for model, tests in self.result.test_results.items()
            },
            "errors": self.result.errors,
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def generate_markdown_report(self, output_path: Path) -> None:
        """Generate a markdown report for easy reading."""
        report = []

        # Header
        report.append("# Tsugite Benchmark Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append(f"- **Duration**: {self.result.total_duration:.2f} seconds")
        report.append(f"- **Models Tested**: {len(self.result.model_performances)}")
        report.append(f"- **Total Tests**: {self.result.summary.get('total_tests', 0)}")
        report.append(f"- **Average Accuracy**: {self.result.summary.get('average_accuracy', 0):.1%}")
        report.append("")

        # Model Rankings
        if "model_rankings" in self.result.summary:
            report.append("## Model Rankings")
            report.append("| Rank | Model | Accuracy | Avg Duration | Total Cost |")
            report.append("|------|-------|----------|--------------|------------|")

            for i, ranking in enumerate(self.result.summary["model_rankings"], 1):
                model = ranking["model"]
                accuracy = f"{ranking['accuracy']:.1%}"
                duration = f"{ranking['avg_duration']:.2f}s"
                cost = f"${ranking['total_cost']:.4f}"
                report.append(f"| {i} | {model} | {accuracy} | {duration} | {cost} |")
            report.append("")

        # Detailed Performance
        report.append("## Detailed Performance")
        for model, performance in self.result.model_performances.items():
            report.append(f"### {model}")
            report.append(
                f"- **Accuracy**: {performance.accuracy:.1%} ({performance.passed_tests}/{performance.total_tests})"
            )
            report.append(f"- **Average Duration**: {performance.average_duration:.2f}s")
            report.append(f"- **Total Cost**: ${performance.total_cost:.4f}")
            report.append(f"- **Performance Tier**: {self._get_performance_tier(performance.accuracy)}")
            report.append("")

        # Category Breakdown
        category_breakdown = self._calculate_category_breakdown()
        if category_breakdown:
            report.append("## Performance by Category")
            for category, data in category_breakdown.items():
                report.append(f"### {category.title()}")
                report.append(f"- **Best Model**: {data['best_model']} ({data['best_score']:.1%})")
                report.append(f"- **Average Score**: {data['average_score']:.1%}")
                report.append("")

        # Test Details
        report.append("## Test Results Details")
        for model in self.result.model_performances.keys():
            report.append(f"### {model}")
            report.append("| Test ID | Category | Result | Score | Duration |")
            report.append("|---------|----------|--------|-------|----------|")

            model_tests = self.result.test_results.get(model, {})
            for test_id, test_result in model_tests.items():
                category = self._get_test_category(test_id)
                result_status = "✅ PASS" if test_result.passed else "❌ FAIL"
                score = f"{test_result.score:.2f}"
                duration = f"{test_result.duration:.2f}s"
                report.append(f"| {test_id} | {category} | {result_status} | {score} | {duration} |")
            report.append("")

        # Errors
        if self.result.errors:
            report.append("## Errors")
            for error in self.result.errors:
                report.append(f"- {error}")
            report.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(report))

    def generate_html_report(self, output_path: Path) -> None:
        """Generate an interactive HTML report."""
        html_content = self._generate_html_content()

        with open(output_path, "w") as f:
            f.write(html_content)

    def generate_csv_summary(self, output_path: Path) -> None:
        """Generate CSV summary for data analysis."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Model",
                    "Test_ID",
                    "Category",
                    "Passed",
                    "Score",
                    "Duration",
                    "Token_Usage",
                    "Cost",
                    "Error",
                ]
            )

            # Data rows
            for model, tests in self.result.test_results.items():
                for test_id, test_result in tests.items():
                    category = self._get_test_category(test_id)
                    token_usage = test_result.token_usage.get("total", 0)

                    writer.writerow(
                        [
                            model,
                            test_id,
                            category,
                            test_result.passed,
                            test_result.score,
                            test_result.duration,
                            token_usage,
                            test_result.cost,
                            test_result.error or "",
                        ]
                    )

    def _get_all_test_ids(self) -> List[str]:
        """Get all unique test IDs across all models."""
        all_test_ids = set()
        for tests in self.result.test_results.values():
            all_test_ids.update(tests.keys())
        return list(all_test_ids)

    def _get_test_category(self, test_id: str) -> str:
        """Extract category from test ID."""
        if test_id.startswith("basic_"):
            return "basic"
        elif test_id.startswith("tools_"):
            return "tools"
        elif test_id.startswith("scenarios_"):
            return "scenarios"
        elif test_id.startswith("performance_"):
            return "performance"
        else:
            return "unknown"

    def _get_performance_tier(self, accuracy: float) -> str:
        """Get performance tier description."""
        if accuracy >= 0.9:
            return "Excellent"
        elif accuracy >= 0.75:
            return "Good"
        elif accuracy >= 0.6:
            return "Fair"
        elif accuracy >= 0.4:
            return "Poor"
        else:
            return "Very Poor"

    def _calculate_category_breakdown(self) -> Dict[str, Any]:
        """Calculate performance breakdown by category."""
        categories = {}

        for model, tests in self.result.test_results.items():
            for test_id, test_result in tests.items():
                category = self._get_test_category(test_id)

                if category not in categories:
                    categories[category] = {}

                if model not in categories[category]:
                    categories[category][model] = []

                categories[category][model].append(test_result.score)

        # Calculate averages and find best/worst
        breakdown = {}
        for category, model_scores in categories.items():
            avg_scores = {model: sum(scores) / len(scores) for model, scores in model_scores.items()}

            if avg_scores:
                best_model = max(avg_scores.items(), key=lambda x: x[1])
                average_score = sum(avg_scores.values()) / len(avg_scores)

                breakdown[category] = {
                    "best_model": best_model[0],
                    "best_score": best_model[1],
                    "average_score": average_score,
                    "model_scores": avg_scores,
                }

        return breakdown

    def _generate_html_content(self) -> str:
        """Generate HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tsugite Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .tier-excellent {{ color: #28a745; font-weight: bold; }}
        .tier-good {{ color: #17a2b8; }}
        .tier-fair {{ color: #ffc107; }}
        .tier-poor {{ color: #fd7e14; }}
        .tier-very-poor {{ color: #dc3545; }}
        .progress-bar {{
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .progress-excellent {{ background: #28a745; }}
        .progress-good {{ background: #17a2b8; }}
        .progress-fair {{ background: #ffc107; }}
        .progress-poor {{ background: #fd7e14; }}
        .progress-very-poor {{ background: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Tsugite Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <h2>📊 Summary</h2>
            <div class="metric">
                <div class="metric-value">{self.result.total_duration:.1f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(self.result.model_performances)}</div>
                <div class="metric-label">Models Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.result.summary.get('total_tests', 0)}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.result.summary.get('average_accuracy', 0):.1%}</div>
                <div class="metric-label">Average Accuracy</div>
            </div>
        </div>

        <h2>🏆 Model Rankings</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Performance</th>
                    <th>Avg Duration</th>
                    <th>Total Cost</th>
                </tr>
            </thead>
            <tbody>
        """

        # Model rankings table
        if "model_rankings" in self.result.summary:
            for i, ranking in enumerate(self.result.summary["model_rankings"], 1):
                accuracy = ranking["accuracy"]
                tier_class = self._get_tier_class(accuracy)
                progress_class = self._get_progress_class(accuracy)

                html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{ranking['model']}</strong></td>
                    <td>
                        {accuracy:.1%}
                        <div class="progress-bar">
                            <div class="progress-fill {progress_class}" style="width: {accuracy*100}%"></div>
                        </div>
                    </td>
                    <td><span class="{tier_class}">{self._get_performance_tier(accuracy)}</span></td>
                    <td>{ranking['avg_duration']:.2f}s</td>
                    <td>${ranking['total_cost']:.4f}</td>
                </tr>
                """

        html += """
            </tbody>
        </table>

        <h2>📈 Detailed Results</h2>
        """

        # Detailed results for each model
        for model, performance in self.result.model_performances.items():
            tier_class = self._get_tier_class(performance.accuracy)
            html += f"""
            <h3>{model}</h3>
            <p>
                <span class="{tier_class}">
                    {self._get_performance_tier(performance.accuracy)}
                </span>
                - {performance.accuracy:.1%} accuracy
                ({performance.passed_tests}/{performance.total_tests} tests passed)
            </p>
            """

        html += """
        </div>
    </body>
</html>
        """

        return html

    def _get_tier_class(self, accuracy: float) -> str:
        """Get CSS class for performance tier."""
        if accuracy >= 0.9:
            return "tier-excellent"
        elif accuracy >= 0.75:
            return "tier-good"
        elif accuracy >= 0.6:
            return "tier-fair"
        elif accuracy >= 0.4:
            return "tier-poor"
        else:
            return "tier-very-poor"

    def _get_progress_class(self, accuracy: float) -> str:
        """Get CSS class for progress bar."""
        if accuracy >= 0.9:
            return "progress-excellent"
        elif accuracy >= 0.75:
            return "progress-good"
        elif accuracy >= 0.6:
            return "progress-fair"
        elif accuracy >= 0.4:
            return "progress-poor"
        else:
            return "progress-very-poor"
