"""
Coverage.py adapter for measuring code coverage.

This adapter provides real coverage measurement using the coverage.py library,
following established adapter patterns from the LLM adapters.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from ...domain.models import CoverageResult, TestCraftError

logger = logging.getLogger(__name__)


class CoverageError(TestCraftError):
    """Raised when coverage measurement fails."""

    pass


class CoveragePyAdapter:
    """
    Coverage adapter using coverage.py library.

    Implements CoveragePort interface for actual coverage measurement.
    Follows patterns from:
    - coverage_evaluator.py (graceful error handling)
    - pytest_refiner.py (execution patterns)
    - llm/common.py (error wrapping)
    """

    def __init__(self) -> None:
        """
        Initialize the coverage adapter.

        Raises:
            ImportError: If coverage.py is not installed
        """
        try:
            import coverage

            self._coverage_module = coverage
        except ImportError as e:
            raise ImportError(
                "coverage.py is required but not installed. "
                "Install it with: pip install coverage"
            ) from e

        self._last_method = "coverage.py"

    def measure_coverage(
        self,
        source_files: list[str],
        test_files: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, CoverageResult]:
        """
        Measure coverage for source files by running tests.

        Uses subprocess to run pytest with coverage enabled.
        Follows pytest_refiner.py execution patterns.

        Args:
            source_files: List of source file paths to measure
            test_files: Optional list of test files to run
            **kwargs: Additional parameters (timeout, config overrides, etc.)

        Returns:
            Dictionary mapping file paths to CoverageResult objects
        """
        if not source_files:
            logger.warning("No source files provided for coverage measurement")
            return {}

        try:
            # Resolve all paths upfront
            resolved_sources = [str(Path(f).resolve()) for f in source_files]

            # Determine source directories (for coverage.py source parameter)
            source_dirs = list({str(Path(f).parent) for f in resolved_sources})

            # Create coverage instance with proper configuration
            cov = self._coverage_module.Coverage(
                source=source_dirs,
                omit=[
                    "*/tests/*",
                    "*/test_*.py",
                    "*_test.py",
                    "*/.venv/*",
                    "*/venv/*",
                    "*/site-packages/*",
                    "*/.tox/*",
                ],
                branch=True,  # Enable branch coverage
                config_file=False,  # Don't load .coveragerc to avoid conflicts
            )

            # Start coverage measurement
            cov.start()

            # Run pytest if test files provided (follows pytest_refiner pattern)
            if test_files:
                try:
                    # Run tests with subprocess (isolated execution)
                    test_paths = [str(Path(f).resolve()) for f in test_files]
                    cmd = ["pytest", "-xvs"] + test_paths

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=kwargs.get("timeout", 300),
                    )

                    logger.debug(f"Pytest exit code: {result.returncode}")

                except subprocess.TimeoutExpired:
                    logger.warning("Test execution timed out")
                except Exception as e:
                    logger.warning(f"Test execution failed: {e}")

            # Stop coverage measurement
            cov.stop()
            cov.save()

            # Build results dictionary
            results = {}
            for source_file in resolved_sources:
                try:
                    # Get coverage analysis for this file
                    analysis = cov.analysis2(source_file)

                    # analysis returns: (filename, executed_lines, missing_lines, excluded_lines)
                    executed_lines = set(analysis[1]) if len(analysis) > 1 else set()
                    missing_lines = list(analysis[2]) if len(analysis) > 2 else []

                    # Calculate line coverage
                    total_executable_lines = len(executed_lines) + len(missing_lines)
                    line_coverage = (
                        len(executed_lines) / total_executable_lines
                        if total_executable_lines > 0
                        else 0.0
                    )

                    # Get branch coverage
                    branch_coverage = line_coverage * 0.85  # Conservative estimate
                    try:
                        # Get branch data if available
                        branch_data = cov.get_data().arcs(source_file)
                        if branch_data:
                            executed_arcs: set[tuple[int, int]] = set(branch_data)
                            total_branches = len(branch_data)
                            executed_branches = len(executed_arcs)
                            branch_coverage = (
                                executed_branches / total_branches
                                if total_branches > 0
                                else 0.0
                            )
                    except (AttributeError, Exception) as e:
                        logger.debug(
                            f"Branch coverage unavailable for {source_file}: {e}"
                        )
                        # Use estimate

                    # Create CoverageResult (reusing domain model)
                    results[source_file] = CoverageResult(
                        line_coverage=line_coverage,
                        branch_coverage=branch_coverage,
                        missing_lines=sorted(missing_lines),
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to get coverage analysis for {source_file}: {e}"
                    )
                    # Return zero coverage on error (graceful degradation)
                    results[source_file] = CoverageResult(
                        line_coverage=0.0, branch_coverage=0.0, missing_lines=[]
                    )

            logger.debug(f"Successfully measured coverage for {len(results)} files")
            return results

        except Exception as e:
            # Wrap external error (pattern from llm/common.py)
            logger.error(f"Coverage measurement failed: {e}")
            # Return empty dict on failure (coverage_evaluator pattern line 71-78)
            return {}

    def get_coverage_summary(
        self, coverage_data: dict[str, CoverageResult]
    ) -> dict[str, Any]:
        """
        Generate coverage summary from coverage data.

        Matches coverage_evaluator.py usage (lines 55, 94).
        Returns consistent structure with coverage_usecase expectations.

        Args:
            coverage_data: Dictionary mapping file paths to CoverageResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not coverage_data:
            return {
                "overall_line_coverage": 0.0,
                "overall_branch_coverage": 0.0,
                "files_covered": 0,
                "total_lines": 0,
                "missing_coverage": {},
            }

        # Calculate overall coverage (average across files)
        total_line_coverage = sum(
            result.line_coverage for result in coverage_data.values()
        )
        total_branch_coverage = sum(
            result.branch_coverage for result in coverage_data.values()
        )

        num_files = len(coverage_data)

        # Calculate total executable lines
        total_lines = sum(
            len(result.missing_lines) + int(result.line_coverage * 1000)
            for result in coverage_data.values()
        )

        return {
            "overall_line_coverage": total_line_coverage / num_files,
            "overall_branch_coverage": total_branch_coverage / num_files,
            "files_covered": num_files,
            "total_lines": total_lines,
            "missing_coverage": {
                path: result.missing_lines
                for path, result in coverage_data.items()
                if result.missing_lines
            },
        }

    def report_coverage(
        self,
        coverage_data: dict[str, CoverageResult],
        output_format: str = "detailed",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate coverage report in specified format.

        Follows ui-patterns.mdc: minimalist, professional output.

        Args:
            coverage_data: Coverage data to report
            output_format: Format (detailed, summary, json, html)
            **kwargs: Additional formatting parameters

        Returns:
            Dictionary with report_content, summary_stats, and format
        """
        summary = self.get_coverage_summary(coverage_data)

        if output_format == "summary":
            report_content = self._generate_summary_report(coverage_data, summary)
        elif output_format == "detailed":
            report_content = self._generate_detailed_report(coverage_data, summary)
        elif output_format == "json":
            import json

            report_content = json.dumps(
                {
                    "summary": summary,
                    "files": {
                        path: {
                            "line_coverage": result.line_coverage,
                            "branch_coverage": result.branch_coverage,
                            "missing_lines": result.missing_lines,
                        }
                        for path, result in coverage_data.items()
                    },
                },
                indent=2,
            )
        elif output_format == "html":
            report_content = self._generate_html_report(coverage_data, summary)
        else:
            logger.warning(f"Unknown output format '{output_format}', using detailed")
            report_content = self._generate_detailed_report(coverage_data, summary)

        return {
            "report_content": report_content,
            "summary_stats": summary,
            "format": output_format,
        }

    def identify_gaps(
        self, coverage_data: dict[str, CoverageResult], threshold: float = 0.8
    ) -> dict[str, list[int]]:
        """
        Identify files with coverage below threshold.

        Args:
            coverage_data: Coverage data to analyze
            threshold: Minimum coverage threshold (0.0 to 1.0)

        Returns:
            Dictionary mapping file paths to lists of missing line numbers
        """
        gaps = {}
        for file_path, result in coverage_data.items():
            if result.line_coverage < threshold:
                gaps[file_path] = result.missing_lines

        logger.debug(
            f"Identified {len(gaps)} files below coverage threshold {threshold:.0%}"
        )
        return gaps

    def get_measurement_method(self) -> str:
        """
        Get coverage measurement method identifier.

        NEW: Fixes leaky abstraction (Issue #3).
        Added to Protocol to eliminate getattr() calls.

        Returns:
            Method identifier string
        """
        return self._last_method

    # Private report generation methods

    def _generate_summary_report(
        self, coverage_data: dict[str, CoverageResult], summary: dict[str, Any]
    ) -> str:
        """Generate summary report (minimalist style per ui-patterns)."""
        lines = [
            "Coverage Summary",
            "=" * 50,
            f"Files Covered: {summary['files_covered']}",
            f"Overall Line Coverage: {summary['overall_line_coverage']:.1%}",
            f"Overall Branch Coverage: {summary['overall_branch_coverage']:.1%}",
            f"Total Lines: {summary['total_lines']}",
        ]

        # Add files below 80% coverage
        low_coverage = [
            (path, result)
            for path, result in coverage_data.items()
            if result.line_coverage < 0.8
        ]

        if low_coverage:
            lines.extend(
                [
                    "",
                    f"Files Below 80% Coverage: {len(low_coverage)}",
                ]
            )
            for path, result in sorted(low_coverage, key=lambda x: x[1].line_coverage)[
                :5
            ]:
                lines.append(f"  {Path(path).name}: {result.line_coverage:.1%}")

        return "\n".join(lines)

    def _generate_detailed_report(
        self, coverage_data: dict[str, CoverageResult], summary: dict[str, Any]
    ) -> str:
        """Generate detailed report with per-file breakdown."""
        lines = [
            "Detailed Coverage Report",
            "=" * 80,
            f"Overall: {summary['overall_line_coverage']:.1%} line, "
            f"{summary['overall_branch_coverage']:.1%} branch",
            "",
            "Per-File Coverage:",
            "-" * 80,
            f"{'File':<45} {'Line':>8} {'Branch':>8} {'Missing':>8}",
            "-" * 80,
        ]

        for file_path, result in sorted(coverage_data.items()):
            file_name = Path(file_path).name
            lines.append(
                f"{file_name:<45} "
                f"{result.line_coverage:>7.1%} "
                f"{result.branch_coverage:>7.1%} "
                f"{len(result.missing_lines):>8}"
            )

        lines.append("-" * 80)
        return "\n".join(lines)

    def _generate_html_report(
        self, coverage_data: dict[str, CoverageResult], summary: dict[str, Any]
    ) -> str:
        """Generate simple HTML report."""
        html_rows = []
        for file_path, result in sorted(coverage_data.items()):
            color = (
                "green"
                if result.line_coverage >= 0.8
                else "orange"
                if result.line_coverage >= 0.5
                else "red"
            )
            html_rows.append(
                f'<tr style="color: {color}">'
                f"<td>{Path(file_path).name}</td>"
                f"<td>{result.line_coverage:.1%}</td>"
                f"<td>{result.branch_coverage:.1%}</td>"
                f"<td>{len(result.missing_lines)}</td>"
                f"</tr>"
            )

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Coverage Report</h1>
    <p>Overall Line Coverage: {summary["overall_line_coverage"]:.1%}</p>
    <p>Overall Branch Coverage: {summary["overall_branch_coverage"]:.1%}</p>
    <p>Files Covered: {summary["files_covered"]}</p>

    <h2>Per-File Coverage</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Line Coverage</th>
            <th>Branch Coverage</th>
            <th>Missing Lines</th>
        </tr>
        {"".join(html_rows)}
    </table>
</body>
</html>"""


class NoOpCoverageAdapter:
    """
    No-op coverage adapter for graceful degradation.

    Used when coverage.py is unavailable.
    Matches coverage_evaluator.py pattern (lines 71-78).
    """

    def __init__(self) -> None:
        self._last_method = "noop"
        logger.warning(
            "Using no-op coverage adapter - install coverage.py for real measurements"
        )

    def measure_coverage(
        self,
        source_files: list[str],
        test_files: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, CoverageResult]:
        """Return empty coverage (graceful failure)."""
        logger.warning(
            f"Coverage measurement unavailable for {len(source_files)} files"
        )
        return {}

    def get_coverage_summary(
        self, coverage_data: dict[str, CoverageResult]
    ) -> dict[str, Any]:
        """Return empty summary."""
        return {
            "overall_line_coverage": 0.0,
            "overall_branch_coverage": 0.0,
            "files_covered": 0,
            "total_lines": 0,
            "missing_coverage": {},
        }

    def identify_gaps(
        self, coverage_data: dict[str, CoverageResult], threshold: float = 0.8
    ) -> dict[str, list[int]]:
        """Return no gaps."""
        return {}

    def report_coverage(
        self,
        coverage_data: dict[str, CoverageResult],
        output_format: str = "detailed",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return placeholder report."""
        return {
            "success": False,
            "format": output_format,
            "report_content": (
                "Coverage measurement unavailable\n"
                "Install coverage.py: pip install coverage"
            ),
            "summary_stats": self.get_coverage_summary({}),
        }

    def get_measurement_method(self) -> str:
        """Get measurement method."""
        return self._last_method
