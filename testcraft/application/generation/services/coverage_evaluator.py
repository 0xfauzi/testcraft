"""
Coverage evaluation service.

Handles coverage measurement, delta calculation, and coverage-based
decision making for test generation workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....ports.coverage_port import CoveragePort
from ....ports.telemetry_port import TelemetryPort

logger = logging.getLogger(__name__)


class CoverageEvaluator:
    """
    Service for coverage measurement and evaluation.

    Provides unified coverage measurement, delta calculation, and
    coverage-based decision making for the test generation workflow.
    """

    def __init__(
        self, coverage_port: CoveragePort, telemetry_port: TelemetryPort
    ) -> None:
        """
        Initialize the coverage evaluator.

        Args:
            coverage_port: Port for coverage measurement operations
            telemetry_port: Port for telemetry operations
        """
        self._coverage = coverage_port
        self._telemetry = telemetry_port

    def measure_initial(self, source_files: list[Path]) -> dict[str, Any]:
        """
        Measure initial code coverage before test generation.

        Args:
            source_files: List of source files to measure coverage for

        Returns:
            Coverage measurement results with graceful failure handling
        """
        with self._telemetry.create_child_span("measure_initial_coverage") as span:
            try:
                file_paths = [str(f) for f in source_files]
                coverage_data = self._coverage.measure_coverage(file_paths)
                summary = self._coverage.get_coverage_summary(coverage_data)

                span.set_attribute("files_measured", len(coverage_data))
                span.set_attribute(
                    "overall_coverage", summary.get("overall_line_coverage", 0)
                )

                return summary

            except Exception as e:
                logger.warning("Failed to measure initial coverage: %s", e)
                # Return empty coverage data rather than failing
                return {
                    "overall_line_coverage": 0.0,
                    "overall_branch_coverage": 0.0,
                    "files_covered": 0,
                    "total_lines": 0,
                    "missing_coverage": {},
                }

    def measure_final(self, source_files: list[Path]) -> dict[str, Any]:
        """
        Measure final code coverage after test generation.

        Args:
            source_files: List of source files to measure coverage for

        Returns:
            Coverage measurement results with graceful failure handling
        """
        with self._telemetry.create_child_span("measure_final_coverage") as span:
            try:
                file_paths = [str(f) for f in source_files]
                coverage_data = self._coverage.measure_coverage(file_paths)
                summary = self._coverage.get_coverage_summary(coverage_data)

                span.set_attribute("files_measured", len(coverage_data))
                span.set_attribute(
                    "final_coverage", summary.get("overall_line_coverage", 0)
                )

                return summary

            except Exception as e:
                logger.warning("Failed to measure final coverage: %s", e)
                # Return empty coverage data rather than failing
                return {
                    "overall_line_coverage": 0.0,
                    "overall_branch_coverage": 0.0,
                    "files_covered": 0,
                    "total_lines": 0,
                    "missing_coverage": {},
                }

    def calculate_delta(
        self, initial_coverage: dict[str, Any], final_coverage: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate the difference in coverage between initial and final states.

        Args:
            initial_coverage: Initial coverage measurements
            final_coverage: Final coverage measurements

        Returns:
            Coverage delta information with error handling
        """
        try:
            initial_line = initial_coverage.get("overall_line_coverage", 0.0)
            final_line = final_coverage.get("overall_line_coverage", 0.0)
            line_delta = final_line - initial_line

            initial_branch = initial_coverage.get("overall_branch_coverage", 0.0)
            final_branch = final_coverage.get("overall_branch_coverage", 0.0)
            branch_delta = final_branch - initial_branch

            initial_lines = initial_coverage.get("total_lines", 0)
            final_lines = final_coverage.get("total_lines", 0)
            lines_delta = final_lines - initial_lines

            return {
                "line_coverage_delta": line_delta,
                "branch_coverage_delta": branch_delta,
                "total_lines_delta": lines_delta,
                "initial_line_coverage": initial_line,
                "final_line_coverage": final_line,
                "initial_branch_coverage": initial_branch,
                "final_branch_coverage": final_branch,
                "improvement_percentage": (line_delta * 100) if line_delta > 0 else 0.0,
            }

        except Exception as e:
            logger.warning("Failed to calculate coverage delta: %s", e)
            return {
                "line_coverage_delta": 0.0,
                "branch_coverage_delta": 0.0,
                "total_lines_delta": 0,
                "improvement_percentage": 0.0,
                "error": str(e),
            }
