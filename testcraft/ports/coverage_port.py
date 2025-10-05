"""
Coverage Port interface definition.

This module defines the interface for code coverage measurement and reporting,
including coverage analysis and metric collection.
"""

from typing import Any

from typing_extensions import Protocol

from ..domain.models import CoverageResult


class CoveragePort(Protocol):
    """
    Interface for code coverage measurement and reporting.

    This protocol defines the contract for coverage analysis operations,
    including measurement, reporting, and coverage data management.
    """

    def measure_coverage(
        self,
        source_files: list[str],
        test_files: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, CoverageResult]:
        """
        Measure code coverage for the specified source files.

        Args:
            source_files: List of source file paths to measure coverage for
            test_files: Optional list of test file paths to include
            **kwargs: Additional measurement parameters

        Returns:
            Dictionary mapping file paths to CoverageResult objects

        Raises:
            CoverageError: If coverage measurement fails
        """
        ...

    def report_coverage(
        self,
        coverage_data: dict[str, CoverageResult],
        output_format: str = "detailed",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate coverage reports in various formats.

        Args:
            coverage_data: Coverage data to report on
            output_format: Format of the report (detailed, summary, html, etc.)
            **kwargs: Additional reporting parameters

        Returns:
            Dictionary containing:
                - 'report_content': The generated report content
                - 'summary_stats': Overall coverage statistics
                - 'format': The format used for the report

        Raises:
            CoverageError: If report generation fails
        """
        ...

    def get_coverage_summary(
        self, coverage_data: dict[str, CoverageResult]
    ) -> dict[str, Any]:
        """
        Get a summary of coverage metrics across all files.

        Args:
            coverage_data: Coverage data to summarize

        Returns:
            Dictionary containing:
                - 'overall_line_coverage': Overall line coverage percentage
                - 'overall_branch_coverage': Overall branch coverage percentage
                - 'files_covered': Number of files with coverage data
                - 'total_lines': Total number of lines analyzed
                - 'missing_coverage': Summary of missing coverage areas
        """
        ...

    def identify_gaps(
        self, coverage_data: dict[str, CoverageResult], threshold: float = 0.8
    ) -> dict[str, list[int]]:
        """
        Identify coverage gaps that fall below the specified threshold.

        Args:
            coverage_data: Coverage data to analyze
            threshold: Minimum coverage threshold (0.0 to 1.0)

        Returns:
            Dictionary mapping file paths to lists of line numbers with poor coverage
        """
        ...

    def get_measurement_method(self) -> str:
        """
        Get coverage measurement method identifier.

        Returns:
            String identifier for the measurement method used
            (e.g., "coverage.py", "pytest-coverage", "ast-fallback", "noop")

        Note:
            This method was added to eliminate leaky abstraction where
            use cases were calling getattr() on adapters directly.
        """
        ...
