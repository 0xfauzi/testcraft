"""
Report Port interface definition.

This module defines the interface for report generation operations,
including analysis reports and summary generation.
"""

from pathlib import Path
from typing import Any

from typing_extensions import Protocol

from ..domain.models import AnalysisReport


class ReportPort(Protocol):
    """
    Interface for report generation operations.

    This protocol defines the contract for generating various types
    of reports, including analysis reports and summaries.
    """

    def generate_report(
        self,
        report_type: str,
        data: dict[str, Any],
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a report of the specified type.

        Args:
            report_type: Type of report to generate (coverage, analysis, summary, etc.)
            data: Data to include in the report
            output_format: Format of the report (markdown, html, json, etc.)
            **kwargs: Additional report generation parameters

        Returns:
            Dictionary containing:
                - 'report_content': Generated report content
                - 'report_format': Format of the generated report
                - 'report_metadata': Additional metadata about the report
                - 'generation_stats': Statistics about report generation

        Raises:
            ReportError: If report generation fails
        """
        ...

    def generate_analysis_report(
        self,
        analysis_data: AnalysisReport,
        include_recommendations: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate an analysis report from AnalysisReport data.

        Args:
            analysis_data: AnalysisReport object containing analysis results
            include_recommendations: Whether to include recommendations
            **kwargs: Additional report parameters

        Returns:
            Dictionary containing:
                - 'report_content': Generated analysis report
                - 'summary': Executive summary of the analysis
                - 'recommendations': List of recommendations (if requested)
                - 'analysis_metadata': Additional analysis metadata

        Raises:
            ReportError: If analysis report generation fails
        """
        ...

    def generate_coverage_report(
        self,
        coverage_data: dict[str, Any],
        report_style: str = "detailed",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a coverage report from coverage data.

        Args:
            coverage_data: Coverage data to include in the report
            report_style: Style of the coverage report (detailed, summary, etc.)
            **kwargs: Additional report parameters

        Returns:
            Dictionary containing:
                - 'report_content': Generated coverage report
                - 'coverage_summary': Summary of coverage metrics
                - 'coverage_trends': Coverage trends and patterns
                - 'coverage_metadata': Additional coverage metadata

        Raises:
            ReportError: If coverage report generation fails
        """
        ...

    def generate_summary_report(
        self,
        project_data: dict[str, Any],
        summary_level: str = "comprehensive",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a summary report for the entire project.

        Args:
            project_data: Project data to summarize
            summary_level: Level of detail in the summary
            **kwargs: Additional summary parameters

        Returns:
            Dictionary containing:
                - 'summary_content': Generated summary report
                - 'key_metrics': Key project metrics
                - 'trends': Project trends and patterns
                - 'summary_metadata': Additional summary metadata

        Raises:
            ReportError: If summary report generation fails
        """
        ...

    def export_report(
        self,
        report_content: str,
        output_path: str | Path,
        export_format: str = "markdown",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Export a report to a file.

        Args:
            report_content: Content of the report to export
            output_path: Path where the report should be saved
            export_format: Format to export the report in
            **kwargs: Additional export parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the export succeeded
                - 'export_path': Path where the report was exported
                - 'file_size': Size of the exported file
                - 'export_metadata': Additional export metadata

        Raises:
            ReportError: If report export fails
        """
        ...
