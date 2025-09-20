"""
Coverage Use Case - Core coverage measurement and reporting orchestration.

This module implements the primary use case for measuring code coverage and generating
reports, orchestrating the coverage port to provide end-to-end coverage functionality
with proper error handling and telemetry.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..domain.models import CoverageResult
from ..ports.coverage_port import CoveragePort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import MetricValue, SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class CoverageUseCaseError(Exception):
    """Base exception for Coverage Use Case errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class CoverageUseCase:
    """
    Core use case for coverage measurement and reporting.

    Orchestrates the coverage measurement workflow including:
    - State synchronization and file discovery
    - Coverage measurement with fallback strategies
    - Report generation in multiple formats
    - Coverage gap identification and analysis
    - Telemetry and metrics recording
    """

    def __init__(
        self,
        coverage_port: CoveragePort,
        state_port: StatePort,
        telemetry_port: TelemetryPort,
        file_discovery_service: FileDiscoveryService | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Coverage Use Case with required ports.

        Args:
            coverage_port: Port for coverage operations
            state_port: Port for state management
            telemetry_port: Port for telemetry and metrics
            file_discovery_service: Service for file discovery (creates default if None)
            config: Optional configuration overrides
        """
        self._coverage = coverage_port
        self._state = state_port
        self._telemetry = telemetry_port

        # Initialize file discovery service
        self._file_discovery = file_discovery_service or FileDiscoveryService()

        # Configuration with sensible defaults
        self._config = {
            "coverage_threshold": 0.8,  # Coverage threshold for reporting
            "include_test_files": False,  # Whether to include test files in coverage
            "output_formats": ["detailed", "summary"],  # Default report formats
            "enable_gap_analysis": True,  # Whether to identify coverage gaps
            "max_files_per_batch": 50,  # Maximum files to process at once
            **(config or {}),
        }

    async def measure_and_report(
        self,
        project_path: str | Path,
        source_files: list[str | Path] | None = None,
        test_files: list[str | Path] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Main entry point for coverage measurement and reporting.

        Args:
            project_path: Root path of the project
            source_files: Optional list of specific source files to measure
            test_files: Optional list of test files to include in measurement
            **kwargs: Additional measurement parameters

        Returns:
            Dictionary containing coverage results, reports, and metadata
        """
        project_path = Path(project_path)

        with self._telemetry.create_span(
            "coverage_measure_and_report",
            kind=SpanKind.INTERNAL,
            attributes={
                "project_path": str(project_path),
                "source_files_count": len(source_files) if source_files else 0,
                "test_files_count": len(test_files) if test_files else 0,
                "config": self._config,
            },
        ) as span:
            try:
                logger.info(
                    "Starting coverage measurement and reporting for project: %s",
                    project_path,
                )

                # Step 1: Sync state and discover files
                discovery_result = await self._sync_state_and_discover_files(
                    project_path, source_files, test_files
                )
                span.set_attribute(
                    "files_discovered", len(discovery_result["source_files"])
                )

                # Step 2: Measure coverage
                coverage_data = await self._measure_coverage(
                    discovery_result["source_files"], discovery_result["test_files"]
                )
                span.set_attribute("files_measured", len(coverage_data))

                # Step 3: Generate coverage summary
                coverage_summary = self._coverage.get_coverage_summary(coverage_data)
                span.set_attribute(
                    "overall_coverage", coverage_summary.get("overall_line_coverage", 0)
                )

                # Step 4: Generate reports in requested formats
                reports = await self._generate_reports(coverage_data)
                span.set_attribute("reports_generated", len(reports))

                # Step 5: Identify coverage gaps if enabled
                coverage_gaps = {}
                if self._config["enable_gap_analysis"]:
                    coverage_gaps = self._coverage.identify_gaps(
                        coverage_data, threshold=self._config["coverage_threshold"]
                    )
                    span.set_attribute("gaps_identified", len(coverage_gaps))

                # Step 6: Record state and telemetry
                await self._record_coverage_state(coverage_data, coverage_summary)
                await self._record_telemetry_and_metrics(
                    span, coverage_data, coverage_summary
                )

                # Compile results
                results = {
                    "success": True,
                    "coverage_data": coverage_data,
                    "coverage_summary": coverage_summary,
                    "reports": reports,
                    "coverage_gaps": coverage_gaps,
                    "files_measured": len(coverage_data),
                    "overall_line_coverage": coverage_summary.get(
                        "overall_line_coverage", 0
                    ),
                    "overall_branch_coverage": coverage_summary.get(
                        "overall_branch_coverage", 0
                    ),
                    "metadata": {
                        "project_path": str(project_path),
                        "config_used": self._config,
                        "timestamp": discovery_result.get("timestamp"),
                        "coverage_method": getattr(
                            self._coverage, "get_last_method_used", lambda: "unknown"
                        )(),
                    },
                }

                logger.info(
                    "Coverage measurement completed successfully. Files: %d, Overall coverage: %.2f%%",
                    len(coverage_data),
                    coverage_summary.get("overall_line_coverage", 0) * 100,
                )

                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Coverage measurement failed: %s", e)
                raise CoverageUseCaseError(
                    f"Coverage measurement failed: {e}", cause=e
                ) from e

    async def _sync_state_and_discover_files(
        self,
        project_path: Path,
        source_files: list[str | Path] | None = None,
        test_files: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronize state and discover source and test files.

        Args:
            project_path: Root path of the project
            source_files: Optional list of specific source files
            test_files: Optional list of specific test files

        Returns:
            Dictionary with discovered files and metadata
        """
        with self._telemetry.create_child_span("sync_state_and_discover") as span:
            try:
                # Load current state
                current_state = self._state.get_all_state("coverage")
                span.set_attribute("previous_state_keys", len(current_state))

                # Discover source files
                discovered_source_files = []
                if source_files:
                    # Use provided source files, filtered for validity
                    discovered_source_files = (
                        self._file_discovery.filter_existing_files(source_files)
                    )
                    span.set_attribute("discovery_method", "provided_source_files")
                else:
                    # Discover source files using file discovery service
                    discovered_source_files = (
                        self._file_discovery.discover_source_files(
                            project_path,
                            include_test_files=self._config.get(
                                "include_test_files", False
                            ),
                        )
                    )
                    span.set_attribute("discovery_method", "pattern_discovery")

                # Discover test files if needed
                discovered_test_files = []
                if self._config.get("include_test_files", False):
                    if test_files:
                        discovered_test_files = (
                            self._file_discovery.filter_existing_files(test_files)
                        )
                    else:
                        discovered_test_files = (
                            self._file_discovery.discover_test_files(project_path)
                        )

                span.set_attribute("source_files_found", len(discovered_source_files))
                span.set_attribute("test_files_found", len(discovered_test_files))

                return {
                    "source_files": discovered_source_files,
                    "test_files": discovered_test_files,
                    "previous_state": current_state,
                    "timestamp": (
                        span.get_trace_context().trace_id
                        if span.get_trace_context()
                        else None
                    ),
                    "project_path": project_path,
                }

            except Exception as e:
                logger.exception("Failed to sync state and discover files: %s", e)
                raise CoverageUseCaseError(
                    f"File discovery failed: {e}", cause=e
                ) from e

    async def _measure_coverage(
        self, source_files: list[str], test_files: list[str] | None = None
    ) -> dict[str, CoverageResult]:
        """
        Measure coverage for the specified files.

        Args:
            source_files: List of source files to measure
            test_files: Optional list of test files

        Returns:
            Dictionary mapping file paths to coverage results
        """
        with self._telemetry.create_child_span("measure_coverage") as span:
            try:
                # Batch files if needed to avoid overwhelming the coverage tool
                max_files = self._config.get("max_files_per_batch", 50)

                if len(source_files) <= max_files:
                    # Process all files at once
                    coverage_data = self._coverage.measure_coverage(
                        source_files, test_files
                    )
                else:
                    # Process in batches
                    coverage_data = {}
                    for i in range(0, len(source_files), max_files):
                        batch = source_files[i : i + max_files]
                        batch_data = self._coverage.measure_coverage(batch, test_files)
                        coverage_data.update(batch_data)

                span.set_attribute("files_measured", len(coverage_data))
                span.set_attribute(
                    "measurement_method",
                    getattr(
                        self._coverage, "get_last_method_used", lambda: "unknown"
                    )(),
                )

                return coverage_data

            except Exception as e:
                logger.exception("Failed to measure coverage: %s", e)
                raise CoverageUseCaseError(
                    f"Coverage measurement failed: {e}", cause=e
                ) from e

    async def _generate_reports(
        self, coverage_data: dict[str, CoverageResult]
    ) -> dict[str, dict[str, Any]]:
        """
        Generate coverage reports in requested formats.

        Args:
            coverage_data: Coverage data to generate reports for

        Returns:
            Dictionary mapping format names to report data
        """
        with self._telemetry.create_child_span("generate_reports") as span:
            reports = {}

            try:
                output_formats = self._config.get("output_formats", ["detailed"])

                for format_name in output_formats:
                    try:
                        report_data = self._coverage.report_coverage(
                            coverage_data, output_format=format_name
                        )
                        reports[format_name] = report_data

                    except Exception as e:
                        logger.warning(f"Failed to generate {format_name} report: {e}")
                        reports[format_name] = {
                            "success": False,
                            "error": str(e),
                            "format": format_name,
                        }

                span.set_attribute("reports_generated", len(reports))
                return reports

            except Exception as e:
                logger.exception("Failed to generate reports: %s", e)
                raise CoverageUseCaseError(
                    f"Report generation failed: {e}", cause=e
                ) from e

    async def _record_coverage_state(
        self, coverage_data: dict[str, CoverageResult], coverage_summary: dict[str, Any]
    ) -> None:
        """
        Record coverage state for future analysis and comparison.

        Args:
            coverage_data: Coverage measurement data
            coverage_summary: Coverage summary statistics
        """
        try:
            # Compile state data
            state_data = {
                "last_coverage_run_timestamp": asyncio.get_event_loop().time(),
                "coverage_summary": coverage_summary,
                "files_measured": len(coverage_data),
                "coverage_method": getattr(
                    self._coverage, "get_last_method_used", lambda: "unknown"
                )(),
                "config_used": self._config.copy(),
                "file_coverage_details": {
                    file_path: {
                        "line_coverage": result.line_coverage,
                        "branch_coverage": result.branch_coverage,
                        "missing_lines_count": len(result.missing_lines),
                    }
                    for file_path, result in coverage_data.items()
                },
            }

            # Record state
            self._state.update_state("last_coverage_run", state_data)

        except Exception as e:
            logger.warning("Failed to record coverage state: %s", e)
            # Don't fail the entire operation for state recording issues

    async def _record_telemetry_and_metrics(
        self,
        span,
        coverage_data: dict[str, CoverageResult],
        coverage_summary: dict[str, Any],
    ) -> None:
        """
        Record telemetry metrics and coverage information.

        Args:
            span: Current telemetry span
            coverage_data: Coverage measurement data
            coverage_summary: Coverage summary statistics
        """
        try:
            from datetime import datetime

            # Record coverage metrics
            metrics = [
                MetricValue(
                    name="coverage_files_measured",
                    value=len(coverage_data),
                    unit="count",
                    labels={
                        "method": getattr(
                            self._coverage, "get_last_method_used", lambda: "unknown"
                        )()
                    },
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="coverage_overall_line_percentage",
                    value=coverage_summary.get("overall_line_coverage", 0) * 100,
                    unit="percent",
                    labels={"type": "line_coverage"},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="coverage_overall_branch_percentage",
                    value=coverage_summary.get("overall_branch_coverage", 0) * 100,
                    unit="percent",
                    labels={"type": "branch_coverage"},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="coverage_files_above_threshold",
                    value=sum(
                        1
                        for result in coverage_data.values()
                        if result.line_coverage >= self._config["coverage_threshold"]
                    ),
                    unit="count",
                    labels={"threshold": str(self._config["coverage_threshold"])},
                    timestamp=datetime.now(),
                ),
            ]

            self._telemetry.record_metrics(metrics)

            # Record span attributes
            span.set_attribute("total_files_measured", len(coverage_data))
            span.set_attribute(
                "overall_line_coverage",
                coverage_summary.get("overall_line_coverage", 0),
            )
            span.set_attribute(
                "overall_branch_coverage",
                coverage_summary.get("overall_branch_coverage", 0),
            )
            span.set_attribute(
                "coverage_method",
                getattr(self._coverage, "get_last_method_used", lambda: "unknown")(),
            )

            # Flush telemetry data
            self._telemetry.flush(timeout_seconds=5.0)

        except Exception as e:
            logger.warning("Failed to record telemetry and metrics: %s", e)
            # Don't fail the entire operation for telemetry issues

    async def get_coverage_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get historical coverage data from state.

        Args:
            limit: Maximum number of historical records to return

        Returns:
            List of historical coverage records
        """
        try:
            # This is a simplified implementation - in a real system you might
            # want to store more detailed historical data
            current_state = self._state.get_all_state("coverage")

            # Extract historical records if available
            history = []
            if "last_coverage_run" in current_state:
                history.append(current_state["last_coverage_run"])

            return history[:limit]

        except Exception as e:
            logger.warning("Failed to get coverage history: %s", e)
            return []

    def calculate_coverage_delta(
        self,
        current_coverage: dict[str, Any],
        previous_coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate coverage delta between two coverage runs.

        Args:
            current_coverage: Current coverage summary
            previous_coverage: Previous coverage summary (None to fetch from state)

        Returns:
            Dictionary containing coverage delta information
        """
        try:
            if previous_coverage is None:
                # Try to get previous coverage from state
                state_data = self._state.get_all_state("coverage")
                previous_run = state_data.get("last_coverage_run", {})
                previous_coverage = previous_run.get("coverage_summary", {})

            if not previous_coverage:
                return {
                    "line_coverage_delta": 0.0,
                    "branch_coverage_delta": 0.0,
                    "has_previous_data": False,
                    "message": "No previous coverage data available",
                }

            current_line = current_coverage.get("overall_line_coverage", 0.0)
            previous_line = previous_coverage.get("overall_line_coverage", 0.0)
            line_delta = current_line - previous_line

            current_branch = current_coverage.get("overall_branch_coverage", 0.0)
            previous_branch = previous_coverage.get("overall_branch_coverage", 0.0)
            branch_delta = current_branch - previous_branch

            return {
                "line_coverage_delta": line_delta,
                "branch_coverage_delta": branch_delta,
                "current_line_coverage": current_line,
                "previous_line_coverage": previous_line,
                "current_branch_coverage": current_branch,
                "previous_branch_coverage": previous_branch,
                "improvement_percentage": (line_delta * 100) if line_delta > 0 else 0.0,
                "has_previous_data": True,
                "trend": (
                    "improving"
                    if line_delta > 0
                    else "declining"
                    if line_delta < 0
                    else "stable"
                ),
            }

        except Exception as e:
            logger.warning("Failed to calculate coverage delta: %s", e)
            return {
                "line_coverage_delta": 0.0,
                "branch_coverage_delta": 0.0,
                "has_previous_data": False,
                "error": str(e),
            }
