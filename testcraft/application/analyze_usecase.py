"""
Analyze Use Case - Analysis of what would be generated and why.

This module implements the analyze use case for determining what files need
test generation and the reasons for generation, providing insights before
actually generating tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..adapters.parsing.test_mapper import TestMapper
from ..domain.models import AnalysisReport
from ..ports.coverage_port import CoveragePort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class AnalyzeUseCaseError(Exception):
    """Exception for Analyze Use Case specific errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class AnalyzeUseCase:
    """
    Use case for analyzing what would be generated and why.

    Provides analysis of files that need test processing without actually
    generating tests, allowing users to understand what work would be done
    and make informed decisions.
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
        Initialize the Analyze Use Case with required ports.

        Args:
            coverage_port: Port for coverage measurement
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
            "coverage_threshold": 0.8,  # Coverage threshold for processing decisions
            **(config or {}),
        }

        # Per-run caches for AST-based test detection
        self._current_project_path: Path | None = None
        self._cached_test_files: list[str] | None = None
        self._test_mapper = TestMapper()

    async def analyze_generation_needs(
        self,
        project_path: str | Path,
        target_files: list[str | Path] | None = None,
        **kwargs: Any,
    ) -> AnalysisReport:
        """
        Analyze what files would need test generation and why.

        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to analyze
            **kwargs: Additional analysis parameters

        Returns:
            AnalysisReport containing files to process, reasons, and test presence info
        """
        project_path = Path(project_path)

        with self._telemetry.create_span(
            "analyze_generation_needs",
            kind=SpanKind.INTERNAL,
            attributes={
                "project_path": str(project_path),
                "target_files_count": len(target_files) if target_files else 0,
                "config": self._config,
            },
        ) as span:
            try:
                logger.info("Starting analysis for project: %s", project_path)

                # Cache project path and discovered test files for AST-based detection
                self._current_project_path = project_path
                try:
                    self._cached_test_files = self._file_discovery.discover_test_files(
                        project_path
                    )
                except Exception:
                    self._cached_test_files = []

                # Step 1: Sync state and discover files
                discovery_result = await self._sync_state_and_discover_files(
                    project_path, target_files
                )
                span.set_attribute("files_discovered", len(discovery_result["files"]))

                # Step 2: Measure initial coverage
                initial_coverage = await self._measure_initial_coverage(
                    discovery_result["files"]
                )
                span.set_attribute(
                    "initial_coverage", initial_coverage.get("overall_line_coverage", 0)
                )

                # Step 3: Decide which files to process
                files_to_process = await self._decide_files_to_process(
                    discovery_result["files"], initial_coverage
                )
                span.set_attribute("files_to_process", len(files_to_process))

                # Step 4: Build reasons and test presence information
                reasons = await self._build_processing_reasons(
                    files_to_process, initial_coverage
                )
                test_presence = await self._build_test_presence_info(files_to_process)

                span.set_attribute("reasons_built", len(reasons))
                span.set_attribute("test_presence_analyzed", len(test_presence))

                # Step 5: Create and return AnalysisReport
                report = AnalysisReport(
                    files_to_process=[str(f) for f in files_to_process],
                    reasons=reasons,
                    existing_test_presence=test_presence,
                )

                logger.info(
                    "Analysis completed. Files needing processing: %d, Files with existing tests: %d",
                    len(files_to_process),
                    sum(1 for has_tests in test_presence.values() if has_tests),
                )

                return report

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Analysis failed: %s", e)
                raise AnalyzeUseCaseError(f"Analysis failed: {e}", cause=e) from e

    async def _build_processing_reasons(
        self, files_to_process: list[Path], coverage_data: dict[str, Any]
    ) -> dict[str, str]:
        """
        Build reasons dictionary explaining why each file needs processing.

        Args:
            files_to_process: Files that need processing
            coverage_data: Coverage information

        Returns:
            Dictionary mapping file paths to reasons
        """
        with self._telemetry.create_child_span("build_processing_reasons") as span:
            reasons = {}

            try:
                for file_path in files_to_process:
                    reason = await self._get_processing_reason(file_path, coverage_data)
                    reasons[str(file_path)] = reason

                span.set_attribute("reasons_generated", len(reasons))
                return reasons

            except Exception as e:
                logger.exception("Failed to build processing reasons: %s", e)
                raise AnalyzeUseCaseError(
                    f"Reason building failed: {e}", cause=e
                ) from e

    async def _build_test_presence_info(
        self, files_to_process: list[Path]
    ) -> dict[str, bool]:
        """
        Build test presence information for each file.

        Args:
            files_to_process: Files to check for existing tests

        Returns:
            Dictionary mapping file paths to test presence boolean
        """
        with self._telemetry.create_child_span("build_test_presence_info") as span:
            test_presence = {}

            try:
                for file_path in files_to_process:
                    has_tests = self._has_existing_tests(file_path)
                    test_presence[str(file_path)] = has_tests

                span.set_attribute("test_presence_analyzed", len(test_presence))
                return test_presence

            except Exception as e:
                logger.exception("Failed to build test presence info: %s", e)
                raise AnalyzeUseCaseError(
                    f"Test presence analysis failed: {e}", cause=e
                ) from e

    async def _sync_state_and_discover_files(
        self, project_path: Path, target_files: list[str | Path] | None = None
    ) -> dict[str, Any]:
        """
        Synchronize state and discover source files to process.

        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to target

        Returns:
            Dictionary with discovered files and metadata
        """
        with self._telemetry.create_child_span("sync_state_and_discover") as span:
            try:
                # Load current state
                current_state = self._state.get_all_state("generation")
                span.set_attribute("previous_state_keys", len(current_state))

                # Discover source files using FileDiscoveryService
                if target_files:
                    # Use provided target files, filtered for validity
                    file_paths = [str(f) for f in target_files]
                    files = [
                        Path(f)
                        for f in self._file_discovery.filter_existing_files(file_paths)
                    ]
                    span.set_attribute("discovery_method", "target_files")
                else:
                    # Discover source files using file discovery service
                    discovered_files = self._file_discovery.discover_source_files(
                        project_path, include_test_files=False
                    )
                    files = [Path(f) for f in discovered_files]
                    span.set_attribute("discovery_method", "pattern_discovery")

                span.set_attribute("files_found", len(files))

                return {
                    "files": files,
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
                raise AnalyzeUseCaseError(f"File discovery failed: {e}", cause=e) from e

    async def _measure_initial_coverage(
        self, source_files: list[Path]
    ) -> dict[str, Any]:
        """
        Measure initial code coverage before processing.

        Args:
            source_files: List of source files to measure coverage for

        Returns:
            Coverage measurement results
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

    async def _decide_files_to_process(
        self, discovered_files: list[Path], coverage_data: dict[str, Any]
    ) -> list[Path]:
        """
        Decide which files need processing based on coverage and other criteria.

        Args:
            discovered_files: All discovered source files
            coverage_data: Current coverage information

        Returns:
            List of files that should be processed
        """
        with self._telemetry.create_child_span("decide_files_to_process") as span:
            files_to_process = []

            try:
                for file_path in discovered_files:
                    # Check if file needs processing
                    needs_processing = await self._file_needs_processing(
                        file_path, coverage_data
                    )

                    if needs_processing:
                        files_to_process.append(file_path)

                span.set_attribute("files_selected", len(files_to_process))
                return files_to_process

            except Exception as e:
                logger.exception("Failed to decide files to process: %s", e)
                raise AnalyzeUseCaseError(
                    f"File processing decision failed: {e}", cause=e
                ) from e

    async def _file_needs_processing(
        self, file_path: Path, coverage_data: dict[str, Any]
    ) -> bool:
        """
        Determine if a specific file needs processing.

        Args:
            file_path: Path to the file to check
            coverage_data: Coverage information

        Returns:
            True if the file needs processing
        """
        try:
            # Check if file has existing tests
            has_existing_tests = self._has_existing_tests(file_path)

            # Check current coverage for this file
            current_coverage = self._get_file_coverage(file_path, coverage_data)

            # File needs processing if:
            # 1. No existing tests, OR
            # 2. Coverage below threshold
            return (
                not has_existing_tests
                or current_coverage < self._config["coverage_threshold"]
            )

        except Exception as e:
            logger.warning(
                "Failed to check if file %s needs processing: %s", file_path, e
            )
            # When in doubt, process the file
            return True

    def _has_existing_tests(self, file_path: Path) -> bool:
        """Determine if the source file already has tests using AST-based mapping.

        Falls back to filename pattern checks on failure.
        """
        try:
            # Ensure we have candidate test files discovered
            if (
                self._cached_test_files is None
                and self._current_project_path is not None
            ):
                # Cache not initialized, discover test files once
                test_files = self._file_discovery.discover_test_files(
                    self._current_project_path,
                    quiet=True,  # Reduce log noise during analysis
                )
                self._cached_test_files = test_files
            else:
                # Use cached result (which may be an empty list if no tests exist)
                test_files = self._cached_test_files or []

            if not test_files:
                return False

            # Parse source elements
            parse_result = self._coverage  # placeholder to keep linter context
            del parse_result
            from ..adapters.parsing.codebase_parser import CodebaseParser

            parser = CodebaseParser()
            parse_result = parser.parse_file(file_path)
            elements = parse_result.get("elements", [])
            if not elements:
                return False

            # Map tests to source elements
            mapping = self._test_mapper.map_tests(elements, existing_tests=test_files)
            coverage_pct = float(mapping.get("coverage_percentage", 0.0))
            return coverage_pct > 0.0

        except Exception:
            # Fallback: simple filename-based heuristics
            potential_test_files = [
                file_path.parent / f"test_{file_path.name}",
                file_path.parent / f"{file_path.stem}_test.py",
                file_path.parent.parent / "tests" / f"test_{file_path.name}",
            ]
            return any(test_file.exists() for test_file in potential_test_files)

    def _get_file_coverage(
        self, file_path: Path, coverage_data: dict[str, Any]
    ) -> float:
        """Get coverage percentage for a specific file."""
        # This is a simplified check - in reality we'd look up specific file coverage
        if coverage_data.get("files_covered", 0) > 0:
            return coverage_data.get("overall_line_coverage", 0)
        return 0.0

    async def _get_processing_reason(
        self, file_path: Path, coverage_data: dict[str, Any]
    ) -> str:
        """
        Get the reason why a file needs processing.

        Args:
            file_path: Path to the file
            coverage_data: Coverage information

        Returns:
            Human-readable reason for processing
        """
        has_tests = self._has_existing_tests(file_path)
        coverage = self._get_file_coverage(file_path, coverage_data)
        threshold = self._config["coverage_threshold"]

        if not has_tests:
            return "No existing tests found"
        elif coverage < threshold:
            return f"Coverage {coverage:.1%} below threshold {threshold:.1%}"
        else:
            return "File meets criteria for processing"
