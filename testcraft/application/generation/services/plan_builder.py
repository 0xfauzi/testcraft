"""
Test generation plan builder service.

Handles file selection decisions and TestGenerationPlan creation,
including logic for detecting existing tests and determining processing needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....adapters.io.file_discovery import FileDiscoveryService
from ....adapters.parsing.test_mapper import TestMapper
from ....domain.models import TestGenerationPlan
from ....ports.parser_port import ParserPort
from ....ports.telemetry_port import TelemetryPort
from .state_discovery import GenerateUseCaseError

logger = logging.getLogger(__name__)


class PlanBuilder:
    """
    Service for building test generation plans.

    Handles file selection decisions and TestGenerationPlan creation,
    including AST-based test detection and coverage-aware processing decisions.
    """

    def __init__(
        self,
        parser_port: ParserPort,
        file_discovery_service: FileDiscoveryService,
        telemetry_port: TelemetryPort,
        coverage_threshold: float = 0.8,
    ):
        """
        Initialize the plan builder service.

        Args:
            parser_port: Port for code parsing operations
            file_discovery_service: Service for file discovery
            telemetry_port: Port for telemetry operations
            coverage_threshold: Coverage threshold for processing decisions
        """
        self._parser = parser_port
        self._file_discovery = file_discovery_service
        self._telemetry = telemetry_port
        self._coverage_threshold = coverage_threshold
        self._test_mapper = TestMapper()

        # Per-run caches for AST-based test detection
        self._current_project_path: Path | None = None
        self._cached_test_files: list[str] | None = None
        # Map generation plans to their source file paths
        self._plan_source_map: dict[int, Path] = {}

    def set_project_context(
        self, project_path: Path, test_files: list[str] | None = None
    ):
        """
        Set the current project context for plan building.

        Args:
            project_path: Root path of the current project
            test_files: Optional pre-discovered test files
        """
        self._current_project_path = project_path
        self._cached_test_files = test_files

    def decide_files_to_process(
        self, discovered_files: list[Path], coverage_data: dict[str, Any]
    ) -> list[Path]:
        """
        Decide which files need test processing based on coverage and other criteria.

        Args:
            discovered_files: All discovered source files
            coverage_data: Current coverage information

        Returns:
            List of files that should be processed

        Raises:
            GenerateUseCaseError: If file processing decision fails
        """
        with self._telemetry.create_child_span("decide_files_to_process") as span:
            files_to_process = []

            try:
                for file_path in discovered_files:
                    # Check if file needs test generation
                    needs_processing = self._file_needs_processing(
                        file_path, coverage_data
                    )

                    if needs_processing:
                        files_to_process.append(file_path)

                span.set_attribute("files_selected", len(files_to_process))
                return files_to_process

            except Exception as e:
                logger.exception("Failed to decide files to process: %s", e)
                raise GenerateUseCaseError(
                    f"File processing decision failed: {e}", cause=e
                )

    def build_plans(self, files_to_process: list[Path]) -> list[TestGenerationPlan]:
        """
        Build TestGenerationPlan objects for each file to process.

        Args:
            files_to_process: Files that need test generation

        Returns:
            List of test generation plans

        Raises:
            GenerateUseCaseError: If generation plan creation fails
        """
        with self._telemetry.create_child_span("build_generation_plans") as span:
            plans = []

            try:
                for file_path in files_to_process:
                    plan = self._create_generation_plan_for_file(file_path)
                    if plan is not None:  # Only add valid plans
                        plans.append(plan)

                span.set_attribute("plans_created", len(plans))
                return plans

            except Exception as e:
                logger.exception("Failed to build generation plans: %s", e)
                raise GenerateUseCaseError(
                    f"Generation plan creation failed: {e}", cause=e
                )

    def get_source_path_for_plan(self, plan: TestGenerationPlan) -> Path | None:
        """
        Get the source file path for a test generation plan.

        Args:
            plan: The test generation plan

        Returns:
            Source file path if available, None otherwise
        """
        return self._plan_source_map.get(id(plan))

    def _file_needs_processing(
        self, file_path: Path, coverage_data: dict[str, Any]
    ) -> bool:
        """
        Determine if a specific file needs test processing.

        Args:
            file_path: Path to the file to check
            coverage_data: Coverage information

        Returns:
            True if the file needs processing
        """
        try:
            # Check if file has existing tests using AST-based detection (with fallback)
            has_existing_tests = self._has_existing_tests(file_path)

            # Check current coverage for this file
            current_coverage = 0.0

            # This is a simplified check - in reality we'd look up specific file coverage
            if coverage_data.get("files_covered", 0) > 0:
                current_coverage = coverage_data.get("overall_line_coverage", 0)

            # File needs processing if:
            # 1. No existing tests, OR
            # 2. Coverage below threshold
            return not has_existing_tests or current_coverage < self._coverage_threshold

        except Exception as e:
            logger.warning(
                "Failed to check if file %s needs processing: %s", file_path, e
            )
            # When in doubt, process the file
            return True

    def _has_existing_tests(self, file_path: Path) -> bool:
        """
        Determine if the source file already has tests using AST-based mapping.

        Falls back to filename pattern checks on failure.

        Args:
            file_path: Path to the source file

        Returns:
            True if existing tests are found
        """
        try:
            # Ensure we have candidate test files discovered
            test_files = self._cached_test_files or []
            if not test_files and self._current_project_path is not None:
                test_files = self._file_discovery.discover_test_files(
                    self._current_project_path
                )
                self._cached_test_files = test_files

            if not test_files:
                return False

            # Parse source elements
            parse_result = self._parser.parse_file(file_path)
            elements = parse_result.get("elements", [])
            if not elements:
                return False

            # Map tests to source elements using AST-aware TestMapper
            mapping = self._test_mapper.map_tests(elements, existing_tests=test_files)
            coverage_pct = float(mapping.get("coverage_percentage", 0.0))
            return coverage_pct > 0.0

        except Exception as e:
            logger.debug(
                "AST-based test detection failed for %s: %s; falling back to pattern checks",
                file_path,
                e,
            )
            # Fallback: simple filename-based heuristics
            potential_test_files = [
                file_path.parent / f"test_{file_path.name}",
                file_path.parent / f"{file_path.stem}_test.py",
                file_path.parent.parent / "tests" / f"test_{file_path.name}",
            ]
            return any(test_file.exists() for test_file in potential_test_files)

    def _create_generation_plan_for_file(
        self, file_path: Path
    ) -> TestGenerationPlan | None:
        """
        Create a TestGenerationPlan for a specific file.

        Args:
            file_path: Path to the file to create a plan for

        Returns:
            TestGenerationPlan for the file, or None if no testable elements found
        """
        try:
            # Parse the file to extract testable elements
            parse_result = self._parser.parse_file(file_path)
            elements = parse_result.get("elements", [])

            # Skip files with no testable elements
            if not elements:
                logger.info("Skipping %s: no testable elements found", file_path)
                return None

            # Find existing test files
            existing_tests = self._find_existing_test_files(file_path)

            # Get current coverage for this file (simplified)
            coverage_before = None  # Would implement file-specific coverage lookup

            plan = TestGenerationPlan(
                elements_to_test=elements,
                existing_tests=existing_tests,
                coverage_before=coverage_before,
            )
            # Record source file path for this plan to enable accurate source extraction
            self._plan_source_map[id(plan)] = file_path

            return plan

        except Exception as e:
            logger.warning("Failed to create generation plan for %s: %s", file_path, e)
            # Return None instead of invalid plan when parsing fails
            return None

    def _find_existing_test_files(self, source_file: Path) -> list[str]:
        """Find existing test files for a source file."""
        existing_tests = []

        # Common test file patterns
        test_patterns = [
            source_file.parent / f"test_{source_file.name}",
            source_file.parent / f"{source_file.stem}_test.py",
            source_file.parent.parent / "tests" / f"test_{source_file.name}",
        ]

        for pattern in test_patterns:
            if pattern.exists():
                existing_tests.append(str(pattern))

        return existing_tests
