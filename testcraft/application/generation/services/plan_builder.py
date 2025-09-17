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
from ....adapters.coverage.quick_probe import CoverageQuickProbeAdapter
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
        coverage_probe: CoverageQuickProbeAdapter | None = None,
        *,
        always_analyze_new_files: bool = False,
    ):
        """
        Initialize the plan builder service.

        Args:
            parser_port: Port for code parsing operations
            file_discovery_service: Service for file discovery
            telemetry_port: Port for telemetry operations
            coverage_threshold: Coverage threshold for processing decisions
            coverage_probe: Optional coverage probe adapter for Tier 3 discovery
        """
        self._parser = parser_port
        self._file_discovery = file_discovery_service
        self._telemetry = telemetry_port
        self._coverage_threshold = coverage_threshold
        self._test_mapper = TestMapper()
        self._coverage_probe = coverage_probe
        self._always_analyze_new_files = always_analyze_new_files

        # Per-run caches for AST-based test detection
        self._current_project_path: Path | None = None
        self._cached_test_files: list[str] | None = None
        self._cached_test_discovery_result: dict[str, Any] | None = None
        
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
            # Check if file has existing tests using multi-tier discovery
            has_existing_tests, test_detection_reason = self._has_existing_tests(file_path)
            
            # Log the detection decision for debugging
            logger.debug(f"Test detection for {file_path}: {test_detection_reason}")

            # Check current coverage for this file
            current_coverage = 0.0

            # This is a simplified check - in reality we'd look up specific file coverage
            if coverage_data.get("files_covered", 0) > 0:
                current_coverage = coverage_data.get("overall_line_coverage", 0)

            # File needs processing if:
            # 1. Always analyze flag is set (overrides other checks)
            # 2. No existing tests, OR
            # 3. Coverage below threshold
            needs_processing = (
                self._always_analyze_new_files
                or (not has_existing_tests)
                or (current_coverage < self._coverage_threshold)
            )
            
            # Enhanced logging and telemetry for the final decision
            decision_reason = ""
            if needs_processing:
                if not has_existing_tests:
                    decision_reason = f"no_existing_tests: {test_detection_reason}"
                    logger.info(f"Will generate tests for {file_path}: {test_detection_reason}")
                else:
                    decision_reason = f"low_coverage: existing tests found but coverage {current_coverage:.1%} < {self._coverage_threshold:.1%}"
                    logger.info(f"Will generate tests for {file_path}: existing tests found but coverage {current_coverage:.1%} < {self._coverage_threshold:.1%}")
            else:
                decision_reason = f"excluded: {test_detection_reason}"
                logger.info(f"Skipping {file_path}: {test_detection_reason}")
            
            # Record telemetry metrics
            with self._telemetry.create_child_span("file_processing_decision") as span:
                span.set_attribute("file_path", str(file_path))
                span.set_attribute("needs_processing", needs_processing)
                span.set_attribute("has_existing_tests", has_existing_tests)
                span.set_attribute("current_coverage", current_coverage)
                span.set_attribute("coverage_threshold", self._coverage_threshold)
                span.set_attribute("decision_reason", decision_reason)
            
            return needs_processing

        except Exception as e:
            logger.warning(
                "Failed to check if file %s needs processing: %s", file_path, e
            )
            # When in doubt, process the file
            return True

    def _has_existing_tests(self, file_path: Path) -> tuple[bool, str]:
        """
        Determine if the source file already has tests using multi-tier discovery.

        Uses the new hybrid discovery system with usage-based mapping,
        optional coverage probing, and detailed reasoning.

        Args:
            file_path: Path to the source file

        Returns:
            Tuple of (has_tests: bool, reason: str) explaining the decision
        """
        try:
            # Ensure we have test discovery results cached
            if self._cached_test_discovery_result is None and self._current_project_path is not None:
                self._cached_test_discovery_result = self._file_discovery.discover_and_classify_tests(
                    self._current_project_path
                )

            discovery_result = self._cached_test_discovery_result
            if not discovery_result or not discovery_result.get('test_files'):
                return False, "No test files found in project"

            test_files = [Path(f) for f in discovery_result['test_files']]
            
            # Tier 2B: Usage-based AST mapping
            try:
                mapping_result = self._test_mapper.map_source_to_tests_usage_based(
                    file_path, test_files, self._current_project_path
                )
                
                mapping_score = sum(mapping_result.get('mapping_scores', {}).values())
                coverage_pct = mapping_result.get('coverage_percentage', 0.0)
                
                # Apply configurable threshold (default: score >= 2)
                min_score = getattr(self._file_discovery.config.test_discovery, 'mapper_min_score', 2)
                
                if mapping_score >= min_score:
                    return True, f"Usage-based mapping found tests (score: {mapping_score}, coverage: {coverage_pct:.1f}%)"
                
                # If mapping is inconclusive, try coverage probe if enabled
                if (self._coverage_probe and 
                    getattr(self._file_discovery.config.test_discovery, 'enable_coverage_probe', False)):
                    
                    probe_result = self._coverage_probe.probe(
                        self._current_project_path, file_path
                    )
                    
                    if probe_result.success and probe_result.executed:
                        return True, f"Coverage probe detected execution ({probe_result.coverage_percentage:.1f}% coverage)"
                    elif probe_result.success and probe_result.executed is False:
                        return False, f"Coverage probe found no execution (mapping score: {mapping_score})"
                
                # Neither mapping nor probe found positive evidence
                return False, f"No evidence after multi-tier analysis (mapping score: {mapping_score})"
            
            except Exception as e:
                logger.debug(f"Usage-based mapping failed for {file_path}: {e}")
                # Fall back to pattern-based test detection
                return self._pattern_based_test_detection(file_path, test_files)

        except Exception as e:
            logger.debug(f"Multi-tier test detection failed for {file_path}: {e}")
            # Final fallback: simple filename patterns
            return self._filename_pattern_fallback(file_path)

    def _pattern_based_test_detection(self, file_path: Path, test_files: list[Path]) -> tuple[bool, str]:
        """Pattern-based test detection using AST element mapping (fallback method)."""
        try:
            # Parse source elements
            parse_result = self._parser.parse_file(file_path)
            elements = parse_result.get("elements", [])
            if not elements:
                return False, "No parseable elements in source file"

            # Map tests to source elements using pattern-based TestMapper
            mapping = self._test_mapper.map_tests(elements, existing_tests=[str(f) for f in test_files])
            coverage_pct = float(mapping.get("coverage_percentage", 0.0))
            
            if coverage_pct > 0.0:
                return True, f"Pattern-based AST mapping found tests ({coverage_pct:.1f}% coverage)"
            else:
                return False, f"Pattern-based AST mapping found no tests"

        except Exception as e:
            logger.debug(f"Pattern-based test detection failed for {file_path}: {e}")
            return self._filename_pattern_fallback(file_path)

    def _filename_pattern_fallback(self, file_path: Path) -> tuple[bool, str]:
        """Final fallback: simple filename-based heuristics."""
        potential_test_files = [
            file_path.parent / f"test_{file_path.name}",
            file_path.parent / f"{file_path.stem}_test.py",
            file_path.parent.parent / "tests" / f"test_{file_path.name}",
        ]
        
        existing_test_files = [f for f in potential_test_files if f.exists()]
        
        if existing_test_files:
            return True, f"Filename pattern found tests: {[str(f) for f in existing_test_files]}"
        else:
            return False, "No tests found via filename patterns"

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

    def build_plans(self, files_to_process: list[Path]) -> list[TestGenerationPlan]:
        """
        Build test generation plans for a list of files.
        
        Args:
            files_to_process: List of files to create plans for
            
        Returns:
            List of TestGenerationPlan objects for files that have testable elements
        """
        plans = []
        for file_path in files_to_process:
            plan = self._create_generation_plan_for_file(file_path)
            if plan:
                plans.append(plan)
        return plans

    def build_plans_for_elements(
        self, 
        files_to_process: list[Path], 
        selected_element_keys: list[str]
    ) -> list[TestGenerationPlan]:
        """
        Build test generation plans filtered by selected planning elements.
        
        This method creates plans where only the selected elements are included
        for testing, based on planning session selections.
        
        Args:
            files_to_process: List of files to create plans for
            selected_element_keys: List of element keys from planning session
            
        Returns:
            List of TestGenerationPlan objects with filtered elements
        """
        plans = []
        
        # Parse element keys to extract element information
        selected_elements_by_file = self._group_element_keys_by_file(selected_element_keys)
        
        for file_path in files_to_process:
            # Get elements for this file from selection
            file_path_str = str(file_path)
            selected_for_file = selected_elements_by_file.get(file_path_str, [])
            # Backward-compatibility: apply global selections (legacy 3-part keys)
            if not selected_for_file and "all_files" in selected_elements_by_file:
                selected_for_file = selected_elements_by_file["all_files"]
            
            if not selected_for_file:
                continue  # Skip files with no selected elements
            
            # Create plan with filtered elements
            plan = self._create_filtered_generation_plan(file_path, selected_for_file)
            if plan:
                plans.append(plan)
                
        return plans

    def _group_element_keys_by_file(self, selected_element_keys: list[str]) -> dict[str, list[dict]]:
        """Group selected element keys by their source file.

        Supports both canonical keys with file path and legacy keys without it.

        Canonical: "{abs_source_path}::{element.type}::{element.name}::{line_start}-{line_end}"
        Legacy:    "{element.name}::{element.type}::{line_start}-{line_end}"
        """
        grouped: dict[str, list[dict]] = {}
        
        for key in selected_element_keys:
            try:
                parts = key.split("::")
                if len(parts) >= 4:
                    # Canonical format with file path
                    source_path, element_type, element_name, line_range = parts[:4]
                    file_key = str(Path(source_path))
                elif len(parts) == 3:
                    # Legacy format without file path
                    element_name, element_type, line_range = parts
                    file_key = "all_files"
                else:
                    logger.warning(f"Unrecognized element key format: {key}")
                    continue

                element_info = {
                    "name": element_name,
                    "type": element_type,
                    "line_range": line_range,
                    "key": key,
                }
                grouped.setdefault(file_key, []).append(element_info)

            except Exception as e:
                logger.warning(f"Failed to parse element key {key}: {e}")
                continue
                
        return grouped

    def _create_filtered_generation_plan(
        self, 
        file_path: Path, 
        selected_elements: list[dict]
    ) -> TestGenerationPlan | None:
        """
        Create a TestGenerationPlan with only selected elements.
        
        Args:
            file_path: Path to the source file
            selected_elements: List of selected element info from planning
            
        Returns:
            TestGenerationPlan with filtered elements, or None if no matches
        """
        try:
            # Parse the file to get all elements
            parse_result = self._parser.parse_file(file_path)
            all_elements = parse_result.get("elements", [])
            
            if not all_elements:
                return None
                
            # Filter elements based on selection
            filtered_elements = []
            for element in all_elements:
                # Match element with selected items by name and type
                for selected in selected_elements:
                    element_type_value = (
                        element.type.value if hasattr(element.type, "value") else str(element.type)
                    )
                    if (element.name == selected["name"] and 
                        element_type_value == selected["type"]):
                        filtered_elements.append(element)
                        break
            
            if not filtered_elements:
                logger.info(f"No matching selected elements found in {file_path}")
                return None
            
            # Find existing test files
            existing_tests = self._find_existing_test_files(file_path)
            
            # Create plan with filtered elements
            plan = TestGenerationPlan(
                elements_to_test=filtered_elements,
                existing_tests=existing_tests,
                coverage_before=None,
            )
            
            # Record source file path for this plan
            self._plan_source_map[id(plan)] = file_path
            
            logger.info(
                f"Created filtered plan for {file_path} with {len(filtered_elements)} selected elements"
            )
            
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to create filtered plan for {file_path}: {e}")
            return None

    def get_source_path_for_plan(self, plan: TestGenerationPlan) -> Path | None:
        """
        Get the source file path for a test generation plan.
        
        Args:
            plan: The test generation plan
            
        Returns:
            Path to the source file, or None if not found
        """
        return self._plan_source_map.get(id(plan))
