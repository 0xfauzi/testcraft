"""
Generate Use Case - Core test generation orchestration.

This module implements the primary use case for generating tests, orchestrating
all the different ports and adapters to provide end-to-end test generation
functionality with proper error handling, batching, and telemetry.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..domain.models import (
    GenerationResult,
    TestGenerationPlan,
)
from ..ports.context_port import ContextPort
from ..ports.coverage_port import CoveragePort
from ..ports.llm_port import LLMPort
from ..ports.parser_port import ParserPort
from ..ports.refine_port import RefinePort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import MetricValue, SpanKind, TelemetryPort
from ..ports.writer_port import WriterPort

logger = logging.getLogger(__name__)


class GenerateUseCaseError(Exception):
    """Exception for Generate Use Case specific errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class GenerateUseCase:
    """
    Core use case for test generation.

    Orchestrates the entire test generation workflow including:
    - State synchronization and file discovery
    - File processing decisions and test planning
    - Context gathering and prompt building
    - LLM-based test generation with validation
    - Test file writing with configured strategies
    - Optional pytest execution and refinement
    - Coverage measurement and telemetry reporting
    """

    def __init__(
        self,
        llm_port: LLMPort,
        writer_port: WriterPort,
        coverage_port: CoveragePort,
        refine_port: RefinePort,
        context_port: ContextPort,
        parser_port: ParserPort,
        state_port: StatePort,
        telemetry_port: TelemetryPort,
        file_discovery_service: FileDiscoveryService | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Generate Use Case with all required ports.

        Args:
            llm_port: Port for LLM operations
            writer_port: Port for file writing operations
            coverage_port: Port for coverage measurement
            refine_port: Port for test refinement
            context_port: Port for context management
            parser_port: Port for code parsing
            state_port: Port for state management
            telemetry_port: Port for telemetry and metrics
            file_discovery_service: Service for file discovery (creates default if None)
            config: Optional configuration overrides
        """
        # Initialize all ports
        self._llm = llm_port
        self._writer = writer_port
        self._coverage = coverage_port
        self._refine = refine_port
        self._context = context_port
        self._parser = parser_port
        self._state = state_port
        self._telemetry = telemetry_port

        # Initialize file discovery service
        self._file_discovery = file_discovery_service or FileDiscoveryService()

        # Configuration with sensible defaults
        self._config = {
            "batch_size": 5,  # Number of files to process in parallel
            "enable_context": True,  # Whether to use context retrieval
            "enable_refinement": True,  # Whether to refine failed tests
            "max_refinement_iterations": 3,  # Max refinement attempts
            "coverage_threshold": 0.8,  # Coverage threshold for reporting
            "test_framework": "pytest",  # Default test framework
            "enable_streaming": False,  # Whether to use streaming LLM responses
            **(config or {}),
        }

        # Initialize thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=self._config["batch_size"])

    async def generate_tests(
        self,
        project_path: str | Path,
        target_files: list[str | Path] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Main entry point for test generation.

        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to process
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generation results, statistics, and metadata
        """
        project_path = Path(project_path)

        with self._telemetry.create_span(
            "generate_tests",
            kind=SpanKind.INTERNAL,
            attributes={
                "project_path": str(project_path),
                "target_files_count": len(target_files) if target_files else 0,
                "config": self._config,
            },
        ) as span:
            try:
                logger.info("Starting test generation for project: %s", project_path)

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

                # Step 4: Build test generation plans for each file
                generation_plans = await self._build_generation_plans(files_to_process)
                span.set_attribute("generation_plans_created", len(generation_plans))

                # Step 5: Build directory tree and gather context if enabled
                context_data = (
                    await self._gather_project_context(project_path, files_to_process)
                    if self._config["enable_context"]
                    else {}
                )

                # Step 6: Apply batching policy and generate tests
                generation_results = await self._execute_test_generation(
                    generation_plans, context_data
                )
                span.set_attribute("tests_generated", len(generation_results))

                # Step 7: Write test files using configured strategy
                write_results = await self._write_test_files(generation_results)
                span.set_attribute("files_written", len(write_results))

                # Step 8: Optional pytest execution and refinement
                refinement_results = []
                if self._config["enable_refinement"]:
                    refinement_results = await self._execute_and_refine_tests(
                        write_results
                    )
                    span.set_attribute("files_refined", len(refinement_results))

                # Step 9: Measure coverage delta and record state
                final_coverage = await self._measure_final_coverage(
                    discovery_result["files"]
                )
                coverage_delta = self._calculate_coverage_delta(
                    initial_coverage, final_coverage
                )
                span.set_attribute(
                    "final_coverage", final_coverage.get("overall_line_coverage", 0)
                )
                span.set_attribute(
                    "coverage_delta", coverage_delta.get("line_coverage_delta", 0)
                )

                # Step 10: Record state and generate report
                await self._record_final_state(
                    generation_results, refinement_results, coverage_delta
                )

                # Step 11: Record telemetry and cost summary
                await self._record_telemetry_and_costs(
                    span, generation_results, refinement_results
                )

                # Compile final results
                results = {
                    "success": True,
                    "files_discovered": len(discovery_result["files"]),
                    "files_processed": len(files_to_process),
                    "tests_generated": len(generation_results),
                    "files_written": len(write_results),
                    "files_refined": len(refinement_results),
                    "initial_coverage": initial_coverage,
                    "final_coverage": final_coverage,
                    "coverage_delta": coverage_delta,
                    "generation_results": generation_results,
                    "refinement_results": refinement_results,
                    "metadata": {
                        "project_path": str(project_path),
                        "config_used": self._config,
                        "timestamp": discovery_result.get("timestamp"),
                    },
                }

                logger.info(
                    "Test generation completed successfully. Files: %d, Tests generated: %d, Coverage delta: %.2f%%",
                    len(files_to_process),
                    len(generation_results),
                    coverage_delta.get("line_coverage_delta", 0) * 100,
                )

                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Test generation failed: %s", e)
                raise GenerateUseCaseError(f"Test generation failed: {e}", cause=e)

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
                raise GenerateUseCaseError(f"File discovery failed: {e}", cause=e)

    async def _measure_initial_coverage(
        self, source_files: list[Path]
    ) -> dict[str, Any]:
        """
        Measure initial code coverage before test generation.

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
        Decide which files need test processing based on coverage and other criteria.

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
                    # Check if file needs test generation
                    needs_processing = await self._file_needs_processing(
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

    async def _file_needs_processing(
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
            # Check if file has existing tests
            potential_test_files = [
                file_path.parent / f"test_{file_path.name}",
                file_path.parent / f"{file_path.stem}_test.py",
                file_path.parent.parent / "tests" / f"test_{file_path.name}",
            ]

            has_existing_tests = any(
                test_file.exists() for test_file in potential_test_files
            )

            # Check current coverage for this file
            current_coverage = 0.0

            # This is a simplified check - in reality we'd look up specific file coverage
            if coverage_data.get("files_covered", 0) > 0:
                current_coverage = coverage_data.get("overall_line_coverage", 0)

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

    async def _build_generation_plans(
        self, files_to_process: list[Path]
    ) -> list[TestGenerationPlan]:
        """
        Build TestGenerationPlan objects for each file to process.

        Args:
            files_to_process: Files that need test generation

        Returns:
            List of test generation plans
        """
        with self._telemetry.create_child_span("build_generation_plans") as span:
            plans = []

            try:
                for file_path in files_to_process:
                    plan = await self._create_generation_plan_for_file(file_path)
                    plans.append(plan)

                span.set_attribute("plans_created", len(plans))
                return plans

            except Exception as e:
                logger.exception("Failed to build generation plans: %s", e)
                raise GenerateUseCaseError(
                    f"Generation plan creation failed: {e}", cause=e
                )

    async def _create_generation_plan_for_file(
        self, file_path: Path
    ) -> TestGenerationPlan:
        """
        Create a TestGenerationPlan for a specific file.

        Args:
            file_path: Path to the file to create a plan for

        Returns:
            TestGenerationPlan for the file
        """
        try:
            # Parse the file to extract testable elements
            parse_result = self._parser.parse_file(file_path)
            elements = parse_result.get("elements", [])

            # Find existing test files
            existing_tests = self._find_existing_test_files(file_path)

            # Get current coverage for this file (simplified)
            coverage_before = None  # Would implement file-specific coverage lookup

            return TestGenerationPlan(
                elements_to_test=elements,
                existing_tests=existing_tests,
                coverage_before=coverage_before,
            )

        except Exception as e:
            logger.warning("Failed to create generation plan for %s: %s", file_path, e)
            # Return minimal plan rather than failing
            return TestGenerationPlan(
                elements_to_test=[], existing_tests=[], coverage_before=None
            )

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

    async def _gather_project_context(
        self, project_path: Path, files_to_process: list[Path]
    ) -> dict[str, Any]:
        """
        Gather project context including directory tree and codebase information.

        Args:
            project_path: Root path of the project
            files_to_process: Files being processed

        Returns:
            Dictionary containing context information
        """
        with self._telemetry.create_child_span("gather_project_context") as span:
            try:
                # Build project context graph
                context_graph = self._context.build_context_graph(project_path)

                # Index files for context retrieval
                indexed_files = {}
                for file_path in files_to_process:
                    try:
                        index_result = self._context.index(file_path)
                        indexed_files[str(file_path)] = index_result
                    except Exception as e:
                        logger.warning("Failed to index %s: %s", file_path, e)

                span.set_attribute("indexed_files", len(indexed_files))

                return {
                    "context_graph": context_graph,
                    "indexed_files": indexed_files,
                    "project_structure": self._build_directory_tree(project_path),
                }

            except Exception as e:
                logger.warning("Failed to gather project context: %s", e)
                return {}

    def _build_directory_tree(self, project_path: Path) -> dict[str, Any]:
        """Build a simplified directory tree representation."""
        try:
            tree = {"name": project_path.name, "type": "directory", "children": []}

            # Add immediate Python files and directories (simplified)
            for item in project_path.iterdir():
                if item.is_file() and item.suffix == ".py":
                    tree["children"].append(
                        {"name": item.name, "type": "file", "path": str(item)}
                    )
                elif item.is_dir() and not item.name.startswith("."):
                    tree["children"].append({"name": item.name, "type": "directory"})

            return tree

        except Exception as e:
            logger.warning("Failed to build directory tree: %s", e)
            return {}

    async def _execute_test_generation(
        self, generation_plans: list[TestGenerationPlan], context_data: dict[str, Any]
    ) -> list[GenerationResult]:
        """
        Execute LLM-based test generation with batching and concurrency.

        Args:
            generation_plans: List of test generation plans
            context_data: Project context information

        Returns:
            List of generation results
        """
        with self._telemetry.create_child_span("execute_test_generation") as span:
            try:
                # Apply batching strategy
                batch_size = self._config["batch_size"]
                generation_results = []

                # Process plans in batches
                for i in range(0, len(generation_plans), batch_size):
                    batch = generation_plans[i : i + batch_size]
                    span.set_attribute(f"batch_{i // batch_size}_size", len(batch))

                    # Process batch concurrently
                    batch_results = await self._process_generation_batch(
                        batch, context_data
                    )
                    generation_results.extend(batch_results)

                span.set_attribute("total_generated", len(generation_results))
                span.set_attribute(
                    "successful_generations",
                    sum(1 for r in generation_results if r.success),
                )

                return generation_results

            except Exception as e:
                logger.exception("Test generation execution failed: %s", e)
                raise GenerateUseCaseError(
                    f"Test generation execution failed: {e}", cause=e
                )

    async def _process_generation_batch(
        self, batch: list[TestGenerationPlan], context_data: dict[str, Any]
    ) -> list[GenerationResult]:
        """Process a batch of generation plans concurrently."""
        tasks = []

        for plan in batch:
            task = self._generate_tests_for_plan(plan, context_data)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed GenerationResult objects
        generation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Generation failed for plan %d: %s", i, result)
                generation_results.append(
                    GenerationResult(
                        file_path=f"plan_{i}",  # Would use actual file path from plan
                        content=None,
                        success=False,
                        error_message=str(result),
                    )
                )
            else:
                generation_results.append(result)

        return generation_results

    async def _generate_tests_for_plan(
        self, plan: TestGenerationPlan, context_data: dict[str, Any]
    ) -> GenerationResult:
        """
        Generate tests for a single test generation plan.

        Args:
            plan: The test generation plan
            context_data: Project context information

        Returns:
            GenerationResult for this plan
        """
        try:
            # Build code content from plan elements
            code_content = self._build_code_content_from_plan(plan)

            # Get relevant context for this file
            relevant_context = await self._get_relevant_context(plan, context_data)

            # Call LLM to generate tests
            llm_result = self._llm.generate_tests(
                code_content=code_content,
                context=relevant_context,
                test_framework=self._config["test_framework"],
            )

            # Extract and validate the generated content
            test_content = llm_result.get("tests", "")
            if not test_content or not test_content.strip():
                return GenerationResult(
                    file_path="unknown",  # Would use actual path
                    content=None,
                    success=False,
                    error_message="LLM returned empty test content",
                )

            # Determine output file path
            output_path = self._determine_test_file_path(plan)

            return GenerationResult(
                file_path=output_path,
                content=test_content,
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning("Test generation failed for plan: %s", e)
            return GenerationResult(
                file_path="unknown", content=None, success=False, error_message=str(e)
            )

    def _build_code_content_from_plan(self, plan: TestGenerationPlan) -> str:
        """Build combined code content from test elements in the plan."""
        # In a real implementation, this would read the actual source file
        # and extract the relevant code sections based on the elements
        code_parts = []

        for element in plan.elements_to_test:
            # Simplified - would read actual code from file
            code_parts.append(f"# {element.type.value}: {element.name}")
            if element.docstring:
                code_parts.append(f'"""{element.docstring}"""')

        return "\n".join(code_parts)

    async def _get_relevant_context(
        self, plan: TestGenerationPlan, context_data: dict[str, Any]
    ) -> str | None:
        """Get relevant context for test generation."""
        if not self._config["enable_context"] or not context_data:
            return None

        try:
            # Build context query from plan elements
            query_parts = [
                element.name for element in plan.elements_to_test[:3]
            ]  # Limit query size
            query = " ".join(query_parts)

            # Retrieve relevant context
            context_result = self._context.retrieve(
                query=query, context_type="general", limit=5
            )

            # Format context for LLM
            if context_result.get("results"):
                context_items = []
                for result in context_result["results"][:3]:  # Limit context size
                    if isinstance(result, dict) and "content" in result:
                        context_items.append(result["content"])

                return "\n".join(context_items) if context_items else None

            return None

        except Exception as e:
            logger.warning("Failed to retrieve context: %s", e)
            return None

    def _determine_test_file_path(self, plan: TestGenerationPlan) -> str:
        """Determine the output path for the test file."""
        # Simplified implementation - would use actual source file paths from plan
        if plan.elements_to_test:
            # Use first element to determine naming
            element = plan.elements_to_test[0]
            return f"test_{element.name.lower()}.py"
        return "test_generated.py"

    async def _write_test_files(
        self, generation_results: list[GenerationResult]
    ) -> list[dict[str, Any]]:
        """
        Write generated test files using configured writing strategy.

        Args:
            generation_results: Results from test generation

        Returns:
            List of write operation results
        """
        with self._telemetry.create_child_span("write_test_files") as span:
            write_results = []

            try:
                for result in generation_results:
                    if not result.success or not result.content:
                        # Skip failed generations
                        continue

                    try:
                        # Write the test file
                        write_result = self._writer.write_test_file(
                            test_path=result.file_path,
                            test_content=result.content,
                            overwrite=True,  # Configuration could control this
                        )

                        write_results.append(
                            {
                                "source_result": result,
                                "write_result": write_result,
                                "success": write_result.get("success", False),
                            }
                        )

                    except Exception as e:
                        logger.warning(
                            "Failed to write test file %s: %s", result.file_path, e
                        )
                        write_results.append(
                            {
                                "source_result": result,
                                "write_result": {"success": False, "error": str(e)},
                                "success": False,
                            }
                        )

                span.set_attribute(
                    "files_written", sum(1 for r in write_results if r["success"])
                )

                return write_results

            except Exception as e:
                logger.exception("Test file writing failed: %s", e)
                raise GenerateUseCaseError(f"Test file writing failed: {e}", cause=e)

    async def _execute_and_refine_tests(
        self, write_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Execute tests with pytest and refine failures.

        Args:
            write_results: Results from test file writing

        Returns:
            List of refinement results
        """
        if not self._config["enable_refinement"]:
            return []

        with self._telemetry.create_child_span("execute_and_refine_tests") as span:
            refinement_results = []

            try:
                successful_writes = [r for r in write_results if r["success"]]

                for write_result in successful_writes:
                    test_file_path = write_result["source_result"].file_path

                    try:
                        # Attempt refinement for this test file
                        refinement_result = await self._refine_test_file(test_file_path)
                        refinement_results.append(refinement_result)

                    except Exception as e:
                        logger.warning(
                            "Refinement failed for %s: %s", test_file_path, e
                        )
                        refinement_results.append(
                            {
                                "test_file": test_file_path,
                                "success": False,
                                "error": str(e),
                                "iterations": 0,
                            }
                        )

                span.set_attribute(
                    "files_refined",
                    sum(1 for r in refinement_results if r.get("success", False)),
                )

                return refinement_results

            except Exception as e:
                logger.exception("Test execution and refinement failed: %s", e)
                raise GenerateUseCaseError(
                    f"Test execution and refinement failed: {e}", cause=e
                )

    async def _refine_test_file(self, test_file_path: str) -> dict[str, Any]:
        """
        Refine a single test file through pytest execution and LLM refinement.

        This method implements the complete refinement workflow:
        1. Run pytest to get failure output
        2. If tests pass, return success
        3. If tests fail, use refine port to fix failures
        4. Repeat until max iterations or tests pass
        5. Detect no-change scenarios to avoid infinite loops

        Args:
            test_file_path: Path to the test file to refine

        Returns:
            Dictionary with refinement results including success status,
            iterations used, final pytest status, and any errors
        """
        max_iterations = self._config["max_refinement_iterations"]
        test_file = Path(test_file_path)

        # Track content between iterations to detect no-change scenarios
        previous_content = None
        if test_file.exists():
            try:
                previous_content = test_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read test file %s: %s", test_file_path, e)

        with self._telemetry.create_child_span("refine_test_file") as span:
            span.set_attribute("test_file", test_file_path)
            span.set_attribute("max_iterations", max_iterations)

            for iteration in range(max_iterations):
                span.set_attribute(f"iteration_{iteration}_started", True)

                try:
                    # Step 1: Run pytest to get current test status
                    pytest_result = await self._run_pytest_for_file(test_file_path)
                    span.set_attribute(
                        f"iteration_{iteration}_pytest_returncode",
                        pytest_result["returncode"],
                    )

                    # Step 2: Check if tests are now passing
                    if pytest_result["returncode"] == 0:
                        # Tests pass! Refinement successful
                        span.set_attribute("refinement_successful", True)
                        span.set_attribute("final_iteration", iteration)

                        return {
                            "test_file": test_file_path,
                            "success": True,
                            "iterations": iteration + 1,
                            "final_status": "passed",
                            "final_output": pytest_result.get("stdout", ""),
                            "refinement_details": f"Tests passing after {iteration + 1} iteration(s)",
                        }

                    # Step 3: Tests are failing, attempt refinement
                    failure_output = self._format_pytest_failure_output(pytest_result)

                    # Build source context for better refinement
                    source_context = await self._build_source_context_for_test(
                        test_file
                    )

                    # Use refine port to fix the failures
                    refine_result = self._refine.refine_from_failures(
                        test_file=test_file,
                        failure_output=failure_output,
                        source_context=source_context,
                        max_iterations=1,  # Handle one iteration at a time for better control
                    )

                    span.set_attribute(
                        f"iteration_{iteration}_refine_success",
                        refine_result.get("success", False),
                    )

                    if not refine_result.get("success"):
                        logger.warning(
                            "Refinement failed on iteration %d for %s: %s",
                            iteration + 1,
                            test_file_path,
                            refine_result.get("error", "Unknown error"),
                        )
                        continue

                    # Step 4: Check if content actually changed (no-change detection)
                    refined_content = refine_result.get("refined_content")
                    if (
                        refined_content
                        and refined_content.strip() == (previous_content or "").strip()
                    ):
                        # Content didn't change, avoid infinite loop
                        logger.warning(
                            "No content changes detected in iteration %d for %s, stopping refinement",
                            iteration + 1,
                            test_file_path,
                        )
                        span.set_attribute("stopped_reason", "no_content_change")

                        return {
                            "test_file": test_file_path,
                            "success": False,
                            "iterations": iteration + 1,
                            "final_status": "no_change_detected",
                            "error": "Refinement made no changes to test content",
                            "last_failure": failure_output,
                        }

                    # Step 5: Content changed, update for next iteration
                    previous_content = refined_content

                except Exception as e:
                    logger.warning(
                        "Refinement iteration %d failed for %s: %s",
                        iteration + 1,
                        test_file_path,
                        e,
                    )
                    span.set_attribute(f"iteration_{iteration}_error", str(e))
                    continue

            # Step 6: All refinement attempts exhausted
            span.set_attribute("refinement_successful", False)
            span.set_attribute("stopped_reason", "max_iterations_exceeded")

            # Run final pytest to get latest status
            try:
                final_pytest = await self._run_pytest_for_file(test_file_path)
                final_status = "passed" if final_pytest["returncode"] == 0 else "failed"
                final_output = self._format_pytest_failure_output(final_pytest)
            except Exception:
                final_status = "unknown"
                final_output = "Could not determine final test status"

            return {
                "test_file": test_file_path,
                "success": False,
                "iterations": max_iterations,
                "final_status": final_status,
                "error": f"Maximum refinement iterations ({max_iterations}) exceeded",
                "last_failure": final_output,
            }

    async def _run_pytest_for_file(self, test_file_path: str) -> dict[str, Any]:
        """
        Run pytest on a specific test file and return results.

        Uses the new async_runner abstraction which wraps the established
        subprocess patterns (python_runner + subprocess_safe) for async workflows.

        Args:
            test_file_path: Path to the test file to run

        Returns:
            Dictionary with pytest execution results including stdout, stderr, returncode
        """
        from ..adapters.io.async_runner import run_python_module_async_with_executor

        try:
            # Use consistent pytest arguments with existing refine adapter patterns
            pytest_args = [
                str(test_file_path),
                "-v",  # verbose output for clear failure details
                "--tb=short",  # shorter tracebacks for cleaner LLM consumption
                "-x",  # stop on first failure for faster refinement cycles
            ]

            # Use the reusable async subprocess abstraction with our existing executor
            stdout, stderr, returncode = await run_python_module_async_with_executor(
                executor=self._executor,
                module_name="pytest",
                args=pytest_args,
                timeout=60,
                raise_on_error=False,  # Handle failures ourselves like refine adapter
            )

            return {
                "stdout": stdout or "",
                "stderr": stderr or "",
                "returncode": returncode,
                "command": f"python -m pytest {' '.join(pytest_args)}",
            }

        except Exception as e:
            logger.warning("Failed to run pytest for %s: %s", test_file_path, e)
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": f"python -m pytest {test_file_path}",
                "error": str(e),
            }

    def _format_pytest_failure_output(self, pytest_result: dict[str, Any]) -> str:
        """
        Format pytest execution results into a clean failure output string.

        Args:
            pytest_result: Results from _run_pytest_for_file

        Returns:
            Formatted failure output suitable for LLM refinement
        """
        parts = []

        # Add command that was run
        if pytest_result.get("command"):
            parts.append(f"Command: {pytest_result['command']}")
            parts.append("")

        # Add return code
        returncode = pytest_result.get("returncode", -1)
        parts.append(f"Exit Code: {returncode}")
        parts.append("")

        # Add stdout (test results)
        stdout = pytest_result.get("stdout", "").strip()
        if stdout:
            parts.append("Test Output:")
            parts.append(stdout)
            parts.append("")

        # Add stderr (errors)
        stderr = pytest_result.get("stderr", "").strip()
        if stderr:
            parts.append("Error Output:")
            parts.append(stderr)
            parts.append("")

        # Add any additional error information
        if pytest_result.get("error"):
            parts.append(f"Execution Error: {pytest_result['error']}")

        return "\n".join(parts)

    async def _build_source_context_for_test(
        self, test_file: Path
    ) -> dict[str, Any] | None:
        """
        Build source context for a test file using AST analysis and context discovery.

        Uses our existing ParserPort and ContextPort capabilities for much more reliable
        context discovery than simple pattern matching. This analyzes the test file's
        imports and dependencies to find related source files.

        Args:
            test_file: Path to the test file being refined

        Returns:
            Dictionary with comprehensive source context information or None if unavailable
        """
        try:
            context = {
                "test_file_path": str(test_file),
                "test_content": "",
                "related_source_files": [],
                "imports_context": [],
                "dependency_analysis": {},
                "retrieved_context": [],
                "project_structure": {},
            }

            # Read current test content
            if test_file.exists():
                try:
                    context["test_content"] = test_file.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to read test file %s: %s", test_file, e)
                    return None

            # Skip context gathering if not enabled
            if not self._config.get("enable_context", True):
                return context

            try:
                # Step 1: Use AST analysis to find test file dependencies
                # This is much more reliable than pattern matching
                dependency_analysis = self._parser.analyze_dependencies(test_file)
                context["dependency_analysis"] = dependency_analysis

                # Extract import information
                imports = dependency_analysis.get("imports", [])
                internal_deps = dependency_analysis.get("internal_deps", [])

                # Step 2: Index the test file for context relationships
                if hasattr(self, "_context"):
                    try:
                        # Index the test file if not already indexed
                        index_result = self._context.index(
                            test_file, content=context["test_content"]
                        )
                        logger.debug(
                            "Indexed test file %s: %s", test_file, index_result
                        )

                        # Step 3: Use context port to find related files
                        related_context = self._context.get_related_context(
                            test_file, relationship_type="all"
                        )

                        # Add related files found through context relationships
                        for related_file_path in related_context.get(
                            "related_files", []
                        ):
                            related_path = Path(related_file_path)
                            if related_path.exists() and related_path.suffix == ".py":
                                try:
                                    # Limit content size for performance
                                    source_content = related_path.read_text(
                                        encoding="utf-8"
                                    )
                                    context["related_source_files"].append(
                                        {
                                            "path": str(related_path),
                                            "content": source_content[:2000],
                                            "relationship": "context_analysis",
                                        }
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to read related file %s: %s",
                                        related_path,
                                        e,
                                    )

                        # Step 4: Build intelligent retrieval queries from test context
                        # Extract key terms from test file for context retrieval
                        retrieval_queries = self._extract_test_context_queries(
                            test_file, context["test_content"]
                        )

                        for query in retrieval_queries[
                            :3
                        ]:  # Limit queries for performance
                            try:
                                retrieval_result = self._context.retrieve(
                                    query=query, context_type="general", limit=3
                                )

                                if retrieval_result.get("results"):
                                    context["retrieved_context"].append(
                                        {
                                            "query": query,
                                            "results": retrieval_result["results"][
                                                :2
                                            ],  # Limit results
                                        }
                                    )

                            except Exception as e:
                                logger.warning(
                                    "Context retrieval failed for query '%s': %s",
                                    query,
                                    e,
                                )

                        # Step 5: Add import-based source file discovery
                        # Look for source files corresponding to internal dependencies
                        for dep in internal_deps:
                            potential_source_paths = self._find_source_files_for_module(
                                test_file, dep
                            )
                            for source_path in potential_source_paths:
                                if source_path.exists():
                                    try:
                                        source_content = source_path.read_text(
                                            encoding="utf-8"
                                        )
                                        context["related_source_files"].append(
                                            {
                                                "path": str(source_path),
                                                "content": source_content[:2000],
                                                "relationship": f"import_dependency: {dep}",
                                            }
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            "Failed to read source file %s: %s",
                                            source_path,
                                            e,
                                        )

                        # Step 6: Add imports context for better LLM understanding
                        for import_info in imports:
                            context["imports_context"].append(
                                {
                                    "module": import_info.get("module", ""),
                                    "items": import_info.get("items", []),
                                    "alias": import_info.get("alias", ""),
                                    "is_internal": import_info.get("module", "")
                                    in internal_deps,
                                }
                            )

                    except Exception as e:
                        logger.warning(
                            "Context port analysis failed for %s: %s", test_file, e
                        )
                        # Continue with basic context even if context port fails

            except Exception as e:
                logger.warning("AST/Context analysis failed for %s: %s", test_file, e)
                # Fall back to basic context if advanced analysis fails

            # Step 7: Add project structure context
            try:
                context["project_structure"] = self._build_directory_tree(
                    test_file.parent.parent
                    if test_file.parent != test_file.parent.parent
                    else test_file.parent
                )
            except Exception as e:
                logger.warning("Failed to build project structure context: %s", e)

            return context

        except Exception as e:
            logger.warning("Failed to build source context for %s: %s", test_file, e)
            return None

    def _extract_test_context_queries(
        self, test_file: Path, test_content: str
    ) -> list[str]:
        """
        Extract intelligent search queries from test file content for context retrieval.

        Analyzes the test file to identify key concepts and functions being tested
        to build targeted search queries for finding relevant context.

        Args:
            test_file: Path to the test file
            test_content: Content of the test file

        Returns:
            List of search query strings for context retrieval
        """
        queries = []

        try:
            # Parse test content using AST to find test functions and their patterns
            import ast
            import re

            try:
                tree = ast.parse(test_content)

                # Extract test function names and build queries from them
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith(
                        "test_"
                    ):
                        # Convert test function name to search query
                        # test_user_login -> "user login"
                        clean_name = node.name[5:]  # Remove "test_" prefix
                        query_words = re.findall(r"[a-z]+", clean_name.lower())
                        if query_words:
                            queries.append(" ".join(query_words))

                        # Extract string literals from test body as additional context
                        for child in ast.walk(node):
                            if isinstance(child, ast.Constant) and isinstance(
                                child.value, str
                            ):
                                # Skip very short or very long strings
                                if (
                                    3 <= len(child.value) <= 30
                                    and child.value.isalnum()
                                ):
                                    queries.append(child.value)

            except SyntaxError as e:
                logger.warning(
                    "Could not parse test file %s for query extraction: %s",
                    test_file,
                    e,
                )

            # Fallback: extract keywords from filename
            if not queries:
                # Convert test_foo_bar.py -> "foo bar"
                test_name = test_file.stem
                if test_name.startswith("test_"):
                    clean_name = test_name[5:]
                    query_words = re.findall(r"[a-z]+", clean_name.lower())
                    if query_words:
                        queries.append(" ".join(query_words))

            # Deduplicate and limit queries
            unique_queries = list(
                dict.fromkeys(queries)
            )  # Preserve order while deduplicating
            return unique_queries[:5]  # Limit to 5 most relevant queries

        except Exception as e:
            logger.warning(
                "Failed to extract context queries from %s: %s", test_file, e
            )
            return [test_file.stem.replace("test_", "").replace("_", " ")]

    def _find_source_files_for_module(
        self, test_file: Path, module_name: str
    ) -> list[Path]:
        """
        Find potential source files for an imported module using intelligent path resolution.

        Uses the test file location and module name to find corresponding source files,
        handling common project structures and naming patterns.

        Args:
            test_file: Path to the test file importing the module
            module_name: Name of the imported module to find

        Returns:
            List of potential source file paths
        """
        potential_paths = []

        try:
            # Convert module.submodule to file paths
            module_parts = module_name.split(".")

            # Strategy 1: Relative to test file location
            base_dirs = [
                test_file.parent,  # Same directory as test
                test_file.parent.parent,  # Parent directory
            ]

            # Strategy 2: Common project structures
            # If test is in tests/ directory, look in parallel source directories
            if "tests" in test_file.parts:
                tests_index = None
                for i, part in enumerate(test_file.parts):
                    if part == "tests":
                        tests_index = i
                        break

                if tests_index is not None and tests_index > 0:
                    project_root = Path(*test_file.parts[:tests_index])
                    base_dirs.extend(
                        [
                            project_root,  # Project root
                            project_root / test_file.parts[0],  # Main package directory
                            project_root / "src",  # Common src/ directory
                            project_root / "lib",  # Common lib/ directory
                        ]
                    )

            # Strategy 3: Build potential file paths from module name
            for base_dir in base_dirs:
                if not base_dir or not base_dir.exists():
                    continue

                # Direct module file: mymodule -> mymodule.py
                direct_file = base_dir / f"{module_parts[-1]}.py"
                if direct_file.exists():
                    potential_paths.append(direct_file)

                # Package module: mypackage.mymodule -> mypackage/mymodule.py
                if len(module_parts) > 1:
                    package_file = base_dir
                    for part in module_parts:
                        package_file = package_file / part
                    package_file = package_file.with_suffix(".py")
                    if package_file.exists():
                        potential_paths.append(package_file)

                # Package init: mypackage -> mypackage/__init__.py
                package_init = base_dir / module_parts[0] / "__init__.py"
                if package_init.exists():
                    potential_paths.append(package_init)

            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for path in potential_paths:
                path_str = str(path)
                if path_str not in seen:
                    seen.add(path_str)
                    unique_paths.append(path)

            return unique_paths

        except Exception as e:
            logger.warning(
                "Failed to find source files for module %s: %s", module_name, e
            )
            return []

    async def _measure_final_coverage(self, source_files: list[Path]) -> dict[str, Any]:
        """
        Measure final code coverage after test generation.

        Args:
            source_files: List of source files to measure coverage for

        Returns:
            Coverage measurement results
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

    def _calculate_coverage_delta(
        self, initial_coverage: dict[str, Any], final_coverage: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate the difference in coverage between initial and final states.

        Args:
            initial_coverage: Initial coverage measurements
            final_coverage: Final coverage measurements

        Returns:
            Coverage delta information
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

    async def _record_final_state(
        self,
        generation_results: list[GenerationResult],
        refinement_results: list[dict[str, Any]],
        coverage_delta: dict[str, Any],
    ) -> None:
        """
        Record final state information for future runs and analysis.

        Args:
            generation_results: Results from test generation
            refinement_results: Results from test refinement
            coverage_delta: Coverage improvement measurements
        """
        try:
            # Compile state data
            state_data = {
                "last_run_timestamp": asyncio.get_event_loop().time(),
                "generation_summary": {
                    "total_files_processed": len(generation_results),
                    "successful_generations": sum(
                        1 for r in generation_results if r.success
                    ),
                    "failed_generations": sum(
                        1 for r in generation_results if not r.success
                    ),
                },
                "refinement_summary": {
                    "total_files_refined": len(refinement_results),
                    "successful_refinements": sum(
                        1 for r in refinement_results if r.get("success", False)
                    ),
                },
                "coverage_improvement": coverage_delta,
                "config_used": self._config.copy(),
            }

            # Record state
            self._state.update_state("last_generation_run", state_data)

            # Persist state for future runs
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._state.persist_state
            )

        except Exception as e:
            logger.warning("Failed to record final state: %s", e)
            # Don't fail the entire operation for state recording issues

    async def _record_telemetry_and_costs(
        self,
        span,
        generation_results: list[GenerationResult],
        refinement_results: list[dict[str, Any]],
    ) -> None:
        """
        Record telemetry metrics and cost information.

        Args:
            span: Current telemetry span
            generation_results: Results from test generation
            refinement_results: Results from test refinement
        """
        try:
            from datetime import datetime

            # Record metrics
            metrics = [
                MetricValue(
                    name="tests_generated_total",
                    value=len(generation_results),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_generated_successful",
                    value=sum(1 for r in generation_results if r.success),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_refined_total",
                    value=len(refinement_results),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_refined_successful",
                    value=sum(1 for r in refinement_results if r.get("success", False)),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
            ]

            self._telemetry.record_metrics(metrics)

            # Record span attributes
            span.set_attribute("total_generation_results", len(generation_results))
            span.set_attribute(
                "successful_generations",
                sum(1 for r in generation_results if r.success),
            )
            span.set_attribute("total_refinement_results", len(refinement_results))
            span.set_attribute(
                "successful_refinements",
                sum(1 for r in refinement_results if r.get("success", False)),
            )

            # Flush telemetry data
            self._telemetry.flush(timeout_seconds=5.0)

        except Exception as e:
            logger.warning("Failed to record telemetry and costs: %s", e)
            # Don't fail the entire operation for telemetry issues

    def __del__(self):
        """Clean up resources."""
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors
