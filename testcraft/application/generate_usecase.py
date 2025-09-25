"""
Generate Use Case - Refactored as thin orchestrator.

This module implements the primary use case for generating tests, now refactored
as a thin orchestrator that delegates to focused, testable services for
improved maintainability and separation of concerns.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..adapters.io.file_status_tracker import FileStatus
from ..domain.models import GenerationResult
from ..ports.context_port import ContextPort
from ..ports.coverage_port import CoveragePort
from ..ports.llm_port import LLMPort
from ..ports.parser_port import ParserPort
from ..ports.refine_port import RefinePort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import MetricValue, SpanKind, TelemetryPort
from ..ports.writer_port import WriterPort
from .generation.config import GenerationConfig
from .generation.services.batch_executor import BatchExecutor
from .generation.services.content_builder import ContentBuilder
from .generation.services.context_assembler import ContextAssembler
from .generation.services.coverage_evaluator import CoverageEvaluator
from .generation.services.generator_guardrails import TestContentValidator
from .generation.services.plan_builder import PlanBuilder
from .generation.services.pytest_refiner import PytestRefiner
from .generation.services.state_discovery import (
    GenerateUseCaseError,
    StateSyncDiscovery,
)
from .generation.services.structure import ModulePathDeriver

logger = logging.getLogger(__name__)


class GenerateUseCase:
    """
    Core use case for test generation - Thin orchestrator.

    Now refactored as a thin orchestrator that delegates to focused services:
    - StateSyncDiscovery: State sync and file discovery
    - CoverageEvaluator: Coverage measurement and deltas
    - PlanBuilder: File selection and plan creation
    - ContentBuilder: Source extraction and test paths
    - ContextAssembler: Context building for generation/refinement
    - BatchExecutor: Concurrent test generation
    - PytestRefiner: Test execution and refinement
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
        Initialize the Generate Use Case orchestrator with all required ports.

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
        # Initialize all ports (unchanged interface)
        self._llm = llm_port
        self._writer = writer_port
        self._coverage = coverage_port
        self._refine = refine_port
        self._context = context_port
        self._parser = parser_port
        self._state = state_port
        self._telemetry = telemetry_port

        # Merge and validate configuration
        self._config = GenerationConfig.merge_config(config)
        GenerationConfig.validate_config(self._config)

        # Initialize file discovery service
        self._file_discovery = file_discovery_service or FileDiscoveryService()

        # Initialize thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=self._config["batch_size"])

        # Initialize all service dependencies
        self._state_discovery = StateSyncDiscovery(
            state_port=state_port,
            file_discovery_service=self._file_discovery,
            telemetry_port=telemetry_port,
        )

        self._coverage_evaluator = CoverageEvaluator(
            coverage_port=coverage_port,
            telemetry_port=telemetry_port,
        )

        self._plan_builder = PlanBuilder(
            parser_port=parser_port,
            file_discovery_service=self._file_discovery,
            telemetry_port=telemetry_port,
            coverage_threshold=self._config["coverage_threshold"],
        )

        self._content_builder = ContentBuilder(parser_port=parser_port)

        self._context_assembler = ContextAssembler(
            context_port=context_port,
            parser_port=parser_port,
            config=self._config,
        )

        # Initialize without status tracker initially - will be set per operation
        self._batch_executor = BatchExecutor(
            telemetry_port=telemetry_port,
            executor=self._executor,
        )

        self._pytest_refiner = PytestRefiner(
            refine_port=refine_port,
            telemetry_port=telemetry_port,
            executor=self._executor,
            max_concurrent_refines=self._config["max_refine_workers"],
            backoff_sec=self._config["refinement_backoff_sec"],
            writer_port=self._writer,
        )

        # Status tracker will be injected per generation operation
        self._current_status_tracker = None

    def set_status_tracker(self, status_tracker) -> None:
        """Set the status tracker for live updates during generation."""
        self._current_status_tracker = status_tracker

        # Update dependent services with the status tracker
        if hasattr(self._batch_executor, "_status_tracker"):
            self._batch_executor._status_tracker = status_tracker
        if hasattr(self._pytest_refiner, "_status_tracker"):
            self._pytest_refiner._status_tracker = status_tracker

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
            (Unchanged from original implementation)
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
                discovery_result = self._state_discovery.sync_and_discover(
                    project_path, target_files
                )
                span.set_attribute("files_discovered", len(discovery_result["files"]))

                # Set up plan builder context with discovered files
                try:
                    test_files = self._file_discovery.discover_test_files(project_path)
                    self._plan_builder.set_project_context(project_path, test_files)
                except Exception:
                    test_files = []
                    self._plan_builder.set_project_context(project_path, [])

                # Step 2: Measure initial coverage
                initial_coverage = self._coverage_evaluator.measure_initial(
                    discovery_result["files"]
                )
                span.set_attribute(
                    "initial_coverage", initial_coverage.get("overall_line_coverage", 0)
                )

                # Step 3: Decide which files to process and build plans
                files_to_process = self._plan_builder.decide_files_to_process(
                    discovery_result["files"], initial_coverage
                )
                span.set_attribute("files_to_process", len(files_to_process))

                generation_plans = self._plan_builder.build_plans(files_to_process)
                span.set_attribute("generation_plans_created", len(generation_plans))

                # Check if we have any valid plans to work with
                if not generation_plans:
                    logger.warning("No testable elements found in any source files")
                    return {
                        "generation_summary": {
                            "total_files_processed": len(files_to_process),
                            "total_tests_generated": 0,
                            "total_elements_tested": 0,
                            "files_with_issues": [],
                        },
                        "files_generated": [],
                        "success": True,
                        "message": "No testable elements found in source files",
                    }

                # Step 4: Build directory tree and gather context if enabled
                context_data = {}
                if self._config["enable_context"]:
                    context_data = self._context_assembler.gather_project_context(
                        project_path, files_to_process
                    )

                # Step 5: Generate, write, and refine tests
                # Branch on immediate refinement mode
                if self._config["immediate_refinement"]:
                    # Immediate per-file pipeline: generate → write → refine for each plan
                    (
                        generation_results,
                        write_results,
                        refinement_results,
                    ) = await self._process_plans_immediately(
                        generation_plans, context_data
                    )
                    span.set_attribute("immediate_mode", True)
                else:
                    # Legacy mode: batch all operations
                    generation_results = await self._batch_executor.run_in_batches(
                        generation_plans,
                        self._config["batch_size"],
                        lambda plan: self._generate_tests_for_plan(plan, context_data),
                    )
                    span.set_attribute("tests_generated", len(generation_results))

                    # Step 6: Write test files using configured strategy
                    write_results = await self._write_test_files(generation_results)
                    span.set_attribute("files_written", len(write_results))

                    # Step 7: Optional pytest execution and refinement
                    refinement_results = []
                    if self._config["enable_refinement"]:
                        refinement_results = await self._execute_and_refine_tests(
                            write_results
                        )
                        span.set_attribute("files_refined", len(refinement_results))
                    span.set_attribute("immediate_mode", False)

                span.set_attribute("tests_generated", len(generation_results))
                span.set_attribute("files_written", len(write_results))
                span.set_attribute("files_refined", len(refinement_results))

                # Step 8: Measure coverage delta and record state
                final_coverage = self._coverage_evaluator.measure_final(
                    discovery_result["files"]
                )
                coverage_delta = self._coverage_evaluator.calculate_delta(
                    initial_coverage, final_coverage
                )
                span.set_attribute(
                    "final_coverage", final_coverage.get("overall_line_coverage", 0)
                )
                span.set_attribute(
                    "coverage_delta", coverage_delta.get("line_coverage_delta", 0)
                )

                # Step 9: Record state and generate report
                await self._record_final_state(
                    generation_results, refinement_results, coverage_delta
                )

                # Step 10: Record telemetry and cost summary
                await self._record_telemetry_and_costs(
                    span, generation_results, refinement_results
                )

                # Compile final results (unchanged format)
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
                raise GenerateUseCaseError(
                    f"Test generation failed: {e}", cause=e
                ) from e

    async def _generate_tests_for_plan(
        self, plan, context_data: dict[str, Any]
    ) -> GenerationResult:
        """
        Generate tests for a single test generation plan using services.

        Args:
            plan: The test generation plan
            context_data: Project context information

        Returns:
            GenerationResult for this plan
        """
        try:
            # Get source path for this plan
            source_path = self._plan_builder.get_source_path_for_plan(plan)

            # Build code content from plan elements
            code_content = self._content_builder.build_code_content(plan, source_path)

            # Get relevant context for this file
            context_result = self._context_assembler.context_for_generation(
                plan, source_path
            )

            # Extract context string and import_map from result
            relevant_context = None
            # import_map = None
            if context_result:
                if isinstance(context_result, dict):
                    relevant_context = context_result.get("context")
                    # TODO: USE REAL import_map
                    # import_map = context_result.get("import_map")
                else:
                    # Backward compatibility: if it's still a string
                    relevant_context = context_result

            # Derive authoritative module path and import suggestions
            module_path_info = {}
            if source_path:
                try:
                    # Get project root from context_data or detect automatically
                    project_root = None
                    if context_data.get("project_structure"):
                        # Try to extract project root from context structure
                        project_structure = context_data["project_structure"]
                        if isinstance(
                            project_structure, dict
                        ) and project_structure.get("name"):
                            # This is a simplified approach - in practice you might need more logic
                            project_root = source_path.parent
                            while project_root != project_root.parent:
                                if (project_root / "pyproject.toml").exists():
                                    break
                                project_root = project_root.parent

                    module_path_info = ModulePathDeriver.derive_module_path(
                        source_path, project_root
                    )

                    if module_path_info.get("module_path"):
                        logger.debug(
                            "Derived module path for %s: %s (status: %s)",
                            source_path,
                            module_path_info["module_path"],
                            module_path_info["validation_status"],
                        )

                        # Record telemetry for module path derivation
                        self._record_module_path_telemetry(
                            module_path_info, source_path
                        )

                except Exception as e:
                    logger.warning(
                        "Failed to derive module path for %s: %s", source_path, e
                    )
                    module_path_info = {}

            # Enhance context with module path information
            enhanced_context = relevant_context

            # Note: import_map is now available from ContextAssembler and includes:
            # - target_import: canonical import statement
            # - sys_path_roots: list of sys.path root directories
            # - needs_bootstrap: whether conftest.py setup is needed
            # - bootstrap_conftest: conftest.py content if needed
            # This information is already included in the enriched context string

            if module_path_info and (
                module_path_info.get("module_path")
                or module_path_info.get("import_suggestion")
            ):
                # Append module path info to context
                module_context = []
                if module_path_info.get("module_path"):
                    module_context.append(
                        f"Module Path: {module_path_info['module_path']}"
                    )
                if module_path_info.get("import_suggestion"):
                    module_context.append(
                        f"Import Hint: {module_path_info['import_suggestion']}"
                    )
                if module_path_info.get("validation_status"):
                    module_context.append(
                        f"Import Status: {module_path_info['validation_status']}"
                    )

                module_info_text = "\n".join(module_context)

                # Combine with existing context
                if enhanced_context:
                    enhanced_context = f"{enhanced_context}\n\n# Module Import Information\n{module_info_text}"
                else:
                    enhanced_context = (
                        f"# Module Import Information\n{module_info_text}"
                    )

            # Call LLM to generate tests with enhanced context
            llm_result = await self._llm.generate_tests(
                code_content=code_content,
                context=enhanced_context,
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

            # Validate and potentially fix the generated test content
            enriched_context_data = self._context_assembler.get_last_enriched_context()
            if enriched_context_data and self._config.get("enable_validation", True):
                try:
                    validated_content, is_valid, issues = (
                        TestContentValidator.validate_and_fix(
                            test_content, enriched_context_data
                        )
                    )

                    if issues:
                        # Log validation issues for telemetry
                        issue_summary = "; ".join(str(issue) for issue in issues[:3])
                        logger.info(
                            "Test validation issues for %s: %s",
                            source_path,
                            issue_summary,
                        )

                        # Record validation metrics
                        self._record_validation_telemetry(issues, source_path)

                    # Use validated content if fixes were applied
                    if validated_content != test_content:
                        logger.debug(
                            "Applied validation fixes to test for %s", source_path
                        )
                        test_content = validated_content

                    # If validation failed with errors, note in result but don't fail generation
                    if not is_valid:
                        error_count = sum(
                            1 for issue in issues if issue.severity == "error"
                        )
                        logger.warning(
                            "Test validation failed for %s with %d errors",
                            source_path,
                            error_count,
                        )

                except Exception as e:
                    logger.warning(
                        "Test validation failed with exception for %s: %s",
                        source_path,
                        e,
                    )
                    # Continue with unvalidated content rather than failing generation

            # Determine output file path
            output_path = self._content_builder.determine_test_path(plan)

            return GenerationResult(
                file_path=output_path,
                content=test_content,
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning("Test generation failed for plan: %s", e)
            return GenerationResult(
                file_path="unknown",
                content=None,
                success=False,
                error_message=str(e),
            )

    async def _process_plans_immediately(
        self, generation_plans, context_data: dict[str, Any]
    ) -> tuple[list[GenerationResult], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Process plans with immediate per-file refinement using batch execution.

        This method handles batching of plans while processing each plan immediately
        (generate → write → refine) to provide faster feedback.

        Args:
            generation_plans: List of test generation plans
            context_data: Project context information

        Returns:
            Tuple of (generation_results, write_results, refinement_results)
        """
        # Process plans immediately without using batch_executor
        # since we need different return types for immediate mode
        immediate_results = []

        # Process in batches manually to maintain some parallelism
        batch_size = self._config["batch_size"]
        for i in range(0, len(generation_plans), batch_size):
            batch = generation_plans[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = [
                self._process_plan_immediate(plan, context_data) for plan in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions in results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning("Plan processing failed: %s", result)
                    immediate_results.append(
                        {
                            "generation_result": GenerationResult(
                                file_path=f"plan_{i + j}",
                                content=None,
                                success=False,
                                error_message=str(result),
                            ),
                            "write_result": {"success": False, "error": str(result)},
                            "refinement_result": {
                                "success": False,
                                "error": str(result),
                            },
                            "success": False,
                            "errors": [str(result)],
                        }
                    )
                else:
                    immediate_results.append(result)

        # Separate results into individual lists
        generation_results = []
        write_results = []
        refinement_results = []

        for result in immediate_results:
            if result.get("generation_result"):
                generation_results.append(result["generation_result"])
            if result.get("write_result"):
                write_results.append(result["write_result"])
            if result.get("refinement_result"):
                refinement_results.append(result["refinement_result"])

        return generation_results, write_results, refinement_results

    async def _process_plan_immediate(
        self, plan, context_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a single plan with immediate write-and-refine workflow.

        Implements: generate → write → pytest → refine (iteratively) → return results

        Args:
            plan: The test generation plan
            context_data: Project context information

        Returns:
            Rich result dict containing generation, write, and refinement results
        """
        result = {
            "plan": plan,
            "generation_result": None,
            "write_result": None,
            "refinement_result": None,
            "success": False,
            "errors": [],
        }

        try:
            # Update status for generation start
            if self._current_status_tracker:
                self._current_status_tracker.update_file_status(
                    plan.file_path,
                    FileStatus.GENERATING,
                    operation="LLM Generation",
                    step="Processing source code with AI",
                    progress=25.0,
                )

            # Step 1: Generate test content
            generation_result = await self._generate_tests_for_plan(plan, context_data)
            result["generation_result"] = generation_result

            if not generation_result.success or not generation_result.content:
                result["errors"].append("Test generation failed")
                if self._current_status_tracker:
                    self._current_status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.FAILED,
                        operation="Generation Failed",
                        step=generation_result.error_message
                        or "Test generation failed",
                        progress=0.0,
                    )
                return result

            # Update status for writing
            if self._current_status_tracker:
                self._current_status_tracker.update_file_status(
                    plan.file_path,
                    FileStatus.WRITING,
                    operation="Writing Tests",
                    step="Saving generated test file to disk",
                    progress=60.0,
                )

            # Step 2: Pre-validate and write test file
            try:
                write_result = await self._write_test_file_immediate(generation_result)
                result["write_result"] = write_result

                if not write_result.get("success", False):
                    result["errors"].append(
                        f"Write failed: {write_result.get('error', 'Unknown')}"
                    )

                    if self._current_status_tracker:
                        self._current_status_tracker.update_file_status(
                            plan.file_path,
                            FileStatus.FAILED,
                            operation="Write Failed",
                            step=write_result.get("error", "Unknown write error"),
                            progress=0.0,
                        )

                    # Rollback if configured to do so
                    if not self._config["keep_failed_writes"]:
                        await self._rollback_failed_write(generation_result.file_path)

                    return result

            except Exception as e:
                result["errors"].append(f"Write error: {e}")
                if self._current_status_tracker:
                    self._current_status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.FAILED,
                        operation="Write Error",
                        step=f"Exception during write: {str(e)}",
                        progress=0.0,
                    )
                if not self._config["keep_failed_writes"]:
                    await self._rollback_failed_write(generation_result.file_path)
                return result

            # Step 3: Immediate refinement if enabled
            if self._config["enable_refinement"]:
                try:
                    # Update status for testing start
                    if self._current_status_tracker:
                        self._current_status_tracker.update_file_status(
                            plan.file_path,
                            FileStatus.TESTING,
                            operation="Initial Testing",
                            step="Running pytest on generated tests",
                            progress=80.0,
                        )

                    # Use semaphore-controlled refinement to respect max_refine_workers
                    refinement_result = await self._refine_test_file_immediate(
                        generation_result.file_path
                    )
                    result["refinement_result"] = refinement_result

                    if refinement_result.get("success", False):
                        result["success"] = True
                        if self._current_status_tracker:
                            self._current_status_tracker.update_file_status(
                                plan.file_path,
                                FileStatus.COMPLETED,
                                operation="Tests Passing",
                                step=f"All tests pass after {refinement_result.get('iterations', 1)} iteration(s)",
                                progress=100.0,
                            )
                    else:
                        result["errors"].append(
                            f"Refinement failed: {refinement_result.get('error', 'Unknown')}"
                        )
                        if self._current_status_tracker:
                            self._current_status_tracker.update_file_status(
                                plan.file_path,
                                FileStatus.FAILED,
                                operation="Refinement Failed",
                                step=refinement_result.get(
                                    "error", "Maximum iterations reached"
                                ),
                                progress=0.0,
                            )

                except Exception as e:
                    result["errors"].append(f"Refinement error: {e}")
                    if self._current_status_tracker:
                        self._current_status_tracker.update_file_status(
                            plan.file_path,
                            FileStatus.FAILED,
                            operation="Refinement Error",
                            step=f"Exception during refinement: {str(e)}",
                            progress=0.0,
                        )
            else:
                # No refinement enabled, mark as success if write succeeded
                result["success"] = True
                if self._current_status_tracker:
                    self._current_status_tracker.update_file_status(
                        plan.file_path,
                        FileStatus.COMPLETED,
                        operation="Tests Written",
                        step="Test file saved (refinement disabled)",
                        progress=100.0,
                    )

            return result

        except Exception as e:
            logger.warning("Immediate plan processing failed: %s", e)
            result["errors"].append(f"Processing error: {e}")
            return result
        finally:
            # Record incremental state for this file
            await self._record_per_file_state(result)

    async def _record_per_file_state(self, file_result: dict[str, Any]) -> None:
        """
        Record incremental state and telemetry for a single file's processing.

        Args:
            file_result: Result dictionary from processing a single plan
        """
        try:
            from datetime import datetime

            # Extract file path for logging
            file_path = "unknown"
            if file_result.get("generation_result"):
                file_path = file_result["generation_result"].file_path

            # Build per-file state entry
            file_state = {
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "success": file_result.get("success", False),
                "errors": file_result.get("errors", []),
                "stages": {},
            }

            # Record generation stage
            if file_result.get("generation_result"):
                gen_result = file_result["generation_result"]
                file_state["stages"]["generation"] = {
                    "success": gen_result.success,
                    "content_length": len(gen_result.content)
                    if gen_result.content
                    else 0,
                    "error": gen_result.error_message,
                }

            # Record write stage
            if file_result.get("write_result"):
                write_result = file_result["write_result"]
                file_state["stages"]["write"] = {
                    "success": write_result.get("success", False),
                    "bytes_written": write_result.get("bytes_written", 0),
                    "formatted": write_result.get("formatted", False),
                    "error": write_result.get("error"),
                }

            # Record refinement stage
            if file_result.get("refinement_result"):
                refine_result = file_result["refinement_result"]
                file_state["stages"]["refinement"] = {
                    "success": refine_result.get("success", False),
                    "iterations": refine_result.get("iterations", 0),
                    "final_status": refine_result.get("final_status"),
                    "error": refine_result.get("error"),
                }

            # Store in state using append strategy for incremental logging
            self._state.update_state(
                "immediate_generation_log", file_state, merge_strategy="append"
            )

            # Record telemetry metrics for this file
            await self._record_per_file_telemetry(file_result, file_path)

        except Exception as e:
            logger.warning("Failed to record per-file state for %s: %s", file_path, e)

    async def _record_per_file_telemetry(
        self, file_result: dict[str, Any], file_path: str
    ) -> None:
        """
        Record per-file telemetry metrics and spans.

        Args:
            file_result: Result dictionary from processing a single plan
            file_path: Path to the file being processed
        """
        try:
            from datetime import datetime

            # Create per-file span for detailed tracing
            with self._telemetry.create_child_span(
                "process_file_immediate"
            ) as file_span:
                file_span.set_attribute("file_path", file_path)
                file_span.set_attribute("success", file_result.get("success", False))

                # Generation stage metrics
                if file_result.get("generation_result"):
                    gen_result = file_result["generation_result"]
                    file_span.set_attribute("generation_success", gen_result.success)
                    if gen_result.content:
                        file_span.set_attribute(
                            "content_length", len(gen_result.content)
                        )

                # Write stage metrics
                if file_result.get("write_result"):
                    write_result = file_result["write_result"]
                    file_span.set_attribute(
                        "write_success", write_result.get("success", False)
                    )
                    file_span.set_attribute(
                        "bytes_written", write_result.get("bytes_written", 0)
                    )

                # Refinement stage metrics
                if file_result.get("refinement_result"):
                    refine_result = file_result["refinement_result"]
                    file_span.set_attribute(
                        "refinement_success", refine_result.get("success", False)
                    )
                    file_span.set_attribute(
                        "refinement_iterations", refine_result.get("iterations", 0)
                    )

                # Record errors if any
                if file_result.get("errors"):
                    file_span.set_attribute("error_count", len(file_result["errors"]))
                    file_span.set_attribute("errors", "; ".join(file_result["errors"]))

            # Record individual stage metrics
            metrics = []

            # Generation metrics
            if file_result.get("generation_result"):
                gen_result = file_result["generation_result"]
                metrics.append(
                    MetricValue(
                        name="file_generation_success",
                        value=1 if gen_result.success else 0,
                        unit="count",
                        labels={
                            "file_path": file_path,
                            "framework": self._config["test_framework"],
                        },
                        timestamp=datetime.now(),
                    )
                )

            # Write metrics
            if file_result.get("write_result"):
                write_result = file_result["write_result"]
                metrics.append(
                    MetricValue(
                        name="file_write_success",
                        value=1 if write_result.get("success", False) else 0,
                        unit="count",
                        labels={"file_path": file_path},
                        timestamp=datetime.now(),
                    )
                )

            # Refinement metrics
            if file_result.get("refinement_result"):
                refine_result = file_result["refinement_result"]
                metrics.extend(
                    [
                        MetricValue(
                            name="file_refinement_success",
                            value=1 if refine_result.get("success", False) else 0,
                            unit="count",
                            labels={"file_path": file_path},
                            timestamp=datetime.now(),
                        ),
                        MetricValue(
                            name="file_refinement_iterations",
                            value=refine_result.get("iterations", 0),
                            unit="count",
                            labels={"file_path": file_path},
                            timestamp=datetime.now(),
                        ),
                    ]
                )

            # Record all metrics
            if metrics:
                self._telemetry.record_metrics(metrics)

        except Exception as e:
            logger.warning(
                "Failed to record per-file telemetry for %s: %s", file_path, e
            )

    def _record_module_path_telemetry(
        self, module_path_info: dict[str, Any], source_path: Path
    ) -> None:
        """
        Record telemetry metrics for module path derivation.

        Args:
            module_path_info: Result from ModulePathDeriver.derive_module_path
            source_path: Path to source file
        """
        try:
            from datetime import datetime

            validation_status = module_path_info.get("validation_status", "unknown")
            module_path = module_path_info.get("module_path", "")

            # Record success/failure metrics
            metrics = [
                MetricValue(
                    name="module_path_derived_total",
                    value=1,
                    unit="count",
                    labels={"file_path": str(source_path), "status": validation_status},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="module_path_derived_success",
                    value=1 if validation_status == "validated" else 0,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
            ]

            # Record validation status breakdown
            if validation_status in ["validated", "unvalidated", "failed", "error"]:
                metrics.append(
                    MetricValue(
                        name=f"module_path_status_{validation_status}",
                        value=1,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    )
                )

            # Record fallback usage
            fallback_paths = module_path_info.get("fallback_paths", [])
            if fallback_paths:
                metrics.append(
                    MetricValue(
                        name="module_path_fallback_used",
                        value=1,
                        unit="count",
                        labels={
                            "file_path": str(source_path),
                            "fallback_count": str(len(fallback_paths)),
                        },
                        timestamp=datetime.now(),
                    )
                )

            # Record module path pattern (for analytics)
            if module_path:
                has_src = "src" in module_path
                path_depth = len(module_path.split("."))
                metrics.extend(
                    [
                        MetricValue(
                            name="module_path_has_src",
                            value=1 if has_src else 0,
                            unit="count",
                            labels={"file_path": str(source_path)},
                            timestamp=datetime.now(),
                        ),
                        MetricValue(
                            name="module_path_depth",
                            value=path_depth,
                            unit="count",
                            labels={"file_path": str(source_path)},
                            timestamp=datetime.now(),
                        ),
                    ]
                )

            # Record all metrics
            if metrics:
                self._telemetry.record_metrics(metrics)

        except Exception as e:
            logger.warning(
                "Failed to record module path telemetry for %s: %s", source_path, e
            )

    def _record_validation_telemetry(self, issues: list, source_path: Path) -> None:
        """
        Record telemetry metrics for test validation.

        Args:
            issues: List of validation issues
            source_path: Path to source file
        """
        try:
            from datetime import datetime

            # Count issues by severity and category
            error_count = sum(1 for issue in issues if issue.severity == "error")
            warning_count = sum(1 for issue in issues if issue.severity == "warning")

            # Count by category
            category_counts = {}
            for issue in issues:
                category = issue.category
                category_counts[category] = category_counts.get(category, 0) + 1

            # Record metrics
            metrics = [
                MetricValue(
                    name="test_validation_total_issues",
                    value=len(issues),
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="test_validation_errors",
                    value=error_count,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="test_validation_warnings",
                    value=warning_count,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
            ]

            # Add category-specific metrics
            for category, count in category_counts.items():
                metrics.append(
                    MetricValue(
                        name=f"test_validation_{category}_issues",
                        value=count,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    )
                )

            # Record all metrics
            if metrics:
                self._telemetry.record_metrics(metrics)

        except Exception as e:
            logger.warning(
                "Failed to record validation telemetry for %s: %s", source_path, e
            )

    async def _write_test_file_immediate(
        self, generation_result: GenerationResult
    ) -> dict[str, Any]:
        """
        Write a single test file with immediate validation and formatting.

        Implements pre-validation, syntax checking, and rollback capability.

        Args:
            generation_result: The generation result to write

        Returns:
            Write result dictionary with success status and details
        """
        if not generation_result.success or not generation_result.content:
            return {
                "success": False,
                "error": "Invalid generation result",
                "file_path": generation_result.file_path,
            }

        try:
            # Pre-validate content before writing (syntax check happens in writer)
            write_result = self._writer.write_test_file(
                test_path=generation_result.file_path,
                test_content=generation_result.content,
                overwrite=True,  # Immediate mode allows overwrite
            )

            # Convert to consistent format
            return {
                "success": write_result.get("success", False),
                "file_path": generation_result.file_path,
                "bytes_written": write_result.get("bytes_written", 0),
                "formatted": write_result.get("formatted", False),
                "error": write_result.get("error")
                if not write_result.get("success")
                else None,
                "source_result": generation_result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": generation_result.file_path,
                "source_result": generation_result,
            }

    async def _rollback_failed_write(self, file_path: str) -> None:
        """
        Rollback a failed write by removing incomplete test files.

        Args:
            file_path: Path to the file to rollback
        """
        try:
            from pathlib import Path

            path = Path(file_path)
            if path.exists():
                # Only remove if it appears to be incomplete (small file)
                # This is a safety check to avoid removing valid existing files
                try:
                    content = path.read_text(encoding="utf-8")
                    if (
                        len(content.strip()) < 50
                    ):  # Very small files are likely incomplete
                        path.unlink()
                        logger.info("Rolled back incomplete test file: %s", file_path)
                except Exception:
                    # If we can't read it, it's likely corrupted, so remove it
                    path.unlink()
                    logger.info("Rolled back corrupted test file: %s", file_path)

        except Exception as e:
            logger.warning("Failed to rollback file %s: %s", file_path, e)

    async def _refine_test_file_immediate(self, test_file_path: str) -> dict[str, Any]:
        """
        Refine a single test file immediately using semaphore-controlled pytest refiner.

        Args:
            test_file_path: Path to the test file to refine

        Returns:
            Refinement result dictionary
        """
        try:
            from pathlib import Path

            test_file = Path(test_file_path)
            if not test_file.exists():
                return {
                    "test_file": test_file_path,
                    "success": False,
                    "error": "Test file does not exist",
                    "iterations": 0,
                }

            # Create context builder function with current test content
            async def build_context_for_refinement(
                test_file: Path, test_content: str
            ) -> dict[str, Any] | None:
                return self._context_assembler.context_for_refinement(
                    test_file, test_content
                )

            # Use the pytest refiner with proper concurrency control
            # (PytestRefiner will be enhanced with semaphore support)
            refinement_result = await self._pytest_refiner.refine_until_pass(
                test_file_path,
                self._config["max_refinement_iterations"],
                build_context_for_refinement,
            )

            return refinement_result

        except Exception as e:
            logger.warning("Immediate refinement failed for %s: %s", test_file_path, e)
            return {
                "test_file": test_file_path,
                "success": False,
                "error": str(e),
                "iterations": 0,
            }

    async def _write_test_files(
        self, generation_results: list[GenerationResult]
    ) -> list[dict[str, Any]]:
        """Write generated test files using configured writing strategy."""
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
                raise GenerateUseCaseError(
                    f"Test file writing failed: {e}", cause=e
                ) from e

    async def _execute_and_refine_tests(
        self, write_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute tests with pytest and refine failures using services."""
        if not self._config["enable_refinement"]:
            return []

        with self._telemetry.create_child_span("execute_and_refine_tests") as span:
            refinement_results = []

            try:
                successful_writes = [r for r in write_results if r["success"]]

                for write_result in successful_writes:
                    test_file_path = write_result["source_result"].file_path

                    try:
                        # Create async wrapper for context_for_refinement
                        async def build_context_for_refinement(
                            test_file: Path, test_content: str
                        ) -> dict[str, Any] | None:
                            return self._context_assembler.context_for_refinement(
                                test_file, test_content
                            )

                        # Use pytest refiner service
                        refinement_result = (
                            await self._pytest_refiner.refine_until_pass(
                                test_file_path,
                                self._config["max_refinement_iterations"],
                                build_context_for_refinement,
                            )
                        )
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
                ) from e

    async def _record_final_state(
        self,
        generation_results: list[GenerationResult],
        refinement_results: list[dict[str, Any]],
        coverage_delta: dict[str, Any],
    ) -> None:
        """Record final state information for future runs and analysis."""
        try:
            # Compile state data
            state_data = {
                "last_run_timestamp": asyncio.get_event_loop().time(),
                "generation_summary": {
                    "total_files_processed": len(generation_results),
                    "successful_generations": sum(
                        1
                        for r in generation_results
                        if (
                            (getattr(r, "success", False))
                            if not isinstance(r, dict)
                            else bool(r.get("success", False))
                        )
                    ),
                    "failed_generations": sum(
                        1
                        for r in generation_results
                        if not (
                            (getattr(r, "success", False))
                            if not isinstance(r, dict)
                            else bool(r.get("success", False))
                        )
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
        """Record telemetry metrics and cost information."""
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
                    value=sum(
                        1
                        for r in generation_results
                        if (
                            (getattr(r, "success", False))
                            if not isinstance(r, dict)
                            else bool(r.get("success", False))
                        )
                    ),
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
                sum(
                    1
                    for r in generation_results
                    if (
                        (getattr(r, "success", False))
                        if not isinstance(r, dict)
                        else bool(r.get("success", False))
                    )
                ),
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
            if hasattr(self, "_content_builder"):
                self._content_builder.clear_cache()
        except Exception:
            pass  # Ignore cleanup errors
