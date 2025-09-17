"""
Pipeline orchestration logic for GenerateUseCase.

This module contains the core pipeline logic extracted from GenerateUseCase,
handling the main test generation workflow coordination.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..config import GenerationConfig
from .batch_executor import BatchExecutor
from .content_builder import ContentBuilder
from .context_assembler import ContextAssembler
from .coverage_evaluator import CoverageEvaluator
from .plan_builder import PlanBuilder
from .pytest_refiner import PytestRefiner
from .state_discovery import StateSyncDiscovery
from .structure import ModulePathDeriver
from .generator_guardrails import TestContentValidator
from .writing import WritingService
from ....domain.models import GenerationResult, PlanningSession
from ....ports.llm_port import LLMPort
from ....ports.telemetry_port import SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class UseCasePipeline:
    """
    Core pipeline orchestration for test generation workflow.
    
    Handles the main sequence: discovery → coverage → planning → generation → writing → refinement
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        llm_port: LLMPort,
        telemetry_port: TelemetryPort,
        state_discovery: StateSyncDiscovery,
        coverage_evaluator: CoverageEvaluator,
        plan_builder: PlanBuilder,
        content_builder: ContentBuilder,
        context_assembler: ContextAssembler,
        batch_executor: BatchExecutor,
        pytest_refiner: PytestRefiner,
        writing_service: WritingService,
    ):
        """Initialize pipeline with all required services."""
        self._config = config
        self._llm = llm_port
        self._telemetry = telemetry_port
        self._state_discovery = state_discovery
        self._coverage_evaluator = coverage_evaluator
        self._plan_builder = plan_builder
        self._content_builder = content_builder
        self._context_assembler = context_assembler
        self._batch_executor = batch_executor
        self._pytest_refiner = pytest_refiner
        self._writing_service = writing_service
        
        # Status tracker will be injected per operation
        self._current_status_tracker = None
    
    def set_status_tracker(self, status_tracker) -> None:
        """Set the status tracker for live updates during generation."""
        self._current_status_tracker = status_tracker
        
        # Update dependent services with the status tracker
        if hasattr(self._batch_executor, '_status_tracker'):
            self._batch_executor._status_tracker = status_tracker
        if hasattr(self._pytest_refiner, '_status_tracker'):
            self._pytest_refiner._status_tracker = status_tracker
    
    async def execute_pipeline(
        self,
        project_path: Path,
        target_files: list[str | Path] | None = None,
        from_planning_session_id: str | None = None,
        selected_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute the main test generation pipeline.
        
        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to process
            from_planning_session_id: Optional planning session ID for element filtering
            selected_only: If True, only generate tests for selected elements
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing pipeline results and metadata
        """
        with self._telemetry.create_span(
            "execute_pipeline",
            kind=SpanKind.INTERNAL,
            attributes={
                "project_path": str(project_path),
                "target_files_count": len(target_files) if target_files else 0,
                "config": self._config,
            },
        ) as span:
            # Step 1: Sync state and discover files
            discovery_result = self._state_discovery.sync_and_discover(
                project_path, target_files
            )
            span.set_attribute("files_discovered", len(discovery_result["files"]))
            
            # Step 2: Set up plan builder context
            test_files = await self._setup_plan_builder_context(project_path)
            
            # Step 3: Measure initial coverage
            initial_coverage = self._coverage_evaluator.measure_initial(
                discovery_result["files"]
            )
            span.set_attribute(
                "initial_coverage", initial_coverage.get("overall_line_coverage", 0)
            )
            
            # Step 4: Decide which files to process and build plans
            files_to_process = self._plan_builder.decide_files_to_process(
                discovery_result["files"], initial_coverage
            )
            span.set_attribute("files_to_process", len(files_to_process))
            
            # Step 5: Load planning session and build generation plans
            planning_session, generation_plans = await self._build_generation_plans(
                files_to_process, from_planning_session_id, selected_only, span
            )
            
            # Check if we have any valid plans to work with
            if not generation_plans:
                return self._create_empty_result(files_to_process)
            
            # Step 6: Build directory tree and gather context if enabled
            context_data = await self._gather_project_context(project_path, files_to_process)
            
            # Step 7: Execute generation workflow (immediate or batch mode)
            generation_results, write_results, refinement_results = await self._execute_generation_workflow(
                generation_plans, context_data, planning_session, span
            )
            
            # Step 8: Measure coverage delta
            final_coverage, coverage_delta = await self._measure_coverage_delta(
                discovery_result["files"], initial_coverage, span
            )
            
            # Step 9: Compile and return results
            return self._compile_pipeline_results(
                discovery_result,
                files_to_process,
                generation_results,
                write_results,
                refinement_results,
                initial_coverage,
                final_coverage,
                coverage_delta,
                project_path,
            )
    
    async def _setup_plan_builder_context(self, project_path: Path) -> list:
        """Set up plan builder context with discovered test files."""
        try:
            from ....adapters.io.file_discovery import FileDiscoveryService
            # This is a simplified approach - in practice, we'd inject the service
            file_discovery = FileDiscoveryService()
            test_files = file_discovery.discover_test_files(project_path)
            self._plan_builder.set_project_context(project_path, test_files)
            return test_files
        except Exception:
            test_files = []
            self._plan_builder.set_project_context(project_path, [])
            return test_files
    
    async def _build_generation_plans(
        self,
        files_to_process: list,
        from_planning_session_id: str | None,
        selected_only: bool,
        span,
    ) -> tuple[Any, list]:
        """Load planning session data and build generation plans."""
        planning_session = None
        selected_element_keys = []
        
        if from_planning_session_id and selected_only:
            planning_session, selected_element_keys = await self._load_planning_session(
                from_planning_session_id
            )
        
        # Build plans (filtered by planning session if available)
        if selected_element_keys:
            generation_plans = self._plan_builder.build_plans_for_elements(
                files_to_process, selected_element_keys
            )
            span.set_attribute("planning_filtered", True)
            span.set_attribute("selected_element_count", len(selected_element_keys))
        else:
            generation_plans = self._plan_builder.build_plans(files_to_process)
            span.set_attribute("planning_filtered", False)
        
        span.set_attribute("generation_plans_created", len(generation_plans))
        return planning_session, generation_plans
    
    async def _load_planning_session(self, session_id: str) -> tuple[Any, list]:
        """Load planning session from state."""
        try:
            # Access state through state_discovery service
            planning_data = self._state_discovery._state.get_state("last_planning_session")
            if planning_data and planning_data.get("session_id") == session_id:
                planning_session = PlanningSession(**planning_data)
                selected_element_keys = planning_session.selected_keys
                logger.info(f"Using planning session {session_id} with {len(selected_element_keys)} selected elements")
                return planning_session, selected_element_keys
            else:
                logger.warning(f"Planning session {session_id} not found or invalid")
        except Exception as e:
            logger.warning(f"Failed to load planning session {session_id}: {e}")
        
        return None, []
    
    async def _gather_project_context(self, project_path: Path, files_to_process: list) -> dict[str, Any]:
        """Build directory tree and gather context if enabled."""
        context_data = {}
        if self._config["enable_context"]:
            context_data = self._context_assembler.gather_project_context(
                project_path, files_to_process
            )
        return context_data
    
    async def _execute_generation_workflow(
        self,
        generation_plans: list,
        context_data: dict[str, Any],
        planning_session: Any,
        span,
    ) -> tuple[list, list, list]:
        """Execute the generation workflow based on configuration mode."""
        if self._config["immediate_refinement"]:
            # Immediate per-file pipeline: generate → write → refine for each plan
            generation_results, write_results, refinement_results = await self._process_plans_immediately(
                generation_plans, context_data, planning_session
            )
            span.set_attribute("immediate_mode", True)
        else:
            # Legacy mode: batch all operations
            generation_results = await self._batch_executor.run_in_batches(
                generation_plans,
                self._config["batch_size"],
                lambda plan: self._generate_tests_for_plan(plan, context_data, planning_session),
            )
            
            # Write and refine in separate phases
            write_results = await self._writing_service.write_test_files_batch(generation_results)
            
            # Note: Batch mode refinement would require coordination across multiple files
            # For now, batch mode focuses on generation and writing; use immediate mode for refinement
            refinement_results = [
                {"success": True, "message": "Batch mode refinement not yet implemented", "final_status": "skipped"}
                for _ in write_results
            ]
            span.set_attribute("immediate_mode", False)
        
        span.set_attribute("tests_generated", len(generation_results))
        span.set_attribute("files_written", len(write_results))
        span.set_attribute("files_refined", len(refinement_results))
        
        return generation_results, write_results, refinement_results
    
    async def _measure_coverage_delta(
        self, files: list, initial_coverage: dict, span
    ) -> tuple[dict, dict]:
        """Measure final coverage and calculate delta."""
        final_coverage = self._coverage_evaluator.measure_final(files)
        coverage_delta = self._coverage_evaluator.calculate_delta(
            initial_coverage, final_coverage
        )
        
        span.set_attribute(
            "final_coverage", final_coverage.get("overall_line_coverage", 0)
        )
        span.set_attribute(
            "coverage_delta", coverage_delta.get("line_coverage_delta", 0)
        )
        
        return final_coverage, coverage_delta
    
    def _create_empty_result(self, files_to_process: list) -> dict[str, Any]:
        """Create result for when no testable elements are found."""
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
    
    def _compile_pipeline_results(
        self,
        discovery_result: dict,
        files_to_process: list,
        generation_results: list,
        write_results: list,
        refinement_results: list,
        initial_coverage: dict,
        final_coverage: dict,
        coverage_delta: dict,
        project_path: Path,
    ) -> dict[str, Any]:
        """Compile final pipeline results."""
        return {
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
    
    async def _generate_tests_for_plan(
        self, plan, context_data: dict[str, Any], planning_session: Any = None
    ) -> GenerationResult:
        """
        Generate tests for a single test generation plan.
        
        This method will be moved to a separate module in the next step.
        """
        try:
            # Get source path for this plan
            source_path = self._plan_builder.get_source_path_for_plan(plan)
            
            # Build code content from plan elements
            code_content = self._content_builder.build_code_content(plan, source_path)
            
            # Get relevant context for this file
            relevant_context = self._context_assembler.context_for_generation(
                plan, source_path
            )
            
            # Derive module path information
            module_path_info = await self._derive_module_path_info(source_path, context_data)
            
            # Enhance context with module path and planning information
            enhanced_context = await self._enhance_generation_context(
                relevant_context, module_path_info, planning_session, plan, source_path
            )
            
            # Call LLM to generate tests
            llm_result = self._llm.generate_tests(
                code_content=code_content,
                context=enhanced_context,
                test_framework=self._config["test_framework"],
                include_docstrings=self._config.get("include_docstrings", True),
                generate_fixtures=self._config.get("generate_fixtures", True),
                parametrize_similar_tests=self._config.get("parametrize_similar_tests", True),
                max_test_methods_per_class=self._config.get("max_test_methods_per_class", 20),
            )
            
            # Extract and validate the generated content
            raw_tests = llm_result.get("tests", "")
            test_content = self._normalize_test_content(raw_tests)
            if not test_content.strip():
                return GenerationResult(
                    file_path="unknown",
                    content=None,
                    success=False,
                    error_message="LLM returned empty test content",
                )
            
            # Validate and potentially fix the generated test content
            test_content = await self._validate_and_fix_content(test_content, source_path)
            
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

    def _normalize_test_content(self, tests_field: Any) -> str:
        """Coerce LLM 'tests' field into a string for downstream processing.

        Handles common shapes:
        - str: returned as-is
        - list[str|dict|Any]: joins items with double newlines; dicts prefer code-like keys
        - dict: tries common keys ('tests', 'code', 'content', 'text') then str()
        - other: str() fallback
        """
        try:
            if tests_field is None:
                return ""
            if isinstance(tests_field, str):
                return tests_field
            if isinstance(tests_field, list):
                parts: list[str] = []
                for item in tests_field:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        for key in ("code", "content", "tests", "text"):
                            if key in item:
                                val = item.get(key)
                                parts.append(val if isinstance(val, str) else str(val))
                                break
                        else:
                            parts.append(str(item))
                    else:
                        parts.append(str(item))
                return "\n\n".join(parts)
            if isinstance(tests_field, dict):
                for key in ("tests", "code", "content", "text"):
                    if key in tests_field:
                        nested = tests_field.get(key)
                        # Recurse to handle nested list/dict
                        return self._normalize_test_content(nested)
                return str(tests_field)
            # Fallback for any other type
            return str(tests_field)
        except Exception:
            # On any unexpected error, use safe string fallback
            return str(tests_field) if tests_field is not None else ""
    
    async def _derive_module_path_info(self, source_path: Path, context_data: dict[str, Any]) -> dict[str, Any]:
        """Derive authoritative module path and import suggestions."""
        module_path_info = {}
        if source_path:
            try:
                # Get project root from context_data or detect automatically
                project_root = None
                if context_data.get("project_structure"):
                    project_structure = context_data["project_structure"]
                    if isinstance(project_structure, dict) and project_structure.get("name"):
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
                        module_path_info["validation_status"]
                    )
                    
            except Exception as e:
                logger.warning("Failed to derive module path for %s: %s", source_path, e)
                module_path_info = {}
        
        return module_path_info
    
    async def _enhance_generation_context(
        self,
        relevant_context: str,
        module_path_info: dict[str, Any],
        planning_session: Any,
        plan,
        source_path: Path,
    ) -> str:
        """Enhance context with module path information and planning details."""
        enhanced_context = relevant_context
        
        # Add module path information
        if module_path_info and (module_path_info.get("module_path") or module_path_info.get("import_suggestion")):
            module_context = []
            if module_path_info.get("module_path"):
                module_context.append(f"Module Path: {module_path_info['module_path']}")
            if module_path_info.get("import_suggestion"):
                module_context.append(f"Import Hint: {module_path_info['import_suggestion']}")
            if module_path_info.get("validation_status"):
                module_context.append(f"Import Status: {module_path_info['validation_status']}")
            
            module_info_text = "\n".join(module_context)
            
            if enhanced_context:
                enhanced_context = f"{enhanced_context}\n\n# Module Import Information\n{module_info_text}"
            else:
                enhanced_context = f"# Module Import Information\n{module_info_text}"
        
        # Inject detailed planning information if available
        if planning_session and hasattr(planning_session, 'items'):
            enhanced_context = await self._inject_planning_context(
                enhanced_context, planning_session, plan, source_path
            )
        
        return enhanced_context
    
    async def _inject_planning_context(
        self, enhanced_context: str, planning_session: Any, plan, source_path: Path
    ) -> str:
        """Inject detailed planning information into generation context."""
        element_plans = []
        for element in plan.elements_to_test:
            for plan_item in planning_session.items:
                if (element.name == plan_item.element.name and 
                    element.type == plan_item.element.type):
                    element_plans.append(plan_item)
                    break
        
        if element_plans:
            planning_context = ["# DETAILED_TEST_PLANS"]
            planning_context.append("The following detailed test plans should guide your implementation:")
            planning_context.append("")
            
            for i, plan_item in enumerate(element_plans, 1):
                element_type = plan_item.element.type.value if hasattr(plan_item.element.type, 'value') else str(plan_item.element.type)
                planning_context.append(f"## Plan {i}: {plan_item.element.name} ({element_type})")
                planning_context.append(f"**Summary:** {plan_item.plan_summary}")
                planning_context.append(f"**Detailed Plan:**\n{plan_item.detailed_plan}")
                if plan_item.confidence:
                    planning_context.append(f"**Confidence:** {plan_item.confidence:.2f}")
                if plan_item.tags:
                    planning_context.append(f"**Tags:** {', '.join(plan_item.tags)}")
                planning_context.append("")
            
            planning_context.append("IMPORTANT: Follow these detailed plans closely. Implement ONLY the listed elements and scenarios.")
            planning_info_text = "\n".join(planning_context)
            
            if enhanced_context:
                enhanced_context = f"{enhanced_context}\n\n{planning_info_text}"
            else:
                enhanced_context = planning_info_text
            
            logger.info(f"Injected {len(element_plans)} detailed plans into generation context for {source_path}")
        
        return enhanced_context
    
    async def _validate_and_fix_content(self, test_content: str, source_path: Path) -> str:
        """Validate and potentially fix the generated test content."""
        enriched_context_data = self._context_assembler.get_last_enriched_context()
        if enriched_context_data and self._config.get("enable_validation", True):
            try:
                validated_content, is_valid, issues = TestContentValidator.validate_and_fix(
                    test_content, enriched_context_data
                )
                
                if issues:
                    issue_summary = "; ".join(str(issue) for issue in issues[:3])
                    logger.info("Test validation issues for %s: %s", source_path, issue_summary)
                
                # Use validated content if fixes were applied
                if validated_content != test_content:
                    logger.debug("Applied validation fixes to test for %s", source_path)
                    test_content = validated_content
                
                if not is_valid:
                    error_count = sum(1 for issue in issues if issue.severity == "error")
                    logger.warning("Test validation failed for %s with %d errors", source_path, error_count)
                    
            except Exception as e:
                logger.warning("Test validation failed with exception for %s: %s", source_path, e)
        
        return test_content
    
    async def _process_plans_immediately(
        self, generation_plans, context_data: dict[str, Any], planning_session: Any = None
    ) -> tuple[list[GenerationResult], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Process plans with immediate per-file refinement.
        
        This method will be moved to a separate module in the next step.
        """
        immediate_results = []
        
        # Process in batches manually to maintain some parallelism
        batch_size = self._config["batch_size"]
        for i in range(0, len(generation_plans), batch_size):
            batch = generation_plans[i : i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self._process_plan_immediate(plan, context_data, planning_session) 
                for plan in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning("Plan processing failed: %s", result)
                    immediate_results.append({
                        "generation_result": GenerationResult(
                            file_path=f"plan_{i + j}",
                            content=None,
                            success=False,
                            error_message=str(result),
                        ),
                        "write_result": {"success": False, "error": str(result)},
                        "refinement_result": {"success": False, "error": str(result)},
                        "success": False,
                        "errors": [str(result)],
                    })
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
        self, plan, context_data: dict[str, Any], planning_session: Any = None
    ) -> dict[str, Any]:
        """
        Process a single plan with immediate write-and-refine workflow.
        
        Implements: generate → write → refine (if enabled) for a single plan.
        """
        # Step 1: Generate tests
        generation_result = await self._generate_tests_for_plan(plan, context_data, planning_session)
        
        if not generation_result.success or not generation_result.content:
            return {
                "plan": plan,
                "generation_result": generation_result,
                "write_result": {"success": False, "error": "Generation failed"},
                "refinement_result": {"success": False, "error": "Skipped due to generation failure"},
                "success": False,
                "errors": [generation_result.error_message or "Generation failed"],
            }
        
        # Step 2: Write tests immediately
        write_result = await self._writing_service.write_test_file_immediate(generation_result)
        
        # Step 3: Run pytest and refine if enabled
        refinement_result = {"success": True, "message": "No refinement needed"}
        
        if self._config.get("enable_refinement", False) and write_result.get("success", False):
            try:
                test_file_path = str(generation_result.file_path)
                
                # Create source context builder function for refinement
                async def build_source_context_fn(test_file: Path, test_content: str) -> dict[str, Any] | None:
                    """Build source context for refinement."""
                    try:
                        # Get source file path from the plan
                        source_path = plan.file_path if hasattr(plan, 'file_path') else None
                        
                        context = {
                            "test_path": str(test_file),
                            "test_content": test_content,
                        }
                        
                        if source_path:
                            context["source_path"] = str(source_path)
                            # Try to get source content for better context
                            try:
                                source_file = Path(source_path)
                                if source_file.exists():
                                    context["source_content"] = source_file.read_text(encoding='utf-8')
                            except Exception as e:
                                logger.debug("Failed to read source file %s: %s", source_path, e)
                        
                        if context_data:
                            context["project_context"] = context_data.get("project_structure", {})
                        
                        return context
                    except Exception as e:
                        logger.warning("Failed to build source context: %s", e)
                        return None
                
                # Use the high-level refine_until_pass method for complete workflow
                refine_result = await self._pytest_refiner.refine_until_pass(
                    test_path=test_file_path,
                    max_iterations=self._config.get("max_refinement_iterations", 3),
                    build_source_context_fn=build_source_context_fn
                )
                
                refinement_result = {
                    "success": refine_result.get("success", False),
                    "iterations": refine_result.get("iterations_used", 0),
                    "message": refine_result.get("rationale", "Refinement completed"),
                    "final_status": refine_result.get("final_status", "unknown"),
                    "pytest_status": refine_result.get("pytest_status", "unknown")
                }
                
            except Exception as e:
                logger.warning("Refinement error for %s: %s", generation_result.file_path, e)
                refinement_result = {
                    "success": False,
                    "error": str(e),
                    "message": f"Refinement failed with exception: {e}",
                    "final_status": "exception"
                }
        
        # Determine overall success - refinement failure doesn't block success if disabled
        refine_enabled = self._config.get("enable_refinement", False)
        overall_success = (
            generation_result.success 
            and write_result.get("success", False)
            and (not refine_enabled or refinement_result.get("success", True))
        )
        
        errors = []
        if not generation_result.success:
            errors.append(generation_result.error_message or "Generation failed")
        if not write_result.get("success", False):
            errors.append(write_result.get("error", "Write failed"))
        if refine_enabled and not refinement_result.get("success", True):
            errors.append(refinement_result.get("error", "Refinement failed"))
        
        return {
            "plan": plan,
            "generation_result": generation_result,
            "write_result": write_result,
            "refinement_result": refinement_result,
            "success": overall_success,
            "errors": errors,
        }
