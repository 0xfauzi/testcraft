"""
Generate Use Case - Thin orchestrator.

This module implements the primary use case for generating tests as a thin
orchestrator that delegates to focused, testable services for improved
maintainability and separation of concerns.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .generation.config import GenerationConfig
from .generation.services.batch_executor import BatchExecutor
from .generation.services.content_builder import ContentBuilder
from .generation.services.context_assembler import ContextAssembler
from .generation.services.coverage_evaluator import CoverageEvaluator
from .generation.services.plan_builder import PlanBuilder
from .generation.services.pytest_refiner import PytestRefiner
from .generation.services.state_discovery import GenerateUseCaseError, StateSyncDiscovery
from .generation.services.usecase_pipeline import UseCasePipeline
from .generation.services.writing import WritingService, ImmediateModeProcessor
from .generation.services.telemetry import TelemetryService
from .generation.services.state_reporting import StateReportingService
from .generation.services.planning_bridge import PlanningBridge
from ..adapters.io.file_discovery import FileDiscoveryService
from ..ports.context_port import ContextPort
from ..ports.coverage_port import CoveragePort
from ..ports.llm_port import LLMPort
from ..ports.parser_port import ParserPort
from ..ports.refine_port import RefinePort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import SpanKind, TelemetryPort
from ..ports.writer_port import WriterPort

logger = logging.getLogger(__name__)

# Re-export GenerateUseCaseError to maintain import compatibility
__all__ = ["GenerateUseCase", "GenerateUseCaseError"]


class GenerateUseCase:
    """
    Core use case for test generation - Thin orchestrator.
    
    Delegates to focused services for improved maintainability and separation of concerns:
    - UseCasePipeline: Main pipeline orchestration
    - WritingService: File writing and persistence
    - TelemetryService: Metrics and span management
    - StateReportingService: State persistence and reporting
    - PlanningBridge: Planning session integration
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
        file_discovery_service: FileDiscoveryService,
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
            file_discovery_service: Service for file discovery
            config: Optional configuration overrides
        """
        # Store ports for service initialization
        self._llm = llm_port
        self._writer = writer_port
        self._coverage = coverage_port
        self._refine = refine_port
        self._context = context_port
        self._parser = parser_port
        self._state = state_port
        self._telemetry = telemetry_port
        self._file_discovery = file_discovery_service
        
        # Merge and validate configuration
        self._config = GenerationConfig.merge_config(config)
        GenerationConfig.validate_config(self._config)
        
        # Initialize thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=self._config["batch_size"])
        
        # Initialize core services
        self._initialize_services()
        
        # Status tracker will be injected per generation operation
        self._current_status_tracker = None
    
    def _initialize_services(self):
        """Initialize all service dependencies."""
        # Core discovery and planning services
        self._state_discovery = StateSyncDiscovery(
            state_port=self._state,
            file_discovery_service=self._file_discovery,
            telemetry_port=self._telemetry,
        )
        
        self._coverage_evaluator = CoverageEvaluator(
            coverage_port=self._coverage,
            telemetry_port=self._telemetry,
        )
        
        coverage_probe = self._config.get("coverage_probe")
        self._plan_builder = PlanBuilder(
            parser_port=self._parser,
            file_discovery_service=self._file_discovery,
            telemetry_port=self._telemetry,
            coverage_threshold=self._config["coverage_threshold"],
            coverage_probe=coverage_probe,
            always_analyze_new_files=bool(self._config.get("always_analyze_new_files", False)),
        )
        
        self._content_builder = ContentBuilder(parser_port=self._parser)
        
        self._context_assembler = ContextAssembler(
            context_port=self._context,
            parser_port=self._parser,
            config=self._config,
        )
        
        self._batch_executor = BatchExecutor(
            telemetry_port=self._telemetry,
            executor=self._executor,
        )
        
        self._pytest_refiner = PytestRefiner(
            refine_port=self._refine,
            telemetry_port=self._telemetry,
            executor=self._executor,
            max_concurrent_refines=self._config["max_refine_workers"],
            backoff_sec=self._config["refinement_backoff_sec"],
            writer_port=self._writer,
        )
        
        # New modular services
        self._writing_service = WritingService(
            writer_port=self._writer,
            telemetry_port=self._telemetry,
            config=self._config,
        )
        
        self._telemetry_service = TelemetryService(
            telemetry_port=self._telemetry,
            config=self._config,
        )
        
        self._state_reporting_service = StateReportingService(
            state_port=self._state,
            executor=self._executor,
            config=self._config,
        )
        
        self._planning_bridge = PlanningBridge(state_port=self._state)
        
        # Main pipeline orchestrator
        self._pipeline = UseCasePipeline(
            config=self._config,
            llm_port=self._llm,
            telemetry_port=self._telemetry,
            state_discovery=self._state_discovery,
            coverage_evaluator=self._coverage_evaluator,
            plan_builder=self._plan_builder,
            content_builder=self._content_builder,
            context_assembler=self._context_assembler,
            batch_executor=self._batch_executor,
            pytest_refiner=self._pytest_refiner,
            writing_service=self._writing_service,
        )
        
        # Immediate mode processor
        self._immediate_processor = ImmediateModeProcessor(
            writing_service=self._writing_service,
            telemetry_port=self._telemetry,
            config=self._config,
        )
    
    def set_status_tracker(self, status_tracker) -> None:
        """Set the status tracker for live updates during generation."""
        self._current_status_tracker = status_tracker
        
        # Propagate to all services that need status tracking
        self._pipeline.set_status_tracker(status_tracker)
        self._immediate_processor.set_status_tracker(status_tracker)
    
    async def generate_tests(
        self,
        project_path: str | Path,
        target_files: list[str | Path] | None = None,
        from_planning_session_id: str | None = None,
        selected_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Main entry point for test generation.
        
        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to process
            from_planning_session_id: Optional planning session ID for element filtering
            selected_only: If True, only generate tests for selected elements
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
                
                # Delegate to pipeline orchestrator
                results = await self._pipeline.execute_pipeline(
                    project_path=project_path,
                    target_files=target_files,
                    from_planning_session_id=from_planning_session_id,
                    selected_only=selected_only,
                    **kwargs
                )
                
                # Record final state and telemetry
                await self._finalize_generation(
                    span,
                    results.get("generation_results", []),
                    results.get("refinement_results", []),
                    results.get("coverage_delta", {}),
                )
                
                logger.info(
                    "Test generation completed successfully. Files: %d, Tests generated: %d, Coverage delta: %.2f%%",
                    results.get("files_processed", 0),
                    results.get("tests_generated", 0),
                    results.get("coverage_delta", {}).get("line_coverage_delta", 0) * 100,
                )
                
                return results
                
            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Test generation failed: %s", e)
                raise GenerateUseCaseError(f"Test generation failed: {e}", cause=e)
    
    async def _finalize_generation(
        self,
        span,
        generation_results: list,
        refinement_results: list,
        coverage_delta: dict[str, Any],
    ) -> None:
        """Finalize generation by recording state and telemetry."""
        # Record final state
        await self._state_reporting_service.record_final_state(
            generation_results, refinement_results, coverage_delta
        )
        
        # Record telemetry and costs
        await self._telemetry_service.record_final_telemetry_and_costs(
            span, generation_results, refinement_results
        )
    
    def __del__(self):
        """Clean up resources."""
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False)
            if hasattr(self, "_content_builder"):
                self._content_builder.clear_cache()
        except Exception:
            pass  # Ignore cleanup errors
