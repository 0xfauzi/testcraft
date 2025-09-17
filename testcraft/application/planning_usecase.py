"""
Planning Use Case - Test plan generation orchestrator.

This module implements the use case for generating test plans for eligible elements,
presenting them to users for selection, and persisting the results for later
generation use.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .generation.services.batch_executor import BatchExecutor
from .generation.services.content_builder import ContentBuilder
from .generation.services.context_assembler import ContextAssembler
from .generation.services.plan_builder import PlanBuilder
from ..adapters.io.file_discovery import FileDiscoveryService
from ..adapters.parsing.test_mapper import TestMapper
from ..domain.models import (
    PlannableElementKey,
    PlanningSession,
    TestElementPlan,
)
from ..ports.context_port import ContextPort
from ..ports.llm_port import LLMPort
from ..ports.parser_port import ParserPort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import MetricValue, SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class PlanningUseCaseError(Exception):
    """Planning use case specific errors."""
    pass


class PlanningUseCase:
    """
    Core use case for test planning - generates plans for eligible test elements.
    
    This orchestrator coordinates:
    - FileDiscoveryService: Identifying files that need attention
    - PlanBuilder: Determining which elements need testing
    - TestMapper: Computing eligibility reasons for elements
    - ContextAssembler: Building context for LLM planning
    - ContentBuilder: Extracting source code content
    - BatchExecutor: Concurrent LLM plan generation
    - StatePort: Persisting planning sessions
    - TelemetryPort: Tracking planning metrics
    """
    
    def __init__(
        self,
        llm_port: LLMPort,
        parser_port: ParserPort,
        context_port: ContextPort,
        state_port: StatePort,
        telemetry_port: TelemetryPort,
        file_discovery_service: FileDiscoveryService,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Planning Use Case orchestrator.
        
        Args:
            llm_port: Port for LLM planning operations
            parser_port: Port for code parsing
            context_port: Port for context management
            state_port: Port for state persistence
            telemetry_port: Port for telemetry and metrics
            file_discovery_service: Service for file discovery
            config: Optional configuration overrides
        """
        self._llm = llm_port
        self._parser = parser_port
        self._context = context_port
        self._state = state_port
        self._telemetry = telemetry_port
        self._file_discovery = file_discovery_service
        self._config = config or {}
        
        # Initialize services
        self._plan_builder = PlanBuilder(
            parser_port=parser_port,
            file_discovery_service=file_discovery_service,
            telemetry_port=telemetry_port,
            coverage_threshold=self._config.get("coverage_threshold", 0.8)
        )
        self._test_mapper = TestMapper()
        self._context_assembler = ContextAssembler(
            context_port=context_port,
            parser_port=parser_port,
            config=self._config
        )
        self._content_builder = ContentBuilder(parser_port=parser_port)
        self._batch_executor = BatchExecutor(
            telemetry_port=telemetry_port
        )
        
        # Planning configuration
        self._max_elements_per_batch = config.get("planning_max_elements_per_batch", 8)
        self._plan_prompt_version = config.get("planning_prompt_version", "v1")

    def generate_planning_session(
        self,
        project_path: str,
        target_files: list[str] | None = None,
    ) -> PlanningSession:
        """
        Generate a comprehensive planning session for the project.
        
        Args:
            project_path: Path to the project root
            target_files: Optional list of specific files to plan for
            
        Returns:
            Complete planning session with element plans and metadata
            
        Raises:
            PlanningUseCaseError: If planning generation fails
        """
        session_id = str(uuid.uuid4())
        created_at = time.time()
        
        with self._telemetry.create_span(
            "planning_session", 
            kind=SpanKind.INTERNAL,
            attributes={
                "session_id": session_id,
                "project_path": project_path,
                "target_files_count": len(target_files) if target_files else 0,
            }
        ) as span:
            try:
                # Step 1: Discover files needing attention
                files_to_process = self._discover_files(project_path, target_files)
                span.set_attribute("files_discovered", len(files_to_process))
                
                # Step 2: Extract eligible test elements with reasons
                eligible_elements = self._extract_eligible_elements(files_to_process)
                span.set_attribute("eligible_elements", len(eligible_elements))
                
                if not eligible_elements:
                    logger.info("No eligible test elements found for planning")
                    return PlanningSession(
                        session_id=session_id,
                        project_path=project_path,
                        created_at=created_at,
                        items=[],
                        selected_keys=[],
                        stats={
                            "files_processed": len(files_to_process),
                            "eligible_elements": 0,
                            "planning_cost": 0.0,
                        }
                    )
                
                # Step 3: Generate plans for eligible elements
                element_plans = self._generate_element_plans(eligible_elements, project_path)
                span.set_attribute("plans_generated", len(element_plans))
                
                # Step 4: Create session with statistics
                stats = {
                    "files_processed": len(files_to_process),
                    "eligible_elements": len(eligible_elements),
                    "plans_generated": len(element_plans),
                    "planning_cost": sum(
                        plan.get("cost", 0.0) for plan in [
                            getattr(plan, "_metadata", {}) for plan in element_plans
                        ]
                    ),
                    "generation_time": time.time() - created_at,
                }
                
                session = PlanningSession(
                    session_id=session_id,
                    project_path=project_path,
                    created_at=created_at,
                    items=element_plans,
                    selected_keys=[],  # Empty until user makes selections
                    stats=stats
                )
                
                # Step 5: Persist session
                self._persist_session(session)
                
                # Record metrics
                self._telemetry.record_metric(
                    MetricValue(
                        name="planning_elements_total",
                        value=len(element_plans),
                        unit="count"
                    )
                )
                
                logger.info(
                    f"Generated planning session {session_id} with {len(element_plans)} element plans"
                )
                
                return session
                
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Planning session generation failed: {e}")
                raise PlanningUseCaseError(f"Failed to generate planning session: {e}") from e

    def _discover_files(self, project_path: str, target_files: list[str] | None) -> list[str]:
        """Discover files that need test planning attention."""
        if target_files:
            # Use explicitly provided files
            return [str(Path(project_path) / f) for f in target_files]
        
        # Discover all Python files in the project
        try:
            discovered = self._file_discovery.discover_source_files(Path(project_path))
            return [str(f) for f in discovered]
        except Exception as e:
            logger.warning(f"File discovery failed: {e}")
            return []

    def _extract_eligible_elements(self, files_to_process: list[str]) -> list[tuple[str, Any, str]]:
        """
        Extract elements eligible for testing with eligibility reasons.
        
        Returns:
            List of (file_path, element, eligibility_reason) tuples
        """
        eligible_elements = []
        
        for file_path in files_to_process:
            try:
                # Create a generation plan to get elements
                plan = self._plan_builder._create_generation_plan_for_file(Path(file_path))
                
                # Handle case where plan creation fails
                if plan is None or not plan.elements_to_test:
                    continue
                
                # For planning, simply assume all elements are eligible
                # We can enhance eligibility logic later when TestMapper interface is stable
                for element in plan.elements_to_test:
                    element_key = PlannableElementKey.from_element(file_path, element)
                    
                    # Simplified eligibility logic for now
                    # Check if test files exist for this source file
                    has_test_files = len(plan.existing_tests) > 0
                    
                    if not has_test_files:
                        reason = "no_existing_tests"
                    else:
                        # For now, assume existing tests might be incomplete
                        reason = "needs_additional_coverage"
                    
                    eligible_elements.append((file_path, element, reason))
                    
            except Exception as e:
                logger.warning(f"Failed to extract elements from {file_path}: {e}")
                continue
        
        return eligible_elements

    def _generate_element_plans(
        self, 
        eligible_elements: list[tuple[str, Any, str]], 
        project_path: str
    ) -> list[TestElementPlan]:
        """Generate LLM plans for eligible elements using a single call per file.

        This replaces the prior per-element planning with per-file planning by
        grouping elements by their source file and requesting plans for all
        elements in that file together. This reduces LLM round-trips and keeps
        plans coherent across elements within the same module.
        """
        from collections import defaultdict

        plans: list[TestElementPlan] = []

        # Group all eligible elements by their source file
        by_file: dict[str, list[tuple[Any, str]]] = defaultdict(list)
        for file_path, element, eligibility_reason in eligible_elements:
            by_file[file_path].append((element, eligibility_reason))

        # Generate plans with a single LLM request per file
        for file_path, items in by_file.items():
            try:
                file_plans = self._generate_file_plans(file_path, items, project_path)
                plans.extend(file_plans)
            except Exception as e:
                logger.warning(
                    f"Failed to generate file-level plans for {file_path}: {e}. Falling back to per-element stubs."
                )
                # Create safe fallback plans for each element on failure
                for element, eligibility_reason in items:
                    plans.append(
                        TestElementPlan(
                            element=element,
                            eligibility_reason=eligibility_reason,
                            plan_summary=f"Plan generation failed for {element.name} due to error",
                            detailed_plan=f"Unable to generate detailed plan due to error: {str(e)}. Manual planning recommended.",
                            confidence=0.0,
                            tags=["error", f"source_file:{file_path}"],
                        )
                    )

        return plans

    def _generate_batch_plans(
        self, 
        element_batch: list[tuple[str, Any, str]], 
        project_path: str
    ) -> list[TestElementPlan]:
        """Generate plans for a batch of elements."""
        plans = []
        
        for file_path, element, eligibility_reason in element_batch:
            try:
                # Create a temporary plan to extract content using existing service
                from ..domain.models import TestGenerationPlan
                temp_plan = TestGenerationPlan(
                    elements_to_test=[element],
                    existing_tests=[],
                    coverage_before=None
                )
                
                # Build content and context for this element
                content = self._content_builder.build_code_content(temp_plan, Path(file_path))
                context = self._context_assembler.context_for_generation(temp_plan, Path(file_path))
                
                # Generate plan using LLM
                plan_result = self._llm.generate_test_plan(
                    code_content=content,
                    context=context
                )
                
                # Create TestElementPlan with file path in tags for reference
                plan_tags = plan_result.get("scenarios", [])
                plan_tags.append(f"source_file:{file_path}")  # Store source file for UI display
                
                # Validate plan content and provide fallbacks
                plan_summary = plan_result.get("plan_summary", "").strip()
                detailed_plan = plan_result.get("detailed_plan", "").strip()
                
                if not plan_summary:
                    element_type = element.type.value if hasattr(element.type, 'value') else str(element.type)
                    plan_summary = f"Test plan for {element.name} ({element_type})"
                
                if not detailed_plan:
                    detailed_plan = f"Comprehensive testing plan for {element.name}:\n1. Test happy path scenarios\n2. Test edge cases and boundary conditions\n3. Test error handling and exceptions\n4. Mock external dependencies as needed"
                
                plan = TestElementPlan(
                    element=element,
                    eligibility_reason=eligibility_reason,
                    plan_summary=plan_summary,
                    detailed_plan=detailed_plan,
                    confidence=plan_result.get("confidence"),
                    tags=plan_tags
                )
                
                plans.append(plan)
                
            except Exception as e:
                logger.warning(
                    f"Failed to generate plan for {element.name} in {file_path}: {e}"
                )
                # Create a fallback plan with proper content
                fallback_plan = TestElementPlan(
                    element=element,
                    eligibility_reason=eligibility_reason,
                    plan_summary=f"Plan generation failed for {element.name} due to LLM error",
                    detailed_plan=f"Unable to generate detailed plan due to error: {str(e)}. Manual planning recommended for comprehensive test coverage.",
                    confidence=0.0,
                    tags=["error", f"source_file:{file_path}"]
                )
                plans.append(fallback_plan)
        
        return plans

    def _generate_file_plans(
        self,
        file_path: str,
        items: list[tuple[Any, str]],
        project_path: str,
    ) -> list[TestElementPlan]:
        """Generate plans for multiple elements within a single file using one LLM call.

        Args:
            file_path: Source file path
            items: List of tuples (element, eligibility_reason)
            project_path: Project root path (currently unused but reserved)

        Returns:
            List of TestElementPlan for each element in the file
        """
        from ..domain.models import TestGenerationPlan
        from pathlib import Path

        # Build a temporary plan aggregating all elements from this file
        elements = [element for (element, _reason) in items]
        tmp_plan = TestGenerationPlan(
            elements_to_test=elements,
            existing_tests=[],
            coverage_before=None,
        )

        # Build content and context once per file
        content = self._content_builder.build_code_content(tmp_plan, Path(file_path))
        context = self._context_assembler.context_for_generation(tmp_plan, Path(file_path))

        # Provide element descriptors to help the LLM return per-element plans
        element_descriptors = []
        for element in elements:
            try:
                start, end = element.line_range
                line_range = f"{start}-{end}"
            except Exception:
                line_range = ""
            element_descriptors.append(
                {
                    "name": element.name,
                    "type": getattr(element.type, "value", str(element.type)),
                    "line_range": line_range,
                }
            )

        # Single LLM call for the file
        result = self._llm.generate_test_plan(
            code_content=content,
            context=context,
            elements=element_descriptors,
        )

        # Map response to per-element TestElementPlan
        plans: list[TestElementPlan] = []

        # Prefer explicit element_plans array if provided
        returned_plans = result.get("element_plans") if isinstance(result, dict) else None

        # Build a quick index by (name, type) for matching
        by_key: dict[tuple[str, str], dict] = {}
        if isinstance(returned_plans, list):
            for p in returned_plans:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name", "")).strip()
                etype = str(p.get("type", "")).strip()
                if name and etype:
                    by_key[(name, etype)] = p

        for element, eligibility_reason in items:
            etype = getattr(element.type, "value", str(element.type))

            # Default values if we don't find a specific plan
            plan_summary = None
            detailed_plan = None
            confidence = None
            scenarios: list[str] = []

            # Try to match by name+type
            matched = by_key.get((element.name, etype))
            if matched:
                plan_summary = (matched.get("plan_summary") or "").strip()
                detailed_plan = (matched.get("detailed_plan") or "").strip()
                confidence = matched.get("confidence")
                scenarios = matched.get("scenarios") or []

            # Fallback: use single-plan fields for all elements if present
            if not plan_summary or not detailed_plan:
                if isinstance(result, dict):
                    plan_summary = plan_summary or (result.get("plan_summary") or "").strip()
                    detailed_plan = detailed_plan or (result.get("detailed_plan") or "").strip()
                    if confidence is None:
                        confidence = result.get("confidence")

            # Final safety defaults
            if not plan_summary:
                plan_summary = f"Test plan for {element.name} ({etype})"
            if not detailed_plan:
                detailed_plan = (
                    f"Comprehensive plan for {element.name}:\n"
                    "1. Test happy paths\n2. Test edge cases\n3. Test error handling\n4. Mock external dependencies as needed"
                )

            plan_tags = list(scenarios) if isinstance(scenarios, list) else []
            plan_tags.append(f"source_file:{file_path}")

            plans.append(
                TestElementPlan(
                    element=element,
                    eligibility_reason=eligibility_reason,
                    plan_summary=plan_summary,
                    detailed_plan=detailed_plan,
                    confidence=confidence,
                    tags=plan_tags,
                )
            )

        return plans

    def _persist_session(self, session: PlanningSession) -> None:
        """Persist the planning session to state storage."""
        try:
            # Store as the most recent session using StatePort interface
            self._state.update_state("last_planning_session", session.model_dump())
            
            # Optionally store in history (truncate to keep recent N)
            history_key = "planning_history"
            history = self._state.get_state(history_key, [])
            history.insert(0, {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "project_path": session.project_path,
                "stats": session.stats
            })
            # Keep last 10 sessions in history
            if len(history) > 10:
                history = history[:10]
            self._state.update_state(history_key, history)
            
            # Persist changes to disk
            self._state.persist_state()
            
        except Exception as e:
            logger.warning(f"Failed to persist planning session: {e}")
            # Don't fail the entire operation for persistence issues

    def get_planning_session(self, session_id: str | None = None) -> PlanningSession | None:
        """
        Retrieve a planning session by ID, or the most recent session.
        
        Args:
            session_id: Specific session ID, or None for most recent
            
        Returns:
            PlanningSession if found, None otherwise
        """
        try:
            if session_id is None:
                # Get most recent session
                session_data = self._state.get_state("last_planning_session")
                if session_data:
                    return PlanningSession(**session_data)
                return None
            
            # Search history for specific session ID
            history = self._state.get_state("planning_history", [])
            for entry in history:
                if entry["session_id"] == session_id:
                    # Need to load full session data - this would need enhancement
                    # For now, just return the most recent if IDs match
                    session_data = self._state.get_state("last_planning_session")
                    if session_data and session_data.get("session_id") == session_id:
                        return PlanningSession(**session_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve planning session {session_id}: {e}")
            return None

    def update_session_selections(
        self, 
        session_id: str, 
        selected_keys: list[str]
    ) -> PlanningSession | None:
        """
        Update the selected elements for a planning session.
        
        Args:
            session_id: Session identifier
            selected_keys: List of selected element keys
            
        Returns:
            Updated session if successful, None otherwise
        """
        try:
            session = self.get_planning_session(session_id)
            if not session:
                logger.warning(f"Planning session {session_id} not found")
                return None
            
            # Update selections
            updated_session = PlanningSession(
                session_id=session.session_id,
                project_path=session.project_path,
                created_at=session.created_at,
                items=session.items,
                selected_keys=selected_keys,
                stats=session.stats
            )
            
            # Persist updated session
            self._persist_session(updated_session)
            
            logger.info(f"Updated session {session_id} with {len(selected_keys)} selected elements")
            return updated_session
            
        except Exception as e:
            logger.error(f"Failed to update session selections: {e}")
            return None
