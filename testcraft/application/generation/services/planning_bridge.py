"""
Planning session integration and bridge utilities for GenerateUseCase.

This module handles integration with planning sessions, element filtering,
and detailed plan injection for enhanced test generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....domain.models import PlanningSession
from ....ports.state_port import StatePort

logger = logging.getLogger(__name__)


class PlanningBridge:
    """
    Bridge service for planning session integration.
    
    Handles loading planning sessions, filtering elements, and injecting
    detailed planning information into the generation context.
    """
    
    def __init__(self, state_port: StatePort):
        """Initialize planning bridge with state port."""
        self._state = state_port
    
    async def load_planning_session(self, session_id: str) -> tuple[PlanningSession | None, list[str]]:
        """
        Load planning session from state.
        
        Args:
            session_id: ID of the planning session to load
            
        Returns:
            Tuple of (planning_session, selected_element_keys)
        """
        try:
            planning_data = self._state.get_state("last_planning_session")
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
    
    def inject_planning_context(
        self,
        base_context: str,
        planning_session: PlanningSession,
        plan,
        source_path: Path,
    ) -> str:
        """
        Inject detailed planning information into generation context.
        
        Args:
            base_context: Base context string
            planning_session: Planning session with detailed plans
            plan: Generation plan containing elements to test
            source_path: Path to source file being processed
            
        Returns:
            Enhanced context with planning information
        """
        if not planning_session or not hasattr(planning_session, 'items'):
            return base_context
        
        # Find plans for elements in this generation plan
        element_plans = self._find_element_plans(planning_session, plan)
        
        if not element_plans:
            return base_context
        
        # Build planning context
        planning_context = self._build_planning_context(element_plans)
        
        # Combine with existing context
        if base_context:
            enhanced_context = f"{base_context}\n\n{planning_context}"
        else:
            enhanced_context = planning_context
        
        logger.info(f"Injected {len(element_plans)} detailed plans into generation context for {source_path}")
        return enhanced_context
    
    def _find_element_plans(self, planning_session: PlanningSession, plan) -> list:
        """Find planning items that match elements in the generation plan."""
        element_plans = []
        for element in plan.elements_to_test:
            for plan_item in planning_session.items:
                if (element.name == plan_item.element.name and 
                    element.type == plan_item.element.type):
                    element_plans.append(plan_item)
                    break
        return element_plans
    
    def _build_planning_context(self, element_plans: list) -> str:
        """Build detailed planning context from element plans."""
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
        return "\n".join(planning_context)


class ElementFilter:
    """
    Service for filtering elements based on planning session selections.
    
    Provides utilities for filtering generation plans based on selected
    elements from planning sessions.
    """
    
    def __init__(self):
        """Initialize element filter."""
        pass
    
    def filter_plans_by_elements(
        self,
        generation_plans: list,
        selected_element_keys: list[str],
        plan_builder,
    ) -> list:
        """
        Filter generation plans to include only selected elements.
        
        Args:
            generation_plans: List of generation plans
            selected_element_keys: List of selected element keys from planning
            plan_builder: Plan builder service for element filtering
            
        Returns:
            Filtered list of generation plans
        """
        if not selected_element_keys:
            return generation_plans
        
        try:
            # Use plan builder's filtering capability
            return plan_builder.build_plans_for_elements(
                generation_plans, selected_element_keys
            )
        except Exception as e:
            logger.warning(f"Failed to filter plans by elements: {e}")
            return generation_plans
    
    def validate_element_selection(
        self,
        selected_element_keys: list[str],
        available_plans: list,
    ) -> dict[str, Any]:
        """
        Validate that selected elements exist in available plans.
        
        Args:
            selected_element_keys: List of selected element keys
            available_plans: List of available generation plans
            
        Returns:
            Validation result with statistics
        """
        # Extract all available element keys from plans
        available_keys = set()
        for plan in available_plans:
            if hasattr(plan, 'elements_to_test'):
                for element in plan.elements_to_test:
                    # Create element key (this logic should match planning session key generation)
                    element_type = element.type.value if hasattr(element.type, 'value') else str(element.type)
                    element_key = f"{element.name}:{element_type}"
                    available_keys.add(element_key)
        
        # Check which selected keys are valid
        valid_keys = []
        invalid_keys = []
        
        for key in selected_element_keys:
            if key in available_keys:
                valid_keys.append(key)
            else:
                invalid_keys.append(key)
        
        return {
            "total_selected": len(selected_element_keys),
            "valid_selections": len(valid_keys),
            "invalid_selections": len(invalid_keys),
            "valid_keys": valid_keys,
            "invalid_keys": invalid_keys,
            "coverage_percentage": len(valid_keys) / len(selected_element_keys) * 100 if selected_element_keys else 0,
        }


class PlanningContextEnhancer:
    """
    Service for enhancing generation context with planning information.
    
    Provides utilities for enriching the generation context with detailed
    planning information, element-specific guidance, and test strategies.
    """
    
    def __init__(self):
        """Initialize planning context enhancer."""
        pass
    
    def enhance_context_with_planning(
        self,
        base_context: str,
        planning_session: PlanningSession | None,
        plan,
        source_path: Path,
    ) -> str:
        """
        Enhance generation context with planning information.
        
        Args:
            base_context: Base generation context
            planning_session: Optional planning session with detailed plans
            plan: Generation plan containing elements to test
            source_path: Path to source file being processed
            
        Returns:
            Enhanced context string
        """
        if not planning_session:
            return base_context
        
        # Get planning-specific context
        planning_context = self._extract_planning_context(planning_session, plan, source_path)
        
        if not planning_context:
            return base_context
        
        # Combine contexts
        if base_context:
            return f"{base_context}\n\n{planning_context}"
        else:
            return planning_context
    
    def _extract_planning_context(
        self,
        planning_session: PlanningSession,
        plan,
        source_path: Path,
    ) -> str:
        """Extract relevant planning context for the given plan."""
        if not hasattr(planning_session, 'items'):
            return ""
        
        # Find matching planning items
        matching_items = []
        for element in plan.elements_to_test:
            for plan_item in planning_session.items:
                if (element.name == plan_item.element.name and 
                    element.type == plan_item.element.type):
                    matching_items.append(plan_item)
        
        if not matching_items:
            return ""
        
        # Build context sections
        context_sections = []
        
        # Add session metadata
        context_sections.append("# PLANNING SESSION CONTEXT")
        context_sections.append(f"Session ID: {planning_session.session_id}")
        if hasattr(planning_session, 'description') and planning_session.description:
            context_sections.append(f"Description: {planning_session.description}")
        context_sections.append("")
        
        # Add element-specific plans
        context_sections.append("# ELEMENT-SPECIFIC TEST PLANS")
        for i, item in enumerate(matching_items, 1):
            context_sections.append(f"## Element {i}: {item.element.name}")
            element_type = item.element.type.value if hasattr(item.element.type, 'value') else str(item.element.type)
            context_sections.append(f"Type: {element_type}")
            if item.plan_summary:
                context_sections.append(f"Summary: {item.plan_summary}")
            if item.detailed_plan:
                context_sections.append(f"Detailed Plan:\n{item.detailed_plan}")
            if item.confidence:
                context_sections.append(f"Confidence: {item.confidence:.2f}")
            if item.tags:
                context_sections.append(f"Tags: {', '.join(item.tags)}")
            context_sections.append("")
        
        # Add implementation guidance
        context_sections.append("# IMPLEMENTATION GUIDANCE")
        context_sections.append("- Follow the detailed plans closely")
        context_sections.append("- Implement ONLY the elements and scenarios listed above")
        context_sections.append("- Use the confidence scores to prioritize test coverage")
        context_sections.append("- Consider the tags for test categorization and organization")
        
        return "\n".join(context_sections)
