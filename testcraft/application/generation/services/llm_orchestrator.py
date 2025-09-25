"""
LLM Orchestrator for PLAN/GENERATE/REFINE loops with symbol resolution.

This service implements the LLM orchestrator as specified in the context assembly
specification, including the missing_symbols resolution loop that fetches precise
definitions on demand during PLAN and REFINE stages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ....domain.models import ContextPack, ResolvedDef
from ....ports.llm_port import LLMPort
from ....ports.parser_port import ParserPort
from .context_assembler import ContextAssembler
from .context_pack import ContextPackBuilder
from .symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    LLM orchestrator implementing PLAN/GENERATE/REFINE loops with symbol resolution.

    Handles the complete test generation workflow including:
    - PLAN stage with missing_symbols resolution
    - GENERATE stage for test creation
    - REFINE stage for test repair with symbol resolution
    - Context re-packing and retry logic
    """

    def __init__(
        self,
        llm_port: LLMPort,
        parser_port: ParserPort,
        context_assembler: ContextAssembler,
        context_pack_builder: ContextPackBuilder | None = None,
        symbol_resolver: SymbolResolver | None = None,
        max_plan_retries: int = 2,
        max_refine_retries: int = 3,
    ) -> None:
        """
        Initialize the LLM orchestrator.

        Args:
            llm_port: LLM port for generating and refining tests
            parser_port: Parser port for extracting symbol definitions
            context_assembler: Service for assembling context
            context_pack_builder: Builder for context packs (optional)
            symbol_resolver: Service for resolving missing symbols (optional)
            max_plan_retries: Maximum retries for PLAN stage
            max_refine_retries: Maximum retries for REFINE stage
        """
        self._llm = llm_port
        self._parser = parser_port
        self._context_assembler = context_assembler
        self._context_pack_builder = context_pack_builder
        self._symbol_resolver = symbol_resolver or SymbolResolver(parser_port)
        self._max_plan_retries = max_plan_retries
        self._max_refine_retries = max_refine_retries

    def plan_and_generate(
        self,
        context_pack: ContextPack,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full PLAN/GENERATE workflow.

        Args:
            context_pack: Context pack for test generation
            project_root: Project root directory

        Returns:
            Dictionary containing generated tests and metadata
        """
        logger.info(
            "Starting PLAN/GENERATE workflow for %s", context_pack.target.object
        )

        # PLAN stage with symbol resolution
        plan = self._plan_stage(context_pack, project_root)
        if not plan:
            raise ValueError("PLAN stage failed - no plan generated")

        # GENERATE stage
        generated_code = self._generate_stage(context_pack, plan)
        if not generated_code:
            raise ValueError("GENERATE stage failed - no code generated")

        return {
            "plan": plan,
            "generated_code": generated_code,
            "context_pack": context_pack,
        }

    def plan_stage(
        self, context_pack: ContextPack, project_root: Path | None = None
    ) -> dict[str, Any] | None:
        """
        Execute the PLAN stage with symbol resolution.

        Args:
            context_pack: Context pack for planning
            project_root: Project root directory

        Returns:
            Plan dictionary if successful, None otherwise
        """
        return self._plan_stage(context_pack, project_root)

    def refine_stage(
        self,
        context_pack: ContextPack,
        existing_code: str,
        feedback: dict[str, Any],
        project_root: Path | None = None,
    ) -> str | None:
        """
        Execute the REFINE stage with symbol resolution.

        Args:
            context_pack: Context pack for refinement
            existing_code: Current test code to refine
            feedback: Execution feedback (traceback, coverage, etc.)
            project_root: Project root directory

        Returns:
            Refined code if successful, None otherwise
        """
        return self._refine_stage(context_pack, existing_code, feedback, project_root)

    def _plan_stage(
        self, context_pack: ContextPack, project_root: Path | None = None
    ) -> dict[str, Any] | None:
        """
        Execute PLAN stage with symbol resolution loop.

        Args:
            context_pack: Context pack for planning
            project_root: Project root directory

        Returns:
            Plan dictionary if successful
        """
        current_context = context_pack
        retry_count = 0

        while retry_count <= self._max_plan_retries:
            try:
                logger.info("Executing PLAN stage (attempt %d)", retry_count + 1)

                # Create PLAN prompt
                plan_prompt = self._create_plan_prompt(current_context)

                # Call LLM for planning
                response = self._llm.generate_test(plan_prompt)
                response_text = self._extract_response_text(response)

                # Parse response as JSON
                try:
                    plan = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse PLAN response as JSON: %s", e)
                    return None

                # Check for missing symbols
                missing_symbols = plan.get("missing_symbols", [])
                if not missing_symbols:
                    # No missing symbols - return the plan
                    return plan

                logger.info(
                    "Found %d missing symbols in PLAN response", len(missing_symbols)
                )

                # Resolve missing symbols
                resolved_defs = self._symbol_resolver.resolve_symbols(
                    missing_symbols, project_root
                )

                if not resolved_defs:
                    logger.warning("Could not resolve any missing symbols")
                    # Continue retrying in case the next LLM call doesn't have missing symbols
                    retry_count += 1
                    if retry_count > self._max_plan_retries:
                        logger.warning("PLAN stage exceeded maximum retries")
                        return None
                    continue

                # Add resolved definitions to context pack
                updated_resolved_defs = (
                    list(current_context.resolved_defs) + resolved_defs
                )
                current_context = self._update_context_pack_resolved_defs(
                    current_context, updated_resolved_defs
                )

                retry_count += 1
                logger.info(
                    "Retrying PLAN stage with resolved symbols (attempt %d)",
                    retry_count + 1,
                )

            except Exception as e:
                logger.exception("Error in PLAN stage: %s", e)
                return None

        logger.warning("PLAN stage exceeded maximum retries")
        return None

    def _generate_stage(
        self, context_pack: ContextPack, plan: dict[str, Any]
    ) -> str | None:
        """
        Execute GENERATE stage.

        Args:
            context_pack: Context pack for generation
            plan: Approved plan from PLAN stage

        Returns:
            Generated test code if successful
        """
        try:
            logger.info("Executing GENERATE stage")

            # Create GENERATE prompt
            generate_prompt = self._create_generate_prompt(context_pack, plan)

            # Call LLM for generation
            response = self._llm.generate_test(generate_prompt)
            response_text = self._extract_response_text(response)

            # Extract code from response
            generated_code = self._extract_code_from_response(response_text)
            if not generated_code:
                logger.warning("No code found in GENERATE response")
                return None

            return generated_code

        except Exception as e:
            logger.exception("Error in GENERATE stage: %s", e)
            return None

    def _refine_stage(
        self,
        context_pack: ContextPack,
        existing_code: str,
        feedback: dict[str, Any],
        project_root: Path | None = None,
    ) -> str | None:
        """
        Execute REFINE stage with symbol resolution loop.

        Args:
            context_pack: Context pack for refinement
            existing_code: Current test code to refine
            feedback: Execution feedback
            project_root: Project root directory

        Returns:
            Refined code if successful
        """
        current_context = context_pack
        retry_count = 0

        while retry_count <= self._max_refine_retries:
            try:
                logger.info("Executing REFINE stage (attempt %d)", retry_count + 1)

                # Create REFINE prompt
                refine_prompt = self._create_refine_prompt(
                    current_context, existing_code, feedback
                )

                # Call LLM for refinement
                response = self._llm.generate_test(refine_prompt)
                response_text = self._extract_response_text(response)

                # Parse response as JSON or extract code
                if response_text.strip().startswith("{"):
                    try:
                        response_data = json.loads(response_text)
                        if "missing_symbols" in response_data:
                            # Handle missing symbols
                            missing_symbols = response_data["missing_symbols"]
                            if missing_symbols:
                                logger.info(
                                    "Found %d missing symbols in REFINE response",
                                    len(missing_symbols),
                                )

                                # Resolve missing symbols
                                resolved_defs = self._symbol_resolver.resolve_symbols(
                                    missing_symbols, project_root
                                )

                                if resolved_defs:
                                    # Add resolved definitions to context pack
                                    updated_resolved_defs = (
                                        list(current_context.resolved_defs)
                                        + resolved_defs
                                    )
                                    current_context = (
                                        self._update_context_pack_resolved_defs(
                                            current_context, updated_resolved_defs
                                        )
                                    )

                                    retry_count += 1
                                    continue  # Retry with resolved symbols

                            # If no symbols to resolve, extract code from response
                            refined_code = response_data.get(
                                "refined_code", existing_code
                            )
                        else:
                            # No missing symbols, extract code
                            refined_code = response_data.get(
                                "refined_code", existing_code
                            )
                    except json.JSONDecodeError:
                        # Not JSON, treat as code
                        refined_code = response_text
                else:
                    # Treat as code
                    refined_code = response_text

                # Extract code from response
                refined_code = self._extract_code_from_response(refined_code)
                if not refined_code:
                    logger.warning("No code found in REFINE response")
                    return existing_code  # Return original code

                return refined_code

            except Exception as e:
                logger.exception("Error in REFINE stage: %s", e)
                retry_count += 1

        logger.warning("REFINE stage exceeded maximum retries")
        return existing_code  # Return original code as fallback

    def _create_plan_prompt(self, context_pack: ContextPack) -> str:
        """Create PLAN stage prompt."""
        return f"""
You are a senior Python test engineer. You write small, correct, deterministic pytest tests.
Do NOT guess missing symbols. List them.

TARGET
- File: {context_pack.target.module_file}
- Object: {context_pack.target.object}
- Canonical import to use in tests (must use exactly this):
  {context_pack.import_map.target_import}

Focal code:
{context_pack.focal.source}

Signature/docstring:
{context_pack.focal.signature}
{context_pack.focal.docstring or ""}

Precise repository context (curated):
Resolved definitions you can rely on:
{self._format_resolved_defs(context_pack.resolved_defs)}

Property-related examples (GIVEN/WHEN/THEN):
GIVEN:
{self._format_gwt_snippets(context_pack.property_context.gwt_snippets.given)}
WHEN:
{self._format_gwt_snippets(context_pack.property_context.gwt_snippets.when)}
THEN:
{self._format_gwt_snippets(context_pack.property_context.gwt_snippets.then)}

Repo conventions:
{self._format_conventions(context_pack.conventions)}

TASK:
1) Produce a TEST PLAN (cases, boundaries, exceptions, side-effects, fixtures/mocks).
2) List "missing_symbols" you need (fully qualified where possible).
3) Confirm the import you will write at the top of the test file.
Output strictly as JSON: {{"plan":[...], "missing_symbols":[...], "import_line":"..."}}
""".strip()

    def _create_generate_prompt(
        self, context_pack: ContextPack, plan: dict[str, Any]
    ) -> str:
        """Create GENERATE stage prompt."""
        return f"""
You are a senior Python test engineer. Output a single runnable pytest module. Use ONLY the provided canonical import.
No network. Use tmp_path for FS. Keep imports minimal.

Canonical import (must appear at top of the file):
{context_pack.import_map.target_import}

Focal code (trimmed):
{context_pack.focal.source}

Resolved definitions (only what you can call):
{self._format_resolved_defs(context_pack.resolved_defs)}

Property-related patterns and test-bundle fragments:
{self._format_property_context(context_pack.property_context)}

Repo conventions / determinism:
{self._format_conventions(context_pack.conventions)}

Approved TEST PLAN:
{plan.get("plan", [])}

REQUIREMENTS:
- Use EXACTLY the canonical import above.
- Prefer pytest parametrization for partitions/boundaries.
- Assertions must check behavior (not just "no exception").
- If side-effects occur, assert on state/IO/logs accordingly.
- Name tests `test_<target_simplename>_<behavior>`.
- Output ONLY the complete test module in one fenced block.
""".strip()

    def _create_refine_prompt(
        self, context_pack: ContextPack, existing_code: str, feedback: dict[str, Any]
    ) -> str:
        """Create REFINE stage prompt."""
        return f"""
You repair Python tests with minimal edits. Keep style and canonical import unchanged.

Last tests (trim to failing parts):
{self._extract_failing_parts(existing_code, feedback)}

Focal code (trimmed):
{context_pack.focal.source}

Canonical import (DO NOT change):
{context_pack.import_map.target_import}

Execution feedback:
- Result: {feedback.get("result", "unknown")}
- Trace excerpt: {feedback.get("trace_excerpt", "")}
- Coverage gaps: {feedback.get("coverage", {})}
- Surviving mutants: {feedback.get("mutants_survived", [])}

Constraints:
- Do NOT introduce new undefined symbols. If truly needed, output {{"missing_symbols":[...]}} and nothing else.

TASK:
1) Brief rationale of changes (compile fix / wrong assumption / new branch case / stronger oracle).
2) Output the corrected full test module.
3) If you require new symbols, output only {{"missing_symbols":[...]}}.
""".strip()

    def _update_context_pack_resolved_defs(
        self, context_pack: ContextPack, new_resolved_defs: list[ResolvedDef]
    ) -> ContextPack:
        """Update context pack with new resolved definitions."""
        # Since ContextPack is immutable, we need to create a new one
        # This is a simplified version - in reality we'd need to handle all fields
        return ContextPack(
            target=context_pack.target,
            import_map=context_pack.import_map,
            focal=context_pack.focal,
            resolved_defs=new_resolved_defs,
            property_context=context_pack.property_context,
            conventions=context_pack.conventions,
            budget=context_pack.budget,
        )

    def _format_resolved_defs(self, resolved_defs: list[ResolvedDef]) -> str:
        """Format resolved definitions for prompt."""
        if not resolved_defs:
            return "None"

        formatted = []
        for defn in resolved_defs:
            formatted.append(f"- {defn.name} ({defn.kind}): {defn.signature}")
            if defn.doc:
                formatted.append(f"  {defn.doc}")

        return "\n".join(formatted)

    def _format_gwt_snippets(self, snippets: list[str]) -> str:
        """Format GWT snippets for prompt."""
        if not snippets:
            return "None"

        return "\n".join(f"- {snippet}" for snippet in snippets)

    def _format_conventions(self, conventions) -> str:
        """Format conventions for prompt."""
        return f"""
- Test style: {conventions.test_style}
- Allowed libs: {", ".join(conventions.allowed_libs)}
- Determinism: seed={conventions.determinism.seed}, tz={conventions.determinism.tz}, freeze_time={conventions.determinism.freeze_time}
- IO policy: network={conventions.io_policy.network}, fs={conventions.io_policy.fs}
""".strip()

    def _format_property_context(self, property_context) -> str:
        """Format property context for prompt."""
        context_parts = []

        if property_context.ranked_methods:
            context_parts.append("Ranked methods:")
            for method in property_context.ranked_methods:
                context_parts.append(
                    f"  - {method.qualname} ({method.level}, {method.relation})"
                )

        if (
            property_context.gwt_snippets.given
            or property_context.gwt_snippets.when
            or property_context.gwt_snippets.then
        ):
            context_parts.append("GWT patterns:")
            if property_context.gwt_snippets.given:
                context_parts.append(
                    f"  GIVEN: {'; '.join(property_context.gwt_snippets.given)}"
                )
            if property_context.gwt_snippets.when:
                context_parts.append(
                    f"  WHEN: {'; '.join(property_context.gwt_snippets.when)}"
                )
            if property_context.gwt_snippets.then:
                context_parts.append(
                    f"  THEN: {'; '.join(property_context.gwt_snippets.then)}"
                )

        return "\n".join(context_parts) if context_parts else "None"

    def _extract_failing_parts(
        self, existing_code: str, feedback: dict[str, Any]
    ) -> str:
        """Extract failing parts from existing code and feedback."""
        # This is a simplified implementation
        # In reality, you'd use AST to extract specific failing lines
        return existing_code  # For now, return the full code

    def _extract_response_text(self, response) -> str:
        """Extract text from LLM response."""
        # Handle different response formats
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _extract_code_from_response(self, response_text: str) -> str | None:
        """Extract code from LLM response."""
        # Look for code blocks
        import re

        # Try to find code between triple backticks
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        match = re.search(code_block_pattern, response_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no code blocks found, try to find the last Python code section
        lines = response_text.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                in_code = True
                code_lines.append(line)
            elif in_code:
                if line.strip() and not line.startswith(" "):
                    # Line doesn't start with space, might be end of code
                    break
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return None
